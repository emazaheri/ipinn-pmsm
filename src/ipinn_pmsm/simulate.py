"""Fixed-step simulation of the closed-loop drive.

The reference study called ``scipy.integrate.odeint`` once per control step,
which runs adaptive LSODA over a 100 us interval. This module uses classical
fixed-step RK4 instead, for three reasons: it is deterministic, it is
portable to any language without an ODE suite, and it is roughly two orders of
magnitude faster. The difference between the two integrators is measured
once and recorded in a fixture rather than assumed to be negligible, so that
"my RK4 agrees with your RK4" never gets confused with "your RK4 agrees with
your odeint". See `ipinn_pmsm.fixtures`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .control import FieldOrientedController
from .inverter import applied_voltage
from .machine import electromagnetic_torque, state_derivative
from .params import InverterParams, MotorParams, SimParams

__all__ = ["Schedule", "Trace", "simulate"]

Schedule = Callable[[float], float]
"""A reference or disturbance as a function of time, in seconds."""

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Trace:
    """One simulated run, decimated to the rate the estimator sees.

    Every array has the same length. Voltages come in two flavours because the
    distinction matters to the estimator: `v_d`/`v_q` are what the controller
    asked for and what a real drive can log, while `v_d_applied`/`v_q_applied`
    include the inverter's magnitude limit, conduction drop and dead time. They
    are equal only for an ideal bridge.

    Attributes:
        t: Sample times, second.
        i_d: d-axis current, ampere.
        i_q: q-axis current, ampere.
        v_d: Commanded d-axis voltage, volt.
        v_q: Commanded q-axis voltage, volt.
        v_d_applied: Applied d-axis voltage, volt.
        v_q_applied: Applied q-axis voltage, volt.
        omega_e: Electrical angular velocity, radian per second.
        omega_m: Mechanical angular velocity, radian per second.
        torque: Electromagnetic torque, newton metre.
        dt: Time between samples, second.
    """

    t: F64
    i_d: F64
    i_q: F64
    v_d: F64
    v_q: F64
    v_d_applied: F64
    v_q_applied: F64
    omega_e: F64
    omega_m: F64
    torque: F64
    dt: float

    def __len__(self) -> int:
        """Number of samples in the trace."""
        return int(self.t.size)

    def with_noise(
        self,
        current_percent: float,
        voltage_percent: float,
        seed: int,
        *,
        use_applied: bool = False,
        legacy_rng: bool = False,
    ) -> Trace:
        """Return a copy with white Gaussian measurement noise added.

        Noise is scaled to a percentage of each signal's RMS over the whole
        record, matching the reference study's case D. Speed is left clean: the
        dynamometer's encoder is far quieter than the current sensors, and the
        reference did not perturb it either.

        Args:
            current_percent: Current noise standard deviation, as a percentage
                of current RMS.
            voltage_percent: Voltage noise standard deviation, as a percentage
                of voltage RMS.
            seed: PRNG seed, so a noisy run is still reproducible.
            use_applied: Perturb the applied voltages as well as the commanded
                ones.
            legacy_rng: Draw from NumPy's legacy Mersenne Twister rather than
                PCG64, in the order id, iq, vd, vq. The reference study called
                the global `np.random.seed` and `np.random.normal`, so this is
                what reproduces its exact noise realisation. PCG64 is the
                better generator and is the default everywhere else.

        Returns:
            A new `Trace`. The original is unchanged.
        """
        rng: np.random.RandomState | np.random.Generator = (
            np.random.RandomState(seed) if legacy_rng else np.random.default_rng(seed)
        )

        def perturb(signal: F64, percent: float) -> F64:
            if percent <= 0.0:
                return signal
            sigma = float(np.sqrt(np.mean(signal**2))) * percent / 100.0
            return signal + rng.normal(0.0, sigma, size=signal.shape)

        return Trace(
            t=self.t,
            i_d=perturb(self.i_d, current_percent),
            i_q=perturb(self.i_q, current_percent),
            v_d=perturb(self.v_d, voltage_percent),
            v_q=perturb(self.v_q, voltage_percent),
            v_d_applied=(
                perturb(self.v_d_applied, voltage_percent)
                if use_applied
                else self.v_d_applied
            ),
            v_q_applied=(
                perturb(self.v_q_applied, voltage_percent)
                if use_applied
                else self.v_q_applied
            ),
            omega_e=self.omega_e,
            omega_m=self.omega_m,
            torque=self.torque,
            dt=self.dt,
        )


def simulate(
    motor: MotorParams,
    inverter: InverterParams,
    sim: SimParams,
    i_d_ref: Schedule,
    i_q_ref: Schedule,
    speed_ref: Schedule,
) -> Trace:
    """Run the closed-loop drive and return the decimated trace.

    The loop is the standard cascade at one rate: sample the state, run the
    current controllers, pass the command through the inverter model, then
    integrate the machine one step with the applied voltage held constant.
    Speed is imposed by the dynamometer, so it is evaluated from the schedule
    at each RK4 stage rather than integrated.

    Args:
        motor: Machine parameters.
        inverter: Inverter parameters.
        sim: Step size, duration and decimation.
        i_d_ref: d-axis current reference, ampere, as a function of time.
        i_q_ref: q-axis current reference, ampere, as a function of time.
        speed_ref: Mechanical angular velocity, radian per second, as a
            function of time.

    Returns:
        The decimated `Trace`.
    """
    controller = FieldOrientedController.from_params(motor, inverter, sim.ts)
    num_steps = sim.num_steps
    keep = range(0, num_steps, sim.decimation)
    count = len(keep)

    out = {name: np.empty(count, dtype=np.float64) for name in _TRACE_FIELDS}

    state: tuple[float, float, float] = (0.0, 0.0, 0.0)
    write = 0

    for step in range(num_steps):
        t = sim.stamp(step)
        i_d, i_q, theta_m = state
        theta_e = motor.pole_pairs * theta_m
        omega_m = speed_ref(t)

        v_d_cmd, v_q_cmd = controller.update(i_d_ref(t), i_q_ref(t), i_d, i_q)
        v_d_app, v_q_app, _d_d, _d_q = applied_voltage(
            v_d_cmd, v_q_cmd, i_d, i_q, theta_e, inverter
        )

        if step % sim.decimation == 0:
            out["t"][write] = t
            out["i_d"][write] = i_d
            out["i_q"][write] = i_q
            out["v_d"][write] = v_d_cmd
            out["v_q"][write] = v_q_cmd
            out["v_d_applied"][write] = v_d_app
            out["v_q_applied"][write] = v_q_app
            out["omega_m"][write] = omega_m
            out["omega_e"][write] = motor.pole_pairs * omega_m
            out["torque"][write] = electromagnetic_torque(i_d, i_q, motor)
            write += 1

        state = _rk4_step(state, t, sim.ts, v_d_app, v_q_app, speed_ref, motor)
        # Keep the angle bounded so cos and sin stay accurate over long runs.
        state = (state[0], state[1], state[2] % (2.0 * np.pi))

    return Trace(dt=sim.sample_dt, **out)


_TRACE_FIELDS = (
    "t",
    "i_d",
    "i_q",
    "v_d",
    "v_q",
    "v_d_applied",
    "v_q_applied",
    "omega_e",
    "omega_m",
    "torque",
)


def _rk4_step(
    state: tuple[float, float, float],
    t: float,
    dt: float,
    v_d: float,
    v_q: float,
    speed_ref: Schedule,
    motor: MotorParams,
) -> tuple[float, float, float]:
    """Advance the machine state one step with classical RK4.

    The applied voltage is held constant across the step, which is what the
    inverter does between switching updates. Speed is read from the schedule at
    each stage time, matching the reference implementation's behaviour of
    calling the schedule from inside the ODE.
    """

    def f(s: tuple[float, float, float], time: float) -> tuple[float, float, float]:
        return state_derivative(s, v_d, v_q, speed_ref(time), motor)

    half = 0.5 * dt
    k1 = f(state, t)
    s2 = tuple(state[i] + half * k1[i] for i in range(3))
    k2 = f(s2, t + half)  # type: ignore[arg-type]
    s3 = tuple(state[i] + half * k2[i] for i in range(3))
    k3 = f(s3, t + half)  # type: ignore[arg-type]
    s4 = tuple(state[i] + dt * k3[i] for i in range(3))
    k4 = f(s4, t + dt)  # type: ignore[arg-type]

    sixth = dt / 6.0
    return (
        state[0] + sixth * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state[1] + sixth * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        state[2] + sixth * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    )
