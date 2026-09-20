"""The permanent-magnet machine's dq-frame state equations.

The rotor speed is imposed by a dynamometer rather than integrated, so the
mechanical equation never appears: the state is ``[id, iq, theta_m]`` and speed
is an input. That is what the test rig does, and it is why the estimator can
treat the electrical angular velocity as a measured signal rather than a state
it has to infer.
"""

from __future__ import annotations

from .params import MotorParams

__all__ = ["electromagnetic_torque", "state_derivative"]

State = tuple[float, float, float]


def state_derivative(
    state: State,
    v_d: float,
    v_q: float,
    omega_m: float,
    motor: MotorParams,
) -> State:
    """Return ``d/dt [id, iq, theta_m]`` at one operating point.

    The electrical equations are the standard dq model with cross-coupling:

        Ld(i) * did/dt = vd - Rs*id + omega_e * Lq(i) * iq
        Lq(i) * diq/dt = vq - Rs*iq - omega_e * (Ld(i)*id + lambda_m)

    where ``omega_e = pole_pairs * omega_m`` and the inductances depend on the
    current magnitude through `MotorParams.saturated`.

    Args:
        state: The ``(id, iq, theta_m)`` triple, in ampere, ampere and radian.
        v_d: Applied d-axis voltage, volt.
        v_q: Applied q-axis voltage, volt.
        omega_m: Mechanical angular velocity, radian per second.
        motor: Machine parameters.

    Returns:
        The time derivative of the state.
    """
    i_d, i_q, _theta_m = state
    ld, lq = motor.saturated(i_d, i_q)
    omega_e = motor.pole_pairs * omega_m

    did_dt = (v_d - motor.rs * i_d + omega_e * lq * i_q) / ld
    diq_dt = (v_q - motor.rs * i_q - omega_e * (ld * i_d + motor.lambda_m)) / lq
    return did_dt, diq_dt, omega_m


def electromagnetic_torque(i_d: float, i_q: float, motor: MotorParams) -> float:
    """Return the electromagnetic torque at an operating point.

    Not used by the estimator, which never sees torque, but recorded in the
    trace because it is the physically meaningful output of the drive.

    Args:
        i_d: d-axis current, ampere.
        i_q: q-axis current, ampere.
        motor: Machine parameters.

    Returns:
        Torque, newton metre. Uses unsaturated inductances for the reluctance
        term, matching the reference implementation.
    """
    magnet = motor.lambda_m * i_q
    reluctance = (motor.ld - motor.lq) * i_d * i_q
    return 1.5 * motor.pole_pairs * (magnet + reluctance)
