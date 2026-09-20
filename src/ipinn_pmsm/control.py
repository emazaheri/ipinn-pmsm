"""Field-oriented current control: two PI loops, one per axis.

The gains follow the reference study's rule, which places the closed-loop
current bandwidth at a fixed fraction of the sampling rate:

    Kp = 0.9 * L / Ts        Ki = 0.9 * Rs / Ts

This is aggressive by design. The point of the drive is to hold the operating
point steady so the estimator sees a clean step response at each of the eight
combinations, not to model a realistic commissioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .params import InverterParams, MotorParams

__all__ = ["FieldOrientedController", "PIController"]


@dataclass(slots=True)
class PIController:
    """A clamped proportional-integral controller.

    Both the integrator and the output are clamped to the same bounds. Clamping
    the integrator is what stops the loop winding up while the inverter is
    saturated, which the drive spends a real fraction of each step transient in.

    Attributes:
        kp: Proportional gain.
        ki: Integral gain.
        limit: Symmetric output bound. Both the integrator state and the output
            are clamped to ``[-limit, +limit]``.
        ts: Sample time, second.
        integrator: Accumulated integral term. Reset with `reset`.
    """

    kp: float
    ki: float
    limit: float
    ts: float
    integrator: float = field(default=0.0)

    def reset(self) -> None:
        """Clear the integrator, so a fresh run starts from a known state."""
        self.integrator = 0.0

    def update(self, error: float) -> float:
        """Advance one sample and return the control output.

        Args:
            error: Reference minus measurement.

        Returns:
            The clamped controller output.
        """
        self.integrator = _clamp(
            self.integrator + self.ki * error * self.ts, self.limit
        )
        return _clamp(self.kp * error + self.integrator, self.limit)


def _clamp(value: float, limit: float) -> float:
    """Clamp a value to a symmetric bound."""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@dataclass(slots=True)
class FieldOrientedController:
    """The pair of current loops that make up the drive's inner control.

    Attributes:
        d_axis: The d-axis current controller.
        q_axis: The q-axis current controller.
    """

    d_axis: PIController
    q_axis: PIController

    @classmethod
    def from_params(
        cls,
        motor: MotorParams,
        inverter: InverterParams,
        ts: float,
        kp_factor: float = 0.9,
        ki_factor: float = 0.9,
    ) -> FieldOrientedController:
        """Build both loops from the machine and inverter parameters.

        Args:
            motor: Machine parameters, which set the gains.
            inverter: Inverter parameters, which set the output bound.
            ts: Sample time, second.
            kp_factor: Numerator factor on the proportional gains.
            ki_factor: Numerator factor on the integral gains.

        Returns:
            A controller with both loops initialised and their integrators zero.
        """
        limit = inverter.v_max_dq
        return cls(
            d_axis=PIController(
                kp_factor * motor.ld / ts, ki_factor * motor.rs / ts, limit, ts
            ),
            q_axis=PIController(
                kp_factor * motor.lq / ts, ki_factor * motor.rs / ts, limit, ts
            ),
        )

    def reset(self) -> None:
        """Clear both integrators."""
        self.d_axis.reset()
        self.q_axis.reset()

    def update(
        self, i_d_ref: float, i_q_ref: float, i_d: float, i_q: float
    ) -> tuple[float, float]:
        """Advance both loops one sample.

        Args:
            i_d_ref: d-axis current reference, ampere.
            i_q_ref: q-axis current reference, ampere.
            i_d: Measured d-axis current, ampere.
            i_q: Measured q-axis current, ampere.

        Returns:
            The commanded ``(v_d, v_q)`` pair, volt.
        """
        return (
            self.d_axis.update(i_d_ref - i_d),
            self.q_axis.update(i_q_ref - i_q),
        )
