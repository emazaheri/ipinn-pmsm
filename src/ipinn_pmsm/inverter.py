"""Voltage-source inverter: magnitude limit, conduction drop and dead time.

The estimator is only ever shown the *commanded* voltage, because that is what
a real drive can log without extra sensors. Everything this module subtracts
from the command is therefore unmodelled disturbance from the estimator's point
of view, which is exactly the point of being able to switch it on.
"""

from __future__ import annotations

from .params import InverterParams
from .transforms import dead_time_projection, phase_signs

__all__ = ["applied_voltage", "limit_magnitude"]


def limit_magnitude(v_d: float, v_q: float, v_max: float) -> tuple[float, float]:
    """Scale a dq voltage vector back onto the realisable circle.

    Args:
        v_d: Commanded d-axis voltage, volt.
        v_q: Commanded q-axis voltage, volt.
        v_max: Largest realisable magnitude, volt.

    Returns:
        The possibly scaled ``(v_d, v_q)`` pair. Direction is preserved, which
        matters: clipping the axes independently would rotate the vector and
        inject a torque error.
    """
    magnitude = (v_d * v_d + v_q * v_q) ** 0.5
    if magnitude <= v_max or magnitude == 0.0:
        return v_d, v_q
    scale = v_max / magnitude
    return v_d * scale, v_q * scale


def applied_voltage(
    v_d_cmd: float,
    v_q_cmd: float,
    i_d: float,
    i_q: float,
    theta_e: float,
    inverter: InverterParams,
) -> tuple[float, float, float, float]:
    """Return the voltage the bridge actually applies to the machine.

    Args:
        v_d_cmd: Commanded d-axis voltage, volt.
        v_q_cmd: Commanded q-axis voltage, volt.
        i_d: Measured d-axis current, ampere.
        i_q: Measured q-axis current, ampere.
        theta_e: Electrical rotor angle, radian.
        inverter: Inverter parameters.

    Returns:
        The ``(v_d, v_q, d_d, d_q)`` tuple: the applied voltages and the
        dead-time projection terms, the latter recorded for diagnostics.
    """
    v_d, v_q = limit_magnitude(v_d_cmd, v_q_cmd, inverter.v_max_dq)

    if inverter.v_dead == 0.0:
        # Skip the trigonometry when there is no dead time to project. The
        # signs are only ever used to scale v_dead, so this is exact, not an
        # approximation.
        d_d = d_q = 0.0
    else:
        d_d, d_q = dead_time_projection(theta_e, phase_signs(i_d, i_q, theta_e))

    return (
        v_d - inverter.r_cd * i_d - inverter.v_dead * d_d,
        v_q - inverter.r_cd * i_q - inverter.v_dead * d_q,
        d_d,
        d_q,
    )
