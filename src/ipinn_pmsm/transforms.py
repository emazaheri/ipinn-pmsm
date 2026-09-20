"""Clarke and Park transforms between the abc, alpha-beta and dq frames.

These are the amplitude-invariant forms used throughout the reference study,
not the power-invariant ones: a balanced set of unit-amplitude phase currents
maps to a dq vector of unit magnitude. Mixing the two conventions rescales
every voltage by sqrt(3/2), which would move the estimated flux linkage by the
same factor, so the convention is asserted in the tests rather than assumed.
"""

from __future__ import annotations

import math

__all__ = [
    "clarke",
    "inverse_clarke",
    "inverse_park",
    "park",
]

_SQRT3 = math.sqrt(3.0)
_TWO_PI_OVER_3 = 2.0 * math.pi / 3.0


def clarke(a: float, b: float, _c: float) -> tuple[float, float]:
    """Map balanced abc quantities onto the stationary alpha-beta frame.

    Args:
        a: Phase a quantity.
        b: Phase b quantity.
        _c: Phase c quantity. Unused: for a balanced set it is implied by the
            other two, and the reference implementation ignores it as well.

    Returns:
        The ``(alpha, beta)`` pair.
    """
    return a, (a + 2.0 * b) / _SQRT3


def inverse_clarke(alpha: float, beta: float) -> tuple[float, float, float]:
    """Map a stationary alpha-beta vector back onto three phase quantities.

    Args:
        alpha: Alpha-axis quantity.
        beta: Beta-axis quantity.

    Returns:
        The ``(a, b, c)`` triple.
    """
    half_beta = 0.5 * _SQRT3 * beta
    return alpha, -0.5 * alpha + half_beta, -0.5 * alpha - half_beta


def park(alpha: float, beta: float, theta_e: float) -> tuple[float, float]:
    """Rotate a stationary vector into the rotor dq frame.

    Args:
        alpha: Alpha-axis quantity.
        beta: Beta-axis quantity.
        theta_e: Electrical rotor angle, radian.

    Returns:
        The ``(d, q)`` pair.
    """
    cos_t, sin_t = math.cos(theta_e), math.sin(theta_e)
    return alpha * cos_t + beta * sin_t, -alpha * sin_t + beta * cos_t


def inverse_park(d: float, q: float, theta_e: float) -> tuple[float, float]:
    """Rotate a dq vector back into the stationary frame.

    Args:
        d: d-axis quantity.
        q: q-axis quantity.
        theta_e: Electrical rotor angle, radian.

    Returns:
        The ``(alpha, beta)`` pair.
    """
    cos_t, sin_t = math.cos(theta_e), math.sin(theta_e)
    return d * cos_t - q * sin_t, d * sin_t + q * cos_t


def phase_signs(i_d: float, i_q: float, theta_e: float) -> tuple[float, float, float]:
    """Return the signs of the three phase currents at an operating point.

    The inverter's dead-time model needs the polarity of each phase current,
    which is only visible in the abc frame.

    Args:
        i_d: d-axis current, ampere.
        i_q: q-axis current, ampere.
        theta_e: Electrical rotor angle, radian.

    Returns:
        The ``(sign_a, sign_b, sign_c)`` triple, each +1.0 or -1.0. Zero counts
        as positive, matching the reference implementation.
    """
    i_alpha, i_beta = inverse_park(i_d, i_q, theta_e)
    i_a, i_b, i_c = inverse_clarke(i_alpha, i_beta)
    return (
        1.0 if i_a >= 0.0 else -1.0,
        1.0 if i_b >= 0.0 else -1.0,
        1.0 if i_c >= 0.0 else -1.0,
    )


def dead_time_projection(
    theta_e: float, signs: tuple[float, float, float]
) -> tuple[float, float]:
    """Project the phase-current polarity pattern onto the dq axes.

    This is the ``Dd``/``Dq`` pair the inverter subtracts from the commanded
    voltage. It is a square wave in the rotor angle, which is why dead-time
    distortion shows up as low-order harmonics rather than as a DC offset.

    Args:
        theta_e: Electrical rotor angle, radian.
        signs: The ``(sign_a, sign_b, sign_c)`` triple from `phase_signs`.

    Returns:
        The ``(Dd, Dq)`` pair.
    """
    sign_a, sign_b, sign_c = signs
    d = (
        math.cos(theta_e) * sign_a
        + math.cos(theta_e - _TWO_PI_OVER_3) * sign_b
        + math.cos(theta_e + _TWO_PI_OVER_3) * sign_c
    )
    q = (
        -math.sin(theta_e) * sign_a
        - math.sin(theta_e - _TWO_PI_OVER_3) * sign_b
        - math.sin(theta_e + _TWO_PI_OVER_3) * sign_c
    )
    return 2.0 * d / 3.0, 2.0 * q / 3.0
