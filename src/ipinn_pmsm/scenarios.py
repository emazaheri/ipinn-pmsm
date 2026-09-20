"""The excitation the drive is asked to follow.

The reference scenario walks eight operating combinations over two seconds,
0.25 s each, built so that the fastest-switching quantity is the one the
estimator is least sensitive to and the slowest is the one it is most
sensitive to.

    id:    0, -2 A          switches 7 times
    iq:    8, 12 A          switches 3 times
    speed: 500, 1000 rpm    switches once, at t = 1.0 s

That last row is the interesting one. The flux linkage enters only the q-axis
residual, multiplied by the electrical angular velocity, so the sensitivity of
the measurement to the parameter is

    d(vq) / d(lambda_m) = omega_e = pole_pairs * omega_m

At 500 rpm with four pole pairs that is 209 V per weber; at 1000 rpm it is 419.
The same voltage error therefore maps to twice the flux error in the first half
of the record as in the second. The scatter in the estimate should visibly
narrow when the speed steps, and at standstill the parameter is not
identifiable at all. That is a prediction the plots either confirm or refute.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .simulate import Schedule

__all__ = [
    "REFERENCE_SEGMENTS",
    "Segment",
    "constant",
    "reference_scenario",
    "rpm_to_rad_s",
    "staircase",
]


def rpm_to_rad_s(rpm: float) -> float:
    """Convert revolutions per minute to mechanical radians per second.

    Args:
        rpm: Speed in revolutions per minute.

    Returns:
        Speed in radians per second.
    """
    return rpm * 2.0 * math.pi / 60.0


@dataclass(frozen=True, slots=True)
class Segment:
    """One operating combination held for a fixed interval.

    Attributes:
        until: The segment ends at this time, second.
        i_d: d-axis current reference, ampere.
        i_q: q-axis current reference, ampere.
        rpm: Dynamometer speed, revolutions per minute.
    """

    until: float
    i_d: float
    i_q: float
    rpm: float


REFERENCE_SEGMENTS: tuple[Segment, ...] = (
    Segment(0.25, 0.0, 8.0, 500.0),
    Segment(0.50, -2.0, 8.0, 500.0),
    Segment(0.75, 0.0, 12.0, 500.0),
    Segment(1.00, -2.0, 12.0, 500.0),
    Segment(1.25, 0.0, 8.0, 1000.0),
    Segment(1.50, -2.0, 8.0, 1000.0),
    Segment(1.75, 0.0, 12.0, 1000.0),
    Segment(2.00, -2.0, 12.0, 1000.0),
)
"""The eight combinations of the reference study, in order."""


def staircase(segments: Sequence[Segment], field: str) -> Schedule:
    """Build a zero-order-hold schedule from a segment list.

    Args:
        segments: Segments in increasing order of `Segment.until`.
        field: Which attribute to read, one of ``i_d``, ``i_q`` or ``rpm``.

    Returns:
        A function of time returning that attribute's value. Times past the
        last segment hold the last value rather than raising, so a simulation
        that overruns by one floating-point step does not fail.
    """
    last = segments[-1]

    def schedule(t: float) -> float:
        for segment in segments:
            if t < segment.until:
                return float(getattr(segment, field))
        return float(getattr(last, field))

    return schedule


def constant(value: float) -> Schedule:
    """Return a schedule that always yields the same value.

    Args:
        value: The constant.

    Returns:
        A function of time ignoring its argument.
    """

    def schedule(_t: float) -> float:
        return value

    return schedule


def reference_scenario(
    segments: Sequence[Segment] = REFERENCE_SEGMENTS,
    speed_scale: float = 1.0,
) -> tuple[Schedule, Schedule, Schedule]:
    """Return the three schedules the simulator needs.

    Args:
        segments: Segments to follow. Defaults to `REFERENCE_SEGMENTS`.
        speed_scale: Multiplier on every speed, so the identifiability of the
            flux linkage can be varied without rewriting the scenario. A scale
            of zero holds the rotor still, which makes the parameter
            unobservable: the demo turns this into a control.

    Returns:
        The ``(i_d_ref, i_q_ref, speed_ref)`` triple. Speed is in mechanical
        radians per second, the other two in amperes.
    """
    i_d = staircase(segments, "i_d")
    i_q = staircase(segments, "i_q")
    rpm = staircase(segments, "rpm")

    def speed(t: float) -> float:
        return rpm_to_rad_s(rpm(t) * speed_scale)

    return i_d, i_q, speed
