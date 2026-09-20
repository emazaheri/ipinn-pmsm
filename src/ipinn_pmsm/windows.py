"""Cutting a record into the short spans each network is fitted to.

One subtlety governs everything downstream. A window of ``B`` samples taken at
spacing ``dt`` spans ``(B - 1) * dt``, not ``B * dt``. At the reference
settings that is 13 ms, not the nominal 15 ms the window length implies. When
the estimator normalises time to the unit interval it must divide the derivative
by the span, and using the nominal length instead puts a 15.4 percent bias on
every ``did/dt`` and ``diq/dt``, which lands straight on the ``Ld * did/dt``
term of the residual. `Window.span` is the only value allowed to be used for
that, and a test asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .params import WindowConfig
from .simulate import Trace

__all__ = ["Window", "split"]

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Window:
    """One span of the record, with the signals the estimator is shown.

    Attributes:
        index: Position in the sequence of windows, from zero.
        t: Sample times, second, absolute rather than relative to the window.
        i_d: d-axis current, ampere.
        i_q: q-axis current, ampere.
        v_d: d-axis voltage, volt.
        v_q: q-axis voltage, volt.
        omega_e: Electrical angular velocity, radian per second.
        dt: Time between samples, second.
    """

    index: int
    t: F64
    i_d: F64
    i_q: F64
    v_d: F64
    v_q: F64
    omega_e: F64
    dt: float

    def __len__(self) -> int:
        """Number of samples in the window."""
        return int(self.t.size)

    @property
    def span(self) -> float:
        """Time from the first sample to the last, second.

        This is ``(len(self) - 1) * dt``, which is the value a normalised time
        coordinate must be scaled by. It is not the nominal window length.
        """
        return (len(self) - 1) * self.dt

    @property
    def centre(self) -> float:
        """Midpoint of the window, second. Used as the estimate's x coordinate."""
        return float(0.5 * (self.t[0] + self.t[-1]))

    @property
    def mean_speed(self) -> float:
        """Mean electrical angular velocity over the window, radian per second.

        The sensitivity of the measurement to the flux linkage is exactly this
        number, so it is what says whether the window can identify the
        parameter at all.
        """
        return float(np.mean(self.omega_e))


def split(
    trace: Trace,
    config: WindowConfig,
    *,
    use_applied: bool = False,
) -> list[Window]:
    """Cut a trace into windows.

    Windows that would run past the end of the record are dropped rather than
    truncated, so every window holds the same number of samples and the
    estimates stay comparable.

    Args:
        trace: The simulated record.
        config: Window length and stride.
        use_applied: Show the estimator the applied voltages rather than the
            commanded ones. The commanded pair is the honest default: it is
            what a drive can log without extra sensors, and the difference
            between the two is exactly the inverter disturbance the estimator
            is not told about.

    Returns:
        The windows, in time order.
    """
    length, stride = config.sample_counts(trace.dt)
    v_d = trace.v_d_applied if use_applied else trace.v_d
    v_q = trace.v_q_applied if use_applied else trace.v_q

    windows: list[Window] = []
    start = 0
    while start + length <= len(trace):
        stop = start + length
        windows.append(
            Window(
                index=len(windows),
                t=trace.t[start:stop],
                i_d=trace.i_d[start:stop],
                i_q=trace.i_q[start:stop],
                v_d=v_d[start:stop],
                v_q=v_q[start:stop],
                omega_e=trace.omega_e[start:stop],
                dt=trace.dt,
            )
        )
        start += stride
    return windows
