"""Per-window scaling of the estimator's inputs and outputs.

The reference implementation feeds the network absolute time in seconds
alongside voltages in the hundreds, and asks it to predict currents of order
one alongside an angular velocity of order four hundred. Within a 13 ms window
the time input varies by about one percent of its own magnitude. Every one of
those is a conditioning problem, and together they are the most likely reason
the per-window spread sits near five percent.

Scaling is optional because the published numbers were produced without it.
`Scaling.identity` reproduces the reference exactly; `Scaling.for_window`
applies the corrections. Which one is in use is recorded in every result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windows import Window

__all__ = ["Scaling"]


@dataclass(frozen=True, slots=True)
class Scaling:
    """Affine maps between physical units and the network's own units.

    The network sees ``[(t - t0) / t_span, vd / v, vq / v]`` and predicts
    ``[id / i, iq / i, we / w]``.

    Attributes:
        t0: Time origin, second. Subtracted before scaling.
        t_span: Time scale, second. This must be the window's true span,
            ``(B - 1) * dt``, not its nominal length.
        v: Voltage scale, volt.
        i: Current scale, ampere.
        w: Angular velocity scale, radian per second.
        enabled: False for the reference preset, where every scale is one and
            the time origin is zero.
    """

    t0: float = 0.0
    t_span: float = 1.0
    v: float = 1.0
    i: float = 1.0
    w: float = 1.0
    enabled: bool = False

    @classmethod
    def identity(cls) -> Scaling:
        """Return the no-op scaling the reference preset uses."""
        return cls()

    @classmethod
    def for_window(cls, window: Window, v_max: float) -> Scaling:
        """Derive scales for one window.

        Voltage uses the drive's own limit rather than a window statistic, so
        that runs stay comparable and the estimator is not quietly told the
        operating point through its normalisation. Current and speed use window
        magnitudes, floored at one so a near-zero window cannot blow the scale
        up.

        Args:
            window: The window to scale.
            v_max: The inverter's largest realisable dq voltage, volt.

        Returns:
            A scaling with `enabled` set.
        """
        i_rms = float(np.sqrt(np.mean(window.i_d**2 + window.i_q**2)))
        w_mean = float(np.mean(np.abs(window.omega_e)))
        return cls(
            t0=float(window.t[0]),
            t_span=window.span,
            v=max(v_max, 1.0),
            i=max(i_rms, 1.0),
            w=max(w_mean, 1.0),
            enabled=True,
        )

    def inputs(self, window: Window) -> np.ndarray:
        """Return the scaled ``[t, vd, vq]`` matrix for a window.

        Args:
            window: The window to encode.

        Returns:
            Shape ``(B, 3)``, float64.
        """
        return np.column_stack(
            [
                (window.t - self.t0) / self.t_span,
                window.v_d / self.v,
                window.v_q / self.v,
            ]
        )

    def targets(self, window: Window) -> np.ndarray:
        """Return the scaled ``[id, iq, we]`` matrix for a window.

        Args:
            window: The window to encode.

        Returns:
            Shape ``(B, 3)``, float64.
        """
        return np.column_stack(
            [window.i_d / self.i, window.i_q / self.i, window.omega_e / self.w]
        )
