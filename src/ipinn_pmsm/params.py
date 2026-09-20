"""Parameter records for the machine, the drive and the estimator.

Everything is a frozen dataclass so a configuration can be passed around,
hashed, printed and serialised without any chance of a stage mutating the
settings of the stage before it. The research code these replace carried the
same values as bare dicts reassembled at the top of each script, which is how
four scripts that were meant to differ in two numbers came to differ in five.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "InverterParams",
    "MotorParams",
    "PinnConfig",
    "SimParams",
    "WindowConfig",
]


@dataclass(frozen=True, slots=True)
class MotorParams:
    """Electrical and magnetic parameters of the permanent-magnet machine.

    Attributes:
        rs: Stator resistance, ohm.
        ld: Unsaturated d-axis inductance, henry.
        lq: Unsaturated q-axis inductance, henry.
        lambda_m: Permanent-magnet flux linkage, weber. This is the quantity
            the inverse PINN recovers, and the one demagnetization reduces.
        pole_pairs: Number of pole pairs, so that omega_e = pole_pairs * omega_m.
        beta_d: d-axis saturation coefficient. Zero disables saturation.
        beta_q: q-axis saturation coefficient. Zero disables saturation.
    """

    rs: float = 0.46
    ld: float = 0.00466
    lq: float = 0.0135
    lambda_m: float = 0.206
    pole_pairs: int = 4
    beta_d: float = 0.0
    beta_q: float = 0.0

    def saturated(self, i_d: float, i_q: float) -> tuple[float, float]:
        """Return the saturated inductances at an operating current.

        The model is ``L = L0 / (1 + beta * (id^2 + iq^2))``. Note that the
        original ``pmsm_foc_simulator_v4.py`` header documented this as
        ``L0 * (1 - beta * |iq|)``, which the code never implemented. The
        divided form below is what produced every published result.

        Args:
            i_d: d-axis current, ampere.
            i_q: q-axis current, ampere.

        Returns:
            The saturated ``(ld, lq)`` pair, henry.
        """
        i_squared = i_d * i_d + i_q * i_q
        return (
            self.ld / (1.0 + self.beta_d * i_squared),
            self.lq / (1.0 + self.beta_q * i_squared),
        )


@dataclass(frozen=True, slots=True)
class InverterParams:
    """Two-level voltage-source inverter non-idealities.

    Attributes:
        v_dc: DC-link voltage, volt.
        r_cd: Lumped conduction-drop resistance, ohm. Zero for an ideal bridge.
        v_dead: Dead-time voltage magnitude, volt. Zero for an ideal bridge.
    """

    v_dc: float = 400.0
    r_cd: float = 0.0
    v_dead: float = 0.0

    @property
    def v_max_dq(self) -> float:
        """Largest realisable dq voltage magnitude, volt."""
        return float(self.v_dc / 3.0**0.5)


@dataclass(frozen=True, slots=True)
class SimParams:
    """Fixed-step simulation settings.

    Attributes:
        ts: Control and integration step, second.
        t_sim: Total simulated time, second.
        decimation: Keep every nth sample when handing data to the estimator.
            The reference study integrates at 100 us and estimates at 1 ms.
        legacy_time_axis: Label samples using ``linspace(0, t_sim, num_steps)``
            rather than ``step * ts``. The reference study did the former,
            which spaces its time stamps by ``t_sim / (num_steps - 1)``, 50
            parts per million wider than the step it actually integrates with.
            The drift is irrelevant to the physics but it changes
            ``int(window_length / dt)`` from 15 to 14, which is the whole
            reason the published run has 142 windows and not 133. Set True to
            reproduce the published window count.
    """

    ts: float = 1e-4
    t_sim: float = 2.0
    decimation: int = 10
    legacy_time_axis: bool = False

    @property
    def num_steps(self) -> int:
        """Number of integration steps."""
        return round(self.t_sim / self.ts)

    @property
    def sample_dt(self) -> float:
        """Time between the samples the estimator actually sees, second."""
        if self.legacy_time_axis:
            return self.t_sim / (self.num_steps - 1) * self.decimation
        return self.ts * self.decimation

    def stamp(self, step: int) -> float:
        """Return the time label for an integration step, second.

        Args:
            step: Zero-based integration step index.

        Returns:
            The sample's time stamp. Only the label changes between the two
            axis conventions; the integration always advances by `ts`.
        """
        if self.legacy_time_axis:
            return step * self.t_sim / (self.num_steps - 1)
        return step * self.ts


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """How the record is cut into estimation windows.

    Attributes:
        length_s: Window length, second.
        stride_s: Distance between window starts, second. When equal to
            ``length_s`` the windows tile without overlap, which is what the
            reference study did despite calling them sliding.
    """

    length_s: float = 0.015
    stride_s: float = 0.015

    def sample_counts(self, sample_dt: float) -> tuple[int, int]:
        """Return ``(samples_per_window, stride_in_samples)``.

        Args:
            sample_dt: Time between estimator samples, second.

        Raises:
            ValueError: If either count rounds to less than one sample.
        """
        length = int(self.length_s / sample_dt)
        stride = int(self.stride_s / sample_dt)
        if length < 2:
            raise ValueError(
                f"window of {self.length_s}s holds {length} samples at "
                f"dt={sample_dt}s; a time derivative needs at least 2"
            )
        if stride < 1:
            raise ValueError(f"stride of {self.stride_s}s rounds to {stride} samples")
        return length, stride


@dataclass(frozen=True, slots=True)
class PinnConfig:
    """Network, loss and optimiser settings for the inverse PINN.

    Attributes:
        hidden: Hidden layer widths. Input is always 3 (t, vd, vq) and output
            always 3 (id, iq, we).
        epochs: Optimiser steps per window.
        learning_rate: Initial rprop step size.
        alpha: Weight on the supervised term.
        beta: Weight on the physics residual term.
        lambda_init: Starting guess for the flux linkage, weber. Deliberately
            wrong, so that convergence is visible.
        seed: PRNG seed for weight initialisation.
        normalize: Scale inputs and outputs per window. False reproduces the
            reference study, which fed absolute time and raw volts to the
            network. See README, "Two presets".
    """

    hidden: tuple[int, ...] = (32, 32)
    epochs: int = 150
    learning_rate: float = 1e-3
    alpha: float = 1.0
    beta: float = 1e-3
    lambda_init: float = 0.1
    seed: int = 0
    normalize: bool = False

    @property
    def layers(self) -> tuple[int, ...]:
        """Full layer widths including the fixed input and output sizes."""
        return (3, *self.hidden, 3)


def to_dict(params: Any) -> dict[str, Any]:
    """Return a plain dict for a frozen dataclass, for logging and fixtures.

    Args:
        params: Any dataclass instance defined in this module.

    Returns:
        A JSON-serialisable mapping of its fields.
    """
    return asdict(params)
