"""Validation cases, loaded from TOML rather than copied into four scripts.

The reference study kept these as four 15 KB files that differed in two or
three literals each, which is how case C came to change something nobody
writing it down afterwards remembered.

The important structure here is the separation of **plant** from **known**. The
plant is the machine the simulator runs. The known parameters are what the
estimator is told. Case C is the case that makes the distinction matter: it
raises the plant's stator resistance by thirteen percent while the estimator
keeps working from the cold value. That single line is the difference between
"2.13 percent error" and "1.02 percent error", and without it the case looks
mysteriously easy.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .params import InverterParams, MotorParams, PinnConfig, SimParams, WindowConfig

__all__ = ["CASES_DIR", "Case", "available_cases", "load_case"]

CASES_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True, slots=True)
class Noise:
    """Measurement noise settings.

    Attributes:
        current_percent: Current noise as a percentage of current RMS.
        voltage_percent: Voltage noise as a percentage of voltage RMS.
        seed: PRNG seed.
        legacy_rng: Use NumPy's legacy generator, reproducing the reference.
    """

    current_percent: float = 0.0
    voltage_percent: float = 0.0
    seed: int = 42
    legacy_rng: bool = True

    @property
    def enabled(self) -> bool:
        """Whether any noise is actually added."""
        return self.current_percent > 0.0 or self.voltage_percent > 0.0


@dataclass(frozen=True, slots=True)
class Case:
    """One validation case, end to end.

    Attributes:
        name: Short identifier, for example ``a``.
        title: One-line description.
        plant: The machine the simulator runs.
        known: What the estimator is told. Its ``lambda_m`` is ignored, since
            that is the quantity being recovered.
        inverter: Inverter parameters.
        sim: Simulation settings.
        window: Windowing settings.
        pinn: Estimator settings.
        noise: Measurement noise settings.
        published_error_pct: The mean absolute error reported in the original
            study, for comparison. None when there is nothing to compare to.
    """

    name: str
    title: str
    plant: MotorParams
    known: MotorParams
    inverter: InverterParams
    sim: SimParams
    window: WindowConfig
    pinn: PinnConfig
    noise: Noise
    published_error_pct: float | None = None

    def tuned(self) -> Case:
        """Return the same case with the corrected estimator preset.

        Returns:
            A copy whose `pinn` has normalisation enabled.
        """
        return replace(self, pinn=replace(self.pinn, normalize=True))


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a TOML table, or an empty one when absent."""
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{key}] must be a table, got {type(value).__name__}")
    return value


def load_case(name: str, directory: Path | None = None) -> Case:
    """Load a case from ``configs/case_<name>.toml``.

    The ``[known]`` table inherits every field of ``[plant]`` that it does not
    itself set, so a config only has to state what the estimator gets *wrong*.
    That keeps the interesting line, case C's stator resistance, impossible to
    miss when reading the file.

    Args:
        name: Case identifier, for example ``a``.
        directory: Where to look. Defaults to the packaged ``configs/``.

    Returns:
        The assembled case.

    Raises:
        FileNotFoundError: If no such config exists.
    """
    path = (directory or CASES_DIR) / f"case_{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no case config at {path}")

    data = tomllib.loads(path.read_text())
    plant_fields = _section(data, "plant")
    plant = MotorParams(**plant_fields)

    # The estimator inherits the plant unless the case deliberately lies to it.
    known = MotorParams(**{**plant_fields, **_section(data, "known")})

    return Case(
        name=name,
        title=str(data.get("title", name)),
        plant=plant,
        known=known,
        inverter=InverterParams(**_section(data, "inverter")),
        sim=SimParams(**_section(data, "sim")),
        window=WindowConfig(**_section(data, "window")),
        pinn=PinnConfig(**_section(data, "pinn")),
        noise=Noise(**_section(data, "noise")),
        published_error_pct=data.get("published_error_pct"),
    )


def available_cases(directory: Path | None = None) -> list[str]:
    """List the case identifiers that have configs.

    Args:
        directory: Where to look. Defaults to the packaged ``configs/``.

    Returns:
        Sorted identifiers, for example ``["a", "b", "c", "d"]``.
    """
    root = directory or CASES_DIR
    return sorted(p.stem.removeprefix("case_") for p in root.glob("case_*.toml"))
