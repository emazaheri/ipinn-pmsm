"""Running a case end to end: simulate, window, estimate, score."""

from __future__ import annotations

from collections.abc import Callable

from .cases import Case
from .pinn.train import RunSummary, WindowResult, train_windows
from .scenarios import reference_scenario
from .simulate import Trace, simulate
from .windows import Window, split

__all__ = ["run_case", "simulate_case", "windows_for"]


def simulate_case(case: Case, speed_scale: float = 1.0) -> Trace:
    """Simulate the plant for a case, noise included.

    Args:
        case: The case to run.
        speed_scale: Multiplier on the dynamometer speed. Zero holds the rotor
            still, which makes the flux linkage unidentifiable.

    Returns:
        The decimated trace the estimator will be shown.
    """
    trace = simulate(
        case.plant,
        case.inverter,
        case.sim,
        *reference_scenario(speed_scale=speed_scale),
    )
    if case.noise.enabled:
        trace = trace.with_noise(
            case.noise.current_percent,
            case.noise.voltage_percent,
            case.noise.seed,
            legacy_rng=case.noise.legacy_rng,
        )
    return trace


def windows_for(case: Case, trace: Trace) -> list[Window]:
    """Cut a trace into the case's windows.

    Args:
        case: The case, for its window configuration.
        trace: The simulated record.

    Returns:
        The windows, in time order.
    """
    return split(trace, case.window)


def run_case(
    case: Case,
    speed_scale: float = 1.0,
    on_result: Callable[[WindowResult], None] | None = None,
) -> RunSummary:
    """Simulate, window and estimate, then score against the truth.

    Args:
        case: The case to run.
        speed_scale: Multiplier on the dynamometer speed.
        on_result: Optional callback invoked with each window result as it
            lands, for progress reporting.

    Returns:
        The run summary, scored against the plant's true flux linkage.
    """
    trace = simulate_case(case, speed_scale)
    return train_windows(
        windows_for(case, trace),
        case.known,
        case.pinn,
        case.inverter.v_max_dq,
        case.plant.lambda_m,
        on_result,
    )
