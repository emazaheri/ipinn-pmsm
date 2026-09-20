"""Command line entry point: run cases, export fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cases import available_cases, load_case
from .pinn.mlp import count_parameters
from .runner import run_case

__all__ = ["main"]


def _run(args: argparse.Namespace) -> int:
    """Run one case or all of them and print a comparison table."""
    names = available_cases() if args.case == "all" else [args.case]
    preset = "tuned" if args.tuned else "reference"

    print(f"preset: {preset}")
    header = (
        f"{'case':5s} {'mean%':>7s} {'std%':>7s} {'median%':>8s} "
        f"{'published%':>11s} {'delta':>7s} {'slow%':>7s} {'fast%':>7s} {'s':>6s}"
    )
    print(header)
    print("-" * len(header))

    failures = 0
    for name in names:
        case = load_case(name)
        if args.tuned:
            case = case.tuned()
        summary = run_case(case)
        slow, fast = summary.split_by_speed()
        published = case.published_error_pct
        delta = (
            f"{summary.mean_error_pct - published:+7.2f}"
            if published is not None
            else f"{'':>7s}"
        )
        print(
            f"{name:5s} {summary.mean_error_pct:7.2f} {summary.std_error_pct:7.2f} "
            f"{summary.median_error_pct:8.2f} "
            f"{published if published is not None else float('nan'):11.2f} {delta} "
            f"{slow:7.2f} {fast:7.2f} {summary.elapsed_s:6.1f}"
        )
        outside = (
            published is not None
            and abs(summary.mean_error_pct - published) > args.tolerance
        )
        if args.check and outside:
            failures += 1

    if args.check and failures:
        print(
            f"\n{failures} case(s) outside +/-{args.tolerance} points", file=sys.stderr
        )
        return 1
    return 0


def _describe(args: argparse.Namespace) -> int:
    """Print what a case actually configures, including where it lies."""
    case = load_case(args.case)
    print(f"{case.name}: {case.title}")
    print(
        f"  plant   rs={case.plant.rs} ld={case.plant.ld} lq={case.plant.lq} "
        f"lambda_m={case.plant.lambda_m} "
        f"beta=({case.plant.beta_d}, {case.plant.beta_q})"
    )
    print(
        f"  known   rs={case.known.rs} ld={case.known.ld} lq={case.known.lq} "
        f"beta=({case.known.beta_d}, {case.known.beta_q})"
    )

    lies = [
        f"{name}: plant {getattr(case.plant, name)} "
        f"vs estimator {getattr(case.known, name)}"
        for name in ("rs", "ld", "lq")
        if getattr(case.plant, name) != getattr(case.known, name)
    ]
    print(f"  mismatch  {'; '.join(lies) if lies else 'none'}")
    if case.plant.beta_d or case.plant.beta_q:
        print(
            "  mismatch  saturation: the estimator's residual uses unsaturated Ld, Lq"
        )
    if case.noise.enabled:
        print(
            f"  noise   {case.noise.current_percent}% current, "
            f"{case.noise.voltage_percent}% voltage, seed {case.noise.seed}"
        )
    print(
        f"  network {case.pinn.layers}, {count_parameters(case.pinn)} trainable, "
        f"{case.pinn.epochs} epochs/window"
    )
    return 0


def _export(args: argparse.Namespace) -> int:
    """Write golden vectors for the TypeScript twin."""
    from .fixtures import export_all

    out = Path(args.out)
    written = export_all(out)
    for path in written:
        print(f"wrote {path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="ipinn-pmsm",
        description="Inverse PINN recovery of a PMSM's magnet flux linkage.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one case or all of them")
    run.add_argument("--case", default="all", help="case name, or 'all'")
    run.add_argument("--tuned", action="store_true", help="use the corrected preset")
    run.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if a case misses its published figure",
    )
    run.add_argument(
        "--tolerance", type=float, default=0.15, help="points, for --check"
    )
    run.set_defaults(func=_run)

    describe = sub.add_parser("describe", help="print what a case configures")
    describe.add_argument("--case", default="a")
    describe.set_defaults(func=_describe)

    export = sub.add_parser("export-fixtures", help="write golden vectors")
    export.add_argument("--out", required=True, help="output directory")
    export.set_defaults(func=_export)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
