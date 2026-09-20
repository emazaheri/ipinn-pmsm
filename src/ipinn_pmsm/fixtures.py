"""Golden vectors for the TypeScript twin that runs in the browser.

The browser port reimplements all of this by hand: the drive simulator, the
network, and forward-over-reverse derivatives in place of JAX. Two independent
implementations of the same arithmetic drift unless something holds them
together, and these files are that something.

What can and cannot be asserted is worth being explicit about.

**Exactly**: the simulator trace, the forward pass, the loss, and the
analytic gradient. These are deterministic arithmetic on the same inputs, and
the only difference is summation order.

**Not at all**: the weights a run starts from. JAX's threefry counter-based
PRNG has no reasonable TypeScript equivalent, so the browser initialises with
the same mulberry32 generator the other lab uses. The initial weights here are
therefore *test data*, injected into the port so the two can be compared, and
never shipped to a visitor. A deployed run matches the published numbers in
aggregate, not window by window.

**Statistically**: the per-window estimates after 150 rprop steps. Rprop moves
by the *sign* of the gradient, so the two implementations track each other
exactly until some component's sign disagrees near zero, then diverge by one
step size in that component. The converged flux linkage agrees far better than
the weights do, because it is the one strongly identified direction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .cases import load_case
from .normalize import Scaling
from .params import PinnConfig
from .pinn.losses import Fixed, loss_fn, ode_residuals
from .pinn.mlp import forward, init_params
from .pinn.train import reference_schedule_value, train_window
from .runner import simulate_case, windows_for

__all__ = ["export_all"]

SIGNIFICANT_DIGITS = 9
"""Enough to pin float32 arithmetic without bloating the files."""

FIXTURE_WINDOW = 120
"""A window well inside one operating point, away from every transition."""


def _round(value: Any) -> Any:
    """Round floats recursively, so the files stay small and diffable."""
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.{SIGNIFICANT_DIGITS}g}")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _round(value.tolist())
    return value


def _write(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write one fixture, pretty-printed and rounded."""
    path = directory / name
    path.write_text(json.dumps(_round(payload), indent=2) + "\n")
    return path


def _layer_arrays(params: list[dict[str, jax.Array]]) -> list[dict[str, Any]]:
    """Serialise the network layers, without the physics dict."""
    return [
        {"W": np.asarray(layer["W"]).tolist(), "B": np.asarray(layer["B"]).tolist()}
        for layer in params[:-1]
    ]


def _simulator_fixture(directory: Path) -> Path:
    """The drive trace, decimated again so the file stays under a megabyte."""
    case = load_case("a")
    trace = simulate_case(case)
    keep = slice(None, None, 10)
    return _write(
        directory,
        "simulator-trace.json",
        {
            "description": (
                "Case A drive trace, every tenth estimator sample. Fixed-step "
                "RK4, commanded voltages. Tolerance 1e-9 relative."
            ),
            "case": "a",
            "dt": trace.dt,
            "decimation": 10,
            "samples": int(trace.t[keep].size),
            "t": trace.t[keep],
            "i_d": trace.i_d[keep],
            "i_q": trace.i_q[keep],
            "v_d": trace.v_d[keep],
            "v_q": trace.v_q[keep],
            "omega_e": trace.omega_e[keep],
        },
    )


def _forward_fixture(directory: Path) -> Path:
    """One forward pass, with the time derivatives the physics term needs."""
    case = load_case("a")
    config = case.pinn
    window = windows_for(case, simulate_case(case))[FIXTURE_WINDOW]
    inputs = jnp.asarray(Scaling.identity().inputs(window))
    params = init_params(config)
    layers = params[:-1]

    def d_output(index: int) -> np.ndarray:
        def scalar(row: jax.Array) -> jax.Array:
            return forward(layers, row.reshape(1, -1))[0, index]

        return np.asarray(jax.vmap(lambda r: jax.grad(scalar)(r)[0])(inputs))

    return _write(
        directory,
        "mlp-forward.json",
        {
            "description": (
                "Forward pass and time derivatives for a fixed set of weights. "
                "The weights are test data: the browser initialises its own. "
                "Tolerance 1e-5 relative."
            ),
            "layers": list(config.layers),
            "activation": "swish",
            "weights": _layer_arrays(params),
            "inputs": np.asarray(inputs).tolist(),
            "outputs": np.asarray(forward(layers, inputs)).tolist(),
            "d_id_dt": d_output(0),
            "d_iq_dt": d_output(1),
        },
    )


def _gradient_fixture(directory: Path) -> Path:
    """Loss and its full gradient at a fixed, reproducible state."""
    case = load_case("a")
    config = case.pinn
    motor = case.known
    window = windows_for(case, simulate_case(case))[FIXTURE_WINDOW]
    scaling = Scaling.identity()
    inputs = jnp.asarray(scaling.inputs(window))
    targets = jnp.asarray(scaling.targets(window))
    fixed = Fixed(rs=motor.rs, ld=motor.ld, lq=motor.lq)
    params = init_params(config)

    parts = loss_fn(params, inputs, targets, fixed, config.alpha, config.beta)
    grads = jax.grad(
        lambda p: loss_fn(p, inputs, targets, fixed, config.alpha, config.beta).total
    )(params)
    r_d, r_q = ode_residuals(params[:-1], inputs, params[-1]["lambda_m"], fixed)

    return _write(
        directory,
        "gradients.json",
        {
            "description": (
                "Loss, residuals and the full gradient at the initial weights. "
                "Gradient tolerance 1e-4 relative; check the port against "
                "finite differences first, which needs no fixture at all."
            ),
            "fixed": {"rs": fixed.rs, "ld": fixed.ld, "lq": fixed.lq},
            "alpha": config.alpha,
            "beta": config.beta,
            "lambda_m": float(params[-1]["lambda_m"][0]),
            "weights": _layer_arrays(params),
            "inputs": np.asarray(inputs).tolist(),
            "targets": np.asarray(targets).tolist(),
            "loss": {
                "total": float(parts.total),
                "supervised": float(parts.supervised),
                "physics": float(parts.physics),
            },
            "residual_d": np.asarray(r_d).tolist(),
            "residual_q": np.asarray(r_q).tolist(),
            "grad_lambda_m": float(grads[-1]["lambda_m"][0]),
            "grad_weights": _layer_arrays([*grads[:-1], {}]),
        },
    )


def _optimizer_fixture(directory: Path) -> Path:
    """The rprop constants, and the constant the reference schedule reduces to."""
    return _write(
        directory,
        "optimizer.json",
        {
            "description": (
                "iRprop-. learning_rate is the initial per-parameter step. The "
                "reference chains a 'cosine one-cycle schedule' after it that "
                "measures as the constant epochs/100 at every step, so the "
                "port multiplies by that scalar rather than evaluating a curve."
            ),
            "rprop": {
                "initial_step": PinnConfig().learning_rate,
                "eta_minus": 0.5,
                "eta_plus": 1.2,
                "min_step": 1e-6,
                "max_step": 50.0,
                "zero_gradient_on_sign_flip": True,
            },
            "reference_schedule": {
                str(epochs): reference_schedule_value(epochs)
                for epochs in (50, 100, 150, 200, 300)
            },
        },
    )


def _trajectory_fixture(directory: Path) -> Path:
    """A whole window's convergence, plus the run summary for every case."""
    case = load_case("a")
    window = windows_for(case, simulate_case(case))[FIXTURE_WINDOW]
    result = train_window(window, case.known, case.pinn)

    summaries = {}
    for name in ("a", "b", "c", "d"):
        other = load_case(name)
        from .runner import run_case

        summary = run_case(other)
        slow, fast = summary.split_by_speed()
        summaries[name] = {
            "true_lambda_m": other.plant.lambda_m,
            "mean_error_pct": summary.mean_error_pct,
            "std_error_pct": summary.std_error_pct,
            "median_error_pct": summary.median_error_pct,
            "published_error_pct": other.published_error_pct,
            "windows": len(summary.results),
            "slow_half_error_pct": slow,
            "fast_half_error_pct": fast,
        }

    return _write(
        directory,
        "run-summary.json",
        {
            "description": (
                "One window's 150-epoch trajectory, and the aggregate for all "
                "four cases. Assert the trajectory's final value to 1e-3 "
                "absolute and its shape, not its every step: rprop makes the "
                "two implementations diverge once any gradient sign disagrees."
            ),
            "window_index": FIXTURE_WINDOW,
            "window_centre": window.centre,
            "epochs": case.pinn.epochs,
            "lambda_history": result.lambda_history,
            "supervised_history": result.supervised_history,
            "physics_history": result.physics_history,
            "cases": summaries,
        },
    )


def export_all(directory: Path) -> list[Path]:
    """Write every fixture into a directory, creating it if needed.

    Args:
        directory: Destination. Usually the website's
            ``components/sections/observer/fixtures``.

    Returns:
        The paths written, in order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return [
        _simulator_fixture(directory),
        _forward_fixture(directory),
        _gradient_fixture(directory),
        _optimizer_fixture(directory),
        _trajectory_fixture(directory),
    ]
