"""Fitting one network per window, and reading the flux linkage off the result.

The optimiser is rprop, which is the reason the unnormalised reference works at
all. Rprop ignores gradient magnitude and moves each parameter by a per
parameter step size whose *sign* follows the gradient. The flux linkage's
gradient is of order ``beta / B * r_q * omega_e``, which with ``beta = 1e-3``
and ``omega_e`` in the hundreds sits orders of magnitude away from a typical
weight gradient. Any magnitude-following optimiser would need those two scaled
to within reach of each other. Rprop does not care, so a badly conditioned
formulation still converges. The optimiser is doing the compensating, not the
formulation.

A note on the schedule. The reference chains
``scale_by_schedule(cosine_onecycle_schedule(1e-3, epochs, 1e-4, 0.1))`` after
rprop. Measured against optax 0.2.8, that expression returns ``epochs / 100``
for every step: one unique value across the whole run. It is a constant, not a
schedule. `REFERENCE_SCHEDULE` reproduces it, quirk and all, including the
consequence that raising the epoch count also raises every step size. `TUNED`
uses a flat one, so epochs and step size stop being the same knob.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from ..normalize import Scaling
from ..params import MotorParams, PinnConfig
from ..windows import Window
from .losses import Fixed, loss_fn
from .mlp import Params, init_params

__all__ = ["WindowResult", "reference_schedule_value", "train_window", "train_windows"]

F64 = npt.NDArray[np.float64]


def reference_schedule_value(epochs: int) -> float:
    """Return the constant the reference's "cosine one-cycle" actually yields.

    Measured, not derived: ``cosine_onecycle_schedule(1e-3, E, 1e-4, 0.1)``
    evaluates to ``E / 100`` at every step, for every ``E`` tested between 50
    and 300.

    Args:
        epochs: Epoch count, which the reference passes as ``peak_value``.

    Returns:
        The multiplier applied to every rprop update.
    """
    return epochs / 100.0


@dataclass(frozen=True, slots=True)
class WindowResult:
    """What fitting one window produced.

    Attributes:
        index: Window position in the record.
        centre: Window midpoint, second.
        mean_speed: Mean electrical angular velocity, radian per second. This
            is the sensitivity of the measurement to the flux linkage, so a
            window with a low value cannot be expected to identify it.
        lambda_m: Final flux-linkage estimate, weber.
        lambda_history: Estimate after each epoch, shape ``(epochs,)``.
        supervised_history: Supervised loss after each epoch.
        physics_history: Physics loss after each epoch.
    """

    index: int
    centre: float
    mean_speed: float
    lambda_m: float
    lambda_history: F64 = field(repr=False)
    supervised_history: F64 = field(repr=False)
    physics_history: F64 = field(repr=False)

    def error_pct(self, truth: float) -> float:
        """Absolute error against a known value, in percent.

        Args:
            truth: The true flux linkage, weber.

        Returns:
            ``100 * |estimate - truth| / truth``.
        """
        return 100.0 * abs(self.lambda_m - truth) / truth


@lru_cache(maxsize=8)
def _build_optimizer(config: PinnConfig) -> optax.GradientTransformation:
    """Chain rprop with the scale the preset calls for."""
    scale = 1.0 if config.normalize else reference_schedule_value(config.epochs)
    return optax.chain(
        optax.rprop(config.learning_rate),
        optax.scale(scale),
    )


@lru_cache(maxsize=8)
def _compiled_step(
    config: PinnConfig, optimizer: optax.GradientTransformation
) -> Callable[..., tuple[Params, optax.OptState, jax.Array, jax.Array]]:
    """Return a jitted training step, compiled once per configuration.

    The window's data and unit scales are arguments rather than closed-over
    constants, so all 142 windows share one compilation. Closing over them
    instead costs a full XLA trace per window, which dominated the runtime by
    roughly thirty to one before this was hoisted.

    Args:
        config: Loss weights. Hashable, which is why `PinnConfig` is frozen.
        optimizer: The chained transformation to apply.

    Returns:
        A function of ``(params, opt_state, inputs, targets, fixed)``.
    """

    def objective(
        p: Params, inputs: jax.Array, targets: jax.Array, fixed: Fixed
    ) -> jax.Array:
        return loss_fn(p, inputs, targets, fixed, config.alpha, config.beta).total

    @jax.jit
    def step(
        p: Params,
        state: optax.OptState,
        inputs: jax.Array,
        targets: jax.Array,
        fixed: Fixed,
    ) -> tuple[Params, optax.OptState, jax.Array, jax.Array]:
        grads = jax.grad(objective)(p, inputs, targets, fixed)
        updates, state = optimizer.update(grads, state)
        p = optax.apply_updates(p, updates)
        parts = loss_fn(p, inputs, targets, fixed, config.alpha, config.beta)
        return p, state, parts.supervised, parts.physics

    return step


def _fixed_for(motor: MotorParams, scaling: Scaling, window: Window) -> Fixed:
    """Assemble the constants the loss needs for one window."""
    if not scaling.enabled:
        return Fixed(rs=motor.rs, ld=motor.ld, lq=motor.lq)
    return Fixed(
        rs=motor.rs,
        ld=motor.ld,
        lq=motor.lq,
        time_span=window.span,
        current_scale=scaling.i,
        speed_scale=scaling.w,
        voltage_scale=scaling.v,
    )


def train_window(
    window: Window,
    motor: MotorParams,
    config: PinnConfig,
    scaling: Scaling | None = None,
) -> WindowResult:
    """Fit a fresh network to one window and return the recovered parameter.

    Every window starts from the same seed and the same deliberately wrong
    flux linkage, so the estimates are independent: nothing carries over from
    the previous window, and a good estimate cannot be inherited.

    Args:
        window: The span to fit.
        motor: The machine. Only ``rs``, ``ld`` and ``lq`` are used; the true
            ``lambda_m`` is never shown to the estimator.
        config: Network, loss and optimiser settings.
        scaling: Input and output scaling. Defaults to the reference's no-op.

    Returns:
        The per-window result, histories included.
    """
    scaling = scaling or Scaling.identity()
    fixed = _fixed_for(motor, scaling, window)

    inputs = jnp.asarray(scaling.inputs(window))
    targets = jnp.asarray(scaling.targets(window))

    params = init_params(config)
    optimizer = _build_optimizer(config)
    opt_state = optimizer.init(params)
    step = _compiled_step(config, optimizer)

    lambda_history = np.empty(config.epochs, dtype=np.float64)
    supervised_history = np.empty(config.epochs, dtype=np.float64)
    physics_history = np.empty(config.epochs, dtype=np.float64)

    for epoch in range(config.epochs):
        params, opt_state, supervised, physics = step(
            params, opt_state, inputs, targets, fixed
        )
        lambda_history[epoch] = float(params[-1]["lambda_m"][0])
        supervised_history[epoch] = float(supervised)
        physics_history[epoch] = float(physics)

    return WindowResult(
        index=window.index,
        centre=window.centre,
        mean_speed=window.mean_speed,
        lambda_m=float(lambda_history[-1]),
        lambda_history=lambda_history,
        supervised_history=supervised_history,
        physics_history=physics_history,
    )


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregate statistics over a whole record.

    Attributes:
        results: Every window result, in time order.
        truth: The true flux linkage the estimates are scored against, weber.
        elapsed_s: Wall-clock time for the run, second.
    """

    results: list[WindowResult]
    truth: float
    elapsed_s: float

    @property
    def estimates(self) -> F64:
        """Final estimate per window, weber."""
        return np.array([r.lambda_m for r in self.results], dtype=np.float64)

    @property
    def errors_pct(self) -> F64:
        """Absolute error per window, percent."""
        return 100.0 * np.abs(self.estimates - self.truth) / self.truth

    @property
    def mean_error_pct(self) -> float:
        """Mean absolute error, percent. The headline number."""
        return float(np.mean(self.errors_pct))

    @property
    def std_error_pct(self) -> float:
        """Standard deviation of the absolute error, percent."""
        return float(np.std(self.errors_pct))

    @property
    def median_error_pct(self) -> float:
        """Median absolute error, percent. Robust to a diverged window."""
        return float(np.median(self.errors_pct))

    def split_by_speed(self) -> tuple[float, float]:
        """Return mean error for the slow and fast halves of the record.

        Sensitivity to the flux linkage is the electrical angular velocity, so
        the slower half should be measurably worse. Splitting at the median
        speed turns that prediction into a number.

        Returns:
            The ``(slow_mean_pct, fast_mean_pct)`` pair. Both are NaN if every
            window ran at the same speed.
        """
        speeds = np.array([r.mean_speed for r in self.results])
        pivot = float(np.median(speeds))
        slow, fast = speeds < pivot, speeds > pivot
        errors = self.errors_pct
        return (
            float(np.mean(errors[slow])) if slow.any() else float("nan"),
            float(np.mean(errors[fast])) if fast.any() else float("nan"),
        )


def train_windows(
    windows: list[Window],
    motor: MotorParams,
    config: PinnConfig,
    v_max: float,
    truth: float,
    on_result: object = None,
) -> RunSummary:
    """Fit every window in turn.

    Args:
        windows: The windows to fit, in time order.
        motor: The machine, minus the parameter being recovered.
        config: Network, loss and optimiser settings.
        v_max: Inverter voltage limit, used as the normalising voltage scale.
        truth: The true flux linkage, used only for scoring afterwards.
        on_result: Optional callable invoked with each `WindowResult` as it
            lands, so a caller can stream progress.

    Returns:
        The run summary.
    """
    started = time.perf_counter()
    results: list[WindowResult] = []
    for window in windows:
        scaling = (
            Scaling.for_window(window, v_max)
            if config.normalize
            else Scaling.identity()
        )
        result = train_window(window, motor, config, scaling)
        results.append(result)
        if callable(on_result):
            on_result(result)
    return RunSummary(results, truth, time.perf_counter() - started)
