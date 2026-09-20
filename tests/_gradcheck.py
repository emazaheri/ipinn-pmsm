"""Standalone float64 gradient check. Run by tests/test_gradients.py.

The physics term differentiates through a derivative-with-respect-to-an-input,
which is the part a hand-written port gets wrong, and it fails *plausibly*:
training still descends and the estimate still moves toward the truth on the
supervised term alone, so nothing looks broken. Only a numerical check finds it.

Two details make this check actually work.

**It runs in float64**, which is why it lives in a subprocess. JAX only honours
``jax_enable_x64`` when it is set before the library is first used, and
toggling it mid-session silently leaves the arithmetic in float32, where the
difference of two nearly-equal values is quantised and the result is noise.

**It differentiates along random directions rather than along axes.** In the
reference preset the objective is of order 1e5 while a single weight's
gradient can be of order 1e-5. Subtracting two values of 1e5 in float64 leaves
about 1e-11 of resolution, so a per-component difference is buried in
roundoff before it is ever compared. A random unit direction aggregates all
1284 components, giving a directional derivative of order one, and it still
catches a single wrong term because that term contributes to every direction.

Run at two settings on purpose. With ``beta = 0`` the physics term is off and
this is plain backprop, which must pass. With ``beta = 1`` the physics term
dominates, so any error in differentiating through the time derivative shows
up immediately.
"""

from __future__ import annotations

import sys

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from ipinn_pmsm import InverterParams, MotorParams, PinnConfig, SimParams, WindowConfig
from ipinn_pmsm.normalize import Scaling
from ipinn_pmsm.pinn.losses import Fixed, loss_fn
from ipinn_pmsm.pinn.mlp import init_params
from ipinn_pmsm.scenarios import reference_scenario
from ipinn_pmsm.simulate import simulate
from ipinn_pmsm.windows import split

TOLERANCE = 1e-6
DIRECTIONS = 8


def check(beta: float, step: float = 1e-6) -> int:
    """Compare analytic and numerical directional derivatives.

    Args:
        beta: Weight on the physics term.
        step: Central-difference step along a unit direction.

    Returns:
        Zero on success, one on failure.
    """
    if not bool(jax.config.jax_enable_x64):
        print("x64 is not enabled; this check would be meaningless", file=sys.stderr)
        return 1

    motor = MotorParams()
    config = PinnConfig(hidden=(8, 8), beta=beta)
    fixed = Fixed(rs=motor.rs, ld=motor.ld, lq=motor.lq)

    trace = simulate(
        motor, InverterParams(), SimParams(legacy_time_axis=True), *reference_scenario()
    )
    window = split(trace, WindowConfig())[40]
    scaling = Scaling.identity()
    inputs = jnp.asarray(scaling.inputs(window))
    targets = jnp.asarray(scaling.targets(window))

    params = init_params(config)
    params = [{k: jnp.asarray(v, jnp.float64) for k, v in p.items()} for p in params]

    def objective(flat_params: jax.Array) -> jax.Array:
        return loss_fn(
            unflatten(flat_params), inputs, targets, fixed, config.alpha, config.beta
        ).total

    flat, unflatten = jax.flatten_util.ravel_pytree(params)
    gradient = jax.grad(objective)(flat)

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(DIRECTIONS):
        direction = rng.normal(size=int(flat.size))
        direction /= np.linalg.norm(direction)
        v = jnp.asarray(direction)

        analytic = float(jnp.dot(gradient, v))
        plus = float(objective(flat + step * v))
        minus = float(objective(flat - step * v))
        numeric = (plus - minus) / (2.0 * step)
        worst = max(worst, abs(numeric - analytic) / max(abs(analytic), 1e-12))

    print(
        f"beta={beta}: {DIRECTIONS} directions, "
        f"worst relative error {worst:.2e} (tolerance {TOLERANCE:.0e})"
    )
    return 0 if worst < TOLERANCE else 1


def check_lambda_gradient() -> int:
    """Verify the flux linkage's gradient against its closed form.

    The parameter enters only the q residual, so with
    ``physics = mean(r_d^2) + mean(r_q^2)`` the derivative is exactly
    ``beta * 2 * mean(r_q * omega_e)``. It is the one gradient a port can
    compute analytically rather than by backpropagation, so it is worth
    pinning separately.

    Returns:
        Zero on success, one on failure.
    """
    from ipinn_pmsm.pinn.losses import ode_residuals
    from ipinn_pmsm.pinn.mlp import forward

    motor = MotorParams()
    config = PinnConfig(hidden=(8, 8), beta=1.0)
    fixed = Fixed(rs=motor.rs, ld=motor.ld, lq=motor.lq)

    trace = simulate(
        motor, InverterParams(), SimParams(legacy_time_axis=True), *reference_scenario()
    )
    window = split(trace, WindowConfig())[40]
    inputs = jnp.asarray(Scaling.identity().inputs(window))
    targets = jnp.asarray(Scaling.identity().targets(window))

    params = init_params(config)
    params = [{k: jnp.asarray(v, jnp.float64) for k, v in p.items()} for p in params]
    layers = params[:-1]

    def of_lambda(lam: jax.Array) -> jax.Array:
        return loss_fn(
            [*layers, {"lambda_m": lam}],
            inputs,
            targets,
            fixed,
            config.alpha,
            config.beta,
        ).total

    lam = jnp.array([0.1], dtype=jnp.float64)
    analytic = float(jax.grad(lambda x: of_lambda(x).sum())(lam)[0])

    _r_d, r_q = ode_residuals(layers, inputs, lam, fixed)
    omega_e = forward(layers, inputs)[:, 2]
    closed_form = float(config.beta * 2.0 * jnp.mean(r_q * omega_e))

    error = abs(analytic - closed_form) / max(abs(closed_form), 1e-12)
    print(f"lambda_m gradient: autodiff vs closed form, relative error {error:.2e}")
    return 0 if error < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(max(check(0.0), check(1.0), check_lambda_gradient()))
