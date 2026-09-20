"""The two-term objective: fit the measurements, obey the machine.

The physics term is where the inverse problem lives. It needs the derivative of
the network's current outputs with respect to the network's *time input*, which
is a derivative of an output with respect to an input, not with respect to a
parameter. Under JAX that is one `jax.grad` per output composed with `vmap`.
A port without autodiff can get the same thing by propagating a tangent through
the forward pass and then differentiating that augmented computation; either
way `ipinn_pmsm.fixtures` is what holds the two to the same answer.

Two things about the residual are easy to get wrong and are therefore spelled
out here.

**Units.** The network may be trained in scaled coordinates, but the machine
equations only hold in volts, amperes and radians per second. So every
quantity is decoded back to physical units before the residual is formed, and
the derivative picks up a factor of ``current_scale / time_span``. The
residual is then divided by a voltage scale, since it *is* a voltage, and
leaving it in volts would let the physics term dominate the objective by four
orders of magnitude.

**Saturation.** The residuals use **unsaturated** ``Ld`` and ``Lq`` while the
plant they are fitted to saturates. That mismatch is deliberate and is not a
bug. It is the realistic case, since a commissioning sheet gives you the
unsaturated values, and it is the reason the estimate carries a bias that
better conditioning cannot remove.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mlp import Params, forward

__all__ = ["Fixed", "LossParts", "loss_fn", "ode_residuals", "physics_loss"]


class LossParts(NamedTuple):
    """The objective broken into its named pieces.

    Attributes:
        total: The weighted sum actually minimised.
        supervised: Mean squared error against the measurements.
        physics: Mean squared ODE residual.
    """

    total: jax.Array
    supervised: jax.Array
    physics: jax.Array


class Fixed(NamedTuple):
    """What the estimator is told, and the units it is working in.

    Attributes:
        rs: Stator resistance, ohm.
        ld: d-axis inductance, henry. Unsaturated, deliberately.
        lq: q-axis inductance, henry. Unsaturated, deliberately.
        time_span: Seconds per unit of the network's time input. One when the
            network is fed absolute seconds. Must be the window's true span,
            ``(B - 1) * dt``, not its nominal length.
        current_scale: Amperes per unit of the network's current outputs.
        speed_scale: Radians per second per unit of the speed output.
        voltage_scale: Volts per unit of the network's voltage inputs, and the
            divisor that renders the residual dimensionless.
    """

    rs: float
    ld: float
    lq: float
    time_span: float = 1.0
    current_scale: float = 1.0
    speed_scale: float = 1.0
    voltage_scale: float = 1.0


def ode_residuals(
    layers: Params,
    inputs: jax.Array,
    lambda_m: jax.Array,
    fixed: Fixed,
) -> tuple[jax.Array, jax.Array]:
    """Return the d- and q-axis residuals of the machine equations.

    In physical units, before the final division by the voltage scale:

        r_d = Ld * did/dt - vd + Rs * id - we * Lq * iq
        r_q = Lq * diq/dt - vq + Rs * iq + we * (Ld * id + lambda_m)

    Args:
        layers: Network layers, without the physics dict.
        inputs: Shape ``(N, 3)`` of ``[t, vd, vq]`` rows, in network units.
        lambda_m: The trainable flux linkage, weber, shape ``(1,)``.
        fixed: Machine parameters and unit scales.

    Returns:
        The ``(r_d, r_q)`` pair, each shape ``(N,)``, dimensionless.
    """

    def output(row: jax.Array, index: int) -> jax.Array:
        return forward(layers, row.reshape(1, -1))[0, index]

    # Differentiate each current output with respect to the whole input row,
    # then keep component 0, which is the time column.
    d_id = jax.vmap(lambda r: jax.grad(output, argnums=0)(r, 0)[0])(inputs)
    d_iq = jax.vmap(lambda r: jax.grad(output, argnums=0)(r, 1)[0])(inputs)

    # Decode to physical units. The derivative carries both scales: amperes on
    # top from the output, seconds underneath from the input.
    to_amps_per_second = fixed.current_scale / fixed.time_span
    d_id = d_id * to_amps_per_second
    d_iq = d_iq * to_amps_per_second

    predicted = forward(layers, inputs)
    i_d = predicted[..., 0] * fixed.current_scale
    i_q = predicted[..., 1] * fixed.current_scale
    omega_e = predicted[..., 2] * fixed.speed_scale
    v_d = inputs[..., 1] * fixed.voltage_scale
    v_q = inputs[..., 2] * fixed.voltage_scale

    r_d = fixed.ld * d_id - v_d + fixed.rs * i_d - omega_e * fixed.lq * i_q
    r_q = fixed.lq * d_iq - v_q + fixed.rs * i_q + omega_e * (fixed.ld * i_d + lambda_m)
    return r_d / fixed.voltage_scale, r_q / fixed.voltage_scale


def physics_loss(
    layers: Params,
    inputs: jax.Array,
    lambda_m: jax.Array,
    fixed: Fixed,
) -> jax.Array:
    """Mean squared ODE residual over both axes.

    Args:
        layers: Network layers.
        inputs: Shape ``(N, 3)``.
        lambda_m: The trainable flux linkage.
        fixed: Machine parameters and unit scales.

    Returns:
        A scalar. This is ``mean(r_d^2) + mean(r_q^2)``, a sum of two means
        rather than the mean of the sum, matching the reference
        implementation. The two differ by a factor of two, which is absorbed
        into ``beta``, so the convention has to be pinned rather than inferred.
    """
    r_d, r_q = ode_residuals(layers, inputs, lambda_m, fixed)
    return jnp.mean(r_d**2) + jnp.mean(r_q**2)


def supervised_loss(layers: Params, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean squared error of the network against the measurements.

    Compared in network units, not physical ones, so that the term stays order
    one whatever the scaling is.

    Args:
        layers: Network layers.
        inputs: Shape ``(N, 3)`` of ``[t, vd, vq]`` rows.
        targets: Shape ``(N, 3)`` of measured ``[id, iq, we]`` rows.

    Returns:
        A scalar, averaged over samples and all three outputs.
    """
    return jnp.mean((forward(layers, inputs) - targets) ** 2)


def loss_fn(
    params: Params,
    inputs: jax.Array,
    targets: jax.Array,
    fixed: Fixed,
    alpha: float,
    beta: float,
) -> LossParts:
    """Evaluate the full objective.

    Args:
        params: Network layers with the physics dict appended.
        inputs: Shape ``(N, 3)`` of ``[t, vd, vq]`` rows.
        targets: Shape ``(N, 3)`` of measured ``[id, iq, we]`` rows.
        fixed: Machine parameters and unit scales.
        alpha: Weight on the supervised term.
        beta: Weight on the physics term.

    Returns:
        The three scalars as a `LossParts`.
    """
    *layers, physics = params
    supervised = supervised_loss(layers, inputs, targets)
    residual = physics_loss(layers, inputs, physics["lambda_m"], fixed)
    return LossParts(alpha * supervised + beta * residual, supervised, residual)
