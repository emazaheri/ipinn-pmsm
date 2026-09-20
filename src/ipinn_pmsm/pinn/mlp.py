"""The small multilayer perceptron the physics is imposed on.

Three inputs ``[t, vd, vq]``, three outputs ``[id, iq, we]``, swish throughout
the hidden layers and a linear head. At the reference width of two 32-unit
layers that is 1283 weights and biases, to which the trainable flux linkage
adds one more.

The network is deliberately tiny. It has to fit fourteen samples, not a
dataset, and a fresh copy is trained for every one of the 142 windows. That
size is the point: 1284 float64 values is 10 KB of state, and the whole
forward pass is three small matrix products, so this runs anywhere.
"""

from __future__ import annotations

from itertools import pairwise

import jax
import jax.numpy as jnp

from ..params import PinnConfig

__all__ = ["Params", "count_parameters", "forward", "init_params"]

Params = list[dict[str, jax.Array]]
"""Per-layer ``{"W", "B"}`` dicts, with the learnable physics dict appended."""


def init_params(config: PinnConfig, lambda_init: float | None = None) -> Params:
    """Initialise the network and the trainable physical parameter.

    Weights use Xavier-uniform bounds; biases are uniform on ``[0, 1)``, which
    is what the reference implementation did. Biases drawn from a unit interval
    rather than zeros is unusual and not defensible on any theoretical ground,
    but it is part of what produced the published numbers, so it stays.

    Args:
        config: Layer widths, seed and the flux linkage's starting value.
        lambda_init: Override the starting flux linkage, weber.

    Returns:
        The parameter list. The final entry is the physics dict, currently
        holding only ``lambda_m``.
    """
    layers = config.layers
    keys = jax.random.split(jax.random.PRNGKey(config.seed), len(layers) - 1)

    params: Params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:], strict=True):
        bound = 1.0 / jnp.sqrt(jnp.asarray(n_in, dtype=jnp.float32))
        params.append(
            {
                "W": jax.random.uniform(
                    key, (n_in, n_out), minval=-bound, maxval=bound
                ),
                "B": jax.random.uniform(key, (n_out,)),
            }
        )

    start = config.lambda_init if lambda_init is None else lambda_init
    params.append({"lambda_m": jnp.array([start])})
    return params


def forward(layers: Params, inputs: jax.Array) -> jax.Array:
    """Evaluate the network.

    Args:
        layers: Network layers only, without the trailing physics dict.
        inputs: Shape ``(N, 3)`` of ``[t, vd, vq]`` rows, or ``(3,)`` for one.

    Returns:
        Shape ``(N, 3)`` of ``[id, iq, we]`` rows.
    """
    x = inputs.reshape(1, -1) if inputs.ndim == 1 else inputs
    *hidden, last = layers
    for layer in hidden:
        x = jax.nn.swish(x @ layer["W"] + layer["B"])
    return x @ last["W"] + last["B"]


def count_parameters(config: PinnConfig) -> int:
    """Return the number of trainable scalars, physics included.

    Args:
        config: Layer widths.

    Returns:
        Total count, weights plus biases plus the one physical parameter.
    """
    layers = config.layers
    total = sum(n_in * n_out + n_out for n_in, n_out in pairwise(layers))
    return total + 1
