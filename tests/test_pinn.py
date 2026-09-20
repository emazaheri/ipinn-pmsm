"""The estimator: its gradients, its optimiser, and what it recovers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from ipinn_pmsm import InverterParams, MotorParams, PinnConfig, SimParams, WindowConfig
from ipinn_pmsm.normalize import Scaling
from ipinn_pmsm.pinn.losses import Fixed, ode_residuals
from ipinn_pmsm.pinn.mlp import count_parameters, forward, init_params
from ipinn_pmsm.pinn.train import (
    WindowResult,
    reference_schedule_value,
    train_window,
)
from ipinn_pmsm.scenarios import reference_scenario
from ipinn_pmsm.simulate import simulate
from ipinn_pmsm.windows import Window, split

SETTLED = 120
"""A window well inside one operating point, away from every transition."""

TRANSITION = 71
"""The one window straddling the 500 to 1000 rpm speed step at t = 1.0 s."""


@pytest.fixture(scope="module")
def windows() -> list[Window]:
    trace = simulate(
        MotorParams(),
        InverterParams(),
        SimParams(legacy_time_axis=True),
        *reference_scenario(),
    )
    return split(trace, WindowConfig())


def test_the_network_has_the_size_the_write_up_claims() -> None:
    assert count_parameters(PinnConfig()) == 1284


def test_forward_accepts_one_row_or_a_batch() -> None:
    layers = init_params(PinnConfig())[:-1]
    batch = forward(layers, jnp.zeros((5, 3)))
    single = forward(layers, jnp.zeros(3))
    assert batch.shape == (5, 3)
    assert single.shape == (1, 3)


def test_the_residual_vanishes_for_an_exact_solution() -> None:
    # Build a network-free check: if the "network" reproduces a constant
    # operating point exactly, the residual must be the algebraic one, which
    # is zero at the true parameter and non-zero at any other.
    motor = MotorParams()
    fixed = Fixed(rs=motor.rs, ld=motor.ld, lq=motor.lq)
    i_d, i_q, omega_e = -2.0, 12.0, 418.879
    v_d = motor.rs * i_d - omega_e * motor.lq * i_q
    v_q = motor.rs * i_q + omega_e * (motor.ld * i_d + motor.lambda_m)

    # A zero-weight network with biases set to the operating point outputs a
    # constant, so every time derivative is exactly zero.
    layers = [
        {"W": jnp.zeros((3, 2)), "B": jnp.zeros(2)},
        {"W": jnp.zeros((2, 3)), "B": jnp.array([i_d, i_q, omega_e])},
    ]
    inputs = jnp.array([[0.0, v_d, v_q]])

    r_d, r_q = ode_residuals(layers, inputs, jnp.array([motor.lambda_m]), fixed)
    assert float(r_d[0]) == pytest.approx(0.0, abs=1e-3)
    assert float(r_q[0]) == pytest.approx(0.0, abs=1e-3)

    # A wrong flux linkage shows up in the q residual alone, scaled by speed.
    wrong = motor.lambda_m + 0.01
    r_d2, r_q2 = ode_residuals(layers, inputs, jnp.array([wrong]), fixed)
    assert float(r_d2[0]) == pytest.approx(float(r_d[0]), abs=1e-6)
    assert float(r_q2[0] - r_q[0]) == pytest.approx(omega_e * 0.01, rel=1e-4)


def test_sensitivity_to_the_parameter_is_exactly_the_electrical_speed() -> None:
    # d(r_q)/d(lambda_m) = omega_e. This is the identifiability statement the
    # whole demo is built around, so it gets asserted rather than asserted in
    # prose.
    motor = MotorParams()
    fixed = Fixed(rs=motor.rs, ld=motor.ld, lq=motor.lq)
    layers = init_params(PinnConfig())[:-1]
    inputs = jnp.array([[0.1, -22.0, 47.0], [0.2, -45.0, 90.0]])

    def q_sum(lam: jax.Array) -> jax.Array:
        return jnp.sum(ode_residuals(layers, inputs, lam, fixed)[1])

    gradient = float(jax.grad(q_sum)(jnp.array([0.2]))[0])
    speeds = forward(layers, inputs)[:, 2]
    assert gradient == pytest.approx(float(jnp.sum(speeds)), rel=1e-4)


def test_the_reference_schedule_is_a_constant_not_a_schedule() -> None:
    # Measured against optax: cosine_onecycle_schedule(1e-3, E, 1e-4, 0.1)
    # returns E/100 at every step. If a future optax changes that, the
    # reference preset silently stops reproducing the published numbers, so
    # this asserts the behaviour rather than trusting it.
    for epochs in (50, 150, 300):
        schedule = optax.cosine_onecycle_schedule(1e-3, epochs, 1e-4, 0.1)
        values = {round(float(schedule(k)), 9) for k in range(epochs)}
        assert len(values) == 1
        assert values.pop() == pytest.approx(reference_schedule_value(epochs))


def test_a_settled_window_recovers_the_hidden_parameter(windows: list[Window]) -> None:
    motor = MotorParams()
    result = train_window(windows[SETTLED], motor, PinnConfig())
    assert result.error_pct(motor.lambda_m) < 0.5
    # It has to arrive there, not start there.
    assert result.lambda_history[0] == pytest.approx(0.1, abs=0.02)
    assert result.lambda_history[-1] == pytest.approx(motor.lambda_m, abs=2e-3)


def test_the_estimate_converges_early_and_then_settles(windows: list[Window]) -> None:
    # rprop grows its step by 1.2 per same-sign epoch from 1e-3, so travelling
    # 0.106 Wb takes on the order of twenty epochs. The tail should be flat.
    result = train_window(windows[SETTLED], MotorParams(), PinnConfig())
    history = result.lambda_history
    assert abs(history[40] - history[-1]) < 5e-3
    early_travel = abs(history[40] - history[0])
    late_travel = abs(history[-1] - history[40])
    assert early_travel > 10 * late_travel


def test_the_estimator_is_never_told_the_answer(windows: list[Window]) -> None:
    # Changing only the plant's flux linkage in the parameters handed to the
    # estimator must not move the estimate at all, because the estimator reads
    # rs, ld and lq from it and nothing else.
    window = windows[SETTLED]
    a = train_window(window, MotorParams(lambda_m=0.206), PinnConfig())
    b = train_window(window, MotorParams(lambda_m=0.150), PinnConfig())
    assert a.lambda_m == pytest.approx(b.lambda_m, abs=1e-12)


def test_normalisation_keeps_both_loss_terms_near_unity(windows: list[Window]) -> None:
    window = windows[SETTLED]
    motor = MotorParams()
    scaling = Scaling.for_window(window, InverterParams().v_max_dq)
    assert scaling.enabled
    assert scaling.t_span == pytest.approx(window.span)

    reference = train_window(window, motor, PinnConfig())
    tuned = train_window(window, motor, PinnConfig(normalize=True), scaling)

    def ratio(result: WindowResult) -> float:
        return float(
            result.physics_history[-1] / max(result.supervised_history[-1], 1e-12)
        )

    # Scaling buys conditioning, so the two terms must end up closer together
    # than they were. It does not have to make them equal.
    assert ratio(tuned) < ratio(reference)
    # And it must not break the estimate it was supposed to help.
    assert tuned.lambda_m == pytest.approx(motor.lambda_m, abs=5e-3)


def test_a_window_straddling_the_speed_step_is_the_hard_one(
    windows: list[Window],
) -> None:
    # One window in the record contains the 500 to 1000 rpm transition, so
    # 209 rad/s of speed change lands inside 13 ms. Its neighbours are
    # settled. This single window is the largest contributor to the spread
    # the study reports, and it is a property of the excitation rather than
    # of the estimator.
    motor = MotorParams()
    transition = windows[TRANSITION]
    before, after = windows[TRANSITION - 1], windows[TRANSITION + 1]

    assert transition.omega_e.max() - transition.omega_e.min() > 200.0
    assert before.omega_e.max() == before.omega_e.min()
    assert after.omega_e.max() == after.omega_e.min()

    hard = train_window(transition, motor, PinnConfig())
    easy = train_window(before, motor, PinnConfig())
    assert easy.error_pct(motor.lambda_m) < 0.1
    assert hard.error_pct(motor.lambda_m) > 10.0
