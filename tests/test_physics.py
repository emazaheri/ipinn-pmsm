"""The simulator has to be right before anything fitted to it means much."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ipinn_pmsm import InverterParams, MotorParams, SimParams, simulate
from ipinn_pmsm.inverter import applied_voltage, limit_magnitude
from ipinn_pmsm.machine import state_derivative
from ipinn_pmsm.params import WindowConfig
from ipinn_pmsm.scenarios import REFERENCE_SEGMENTS, reference_scenario, rpm_to_rad_s
from ipinn_pmsm.transforms import clarke, inverse_clarke, inverse_park, park
from ipinn_pmsm.windows import split


def test_park_round_trips() -> None:
    for theta in (0.0, 0.7, 2.5, -1.3):
        d, q = park(*inverse_park(3.0, -4.0, theta), theta)
        assert d == pytest.approx(3.0, abs=1e-12)
        assert q == pytest.approx(-4.0, abs=1e-12)


def test_clarke_round_trips_for_a_balanced_set() -> None:
    alpha, beta = 2.0, -0.5
    a, b, c = inverse_clarke(alpha, beta)
    assert a + b + c == pytest.approx(0.0, abs=1e-12)
    assert clarke(a, b, c) == pytest.approx((alpha, beta), abs=1e-12)


def test_clarke_is_amplitude_invariant_not_power_invariant() -> None:
    # A balanced unit-amplitude set must map to a unit-magnitude vector. The
    # power-invariant convention would give sqrt(3/2) instead, which would
    # rescale every estimated flux linkage by the same factor.
    angle = 0.4
    a = math.cos(angle)
    b = math.cos(angle - 2 * math.pi / 3)
    c = math.cos(angle + 2 * math.pi / 3)
    alpha, beta = clarke(a, b, c)
    assert math.hypot(alpha, beta) == pytest.approx(1.0, abs=1e-12)


def test_voltage_limit_preserves_direction() -> None:
    v_d, v_q = limit_magnitude(300.0, 400.0, 100.0)
    assert math.hypot(v_d, v_q) == pytest.approx(100.0)
    # Clipping the axes independently would rotate the vector; this must not.
    assert v_q / v_d == pytest.approx(400.0 / 300.0)


def test_voltage_below_the_limit_is_untouched() -> None:
    assert limit_magnitude(3.0, 4.0, 100.0) == (3.0, 4.0)


def test_an_ideal_inverter_applies_what_was_commanded() -> None:
    ideal = InverterParams(v_dc=400.0)
    v_d, v_q, d_d, d_q = applied_voltage(10.0, 20.0, 1.0, 2.0, 0.3, ideal)
    assert (v_d, v_q, d_d, d_q) == (10.0, 20.0, 0.0, 0.0)


def test_derivative_vanishes_at_the_analytic_operating_point() -> None:
    # At steady state the dq equations reduce to an algebraic pair. Solving
    # them and feeding the result back must give zero current derivative.
    motor = MotorParams()
    i_d, i_q, omega_m = -2.0, 12.0, rpm_to_rad_s(1000.0)
    omega_e = motor.pole_pairs * omega_m
    ld, lq = motor.saturated(i_d, i_q)
    v_d = motor.rs * i_d - omega_e * lq * i_q
    v_q = motor.rs * i_q + omega_e * (ld * i_d + motor.lambda_m)

    did, diq, dtheta = state_derivative((i_d, i_q, 0.0), v_d, v_q, omega_m, motor)
    assert did == pytest.approx(0.0, abs=1e-9)
    assert diq == pytest.approx(0.0, abs=1e-9)
    assert dtheta == pytest.approx(omega_m)


def test_saturation_lowers_inductance_and_is_off_by_default() -> None:
    assert MotorParams().saturated(10.0, 10.0) == (MotorParams().ld, MotorParams().lq)
    saturating = MotorParams(beta_d=0.000575, beta_q=0.000891)
    ld, lq = saturating.saturated(0.0, 12.0)
    assert ld < saturating.ld
    assert lq < saturating.lq


def test_the_drive_reaches_every_commanded_operating_point() -> None:
    motor, inverter, sim = MotorParams(), InverterParams(), SimParams()
    trace = simulate(motor, inverter, sim, *reference_scenario())

    for segment in REFERENCE_SEGMENTS:
        # Sample just before the segment ends, once the transient has settled.
        index = int((segment.until - 0.01) / trace.dt)
        assert trace.i_d[index] == pytest.approx(segment.i_d, abs=1e-3)
        assert trace.i_q[index] == pytest.approx(segment.i_q, abs=1e-3)
        expected = motor.pole_pairs * rpm_to_rad_s(segment.rpm)
        assert trace.omega_e[index] == pytest.approx(expected, rel=1e-9)


def test_steady_state_voltages_match_the_algebraic_solution() -> None:
    motor, inverter, sim = MotorParams(), InverterParams(), SimParams()
    trace = simulate(motor, inverter, sim, *reference_scenario())

    index = int(0.24 / trace.dt)
    omega_e = trace.omega_e[index]
    i_d, i_q = trace.i_d[index], trace.i_q[index]
    assert trace.v_d[index] == pytest.approx(
        motor.rs * i_d - omega_e * motor.lq * i_q, rel=1e-4
    )
    assert trace.v_q[index] == pytest.approx(
        motor.rs * i_q + omega_e * (motor.ld * i_d + motor.lambda_m), rel=1e-4
    )


def test_an_ideal_bridge_differs_from_the_command_only_where_it_clips() -> None:
    # The DC-link magnitude limit is not an imperfection, it is what the
    # hardware can reach, so it applies even with zero conduction drop and
    # zero dead time. Everywhere below the limit the two must agree exactly.
    inverter = InverterParams()
    trace = simulate(MotorParams(), inverter, SimParams(), *reference_scenario())

    commanded = np.hypot(trace.v_d, trace.v_q)
    clipped = commanded > inverter.v_max_dq + 1e-9
    assert np.allclose(trace.v_d[~clipped], trace.v_d_applied[~clipped])
    assert np.allclose(trace.v_q[~clipped], trace.v_q_applied[~clipped])

    # The only places the command exceeds the bus are the reference steps,
    # where the deliberately aggressive PI gains ask for more than 400 V can
    # deliver for a single sample. Nowhere else, and never for long.
    boundaries = {0.5, 1.0, 1.5}
    clipped_times = {round(float(t), 3) for t in trace.t[clipped]}
    assert clipped_times <= boundaries
    assert clipped.sum() <= len(boundaries)

    applied = np.hypot(trace.v_d_applied, trace.v_q_applied)
    assert np.all(applied <= inverter.v_max_dq + 1e-9)


def test_zero_speed_makes_the_flux_linkage_unobservable() -> None:
    # The parameter enters only through omega_e, so at standstill it cannot
    # affect any measurement. Two very different magnets must produce
    # identical currents.
    inverter, sim = InverterParams(), SimParams(t_sim=0.2)
    weak = simulate(
        MotorParams(lambda_m=0.15), inverter, sim, *reference_scenario(speed_scale=0.0)
    )
    strong = simulate(
        MotorParams(lambda_m=0.25), inverter, sim, *reference_scenario(speed_scale=0.0)
    )
    assert np.allclose(weak.i_d, strong.i_d, atol=1e-12)
    assert np.allclose(weak.i_q, strong.i_q, atol=1e-12)


def test_noise_leaves_the_clean_trace_alone() -> None:
    trace = simulate(
        MotorParams(), InverterParams(), SimParams(t_sim=0.1), *reference_scenario()
    )
    original = trace.i_d.copy()
    noisy = trace.with_noise(10.0, 5.0, seed=1)
    assert np.array_equal(trace.i_d, original)
    assert not np.array_equal(noisy.i_d, original)
    # Speed is measured by the dynamometer encoder and stays clean.
    assert np.array_equal(noisy.omega_e, trace.omega_e)


def test_window_span_counts_intervals_not_samples() -> None:
    # 14 samples at 1 ms span 13 ms. Using the nominal 15 ms instead puts a
    # 15% bias on every time derivative in the residual.
    trace = simulate(
        MotorParams(),
        InverterParams(),
        SimParams(legacy_time_axis=True),
        *reference_scenario(),
    )
    window = split(trace, WindowConfig())[0]
    assert len(window) == 14
    assert window.span == pytest.approx(13 * trace.dt)
    assert window.span < 0.015


def test_the_legacy_time_axis_is_what_produces_142_windows() -> None:
    motor, inverter = MotorParams(), InverterParams()
    scenario = reference_scenario()

    legacy = split(
        simulate(motor, inverter, SimParams(legacy_time_axis=True), *scenario),
        WindowConfig(),
    )
    consistent = split(
        simulate(motor, inverter, SimParams(), *scenario), WindowConfig()
    )
    assert (len(legacy), len(legacy[0])) == (142, 14)
    assert (len(consistent), len(consistent[0])) == (133, 15)
