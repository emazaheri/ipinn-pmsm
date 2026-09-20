"""The published numbers, reproduced.

This is the test that says the port is the research rather than something
inspired by it. Cases A, B and C land on their published figures to two
decimal places; case D is within a twentieth of a point, the residue of a
different noise realisation and of fixed-step RK4 in place of adaptive LSODA.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipinn_pmsm.cases import available_cases, load_case
from ipinn_pmsm.runner import run_case

TOLERANCE_POINTS = 0.1


def test_every_case_config_loads() -> None:
    assert available_cases() == ["a", "b", "c", "d"]
    for name in available_cases():
        case = load_case(name)
        assert case.published_error_pct is not None
        assert case.title


def test_the_estimator_inherits_the_plant_unless_told_otherwise() -> None:
    # Case B lies about nothing, so known must equal plant.
    b = load_case("b")
    assert b.known.rs == b.plant.rs
    assert b.known.ld == b.plant.ld

    # Case C is the one that lies, and only about stator resistance.
    c = load_case("c")
    assert c.plant.rs == pytest.approx(0.52)
    assert c.known.rs == pytest.approx(0.46)
    assert c.known.ld == c.plant.ld
    assert c.known.lq == c.plant.lq


@pytest.mark.parametrize("name", ["a", "b", "c", "d"])
def test_case_reproduces_its_published_error(name: str) -> None:
    case = load_case(name)
    summary = run_case(case)
    assert case.published_error_pct is not None
    assert summary.mean_error_pct == pytest.approx(
        case.published_error_pct, abs=TOLERANCE_POINTS
    )


def test_case_c_is_hard_because_of_the_resistance_not_the_magnet() -> None:
    # Recovering a demagnetised magnet is no harder than a healthy one. What
    # makes case C cost 2.13% instead of about 1% is the stator resistance
    # the estimator has wrong. Removing that one lie should roughly halve the
    # error while leaving the magnet weak.
    case = load_case("c")
    honest = replace(case, known=case.plant)

    assert run_case(honest).mean_error_pct < run_case(case).mean_error_pct / 1.5


def test_a_stalled_rotor_makes_the_parameter_unrecoverable() -> None:
    # Sensitivity is the electrical speed, so at standstill there is nothing
    # to recover and the estimator should be badly wrong rather than quietly
    # plausible.
    case = load_case("a")
    assert run_case(case, speed_scale=0.0).mean_error_pct > 20.0


def test_slower_operation_is_measurably_worse_than_faster() -> None:
    # The record steps from 500 to 1000 rpm halfway through. Sensitivity
    # doubles with it, so the first half must scatter wider than the second.
    summary = run_case(load_case("d"))
    slow, fast = summary.split_by_speed()
    assert slow > fast
