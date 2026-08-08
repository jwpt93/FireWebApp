"""Unit tests for Marsden-Smedley 1995 buttongrass empirical ROS.

Rule #18 (unit tests required for every new module): the new
`marsden_smedley_ros_m_per_s` and `marsden_smedley_p_sustain` functions
in `model_outdoor/empirical_ros.py` require these tests to ship with
the same commit that introduces them.

Coverage:
  1. Bit-exact determinism (same inputs → same outputs to last digit)
  2. Known regression point (Phase 22 CAL: U=2 m/s, M=30%, age=10)
  3. Monotonicity in each independent variable
  4. Boundary cases (zero-U, zero-age → zero ROS)
  5. Age asymptote (age=100 ≈ age=∞)
  6. Extinction logistic reproduces Tasmania 2009 Appendix 2 boundary values
  7. Dispatch via evaluate_empirical_ros routes correctly
"""
import math
import pytest
from model_outdoor.empirical_ros import (
    marsden_smedley_ros_m_per_s,
    marsden_smedley_p_sustain,
    evaluate_empirical_ros,
    MS_1995_CONST, MS_1995_U_EXP, MS_1995_B_MF, MS_1995_AGE_LAMBDA,
)


# ── 1. Determinism ───────────────────────────────────────────────────
def test_marsden_smedley_ros_determinism():
    a = marsden_smedley_ros_m_per_s(2.0, 0.30, 10.0)
    b = marsden_smedley_ros_m_per_s(2.0, 0.30, 10.0)
    assert repr(a) == repr(b), (
        f"Non-deterministic (Rule #17): {a!r} vs {b!r}"
    )


def test_marsden_smedley_p_sustain_determinism():
    a = marsden_smedley_p_sustain(1.5, 0.50, productivity=1)
    b = marsden_smedley_p_sustain(1.5, 0.50, productivity=1)
    assert repr(a) == repr(b)


# ── 2. Known regression point (Phase 22 CAL) ─────────────────────────
def test_marsden_smedley_cal_point():
    """CAL: U_1.7=2 m/s, M=30%, age=10 → 0.04988 m/s = 2.99 m/min.

    Value computed at Phase 22 plan time from the closed-form regression;
    freezes the constant against accidental refactor.
    """
    ros = marsden_smedley_ros_m_per_s(2.0, 0.30, 10.0)
    assert ros == pytest.approx(0.04988, abs=1e-5), (
        f"CAL point drifted from Phase 22 plan value 0.04988 m/s: got {ros}"
    )


def test_marsden_smedley_val_points():
    """VAL-A/B/C/D points frozen at Phase 22 plan values."""
    expected = [
        # (U_m_s, M_frac, age_yr, expected_ros_m_s)
        (4.0, 0.15, 10.0, 0.17832),  # VAL-A
        (6.0, 0.55, 10.0, 0.11484),  # VAL-B
        (1.5, 0.50, 10.0, 0.02104),  # VAL-C (1.26 m/min)
        (0.5, 0.70, 10.0, 0.00306),  # VAL-D
    ]
    for U, M, age, exp in expected:
        ros = marsden_smedley_ros_m_per_s(U, M, age)
        assert ros == pytest.approx(exp, abs=1e-4), (
            f"U={U},M={M},age={age}: expected {exp}, got {ros}"
        )


# ── 3. Monotonicity ──────────────────────────────────────────────────
def test_ros_increases_with_wind():
    r1 = marsden_smedley_ros_m_per_s(1.0, 0.30, 10.0)
    r2 = marsden_smedley_ros_m_per_s(4.0, 0.30, 10.0)
    assert r2 > r1


def test_ros_decreases_with_moisture():
    r_dry = marsden_smedley_ros_m_per_s(2.0, 0.10, 10.0)
    r_wet = marsden_smedley_ros_m_per_s(2.0, 0.60, 10.0)
    assert r_wet < r_dry


def test_ros_increases_with_age_below_asymptote():
    r_young = marsden_smedley_ros_m_per_s(2.0, 0.30, 3.0)
    r_old   = marsden_smedley_ros_m_per_s(2.0, 0.30, 20.0)
    assert r_old > r_young


# ── 4. Boundary cases ────────────────────────────────────────────────
def test_zero_wind_gives_zero():
    assert marsden_smedley_ros_m_per_s(0.0, 0.30, 10.0) == 0.0


def test_negative_wind_gives_zero():
    assert marsden_smedley_ros_m_per_s(-1.0, 0.30, 10.0) == 0.0


def test_zero_age_gives_zero():
    assert marsden_smedley_ros_m_per_s(2.0, 0.30, 0.0) == 0.0


def test_never_returns_negative():
    for U, M, age in [(0.1, 0.99, 0.1), (10.0, 0.99, 40.0), (0.5, 0.01, 100.0)]:
        assert marsden_smedley_ros_m_per_s(U, M, age) >= 0.0


# ── 5. Age asymptote ─────────────────────────────────────────────────
def test_age_asymptote_within_1e_minus_4():
    """(1 - exp(-0.116·age)) → 1.0 as age → ∞.  At age=100 the
    residual is exp(-11.6) ≈ 9e-6."""
    r_100 = marsden_smedley_ros_m_per_s(2.0, 0.30, 100.0)
    r_inf = MS_1995_CONST * (7.2 ** MS_1995_U_EXP) * \
            math.exp(-MS_1995_B_MF * 30.0)  # asymptote factor = 1
    assert abs(r_100 - r_inf) < 1e-4


# ── 6. Extinction logistic against Tasmania 2009 Appendix 2 ─────────
def test_p_sustain_boundary_M30():
    """M=30%, U_1.7=1.76 km/h (0.489 m/s), prod=1 → P ≈ 0.5."""
    p = marsden_smedley_p_sustain(0.489, 0.30, productivity=1)
    assert p == pytest.approx(0.500, abs=0.005), (
        f"P_sustain at M=30, U=0.489 m/s, prod=1 = {p} (expected ≈0.5)"
    )


def test_p_sustain_higher_wind_more_likely():
    p_lo = marsden_smedley_p_sustain(0.5, 0.50, 1)
    p_hi = marsden_smedley_p_sustain(3.0, 0.50, 1)
    assert p_hi > p_lo


def test_p_sustain_wetter_less_likely():
    p_dry = marsden_smedley_p_sustain(2.0, 0.20, 1)
    p_wet = marsden_smedley_p_sustain(2.0, 0.80, 1)
    assert p_wet < p_dry


def test_p_sustain_productivity_effect():
    """Higher productivity → more likely to sustain (positive coefficient
    in the logistic argument)."""
    p_low  = marsden_smedley_p_sustain(2.0, 0.50, productivity=1)
    p_med  = marsden_smedley_p_sustain(2.0, 0.50, productivity=2)
    assert p_med > p_low


# ── 7. Dispatch routing ──────────────────────────────────────────────
def test_dispatch_routes_marsden_smedley():
    """`evaluate_empirical_ros` with model='marsden_smedley' calls the
    right function with age_yr kwarg."""
    direct = marsden_smedley_ros_m_per_s(2.0, 0.30, 10.0)
    via_dispatch = evaluate_empirical_ros(
        "marsden_smedley", 2.0, 0.30, a_ch=999.0,  # a_ch must be ignored
        age_yr=10.0,
    )
    assert direct == via_dispatch


def test_dispatch_default_age_when_missing():
    """When age_yr kwarg is missing, defaults to 10 yr."""
    default_age = evaluate_empirical_ros(
        "marsden_smedley", 2.0, 0.30, a_ch=999.0,
    )
    explicit = marsden_smedley_ros_m_per_s(2.0, 0.30, 10.0)
    assert default_age == explicit


def test_dispatch_ignores_age_yr_for_cheney():
    """`age_yr` kwarg must not affect the cheney_eq6 branch."""
    r_no_age = evaluate_empirical_ros(
        "cheney_eq6", 4.0, 0.04, a_ch=0.406,
    )
    r_with_age = evaluate_empirical_ros(
        "cheney_eq6", 4.0, 0.04, a_ch=0.406, age_yr=99.0,
    )
    assert r_no_age == r_with_age


# ── 8. Class-based FuelModel interface (Phase 22.5 refactor) ─────────
def test_registry_lists_expected_models():
    from model_outdoor.empirical_ros import list_models
    names = list_models()
    for expected in ("cheney_eq6", "marsden_smedley", "rothermel"):
        assert expected in names, f"missing model {expected!r}: {names}"


def test_get_model_returns_correct_instance():
    from model_outdoor.empirical_ros import get_model, MarsdenSmedley
    m = get_model("marsden_smedley")
    assert isinstance(m, MarsdenSmedley)
    assert m.name == "marsden_smedley"
    assert "age_yr" in m.schema


def test_get_model_unknown_raises():
    from model_outdoor.empirical_ros import get_model
    with pytest.raises(ValueError, match="Unknown fuel model"):
        get_model("no_such_model")


def test_class_ros_matches_free_function():
    """CheneyEq6.ros and MarsdenSmedley.ros bit-exact against the
    corresponding free functions (this is the whole point of the
    backward-compat guarantee)."""
    from model_outdoor.empirical_ros import get_model
    r_cls = get_model("cheney_eq6").ros(4.0, moisture_frac=0.04, a_ch=0.406)
    r_free = evaluate_empirical_ros("cheney_eq6", 4.0, 0.04, 0.406)
    assert repr(r_cls) == repr(r_free)

    r_cls = get_model("marsden_smedley").ros(2.0, moisture_frac=0.30, age_yr=10.0)
    r_free = marsden_smedley_ros_m_per_s(2.0, 0.30, 10.0)
    assert repr(r_cls) == repr(r_free)


def test_class_p_sustain_default_is_one():
    """FuelModel base class default p_sustain returns 1.0."""
    from model_outdoor.empirical_ros import get_model
    assert get_model("cheney_eq6").p_sustain(4.0) == 1.0


def test_class_p_sustain_marsden_smedley_uses_logistic():
    from model_outdoor.empirical_ros import get_model
    p = get_model("marsden_smedley").p_sustain(
        0.489, moisture_frac=0.30, productivity=1,
    )
    assert p == pytest.approx(0.500, abs=0.005)


def test_class_ros_samples_default_is_delta():
    """Default ros_samples returns n identical copies of ros()."""
    from model_outdoor.empirical_ros import get_model
    m = get_model("marsden_smedley")
    samples = m.ros_samples(2.0, n=5, moisture_frac=0.30, age_yr=10.0)
    assert len(samples) == 5
    assert all(s == samples[0] for s in samples)
    # BHM-ready hook: a fitted subclass would return distinct samples here
    assert samples[0] == m.ros(2.0, moisture_frac=0.30, age_yr=10.0)
