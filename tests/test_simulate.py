from dataclasses import replace

import numpy as np
import pytest

from noten.simulate import SCENARIOS, SimConfig, ordinal_from_latent, quick_config, simulate


def test_shapes_and_ranges():
    cfg = SimConfig()
    d = simulate(cfg, seed=1)
    n = cfg.n_students
    assert d.S.shape == (n, cfg.n_s)
    assert d.S_holdout.shape == (n, cfg.n_s_holdout)
    assert d.M.shape == (n, cfg.n_m)
    assert d.V_s.shape == d.V_m.shape == (n, cfg.n_years)
    assert d.course.shape == (n,)
    assert np.bincount(d.course).tolist() == [cfg.n_per_course] * cfg.n_courses
    for arr in (d.S, d.S_holdout, d.M, d.V_s, d.V_m):
        assert arr.dtype.kind == "i"
        assert arr.min() >= 1 and arr.max() <= 6
    assert d.D is None


def test_deterministic_seed():
    a = simulate(SimConfig(), seed=42)
    b = simulate(SimConfig(), seed=42)
    c = simulate(SimConfig(), seed=43)
    np.testing.assert_array_equal(a.S, b.S)
    np.testing.assert_array_equal(a.truth["mu_s"], b.truth["mu_s"])
    assert not np.array_equal(a.S, c.S)


def test_ordinal_from_latent_boundaries():
    lat = np.array([-5.0, 1.5, 1.51, 2.49, 3.0, 5.5, 5.51, 9.0])
    # Wert genau auf einer Schwelle gehört zur besseren (kleineren) Note: P(c_{k-1} < x <= c_k)
    assert ordinal_from_latent(lat).tolist() == [1, 1, 2, 2, 3, 5, 6, 6]


def test_truth_consistency():
    cfg = SimConfig()
    d = simulate(cfg, seed=3)
    t = d.truth
    np.testing.assert_allclose(t["theta"], cfg.w * t["mu_s"] + (1 - cfg.w) * t["mu_m"])
    assert t["b"].shape == (cfg.n_courses,)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_rho_is_simulated(rho):
    cfg = replace(SimConfig(), rho=rho, tau_b=0.0, n_courses=40, n_per_course=100)
    d = simulate(cfg, seed=5)
    u_s = d.truth["mu_s"] - cfg.alpha_s
    u_m = d.truth["mu_m"] - cfg.alpha_m
    r = np.corrcoef(u_s, u_m)[0, 1]
    assert abs(r - rho) < 0.05


def test_no_teacher_effect():
    d = simulate(SCENARIOS["c_ohne_lehrer"], seed=1)
    assert np.all(d.truth["b"] == 0)


def test_vornoten_drift_older_years_less_informative():
    cfg = replace(SimConfig(), sigma_drift=0.8, n_courses=40, n_per_course=100, n_years=3)
    d = simulate(cfg, seed=9)
    mu = d.truth["mu_s"] - d.truth["b"][d.course]
    corr = [np.corrcoef(mu, d.V_s[:, y])[0, 1] for y in range(3)]
    assert corr[0] > corr[1] > corr[2]


def test_teacher_text_channel():
    cfg = replace(SimConfig(), with_teacher_text=True)
    d = simulate(cfg, seed=2)
    assert d.D.shape == (cfg.n_students,)
    assert d.D.min() >= 1 and d.D.max() <= 6


def test_scenarios_cover_requirements():
    s = SCENARIOS
    assert s["a_rho0"].rho == 0.0 and s["b_rho05"].rho == 0.5
    assert s["c_ohne_lehrer"].tau_b == 0.0 and s["b_rho05"].tau_b > 0
    assert s["d_wenig"].n_s < s["b_rho05"].n_s < s["d_viel"].n_s
    assert s["e_drift"].sigma_drift > s["b_rho05"].sigma_drift
    for cfg in s.values():
        assert cfg.n_courses >= 8 and cfg.n_per_course >= 25
    q = quick_config(s["b_rho05"])
    assert q.n_students < s["b_rho05"].n_students
