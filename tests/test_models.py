from dataclasses import replace

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from noten import evaluate as ev
from noten.mapping import full_cutpoints, grade_probabilities
from noten.models import QUICK_SAMPLER, ModelSpec, _probit_interval_logp, build_model, fit
from noten.simulate import SimConfig, simulate

TINY = replace(SimConfig(), n_courses=3, n_per_course=6, with_teacher_text=True)


def test_custom_ordinal_matches_pymc_orderedprobit():
    cut = np.array([1.5, 2.3, 3.6, 4.4, 5.5])
    eta = np.array([0.0, 2.0, 3.3, 5.0, 8.0, 3.0, -4.0])
    y = np.array([0, 1, 2, 5, 5, 3, 0])
    padded = np.concatenate([[-1e3], cut, [1e3]])
    ours = _probit_interval_logp(
        pt.as_tensor(eta), 0.7, pt.as_tensor(padded[y]), pt.as_tensor(padded[y + 1])
    ).eval()
    ref = pm.logp(pm.OrderedProbit.dist(eta=eta, cutpoints=cut, sigma=0.7), y).eval()
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "spec",
    [
        ModelSpec(),
        ModelSpec(rho_free=False),
        ModelSpec(use_vornoten=False),
        ModelSpec(teacher_effect=False),
        ModelSpec(use_teacher_text=True),
    ],
)
def test_model_variants_build_and_logp_finite(spec):
    data = simulate(TINY, seed=1)
    m = build_model(data, spec)
    lp = m.compile_logp()(m.initial_point())
    assert np.isfinite(lp)
    names = {v.name for v in m.free_RVs}
    assert ("chol_u" in names) == spec.rho_free
    assert ("sigma_drift" in names) == spec.use_vornoten
    assert ("tau_b" in names) == spec.teacher_effect


def test_fit_smoke():
    """Schneller Fit-Smoketest: läuft durch, liefert endliche Größen und gültige Notenverteilung."""
    data = simulate(TINY, seed=3)
    sampler = replace(QUICK_SAMPLER, draws=150, tune=150, chains=2)
    _, idata = fit(data, ModelSpec(), sampler, seed=1, log_likelihood=True)
    post = idata.posterior
    for v in ("mu_s", "mu_m", "theta", "rho", "b", "sigma_s", "sigma_m", "cut_inner"):
        assert v in post.data_vars
        assert np.isfinite(np.asarray(post[v].values)).all()
    # Zero-Sum der Kurseffekte
    np.testing.assert_allclose(ev.draws(idata, "b").sum(axis=1), 0.0, atol=1e-8)
    cut = full_cutpoints(ev.draws(idata, "cut_inner"))
    assert (np.diff(cut, axis=1) > 0).all()
    probs = grade_probabilities(ev.draws(idata, "theta"), cut)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    d = ev.diagnostics(idata)
    assert set(d) >= {"rhat_max", "ess_bulk_min", "ess_tail_min", "divergences", "ok"}
    rows = ev.recovery(idata, data)
    assert {r["param"] for r in rows} >= {"mu_s", "rho", "b", "sigma_s"}
    ll = ev.combined_log_likelihood(idata)
    assert ll.sizes["obs"] == data.S.size + data.M.size
