"""Auswertung: Diagnostik, Parameter-Recovery, Kalibrierung, LOO."""

from __future__ import annotations

import arviz as az
import numpy as np
import xarray as xr

from .mapping import (
    central_interval,
    discrete_quantile,
    full_cutpoints,
    predictive_probabilities,
)
from .simulate import GradeData

LEVELS = (0.5, 0.8, 0.95)

# Größen, die geprüft und berichtet werden (sofern im Modell vorhanden)
REPORT_VARS = [
    "alpha",
    "tau",
    "rho",
    "tau_b",
    "b",
    "sigma_s",
    "sigma_m",
    "cut_inner",
    "sigma_drift",
    "sigma_v",
    "delta_v",
    "sigma_d",
    "mu_s",
    "mu_m",
    "theta",
]


def draws(idata, name: str) -> np.ndarray:
    """Posterior-Draws als (D, ...) Array (Chains zusammengelegt)."""
    da = idata.posterior[name]
    arr = np.asarray(da.values)
    return arr.reshape((-1,) + arr.shape[2:])


def present_vars(idata, names=REPORT_VARS) -> list[str]:
    post = idata.posterior
    return [v for v in names if v in post.data_vars]


def diagnostics(idata, var_names=None) -> dict:
    """R-hat, ESS (bulk/tail) und Divergenzen eines Fits."""
    var_names = var_names or present_vars(idata)
    post = idata.posterior.to_dataset()[var_names]
    rhat = az.rhat(post)
    ess_b = az.ess(post, method="bulk")
    ess_t = az.ess(post, method="tail")

    def _max(ds):
        return float(max(np.nanmax(np.asarray(ds[v].values)) for v in ds.data_vars))

    def _min(ds):
        return float(min(np.nanmin(np.asarray(ds[v].values)) for v in ds.data_vars))

    div = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    worst = max(rhat.data_vars, key=lambda v: float(np.nanmax(np.asarray(rhat[v].values))))
    out = {
        "rhat_worst_var": str(worst),
        "rhat_max": _max(rhat),
        "ess_bulk_min": _min(ess_b),
        "ess_tail_min": _min(ess_t),
        "divergences": div,
    }
    out["ok"] = (
        out["rhat_max"] < 1.01
        and out["ess_bulk_min"] > 400
        and out["ess_tail_min"] > 400
        and out["divergences"] == 0
    )
    return out


def interval_hits(samples: np.ndarray, truth: np.ndarray, levels=LEVELS) -> dict[float, np.ndarray]:
    """Für jede Ebene: Bool-Array, ob die Wahrheit im zentralen Intervall liegt."""
    truth = np.asarray(truth, dtype=float)
    out = {}
    for lev in levels:
        a = (1 - lev) / 2
        lo = np.quantile(samples, a, axis=0)
        hi = np.quantile(samples, 1 - a, axis=0)
        out[lev] = (truth >= lo) & (truth <= hi)
    return out


def recovery(idata, data: GradeData) -> list[dict]:
    """Recovery-Zeilen (eine je Parametergruppe) mit Trefferquoten in 50/80/95-%-Intervallen."""
    t = data.truth
    # Vergleichsgrößen passend zur Parametrisierung des Modells:
    # - Kurseffekte sind sum-to-zero -> Wahrheit b_t - mean(b), Kanalmittel alpha_c + mean(b),
    #   Vornoten-Offset delta_v - mean(b)
    # - sigma_v absorbiert den Rundungsfehler der Vornoten (Var 1/12) -> effektiver Wert
    bbar = float(np.mean(t["b"]))
    t = dict(t)
    t["b"] = t["b"] - bbar
    t["alpha_s"] = t["alpha_s"] + bbar
    t["alpha_m"] = t["alpha_m"] + bbar
    # Vornoten-Mittel = alpha_c + delta_v (ohne b) -> im Modell entspricht delta_v - mean(b)
    t["delta_v"] = t["delta_v"] - bbar
    t["sigma_v"] = float(np.sqrt(t["sigma_v"] ** 2 + 1 / 12))
    rows = []
    groups = {
        "alpha_s": ("alpha", lambda d: d[:, 0], t["alpha_s"]),
        "alpha_m": ("alpha", lambda d: d[:, 1], t["alpha_m"]),
        "tau_s": ("tau", lambda d: d[:, 0], t["tau_s"]),
        "tau_m": ("tau", lambda d: d[:, 1], t["tau_m"]),
        "rho": ("rho", None, t["rho"]),
        "tau_b": ("tau_b", None, t["tau_b"]),
        "b": ("b", None, t["b"]),
        "sigma_s": ("sigma_s", None, t["sigma_s"]),
        "sigma_m": ("sigma_m", None, t["sigma_m"]),
        "cut_inner": ("cut_inner", None, t["cut_inner"]),
        "sigma_drift": ("sigma_drift", None, t["sigma_drift"]),
        "sigma_v": ("sigma_v", None, t["sigma_v"]),
        "delta_v": ("delta_v", None, t["delta_v"]),
        "mu_s": ("mu_s", None, t["mu_s"]),
        "mu_m": ("mu_m", None, t["mu_m"]),
        "theta": ("theta", None, t["theta"]),
    }
    post = idata.posterior
    for label, (var, sel, truth) in groups.items():
        if var not in post.data_vars:
            continue
        d = draws(idata, var)
        if sel is not None:
            d = sel(d)
        truth = np.atleast_1d(truth)
        d = d.reshape(d.shape[0], -1)
        hits = interval_hits(d, truth)
        med = np.median(d, axis=0)
        q10, q90 = np.quantile(d, [0.1, 0.9], axis=0)
        rows.append(
            {
                "param": label,
                "n": truth.size,
                "cov50": hits[0.5].mean(),
                "cov80": hits[0.8].mean(),
                "cov95": hits[0.95].mean(),
                "bias": float(np.mean(med - truth)),
                "rmse": float(np.sqrt(np.mean((med - truth) ** 2))),
                "width80": float(np.mean(q90 - q10)),
                "truth_mean": float(np.mean(truth)),
                "post_median_mean": float(np.mean(med)),
            }
        )
    return rows


def grade_recovery(idata, data: GradeData) -> dict:
    """Trifft das 80-%-Intervall der Zeugnisnote die wahre Note round(theta_wahr)?"""
    from .mapping import grade_probabilities
    from .simulate import ordinal_from_latent

    theta = draws(idata, "theta")
    cut = full_cutpoints(draws(idata, "cut_inner"))
    probs = grade_probabilities(theta, cut)
    lo, hi = central_interval(probs, 0.8)
    true_grade = ordinal_from_latent(data.truth["theta"], data.config.cutpoints)
    med = discrete_quantile(probs, 0.5)
    return {
        "param": "note",
        "n": true_grade.size,
        "cov80": float(((true_grade >= lo) & (true_grade <= hi)).mean()),
        "width80": float(np.mean(hi - lo + 1)),
        "p_true": float(np.mean(probs[np.arange(true_grade.size), true_grade - 1])),
        "median_correct": float(np.mean(med == true_grade)),
        "single_grade_share": float(np.mean(lo == hi)),
    }


def predictive_for_holdout(idata) -> np.ndarray:
    """Prädiktive Wahrscheinlichkeiten (n, 6) für eine neue Klausur jedes Schülers."""
    mu = draws(idata, "mu_s")
    cut = full_cutpoints(draws(idata, "cut_inner"))
    sig = draws(idata, "sigma_s")
    return predictive_probabilities(mu, cut, sig)


def calibration(probs: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Kalibrierung der Vorhersage einer neuen Klausur (y: (n,) Noten 1..6).

    - hit80: Anteil Treffer im zentralen diskreten 80-%-Intervall [q10, q90]
      (wegen Diskretheit ist die tatsächliche Masse dieser Intervalle > 80 %)
    - mass80: mittlere vorhergesagte Wahrscheinlichkeit dieser Intervalle (Sollwert für hit80)
    - pit80: Anteil randomisierter PIT-Werte in (0.1, 0.9); kalibriert => 0.80
    - log_score: mittlere log-Wahrscheinlichkeit der wahren Note
    """
    y = np.asarray(y).astype(int)
    n = y.size
    lo, hi = central_interval(probs, 0.8)
    hit = (y >= lo) & (y <= hi)
    cdf = np.cumsum(probs, axis=1)
    mass = np.array([probs[i, lo[i] - 1 : hi[i]].sum() for i in range(n)])
    F_below = np.where(y > 1, cdf[np.arange(n), np.maximum(y - 2, 0)], 0.0)
    p_y = probs[np.arange(n), y - 1]
    pit = F_below + rng.uniform(size=n) * p_y
    return {
        "hit80": float(hit.mean()),
        "mass80": float(mass.mean()),
        "pit80": float(((pit > 0.1) & (pit < 0.9)).mean()),
        "log_score": float(np.mean(np.log(np.clip(p_y, 1e-12, None)))),
        "width80": float(np.mean(hi - lo + 1)),
        "pit": pit,
    }


def combined_log_likelihood(idata) -> xr.DataArray:
    """Punktweise Log-Likelihood der AKTUELLEN Noten (S und M zusammen) als ein Vektor.

    Vornoten gehen bewusst nicht ein, damit Modelle mit/ohne Vornoten auf denselben
    Beobachtungen verglichen werden.
    """
    ll = idata.log_likelihood
    s = ll["S"].rename({"obs_S": "obs"})
    m = ll["M"].rename({"obs_M": "obs"})
    s = s.assign_coords(obs=np.arange(s.sizes["obs"]))
    m = m.assign_coords(obs=np.arange(m.sizes["obs"]) + s.sizes["obs"])
    return xr.concat([s, m], dim="obs").rename("grades")


def loo_data(idata):
    """Minimaler DataTree mit posterior + log_likelihood('grades') für az.loo/az.compare."""
    ll = combined_log_likelihood(idata)
    return xr.DataTree.from_dict(
        {
            "posterior": idata.posterior.to_dataset()[["sigma_s"]],
            "log_likelihood": xr.Dataset({"grades": ll}),
        }
    )
