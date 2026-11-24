"""Experimente zum hierarchischen Notenmodell (nur SIMULIERTE Daten).

Beispiele:
    python scripts/run_experiments.py all                 # volle Konfiguration
    python scripts/run_experiments.py all --quick         # schnelle CI-Konfiguration
    python scripts/run_experiments.py recovery --n-runs 30 --workers 4
    python scripts/run_experiments.py prior loo sensitivity grades

Ergebnisse: <out>/tables/*.md|csv, <out>/figures/*.png, <out>/results/*.csv
(Default out = reports/, mit --quick = reports/quick/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

from noten import evaluate as ev  # noqa: E402
from noten.mapping import full_cutpoints, summarize_grades  # noqa: E402
from noten.models import QUICK_SAMPLER, ModelSpec, SamplerConfig, build_model, fit  # noqa: E402
from noten.simulate import (  # noqa: E402
    BASE,
    SCENARIOS,
    ordinal_from_latent,
    quick_config,
    simulate,
)

ROOT = Path(__file__).resolve().parents[1]
CORE = list(SCENARIOS)
SCEN_INDEX = {name: i for i, name in enumerate(CORE)}


class Ctx:
    def __init__(self, args):
        self.quick = args.quick
        default_out = ROOT / ("reports/quick" if args.quick else "reports")
        self.out = Path(args.out) if args.out else default_out
        for sub in ("tables", "figures", "results"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.n_runs = args.n_runs if args.n_runs is not None else (2 if args.quick else 30)
        self.workers = args.workers
        self.sampler = QUICK_SAMPLER if args.quick else SamplerConfig(cores=1)
        self.loo_reps = args.loo_reps if args.loo_reps is not None else (1 if args.quick else 3)

    def scen(self, name):
        cfg = SCENARIOS[name]
        return quick_config(cfg) if self.quick else cfg


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """DataFrame -> Markdown-Tabelle (ohne tabulate-Abhängigkeit)."""
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_table(ctx: Ctx, df: pd.DataFrame, name: str, floatfmt=".3f"):
    df.to_csv(ctx.out / "tables" / f"{name}.csv", index=False)
    (ctx.out / "tables" / f"{name}.md").write_text(md_table(df, floatfmt), encoding="utf-8")


# --------------------------------------------------------------------------------------
# 1) Prior Predictive Checks
# --------------------------------------------------------------------------------------
def run_prior(ctx: Ctx):
    import pymc as pm

    data = simulate(ctx.scen("b_rho05"), seed=11)
    model = build_model(data, ModelSpec())
    n_draws = 200 if ctx.quick else 1000
    with model:
        prior = pm.sample_prior_predictive(
            draws=n_draws,
            var_names=["mu_s", "mu_m", "theta", "sigma_s", "sigma_m", "cut_inner"],
            random_seed=2024,
        )
    pr = prior.prior

    def get(v):
        a = np.asarray(pr[v].values)
        return a.reshape((-1,) + a.shape[2:])

    rng = np.random.default_rng(7)
    mu_s, mu_m, theta = get("mu_s"), get("mu_m"), get("theta")
    sig_s, sig_m = get("sigma_s"), get("sigma_m")
    cut = full_cutpoints(get("cut_inner"))
    n_s, n_m = data.S.shape[1], data.M.shape[1]

    def sim_grades(mu, sig, k):
        lat = mu[:, :, None] + sig[:, None, None] * rng.normal(size=mu.shape + (k,))
        return 1 + (lat[..., None] > cut[:, None, None, :]).sum(-1)

    S = sim_grades(mu_s, sig_s, n_s).reshape(n_draws, -1)
    M = sim_grades(mu_m, sig_m, n_m).reshape(n_draws, -1)
    G = 1 + (theta[..., None] > cut[:, None, :]).sum(-1)

    def freqs(x):
        return np.stack([(x == k).mean(axis=1) for k in range(1, 7)], axis=1)

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (lab, x) in zip(
        axes, [("Klausuren S", S), ("Mündlich M", M), ("Zeugnisnote aus θ", G)], strict=True
    ):
        f = freqs(x)
        q = np.quantile(f, [0.1, 0.5, 0.9], axis=0)
        ax.bar(np.arange(1, 7), q[1], color="#4C72B0", alpha=0.8, label="Median über Prior-Draws")
        ax.errorbar(
            np.arange(1, 7),
            q[1],
            yerr=[q[1] - q[0], q[2] - q[1]],
            fmt="none",
            ecolor="k",
            capsize=3,
            label="10–90 %",
        )
        ax.set_title(lab)
        ax.set_xlabel("Note")
        degenerate = (f.max(axis=1) > 0.8).mean()
        extreme = ((f[:, 0] > 0.5) | (f[:, 5] > 0.5)).mean()
        rows.append(
            {
                "größe": lab,
                **{f"anteil_{k}": float(np.mean(f[:, k - 1])) for k in range(1, 7)},
                "P(>80% eine Note)": float(degenerate),
                "P(>50% Note 1 oder 6)": float(extreme),
                "mittel_note": float(np.mean(x)),
            }
        )
    axes[0].set_ylabel("Anteil der Noten")
    axes[0].legend(fontsize=8)
    fig.suptitle("Prior Predictive: Notenverteilungen unter den Priors")
    fig.tight_layout()
    fig.savefig(ctx.out / "figures" / "prior_predictive.png", dpi=110)
    plt.close(fig)

    # Verteilung der Kursmittel und Streuungen der Noten pro Prior-Draw
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(S.mean(axis=1), bins=40, color="#4C72B0")
    axes[0].set_title("Mittlere Klausurnote je Prior-Draw")
    axes[1].hist(S.std(axis=1), bins=40, color="#DD8452")
    axes[1].set_title("SD der Klausurnoten je Prior-Draw")
    fig.tight_layout()
    fig.savefig(ctx.out / "figures" / "prior_predictive_moments.png", dpi=110)
    plt.close(fig)
    save_table(ctx, pd.DataFrame(rows), "prior_predictive")
    print("[prior] fertig")


# --------------------------------------------------------------------------------------
# 2) Parameter Recovery + 3) Kalibrierung (n+1-te Klausur)
# --------------------------------------------------------------------------------------
def _recovery_job(scen_name: str, run: int, quick: bool, sampler: SamplerConfig) -> dict:
    cfg = SCENARIOS[scen_name]
    if quick:
        cfg = quick_config(cfg)
    seed = 100_000 * (SCEN_INDEX[scen_name] + 1) + run
    data = simulate(cfg, seed=seed)
    spec = ModelSpec(use_teacher_text=cfg.with_teacher_text)
    t0 = time.time()
    _, idata = fit(data, spec, sampler, seed=seed + 7)
    diag = ev.diagnostics(idata)
    rows = ev.recovery(idata, data)
    rows.append(ev.grade_recovery(idata, data))
    probs = ev.predictive_for_holdout(idata)
    cal = ev.calibration(probs, data.S_holdout[:, 0], np.random.default_rng(seed))
    pit = cal.pop("pit")
    rho_d = ev.draws(idata, "rho") if "rho" in idata.posterior else None
    extra = {}
    if rho_d is not None:
        lo, hi = np.quantile(rho_d, [0.025, 0.975])
        extra = {
            "rho_median": float(np.median(rho_d)),
            "rho_lo95": float(lo),
            "rho_hi95": float(hi),
            "rho_excl0": bool(lo > 0 or hi < 0),
        }
    return {
        "scenario": scen_name,
        "run": run,
        "seed": seed,
        "seconds": time.time() - t0,
        "delta_fix": True,  # Vergleichswert delta_v - mean(b) (ab Commit "Recovery: ...delta_v")
        **diag,
        **cal,
        **extra,
        "rows": rows,
        "pit": pit.tolist(),
    }


def run_recovery(ctx: Ctx, scenarios=None):
    scenarios = scenarios or CORE
    raw_path = ctx.out / "results" / "recovery_runs.jsonl"
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["scenario"], r["run"]))
    jobs = [(s, r) for s in scenarios for r in range(ctx.n_runs) if (s, r) not in done]
    print(f"[recovery] {len(jobs)} Fits offen ({len(done)} bereits vorhanden)")
    if jobs:
        import multiprocessing as mp

        with ProcessPoolExecutor(
            ctx.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=8
        ) as ex:
            futs = {ex.submit(_recovery_job, s, r, ctx.quick, ctx.sampler): (s, r) for s, r in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                s, r = futs[f]
                try:
                    res = f.result()
                except Exception:
                    print(f"[recovery] FEHLER {s} run {r}:\n{traceback.format_exc()}")
                    continue
                with raw_path.open("a") as fh:
                    fh.write(json.dumps(res) + "\n")
                print(
                    f"[recovery] {i}/{len(jobs)} {s} run {r}: {res['seconds']:.0f}s "
                    f"ok={res['ok']} rhat={res['rhat_max']:.3f} ess={res['ess_bulk_min']:.0f} "
                    f"div={res['divergences']}",
                    flush=True,
                )
    summarize_recovery(ctx, scenarios)


def _load_runs(ctx: Ctx, scenarios):
    raw_path = ctx.out / "results" / "recovery_runs.jsonl"
    runs = [json.loads(line) for line in raw_path.read_text().splitlines()]
    runs = [r for r in runs if r["scenario"] in scenarios and r["run"] < ctx.n_runs]
    return runs


def summarize_recovery(ctx: Ctx, scenarios=None):
    scenarios = scenarios or CORE
    runs = _load_runs(ctx, scenarios)
    if not runs:
        return
    # Diagnostik
    diag = (
        pd.DataFrame(runs)
        .groupby("scenario", sort=False)
        .agg(
            fits=("ok", "size"),
            ok=("ok", "sum"),
            rhat_max=("rhat_max", "max"),
            ess_bulk_min=("ess_bulk_min", "min"),
            ess_tail_min=("ess_tail_min", "min"),
            divergenzen=("divergences", "sum"),
            sek_pro_fit=("seconds", "mean"),
        )
        .reset_index()
    )
    save_table(ctx, diag, "diagnostics", ".2f")

    # Recovery-Abdeckung
    rec = []
    for r in runs:
        for row in r["rows"]:
            if row["param"] == "delta_v" and not r.get("delta_fix", False):
                continue  # alter, falsch definierter Vergleichswert (siehe ERGEBNISSE.md)
            rec.append({"scenario": r["scenario"], "run": r["run"], "ok": r["ok"], **row})
    rec = pd.DataFrame(rec)
    rec.to_csv(ctx.out / "results" / "recovery_rows.csv", index=False)
    params = rec[rec.param != "note"]
    agg = (
        params.groupby(["scenario", "param"], sort=False)
        .agg(
            runs=("run", "nunique"),
            cov50=("cov50", "mean"),
            cov80=("cov80", "mean"),
            cov95=("cov95", "mean"),
            bias=("bias", "mean"),
            rmse=("rmse", "mean"),
            breite80=("width80", "mean"),
        )
        .reset_index()
    )
    save_table(ctx, agg, "recovery_long")
    # kompakte Breitform: cov80 je Parameter x Szenario
    wide = agg.pivot(index="param", columns="scenario", values="cov80")
    wide = wide.reindex(index=agg.param.unique(), columns=[s for s in scenarios if s in wide])
    wide = wide.reset_index()
    save_table(ctx, wide, "recovery_cov80", ".2f")
    for lev in ("cov50", "cov95"):
        w = agg.pivot(index="param", columns="scenario", values=lev)
        w = w.reindex(index=agg.param.unique(), columns=[s for s in scenarios if s in w])
        save_table(ctx, w.reset_index(), f"recovery_{lev}", ".2f")

    notes = rec[rec.param == "note"]
    note_agg = (
        notes.groupby("scenario", sort=False)
        .agg(
            cov80_note=("cov80", "mean"),
            breite80_noten=("width80", "mean"),
            anteil_einzelnote=("single_grade_share", "mean"),
            median_korrekt=("median_correct", "mean"),
            p_wahre_note=("p_true", "mean"),
        )
        .reset_index()
    )
    theta = agg[agg.param == "theta"][["scenario", "breite80"]].rename(
        columns={"breite80": "theta_breite80"}
    )
    note_agg = note_agg.merge(theta, on="scenario", how="left")
    save_table(ctx, note_agg, "grade_intervals")

    # Kalibrierung n+1-te Klausur
    df = pd.DataFrame(runs)
    cal = (
        df.groupby("scenario", sort=False)
        .agg(
            runs=("run", "size"),
            treffer80=("hit80", "mean"),
            sollmasse80=("mass80", "mean"),
            pit80=("pit80", "mean"),
            pit80_sd=("pit80", "std"),
            breite80=("width80", "mean"),
            log_score=("log_score", "mean"),
        )
        .reset_index()
    )
    save_table(ctx, cal, "calibration")

    # rho-Identifizierbarkeit
    if "rho_median" in df:
        rh = (
            df.groupby("scenario", sort=False)
            .agg(
                rho_median=("rho_median", "mean"),
                rho_lo95=("rho_lo95", "mean"),
                rho_hi95=("rho_hi95", "mean"),
                anteil_ci95_ohne_0=("rho_excl0", "mean"),
            )
            .reset_index()
        )
        rh["wahres_rho"] = [SCENARIOS[s].rho for s in rh.scenario]
        save_table(ctx, rh, "rho_identification")

    # PIT-Histogramme
    fig, axes = plt.subplots(2, (len(scenarios) + 1) // 2, figsize=(14, 6), squeeze=False)
    for ax, s in zip(axes.ravel(), scenarios, strict=False):
        pit_list = [r["pit"] for r in runs if r["scenario"] == s]
        if not pit_list:
            ax.axis("off")
            continue
        pits = np.concatenate(pit_list)
        ax.hist(pits, bins=10, range=(0, 1), color="#4C72B0", edgecolor="w")
        ax.axhline(len(pits) / 10, color="k", ls="--", lw=1)
        ax.set_title(s, fontsize=10)
    for ax in axes.ravel()[len(scenarios) :]:
        ax.axis("off")
    fig.suptitle("Randomisierte PIT der vorhergesagten n+1-ten Klausur (flach = kalibriert)")
    fig.tight_layout()
    fig.savefig(ctx.out / "figures" / "calibration_pit.png", dpi=110)
    plt.close(fig)

    # Abdeckungsplot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sub = agg[agg.param.isin(["mu_s", "mu_m", "theta", "rho", "b", "sigma_s", "sigma_m", "tau_b"])]
    for lev, mk in (("cov50", "o"), ("cov80", "s"), ("cov95", "^")):
        for s_i, s in enumerate(scenarios):
            d = sub[sub.scenario == s]
            x = np.arange(len(d)) + (s_i - len(scenarios) / 2) * 0.1
            ax.scatter(x, d[lev], marker=mk, s=18, label=f"{s}" if lev == "cov80" else None)
    params_order = sub.param.unique()
    ax.set_xticks(np.arange(len(params_order)))
    ax.set_xticklabels(params_order)
    for lev in (0.5, 0.8, 0.95):
        ax.axhline(lev, color="grey", ls=":", lw=1)
    ax.set_ylabel("Abdeckung")
    ax.set_title("Recovery-Abdeckung (○ 50 %, □ 80 %, △ 95 %)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(ctx.out / "figures" / "recovery_coverage.png", dpi=110)
    plt.close(fig)
    print("[recovery] Zusammenfassung geschrieben")


# --------------------------------------------------------------------------------------
# 4) LOO-Modellvergleich
# --------------------------------------------------------------------------------------
VARIANTS = {
    "voll": ModelSpec(),
    "rho0": ModelSpec(rho_free=False),
    "ohne_vornoten": ModelSpec(use_vornoten=False),
    "ohne_lehrer": ModelSpec(teacher_effect=False),
}
LOO_SCENARIOS = ["a_rho0", "b_rho05", "c_ohne_lehrer", "e_drift"]


def _loo_job(scen_name, rep, variant, quick, sampler):
    import arviz as az

    cfg = SCENARIOS[scen_name]
    if quick:
        cfg = quick_config(cfg)
    seed = 500_000 + 1000 * SCEN_INDEX[scen_name] + rep
    data = simulate(cfg, seed=seed)
    _, idata = fit(data, VARIANTS[variant], sampler, seed=seed + 3, log_likelihood=True)
    diag = ev.diagnostics(idata)
    loo = az.loo(ev.loo_data(idata), pointwise=True)
    k = np.asarray(loo.pareto_k).ravel()
    return {
        "scenario": scen_name,
        "rep": rep,
        "variant": variant,
        "elpd": float(loo.elpd),
        "se": float(loo.se),
        "p_loo": float(loo.p),
        "k_gt_07": int((k > 0.7).sum()),
        "k_max": float(k.max()),
        "pointwise": np.asarray(loo.elpd_i).ravel().tolist(),
        **diag,
    }


def run_loo(ctx: Ctx):
    import multiprocessing as mp

    jobs = [(s, r, v) for s in LOO_SCENARIOS for r in range(ctx.loo_reps) for v in VARIANTS]
    res = []
    with ProcessPoolExecutor(
        ctx.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=8
    ) as ex:
        futs = [ex.submit(_loo_job, *j, ctx.quick, ctx.sampler) for j in jobs]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                print(f"[loo] FEHLER:\n{traceback.format_exc()}")
                continue
            res.append(r)
            print(f"[loo] {r['scenario']} rep {r['rep']} {r['variant']}: elpd={r['elpd']:.1f}")
    rows = []
    for (s, rep), grp in pd.DataFrame(res).groupby(["scenario", "rep"]):
        ref = grp.loc[grp.elpd.idxmax()]
        for _, g in grp.iterrows():
            diff = np.asarray(g.pointwise) - np.asarray(ref.pointwise)
            rows.append(
                {
                    "scenario": s,
                    "rep": rep,
                    "variante": g.variant,
                    "elpd_loo": g.elpd,
                    "elpd_diff": g.elpd - ref.elpd,
                    "se_diff": float(np.sqrt(diff.size * np.var(diff))),
                    "p_loo": g.p_loo,
                    "k>0.7": g.k_gt_07,
                    "k_max": g.k_max,
                    "diag_ok": g.ok,
                }
            )
    df = pd.DataFrame(rows).sort_values(["scenario", "rep", "elpd_diff"], ascending=[1, 1, 0])
    save_table(ctx, df, "loo_compare", ".2f")
    # Aggregat über Replikate
    agg = (
        df.groupby(["scenario", "variante"])
        .agg(
            elpd_diff_mittel=("elpd_diff", "mean"),
            se_diff_mittel=("se_diff", "mean"),
            rang1_anteil=("elpd_diff", lambda x: float(np.mean(x == 0))),
            k_gt_07_summe=("k>0.7", "sum"),
            k_max=("k_max", "max"),
        )
        .reset_index()
        .sort_values(["scenario", "elpd_diff_mittel"], ascending=[1, 0])
    )
    save_table(ctx, agg, "loo_summary", ".2f")
    print("[loo] fertig")


# --------------------------------------------------------------------------------------
# 5) Prior-Sensitivität der Vornoten
# --------------------------------------------------------------------------------------
SENS = {
    "stark (HN 0.05)": dict(sigma_drift_sd=0.05, sigma_v_sd=0.05),
    "default (HN 0.5)": dict(sigma_drift_sd=0.5, sigma_v_sd=0.5),
    "schwach (HN 2.0)": dict(sigma_drift_sd=2.0, sigma_v_sd=2.0),
}


def _sens_job(scen_name, label, quick, sampler, n_years):
    cfg = SCENARIOS[scen_name]
    if quick:
        cfg = quick_config(cfg)
    cfg = replace(cfg, n_years=n_years)
    seed = 700_000 + SCEN_INDEX[scen_name]
    data = simulate(cfg, seed=seed)
    spec = ModelSpec().with_priors(**SENS[label])
    _, idata = fit(data, spec, sampler, seed=seed + 1)
    th = ev.draws(idata, "theta")
    cut = full_cutpoints(ev.draws(idata, "cut_inner"))
    summ = summarize_grades(th, cut)
    return {
        "scenario": scen_name,
        "n_years": n_years,
        "label": label,
        "theta_median": summ.theta_median.tolist(),
        "theta_w80": (summ.theta_hi - summ.theta_lo).tolist(),
        "note_median": summ.note_median.tolist(),
        "sigma_drift": float(np.median(ev.draws(idata, "sigma_drift"))),
        "sigma_v": float(np.median(ev.draws(idata, "sigma_v"))),
        "theta_true": data.truth["theta"].tolist(),
        **ev.diagnostics(idata),
    }


def run_sensitivity(ctx: Ctx):
    import multiprocessing as mp

    jobs = [(s, lab, ny) for s in ("b_rho05", "e_drift") for ny in (3, 1) for lab in SENS]
    res = []
    with ProcessPoolExecutor(
        ctx.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=8
    ) as ex:
        futs = [ex.submit(_sens_job, s, lab, ctx.quick, ctx.sampler, ny) for s, lab, ny in jobs]
        for f in as_completed(futs):
            res.append(f.result())
    rows = []
    for (s, ny), grp in pd.DataFrame(res).groupby(["scenario", "n_years"], sort=False):
        ref = grp[grp.label == "default (HN 0.5)"].iloc[0]
        for lab in SENS:
            g = grp[grp.label == lab].iloc[0]
            d = np.asarray(g.theta_median) - np.asarray(ref.theta_median)
            err = np.asarray(g.theta_median) - np.asarray(g.theta_true)
            rows.append(
                {
                    "scenario": s,
                    "vornoten_jahre": ny,
                    "prior": lab,
                    "sigma_drift_post": g.sigma_drift,
                    "sigma_v_post": g.sigma_v,
                    "mittl_|Δθ|": float(np.mean(np.abs(d))),
                    "max_|Δθ|": float(np.max(np.abs(d))),
                    "anteil_note_geändert": float(
                        np.mean(np.asarray(g.note_median) != np.asarray(ref.note_median))
                    ),
                    "θ_breite80": float(np.mean(g.theta_w80)),
                    "rmse_θ": float(np.sqrt(np.mean(err**2))),
                    "diag_ok": g.ok,
                }
            )
    save_table(ctx, pd.DataFrame(rows), "sensitivity")
    print("[sensitivity] fertig")


# --------------------------------------------------------------------------------------
# 6) Note mit und ohne Vornoten
# --------------------------------------------------------------------------------------
def run_grades(ctx: Ctx):
    cfg = ctx.scen("b_rho05")
    seed = 900_001
    data = simulate(cfg, seed=seed)
    out = {}
    diags = {}
    for lab, spec in (("mit", ModelSpec()), ("ohne", ModelSpec(use_vornoten=False))):
        _, idata = fit(data, spec, replace(ctx.sampler, cores=min(4, ctx.workers)), seed=seed)
        diags[lab] = ev.diagnostics(idata)
        out[lab] = summarize_grades(
            ev.draws(idata, "theta"), full_cutpoints(ev.draws(idata, "cut_inner"))
        )
    true_grade = ordinal_from_latent(data.truth["theta"], cfg.cutpoints)
    S_mean = data.S.mean(axis=1)
    M_mean = data.M.mean(axis=1)
    naive = np.rint(cfg.w * S_mean + (1 - cfg.w) * M_mean).astype(int)
    tab = pd.DataFrame(
        {
            "schüler": np.arange(data.n_students),
            "kurs": data.course,
            "Ø_S": S_mean.round(2),
            "Ø_M": M_mean.round(2),
            "Ø_Vornote_S": data.V_s.mean(axis=1).round(2),
            "naiv_gerundet": naive,
            "mit_vornoten": [
                f"{m} [{lo}–{hi}]"
                for m, lo, hi in zip(
                    out["mit"].note_median, out["mit"].note_lo, out["mit"].note_hi, strict=True
                )
            ],
            "θ_mit": [
                f"{m:.2f} [{lo:.2f}; {hi:.2f}]"
                for m, lo, hi in zip(
                    out["mit"].theta_median, out["mit"].theta_lo, out["mit"].theta_hi, strict=True
                )
            ],
            "ohne_vornoten": [
                f"{m} [{lo}–{hi}]"
                for m, lo, hi in zip(
                    out["ohne"].note_median, out["ohne"].note_lo, out["ohne"].note_hi, strict=True
                )
            ],
            "θ_ohne": [
                f"{m:.2f} [{lo:.2f}; {hi:.2f}]"
                for m, lo, hi in zip(
                    out["ohne"].theta_median,
                    out["ohne"].theta_lo,
                    out["ohne"].theta_hi,
                    strict=True,
                )
            ],
            "θ_wahr": data.truth["theta"].round(2),
            "note_wahr": true_grade,
        }
    )
    tab.to_csv(ctx.out / "results" / "noten_mit_ohne_vornoten.csv", index=False)
    save_table(ctx, tab.head(15), "grades_side_by_side_head")

    summ = []
    for lab in ("mit", "ohne"):
        o = out[lab]
        summ.append(
            {
                "modell": f"{lab} Vornoten",
                "Ø_breite_θ80": float(np.mean(o.theta_hi - o.theta_lo)),
                "Ø_breite_note80": float(np.mean(o.note_hi - o.note_lo + 1)),
                "anteil_einzelnote": float(np.mean(o.note_lo == o.note_hi)),
                "median=wahr": float(np.mean(o.note_median == true_grade)),
                "abdeckung80_note": float(
                    np.mean((true_grade >= o.note_lo) & (true_grade <= o.note_hi))
                ),
                "rmse_θ": float(np.sqrt(np.mean((o.theta_median - data.truth["theta"]) ** 2))),
                "diag_ok": diags[lab]["ok"],
            }
        )
    summ.append(
        {
            "modell": "naiv: gewichteter Schnitt gerundet",
            "Ø_breite_θ80": np.nan,
            "Ø_breite_note80": np.nan,
            "anteil_einzelnote": np.nan,
            "median=wahr": float(np.mean(naive == true_grade)),
            "abdeckung80_note": np.nan,
            "rmse_θ": float(
                np.sqrt(np.mean((cfg.w * S_mean + (1 - cfg.w) * M_mean - data.truth["theta"]) ** 2))
            ),
            "diag_ok": np.nan,
        }
    )
    save_table(ctx, pd.DataFrame(summ), "grades_summary")

    # Plot: ein Kurs (25 Schüler)
    idx = np.where(data.course == 0)[0]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for off, lab, col in ((-0.15, "mit", "#4C72B0"), (0.15, "ohne", "#DD8452")):
        o = out[lab].iloc[idx]
        x = np.arange(idx.size) + off
        ax.errorbar(
            x,
            o.theta_median,
            yerr=[o.theta_median - o.theta_lo, o.theta_hi - o.theta_median],
            fmt="o",
            ms=4,
            color=col,
            label=f"θ {lab} Vornoten (Median, 80 %)",
        )
    ax.scatter(np.arange(idx.size), data.truth["theta"][idx], marker="x", color="k", label="θ wahr")
    for c in cfg.cutpoints:
        ax.axhline(c, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Schüler in Kurs 0")
    ax.set_ylabel("latente Note θ (Notenskala)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title("Zeugnisnote als Verteilung: mit vs. ohne Vornoten (simuliert)")
    fig.tight_layout()
    fig.savefig(ctx.out / "figures" / "grades_mit_ohne_vornoten.png", dpi=110)
    plt.close(fig)
    print("[grades] fertig")


# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "steps",
        nargs="+",
        choices=["prior", "recovery", "summary", "loo", "sensitivity", "grades", "all"],
    )
    ap.add_argument("--quick", action="store_true", help="schnelle CI-Konfiguration")
    ap.add_argument("--n-runs", type=int, default=None, help="Simulationsläufe je Szenario")
    ap.add_argument("--loo-reps", type=int, default=None, help="Datensätze je LOO-Szenario")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    ctx = Ctx(args)
    steps = args.steps
    if "all" in steps:
        steps = ["prior", "recovery", "loo", "sensitivity", "grades"]
    print(f"Ausgabe: {ctx.out} | quick={ctx.quick} | n_runs={ctx.n_runs} | base={BASE.name}")
    for st in steps:
        t0 = time.time()
        if st == "prior":
            run_prior(ctx)
        elif st == "recovery":
            run_recovery(ctx, args.scenarios)
        elif st == "summary":
            summarize_recovery(ctx, args.scenarios)
        elif st == "loo":
            run_loo(ctx)
        elif st == "sensitivity":
            run_sensitivity(ctx)
        elif st == "grades":
            run_grades(ctx)
        print(f"== {st}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
