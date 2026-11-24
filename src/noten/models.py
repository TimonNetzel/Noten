"""Hierarchisches Bayes-Modell (PyMC) für die Zeugnisnote.

Siehe README.md für die Mathematik. Kurzfassung (Kanal c in {s, m}):

    S_ij ~ OrderedProbit(mu_s,i, c, sigma_s)          M_ik ~ OrderedProbit(mu_m,i, c, sigma_m)
    mu_c,i = alpha_c + b_t(i) + u_c,i
    u_i = L z_i,  z_i ~ N(0, I),  L L^T = Sigma = diag(tau) R diag(tau),  R ~ LKJ(eta)
    b_t = tau_b * z_b,t,  z_b,t ~ N(0, 1)                    (nicht-zentriert)
    V_c,i,1..Y ~ MVN(alpha_c + delta_v + u_c,i, K),  K_yy' = min(y, y') sigma_drift^2
                                                          + 1[y = y'] sigma_v^2
    Schwellen: c_1 = 1.5, c_5 = 5.5 fest, (c_2, c_3, c_4) = 1.5 + 4 * cumsum(Dirichlet(a))

Identifizierbarkeit: Ein ordinales Probit-Modell ist nur bis auf eine affine Transformation
(Lage + Skala) der latenten Achse bestimmt. Die beiden festen äußeren Schwellen legen genau
diese zwei Freiheitsgrade fest. Damit sind alpha_c, sigma_s, sigma_m und die inneren
Schwellen identifiziert und die latente Achse ist in "Notenpunkten" interpretierbar.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .simulate import GradeData

C_LOW, C_HIGH = 1.5, 5.5


@dataclass(frozen=True)
class PriorConfig:
    """Hyperparameter der Priors (alle Streuungen Half-Normal)."""

    alpha_mu: float = 3.5
    alpha_sd: float = 1.0
    tau_sd: float = 1.0  # Streuung der Schülerniveaus
    sigma_sd: float = 1.0  # Residual-Streuung pro Kanal
    tau_b_sd: float = 0.5  # Lehrer-/Kurseffekt
    lkj_eta: float = 2.0
    sigma_drift_sd: float = 0.5  # Random-Walk-Drift der Vornoten pro Jahr
    sigma_v_sd: float = 0.5  # Messfehler einer Vornote
    delta_sd: float = 0.5  # systematische Verschiebung Vornoten vs. aktueller Kurs
    cut_conc: float = 4.0  # Dirichlet-Konzentration der Schwellenabstände
    sigma_d_sd: float = 1.0  # Lehrerurteil


@dataclass(frozen=True)
class ModelSpec:
    """Welche Modellteile aktiv sind."""

    rho_free: bool = True
    use_vornoten: bool = True
    teacher_effect: bool = True
    use_teacher_text: bool = False
    w: float = 2 / 3
    priors: PriorConfig = field(default_factory=PriorConfig)

    def with_priors(self, **kw) -> ModelSpec:
        return replace(self, priors=replace(self.priors, **kw))


def _cutpoints(conc: float):
    gaps = pm.Dirichlet("cut_gaps", a=np.full(4, conc))
    inner = C_LOW + (C_HIGH - C_LOW) * pt.cumsum(gaps)[:3]
    pm.Deterministic("cut_inner", inner)
    return pt.concatenate([pt.as_tensor([C_LOW]), inner, pt.as_tensor([C_HIGH])])


def _probit_interval_logp(eta, sigma, c_lo, c_up):
    """log[Phi((c_up - eta)/sigma) - Phi((c_lo - eta)/sigma)], numerisch stabil."""
    upper = (c_up - eta) / sigma
    lower = (c_lo - eta) / sigma
    # Symmetrie: im oberen Schwanz Phi(u) - Phi(l) = Phi(-l) - Phi(-u)
    flip = (upper + lower) > 0
    u = pt.where(flip, -lower, upper)
    lo = pt.where(flip, -upper, lower)
    p = 0.5 * (pt.erfc(-u / np.sqrt(2.0)) - pt.erfc(-lo / np.sqrt(2.0)))
    return pt.log(pt.maximum(p, 1e-300))


def _ordinal(name, eta, sigma, cut, observed, dims):
    """Cumulative-Probit-Likelihood, observed in 0..5.

    Äquivalent zu pm.OrderedProbit (siehe Test), aber schneller: je Beobachtung werden nur die
    zwei relevanten Schwellen ausgewertet statt aller sechs Kategorien.
    """
    observed = np.asarray(observed, dtype="int64")
    big = pt.as_tensor([1e3])
    padded = pt.concatenate([-big, cut, big])
    c_lo = padded[observed]
    c_up = padded[observed + 1]

    def logp(value, eta, sigma, c_lo, c_up):
        return _probit_interval_logp(eta, sigma, c_lo, c_up)

    return pm.CustomDist(
        name,
        eta,
        sigma,
        c_lo,
        c_up,
        logp=logp,
        observed=observed,
        dims=dims,
        dtype="int64",
    )


def build_model(data: GradeData, spec: ModelSpec | None = None) -> pm.Model:
    """Baue das PyMC-Modell für einen Datensatz."""
    spec = spec or ModelSpec()
    p = spec.priors
    n, T = data.n_students, data.n_courses
    course = data.course
    S_idx = np.repeat(np.arange(n), data.S.shape[1])
    M_idx = np.repeat(np.arange(n), data.M.shape[1])
    coords = {
        "student": np.arange(n),
        "course": np.arange(T),
        "channel": ["s", "m"],
        "obs_S": np.arange(S_idx.size),
        "obs_M": np.arange(M_idx.size),
    }

    with pm.Model(coords=coords) as model:
        cut = _cutpoints(p.cut_conc)
        alpha = pm.Normal("alpha", p.alpha_mu, p.alpha_sd, dims="channel")

        # Schülerniveaus (nicht-zentriert)
        z = pm.Normal("z_u", 0.0, 1.0, dims=("student", "channel"))
        if spec.rho_free:
            chol, corr, tau = pm.LKJCholeskyCov(
                "chol_u",
                n=2,
                eta=p.lkj_eta,
                sd_dist=pm.HalfNormal.dist(p.tau_sd, shape=2),
                compute_corr=True,
            )
            pm.Deterministic("rho", corr[0, 1])
            pm.Deterministic("tau", tau, dims="channel")
            u = pt.dot(z, chol.T)
        else:
            tau = pm.HalfNormal("tau", p.tau_sd, dims="channel")
            u = z * tau

        # Lehrer-/Kurseffekt (nicht-zentriert, additiv in beiden Kanälen)
        if spec.teacher_effect:
            tau_b = pm.HalfNormal("tau_b", p.tau_b_sd)
            # Zero-Sum (Summe der Kurseffekte = 0): trennt b_t sauber vom Kanalmittel alpha_c
            # (Reparametrisierungsversuch 1, siehe README "Annahmen"). Skalierung so, dass
            # Var(b_t) = tau_b^2 wie bei unabhängigen N(0, tau_b^2)-Effekten.
            z_b = pm.ZeroSumNormal("z_b", sigma=1.0, dims="course")
            b = pm.Deterministic("b", tau_b * z_b * np.sqrt(T / (T - 1)), dims="course")
            b_i = b[course]
        else:
            b_i = pt.zeros(n)

        mu_s = pm.Deterministic("mu_s", alpha[0] + b_i + u[:, 0], dims="student")
        mu_m = pm.Deterministic("mu_m", alpha[1] + b_i + u[:, 1], dims="student")
        theta = pm.Deterministic("theta", spec.w * mu_s + (1 - spec.w) * mu_m, dims="student")

        # Beobachtungen: gepoolte Residual-Streuungen pro Kanal
        sigma_s = pm.HalfNormal("sigma_s", p.sigma_sd)
        sigma_m = pm.HalfNormal("sigma_m", p.sigma_sd)
        _ordinal("S", mu_s[S_idx], sigma_s, cut, data.S.ravel() - 1, "obs_S")
        _ordinal("M", mu_m[M_idx], sigma_m, cut, data.M.ravel() - 1, "obs_M")

        # Vornoten: marginalisierter Random Walk -> MVN über die Jahre
        if spec.use_vornoten:
            Y = data.V_s.shape[1]
            sigma_drift = pm.HalfNormal("sigma_drift", p.sigma_drift_sd)
            sigma_v = pm.HalfNormal("sigma_v", p.sigma_v_sd)
            delta_v = pm.Normal("delta_v", 0.0, p.delta_sd)
            yy = np.arange(1, Y + 1)
            K = sigma_drift**2 * np.minimum.outer(yy, yy) + sigma_v**2 * np.eye(Y)
            chol_K = pt.linalg.cholesky(K)
            pm.MvNormal(
                "V_s",
                mu=(alpha[0] + delta_v + u[:, 0])[:, None],
                chol=chol_K,
                observed=data.V_s.astype(float),
            )
            pm.MvNormal(
                "V_m",
                mu=(alpha[1] + delta_v + u[:, 1])[:, None],
                chol=chol_K,
                observed=data.V_m.astype(float),
            )

        # Optionaler dritter Kanal: Lehrerurteil als ordinale Kategorie, bedingt auf theta
        if spec.use_teacher_text:
            if data.D is None:
                raise ValueError("use_teacher_text=True, aber data.D ist None")
            sigma_d = pm.HalfNormal("sigma_d", p.sigma_d_sd)
            _ordinal("D", theta, sigma_d, cut, data.D - 1, "student")
    return model


def _default_backend() -> str:
    try:
        import nutpie  # noqa: F401

        return "nutpie"
    except ImportError:
        return "pymc"


@dataclass(frozen=True)
class SamplerConfig:
    draws: int = 1500
    tune: int = 1000
    chains: int = 4
    cores: int = 1
    target_accept: float = 0.9
    backend: str = field(default_factory=_default_backend)


QUICK_SAMPLER = SamplerConfig(draws=300, tune=300, chains=2, cores=1, target_accept=0.9)


def fit(
    data: GradeData,
    spec: ModelSpec | None = None,
    sampler: SamplerConfig | None = None,
    seed: int = 0,
    log_likelihood: bool = False,
):
    """Fitte das Modell per NUTS (nutpie, falls installiert, sonst PyMC). -> (model, idata)."""
    sampler = sampler or SamplerConfig()
    model = build_model(data, spec)
    with model:
        if sampler.backend == "nutpie":
            import nutpie

            compiled = nutpie.compile_pymc_model(model)
            idata = nutpie.sample(
                compiled,
                draws=sampler.draws,
                tune=sampler.tune,
                chains=sampler.chains,
                cores=sampler.cores,
                target_accept=sampler.target_accept,
                seed=seed,
                progress_bar=False,
            )
        else:
            idata = pm.sample(
                draws=sampler.draws,
                tune=sampler.tune,
                chains=sampler.chains,
                cores=sampler.cores,
                target_accept=sampler.target_accept,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
        if log_likelihood:
            pm.compute_log_likelihood(idata, var_names=["S", "M"], progressbar=False)
    return model, idata
