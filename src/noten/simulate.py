"""Simulator für Notendaten mit bekannter Wahrheit.

Alle Daten sind SIMULIERT. Es werden keine echten Schülerdaten verwendet.

Latente Skala: Notenskala ("höher = schlechter"), Schwellen c_k = k + 0.5 in der Wahrheit,
d. h. ein latenter Wert von 2,3 entspricht "knapp unter 2-3-Grenze" und wird zur Note 2.

Generatives Modell (pro Schüler i in Kurs t(i), Kanal c in {s, m}):

    u_i = (u_s,i, u_m,i) ~ MVN(0, diag(tau) R(rho) diag(tau))
    b_t ~ N(0, tau_b^2)                                  (Lehrer-/Kurseffekt, beide Kanäle)
    mu_c,i = alpha_c + b_t(i) + u_c,i                     (aktuelles Niveau im Kurs)
    S_ij  = ordinal(mu_s,i + eps),  eps ~ N(0, sigma_s^2) (Klausuren)
    M_ik  = ordinal(mu_m,i + eps),  eps ~ N(0, sigma_m^2) (mündlich/sonstige Mitarbeit)
    Vornoten (Random Walk rückwärts in der Zeit, y = 1 letztes Schuljahr, y = 2 vorletztes, ...):
        lambda_c,i,0 = u_c,i,  lambda_c,i,y = lambda_c,i,y-1 + eta,  eta ~ N(0, sigma_drift^2)
        V_c,i,y = clip(round(alpha_c + delta_v + lambda_c,i,y + e), 1, 6),  e ~ N(0, sigma_v^2)
    optional: D_i = ordinal(theta_i + e_D),  theta_i = w mu_s,i + (1-w) mu_m,i  (Lehrerurteil)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

GRADES = np.arange(1, 7)
TRUE_CUTPOINTS = (1.5, 2.5, 3.5, 4.5, 5.5)


@dataclass(frozen=True)
class SimConfig:
    """Parameter eines Simulationsszenarios (Wahrheit + Design)."""

    name: str = "basis"
    n_courses: int = 8
    n_per_course: int = 25
    n_s: int = 3  # Klausuren, die zum Fitten benutzt werden
    n_s_holdout: int = 1  # zusätzliche Klausur(en) für die Vorhersageprüfung
    n_m: int = 5  # mündliche Noten
    n_years: int = 3  # Anzahl zurückliegender Schuljahre mit Vornoten
    alpha_s: float = 3.1
    alpha_m: float = 2.8
    tau_s: float = 0.8
    tau_m: float = 0.7
    rho: float = 0.5
    tau_b: float = 0.3
    sigma_s: float = 0.7
    sigma_m: float = 0.9
    sigma_drift: float = 0.3
    sigma_v: float = 0.3
    delta_v: float = -0.2
    cutpoints: tuple[float, ...] = TRUE_CUTPOINTS
    w: float = 2 / 3
    with_teacher_text: bool = False
    sigma_d: float = 0.5

    @property
    def n_students(self) -> int:
        return self.n_courses * self.n_per_course

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GradeData:
    """Ein simulierter Datensatz. Noten als ganze Zahlen 1..6."""

    course: np.ndarray  # (n,) Kursindex 0..T-1
    S: np.ndarray  # (n, n_s) Klausuren für den Fit
    S_holdout: np.ndarray  # (n, n_s_holdout) zurückgehaltene Klausur(en)
    M: np.ndarray  # (n, n_m)
    V_s: np.ndarray  # (n, n_years) schriftliche Vornoten, Spalte 0 = letztes Jahr
    V_m: np.ndarray  # (n, n_years)
    D: np.ndarray | None  # (n,) optionales Lehrerurteil
    n_courses: int
    truth: dict = field(default_factory=dict)
    config: SimConfig | None = None

    @property
    def n_students(self) -> int:
        return self.course.shape[0]


def ordinal_from_latent(latent: np.ndarray, cutpoints=TRUE_CUTPOINTS) -> np.ndarray:
    """Latente Werte -> Noten 1..6 (Note = 1 + Anzahl überschrittener Schwellen)."""
    cut = np.asarray(cutpoints)
    return 1 + np.searchsorted(cut, latent, side="left")


def simulate(cfg: SimConfig, seed: int) -> GradeData:
    """Simuliere einen Datensatz mit bekannter Wahrheit (deterministisch über `seed`)."""
    rng = np.random.default_rng(seed)
    n, T = cfg.n_students, cfg.n_courses
    course = np.repeat(np.arange(T), cfg.n_per_course)

    b = rng.normal(0.0, cfg.tau_b, size=T) if cfg.tau_b > 0 else np.zeros(T)
    cov = np.array(
        [
            [cfg.tau_s**2, cfg.rho * cfg.tau_s * cfg.tau_m],
            [cfg.rho * cfg.tau_s * cfg.tau_m, cfg.tau_m**2],
        ]
    )
    u = rng.multivariate_normal(np.zeros(2), cov, size=n)
    mu_s = cfg.alpha_s + b[course] + u[:, 0]
    mu_m = cfg.alpha_m + b[course] + u[:, 1]

    n_s_total = cfg.n_s + cfg.n_s_holdout
    S_all = ordinal_from_latent(
        mu_s[:, None] + rng.normal(0, cfg.sigma_s, size=(n, n_s_total)), cfg.cutpoints
    )
    M = ordinal_from_latent(
        mu_m[:, None] + rng.normal(0, cfg.sigma_m, size=(n, cfg.n_m)), cfg.cutpoints
    )

    def vornoten(alpha: float, u_c: np.ndarray) -> np.ndarray:
        steps = rng.normal(0.0, cfg.sigma_drift, size=(n, cfg.n_years))
        lam = u_c[:, None] + np.cumsum(steps, axis=1)
        raw = alpha + cfg.delta_v + lam + rng.normal(0, cfg.sigma_v, size=(n, cfg.n_years))
        return np.clip(np.rint(raw), 1, 6).astype(int)

    V_s = vornoten(cfg.alpha_s, u[:, 0])
    V_m = vornoten(cfg.alpha_m, u[:, 1])

    theta = cfg.w * mu_s + (1 - cfg.w) * mu_m
    D = None
    if cfg.with_teacher_text:
        D = ordinal_from_latent(theta + rng.normal(0, cfg.sigma_d, size=n), cfg.cutpoints)

    truth = {
        "alpha_s": cfg.alpha_s,
        "alpha_m": cfg.alpha_m,
        "tau_s": cfg.tau_s,
        "tau_m": cfg.tau_m,
        "rho": cfg.rho,
        "tau_b": cfg.tau_b,
        "sigma_s": cfg.sigma_s,
        "sigma_m": cfg.sigma_m,
        "sigma_drift": cfg.sigma_drift,
        "sigma_v": cfg.sigma_v,
        "delta_v": cfg.delta_v,
        "cut_inner": np.asarray(cfg.cutpoints[1:4]),
        "b": b,
        "mu_s": mu_s,
        "mu_m": mu_m,
        "theta": theta,
    }
    return GradeData(
        course=course,
        S=S_all[:, : cfg.n_s],
        S_holdout=S_all[:, cfg.n_s :],
        M=M,
        V_s=V_s,
        V_m=V_m,
        D=D,
        n_courses=T,
        truth=truth,
        config=cfg,
    )


# --------------------------------------------------------------------------------------
# Szenarien
# --------------------------------------------------------------------------------------
BASE = SimConfig()

SCENARIOS: dict[str, SimConfig] = {
    # (a) echte Unabhängigkeit der beiden Kanäle
    "a_rho0": replace(BASE, name="a_rho0", rho=0.0),
    # (b) moderate Korrelation (Kernszenario)
    "b_rho05": replace(BASE, name="b_rho05", rho=0.5),
    # (c) ohne Lehrereffekt (mit Lehrereffekt = b_rho05)
    "c_ohne_lehrer": replace(BASE, name="c_ohne_lehrer", tau_b=0.0),
    # (d) wenige vs. viele Beobachtungen
    "d_wenig": replace(BASE, name="d_wenig", n_s=2, n_m=2),
    "d_viel": replace(BASE, name="d_viel", n_s=6, n_m=10),
    # (e) Vornoten stark gedriftet (informativ = b_rho05 mit sigma_drift = 0.3)
    "e_drift": replace(BASE, name="e_drift", sigma_drift=1.2),
}


def quick_config(cfg: SimConfig) -> SimConfig:
    """Verkleinerte Version eines Szenarios für CI/Tests."""
    return replace(cfg, n_courses=8, n_per_course=6)
