"""Abbildung latenter Posterior-Samples auf Notenwahrscheinlichkeiten.

P(Note = k | Daten) = P(c_{k-1} < theta_i <= c_k | Daten), c_0 = -inf, c_6 = +inf,
geschätzt als Anteil der Posterior-Draws (theta_i^(d), c^(d)), die in Intervall k fallen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

N_GRADES = 6


def full_cutpoints(cut_inner: np.ndarray, low: float = 1.5, high: float = 5.5) -> np.ndarray:
    """(D, 3) innere Schwellen -> (D, 5) alle Schwellen."""
    cut_inner = np.atleast_2d(cut_inner)
    D = cut_inner.shape[0]
    return np.column_stack([np.full(D, low), cut_inner, np.full(D, high)])


def grade_probabilities(theta: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Notenwahrscheinlichkeiten aus Posterior-Draws.

    theta: (D, n) Draws von theta_i; cut: (D, 5) oder (5,) Schwellen.
    Rückgabe: (n, 6), Zeilen summieren zu 1.
    """
    theta = np.atleast_2d(theta)
    D, n = theta.shape
    cut = np.broadcast_to(np.atleast_2d(cut), (D, cut.shape[-1]))
    # Note = 1 + Anzahl Schwellen c_k mit theta > c_k  (theta == c_k -> untere Note, "<=")
    grade = 1 + (theta[:, :, None] > cut[:, None, :]).sum(axis=2)  # (D, n)
    probs = np.stack([(grade == k).mean(axis=0) for k in range(1, N_GRADES + 1)], axis=1)
    return probs


def predictive_probabilities(mu: np.ndarray, cut: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Posterior-prädiktive Verteilung einer NEUEN Einzelnote (z. B. nächste Klausur).

    Rao-Blackwellisiert: Mittel über Draws von Phi((c_k - mu)/sigma)-Differenzen.
    mu: (D, n), cut: (D, 5), sigma: (D,). Rückgabe (n, 6).
    """
    mu = np.atleast_2d(mu)
    cdf = norm.cdf((cut[:, None, :] - mu[:, :, None]) / sigma[:, None, None])  # (D, n, 5)
    D, n, _ = cdf.shape
    cdf = np.concatenate([np.zeros((D, n, 1)), cdf, np.ones((D, n, 1))], axis=2)
    return np.diff(cdf, axis=2).mean(axis=0)


def discrete_quantile(probs: np.ndarray, q: float) -> np.ndarray:
    """Kleinste Note k mit P(Note <= k) >= q (zeilenweise)."""
    cdf = np.cumsum(probs, axis=1)
    return 1 + (cdf < q - 1e-12).sum(axis=1)


def central_interval(probs: np.ndarray, prob: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - prob) / 2
    return discrete_quantile(probs, a), discrete_quantile(probs, 1 - a)


def summarize_grades(theta: np.ndarray, cut: np.ndarray, prob: float = 0.8) -> pd.DataFrame:
    """Pro Schüler: Median + Intervall der Note sowie des kontinuierlichen theta."""
    probs = grade_probabilities(theta, cut)
    lo, hi = central_interval(probs, prob)
    a = (1 - prob) / 2
    df = pd.DataFrame(
        {
            "note_median": discrete_quantile(probs, 0.5),
            "note_lo": lo,
            "note_hi": hi,
            "note_modus": probs.argmax(axis=1) + 1,
            "p_modus": probs.max(axis=1),
            "theta_median": np.median(theta, axis=0),
            "theta_lo": np.quantile(theta, a, axis=0),
            "theta_hi": np.quantile(theta, 1 - a, axis=0),
        }
    )
    for k in range(N_GRADES):
        df[f"p{k + 1}"] = probs[:, k]
    return df
