import numpy as np
import pytest

from noten.mapping import (
    central_interval,
    discrete_quantile,
    full_cutpoints,
    grade_probabilities,
    predictive_probabilities,
    summarize_grades,
)

CUT = np.array([1.5, 2.5, 3.5, 4.5, 5.5])


def test_probabilities_sum_to_one_random():
    rng = np.random.default_rng(0)
    theta = rng.normal(3.5, 2.0, size=(500, 40))
    inner = np.sort(rng.uniform(1.6, 5.4, size=(500, 3)), axis=1)
    probs = grade_probabilities(theta, full_cutpoints(inner))
    assert probs.shape == (40, 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_probabilities_known_values():
    theta = np.array([[1.0], [2.0], [2.0], [6.0]])  # 4 Draws, 1 Schüler
    probs = grade_probabilities(theta, CUT)
    np.testing.assert_allclose(probs[0], [0.25, 0.5, 0, 0, 0, 0.25])


def test_boundary_belongs_to_lower_grade():
    probs = grade_probabilities(np.array([[2.5]]), CUT)
    assert probs[0, 1] == 1.0  # P(c_1 < 2.5 <= c_2) -> Note 2


def test_discrete_quantile_and_interval():
    probs = np.array([[0.05, 0.2, 0.5, 0.2, 0.05, 0.0], [0, 0, 0, 0, 0, 1.0]])
    assert discrete_quantile(probs, 0.5).tolist() == [3, 6]
    lo, hi = central_interval(probs, 0.8)
    assert lo.tolist() == [2, 6] and hi.tolist() == [4, 6]


def test_predictive_probabilities_sum_and_symmetry():
    rng = np.random.default_rng(1)
    mu = rng.normal(3.5, 1, size=(200, 10))
    sig = np.abs(rng.normal(0.8, 0.1, size=200))
    p = predictive_probabilities(mu, full_cutpoints(np.tile([2.5, 3.5, 4.5], (200, 1))), sig)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    # mu genau in der Mitte der symmetrischen Schwellen -> symmetrische Verteilung
    p0 = predictive_probabilities(np.full((1, 1), 3.5), CUT[None, :], np.array([1.0]))
    np.testing.assert_allclose(p0[0], p0[0][::-1], atol=1e-12)


def test_summarize_grades_columns():
    rng = np.random.default_rng(2)
    theta = rng.normal([1.2, 3.4, 5.8], 0.3, size=(1000, 3))
    df = summarize_grades(theta, CUT)
    assert df.note_median.tolist() == [1, 3, 6]
    np.testing.assert_allclose(df[[f"p{k}" for k in range(1, 7)]].sum(axis=1), 1.0)
    assert (df.note_lo <= df.note_median).all() and (df.note_median <= df.note_hi).all()


@pytest.mark.parametrize("n_draws", [1, 7])
def test_full_cutpoints(n_draws):
    c = full_cutpoints(np.tile([2.4, 3.5, 4.6], (n_draws, 1)))
    assert c.shape == (n_draws, 5)
    assert (np.diff(c, axis=1) > 0).all()
