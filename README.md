# Noten – hierarchisches Bayes-Modell zur Zeugnisnote

> **Nur simulierte Daten.** Dieses Repository enthält und verarbeitet keine echten Schülerdaten.
> Alle Datensätze werden von `src/noten/simulate.py` mit festen Seeds erzeugt.

Statt einer einzelnen Ganzzahl liefert das Modell eine **Verteilung über die Zeugnisnote**:
`P(Note = k | Daten)` für k = 1…6, berichtet als **Median + 80-%-Intervall**
(z. B. „3 [2–3]“) sowie als kontinuierliche latente Note θ (z. B. „2,71 [2,38; 3,05]“).

Ergebnisse der Simulationsstudie: [`reports/ERGEBNISSE.md`](reports/ERGEBNISSE.md).

## Modell in Kurzform

Notation: Schüler i, Kurs/Lehrkraft t(i) (T ≥ 8 Kurse à ca. 25), Kanal c ∈ {s (schriftlich), m (mündlich)}.
Die latente Achse ist die **Notenskala** (größer = schlechter).

**Beobachtungen (Cumulative Probit, gemeinsame geordnete Schwellen c₁ < … < c₅):**

```
S_ij ∈ {1..6}:  P(S_ij ≤ k) = Φ((c_k − μ_s,i) / σ_s)      j = 1..n_s  (Klausuren)
M_ik ∈ {1..6}:  P(M_ik ≤ k) = Φ((c_k − μ_m,i) / σ_m)      k = 1..n_m  (mündlich/sonstige Mitarbeit)
optional D_i:   P(D_i ≤ k)  = Φ((c_k − θ_i) / σ_D)                    (Lehrerurteil, bedingt auf μ_s, μ_m)
```

**Latente Niveaus:**

```
μ_c,i = α_c + b_t(i) + u_c,i
u_i = (u_s,i, u_m,i) ~ MVN(0, Σ),   Σ = diag(τ) · R · diag(τ),   R ~ LKJ(η = 2),  ρ = R₁₂
b_t = τ_b · z_t · sqrt(T/(T−1)),    z ~ ZeroSumNormal(1)          (Lehrer-/Kurseffekt, beide Kanäle)
```

Nicht-zentriert: `u_i = L · z_i` mit `z_i ~ N(0, I)` und Cholesky-Faktor L (LKJCholeskyCov);
`b_t` ebenfalls als `τ_b · z_t`. Das Modell „ρ = 0“ ersetzt R durch die Einheitsmatrix
(echte Unabhängigkeit der Kanäle, gegeben Kurs und Population).

**Vornoten als Random Walk (Daten, keine Unbekannten).** y = 1 ist das letzte, y = 2 das vorletzte Schuljahr.
Rückwärts in der Zeit: λ_c,i,0 = u_c,i, λ_c,i,y = λ_c,i,y−1 + η_y mit η_y ~ N(0, σ_drift²),
gemessen als V_c,i,y = α_c + δ_v + λ_c,i,y + e mit e ~ N(0, σ_v²). Der Random Walk wird analytisch
marginalisiert:

```
(V_c,i,1, …, V_c,i,Y) ~ MVN(α_c + δ_v + u_c,i,  K),   K_yy' = min(y, y') · σ_drift² + 1[y = y'] · σ_v²
```

Ältere Noten haben dadurch größere Varianz (y · σ_drift²) und damit **automatisch weniger Gewicht**; wie
viel weniger, lernt das Modell über σ_drift. Kombiniert mit dem Populationsprior ist das exakt ein
Prior `(μ_s,i, μ_m,i) ~ MVN(Kursmittel + Vornoten-Beitrag, Σ_i)` (konjugiertes Gauss-Update; der
„Vornoten-Beitrag“ ist die präzisionsgewichtete Abweichung der Vornoten vom Mittel). Der Kurseffekt b_t
geht bewusst **nicht** in die Vornoten ein (andere Lehrkräfte in den Vorjahren) – dadurch helfen die
Vornoten auch, Lehrereffekt und Kurszusammensetzung zu trennen.

**Zeugnisnote:**

```
θ_i = w · μ_s,i + (1 − w) · μ_m,i            (w konfigurierbar, Default 2/3)
P(Note = k | Daten) = P(c_{k−1} < θ_i ≤ c_k | Daten),   c_0 = −∞, c_6 = +∞
```

geschätzt als Anteil der Posterior-Draws (θ_i, c) im Intervall k (`noten.mapping.grade_probabilities`).
Median und 80-%-Intervall sind die 50-, 10- und 90-%-Quantile dieser diskreten Verteilung.

**Priors (schwach informativ, Streuungen Half-Normal):**
α_c ~ N(3,5; 1), τ_c ~ HN(1), σ_s, σ_m ~ HN(1), τ_b ~ HN(0,5), R ~ LKJ(2), σ_drift ~ HN(0,5),
σ_v ~ HN(0,5), δ_v ~ N(0; 0,5), Schwellenabstände ~ Dirichlet(4,4,4,4), optional σ_D ~ HN(1).
Alle Hyperparameter in `noten.models.PriorConfig`.

### Identifizierbarkeit

Ein ordinales Probit-Modell bestimmt die latente Achse nur bis auf eine affine Transformation
x ↦ a + b·x: verschiebt man alle μ und alle Schwellen um a und skaliert alles (inkl. σ) mit b, bleiben
alle Wahrscheinlichkeiten gleich. Das sind genau **zwei** Freiheitsgrade (Lage, Skala). Sie werden
festgelegt, indem die **äußeren Schwellen fix** sind: **c₁ = 1,5 und c₅ = 5,5**. Die inneren Schwellen
c₂, c₃, c₄ = 1,5 + 4 · cumsum(Dirichlet) sind frei, aber geordnet und zwischen c₁ und c₅. Folgen:

* Lage und Skala sind fixiert → α_c, σ_s, σ_m, τ, τ_b und c₂…c₄ sind identifiziert
  (σ_s und σ_m sind getrennt identifiziert, weil beide Kanäle *dieselben* Schwellen teilen).
* Die latente Achse ist in **Notenpunkten** interpretierbar (θ = 2,7 heißt „zwischen 2 und 3, näher an 3“).
* Kanalmittel α_c vs. Kurseffekte: durch die Zero-Sum-Bedingung Σ_t b_t = 0 ist α_c das Mittel über
  die Kurse, b_t die Abweichung davon (ohne diese Bedingung sind α_c und mean(b) nur schwach getrennt;
  siehe „Annahmen“, Reparametrisierung).
* Vornoten haben ein eigenes Offset δ_v; σ_drift und σ_v sind über die Kovarianzstruktur mehrerer
  Jahre getrennt identifiziert (Var steigt mit y, Kovarianz = min(y, y')·σ_drift²). Mit nur **einem**
  Vornotenjahr sind σ_drift und σ_v *nicht* getrennt identifiziert (nur ihre Summe).
* ρ ist identifiziert, weil jeder Schüler mehrere Beobachtungen in beiden Kanälen hat; wie *gut*,
  hängt von der Datenmenge ab (siehe Ergebnisse).

## Annahmen

Bei Unklarheiten wurde ohne Rückfrage eine begründete Annahme getroffen:

1. **Latente Achse = Notenskala, äußere Schwellen fix (1,5 / 5,5).** Alternative wäre Residual-SD = 1 mit
   freien Schwellen; die gewählte Variante hält die Achse in Notenpunkten, erlaubt *getrennte*
   Residualstreuungen σ_s, σ_m und lässt die inneren Schwellen frei (Lehrkräfte vergeben z. B. 2en
   und 3en leichter als 5en).
2. **σ_s, σ_m sind die Residualstreuungen der Einzelnoten um das Schülerniveau** (über alle Schüler gepoolt).
   τ_s, τ_m sind die Streuungen der Niveaus zwischen Schülern.
3. **Lehrereffekt b_t ist ein Skalar je Kurs, additiv in beiden Kanälen** und wird in θ_i mitgenommen
   (es ist die Note *in diesem Kurs*). Eine „lehrerbereinigte“ Note wäre θ_i − b_t(i).
4. **Zero-Sum-Kurseffekte** (Reparametrisierungsversuch 1): Der erste Fit mit unabhängigen
   b_t ~ N(0, τ_b²) hatte R-hat 1,03 und ESS ≈ 210 für α_c, δ_v und b_t (Kamm zwischen α und mean(b)).
   Mit ZeroSumNormal verschwand das Problem. Konsequenz für die Recovery: verglichen wird mit
   b_t − mean(b) und α_c + mean(b) (die im Modell definierten Größen).
5. **Vornoten werden als stetige Messungen** der latenten Note behandelt (nicht ordinal). Rundung auf
   ganze Noten (Varianz 1/12) und Abschneiden bei 1 und 6 werden **nicht** modelliert; σ_v absorbiert die
   Rundung, deshalb wird σ_v in der Recovery mit sqrt(σ_v² + 1/12) verglichen. Das Abschneiden ist eine
   bewusste, kleine Fehlspezifikation.
6. **Ein gemeinsames σ_drift und σ_v für beide Kanäle**, Drift-Schritte der Kanäle unabhängig, ein
   gemeinsames Offset δ_v (z. B. „Notenschock“ beim Übergang in die Oberstufe).
7. **Wahrheit in der Simulation:** α_s = 3,1, α_m = 2,8, τ_s = 0,8, τ_m = 0,7, τ_b = 0,3, σ_s = 0,7,
   σ_m = 0,9, σ_drift = 0,3 (informativ) bzw. 1,2 (stark gedriftet), σ_v = 0,3, δ_v = −0,2,
   wahre Schwellen äquidistant (k + 0,5), 3 Vornotenjahre. Basisdesign: 8 Kurse × 25 Schüler,
   3 Klausuren + 5 mündliche Noten (+1 zurückgehaltene Klausur für die Vorhersage).
   Wenig: 2 + 2; viel: 6 + 10. Die Kurseffekte b_t werden je Datensatz neu gezogen.
8. **Recovery = frequentistische Abdeckung bei fester Wahrheit** (keine Simulation-Based Calibration
   mit Ziehung aus dem Prior). Bei 30 Läufen hat eine Abdeckung von 80 % für *Einzelparameter*
   einen Standardfehler von ≈ 7 Prozentpunkten; für μ_s,i (200 Schüler × 30 Läufe) ist sie präzise.
9. **„n+1-te Klausur“:** Jeder Datensatz enthält n_s + 1 Klausuren; gefittet wird auf n_s, vorhergesagt
   wird die letzte (posterior-prädiktiv, Rao-Blackwellisiert über Φ). Da Noten diskret sind, haben
   zentrale 80-%-Intervalle [q₁₀, q₉₀] mehr als 80 % Masse; berichtet werden deshalb Trefferquote,
   die vorhergesagte Sollmasse und die **randomisierte PIT** (Anteil in (0,1; 0,9) – soll 0,80 sein).
10. **LOO** wird nur auf den aktuellen Noten (S und M, punktweise) berechnet, nicht auf den Vornoten –
    sonst wären Modelle mit/ohne Vornoten nicht vergleichbar.
11. **Prior-Sensitivität der Vornoten:** variiert werden die Half-Normal-Skalen von σ_drift *und* σ_v
    gemeinsam (0,05 „stark“ = Vornoten gelten als sehr verlässlich; 0,5 Default; 2,0 „schwach“), jeweils
    mit 3 und mit nur 1 Vornotenjahr.
12. **Sampler:** nutpie (NUTS) mit 4 Ketten × 1500 Draws, 1000 Tuning, target_accept 0,9. Falls nutpie
    fehlt, wird `pm.sample` verwendet. Die Ordinal-Likelihood ist eine eigene, zu `pm.OrderedProbit`
    äquivalente Implementierung (Test `test_custom_ordinal_matches_pymc_orderedprobit`), die nur die zwei
    relevanten Schwellen je Beobachtung auswertet (~2× schneller).
13. **Repository war leer** (kein `main`). Der Pull Request braucht einen `main`-Branch als Basis
    (siehe PR-Beschreibung).

## Struktur

```
src/noten/simulate.py   Simulator + Szenarien (a) ρ=0, (b) ρ=0,5, (c) ohne Lehrereffekt, (d) wenig/viel, (e) Drift
src/noten/models.py     PyMC-Modell, Varianten (ρ=0, ohne Vornoten, ohne Lehrereffekt, Lehrerurteil), Fit
src/noten/mapping.py    θ-Draws → P(Note=k), Median, Intervalle; prädiktive Verteilung neuer Noten
src/noten/evaluate.py   Diagnostik (R-hat, ESS, Divergenzen), Recovery, Kalibrierung, LOO-Hilfen
scripts/run_experiments.py   alle Experimente (voll oder --quick)
tests/                  pytest
reports/ERGEBNISSE.md   Ergebnisse; Tabellen in reports/tables/, Plots in reports/figures/
```

## Ausführung

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

```bash
ruff check . && ruff format --check .
```

Alle Experimente in Schnellkonfiguration (CI, wenige Minuten, Ausgabe nach `reports/quick/`, nicht versioniert):

```bash
python scripts/run_experiments.py all --quick --workers 2
```

Volle Konfiguration (30 Läufe je Kernszenario, ≈ 1,5 h auf 4 Kernen):

```bash
python scripts/run_experiments.py all --workers 4
```

Einzelne Schritte: `prior`, `recovery`, `summary`, `loo`, `sensitivity`, `grades`; Optionen
`--n-runs N`, `--loo-reps N`, `--scenarios a_rho0 b_rho05 …`, `--out DIR`. Die Recovery ist
fortsetzbar (bereits gerechnete Läufe in `results/recovery_runs.jsonl` werden übersprungen).

Minimalbeispiel in Python:

```python
from noten.simulate import SCENARIOS, simulate
from noten.models import ModelSpec, fit
from noten.evaluate import draws
from noten.mapping import full_cutpoints, summarize_grades

data = simulate(SCENARIOS["b_rho05"], seed=1)
_, idata = fit(data, ModelSpec(w=2 / 3), seed=1)
print(summarize_grades(draws(idata, "theta"), full_cutpoints(draws(idata, "cut_inner"))).head())
```
