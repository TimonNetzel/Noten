"""Erzeugt docs/notenmodell_mathematik.pdf (mathematische Beschreibung des Notenmodells).

Formeln: LaTeX-Syntax, gesetzt mit matplotlib-mathtext (Computer-Modern-Schrift),
Seitenlayout: ReportLab. Keine TeX-Installation nötig.

    pip install reportlab matplotlib numpy scipy
    python docs/build_modell_pdf.py
"""

from __future__ import annotations

import io
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import norm

OUT = pathlib.Path(__file__).with_name("notenmodell_mathematik.pdf")
FD = "/usr/share/fonts/truetype/liberation/"
for n, f in [("R", "Regular"), ("B", "Bold"), ("I", "Italic"), ("BI", "BoldItalic")]:
    pdfmetrics.registerFont(TTFont(f"Serif{n}", f"{FD}LiberationSerif-{f}.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
pdfmetrics.registerFontFamily(
    "Serif", normal="SerifR", bold="SerifB", italic="SerifI", boldItalic="SerifBI"
)

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "axes.unicode_minus": False})
ACC = colors.HexColor("#1f3a5f")
GREY = colors.HexColor("#555555")

# ----------------------------------------------------------------------------- Stile
body = ParagraphStyle("body", fontName="SerifR", fontSize=10.3, leading=13.6, alignment=TA_JUSTIFY,
                      spaceAfter=4.5)
small = ParagraphStyle("small", parent=body, fontSize=8.8, leading=11.2, textColor=GREY)
h1 = ParagraphStyle("h1", fontName="SerifB", fontSize=13, leading=16, textColor=ACC,
                    spaceBefore=9, spaceAfter=4, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="SerifB", fontSize=10.8, leading=13.5, spaceBefore=4,
                    spaceAfter=2, keepWithNext=1)
title = ParagraphStyle("title", fontName="SerifB", fontSize=19, leading=23, textColor=ACC,
                       spaceAfter=3)
sub = ParagraphStyle("sub", fontName="SerifI", fontSize=10.8, leading=14, textColor=GREY,
                     spaceAfter=8)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=2, spaceAfter=2.5)
cell = ParagraphStyle("cell", fontName="SerifR", fontSize=9.2, leading=11.4)
cellb = ParagraphStyle("cellb", parent=cell, fontName="SerifB")
cap = ParagraphStyle("cap", parent=small, spaceAfter=7, spaceBefore=2)

PAGE_W, _ = A4
CONTENT_W = PAGE_W - 4.4 * cm

_eq = [0]


_FALLBACK = "∈↦∝⊤⇒"


def fx(t):
    """Zeichen, die Liberation Serif nicht kennt, aus DejaVu Serif setzen."""
    return "".join(f"<font name='DejaVu'>{ch}</font>" if ch in _FALLBACK else ch for ch in t)


def P(t, s=body):
    return Paragraph(fx(t), s)


def B(t):
    return Paragraph(fx(t), bullet, bulletText="•")


def tex_img(tex: str, size: float = 12.0, dpi: int = 420) -> Image:
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${tex}$", fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03,
                transparent=True)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as PI

    w, h = PI.open(buf).size
    buf.seek(0)
    k = 72.0 / dpi
    return Image(buf, width=w * k, height=h * k)


def eq(*texs: str, numbered: bool = True):
    """Display-Gleichung (mehrere Zeilen = mehrere Bilder, Nummer in der ersten)."""
    if numbered:
        _eq[0] += 1
    rows = []
    for j, t in enumerate(texs):
        img = tex_img(t)
        rows.append([img, f"({_eq[0]})" if (j == 0 and numbered) else ""])
    tb = Table(rows, colWidths=[CONTENT_W - 1.3 * cm, 1.3 * cm])
    tb.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, -1), "CENTER"), ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("FONTNAME", (1, 0), (1, -1), "SerifR"),
        ("FONTSIZE", (1, 0), (1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return tb


def table(data, widths, header=True, zebra=True, align_first_left=True):
    rows = [[Paragraph(fx(str(c)), cellb if (header and i == 0) else cell) for c in r]
            for i, r in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cf")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf4")))
    t.setStyle(TableStyle(st))
    return t


# ----------------------------------------------------------------------------- Rechnungen
def kappa(Y, s_drift, s_v):
    yy = np.arange(1, Y + 1)
    K = s_drift**2 * np.minimum.outer(yy, yy) + s_v**2 * np.eye(Y)
    o = np.ones(Y)
    return float(o @ np.linalg.solve(K, o))


def sd_theta(ns, nm, Y, *, tau=(0.8, 0.7), rho=0.5, sig=(0.7, 0.9), sd=0.3, sv=0.3, w=2 / 3):
    """Gauß-Näherung: Posterior-SD von theta bei bekannten Populationsparametern."""
    S = np.array([[tau[0] ** 2, rho * tau[0] * tau[1]], [rho * tau[0] * tau[1], tau[1] ** 2]])
    k = kappa(Y, sd, np.sqrt(sv**2 + 1 / 12)) if Y > 0 else 0.0
    Lam = np.linalg.inv(S) + np.diag([ns / (sig[0] ** 2 + 1 / 12) + k,
                                      nm / (sig[1] ** 2 + 1 / 12) + k])
    v = np.array([w, 1 - w])
    return float(np.sqrt(v @ np.linalg.inv(Lam) @ v))


Z80 = norm.ppf(0.9)


def fig_probit() -> Image:
    mu, sg = 3.1, 0.7
    cut = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    x = np.linspace(0.3, 6.2, 600)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    edges = np.concatenate([[x[0]], cut, [x[-1]]])
    pal = ["#c9d8ea", "#9fb8d6", "#6f94c0", "#9fb8d6", "#c9d8ea", "#e4ebf4"]
    pal = ["#dfe8f3", "#b9cde5", "#8fb0d3", "#b9cde5", "#dfe8f3", "#f0f4f9"]
    cdf = norm.cdf((cut - mu) / sg)
    probs = np.diff(np.concatenate([[0], cdf, [1]]))
    for k in range(6):
        m = (x >= edges[k]) & (x <= edges[k + 1])
        ax.fill_between(x[m], norm.pdf(x[m], mu, sg), color=pal[k], lw=0)
        xm = 0.5 * (edges[k] + edges[k + 1])
        ax.text(xm, -0.085, f"$k={k + 1}$", ha="center", va="top", fontsize=9)
        ax.text(xm, 0.03, f"{probs[k]:.2f}".replace(".", ","), ha="center", fontsize=9)
    ax.plot(x, norm.pdf(x, mu, sg), color="#1f3a5f", lw=1.4)
    for j, c in enumerate(cut):
        ax.axvline(c, color="#1f3a5f", lw=0.8, ls=":")
        ax.text(c, 0.60, f"$c_{j + 1}$", ha="center", fontsize=9, color="#1f3a5f",
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.axvline(mu, color="#a33", lw=1.0)
    ax.text(mu + 0.05, 0.575, r"$\mu_{s,i}$", color="#a33", fontsize=9, va="top")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 0.66)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.5, -0.27, r"latente Achse (Notenpunkte), Beispiel $\mu_{s,i}=3{,}1,\ \sigma_s=0{,}7$",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=13.4 * cm, height=13.4 * cm * 2.7 / 7.2 * 0.97)


def fig_info() -> Image:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.7))
    Ys = np.arange(1, 9)
    for sd_, sv_, c, lab in [(0.3, 0.3, "#1f3a5f", r"$\sigma_{drift}=0{,}3$"),
                             (1.2, 0.3, "#b5532a", r"$\sigma_{drift}=1{,}2$")]:
        a.plot(Ys, [kappa(y, sd_, sv_) for y in Ys], "o-", color=c, ms=3.5, lw=1.2, label=lab)
        a.axhline(1 / sd_**2, color=c, ls="--", lw=0.8)
    a.set_yscale("log")
    a.set_xlabel(r"Anzahl Vornotenjahre $Y$", fontsize=9)
    a.set_ylabel(r"Information $\kappa=\mathbf{1}^{\top}K^{-1}\mathbf{1}$", fontsize=9)
    a.text(8, 1 / 0.09 * 1.08, r"Schranke $1/\sigma_{drift}^2$", ha="right", fontsize=8,
           color="#1f3a5f")
    a.text(8, 1 / 1.44 * 1.12, r"Schranke $1/\sigma_{drift}^2$", ha="right", fontsize=8,
           color="#b5532a")
    a.set_ylim(0.4, 25)
    a.legend(fontsize=8, frameon=False, loc="center right")
    a.tick_params(labelsize=8)
    ns = np.arange(1, 11)
    for Y, sd_, c, ls, lab in [(0, 0.3, "#555", "-", "ohne Vornoten"),
                               (1, 0.3, "#6f94c0", "-", r"$Y=1$"),
                               (3, 0.3, "#1f3a5f", "-", r"$Y=3$"),
                               (3, 1.2, "#b5532a", "--", r"$Y=3,\ \sigma_{drift}=1{,}2$")]:
        b.plot(ns, [sd_theta(n, n, Y, sd=sd_) for n in ns], ls, color=c, lw=1.3, label=lab)
    b.set_xlabel(r"Einzelnoten je Kanal ($n_s=n_m=n$)", fontsize=9)
    b.set_ylabel(r"$\mathrm{SD}(\theta_i\mid\mathcal{D})$", fontsize=9)
    b.legend(fontsize=8, frameon=False)
    b.tick_params(labelsize=8)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=CONTENT_W, height=CONTENT_W * 2.7 / 7.4 * 0.98)


# ----------------------------------------------------------------------------- Inhalt
def fig_example() -> Image:
    th, sd = 2.71, 0.22
    cut = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    probs = np.diff(np.concatenate([[0], norm.cdf((cut - th) / sd), [1]]))
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    ax.bar(range(1, 7), probs, color=["#b9cde5" if p < 0.5 else "#1f3a5f" for p in probs], width=0.6)
    for k, pr in enumerate(probs, 1):
        ax.text(k, pr + 0.025, f"{100 * pr:.0f} %", ha="center", fontsize=8.5)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Note", fontsize=9)
    ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=8.6 * cm, height=8.6 * cm * 1.9 / 5.2 * 0.95)


def summary_pages():
    s = []
    s += [P("Zusammenfassung des Modells", h1),
          P("<b>N</b><sub>i</sub> ist die Zeugnisnote von Schüler:in <i>i</i>, <b>Θ</b> der Vektor aller Parameter und latenten Größen "
            "(Niveaus μ, u, b, Kanalmittel α, Streuungen, Schwellen c, Vornoten-Parameter). Daten sind die Klausuren <b>S</b>, "
            "die mündlichen Noten <b>M</b>, die Vornoten <b>V</b> und, optional, das Lehrerurteil <b>D</b>. "
            "Das Modell liefert die gemeinsame Posterior-Verteilung von Note und Parametern:"),
          eq(r"P(N,\Theta\mid S,M,V,D)\ \propto\ P(N\mid\Theta)\cdot P(S\mid\Theta)\cdot P(M\mid\Theta)\cdot P(V\mid\Theta)"
             r"\cdot P(D\mid\Theta)\cdot P(\Theta)", numbered=False),
          P("Die Faktoren im Einzelnen (Schüler:in <i>i</i> in Kurs t(i), Kanal c ∈ {s, m}, Φ = Standardnormal-Verteilungsfunktion, "
            "c<sub>0</sub> = −∞, c<sub>6</sub> = +∞):", body)]
    rows = [
        ("Zeugnisnote", r"P(N_i=k\mid\Theta)=\mathbf{1}\!\left[c_{k-1}<\theta_i\leq c_k\right],\qquad \theta_i=w\,\mu_{s,i}+(1-w)\,\mu_{m,i}"),
        ("Klausuren", r"P(S\mid\Theta)=\prod_{i,j}\left[\Phi\!\left(\frac{c_{S_{ij}}-\mu_{s,i}}{\sigma_s}\right)-\Phi\!\left(\frac{c_{S_{ij}-1}-\mu_{s,i}}{\sigma_s}\right)\right]"),
        ("mündlich", r"P(M\mid\Theta)=\prod_{i,k}\left[\Phi\!\left(\frac{c_{M_{ik}}-\mu_{m,i}}{\sigma_m}\right)-\Phi\!\left(\frac{c_{M_{ik}-1}-\mu_{m,i}}{\sigma_m}\right)\right]"),
        ("Vornoten", r"P(V\mid\Theta)=\prod_{i}\prod_{c}\mathcal{N}\!\left(V_{c,i}\ ;\ (\alpha_c+\delta_v+u_{c,i})\mathbf{1},\ K\right),\quad K_{yy'}=\sigma_{drift}^{2}\min(y,y')+\sigma_v^{2}\mathbf{1}[y=y']"),
        ("Lehrerurteil", r"P(D\mid\Theta)=\prod_{i}\left[\Phi\!\left(\frac{c_{D_i}-\theta_i}{\sigma_D}\right)-\Phi\!\left(\frac{c_{D_i-1}-\theta_i}{\sigma_D}\right)\right]\quad(\mathrm{optional})"),
        ("Niveaus", r"\mu_{c,i}=\alpha_c+b_{t(i)}+u_{c,i}"),
        ("Hierarchie", r"P(\Theta)=\prod_{i}\mathcal{N}(u_i\mid\mathbf{0},\Sigma)\ \prod_{t}\mathcal{N}(b_t\mid0,\tau_b^{2})\ \cdot\ P(\alpha)\,P(\tau)\,P(\rho)\,P(\tau_b)\,P(\sigma)\,P(\sigma_D)\,P(c)\,P(\delta_v)\,P(\sigma_{drift})\,P(\sigma_v)"),
        ("Kovarianz", r"\Sigma=\mathrm{diag}(\tau)\,R(\rho)\,\mathrm{diag}(\tau),\qquad \Sigma_{sm}=\rho\,\tau_s\tau_m"),
    ]
    for lab, tx in rows:
        img_ = tex_img(tx, 11.0)
        avail = CONTENT_W - 2.3 * cm - 0.3 * cm
        if img_.drawWidth > avail:
            k_ = avail / img_.drawWidth
            img_.drawWidth *= k_
            img_.drawHeight *= k_
        t_ = Table([[Paragraph(f"<b>{lab}</b>", cell), img_]], colWidths=[2.3 * cm, CONTENT_W - 2.3 * cm])
        t_.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5dce6"))]))
        s.append(t_)
    s += [Spacer(1, 4),
          P("Hyperpriors: α<sub>c</sub> ~ N(3,5; 1²); τ<sub>c</sub>, σ<sub>c</sub>, σ<sub>D</sub> ~ Half-Normal(1); τ<sub>b</sub>, σ<sub>drift</sub>, σ<sub>v</sub> ~ Half-Normal(0,5); "
            "δ<sub>v</sub> ~ N(0; 0,5²); R ~ LKJ(2); innere Schwellen c<sub>2</sub>, c<sub>3</sub>, c<sub>4</sub> über Dirichlet(4,4,4,4)-Abstände "
            "bei festen c<sub>1</sub> = 1,5 und c<sub>5</sub> = 5,5. In dieser Zusammenfassung sind u<sub>i</sub> und b<sub>t</sub> "
            "direkt als Normalverteilungen geschrieben; die Implementierung nutzt eine äquivalente nicht-zentrierte Form "
            "(Details in §3).", small),
          P("Gesucht ist die Randverteilung der Note: P(N<sub>i</sub> = k | S, M, V, D) = ∫ P(N<sub>i</sub> = k | Θ) P(Θ | S, M, V, D) dΘ, "
            "geschätzt als Anteil der Posterior-Ziehungen mit c<sub>k−1</sub> < θ<sub>i</sub> ≤ c<sub>k</sub>.", body),
          PageBreak()]

    # ---------------- Laienseite
    th, sd = 2.71, 0.22
    s += [P("Das Modell in einfachen Worten", h1),
          P("<b>Grundidee.</b> Eine einzelne Note ist eine Momentaufnahme: Tagesform, Aufgabenauswahl und Rundung "
            "verfälschen sie ein Stück. Hinter allen Noten steht aber ein „wahres“ Leistungsniveau. Das Modell "
            "sucht dasjenige Niveau, das am besten zu <i>allen</i> vorhandenen Noten passt, und sagt ehrlich dazu, "
            "wie sicher diese Aussage ist. Heraus kommt keine einzelne Zahl, sondern eine Wahrscheinlichkeit für jede Note."),
          P("Welche Informationen einfließen", h2)]
    zut = [["Information", "Wie sie wirkt"],
           ["Klausuren", "Hauptsäule: Gewicht 2/3 an der Zeugnisnote."],
           ["Mündliche Noten", "Gewicht 1/3. Wer schriftlich stark ist, ist meist auch mündlich stark; das nutzt das Modell."],
           ["Vornoten", "Frühere Schuljahre zählen mit, aber ältere weniger. Sie helfen, wenn aktuell wenig Noten da sind."],
           ["Kurs / Lehrkraft", "Manche Kurse benoten strenger oder milder. Das Modell rechnet diesen Unterschied heraus, bleibt aber in der Endnote sichtbar."],
           ["Lehrerurteil (optional)", "Eine zusätzliche Einschätzung als weiterer Hinweis."]]
    s += [table(zut, [4.0 * cm, CONTENT_W - 4.0 * cm]), Spacer(1, 4),
          P("Vom Datensatz zur Zeugnisnote in vier Schritten", h2),
          B("<b>1. Niveau schätzen.</b> Aus Klausuren und mündlichen Noten wird je Schüler:in ein Leistungsniveau je Kanal geschätzt, "
            "ein Wert wie 2,7 auf der Notenskala."),
          B("<b>2. Vorsichtig sein.</b> Gibt es nur wenige Noten, wird das Ergebnis zum Durchschnitt des Kurses hin „gezogen“. "
            "Je mehr Noten vorliegen, desto mehr zählt die Person selbst."),
          B("<b>3. Zusammenfassen.</b> Schriftliches und mündliches Niveau werden gewichtet (2/3 zu 1/3) zu einem Gesamtniveau verrechnet."),
          B("<b>4. In Noten übersetzen.</b> Das Gesamtniveau ist nicht exakt bekannt. Das Modell spielt tausende plausible Werte durch "
            "und zählt, wie oft sich welche Note ergibt."),
          P("Ein Beispiel (erfunden)", h2),
          P("Angenommen, das Gesamtniveau einer Schülerin liegt bei etwa 2,7 mit einer Unsicherheit von rund ±0,2. "
            "Dann ergibt sich:")]
    ex = Table([[fig_example(),
                 Paragraph("<b>So liest man das:</b><br/>• Wahrscheinlichste Note: <b>3</b> (Median).<br/>"
                           "• Realistische Spanne: <b>2 bis 3</b>. Notiert als „3 [2–3]“.<br/>"
                           "• Eine 2 ist möglich (hier 17 %); eine 4 oder schlechter ist nach den Daten praktisch ausgeschlossen.", cell)]],
               colWidths=[9.0 * cm, CONTENT_W - 9.0 * cm])
    ex.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    s += [ex, Spacer(1, 3),
          P("Was man daraus lernen kann", h2),
          B("Wenn die Spanne „2–3“ lautet, sind die Daten nicht genauer. Das ist eine ehrliche Aussage und kein Mangel des Verfahrens: "
            "Noten sind ganze Zahlen, das Niveau liegt aber oft zwischen zwei Stufen."),
          B("Mit mehr Noten und mit Vornoten wird die Spanne enger. Im Simulationstest bekommt etwa jede:r Zweite mit vollen Daten eine eindeutige Note, "
            "ohne Vornoten nur etwa jede:r Vierte."),
          B("Das Modell <b>ersetzt kein pädagogisches Urteil</b>. Es zeigt, wie belastbar eine Note ist, und macht Grenzfälle sichtbar."),
          B("Alle bisherigen Tests nutzen ausschließlich <b>simulierte</b> Daten, keine echten Schülerdaten."),
          PageBreak()]
    return s


def story():
    s = []
    s += [P("Das hierarchische Bayes-Modell zur Zeugnisnote", title),
          P("Mathematische Beschreibung und methodischer Hintergrund des Algorithmus", sub),
          P("Grundlage: Branch <font name='SerifI'>feature/bayes-notenmodell</font> "
            "(<font name='SerifI'>src/noten/models.py, simulate.py, mapping.py</font>). "
            "Alle Daten sind simuliert; Zahlen aus Simulationsstudien stammen aus "
            "<font name='SerifI'>reports/ERGEBNISSE.md</font>.", small)]

    s += summary_pages()

    # 1 ------------------------------------------------------------------------------------
    s += [P("1  Zielsetzung und Aufbau", h1),
          P("Eine Zeugnisnote ist eine ganze Zahl, die Information dahinter aber nicht: "
            "Klausuren, mündliche Noten und Vornoten sind verrauschte, diskret gerundete "
            "Messungen eines zugrunde liegenden Leistungsniveaus. Das Modell behandelt dieses "
            "Niveau als <i>latente stetige Größe</i> θ<sub>i</sub> je Schüler:in <i>i</i> und "
            "liefert statt einer Zahl die Verteilung P(Note = k | Daten), k = 1,…,6. "
            "Die Berechnung folgt drei Stufen:"),
          B("<b>Messmodell</b> (§2): Einzelnoten entstehen aus einem latenten Niveau plus "
            "Rauschen durch Rundung an geordneten Schwellen (kumulatives Probit)."),
          B("<b>Latentes Strukturmodell</b> (§3, §4): Niveaus sind hierarchisch gegliedert "
            "(Population → Kurs/Lehrkraft → Schüler:in, getrennt für schriftlich und mündlich); "
            "Vornoten gehen als Random Walk in der Zeit ein."),
          B("<b>Aggregation</b> (§5): Gewichtetes Mittel θ<sub>i</sub> der beiden Kanäle, "
            "abgebildet über die Schwellen auf eine Notenverteilung.")]

    s += [P("Notation", h2)]
    sym = [
        ["Symbol", "Bedeutung"],
        ["i = 1,…,n;  t(i) ∈ {1,…,T}", "Schüler:in; Kurs bzw. Lehrkraft von i (Basisdesign: T = 8, n = 200)"],
        ["c ∈ {s, m}", "Kanal: schriftlich (<i>s</i>), mündlich/sonstige Mitarbeit (<i>m</i>)"],
        ["S<sub>ij</sub>, M<sub>ik</sub> ∈ {1,…,6}", "Klausurnoten (j = 1,…,n<sub>s</sub>), mündliche Noten (k = 1,…,n<sub>m</sub>)"],
        ["V<sub>c,i,y</sub>", "Vornote Kanal c, Schuljahr y (y = 1 letztes, y = 2 vorletztes, …, Y)"],
        ["c<sub>1</sub> < … < c<sub>5</sub>", "Schwellen der latenten Achse (c<sub>0</sub> = −∞, c<sub>6</sub> = +∞)"],
        ["μ<sub>c,i</sub>", "aktuelles latentes Niveau von i im Kanal c (Notenpunkte, größer = schlechter)"],
        ["α<sub>c</sub>, b<sub>t</sub>, u<sub>c,i</sub>", "Populationsmittel, Kurseffekt, individuelle Abweichung"],
        ["σ<sub>c</sub>; τ<sub>c</sub>; τ<sub>b</sub>; ρ", "Streuung der Einzelnote; der Schülerniveaus; der Kurseffekte; Korrelation s–m"],
        ["σ<sub>drift</sub>, σ<sub>v</sub>, δ<sub>v</sub>", "Jahresdrift, Messfehler und Niveauverschiebung der Vornoten"],
        ["θ<sub>i</sub>, w", "latente Zeugnisnote; Gewicht der Klausuren (w = 2/3)"],
    ]
    s += [table(sym, [5.0 * cm, CONTENT_W - 5.0 * cm]), Spacer(1, 4)]

    # 3 ------------------------------------------------------------------------------------
    s += [P("2  Messmodell: kumulatives Probit", h1),
          P("Jede Einzelnote entsteht aus einer latenten stetigen Leistung, die an den Schwellen "
            "c<sub>k</sub> in Noten zerlegt wird. Für eine Klausur j von Schüler:in i:")]
    s += [eq(r"S^{*}_{ij}=\mu_{s,i}+\varepsilon_{ij},\quad \varepsilon_{ij}\sim\mathcal{N}(0,\sigma_s^{2}),"
             r"\qquad S_{ij}=k\Longleftrightarrow c_{k-1}<S^{*}_{ij}\leq c_k"),
          eq(r"P(S_{ij}=k\mid\mu_{s,i})=\Phi\!\left(\frac{c_k-\mu_{s,i}}{\sigma_s}\right)"
             r"-\Phi\!\left(\frac{c_{k-1}-\mu_{s,i}}{\sigma_s}\right)")]
    s += [P("Φ ist die Verteilungsfunktion der Standardnormalverteilung. Für mündliche Noten M<sub>ik</sub> "
            "gilt dasselbe mit (μ<sub>m,i</sub>, σ<sub>m</sub>); beide Kanäle <b>teilen dieselben Schwellen</b>. "
            "Bei bedingter Unabhängigkeit gegeben μ ist die Likelihood ein Produkt über alle Beobachtungen:"),
          eq(r"\mathcal{L}_S=\prod_{i=1}^{n}\prod_{j=1}^{n_s}\left[\Phi\!\left(\frac{c_{S_{ij}}-\mu_{s,i}}"
             r"{\sigma_s}\right)-\Phi\!\left(\frac{c_{S_{ij}-1}-\mu_{s,i}}{\sigma_s}\right)\right]")]
    s += [fig_probit(),
          P("<b>Abb. 1.</b> Notenwahrscheinlichkeiten eines Schülerniveaus. Die Fläche zwischen "
            "zwei Schwellen ist die Wahrscheinlichkeit der jeweiligen Note.", cap)]
    s += [P("Schwellen und Identifizierbarkeit", h2),
          P("Eine Transformation x ↦ a + b·x der latenten Achse (b > 0) auf alle μ, c<sub>k</sub> und σ lässt "
            "alle Notenwahrscheinlichkeiten unverändert. Lage und Skala sind damit nicht identifiziert:"),
          eq(r"\Phi\!\left(\frac{(a+bc_k)-(a+b\mu)}{b\sigma}\right)=\Phi\!\left(\frac{c_k-\mu}{\sigma}\right)"),
          P("Das Modell fixiert diese zwei Freiheitsgrade durch <b>feste äußere Schwellen</b> c<sub>1</sub> = 1,5 und "
            "c<sub>5</sub> = 5,5. Die inneren Schwellen sind frei, aber geordnet, über Abstände "
            "g ~ Dirichlet(4,4,4,4):"),
          eq(r"c_{1+j}=\frac{3}{2}+4\sum_{l=1}^{j}g_l,\quad j=1,2,3;\qquad "
             r"\mathbf{g}=(g_1,\dots,g_4)\sim\mathrm{Dirichlet}(4,4,4,4)"),
          P("Dann liegt die Achse in Notenpunkten vor, α<sub>c</sub>, σ<sub>c</sub> und τ<sub>c</sub> sind "
            "interpretierbar, und die Residualstreuungen σ<sub>s</sub>, σ<sub>m</sub> sind getrennt identifiziert, weil beide "
            "Kanäle dieselben Schwellen verwenden. Der Prior zentriert die inneren Schwellen auf 2,5 / 3,5 / 4,5 "
            "(Standardabweichungen ≈ 0,42 / 0,49 / 0,42); Lehrkräfte dürfen also z. B. großzügiger bei 2 und 3 "
            "als bei 5 sein.")]

    # 4 ------------------------------------------------------------------------------------
    s += [P("3  Hierarchische Struktur der latenten Niveaus", h1),
          P("Das Niveau setzt sich additiv aus Population, Kurs und individueller Abweichung zusammen:"),
          eq(r"\mu_{c,i}=\alpha_c+b_{t(i)}+u_{c,i},\qquad c\in\{s,m\}"),
          P("<b>Individuelle Abweichungen, korreliert über die Kanäle.</b> Wer schriftlich stark ist, ist "
            "meist auch mündlich stark; das bildet der Vektor u<sub>i</sub> = (u<sub>s,i</sub>, u<sub>m,i</sub>) ab:"),
          eq(r"u_i\sim\mathcal{N}(\mathbf{0},\Sigma),\quad \Sigma=\mathrm{diag}(\tau)\,R\,\mathrm{diag}(\tau),\quad "
             r"\Sigma_{ss}=\tau_s^2,\ \Sigma_{mm}=\tau_m^2,\ \Sigma_{sm}=\rho\,\tau_s\tau_m"),
          P("Die Korrelation ρ = R<sub>12</sub> erhält einen LKJ(η = 2)-Prior; im 2×2-Fall ist das die Dichte "
            "p(ρ) ∝ (1 − ρ²), also ρ ~ 2·Beta(2,2) − 1 mit Standardabweichung ≈ 0,45. "
            "Gesampelt wird <i>nicht-zentriert</i>: mit z<sub>1</sub>, z<sub>2</sub> ~ iid N(0,1) gilt"),
          eq(r"u_{s,i}=\tau_s z_{1,i},\qquad u_{m,i}=\tau_m\left(\rho\, z_{1,i}+\sqrt{1-\rho^{2}}\;z_{2,i}\right)"),
          P("Das beseitigt die „Trichter“-Geometrie hierarchischer Posteriors und macht NUTS effizient."),
          P("<b>Lehrer-/Kurseffekt.</b> Ein Skalar je Kurs, additiv in beiden Kanälen. Zur sauberen Trennung vom "
            "Kanalmittel gilt die Nebenbedingung Σ<sub>t</sub> b<sub>t</sub> = 0:"),
          eq(r"b_t=\tau_b\sqrt{\frac{T}{T-1}}\;z_t,\quad z\sim\mathrm{ZeroSumNormal}(1)\;\Rightarrow\;"
             r"\mathrm{Var}(b_t)=\tau_b^2,\ \ \mathrm{Cov}(b_t,b_{t'})=-\frac{\tau_b^2}{T-1}"),
          P("Ohne Zero-Sum waren α<sub>c</sub> und mean(b) nur schwach getrennt (Kamm im Posterior, R-hat ≈ 1,03). "
            "Marginal folgt für Schüler:innen im selben Kurs "
            "Cov(μ<sub>s,i</sub>, μ<sub>m,i′</sub>) = ρ τ<sub>s</sub>τ<sub>m</sub>·1[i = i′] + τ<sub>b</sub>²: "
            "Der Kurseffekt erzeugt eine zusätzliche, über Schüler hinweg gemeinsame Korrelation.")]

    # 5 ------------------------------------------------------------------------------------
    s += [P("4  Vornoten als Random Walk", h1),
          P("Vornoten gehören zu früheren Lehrkräften und Kursen, daher geht b<sub>t</sub> hier <i>nicht</i> ein. "
            "Das Niveau driftet rückwärts in der Zeit als Random Walk, gemessen mit Fehler und einem "
            "Offset δ<sub>v</sub> (Notenniveau früherer Jahre vs. aktueller Kurs):"),
          eq(r"\lambda_{c,i,0}=u_{c,i},\qquad \lambda_{c,i,y}=\lambda_{c,i,y-1}+\eta_y,\quad \eta_y\sim\mathcal{N}(0,\sigma_{drift}^2)",
             r"V_{c,i,y}=\alpha_c+\delta_v+\lambda_{c,i,y}+e,\quad e\sim\mathcal{N}(0,\sigma_v^2)"),
          P("Die latente Bahn λ wird analytisch marginalisiert. Der Vektor der Vornoten V<sub>c,i</sub> = "
            "(V<sub>c,i,1</sub>,…,V<sub>c,i,Y</sub>) ist multivariat normal mit gemeinsamem Mittel und Wiener-Kovarianz:"),
          eq(r"V_{c,i}\sim\mathcal{N}\!\left((\alpha_c+\delta_v+u_{c,i})\,\mathbf{1},\ K\right),\qquad "
             r"K_{yy'}=\sigma_{drift}^{2}\min(y,y')+\sigma_v^{2}\,\mathbf{1}[y=y']"),
          P("Ältere Noten haben die Varianz y·σ<sub>drift</sub>² + σ<sub>v</sub>² und werden automatisch "
            "weniger gewichtet; wie stark, lernt das Modell aus den Daten. Als Information über u<sub>c,i</sub> "
            "wirken die Vornoten wie <i>eine</i> einzige Gauß-Messung mit Präzision κ (verallgemeinerter "
            "Kleinste-Quadrate-Mittelwert V̄ = 1<sup>⊤</sup>K<sup>−1</sup>V / κ):"),
          eq(r"\bar V_{c,i}\mid u_{c,i}\sim\mathcal{N}\!\left(\alpha_c+\delta_v+u_{c,i},\ \kappa^{-1}\right),\qquad "
             r"\kappa=\mathbf{1}^{\top}K^{-1}\mathbf{1}\ \leq\ \frac{1}{\sigma_{drift}^{2}}"),
          P("<b>Schranke.</b> Mit M<sub>yy′</sub> = min(y,y′) ist K − σ<sub>drift</sub>² M = σ<sub>v</sub>² I positiv semidefinit, also K<sup>−1</sup> höchstens σ<sub>drift</sub><sup>−2</sup> M<sup>−1</sup> (Loewner-Ordnung). "
            "Aus M e<sub>1</sub> = 1 folgt M<sup>−1</sup>1 = e<sub>1</sub> und damit 1<sup>⊤</sup>M<sup>−1</sup>1 = 1, also κ ≤ σ<sub>drift</sub><sup>−2</sup>. "
            "<i>Mehr Vornotenjahre können u<sub>c,i</sub> nie genauer bestimmen als die Drift eines einzelnen Jahres zulässt</i>; "
            "für Y = 1 ist κ = 1/(σ<sub>drift</sub>² + σ<sub>v</sub>²). Bei σ<sub>drift</sub> = σ<sub>v</sub> = 0,3 liegt κ bei "
            "5,6 / 6,7 / 6,8 für Y = 1 / 2 / 3 (Schranke 11,1); bei starker Drift (σ<sub>drift</sub> = 1,2) bei nur 0,65.")]

    # 6 ------------------------------------------------------------------------------------
    s += [P("5  Zeugnisnote als Verteilung", h1),
          P("Aus den Niveaus beider Kanäle entsteht die latente Zeugnisnote als gewichtetes Mittel, "
            "und über die geschätzten Schwellen die diskrete Notenverteilung:"),
          eq(r"\theta_i=w\,\mu_{s,i}+(1-w)\,\mu_{m,i},\qquad w=\frac{2}{3}"),
          eq(r"P(G_i=k\mid\mathcal{D})=P\!\left(c_{k-1}<\theta_i\leq c_k\mid\mathcal{D}\right)"
             r"\approx\frac{1}{D}\sum_{d=1}^{D}\mathbf{1}\!\left[c^{(d)}_{k-1}<\theta_i^{(d)}\leq c^{(d)}_k\right]"),
          P("(θ<sub>i</sub><sup>(d)</sup>, c<sup>(d)</sup>) sind die D Posterior-Ziehungen (4 Ketten × 1500). "
            "Berichtet werden Median und zentrales 80-%-Intervall der diskreten Verteilung "
            "(q-Quantil: kleinstes k mit P(G ≤ k) ≥ q) sowie θ mit Quantilintervall:"),
          eq(r"q_p(i)=\min\{k:\ P(G_i\leq k\mid D)\geq p\}",
             r"\mathrm{Median}_i=q_{0{,}5}(i),\qquad [\,q_{0{,}1}(i),\,q_{0{,}9}(i)\,]\ \ (80\,\%)"),
          P("Die posterior-prädiktive Verteilung einer <i>neuen</i> Einzelnote (z. B. nächste Klausur) enthält "
            "zusätzlich das Messrauschen σ<sub>s</sub> und wird Rao-Blackwell-geglättet berechnet:"),
          eq(r"P(S^{new}_{i}=k\mid\mathcal{D})=\frac{1}{D}\sum_{d=1}^{D}\left[\Phi\!\left(\frac{c^{(d)}_k-\mu^{(d)}_{s,i}}"
             r"{\sigma_s^{(d)}}\right)-\Phi\!\left(\frac{c^{(d)}_{k-1}-\mu^{(d)}_{s,i}}{\sigma_s^{(d)}}\right)\right]")]

    # 7 ------------------------------------------------------------------------------------
    s += [P("6  Gesamtmodell und Schätzung", h1),
          P("Mit Parametervektor Θ = (α, τ, ρ, τ<sub>b</sub>, σ<sub>s</sub>, σ<sub>m</sub>, σ<sub>drift</sub>, σ<sub>v</sub>, δ<sub>v</sub>, g) "
            "und latenten Größen (z, b) ist das Posterior bis auf eine Konstante"),
          eq(r"p(\Theta,z,b\mid\mathcal{D})\ \propto\ \mathcal{L}_S\,\mathcal{L}_M\,\mathcal{L}_V\;"
             r"p(z)\,p(b\mid\tau_b)\,p(\Theta)"),
          P("mit der Probit-Likelihood (1)–(3) für S und M sowie der multivariaten Normalverteilung (11) für die Vornoten "
            "V<sub>s</sub> und V<sub>m</sub> (beide mit gleichem K). Das Posterior wird per NUTS (Hamiltonian Monte Carlo, "
            "4 Ketten, 1000 Warm-up, 1500 Ziehungen, Ziel-Akzeptanz 0,9) approximiert; die Likelihood nutzt je Beobachtung nur "
            "die beiden angrenzenden Schwellen. Konvergenzkriterien: R-hat < 1,01, ESS > 400, keine Divergenzen."),
          P("Priors (schwach informativ)", h2)]
    pri = [["Parameter", "Prior", "Parameter", "Prior"],
           ["α<sub>c</sub>", "N(3,5; 1²)", "σ<sub>drift</sub>, σ<sub>v</sub>", "Half-Normal(0,5)"],
           ["τ<sub>s</sub>, τ<sub>m</sub>", "Half-Normal(1)", "δ<sub>v</sub>", "N(0; 0,5²)"],
           ["σ<sub>s</sub>, σ<sub>m</sub>", "Half-Normal(1)", "ρ (über R)", "LKJ(η = 2)"],
           ["τ<sub>b</sub>", "Half-Normal(0,5)", "innere Schwellenabstände g", "Dirichlet(4,4,4,4)"]]
    s += [table(pri, [3.3 * cm, 3.9 * cm, 5.3 * cm, CONTENT_W - 12.5 * cm]), Spacer(1, 3),
          P("Optional kann ein Lehrerurteil D<sub>i</sub> als dritter ordinaler Kanal D<sub>i</sub> ~ OrdProbit(θ<sub>i</sub>, σ<sub>D</sub>) "
            "mit denselben Schwellen eingehen (Standard: aus).", small)]

    # 8 ------------------------------------------------------------------------------------
    s += [P("7  Analytische Näherung: Schrumpfung und Unsicherheit", h1),
          P("Bedingt auf die Populationsparameter und bei Näherung der Rundung als zusätzliche Varianz "
            "σ̃<sub>c</sub>² = σ<sub>c</sub>² + 1/12 (Sheppard) ist das Posterior von u<sub>i</sub> Gauß-verteilt. "
            "Mit κ aus (12) und n<sub>c</sub> Einzelnoten im Kanal c gilt für die Präzisionsmatrix von u<sub>i</sub> und die Unsicherheit der Note:"),
          eq(r"\Lambda_i=\Sigma^{-1}+\mathrm{diag}\!\left(\frac{n_s}{\tilde\sigma_s^{2}}+\kappa,\ \frac{n_m}{\tilde\sigma_m^{2}}+\kappa\right)",
             r"\mathrm{Var}(\theta_i\mid D)=\mathbf{w}^{\top}\Lambda_i^{-1}\mathbf{w},\qquad \mathbf{w}=(w,\,1-w)^{\top}"),
          P("Der Posterior-Mittelwert ist Λ<sub>i</sub><sup>−1</sup> mal die präzisionsgewichtete Summe der Abweichungen der "
            "Einzelnoten-Mittel S̄<sub>i</sub>, M̄<sub>i</sub> und der Vornoten V̄<sub>c,i</sub> von α<sub>c</sub> + b<sub>t(i)</sub> bzw. α<sub>c</sub> + δ<sub>v</sub>: "
            "Er schrumpft je Kanal zum Kursmittel, und die Kopplung über ρ (Σ<sup>−1</sup>) leiht Information vom besser "
            "gemessenen Kanal an den schwächer gemessenen. Die Näherung reproduziert die Simulationsstudie gut:")]
    rows = [["Szenario", "n<sub>s</sub> + n<sub>m</sub>, Vornoten", "σ<sub>drift</sub>", "SD(θ|D)  Näherung", "80-%-Breite  Näherung", "80-%-Breite  Simulation"]]
    for lab, ns, nm, Y, sd_, sim in [
        ("Basis", 3, 5, 3, 0.3, 0.566), ("wenig Daten", 2, 2, 3, 0.3, 0.642),
        ("viele Daten", 6, 10, 3, 0.3, 0.468), ("starke Drift", 3, 5, 3, 1.2, 0.714),
        ("Basis ohne Vornoten", 3, 5, 0, 0.3, 0.74),
    ]:
        v = sd_theta(ns, nm, Y, sd=sd_)
        f = lambda x: f"{x:.2f}".replace(".", ",")  # noqa: E731
        rows.append([lab, f"{ns} + {nm}, Y = {Y}", f(sd_) if Y else "–", f(v), f(2 * Z80 * v), f(sim)])
    s += [table(rows, [3.4 * cm, 3.4 * cm, 1.7 * cm, 2.9 * cm, 2.9 * cm, CONTENT_W - 14.3 * cm]),
          Spacer(1, 3),
          P("Abweichungen entstehen durch die Näherung (feste Populationsparameter, Rundung als Zusatzvarianz, Normalverteilung der "
            "Diskretisierung). Die Simulationswerte sind Mittel über 30 Läufe je Szenario.", small),
          fig_info(),
          P("<b>Abb. 2.</b> Links: Informationsgehalt der Vornoten κ und obere Schranke 1/σ<sub>drift</sub>². "
            "Rechts: Unsicherheit der latenten Note in Gauß-Näherung bei Basisparametern.", cap)]

    # 9 ------------------------------------------------------------------------------------
    s += [P("8  Annahmen, Eigenschaften und Grenzen", h1),
          B("<b>Diskretheit.</b> Die Intervalle der latenten Note θ (Breite ≈ 0,57 Punkte) umfassen im Mittel "
            "1,6 Notenstufen; nur 44 % der Schüler:innen erhalten eine eindeutige Note (ohne Vornoten 24 %). Die "
            "Mehrdeutigkeit ist eine Eigenschaft der Daten, nicht des Verfahrens."),
          B("<b>Kalibrierung (Simulation).</b> Die Posterior-Abdeckung von μ<sub>s</sub>, μ<sub>m</sub>, θ beträgt 0,49 / 0,79 / 0,94 "
            "(Soll 0,50 / 0,80 / 0,95), die randomisierte PIT der Klausurvorhersage 0,79–0,81 (Soll 0,80). "
            "Bei ρ = 0,5 schließt das 95-%-Intervall von ρ die 0 in 100 % der Läufe aus."),
          B("<b>Fehlspezifikation Vornoten.</b> Vornoten werden stetig behandelt; Rundung (Varianz 1/12 → σ<sub>v,eff</sub>² = σ<sub>v</sub>² + 1/12) "
            "und Abschneiden bei 1 und 6 sind nicht modelliert. Bei starker Drift liegen 32 % der Vornoten am Rand; σ<sub>drift</sub> "
            "wird dann systematisch unterschätzt (Median 0,93 statt 1,2). Auf μ und θ schlägt das kaum durch (Abdeckung 0,79)."),
          B("<b>Strukturannahmen.</b> Gemeinsame σ<sub>drift</sub>, σ<sub>v</sub>, δ<sub>v</sub> für beide Kanäle; "
            "Residualstreuungen σ<sub>c</sub> über alle Schüler gepoolt (keine individuelle Heteroskedastie); "
            "Kurseffekt additiv und gleich in beiden Kanälen; w fest (normative Setzung, kein Schätzparameter)."),
          B("<b>Schwellenübertragung.</b> θ wird mit denselben Schwellen wie die Einzelnoten diskretisiert. Das setzt voraus, "
            "dass die Zeugnisnote die Skala der Einzelnoten erbt; das ist eine Modell-, keine Datenannahme."),
          B("<b>Kurseffekt in θ.</b> b<sub>t</sub> bleibt in θ<sub>i</sub> enthalten (Note <i>in diesem Kurs</i>); "
            "die lehrerbereinigte Note wäre θ<sub>i</sub> − b<sub>t(i)</sub>. Bei nur T = 8 Kursen ist τ<sub>b</sub> schwach bestimmt "
            "und trägt die meisten Sampler-Auffälligkeiten (81 % der Fits erfüllen alle Kriterien)."),
          B("<b>Identifizierbarkeit der Drift.</b> Mit nur einem Vornotenjahr sind σ<sub>drift</sub> und σ<sub>v</sub> nicht getrennt "
            "bestimmbar (nur σ<sub>drift</sub>² + σ<sub>v</sub>² geht ein, vgl. κ); getrennt identifiziert sind sie erst ab Y ≥ 2.")]
    return s


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("SerifI", 8.5)
    canvas.setFillColor(GREY)
    canvas.drawString(2.2 * cm, 1.2 * cm, "Hierarchisches Bayes-Modell zur Zeugnisnote – mathematische Beschreibung")
    canvas.drawRightString(PAGE_W - 2.2 * cm, 1.2 * cm, f"Seite {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#b8c2cf"))
    canvas.setLineWidth(0.4)
    canvas.line(2.2 * cm, 1.55 * cm, PAGE_W - 2.2 * cm, 1.55 * cm)
    canvas.restoreState()


def main() -> None:
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                          topMargin=2.0 * cm, bottomMargin=2.2 * cm,
                          title="Hierarchisches Bayes-Modell zur Zeugnisnote – Mathematische Beschreibung",
                          author="Noten-Projekt")
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=decorate)])
    doc.build(story())
    print("geschrieben:", OUT)


if __name__ == "__main__":
    main()
