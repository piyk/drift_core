"""make_figures.py  |  Figures of the revised manuscript.

Conventions: IEEE column widths (3.5 in single, 7.16 in double), the
Okabe-Ito colour-blind-safe palette with redundant marker and line-style
coding, 8 pt type at final size, vector PDF with embedded TrueType fonts.

Run:  python revise1/src/make_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import drift_core as dc

FIG_DIR = dc.ROOT / "revise1" / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
OI = {"DRIFT": "#D55E00", "IsolationForest": "#0072B2", "DeepAutoencoder": "#009E73",
      "Heuristic": "#666666", "ctrl": "#CC79A7"}
STYLE = {"DRIFT": dict(ls="-", marker="o"), "IsolationForest": dict(ls="--", marker="s"),
         "DeepAutoencoder": dict(ls="-.", marker="^"), "Heuristic": dict(ls=":", marker="D")}
LABEL = {"DRIFT": "DRIFT", "IsolationForest": "Isolation Forest",
         "DeepAutoencoder": "Autoencoder", "Heuristic": "Heuristic"}
DETS = ["Heuristic", "IsolationForest", "DeepAutoencoder", "DRIFT"]

d = json.load(open(dc.OUT_DIR / "experiments.json"))
grid = np.array(d["budget_grid"]) * 100.0


def fig_budget_curves():
    """Recall as a function of the review budget, overall and per typology."""
    panels = [("recall", "All four typologies"), ("peeling_chain", "Peeling chain"),
              ("mixer_passthrough", "Mixer pass-through"), ("rapid_cashout", "Rapid cash-out"),
              ("dormant_burst", "Dormant burst")]
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.75), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        for m in DETS:
            mu = np.array(d["E1_curves"][m][key]["mean"])
            sd = np.array(d["E1_curves"][m][key]["std"])
            ax.fill_between(grid, mu - sd, mu + sd, color=OI[m], alpha=0.15, lw=0)
            ax.plot(grid, mu, color=OI[m], markevery=8, label=LABEL[m], **STYLE[m])
        ax.plot([0, 20], [0, 0.20], color="k", lw=0.5, ls=(0, (1, 2)))
        ax.set_title(title, pad=3)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xlabel("Review budget (%)")
    axes[0].set_xticks([0, 5, 10, 15, 20])
    axes[0].set_ylabel("Recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, handlelength=2.6,
               bbox_to_anchor=(0.5, 1.13), columnspacing=2.0)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(FIG_DIR / "fig_budget_curves.pdf")
    plt.close(fig)


def fig_robustness():
    """(a) Noise-matched negatives, cited first in the text. (b) Contamination of the reference split."""
    fig, (b, a) = plt.subplots(1, 2, figsize=(3.5, 1.75))
    levels = ["0.005", "0.01", "0.02", "0.05"]
    xs = [0, 1, 2, 3, 4]
    for m in DETS:
        mu = [d["E1_main"][m]["pr_auc"]["mean"]] + [d["E8_contaminated_fit"][c][m]["pr_auc"]["mean"] for c in levels]
        sd = [d["E1_main"][m]["pr_auc"]["std"]] + [d["E8_contaminated_fit"][c][m]["pr_auc"]["std"] for c in levels]
        a.errorbar(xs, mu, yerr=sd, color=OI[m], capsize=1.5, elinewidth=0.6, label=LABEL[m], **STYLE[m])
    a.set_xlabel("Injected share of fit split (%)")
    a.set_ylabel("PR-AUC")
    a.set_xticks(xs)
    a.set_xticklabels(["0", "0.5", "1", "2", "5"])
    a.set_ylim(0, 0.46)
    a.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    a.set_title("(b)", loc="left", pad=2)
    short = {"Heuristic": "Heuristic", "IsolationForest": "IF", "DeepAutoencoder": "AE", "DRIFT": "DRIFT"}
    h, l = a.get_legend_handles_labels()
    a.legend([x[0] for x in h], [short[m] for m in DETS], frameon=False, loc="lower left",
             ncol=2, handlelength=2.2, borderaxespad=0.0, labelspacing=0.2, columnspacing=0.8,
             fontsize=6)

    from matplotlib.lines import Line2D
    names = ["tight", "default", "wide", "very_wide"]
    x = np.arange(4)
    ends = {}
    for m in ("DeepAutoencoder", "DRIFT"):
        for ctrl, ls, fill in (("ctrl_G1_noise", "-", True), ("ctrl_consistent_noise", "--", False)):
            mu = [d["E4_noise"][n][ctrl][m]["auc_perturbed_vs_original"]["mean"] for n in names]
            b.plot(x, mu, color=OI[m], ls=ls, marker=STYLE[m]["marker"],
                   mfc=OI[m] if fill else "white")
            ends[(m, fill)] = mu[-1]
    # direct labels at the right end of each line
    lab = dict(xytext=(5, 0), textcoords="offset points", va="center", fontsize=6)
    b.annotate("AE", xy=(3, ends[("DeepAutoencoder", True)]), color=OI["DeepAutoencoder"], **lab)
    b.annotate("DRIFT", xy=(3, ends[("DRIFT", True)]), color=OI["DRIFT"], **lab)
    b.annotate("AE,\nDRIFT", xy=(3, 0.5 * (ends[("DeepAutoencoder", False)] + ends[("DRIFT", False)])),
               color="k", **lab)
    b.axhline(0.5, color="k", lw=0.5, ls=(0, (1, 2)))
    b.set_xticks(x)
    b.set_xticklabels(["$\\pm$5%", "default", "wide", "very\nwide"])
    b.set_xlim(-0.2, 4.1)
    b.set_xlabel("Noise width")
    b.set_ylabel("Control AUC")
    b.set_ylim(0.45, 1.0)
    b.set_title("(a)", loc="left", pad=2)
    b.legend([Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--")],
             ["G1 noise", "G2 noise"], frameon=False, loc="upper left", handlelength=2.2,
             borderaxespad=0.0, labelspacing=0.2, fontsize=6)
    fig.subplots_adjust(wspace=0.5)
    fig.savefig(FIG_DIR / "fig_robustness.pdf")
    plt.close(fig)


def fig_ofac(tag):
    """(a) Density of the DRIFT score for corpus wallets, sanctioned addresses and
    matched controls. (b) Percentile ranks within the corpus, by decile.
    Data from ofac_figure_data.py."""
    path = dc.OUT_DIR / f"ofac_figure_{tag}.json"
    if not path.exists():
        print("skip OFAC figure:", path)
        return
    f = json.load(open(path))
    blue, verm = OI["IsolationForest"], OI["DRIFT"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 2.35))

    # (a) score densities
    edges = np.array(f["score_bins"])
    w = np.diff(edges)
    n_corpus = f"{f['n_corpus']:,}".replace(",", "{,}")
    h1 = a.bar(edges[:-1], f["density_corpus"], width=w, align="edge", color=blue, alpha=0.55, lw=0)
    h2 = a.bar(edges[:-1], f["density_sanctioned"], width=w, align="edge", color=verm, alpha=0.60, lw=0)
    h3 = a.stairs(f["density_controls"], edges, color="k", lw=0.9)
    h4 = a.axvline(f["corpus_p90_score"], color="k", ls="--", lw=0.8)
    a.set_xlim(0, 1)
    a.set_xlabel("DRIFT anomaly score")
    a.set_ylabel("Probability density")
    a.set_title("(a)", loc="left", pad=2)
    a.legend([h1, h2, h3, h4],
             [f"Corpus wallets ($n={n_corpus}$)", f"Sanctioned addresses ($n={f['n_sanctioned']}$)",
              "Matched corpus controls", "Corpus 90th percentile"],
             frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.25, fontsize=6.5)

    # (b) percentile ranks by decile: within each decile, sanctioned on the left, controls on the right
    left = np.arange(0, 100, 10)
    g1 = b.bar(left + 1.0, f["decile_share_sanctioned"], width=4.0, align="edge", color=verm, alpha=0.85, lw=0)
    g2 = b.bar(left + 5.0, f["decile_share_controls"], width=4.0, align="edge", color="white",
               edgecolor="k", lw=0.6, hatch="////")
    g3 = b.axhline(10, color="k", ls="--", lw=0.8)
    b.set_xlim(0, 100)
    b.set_xticks(np.arange(0, 101, 10))
    b.set_xlabel("DRIFT percentile within the corpus")
    b.set_ylabel("Share of addresses (%)")
    b.set_title("(b)", loc="left", pad=2)
    b.legend([g1, g2, g3], ["Sanctioned addresses", "Matched corpus controls", "Whole corpus (uniform)"],
             frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.25, fontsize=6.5)
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(FIG_DIR / "fig_ofac.pdf")
    plt.close(fig)


def fig_budget_h():
    """Recall at fixed review budgets on G2 and on the held-out typologies H,
    for DRIFT, its components, the heuristic and the ranking by received value."""
    x = json.load(open(dc.OUT_DIR / "extensions.json"))
    budgets = ["0.01", "0.02", "0.05", "0.1"]
    xs = [1, 2, 5, 10]
    col = dict(OI, VolumeReceived="#000000")
    sty = dict(STYLE, VolumeReceived=dict(ls=(0, (5, 1.5)), marker="v"))
    lab = dict(LABEL, VolumeReceived="Received value")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.45), sharey=True)
    for ax, exp, gen, title in ((axes[0], "E1_main", "G2", "(a) G2"), (axes[1], "E2_H", "H", "(b) H")):
        for m in ["VolumeReceived"] + DETS:
            src = x["E11_volume"][gen][m] if m == "VolumeReceived" else d[exp][m]
            mu = np.array([src["R@" + b]["mean"] for b in budgets])
            sd = np.array([src["R@" + b]["std"] for b in budgets])
            ax.fill_between(xs, mu - sd, mu + sd, color=col[m], alpha=0.13, lw=0)
            ax.plot(xs, mu, color=col[m], label=lab[m], **sty[m])
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(["1", "2", "5", "10"])
        ax.minorticks_off()
        ax.set_xlabel("Review budget (%)")
        ax.set_ylim(0, 0.9)
        ax.set_title(title, loc="left", pad=2)
    axes[0].set_ylabel("Recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, handlelength=2.6,
               bbox_to_anchor=(0.5, 1.2), columnspacing=1.0, labelspacing=0.2, fontsize=6.5)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(FIG_DIR / "fig_budget_h.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    fig_budget_curves()
    fig_budget_h()
    fig_robustness()
    fig_ofac(sys.argv[1] if len(sys.argv) > 1 else "full")
    print("figures ->", FIG_DIR)
