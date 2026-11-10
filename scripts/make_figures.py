"""Paper figures, regenerated from the results directory.

Each panel is saved separately (PNG + PDF) to --out; panels whose inputs are missing are
skipped with a message, so the script can run at any stage of the project.

    python scripts/make_figures.py --results results --data data --out figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

# Journal style (RSC Digital Discovery): Arial 7 pt, thin lines, full frame with inward ticks on all four
# sides, single-column width 8.3 cm (3.3 in), double-column 17.1 cm (6.7 in).
STYLE = {
    "font.family": "Arial",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.titlepad": 5,
    "axes.labelsize": 7,
    "axes.linewidth": 0.75,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 2.5,
    "xtick.major.pad": 2.5,
    "xtick.major.width": 0.75,
    "ytick.major.size": 2.5,
    "ytick.major.pad": 2.5,
    "ytick.major.width": 0.75,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.frameon": False,
    "legend.fontsize": 6,
    "lines.linewidth": 0.75,
    "lines.markersize": 3,
    "errorbar.capsize": 1.5,
    "figure.figsize": (3.3, 2.6),
    "figure.subplot.wspace": 0.4,
    "figure.subplot.hspace": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,  # embed TrueType fonts (journal requirement)
    "mathtext.fontset": "custom",  # sub/superscripts in Arial as well
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "ps.fonttype": 42,
}
SINGLE = (3.3, 2.6)  # single column, inches
# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999")  # fmt: skip
SALT_COLORS = {"LiFSI": BLUE, "LiPF6": VERMILION, "LiTFSI": GREEN}
CYCLE = [BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, GREY]


def setup_style(font_dir: Path | None) -> None:
    """Apply the journal style; register Arial from ``font_dir`` if given, else fall back."""
    from matplotlib import font_manager as fm

    if font_dir is not None and font_dir.exists():
        for f in fm.findSystemFonts(str(font_dir)):
            fm.fontManager.addfont(f)
    plt.rcParams.update(STYLE)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CYCLE)
    if "Arial" not in {f.name for f in fm.fontManager.ttflist}:
        print("Arial not found; using Liberation Sans/DejaVu Sans (pass --font-dir with arial.ttf)")
        plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
        for k in ("mathtext.rm", "mathtext.it", "mathtext.bf"):
            plt.rcParams[k] = plt.rcParams[k].replace("Arial", "Liberation Sans")


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):  # vector PDF for submission, 600 dpi PNG for checking
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {name}")


def _canvas(ax, size):
    """Use the given axes (composite figure) or create a stand-alone figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        return fig, ax, True
    return ax.figure, ax, False


def _canvas_multi(axes, nrows: int, ncols: int, size, **kw):
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=size, **kw)
        return fig, axes, True
    axes = np.asarray(axes)
    return axes.flat[0].figure, axes, False


def _label(ax, letter: str) -> None:
    """Panel label '(a)' outside the top-left corner of the axes."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-24, 4),
                textcoords="offset points", fontsize=8, fontweight="bold", va="bottom")  # fmt: skip


def _inner_legend(ax, **kw):
    """Legend inside the axes with a thin frame so it is not mistaken for data."""
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="0.6", framealpha=1.0, handletextpad=0.3,
                    borderpad=0.4, **kw)  # fmt: skip
    leg.get_frame().set_linewidth(0.5)
    return leg


def _place_labels(ax, xs, ys, labels, fontsize: float = 5.0) -> None:
    """Annotate points, trying right/left/above/below so labels avoid each other and the markers."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    axbox = ax.get_window_extent(renderer)
    placed = []
    cands = [(3, 0, "left", "center"), (-3, 0, "right", "center"), (0, 3, "center", "bottom"),
             (0, -3, "center", "top"), (3, 3, "left", "bottom"), (-3, 3, "right", "bottom"),
             (3, -3, "left", "top"), (-3, -3, "right", "top")]  # fmt: skip
    for k, (x, y, lab) in enumerate(zip(xs, ys, labels, strict=True)):
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=fontsize)  # fmt: skip
            bb = t.get_window_extent(renderer).expanded(1.04, 1.1)
            hits = sum(bb.overlaps(o) for o in placed) + sum(
                bb.contains(*pts[j]) for j in range(len(pts)) if j != k
            )
            inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
            score = hits + (0 if inside else 5)
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, bb)
            else:
                t.remove()
            if score == 0:
                break
        placed.append(best[2])


def summary(run: Path) -> dict:
    return json.loads((run / "run.json").read_text())["summary"]


def fig_consistency(data: Path, out: Path, ax=None) -> None:
    """In-house MD (363 K) vs OEDB (298 K) on shared Li systems."""
    c = pd.read_csv(data / "overlap_comparison.csv")
    fig, ax, own = _canvas(ax, SINGLE)
    for salt, g in c.groupby("salt"):
        rho = spearmanr(g.sigma_oedb_298K, g.sigma_inhouse_363K)[0]
        ax.scatter(g.sigma_oedb_298K, g.sigma_inhouse_363K, s=8, color=SALT_COLORS.get(salt),
                   label=f"{salt.replace('PF6', 'PF$_6$')} (\u03c1 = {rho:.2f})")  # fmt: skip
    ax.set(xscale="log", yscale="log", xlabel="OEDB \u03c3, 298 K (mS cm$^{-1}$)",
           ylabel="In-house MD \u03c3, 363 K (mS cm$^{-1}$)")  # fmt: skip
    rho = spearmanr(c.sigma_oedb_298K, c.sigma_inhouse_363K)[0]
    ax.text(0.04, 0.96, f"n = {len(c)}\nSpearman \u03c1 = {rho:.2f}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False)
    if own:
        save(fig, out, "fig_consistency")


def fig_coverage(results: Path, out: Path, ax=None) -> None:
    """Prediction error vs similarity to the nearest training solvent."""
    r = pd.read_csv(results / "log10" / "coverage_vs_error.csv")
    fig, ax, own = _canvas(ax, SINGLE)
    markers = {"fold_solvent": "o", "fold_class": "s", "fold_ethers": "^"}
    labels = {"fold_solvent": "Leave-solvent-out", "fold_class": "Leave-class-out",
              "fold_ethers": "Leave-ethers-out"}  # fmt: skip
    colors = {"fold_solvent": BLUE, "fold_class": ORANGE, "fold_ethers": VERMILION}
    for split, g in r.groupby("split"):
        ax.scatter(g.nn_similarity, 10**g.log_err, s=14, marker=markers.get(split, "o"), facecolors="none",
                   edgecolors=colors.get(split, "k"), linewidths=1.0,
                   label=labels.get(split, split))  # fmt: skip
    bins = [0, 0.3, 0.45, 0.6, 1.0]
    r["bin"] = pd.cut(r.nn_similarity, bins)
    med = r.groupby("bin", observed=True).agg(x=("nn_similarity", "median"), y=("log_err", "median"))
    ax.plot(med.x, 10**med.y, "k-", lw=1.2, label="Binned median")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5, 10, 20]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    ax.set(xlabel="Tanimoto similarity to nearest training solvent",
           ylabel="Error factor")  # fmt: skip
    _inner_legend(ax, loc="upper right")
    if own:
        save(fig, out, "fig_coverage")


def fig_learning_curve(results: Path, out: Path, ax=None) -> None:
    """Leave-ethers-out error vs fraction of ether data added back."""
    if not list((results / "lc_ethers").glob("ab*_none")):
        raise FileNotFoundError(2, "no learning-curve runs", str(results / "lc_ethers"))
    fig, ax, own = _canvas(ax, SINGLE)
    for arm, color in [("none", "k"), ("structure", VERMILION)]:
        xs, ms, sds = [], [], []
        zero = results / "ether_seeds" / f"gat_{arm}"
        if (zero / "run.json").exists():
            s = summary(zero)["MAE_log10"]
            xs.append(0.0), ms.append(s["mean"]), sds.append(s["std"])
        for run in sorted((results / "lc_ethers").glob(f"ab*_{arm}")):
            s = summary(run)["MAE_log10"]
            xs.append(float(run.name[2:].split("_")[0])), ms.append(s["mean"]), sds.append(s["std"])
        if xs:
            order = np.argsort(xs)
            label = "no OEDB features" if arm == "none" else "OEDB structure features"
            x, m, sd = (np.array(v)[order] for v in (xs, ms, sds))
            ax.errorbar(x * 100, m, yerr=sd, color=color, marker="o", capsize=1.5, label=label)
    ax.set(
        xlabel="Ether compositions added to training (%)", ylabel="log$_{10}$ MAE on ethers", ylim=(0, None)
    )
    ax.legend(frameon=False)
    if own:
        save(fig, out, "fig_learning_curve")


def fig_gat_gcn_classes(results: Path, data: Path, out: Path, ax=None) -> None:
    """GAT vs GCN per held-out solvent class (10 seeds)."""
    f = pd.read_csv(data / "folds_inhouse.csv")[["solvent_class", "fold_class"]].drop_duplicates()
    names = dict(zip("fold_class=" + f.fold_class.astype(str), f.solvent_class, strict=True))
    a = pd.read_csv(results / "class_seeds" / "gat_none" / "fold_metrics.csv")
    b = pd.read_csv(results / "class_seeds" / "gcn_none" / "fold_metrics.csv")
    m = a.merge(b, on=["seed", "split"], suffixes=("_gat", "_gcn"))
    g = m.groupby("split")[["MAE_log10_gat", "MAE_log10_gcn"]].agg(["mean", "std"])
    g.index = [names[s] for s in g.index]
    g = g.sort_values(("MAE_log10_gat", "mean"))
    y = np.arange(len(g))
    fig, ax, own = _canvas(ax, SINGLE)
    for col, color, off, lab in [
        ("MAE_log10_gat", BLUE, -0.15, "GAT"),
        ("MAE_log10_gcn", ORANGE, 0.15, "GCN"),
    ]:
        ax.errorbar(
            g[(col, "mean")], y + off, xerr=g[(col, "std")], fmt="o", color=color, capsize=1.5, label=lab
        )
    ax.set_yticks(y, g.index)
    ax.set_xlabel("log$_{10}$ MAE (held-out class, 10 seeds)")
    ax.legend(frameon=False)
    if own:
        save(fig, out, "fig_gat_gcn_classes")


def fig_seed_variance(results: Path, out: Path, ax=None) -> None:
    """Per-seed error on the leave-ethers-out split: ten seeds vs the first three."""
    runs = [
        ("gat_none", "GAT"),
        ("gat_structure", "GAT +\nOEDB"),
        ("gcn_none", "GCN"),
        ("gcn_structure", "GCN +\nOEDB"),
    ]
    vals = [(lab, pd.read_csv(results / "ether_seeds" / r / "fold_metrics.csv").MAE_log10.to_numpy())
            for r, lab in runs]  # fmt: skip
    fig, ax, own = _canvas(ax, SINGLE)
    for i, (_, v) in enumerate(vals):
        ax.scatter(np.full(len(v), i) + np.linspace(-0.1, 0.1, len(v)), v, s=4, facecolors="none",
                   edgecolors="0.35", linewidths=0.6)  # fmt: skip
        ax.plot([i - 0.22, i + 0.22], [v.mean()] * 2, "-", color=VERMILION, lw=1.2,
                label="Mean of 10 seeds" if i == 0 else None)  # fmt: skip
        ax.plot([i - 0.22, i + 0.22], [v[:3].mean()] * 2, "--", color=BLUE, lw=1.0,
                label="Mean of seeds 0\u20132" if i == 0 else None)  # fmt: skip
    ax.set_xticks(range(len(vals)), [lab for lab, _ in vals])
    ax.set_ylabel("log$_{10}$ MAE on ethers")
    _inner_legend(ax, loc="upper left")
    if own:
        save(fig, out, "fig_seed_variance")


def fig_early_stopping(results: Path, out: Path, ax=None) -> None:
    """LCE: early stopping on the test fold (original protocol) vs a held-out part of the training fold."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from elecgnn.metrics import compare_runs

    a = pd.read_csv(results / "original_protocol" / "lce_gat" / "fold_metrics.csv")
    b = pd.read_csv(results / "lce_gat_inner_dipole" / "fold_metrics.csv")
    m = a.merge(b, on=["seed", "split"], suffixes=("_outer", "_inner"))
    test = compare_runs(a, b, "MAE")
    fig, ax, own = _canvas(ax, SINGLE)
    for _, r in m.iterrows():
        ax.plot([0, 1], [r.MAE_outer, r.MAE_inner], "-", color=SKY, lw=0.6, alpha=0.8)
    ax.scatter(np.zeros(len(m)), m.MAE_outer, s=6, c=SKY, linewidths=0, zorder=3)
    ax.scatter(np.ones(len(m)), m.MAE_inner, s=6, c=SKY, linewidths=0, zorder=3)
    ax.plot([0, 1], [m.MAE_outer.mean(), m.MAE_inner.mean()], "-", color="k", lw=1.4, zorder=4)
    ax.scatter([0, 1], [m.MAE_outer.mean(), m.MAE_inner.mean()], s=22, c="w", edgecolors="k",
               linewidths=1.2, zorder=5, label="Mean")  # fmt: skip
    ax.set_xticks([0, 1], ["Early stopping\non test fold", "Early stopping\non held-out data"])
    ax.set(ylabel="LCE MAE (per fold and seed)", xlim=(-0.35, 1.35))
    note = f"Held-out \u2212 test-fold: {-test['mean_diff']:+.3f} (p = {test['p']:.2f})"
    ax.text(0.5, 0.97, note, transform=ax.transAxes, ha="center", va="top")
    if own:
        save(fig, out, "fig_early_stopping")


def fig_oedb_features(results: Path, out: Path, ax=None) -> None:
    """Change in log10 MAE from each OEDB feature set, per split (mean +/- sd over paired folds and seeds).

    Leave-ethers-out is a single held-out group, so it is shown only where ten seeds exist
    (structure features; results/ether_seeds). The three-seed ether runs are not plotted: their
    apparent gain is the false positive discussed with the seed-variance panel.
    """
    root = results / "log10" / "oedb_features"
    sets = ["transport", "structure", "physics", "all", "eta", "sigma"]
    set_labels = ["Transport", "Structure", "Physics", "All", "\u03b7", "\u03c3"]
    fig, ax, own = _canvas(ax, SINGLE)

    def delta(a_dir: Path, b_dir: Path):
        a, b = pd.read_csv(a_dir / "fold_metrics.csv"), pd.read_csv(b_dir / "fold_metrics.csv")
        m = a.merge(b, on=["seed", "split"], suffixes=("", "_0"))
        d = m.MAE_log10 - m.MAE_log10_0
        return d.mean(), d.std()

    for k, (d, lab) in enumerate([("ioncon_fold_solvent", "Leave-solvent-out"),
                                  ("ioncon_fold_class", "Leave-class-out")]):  # fmt: skip
        diffs = [delta(root / d / s, root / d / "none") for s in sets]
        x = np.arange(len(sets)) + (k - 1) * 0.25
        ax.errorbar(x, [v[0] for v in diffs], yerr=[v[1] for v in diffs], fmt="o", capsize=1.5,
                    label=lab)  # fmt: skip
    eth = results / "ether_seeds"
    if (eth / "gat_structure" / "fold_metrics.csv").exists():
        m, sd = delta(eth / "gat_structure", eth / "gat_none")
        ax.errorbar([sets.index("structure") + 0.25], [m], yerr=[sd], fmt="o", capsize=1.5, color=GREEN,
                    label="Leave-ethers-out (10 seeds)")  # fmt: skip
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(sets)), set_labels, rotation=25, ha="right", rotation_mode="anchor")
    ax.set(ylabel="\u0394 log$_{10}$ MAE vs no features", xlabel="OEDB-derived feature set")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0,
        handletextpad=0.3,
        columnspacing=0.8,
        ncol=2,
    )
    if own:
        save(fig, out, "fig_oedb_features")


def fig_external_lce(results: Path, out: Path, ax=None) -> None:
    """RMSE on Kim et al.'s 13 external electrolytes: this work (10 seeds) vs published models."""
    ours = [("GCN", "gcn_none"), ("GAT", "gat_none"), ("GAT + OEDB physics", "gat_physics"),
            ("GAT + OEDB all", "gat_all")]  # fmt: skip
    rows = []
    for label, run in ours:
        s = summary(results / "lce_ext13" / run)["RMSE"]
        rows.append((label, s["mean"], s["std"], "this work (10 seeds)"))
    published = [("Kim linear, published", 0.162), ("BART-SA", 0.148), ("MM-MoLFormer", 0.195),
                 ("MoLFormer", 0.213), ("F-GCN (TL)", 0.39)]  # fmt: skip
    rows += [(lab, v, 0.0, "published") for lab, v in published]
    linear = results / "lce_ext13" / "linear_kim_features.json"
    if not linear.exists():
        raise FileNotFoundError(
            2, "run scripts/kim_linear_baseline.py first (refitted Kim baselines)", str(linear)
        )
    kim = json.loads(linear.read_text())
    rows.append(("Kim linear, refit", kim["RMSE"], 0.0, "this work"))
    rows.append(("Kim RF, refit", kim["rf_RMSE_mean"], kim["rf_RMSE_sd"], "this work"))
    rows.sort(key=lambda r: r[1])
    fig, ax, own = _canvas(ax, SINGLE)
    colors = {"this work (10 seeds)": BLUE, "this work": GREEN, "published": GREY}
    for i, (_, m, sd, src) in enumerate(rows):
        ax.barh(i, m, xerr=sd if sd else None, color=colors[src], capsize=1.5)
    ax.axvline(0.05, color=VERMILION, ls="--", lw=0.75, label="replicate noise")
    ax.legend(loc="lower right")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("RMSE, 13 external electrolytes (LCE)")
    if own:
        save(fig, out, "fig_external_lce")


def fig_attention(results: Path, out: Path, ax=None) -> None:
    """Degree-corrected GAT attention by atom environment (conductivity model, solvent encoder)."""
    s = pd.read_csv(results / "interp" / "class_gat" / "interpret" / "attribution_summary.csv")
    s = s[(s["role"] == "solvent") & (s["n"] >= 15)].sort_values("attn_ratio")
    fig, ax, own = _canvas(ax, SINGLE)
    ax.barh(s["group"], s["attn_ratio"], color=np.where(s["attn_ratio"] > 1, VERMILION, BLUE))
    ax.axvline(1, color="k", lw=0.6)
    ax.set_xlabel("Attention received / uniform")
    if own:
        save(fig, out, "fig_attention")


def fig_graph_vs_descriptors(results: Path, out: Path, ax=None) -> None:
    """log10 MAE per split: graph-only, descriptors (LightGBM, molar), hybrid (graph + descriptors)."""
    runs = {
        "fold_solvent": {"GAT": "log10/oedb_features/ioncon_fold_solvent/none",
                         "GCN": "pretrain_study/ioncon_solvent/gcn_none"},
        "fold_class": {"GAT": "log10/oedb_features/ioncon_fold_class/none", "GCN": "class_seeds/gcn_none"},
        "fold_ethers": {"GAT": "ether_seeds/gat_none", "GCN": "ether_seeds/gcn_none"},
    }  # fmt: skip
    labels = {
        "fold_solvent": "Leave-solvent-out",
        "fold_class": "Leave-class-out",
        "fold_ethers": "Leave-ethers-out",
    }
    arms = [("GAT", BLUE), ("GCN", ORANGE), ("LightGBM descriptors", GREEN),
            ("GAT + descriptors", PURPLE), ("GCN + descriptors", SKY)]  # fmt: skip
    fig, ax, own = _canvas(ax, SINGLE)
    w = 0.16
    for k, (split, graph) in enumerate(runs.items()):
        paths = {"GAT": graph["GAT"], "GCN": graph["GCN"],
                 "LightGBM descriptors": f"descriptors/{split}/LGBM_molar",
                 "GAT + descriptors": f"hybrid/{split}/gat_desc_molar",
                 "GCN + descriptors": f"hybrid/{split}/gcn_desc_molar"}  # fmt: skip
        for j, (arm, color) in enumerate(arms):
            s = summary(results / paths[arm])["MAE_log10"]
            ax.bar(k + (j - 2) * w, s["mean"], w, yerr=s["std"], color=color, capsize=1.5,
                   label=arm if k == 0 else None)  # fmt: skip
    ax.set_xticks(range(len(runs)), [labels[s] for s in runs])
    ax.set_ylabel("log$_{10}$ MAE (held-out solvents)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0, handlelength=1.0)
    if own:
        save(fig, out, "fig_graph_vs_descriptors")


def fig_pretraining_targets(results: Path, out: Path, ax=None) -> None:
    """Change in error from molecular pretraining targets, relative to no pretraining."""
    tasks = [
        ("ioncon_solvent", "MAE_log10", "in-house MD"),
        ("oedb_sigma", "MAE", "OEDB"),
    ]
    arms = ["dipole", "homo", "lumo", "gap", "em", "all4"]
    fig, ax, own = _canvas(ax, SINGLE)
    for k, (task, metric, lab) in enumerate(tasks):
        for m, marker in [("gat", "o"), ("gcn", "s")]:
            base = pd.read_csv(results / "pretrain_study" / task / f"{m}_none" / "fold_metrics.csv")
            mean, sd = [], []
            for a in arms:
                d = pd.read_csv(results / "pretrain_study" / task / f"{m}_{a}" / "fold_metrics.csv")
                diff = d.merge(base, on=["seed", "split"], suffixes=("", "_0"))
                diff = diff[metric] - diff[f"{metric}_0"]
                mean.append(diff.mean()), sd.append(diff.std())
            x = np.arange(len(arms)) + (k * 2 + (m == "gcn") - 1.5) * 0.12
            ax.errorbar(x, mean, yerr=sd, fmt=marker, capsize=1.5, label=f"{m.upper()}, {lab}")
    ax.axhline(0, color="k", lw=0.6)
    ticks = ["Dipole\n(raw)", "HOMO", "LUMO", "Gap", "Dipole\n(scaled)", "All\nfour"]
    ax.set_xticks(range(len(arms)), ticks)
    ax.set(ylabel="\u0394 log$_{10}$ MAE vs no pretraining", xlabel="Molecular pretraining target")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0,
        handletextpad=0.3,
        columnspacing=0.8,
        ncol=2,
    )
    if own:
        save(fig, out, "fig_pretraining_targets")


DOUBLE = (6.7, 2.4)  # double column, inches
FEATURE_LABELS = {
    "HOMO": "HOMO", "LUMO": "LUMO", "VIP": "VIP", "VEA": "VEA", "AIP": "AIP", "AEA": "AEA",
    "Red": "Red. pot.", "Ox": "Ox. pot.", "Solv_Ener": "\u0394G$_{solv}$", "Reorg_Ener": "Reorg. energy",
    "Mol_Vol": "Mol. volume", "Dipole": "Dipole", "MP": "Melting pt.", "BP": "Boiling pt.",
    "Dielec": "Dielectric const.", "MolWt": "Mol. weight", "ValElec": "Valence e$^-$",
    "MinParCharge": "Min. charge", "MaxParCharge": "Max. charge", "Electroneg": "Electroneg.",
    "MolWt_Ani": "Anion MW", "ValElec_Ani": "Anion valence e$^-$", "Electroneg_Ani": "Anion electroneg.",
    "O": "O frac.", "sO": "Solvent O frac.", "aO": "Anion O frac.", "sC": "Solvent C frac.",
    "F": "F frac.", "sF": "Solvent F frac.", "aF": "Anion F frac.", "Salt_Solv_Ratio": "Salt/solvent",
    "x_solvent": "x$_{solvent}$", "x_salt": "x$_{salt}$",
}  # fmt: skip
ETHER_CLASSES = {"Ether", "Fluorinated ether"}


def _row_means(pred_csv: Path) -> pd.DataFrame:
    """Per-row mean and seed spread of log10 predictions (rows predicted out of fold)."""
    p = pd.read_csv(pred_csv)
    p = p[p["split"] != "holdout"]
    p["log_pred"] = np.log10(np.clip(p["y_pred"], 1e-2, None))
    g = p.groupby("row_id").agg(y_true=("y_true", "first"), log_pred=("log_pred", "mean"),
                                spread=("log_pred", "std"))  # fmt: skip
    g["log_true"] = np.log10(g["y_true"])
    g["abs_err"] = (g["log_pred"] - g["log_true"]).abs()
    return g.reset_index()


def _abbrev(name: str) -> str:
    """'Dimethyl carbonate (DMC)' -> 'DMC'; 'Tetramethylene sulfone (TMS/Sulfolane)' -> 'TMS'."""
    import re

    special = {"Diethyl ether": "Et$_2$O", "Dioxolane (1,3-dioxolane (DOL))": "DOL"}
    if str(name) in special:
        return special[str(name)]
    m = re.search(r"\(([^()]*)\)\s*$", str(name))
    return (m.group(1) if m else str(name)).split("/")[0].strip()


def _fingerprints(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=bool)


def fig_chemical_space(data: Path, out: Path, ax=None) -> None:
    """(A) Solvent chemical space of the three data sources (t-SNE of Morgan fingerprints, Jaccard)."""
    from rdkit import Chem
    from sklearn.manifold import TSNE

    can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))  # noqa: E731
    u = pd.read_csv(data / "unified_md.csv")
    inhouse = set(u.loc[u["source"] == "inhouse", "solvent_smiles"])
    oedb = set(u.loc[u["source"] == "oedb", "solvent_smiles"])
    lce = set()
    lce_df = pd.read_excel(data / "Ref_LCEdata.xlsx")
    for j in range(1, 7):
        for smi, c in zip(lce_df[f"smiles{j}"], lce_df[f"conc{j}"], strict=True):
            if isinstance(smi, str) and c > 0 and smi != "O" and "Li" not in smi and Chem.MolFromSmiles(smi):
                lce.add(can(smi))
    allsm = sorted(inhouse | oedb | lce)
    xy = TSNE(2, metric="jaccard", init="random", perplexity=min(15, len(allsm) - 1),
              random_state=0).fit_transform(_fingerprints(allsm))  # fmt: skip
    df = pd.DataFrame({"smiles": allsm, "x": xy[:, 0], "y": xy[:, 1]})
    df["group"] = np.select([df.smiles.isin(inhouse) & df.smiles.isin(oedb), df.smiles.isin(inhouse),
                             df.smiles.isin(oedb)], ["Both MD sets", "In-house MD only", "OEDB only"],
                            "LCE only")  # fmt: skip
    df.to_csv(out / "fig_chemical_space_data.csv", index=False)
    fig, ax, own = _canvas(ax, (3.3, 3.0))
    style = {"LCE only": (GREY, 8, "o"), "OEDB only": (BLUE, 18, "s"), "Both MD sets": (PURPLE, 20, "D"),
             "In-house MD only": (VERMILION, 18, "o")}  # fmt: skip
    for grp, (c, size, m) in style.items():
        d = df[df.group == grp]
        ax.scatter(d.x, d.y, s=size, c=c, marker=m, label=f"{grp} ({len(d)})", linewidths=0)
    in_lce = df[df.smiles.isin(lce) & (df.group != "LCE only")]
    ax.scatter(in_lce.x, in_lce.y, s=34, facecolors="none", edgecolors="k", linewidths=0.5,
               label=f"Also in LCE set ({len(in_lce)})")  # fmt: skip
    names = pd.read_csv(data / "solvent_db_1_clean.csv").assign(can=lambda d: d.SMILES.map(can))
    names = names.groupby("can")["Solvent"].first().map(_abbrev)
    show = {"DME", "THF", "BTFE", "FEPE", "EC", "DMC", "TMS", "FEC"}
    lab = df[df.smiles.isin(names.index) & df.smiles.map(names).isin(show)]
    _place_labels(ax, lab.x.to_numpy(), lab.y.to_numpy(), lab.smiles.map(names).tolist(), fontsize=5.5)
    ax.set(xticks=[], yticks=[], xlabel="t-SNE 1 (solvent fingerprints)", ylabel="t-SNE 2")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, handletextpad=0.3, columnspacing=0.8)
    if own:
        save(fig, out, "fig_chemical_space")


def fig_parity_failure(results: Path, data: Path, out: Path, axes=None) -> None:
    """(B) Same GAT: random split vs all ethers held out vs graph + descriptors (log10 sigma)."""
    cls = pd.read_csv(data / "unified_md.csv").set_index("row_id")["solvent_class"]
    panels = [("Random split", results / "log10/random_none/predictions.csv"),
              ("Ethers held out: GAT", results / "ether_seeds/gat_none/predictions.csv"),
              ("Ethers held out: GAT + descriptors",
               results / "hybrid/fold_ethers/gat_desc_molar/predictions.csv")]  # fmt: skip
    fig, axes, own = _canvas_multi(axes, 1, 3, DOUBLE, sharex=True, sharey=True)
    for ax, (_title, path) in zip(axes, panels, strict=True):
        g = _row_means(path)
        ether = g["row_id"].map(cls).isin(ETHER_CLASSES)
        ax.scatter(
            g.y_true[~ether], 10 ** g.log_pred[~ether], s=6, c=GREY, linewidths=0, label="Other solvents"
        )
        ax.scatter(g.y_true[ether], 10 ** g.log_pred[ether], s=6, c=VERMILION, linewidths=0, label="Ethers")
        lim = (0.03, 60)
        ax.plot(lim, lim, "k-", lw=0.6)
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="MD \u03c3 (mS cm$^{-1}$)")
        note = f"log$_{{10}}$ MAE {g.abs_err.mean():.2f}\nmedian error {10 ** g.abs_err.median():.1f}\u00d7"
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top")
    axes[0].set_ylabel("Predicted \u03c3 (mS cm$^{-1}$)")
    _inner_legend(axes[0], loc="lower right", markerscale=1.5)
    if own:
        save(fig, out, "fig_parity_failure")


def fig_screening(results: Path, out: Path, top: float = 0.10, ax=None) -> None:
    """(C) Leave-class-out screening: recall of the true top-10% conductors among the top-k predicted."""
    runs = [("GAT", "class_seeds/gat_none", BLUE), ("GCN", "class_seeds/gcn_none", ORANGE),
            ("LightGBM descriptors", "descriptors/fold_class/LGBM_molar", GREEN),
            ("GAT + descriptors", "hybrid/fold_class/gat_desc_molar", PURPLE)]  # fmt: skip
    short = {"LightGBM descriptors": "LightGBM", "GAT + descriptors": "GAT + desc."}
    fig, ax, own = _canvas(ax, SINGLE)
    fr = np.linspace(0.01, 0.5, 50)
    rows = []
    for label, run, color in runs:
        g = _row_means(results / run / "predictions.csv")
        n_top = int(np.ceil(top * len(g)))
        truth = set(g.nlargest(n_top, "y_true").row_id)
        order = g.sort_values("log_pred", ascending=False).row_id.to_numpy()
        rec = [len(truth & set(order[: max(1, int(round(f * len(g))))])) / n_top for f in fr]
        ef = len(truth & set(order[:n_top])) / n_top / top
        rows.append({"model": label, "recall_at_10pct": ef * top, "enrichment_at_10pct": ef})
        ax.plot(fr * 100, rec, color=color, label=f"{short.get(label, label)} ({ef:.1f})")
    ax.plot(fr * 100, fr, "k--", lw=0.6, label="Random (1.0)")
    pd.DataFrame(rows).to_csv(out / "fig_screening_data.csv", index=False)
    ax.set(xlabel="Candidates selected (% of compositions)", ylabel="Recall of true top-10% conductors",
           xlim=(0, 50), ylim=(0, 1.02))  # fmt: skip
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), borderaxespad=0, ncol=3, handlelength=1.4,
              handletextpad=0.3, columnspacing=0.7, title="EF$_{10\\%}$ in parentheses",
              title_fontsize=5.5)  # fmt: skip
    if own:
        save(fig, out, "fig_screening")


def _tsne_layout(fig, gs=None):
    """2 x 2 t-SNE panels plus a side column (legend for row 1, colour bar for row 2) of equal widths."""
    gs = gs or fig.add_gridspec(1, 1)[0]
    split = gs.subgridspec(1, 2, width_ratios=[2.0, 0.42], wspace=0.03)
    grid = split[0].subgridspec(2, 2, wspace=0.32, hspace=0.18)
    sides = split[1].subgridspec(2, 1, hspace=0.18)
    axes = np.array([[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(2)])
    side = [fig.add_subplot(sides[i]) for i in range(2)]
    for ax in side:
        ax.axis("off")
    return axes, side


def fig_tsne_control(results: Path, out: Path, axes=None, side=None) -> None:
    """(D) t-SNE of learned descriptors vs a random-weight network (k-NN R2 of the target in titles).

    Conductivity panels are coloured by solvent class (the clusters), CE panels by LCE.
    """
    sets = [("Ionic conductivity", results / "interp/class_gat/interpret_emb", "class"),
            ("Coulombic efficiency", results / "lce_rep/gat_none/interpret_emb", "LCE")]  # fmt: skip
    own = axes is None
    if own:
        fig = plt.figure(figsize=(6.7, 5.4))
        axes, side = _tsne_layout(fig)
    fig = axes.flat[0].figure
    for i, (_name, d, colour_by) in enumerate(sets):
        emb = pd.read_csv(d / "embeddings.csv")
        sc = json.loads((d / "embedding_scores.json").read_text())
        variants = [
            ("trained GAT", "knn_r2_trained", "trained"),
            ("random-weight GAT", "knn_r2_random_init", "random"),
        ]
        for j, (_kind, key, tag) in enumerate(variants):
            ax = axes[i, j]
            x, y = emb[f"tsne_{tag}_1"], emb[f"tsne_{tag}_2"]
            if colour_by == "class" and "solvent_class" in emb:
                classes = sorted(emb["solvent_class"].dropna().unique())
                palette = plt.get_cmap("tab20")
                for k, c in enumerate(classes):
                    sel = emb["solvent_class"] == c
                    ax.scatter(x[sel], y[sel], s=7, color=palette(k % 20), edgecolors="k", linewidths=0.2,
                               label=c)  # fmt: skip
            else:
                h = ax.scatter(
                    x, y, c=emb["y_transformed"], s=7, cmap="viridis", edgecolors="k", linewidths=0.2
                )
            ax.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2")
            ax.text(0.03, 0.97, f"k-NN R$^2$ = {sc[key]:.2f}", transform=ax.transAxes, va="top", fontsize=6.5)
        if colour_by == "class" and "solvent_class" in emb:
            handles, labels = axes[i, 1].get_legend_handles_labels()
            side[i].legend(handles, labels, loc="center left", fontsize=5.5, title="Solvent class",
                           title_fontsize=6, handletextpad=0.2, borderaxespad=0)  # fmt: skip
        else:
            cax = side[i].inset_axes([0.0, 0.08, 0.09, 0.84])
            fig.colorbar(h, cax=cax, label=colour_by)
    if own:
        save(fig, out, "fig_tsne_control")


def fig_shap_extrapolation(results: Path, out: Path, n_top: int = 10, axes=None) -> None:
    """(E) LightGBM (molar) SHAP importance: random split vs all ethers held out."""
    fig, axes, own = _canvas_multi(axes, 1, 2, DOUBLE)
    for ax, (split, _title) in zip(axes, [("fold_random_s0", "Random split"),
                                         ("fold_ethers", "Ethers held out")], strict=True):  # fmt: skip
        imp = pd.read_csv(results / "descriptors" / split / "LGBM_molar" / "shap_importance.csv").head(n_top)
        imp = imp.iloc[::-1]
        ax.barh([FEATURE_LABELS.get(f, f) for f in imp.feature], imp.mean_abs_shap, color=GREEN)
        ax.set(xlabel="mean |SHAP| (log$_{10}$ \u03c3)")
    if own:
        save(fig, out, "fig_shap_extrapolation")


def fig_descriptor_benchmark(results: Path, out: Path, ax=None) -> None:
    """(SI) Nine descriptor models x two weightings x four splits: log10 MAE heatmap."""
    splits = ["fold_random_s0", "fold_solvent", "fold_class", "fold_ethers"]
    models = ["RFR", "ADR", "XGBR", "LGBM", "SVR", "RDG", "LSO", "GPR", "MLP"]
    rows, labels = [], []
    for m in models:
        for w in ["none", "molar"]:
            rows.append(
                [summary(results / "descriptors" / s / f"{m}_{w}")["MAE_log10"]["mean"] for s in splits]
            )
            labels.append(f"{m} ({w})")
    M = np.array(rows)
    fig, ax, own = _canvas(ax, (3.3, 4.4))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="w" if v > M.mean() else "k")
    ax.set_xticks(
        range(len(splits)),
        ["Random", "Leave-solvent-out", "Leave-class-out", "Leave-ethers-out"],
        rotation=30,
    )
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(im, ax=ax, shrink=0.6, label="log$_{10}$ MAE")
    if own:
        save(fig, out, "fig_si_descriptor_benchmark")


def fig_uncertainty(results: Path, data: Path, out: Path, axes=None) -> None:
    """(F) Does seed-ensemble spread flag large errors? Compared with similarity to training data."""
    from scipy.stats import spearmanr

    cov = pd.read_csv(results / "log10" / "coverage_vs_error.csv").query("split == 'fold_class'")
    sim = cov.set_index("solvent")["nn_similarity"]
    u = pd.read_csv(data / "unified_md.csv").set_index("row_id")["solvent_smiles"]
    fig, axes, own = _canvas_multi(axes, 1, 2, DOUBLE)
    bars = []
    for label, run, color in [("GAT", "class_seeds/gat_none", BLUE), ("GCN", "class_seeds/gcn_none", ORANGE)]:
        g = _row_means(results / run / "predictions.csv")
        rho_row = spearmanr(g.spread, g.abs_err)[0]
        if label == "GAT":
            axes[0].scatter(g.spread, g.abs_err, s=5, c=color, linewidths=0)
            axes[0].set(xlabel="Seed-ensemble spread (log$_{10}$)", ylabel="|log$_{10}$ error|")
            axes[0].text(0.97, 0.96, f"Spearman \u03c1 = {rho_row:.2f}", transform=axes[0].transAxes,
                         ha="right", va="top")  # fmt: skip
        g["solvent"] = g.row_id.map(u)
        per = g.groupby("solvent")[["spread", "abs_err"]].mean().join(sim)
        bars.append(
            (label, spearmanr(per.spread, per.abs_err)[0], spearmanr(-per.nn_similarity, per.abs_err)[0])
        )
    x = np.arange(len(bars))
    axes[1].bar(x - 0.18, [b[1] for b in bars], 0.36, color=SKY, label="Ensemble spread")
    axes[1].bar(x + 0.18, [b[2] for b in bars], 0.36, color=GREY, label="Dissimilarity to training")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xticks(x, [b[0] for b in bars])
    lo = min(0.0, *(v for b in bars for v in b[1:]))
    axes[1].set_ylim(lo - 0.05 if lo < 0 else 0, 1)
    axes[1].set(ylabel="Spearman \u03c1 with per-solvent error")
    axes[1].legend(loc="upper right")
    if own:
        save(fig, out, "fig_uncertainty")


def fig_si_distributions(data: Path, out: Path, axes=None) -> None:
    """(SI) In-house MD conductivity by salt/solvent ratio and by salt (268 compositions)."""
    d = pd.read_csv(data / "solvent_db_1_clean.csv")
    d["sigma"] = d["Ion_Con"] * 10
    d["ratio"] = (d["SOL #"] / d["Li/FSI #"]).round().astype(int)
    fig, axes, own = _canvas_multi(axes, 1, 2, DOUBLE)
    bins = np.logspace(np.log10(d.sigma.min()), np.log10(d.sigma.max()), 20)
    for r, c in zip(sorted(d.ratio.unique()), [BLUE, ORANGE, GREEN], strict=False):
        axes[0].hist(d.sigma[d.ratio == r], bins=bins, alpha=0.6, color=c, label=f"1:{r}")
    axes[0].set(xscale="log", xlabel="\u03c3 at 363 K (mS cm$^{-1}$)", ylabel="Count")
    axes[0].legend(title="salt:solvent")
    salts = sorted(d.Salt.unique())
    axes[1].boxplot([d.sigma[d.Salt == s] for s in salts], widths=0.5, flierprops={"markersize": 2})
    axes[1].set_xticks(range(1, len(salts) + 1), [s.replace("PF6", "PF$_6$") for s in salts])
    axes[1].set(yscale="log", ylabel="\u03c3 at 363 K (mS cm$^{-1}$)")
    if own:
        save(fig, out, "fig_si_distributions")


def fig_si_correlation(data: Path, out: Path, ax=None) -> None:
    """(SI) Pearson correlation of the molecular descriptors and log10 sigma."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from elecgnn.descriptors import build_features

    d = pd.read_csv(data / "solvent_db_1_clean.csv")
    X = build_features(d, "none")
    X["log10 \u03c3"] = np.log10(d["Ion_Con"] * 10)
    C = X.corr()
    fig, ax, own = _canvas(ax, (6.7, 6.0))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    lab = [FEATURE_LABELS.get(c, c) for c in C.columns]
    ax.set_xticks(range(len(lab)), lab, rotation=90)
    ax.set_yticks(range(len(lab)), lab)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    if own:
        save(fig, out, "fig_si_correlation")


def fig_si_tradeoff(data: Path, out: Path, ax=None) -> None:
    """(SI) Conductivity vs oxidation potential per solvent, coloured by HOMO."""
    from scipy.stats import spearmanr

    d = pd.read_csv(data / "solvent_db_1_clean.csv")
    d["sigma"] = d["Ion_Con"] * 10
    g = (
        d.groupby("Solvent")
        .agg(sigma=("sigma", "median"), Ox=("Ox", "first"), HOMO=("HOMO", "first"))
        .dropna()
    )
    fig, ax, own = _canvas(ax, SINGLE)
    h = ax.scatter(g.Ox, g.sigma, c=g.HOMO, cmap="coolwarm", s=12, linewidths=0)
    rho = spearmanr(g.Ox, g.sigma)[0]
    ax.set(yscale="log", xlabel="Oxidation potential (V vs Li/Li$^+$)",
           ylabel="Median \u03c3 at 363 K (mS cm$^{-1}$)",
           xlim=(g.Ox.min() - 0.15, g.Ox.max() + 0.55))  # fmt: skip
    ax.text(0.04, 0.04, f"Spearman \u03c1 = {rho:.2f}\n{len(g)} solvents", transform=ax.transAxes)
    fig.colorbar(h, ax=ax, label="HOMO (eV)")
    _place_labels(ax, g.Ox.to_numpy(), g.sigma.to_numpy(), [_abbrev(n) for n in g.index], fontsize=4.5)
    if own:
        save(fig, out, "fig_si_tradeoff")


# Classes for LCE solvents that are absent from the MD datasets (assigned by structure).
LCE_EXTRA_CLASSES = {
    "C1COCCO1": "Ether", "C1COCCOCCOCCO1": "Ether", "C1COCCOCCOCCOCCOCCO1": "Ether", "CC1CCC(C)O1": "Ether",
    "COC(C)C(C)OC": "Ether", "COCCOCCOCCOCCOC": "Ether",
    "CCOCCOCC(F)(F)F": "Fluorinated ether", "COC(C(F)(F)F)C(F)(F)F": "Fluorinated ether",
    "COCC(F)(F)C(F)(F)C(F)(F)C(F)(F)COC": "Fluorinated ether", "COCC(F)(F)C(F)(F)COC": "Fluorinated ether",
    "COCCOCCOCC(F)(F)OC(F)(F)OC(F)(F)COCCOCCOC": "Fluorinated ether",
    "FC(F)(F)COC(OCC(F)(F)F)OCC(F)(F)F": "Fluorinated ether", "FC(F)(F)COCCOCC(F)(F)F": "Fluorinated ether",
    "FC(F)C(F)(F)OCC(F)(F)F": "Fluorinated ether", "FC(F)COCCOCC(F)(F)F": "Fluorinated ether",
    "FC(F)COCCOCC(F)F": "Fluorinated ether",
    "CCOC(=O)OCC(F)(F)F": "Fluorinated carbonate", "COC(=O)OCCF": "Fluorinated carbonate",
    "O=C(OCC(F)(F)F)OCC(F)(F)F": "Fluorinated carbonate",
    "COC(=O)CC(F)(F)F": "Fluorinated ester", "CN(C)C(=O)C(F)(F)F": "Amide/urea",
}  # fmt: skip
CLASS_ORDER = ["Ether", "Fluorinated ether", "Linear carbonate", "Cyclic carbonate", "Fluorinated carbonate",
               "Ester", "Fluorinated ester", "Lactone", "Sulfone/sulfoxide", "Amide/urea", "Nitrile",
               "Phosphate", "Ketone", "Other"]  # fmt: skip


def fig_coverage_matrix(data: Path, out: Path, ax=None) -> None:
    """Number of distinct solvents per chemical class in each data source (zero cells highlighted)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    can = lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi))  # noqa: E731
    u = pd.read_csv(data / "unified_md.csv")
    known = dict(zip(u.solvent_smiles, u.solvent_class, strict=True))
    sources = {"In-house MD": set(u.loc[u.source == "inhouse", "solvent_smiles"]),
               "OEDB": set(u.loc[u.source == "oedb", "solvent_smiles"]),
               "Experimental CE": set()}  # fmt: skip
    lce = pd.read_excel(data / "Ref_LCEdata.xlsx")
    for j in range(1, 7):
        for smi, c in zip(lce[f"smiles{j}"], lce[f"conc{j}"], strict=True):
            if isinstance(smi, str) and c > 0 and smi != "O" and "Li" not in smi and "Rb" not in smi:
                if Chem.MolFromSmiles(smi):
                    sources["Experimental CE"].add(can(smi))
    rows = [{"source": src, "smiles": smi, "class": known.get(smi) or LCE_EXTRA_CLASSES.get(smi, "Other")}
            for src, sm in sources.items() for smi in sm]  # fmt: skip
    long = pd.DataFrame(rows)
    long.to_csv(out / "fig_coverage_matrix_classes.csv", index=False)
    M = long.groupby(["class", "source"]).size().unstack(fill_value=0)
    M = M.reindex(index=[c for c in CLASS_ORDER if c in M.index], columns=list(sources), fill_value=0)
    fig, ax, own = _canvas(ax, (3.3, 3.2))
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.12, 0.85, 256)))
    ax.imshow(np.where(M.values > 0, M.values, np.nan), cmap=cmap, aspect="auto", vmin=1, vmax=M.values.max())
    for (i, j), v in np.ndenumerate(M.values):
        if v == 0:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F2F2F2", edgecolor="none"))
            ax.text(j, i, "0", ha="center", va="center", color=VERMILION, fontweight="bold")
        else:
            ax.text(j, i, str(v), ha="center", va="center", color="w" if v > 0.6 * M.values.max() else "k")
    ax.set_xticks(range(M.shape[1]), M.columns)
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.tick_params(top=False, right=False, length=0)
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="w", lw=1.0)
    ax.tick_params(which="minor", length=0)
    if own:
        save(fig, out, "fig_coverage_matrix")


# ── Experimental Raman validation ───────────────────────────────────────────────────────────────
ANION_COLORS = {"FSI": BLUE, "TFSI": ORANGE, "BF4": GREEN, "ClO4": VERMILION, "PF6": GREY, "TfO": PURPLE}


def _anion_label(a: str) -> str:
    return {"PF6": "PF$_6$", "BF4": "BF$_4$", "ClO4": "ClO$_4$"}.get(a, a)


def fig_raman_example(data: Path, out: Path, ax=None, anion: str = "FSI", solvent: str = "DME") -> None:
    """Anion band after solvent subtraction at three concentrations, with the fitted components."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from elecgnn.raman import COMPONENTS, _gauss, fit_components, load_spectra, subtract_solvent

    meta, nu, S = load_spectra(data / "oedb-experimental-v2026-05-11.csv")
    lo, hi, comps = COMPONENTS[anion]
    pure = S[meta.index[(meta.solvent == solvent) & (meta.molality == 0)][0]]
    sel = meta[(meta.anion == anion) & (meta.solvent == solvent) & (meta.molality > 0)]
    picks = [sel.molality.sub(m).abs().idxmin() for m in (0.5, 1.5, 3.0)]
    fig, ax, own = _canvas(ax, SINGLE)
    win = (nu >= lo - 10) & (nu <= hi + 10)
    for k, i in enumerate(picks):
        r, _ = subtract_solvent(nu, S[i], pure, lo, hi)
        ax.plot(nu[win], r[win] / r[win].max(), color=[SKY, BLUE, "k"][k], lw=1.0,
                label=f"{meta.molality[i]:g} mol kg$^{{-1}}$")  # fmt: skip
    r, _ = subtract_solvent(nu, S[picks[-1]], pure, lo, hi)
    w = (nu >= lo) & (nu <= hi)
    y = np.clip(r[w], 0, None)
    fit = fit_components(nu[w], y, comps)
    scale = r[win].max()
    for (name, _, _), col in zip(comps, [GREEN, ORANGE, VERMILION], strict=False):
        area, cen, fwhm = fit[f"area_{name}"], fit[f"centre_{name}"], fit[f"fwhm_{name}"]
        amp = area / fwhm
        label = {"free": "Free anion", "CIP": "Contact ion pair", "AGG": "Aggregate", "bound": "Coordinated"}[
            name
        ]
        ax.fill_between(nu[win], _gauss(nu[win], amp, cen, fwhm) / scale, color=col, alpha=0.25, lw=0,
                        label=label)  # fmt: skip
    ax.set(xlabel="Raman shift (cm$^{-1}$)", ylabel="Normalized intensity")
    _inner_legend(ax, loc="upper right", fontsize=5.5)
    if own:
        save(fig, out, "fig_raman_example")


def fig_raman_md(results: Path, out: Path, ax=None) -> None:
    """Raman coordinated-anion fraction vs MD Li<-anion coordination (linked OEDB compositions)."""
    from scipy.stats import spearmanr

    df = pd.read_csv(results / "raman" / "raman_fractions.csv")
    df = df[~df.solvent_overlap & df.CN_anion_MD.notna() & df.f_bound.notna()]
    # TFSI and PF6 shift too little on coordination to resolve; they are reported in the SI only
    df = df[df.anion.isin(["FSI", "BF4", "ClO4"])]
    fig, ax, own = _canvas(ax, SINGLE)
    for a in ["FSI", "BF4", "ClO4"]:
        g = df[df.anion == a]
        rho = spearmanr(g.CN_anion_MD, g.f_bound)[0]
        ax.scatter(g.CN_anion_MD, g.f_bound, s=7, color=ANION_COLORS.get(a, "k"), linewidths=0,
                   label=f"{_anion_label(a)} ({rho:.2f})")  # fmt: skip
    ax.set(xlabel="MD coordination, Li$^+$ \u2190 anion", ylabel="Raman coordinated-anion fraction",
           ylim=(0, 1.02))  # fmt: skip
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), borderaxespad=0, handletextpad=0.1,
              columnspacing=0.6, ncol=3, markerscale=1.3)  # fmt: skip
    if own:
        save(fig, out, "fig_raman_md")


def fig_raman_predictor(results: Path, out: Path, ax=None) -> None:
    """Agreement with Raman: MD, and the OEDB predictor inside vs outside OEDB's solvents.

    Black diamonds: ranking by salt concentration alone (the baseline a predictor must beat).
    TFSI and PF6 are omitted: their bands shift too little on coordination to resolve.
    """
    s = pd.read_csv(results / "raman" / "raman_summary.csv")
    s = s[s.group == "pooled"]
    if not s.comparison.str.startswith("OEDB predictor").any():
        raise FileNotFoundError(2, "no predictor results (run elecgnn raman --predict)",
                                str(results / "raman"))  # fmt: skip
    inside, outside = "solvents in OEDB MD", "solvents absent from OEDB MD"
    arms = [("MD (linked OEDB compositions)", None, "MD, OEDB", GREY),
            (f"OEDB predictor, {inside}", f"Molality only, {inside}", "Predictor, in OEDB", BLUE),
            (f"OEDB predictor, {outside}", f"Molality only, {outside}", "Predictor, outside OEDB",
             VERMILION)]  # fmt: skip
    anions = [a for a in ["FSI", "BF4", "ClO4", "TfO"] if a in set(s.anion)]
    fig, ax, own = _canvas(ax, SINGLE)
    w = 0.26
    for k, (comp, base, label, col) in enumerate(arms):
        x = np.arange(len(anions)) + (k - 1) * w
        vals = [s[(s.anion == a) & (s.comparison == comp)].spearman.mean() for a in anions]
        ax.bar(x, vals, w, color=col, label=label)
        if base is not None and s.comparison.eq(base).any():
            b = [s[(s.anion == a) & (s.comparison == base)].spearman.mean() for a in anions]
            ax.scatter(x, b, marker="D", s=10, c="k", zorder=3,
                       label="Conc. only" if k == 1 else None)  # fmt: skip
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(anions)), [_anion_label(a) for a in anions])
    ax.set(ylabel="Spearman \u03c1 with Raman coordinated fraction", ylim=(min(-0.45, ax.get_ylim()[0]), 1))
    handles, labels = ax.get_legend_handles_labels()
    keys = ["MD, OEDB", "Conc. only", "Predictor, in OEDB", "Predictor, outside OEDB"]
    order = [labels.index(k) for k in keys]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right",
              bbox_to_anchor=(1.0, 1.02), borderaxespad=0, ncol=2, handletextpad=0.3, columnspacing=0.6,
              handlelength=1.2)  # fmt: skip
    if own:
        save(fig, out, "fig_raman_predictor")


def figure7_raman(results: Path, data: Path, out: Path) -> None:
    """Fig. 7  Experimental Raman check of MD solvation structure and of OEDB-based predictions."""
    fig = plt.figure(figsize=(6.7, 2.75))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.995, top=0.83, wspace=0.36, width_ratios=[1, 1.12, 1.05])
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    fig_raman_example(data, out, ax=a)
    fig_raman_md(results, out, ax=b)
    fig_raman_predictor(results, out, ax=c)
    _letters([a, b, c])
    save(fig, out, "Figure7_raman_validation")


def fig_si_raman_fits(data: Path, out: Path) -> None:
    """(SI) One representative fitted anion band per anion (about 2 mol/kg, first solvent without overlap)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from elecgnn.raman import COMPONENTS, _gauss, fit_components, load_spectra, subtract_solvent

    meta, nu, S = load_spectra(data / "oedb-experimental-v2026-05-11.csv")
    pure = {r.solvent: S[i] for i, r in meta.iterrows() if r.molality == 0}
    picks = {"FSI": "DMC", "TFSI": "DMC", "PF6": "DMC", "BF4": "AN", "ClO4": "AN", "TfO": "AN"}
    fig, axes = plt.subplots(2, 3, figsize=(6.7, 4.4))
    for ax, (anion, solvent) in zip(axes.flat, picks.items(), strict=True):
        sel = meta[(meta.anion == anion) & (meta.solvent == solvent) & (meta.molality > 0)]
        if sel.empty or solvent not in pure:
            ax.set_axis_off()
            continue
        i = sel.molality.sub(2.0).abs().idxmin()
        lo, hi, comps = COMPONENTS[anion]
        r, _ = subtract_solvent(nu, S[i], pure[solvent], lo, hi)
        w = (nu >= lo) & (nu <= hi)
        y = np.clip(r[w], 0, None)
        fit = fit_components(nu[w], y, comps)
        scale = y.max()
        ax.plot(nu[w], y / scale, "k-", lw=1.0, label="Data")
        total = np.zeros(w.sum())
        for (name, _, _), col in zip(comps, [GREEN, ORANGE, VERMILION], strict=False):
            amp = fit[f"area_{name}"] / fit[f"fwhm_{name}"]
            g = _gauss(nu[w], amp, fit[f"centre_{name}"], fit[f"fwhm_{name}"])
            total += g
            ax.fill_between(nu[w], g / scale, color=col, alpha=0.3, lw=0, label=name)
        ax.plot(nu[w], total / scale, "--", color=BLUE, lw=0.8, label="Fit")
        ax.set_title(f"Li{_anion_label(anion)} in {solvent}")
        ax.text(0.03, 0.96, f"{meta.molality[i]:g} mol kg$^{{-1}}$\nf$_{{coord}}$ = {fit['f_bound']:.2f}",
                transform=ax.transAxes, va="top", fontsize=5.5)  # fmt: skip
        ax.set(xlabel="Raman shift (cm$^{-1}$)", yticks=[])
        # the TfO band fills the upper right, and the upper left holds the concentration note
        ax.legend(loc="center left" if anion == "TfO" else "upper right", fontsize=5)
    save(fig, out, "fig_si_raman_fits")


# ── Composite figures for the main text (Results and Discussion order) ──────────────────────────
def _letters(axes) -> None:
    for ax, letter in zip(axes, "abcdefghij", strict=False):
        _label(ax, letter)


def figure1_data(results: Path, data: Path, out: Path, schematic: Path | None = None) -> None:
    """Fig. 1  Study overview (schematic, drawn separately) with chemical coverage and MD consistency."""
    fig = plt.figure(figsize=(6.7, 6.9))
    top = fig.add_subplot(fig.add_gridspec(1, 1, left=0.0, right=1.0, top=0.975, bottom=0.535)[0])
    top.axis("off")
    if schematic is not None and not Path(schematic).exists():
        raise FileNotFoundError(2, "schematic image not found", str(schematic))
    if schematic is not None:
        top.imshow(plt.imread(schematic), interpolation="lanczos")
    else:
        print("WARNING: Figure 1a drawn without the schematic (pass --schematic path/to/Figure1a.png)")
        top.add_patch(plt.Rectangle((0, 0), 1, 1, transform=top.transAxes, fill=False, ls="--", lw=0.75,
                                    edgecolor=GREY))  # fmt: skip
        top.text(0.5, 0.5, "Schematic (PowerPoint) \u2014 pass --schematic Figure1a.png", ha="center",
                 va="center", transform=top.transAxes, color=GREY)  # fmt: skip
    low = fig.add_gridspec(1, 2, left=0.235, right=0.985, top=0.47, bottom=0.075, wspace=0.42,
                           width_ratios=[0.85, 1.2])  # fmt: skip
    b, c = fig.add_subplot(low[0, 0]), fig.add_subplot(low[0, 1])
    fig_coverage_matrix(data, out, ax=b)
    fig_consistency(data, out, ax=c)
    b.set_xticks(b.get_xticks(), ["In-house\nMD", "OEDB", "Experimental\nCE"])
    for ax in (b, c):  # panels (b) and (c) are read at full page width, so use larger text
        ax.tick_params(labelsize=8.5)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        for t in ax.texts:
            t.set_fontsize(8.5)
        if ax.get_legend() is not None:
            for t in ax.get_legend().get_texts():
                t.set_fontsize(8)
    _letters_from([b, c], "bc")
    fig.text(0.005, 0.985, "(a)", fontsize=8, fontweight="bold", va="top")
    save(fig, out, "Figure1_overview_and_coverage")


def _letters_from(axes, letters: str) -> None:
    for ax, letter in zip(axes, letters, strict=True):
        _label(ax, letter)


def figure2_extrapolation(results: Path, data: Path, out: Path) -> None:
    """Fig. 2  Coverage decides extrapolation: error vs similarity, ether learning curve, parity."""
    fig = plt.figure(figsize=(6.7, 5.0))
    gs = fig.add_gridspec(2, 6, wspace=1.2, hspace=0.45)
    a, b = fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])
    c = fig.add_subplot(gs[1, 0:2])
    d, e = fig.add_subplot(gs[1, 2:4], sharex=c, sharey=c), fig.add_subplot(gs[1, 4:6], sharex=c, sharey=c)
    fig_coverage(results, out, ax=a)
    fig_learning_curve(results, out, ax=b)
    fig_parity_failure(results, data, out, axes=np.array([c, d, e]))
    for ax in (d, e):
        ax.tick_params(labelleft=False)
    _letters([a, b, c, d, e])
    save(fig, out, "Figure2_coverage_and_extrapolation")


def figure3_information(results: Path, data: Path, out: Path) -> None:
    """Fig. 3  Physical descriptors restore extrapolation; external MD and pretraining do not."""
    fig = plt.figure(figsize=(6.7, 7.3))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 2.3], hspace=0.34)
    a = fig.add_subplot(outer[0])
    low = outer[1].subgridspec(2, 2, wspace=0.55, hspace=0.55)
    b, c = fig.add_subplot(low[0, 0]), fig.add_subplot(low[0, 1])
    d, e = fig.add_subplot(low[1, 0]), fig.add_subplot(low[1, 1])
    fig_graph_vs_descriptors(results, out, ax=a)
    fig_pretraining_targets(results, out, ax=b)
    fig_oedb_features(results, out, ax=c)
    fig_shap_extrapolation(results, out, axes=np.array([d, e]))
    _letters([a, b, c, d, e])
    save(fig, out, "Figure3_physical_information")


def figure4_benchmarking(results: Path, data: Path, out: Path) -> None:
    """Fig. 4  Benchmarking practice: early stopping, seeds, architecture per class, external CE test."""
    fig = plt.figure(figsize=(6.7, 5.0))
    gs = fig.add_gridspec(2, 2, wspace=0.7, hspace=0.45)
    a, b, c, d = (fig.add_subplot(gs[i // 2, i % 2]) for i in range(4))
    fig_early_stopping(results, out, ax=a)
    fig_seed_variance(results, out, ax=b)
    fig_gat_gcn_classes(results, data, out, ax=c)
    fig_external_lce(results, out, ax=d)
    _letters([a, b, c, d])
    save(fig, out, "Figure4_benchmarking_practice")


def figure5_reliability(results: Path, data: Path, out: Path) -> None:
    """Fig. 5  Interpretation, screening value and when to trust a prediction."""
    fig = plt.figure(figsize=(6.7, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.42, top=0.9)
    a, b, c, d = (fig.add_subplot(gs[i // 2, i % 2]) for i in range(4))
    fig_attention(results, out, ax=a)
    fig_screening(results, out, ax=b)
    fig_uncertainty(results, data, out, axes=np.array([c, d]))
    _letters([a, b, c, d])
    save(fig, out, "Figure5_interpretation_and_reliability")


def figure6_embeddings(results: Path, data: Path, out: Path) -> None:
    """Fig. 6  Learned embeddings against a random-weight control."""
    fig = plt.figure(figsize=(6.7, 5.0))
    axes, side = _tsne_layout(fig)
    fig_tsne_control(results, out, axes=axes, side=side)
    _letters(axes.flatten())
    save(fig, out, "Figure6_embeddings")


COMPOSITES = [figure1_data, figure2_extrapolation, figure3_information, figure4_benchmarking,
              figure5_reliability, figure6_embeddings, figure7_raman]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--schematic", type=Path, help="PNG of the Figure 1a schematic exported from PowerPoint")
    ap.add_argument("--font-dir", type=Path, default=None,
                    help="directory containing arial.ttf (else a similar sans-serif)")  # fmt: skip
    a = ap.parse_args()
    setup_style(a.font_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    panels = [
        ("consistency", lambda: fig_consistency(a.data, a.out)),
        ("coverage", lambda: fig_coverage(a.results, a.out)),
        ("learning curve", lambda: fig_learning_curve(a.results, a.out)),
        ("GAT vs GCN per class", lambda: fig_gat_gcn_classes(a.results, a.data, a.out)),
        ("seed variance", lambda: fig_seed_variance(a.results, a.out)),
        ("early stopping", lambda: fig_early_stopping(a.results, a.out)),
        ("OEDB features", lambda: fig_oedb_features(a.results, a.out)),
        ("external LCE test", lambda: fig_external_lce(a.results, a.out)),
        ("attention", lambda: fig_attention(a.results, a.out)),
        ("graph vs descriptors", lambda: fig_graph_vs_descriptors(a.results, a.out)),
        ("pretraining targets", lambda: fig_pretraining_targets(a.results, a.out)),
        ("A chemical space", lambda: fig_chemical_space(a.data, a.out)),
        ("B parity failure", lambda: fig_parity_failure(a.results, a.data, a.out)),
        ("C screening", lambda: fig_screening(a.results, a.out)),
        ("D t-SNE control", lambda: fig_tsne_control(a.results, a.out)),
        ("E SHAP extrapolation", lambda: fig_shap_extrapolation(a.results, a.out)),
        ("SI descriptor benchmark", lambda: fig_descriptor_benchmark(a.results, a.out)),
        ("F uncertainty", lambda: fig_uncertainty(a.results, a.data, a.out)),
        ("SI distributions", lambda: fig_si_distributions(a.data, a.out)),
        ("SI correlation", lambda: fig_si_correlation(a.data, a.out)),
        ("SI trade-off", lambda: fig_si_tradeoff(a.data, a.out)),
        ("coverage matrix", lambda: fig_coverage_matrix(a.data, a.out)),
        ("Raman example", lambda: fig_raman_example(a.data, a.out)),
        ("Raman vs MD", lambda: fig_raman_md(a.results, a.out)),
        ("Raman vs predictor", lambda: fig_raman_predictor(a.results, a.out)),
        ("SI Raman fits", lambda: fig_si_raman_fits(a.data, a.out)),
    ]
    for name, fn in panels:
        try:
            fn()
        except FileNotFoundError as err:
            print(f"skipped {name}: missing {Path(err.filename).as_posix() if err.filename else err}")
    for fn in COMPOSITES:  # grouped figures for the main text
        try:
            if fn is figure1_data:
                fn(a.results, a.data, a.out, schematic=a.schematic)
            else:
                fn(a.results, a.data, a.out)
        except FileNotFoundError as err:
            plt.close("all")
            print(f"skipped {fn.__name__}: missing {Path(err.filename).as_posix() if err.filename else err}")


if __name__ == "__main__":
    main()
