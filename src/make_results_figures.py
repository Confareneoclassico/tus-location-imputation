#!/usr/bin/env python3
"""Figures and LaTeX tables for the results section, built from the aggregate outputs of
location_imputation.py (the files in imputation_out/ that can be shared) of version 4.1. Some
branches below also read the outputs of earlier versions of the pipeline; comments mark them.
Per-place scores of countries are drawn only where the pipeline published them, with enough
respondents behind them.

If the output folder holds paper.tex, the generated tables are also copied into it, between
the lines '% <<< generated: tab_name >>>' and '% <<< end of tab_name >>>', so that the
manuscript stays a single file. Edit the tables through this script, not in paper.tex.

Usage:
    python make_results_figures.py path/to/imputation_out [output_folder]

Writes figures/*.pdf and tables/*.tex into the output folder (default: next to this script).
Only aggregate files are read; the PRIVATE_ microdata files are never touched.
"""
import glob
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --------------------------------------------------------------------------- style
# Latin Modern Roman, the font of the manuscript, if a TeX installation provides it
for pattern in ("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf",
                "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
                "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
                "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
                "C:/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf"):
    for f in glob.glob(pattern):
        font_manager.fontManager.addfont(f)
SERIF = "LM Roman 10" if any(f.name == "LM Roman 10" for f in font_manager.fontManager.ttflist) else "serif"
plt.rcParams.update({
    "font.family": "serif", "font.serif": [SERIF, "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
WIDTH = 6.3                      # inches: the manuscript's text width (A4, 2 cm and 3 cm margins)
BLUE, ORANGE, GREY, LIGHT = "#0072B2", "#D55E00", "#7F7F7F", "#BDBDBD"

FULL_LABEL = {
    "home": "All at home",
    "activity_modal": "Activity, modal",
    "slot_time": "Slot, time",
    "slot_time_age": "Slot, time, age",
    "episode_time_age": "Episode, time, age",
    "markov_time": "Markov, time",
    "markov_time_age": "Markov, time, age",
    "markov_time_age_day": "Markov, time, age, day",
    "markov_time_age_modal": "Markov, time, age, modal",
    "markov_time_age_kappa1": r"Markov, time, age, $\alpha = 1$",
    "markov_time_age_kappa20": r"Markov, time, age, $\alpha = 20$",
}
MAIN, MODAL = "markov_time_age", "markov_time_age_modal"
PLACES = ["home", "second home", "workplace or school", "other people's home", "restaurant, cafe or pub",
          "shops and services", "hotel or camping", "other or unspecified place", "in transit"]
PLACE_COLOURS = {
    "home": "#0072B2", "second home": "#56B4E9", "other people's home": "#CC79A7",
    "hotel or camping": "#009E73", "workplace or school": "#E69F00", "shops and services": "#F0E442",
    "restaurant, cafe or pub": "#D55E00", "other or unspecified place": "#C8C8C8",
    "in transit": "#8C8C8C", "travelling": "#4D4D4D", "transport": "#4D4D4D",
}
# Places at which some country has few respondents: their per-country scores are not drawn from
# outputs older than version 3.1, which carry no respondent counts (Eurostat publication rules)
RARE_PLACES = ("second home", "hotel or camping", "in transit")
STACK = ["home", "second home", "other people's home", "hotel or camping", "workplace or school",
         "shops and services", "restaurant, cafe or pub", "other or unspecified place", "transport"]
STACK_V3 = STACK[:-1] + ["in transit", "travelling"]     # outputs before version 4.0


def stack_of(labels):
    """Location categories in display order for the outputs at hand (version 4.0 or older)."""
    return STACK if "transport" in set(labels) else STACK_V3


HOUR_TICKS = [0, 36, 72, 108, 143]          # 04:00, 10:00, 16:00, 22:00, 03:50
HOUR_LABELS = ["04:00", "10:00", "16:00", "22:00", "04:00"]

# Global sensitivity analysis (version 3 outputs)
RANGE_LO, RANGE_HI = "q05", "q95"           # range reported for the German estimates: 90% of the combinations
GSA_A_FACTORS = [("unit", "Unit of imputation"), ("time", "Time of day"), ("age", "Age"),
                 ("day_type", "Day type"), ("assignment", "Assignment"), ("alpha", r"$\alpha$")]
GSA_A_METRICS = [("accuracy", "Accuracy"), ("kappa", r"$\kappa$"), ("home_share_mae_pp", "Home share error"),
                 ("hourly_tvd", "Hourly TVD"), ("changes_error_pct", "Change error")]
GSA_B_FACTORS_V41 = [("unit", "Unit of imputation"), ("time", "Time of day"), ("age", "Age"),
                     ("day_type", "Day type"), ("assignment", "Assignment"), ("alpha", r"$\alpha$"),
                     ("code_13", "Code 13"), ("draws", "Random draws"), ("transfer_error", "Transfer error")]
GSA_B_FACTORS = [("transfer_error", "Transfer error"), ("alpha", r"$\alpha$"), ("age", "Age"),
                 ("day_type", "Day type"), ("code_13", "Code 13"), ("draws", "Random draws")]
GSA_B_FACTORS_V3 = [("transfer_error", "Transfer error"), ("alpha", r"$\alpha$"), ("age", "Age"),
                    ("day_type", "Day type"), ("in_transit", "In-transit draws"), ("code_13", "Code 13")]
GSA_B_OUTPUTS = [("home", "Home"), ("workplace or school", "Workplace or school"),
                 ("other or unspecified place", "Other or unspecified"),
                 ("other people's home", "Other people's home"), ("changes per diary", "Changes per diary")]
# Rows of the tables of indices (appendix): every metric of part A and every output of part B
GSA_A_TABLE_FACTORS = [("unit", "Unit"), ("time", "Time"), ("age", "Age"), ("day_type", "Day type"),
                       ("assignment", "Assignment"), ("alpha", r"$\alpha$")]
GSA_A_TABLE_ROWS = [("accuracy", "Accuracy"), ("kappa", r"$\kappa$"), ("macro_f1", "Macro-F1"),
                    ("home_share_mae_pp", "Home share error"), ("home_share_bias_pp", "Home share bias"),
                    ("hourly_tvd", "Hourly TVD"), ("changes_error_pct", "Change error")]
GSA_B_TABLE_FACTORS_V41 = [("unit", "Unit"), ("time", "Time"), ("age", "Age"), ("day_type", r"\makecell{Day\\type}"),
                           ("assignment", r"\makecell{Assign-\\ment}"), ("alpha", r"$\alpha$"),
                           ("code_13", r"\makecell{Code\\13}"), ("draws", r"\makecell{Random\\draws}"),
                           ("transfer_error", r"\makecell{Transfer\\error}")]
GSA_B_TABLE_FACTORS = [("transfer_error", r"\makecell{Transfer\\error}"), ("alpha", r"$\alpha$"), ("age", "Age"),
                       ("day_type", r"\makecell{Day\\type}"), ("code_13", r"\makecell{Code\\13}"),
                       ("draws", r"\makecell{Random\\draws}")]
GSA_B_TABLE_FACTORS_V3 = [("transfer_error", r"\makecell{Transfer\\error}"), ("alpha", r"$\alpha$"), ("age", "Age"),
                          ("day_type", r"\makecell{Day\\type}"), ("in_transit", r"\makecell{In-transit\\draws}"),
                          ("code_13", r"\makecell{Code\\13}")]


def b_factors(idx, table=False):
    """Factors of part B for the outputs at hand: the method factors of part A from version 4.1,
    random draws from version 4.0, in-transit draws before."""
    present = set(idx.factor) if idx is not None else set()
    if "unit" in present:
        return GSA_B_TABLE_FACTORS_V41 if table else GSA_B_FACTORS_V41
    if "draws" in present:
        return GSA_B_TABLE_FACTORS if table else GSA_B_FACTORS
    return GSA_B_TABLE_FACTORS_V3 if table else GSA_B_FACTORS_V3
GSA_B_TABLE_ROWS = [("home", "Home"), ("second home", "Second home"), ("workplace or school", "Workplace or school"),
                    ("other people's home", "Other people's home"),
                    ("restaurant, cafe or pub", "Restaurant, cafe or pub"),
                    ("shops and services", "Shops and services"), ("hotel or camping", "Hotel or camping"),
                    ("other or unspecified place", "Other or unspecified place"), ("in transit", "In transit"),
                    ("changes per diary", "Changes per diary")]
GSA_COLOURS = {"transfer_error": "#0072B2", "alpha": "#E69F00", "age": "#009E73", "day_type": "#CC79A7",
               "in_transit": "#56B4E9", "draws": "#56B4E9", "code_13": "#D55E00", "interactions": "#D9D9D9",
               "unit": "#F0E442", "time": "#44AA99", "assignment": "#882255"}
# Evaluation metrics of part A and the direction in which each improves (1 higher, -1 lower, 0 closer to zero)
GSA_A_DIRECTIONS = dict(accuracy=1, kappa=1, macro_f1=1, home_share_mae_pp=-1, home_share_bias_pp=0,
                        hourly_tvd=-1, changes_error_pct=0)


def save(fig, path):
    """Save a figure as PDF. If Ghostscript is installed, re-embed the fonts so that the
    file passes strict PDF checks (matplotlib embeds OpenType fonts with a TrueType tag)."""
    fig.savefig(path)
    plt.close(fig)
    gs = shutil.which("gs") or shutil.which("gswin64c")
    if gs:
        tmp = path.with_suffix(".gs.pdf")
        done = subprocess.run([gs, "-q", "-o", str(tmp), "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/prepress",
                               "-dEmbedAllFonts=true", "-dSubsetFonts=true", str(path)], capture_output=True)
        if done.returncode == 0 and tmp.exists():
            tmp.replace(path)
        elif tmp.exists():
            tmp.unlink()


def tex_label(s):
    """Variant labels are already valid LaTeX."""
    return s


def hour_axis(ax, compact=False, end_label=True):
    """Clock-time axis for the 144 ten-minute slots of a HETUS diary day (04:00 to 04:00)."""
    ax.set_xlim(0, 144)
    ax.set_xticks([0, 36, 72, 108, 144])
    labels = ["04", "10", "16", "22", ""] if compact else list(HOUR_LABELS)
    if not end_label:
        labels[-1] = ""
    ax.set_xticklabels(labels)


# --------------------------------------------------------------------------- data
def load(results):
    r = Path(results)
    overall = pd.read_csv(r / "loco_overall.csv")
    summary = pd.read_csv(r / "loco_summary.csv")
    by_class = pd.read_csv(r / "loco_by_class.csv")
    home = pd.read_csv(r / "loco_home_by_slot.csv")
    de_slot = pd.read_csv(r / "germany_shares_by_slot.csv")
    de_day = pd.read_csv(r / "germany_hours_per_day.csv")
    coding = overall.groupby("country").coding.first()
    by_class["coding"] = by_class.country.map(coding)
    home["coding"] = home.country.map(coding)
    home["error_pp"] = 100 * (home.imputed - home.observed)
    d = dict(overall=overall, summary=summary, by_class=by_class, home=home,
             de_slot=de_slot, de_day=de_day, coding=coding)
    for name in ("loco_by_class_means", "loco_hours_per_day", "loco_calibration_summary",
                 "loco_repeated_imputations_summary", "loco_intervals_home_by_slot",
                 "germany_sensitivity", "germany_changes_per_diary",
                 "gsa_loco", "gsa_loco_indices", "gsa_loco_combinations", "gsa_germany_indices",
                 "gsa_germany_uncertainty", "gsa_germany_uncertainty_home_by_slot", "gsa_germany_indices_home_by_slot",
                 "gsa_germany_uncertainty_pareto", "gsa_germany_uncertainty_home_by_slot_pareto",
                 "gsa_germany_combinations"):
        f = r / f"{name}.csv"
        d[name] = pd.read_csv(f) if f.is_file() else None
    if d["gsa_loco_combinations"] is None and d["gsa_loco"] is not None:     # outputs before version 4.1
        m = d["gsa_loco"].groupby([f for f, _ in GSA_A_FACTORS])[list(GSA_A_DIRECTIONS)].mean().reset_index()
        m["pareto_optimal"] = pareto_optimal(m)
        d["gsa_loco_combinations"] = m
    d["transfer"] = transfer_errors(d)
    return d


def pareto_optimal(means):
    """Combinations that no other one matches or beats on every metric of GSA_A_DIRECTIONS while
    beating it on one (as in location_imputation.py)."""
    X = np.column_stack([means[m].to_numpy(float) * s if s else -np.abs(means[m].to_numpy(float))
                         for m, s in GSA_A_DIRECTIONS.items()])
    geq = (X[:, None, :] >= X[None, :, :]).all(2)
    gt = (X[:, None, :] > X[None, :, :]).any(2)
    return ~(geq & gt).any(0)


def group_rows(frame, group):
    """Rows of one day-type group (version 3 files by slot have a group column)."""
    return frame[frame.group == group] if "group" in frame.columns else frame


# --------------------------------------------------------------------------- transfer errors
def coverage(R, q):
    """Mean over countries of the share of units within q."""
    return float(np.mean([np.mean(r <= q) for r in R if len(r)]))


def smallest_bound(R, target=0.95):
    """Smallest q such that, on average over countries, the target share of units is within q."""
    cand = np.unique(np.concatenate(R + [np.zeros(1)]))
    lo, hi = 0, len(cand) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if coverage(R, cand[mid]) >= target:
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo])


def transfer_errors(d):
    """What the held-out countries say about the error to expect for Germany.

    home_q: constant half-width (share) that covers 95% of the held-out slots on average, from
    the mean of the fixed-pool imputations (version 2) or the single imputation (version 1);
    home_cv: its coverage when fitted on the other countries only. hours: per place, the
    smallest and largest error (imputed minus recorded hours per day) among the 13 donors.
    changes: the same for location changes per diary."""
    iv = d["loco_intervals_home_by_slot"]
    if iv is not None:
        if "gamma" in iv.columns:          # versions 2 to 3.1: keep the fixed donor pool
            iv = iv[np.isinf(iv.gamma.astype(float))]
        pairs = [(g.recorded.to_numpy(), g["mean"].to_numpy()) for _, g in iv.groupby("country", sort=False)]
    else:
        h = d["home"][d["home"].variant == MAIN]
        pairs = [(g.observed.to_numpy(), g.imputed.to_numpy()) for _, g in h.groupby("country", sort=False)]
    R = [np.abs(rec - imp) for rec, imp in pairs]
    q = smallest_bound(R)
    cv = [float(np.mean(R[j] <= smallest_bound(R[:j] + R[j + 1:]))) for j in range(len(R))]
    out = dict(home_q=q, home_cv=float(np.mean(cv)), home_cv_by_country=cv, countries=len(R))
    lh = d["loco_hours_per_day"]
    if lh is not None:
        m = lh[(lh.variant == MAIN) & (lh.coding == "full")].copy()
        m["err"] = m.imputed_hours - m.recorded_hours
        out["hours"] = m.groupby("place").err.agg(["min", "max", "mean"]).rename(
            columns={"min": "err_min", "max": "err_max", "mean": "err_mean"})
    o = d["overall"]
    oo = o[(o.variant == MAIN) & (o.coding == "full")]
    e = oo.changes_imputed - oo.changes_observed
    out["changes"] = dict(err_min=float(e.min()), err_max=float(e.max()))
    return out


# --------------------------------------------------------------------------- figures
def fig_variants(d, out):
    """The seven evaluation metrics of Table 1 for each variant: the held-out countries and their
    mean (version 4.1: all seven, in a grid of two rows)."""
    o = metric_means(d)[1]
    o = o[o.coding == "full"].copy()
    o["change_ratio"] = o.changes_imputed / o.changes_observed
    rows = [("Random draw", ["slot_time", "slot_time_age", "episode_time_age",
                             "markov_time", MAIN, "markov_time_age_day"]),
            ("Modal assignment", ["activity_modal", MODAL])]
    order, ypos, headers, y = [], [], [], 0
    for group, variants in rows:
        headers.append((group, y)); y += 1
        for v in variants:
            order.append(v); ypos.append(y); y += 1
        y += 0.3
    ypos = np.array(ypos)
    panels = [("accuracy", "(a) Accuracy", None, None),
              ("kappa", r"(b) Cohen's $\kappa$", None, None),
              ("macro_f1", "(c) Macro-F1", None, None),
              ("home_share_mae_pp", "(d) Home share error (pp)", None, None),
              ("home_share_bias_pp", "(e) Home share bias (pp)", None, 0.0),
              ("hourly_tvd", "(f) Hourly TVD", None, None),
              ("change_ratio", "(g) Changes, imputed/recorded", "log", 1.0)]
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 5.0), sharey=True)
    rng = np.random.default_rng(1)
    for ax, (metric, title, scale, ref) in zip(axes.ravel(), panels):
        for v, yy in zip(order, ypos):
            vals = o.loc[o.variant == v, metric].to_numpy()
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(vals, yy + jitter, s=6, color=LIGHT, linewidths=0, zorder=2)
            colour = BLUE if v == MAIN else (ORANGE if v == MODAL else "#333333")
            ax.scatter([vals.mean()], [yy], s=22, color=colour, marker="D" if v in (MAIN, MODAL) else "o",
                       zorder=3, linewidths=0)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(0.75, 10)
            ax.set_xticks([1, 2, 4, 8])
            ax.set_xticklabels(["1", "2", "4", "8"])
            ax.minorticks_off()
        if ref is not None:
            ax.axvline(ref, color=GREY, lw=0.6, ls="--", zorder=1)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
        ax.tick_params(axis="y", length=0)
    ticks = list(ypos) + [yy for _, yy in headers]
    labels = [FULL_LABEL[v] for v in order] + [g for g, _ in headers]
    for row in range(2):
        axes[row, 0].set_yticks(ticks)
        axes[row, 0].set_yticklabels(labels)
        for lab in axes[row, 0].get_yticklabels()[len(order):]:
            lab.set_color(GREY)
    axes[0, 0].set_ylim(y - 0.6, -0.6)
    legend_ax = axes[1, 3]
    legend_ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=LIGHT, markersize=3, label="held-out country"),
               Line2D([], [], marker="o", ls="", color="#333333", markersize=4, label="mean of 13 countries"),
               Line2D([], [], marker="D", ls="", color=BLUE, markersize=4, label="main specification"),
               Line2D([], [], marker="D", ls="", color=ORANGE, markersize=4, label="its modal version")]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, bbox_to_anchor=(0.0, 0.5))
    fig.subplots_adjust(wspace=0.12, hspace=0.35)
    save(fig, out / "fig_variants.pdf")


def fig_home_error(d, out):
    """Imputed minus observed home share by time of day, each held-out country."""
    h = d["home"]
    o = d["overall"]
    countries = sorted(h.country.unique())
    fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 5.4), sharex=True, sharey=True)
    q = 100 * d["transfer"]["home_q"]
    for ax, c in zip(axes.flat, countries):
        ax.axhspan(-q, q, color="#F0F0F0", lw=0, zorder=0)
        ax.axhline(0, color=GREY, lw=0.5)
        for v, colour, ls in ((MODAL, ORANGE, "--"), (MAIN, BLUE, "-")):
            x = h[(h.country == c) & (h.variant == v)].sort_values("slot")
            ax.plot(x.slot, x.error_pp, color=colour, ls=ls, lw=0.9)
        mae = o.loc[(o.country == c) & (o.variant == MAIN), "home_share_mae_pp"].iloc[0]
        partial = d["coding"][c] == "partial"
        ax.set_title(f"{c}{' (partial coding)' if partial else ''}", loc="left", fontsize=7.5, pad=2)
        ax.text(0.98, 0.04, f"MAE {mae:.1f} pp", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=BLUE, bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))
        hour_axis(ax, compact=True)
        ax.set_ylim(-7.5, 12)
        ax.set_yticks([-5, 0, 5, 10])
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    fig.supxlabel("Hour of the diary day", fontsize=8, y=0.045)
    fig.supylabel("Imputed minus recorded share at home (pp)", fontsize=8, x=0.045)
    handles = [Line2D([], [], color=BLUE, lw=1, label="Markov, time, age (main specification, random draw)"),
               Line2D([], [], color=ORANGE, lw=1, ls="--", label="Markov, time, age, modal"),
               Patch(facecolor="#F0F0F0", edgecolor="none", label=f"$\\pm${q:.1f} pp")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.subplots_adjust(hspace=0.4, wspace=0.08, top=0.94)
    save(fig, out / "fig_home_error.pdf")


def fig_places(d, out):
    """Time, recall and precision by place, main specification, full-coding countries.
    Panel (a) shows weighted hours per day when loco_hours_per_day.csv (version 2) is present,
    and the unweighted share of scored slots otherwise."""
    b = d["by_class"]
    b = b[b.coding == "full"]
    main = b[b.variant == MAIN]
    modal = b[b.variant == MODAL]
    lh = d["loco_hours_per_day"]
    if lh is not None:
        lh = lh[(lh.coding == "full") & lh.place.isin(PLACES)]
        lm, lmod = lh[lh.variant == MAIN], lh[lh.variant == MODAL]
        rec_each = lm.rename(columns={"recorded_hours": "value"})[["place", "value"]]
        obs = lm.groupby("place").recorded_hours.mean()
        imp = lm.groupby("place").imputed_hours.mean()
        imp_modal = lmod.groupby("place").imputed_hours.mean()
        title = "(a) Hours per day"
    else:
        rec_each = main.assign(value=100 * main.observed_share)[["place", "value"]]
        obs = main.groupby("place").observed_share.mean() * 100
        imp = main.groupby("place").imputed_share.mean() * 100
        imp_modal = modal.groupby("place").imputed_share.mean() * 100
        title = "(a) Share of scored time (%)"
    order = obs.reindex([p for p in PLACES if p in obs.index]).sort_values(ascending=False).index.tolist()
    obs, imp, imp_modal = obs.reindex(order), imp.reindex(order), imp_modal.reindex(order)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    rng = np.random.default_rng(2)
    ax = axes[0]
    for yy, p in zip(y, order):
        vals = rec_each.loc[rec_each.place == p, "value"].dropna().to_numpy()
        vals = vals[vals > 0]                                   # zero values cannot be drawn on a log axis
        ax.scatter(vals, yy + rng.uniform(-0.18, 0.18, len(vals)), s=5, color=LIGHT, linewidths=0, zorder=2)
    ax.scatter(obs, y, s=24, facecolors="white", edgecolors="#333333", linewidths=0.8, zorder=3, label="observed")
    ax.scatter(imp, y, s=14, color=BLUE, zorder=4, label="imputed, random draw")
    shown = (imp_modal > 0).to_numpy()
    ax.scatter(imp_modal[shown], y[shown], s=14, color=ORANGE, marker="D", zorder=4, label="imputed, modal")
    ax.set_xscale("log")
    lowest = min(v for v in np.concatenate([rec_each.value.dropna().to_numpy(),
                                            imp.to_numpy(), imp_modal.to_numpy()]) if v > 0)
    ax.set_xlim(10 ** np.floor(np.log10(lowest) - 0.2), 1.5 * max(obs.max(), imp_modal.max()))
    ax.set_xticks([t for t in (0.001, 0.01, 0.1, 1, 10, 100) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_title(title, loc="left", fontsize=7.5)
    handles = [Line2D([], [], marker="o", ls="", color=LIGHT, markersize=3, label="each held-out country"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#333333",
                      markersize=4.5, label="recorded, mean"),
               Line2D([], [], marker="o", ls="", color=BLUE, markersize=4, label="main specification, mean"),
               Line2D([], [], marker="D", ls="", color=ORANGE, markersize=4, label="its modal version, mean")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, -0.08))
    means = d["loco_by_class_means"]
    if means is not None:     # version 3.1: means over all countries, written before any score is withheld
        means = means[means.coding == "full"]
        mean_main = means[means.variant == MAIN].set_index("place")
        mean_modal = means[means.variant == MODAL].set_index("place")
    else:
        mean_main, mean_modal = main.groupby("place").mean(numeric_only=True), \
            modal.groupby("place").mean(numeric_only=True)
    if "respondents_recorded" in main.columns:
        dots = main               # version 3.1: scores on too few respondents are already withheld
    else:                     # older outputs carry no respondent counts: no country scores for rare places
        dots = main[~main.place.isin(RARE_PLACES)]
    for ax, metric, title in ((axes[1], "recall", "(b) Recall"), (axes[2], "precision", "(c) Precision")):
        for yy, p in zip(y, order):
            vals = dots.loc[dots.place == p, metric].dropna().to_numpy()
            ax.scatter(vals, yy + rng.uniform(-0.18, 0.18, len(vals)), s=5, color=LIGHT, linewidths=0, zorder=2)
        ax.scatter(mean_main[metric].reindex(order), y, s=16, color=BLUE, zorder=4)
        ax.scatter(mean_modal[metric].reindex(order), y, s=14, color=ORANGE, marker="D", zorder=3)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(title, loc="left", fontsize=7.5)
    for ax in axes:
        ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    fig.subplots_adjust(wspace=0.1)
    save(fig, out / "fig_places.pdf")


def fig_germany_tempogram(d, out):
    s = d["de_slot"]
    stack = stack_of(s.place.unique())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, (group, title) in zip(axes, (("weekday", "(a) Weekdays"), ("weekend", "(b) Weekend days"))):
        g = s[s.group == group].pivot(index="slot", columns="place", values="mean")[stack] * 100
        ax.stackplot(g.index, g.T.to_numpy(), colors=[PLACE_COLOURS[p] for p in stack], linewidth=0)
        n = int(s.loc[s.group == group, "diaries"].iloc[0])
        ax.set_title(f"{title} ({n:,} diaries)", loc="left", fontsize=7.5)
        hour_axis(ax, end_label=ax is axes[-1])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines["left"].set_visible(True)
    axes[0].set_ylabel("Share of diaries (%)")
    handles = [Patch(facecolor=PLACE_COLOURS[p], edgecolor="none", label=p) for p in stack]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.13),
               handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.06)
    save(fig, out / "fig_germany_tempogram.pdf")


def fig_germany_home(d, out):
    h, s = d["home"], d["de_slot"]
    obs = h[h.variant == MAIN].pivot(index="slot", columns="country", values="observed") * 100
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    ax = axes[0]
    for c in obs.columns:
        ax.plot(obs.index, obs[c], color=LIGHT, lw=0.6, zorder=1)
    de = s[(s.group == "all") & (s.place == "home")].sort_values("slot")
    env = d["gsa_germany_uncertainty_home_by_slot"]
    if env is not None:      # version 3: the range of 90% of the combinations of the global analysis
        env = group_rows(env, "all").sort_values("slot")
        lo, hi = 100 * env[RANGE_LO], 100 * env[RANGE_HI]
        band_label = "90% of combinations"
    else:                    # version 2: constant band from the held-out errors
        q = 100 * d["transfer"]["home_q"]
        lo, hi = np.clip(100 * de["mean"] - q, 0, 100), np.clip(100 * de["mean"] + q, 0, 100)
        band_label = f"$\\pm${q:.1f} pp transfer error"
    ax.fill_between(de.slot, lo, hi, color=BLUE, alpha=0.18, lw=0, zorder=2)
    handles = [Line2D([], [], color=BLUE, lw=1.2, label="Germany (imputed)"),
               Patch(facecolor=BLUE, alpha=0.18, edgecolor="none", label=band_label)]
    best = d.get("gsa_germany_uncertainty_home_by_slot_pareto")
    if env is not None and best is not None:     # version 4.1: the Pareto-optimal methods of part A
        b = group_rows(best, "all").sort_values("slot")
        ax.fill_between(b.slot, 100 * b[RANGE_LO], 100 * b[RANGE_HI], color=BLUE, alpha=0.25, lw=0, zorder=2.5)
        handles.append(Patch(facecolor=BLUE, alpha=0.40, edgecolor="none", label="90%, Pareto-optimal methods"))
    ax.plot(de.slot, 100 * de["mean"], color=BLUE, lw=1.2, zorder=3)
    ax.set_title("(a) Germany and the 16 scored countries", loc="left", fontsize=7.5)
    ax.legend(handles=handles + [Line2D([], [], color=LIGHT, lw=0.8, label="other countries (recorded)")],
              loc="lower left", frameon=False)
    ax = axes[1]
    for group, colour in (("age 10-17", "#E69F00"), ("age 18-64", BLUE), ("age 65+", "#009E73")):
        g = s[(s.group == group) & (s.place == "home")].sort_values("slot")
        ax.plot(g.slot, 100 * g["mean"], color=colour, lw=1.0, label=group.replace("age ", "aged ").replace("-", "–"))
    ax.set_title("(b) Germany, imputed, by age", loc="left", fontsize=7.5)
    ax.legend(loc="lower left", frameon=False)
    for ax in axes:
        hour_axis(ax, end_label=ax is axes[-1])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    axes[0].set_ylabel("At home (%)")
    fig.subplots_adjust(wspace=0.06)
    save(fig, out / "fig_germany_home.pdf")


def gsa_bars(idx, factors, outputs, path, letters="abcd"):
    """One row of panels, one per output: first-order (dark) and total (light) Sobol' index of
    each factor."""
    height = max(1.9, 0.75 + 0.19 * len(factors))
    fig, axes = plt.subplots(1, len(outputs), figsize=(WIDTH, height), sharey=True)
    y = np.arange(len(factors))
    for ax, (key, title), letter in zip(np.atleast_1d(axes), outputs, letters):
        sub = idx[idx.output == key].set_index("factor")
        first = sub.first_order.reindex([f for f, _ in factors]).to_numpy(float)
        total = sub.total.reindex([f for f, _ in factors]).to_numpy(float)
        ax.barh(y + 0.19, np.nan_to_num(np.clip(total, 0, 1)), height=0.36, color=LIGHT, zorder=2)
        ax.barh(y - 0.19, np.nan_to_num(np.clip(first, 0, 1)), height=0.36, color=BLUE, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=7.5)
    axes = np.atleast_1d(axes)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab in factors])
    axes[0].invert_yaxis()
    handles = [Patch(facecolor=BLUE, edgecolor="none", label="first-order index $S_i$"),
               Patch(facecolor=LIGHT, edgecolor="none", label="total index $S_{Ti}$")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.19 / height))
    fig.subplots_adjust(wspace=0.1, bottom=0.42 / height)          # the same margins in inches at any height
    save(fig, path)


def fig_gsa_methods(d, out):
    """Part A: first-order and total Sobol' indices and the optimal-transport index of each factor,
    for the seven evaluation metrics over the 144 combinations of method choices (version 4.1)."""
    idx = d["gsa_loco_indices"]
    if idx is None:
        return
    ot = ot_by_metric(d)
    factors = GSA_A_FACTORS
    metrics = [("accuracy", "(a) Accuracy"), ("kappa", r"(b) Cohen's $\kappa$"), ("macro_f1", "(c) Macro-F1"),
               ("home_share_mae_pp", "(d) Home share error"), ("home_share_bias_pp", "(e) Home share bias"),
               ("hourly_tvd", "(f) Hourly TVD"), ("changes_error_pct", "(g) Change error")]
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 4.6), sharey=True, sharex=True)
    y = np.arange(len(factors))
    h = 0.26
    for ax, (key, title) in zip(axes.ravel(), metrics):
        sub = idx[idx.output == key].set_index("factor")
        first = np.nan_to_num(np.clip(sub.first_order.reindex([f for f, _ in factors]).to_numpy(float), 0, 1))
        total = np.nan_to_num(np.clip(sub.total.reindex([f for f, _ in factors]).to_numpy(float), 0, 1))
        ax.barh(y - h, first, height=h, color=BLUE, zorder=3)
        ax.barh(y, total, height=h, color=LIGHT, zorder=2)
        if ot is not None:
            o = ot[ot.output == key].set_index("factor").ot_index.reindex([f for f, _ in factors]).to_numpy(float)
            ax.barh(y + h, np.nan_to_num(np.clip(o, 0, 1)), height=h, color=ORANGE, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_title(title, loc="left", fontsize=7.5)
    for row in range(2):
        axes[row, 0].set_yticks(y)
        axes[row, 0].set_yticklabels([lab for _, lab in factors])
    axes[0, 0].invert_yaxis()
    legend_ax = axes[1, 3]
    legend_ax.axis("off")
    handles = [Patch(facecolor=BLUE, edgecolor="none", label="first-order Sobol' index $S_i$"),
               Patch(facecolor=LIGHT, edgecolor="none", label="total Sobol' index $S_{Ti}$"),
               Patch(facecolor=ORANGE, edgecolor="none", label=r"optimal-transport index $\iota_i$")]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, bbox_to_anchor=(0.0, 0.5))
    legend_ax.tick_params(labelbottom=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    save(fig, out / "fig_gsa_methods.pdf")


def fig_pareto(d, out):
    """Part A: the 144 combinations of method choices by two pairs of metrics (means over the 13
    donors), Pareto-optimal ones over all seven metrics outlined (version 4.1)."""
    m = d.get("gsa_loco_combinations")
    if m is None or "pareto_optimal" not in m.columns:
        return
    markers = {"slot": "s", "episode": "^", "markov": "o"}
    colours = {"draw": BLUE, "most likely": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    panels = [("home_share_mae_pp", "accuracy", "Home share error (pp)", "Accuracy", None),
              ("hourly_tvd", "changes_error_pct", "Hourly TVD", "Change error (%), absolute", "log")]
    for ax, (x, y, xl, yl, yscale), letter in zip(axes, panels, "ab"):
        for (unit, assign), sub in m.groupby(["unit", "assignment"]):
            yy = sub[y].abs() if yscale == "log" else sub[y]
            ax.scatter(sub[x], yy, s=14, marker=markers[unit], color=colours[assign], alpha=0.75, linewidths=0, zorder=2)
            best = sub[sub.pareto_optimal]
            if len(best):
                yb = best[y].abs() if yscale == "log" else best[y]
                ax.scatter(best[x], yb, s=34, marker=markers[unit], facecolors="none", edgecolors="black",
                           linewidths=0.7, zorder=3)
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"({letter})", loc="left", fontsize=7.5)
        ax.grid(color="#EEEEEE", lw=0.5, zorder=0)
    handles = ([Line2D([], [], marker=markers[u], ls="", color=GREY, markersize=4, label=lab)
                for u, lab in (("slot", "slot"), ("episode", "episode"), ("markov", "Markov chain"))]
               + [Patch(facecolor=colours[a], edgecolor="none", label=lab)
                  for a, lab in (("draw", "random draw"), ("most likely", "most likely class"))]
               + [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black", markersize=5,
                         label="Pareto-optimal")])
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.3, bottom=0.25)
    save(fig, out / "fig_pareto.pdf")


def fig_gsa_germany(d, out):
    """Part B: indices of the German estimates over all combinations."""
    if d["gsa_germany_indices"] is not None:
        gsa_bars(d["gsa_germany_indices"], b_factors(d["gsa_germany_indices"]),
                 [o for o in GSA_B_OUTPUTS if o[0] != "other people's home"], out / "fig_gsa_germany.pdf")


def fig_gsa_home(d, out):
    """Share of the variance of the German share at home in each slot due to each factor alone
    (first-order indices, stacked) and to interactions. Version 3 outputs."""
    idx = d["gsa_germany_indices_home_by_slot"]
    if idx is None:
        return
    idx = group_rows(idx, "all")
    factors = b_factors(idx)
    p = idx.pivot(index="slot", columns="factor", values="first_order")[[f for f, _ in factors]]
    p = p.clip(lower=0).fillna(0)
    p["interactions"] = (1 - p.sum(axis=1)).clip(lower=0)
    keys = [f for f, _ in factors] + ["interactions"]
    labels = dict(factors, interactions="Interactions")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.2))
    ax.stackplot(p.index, p[keys].T.to_numpy(), colors=[GSA_COLOURS[k] for k in keys], linewidth=0)
    hour_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of variance")
    ax.legend(handles=[Patch(facecolor=GSA_COLOURS[k], edgecolor="none", label=labels[k]) for k in keys[::-1]],
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)     # top to bottom, as stacked
    save(fig, out / "fig_gsa_home.pdf")


# --------------------------------------------------------------------------- tables
def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def metric_means(d):
    """The seven evaluation metrics of each variant, unweighted means over the countries of each
    coding group: those of loco_summary.csv, the home share bias (mean over the slots of imputed
    minus recorded weighted share at home, in pp) and the change error (imputed over recorded mean
    number of location changes per diary, minus one, in percent)."""
    o = d["overall"].copy()
    o["changes_error_pct"] = 100 * (o.changes_imputed / o.changes_observed - 1)
    bias = d["home"].groupby(["variant", "country"]).error_pp.mean().rename("home_share_bias_pp").reset_index()
    o = o.merge(bias, on=["variant", "country"], how="left")
    cols = ["accuracy", "kappa", "macro_f1", "home_share_mae_pp", "home_share_bias_pp", "hourly_tvd",
            "changes_error_pct", "changes_observed", "changes_imputed"]
    return o.groupby(["coding", "variant"])[cols].mean(), o


def table_variants(d, out):
    """The seven evaluation metrics of Table 1 for each variant, means over the 13 donors."""
    s = metric_means(d)[0].loc["full"]
    observed = s.loc[MAIN, "changes_observed"]
    groups = [("Baselines", ["home", "activity_modal"]),
              ("Random draw", ["slot_time", "slot_time_age", "episode_time_age", "markov_time", MAIN,
                               "markov_time_age_day"]),
              ("Modal assignment", [MODAL])]       # alpha is varied in the global analysis (version 3)
    lines = [r"\begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] "
             r"S[table-format=2.2] S[table-format=-1.2] S[table-format=1.3] S[table-format=-3.1]}",
             r"\toprule",
             r"Variant & {Accuracy} & {$\kappa$} & {Macro-F1} & {\makecell{Home share\\error (pp)}} & "
             r"{\makecell{Home share\\bias (pp)}} & {\makecell{Hourly\\TVD}} & {\makecell{Change\\error (\%)}} \\",
             r"\midrule"]
    # best value of each metric over all variants: highest, lowest, or closest to zero (GSA_A_DIRECTIONS);
    # every variant whose printed value equals the printed best is marked
    cols = [("accuracy", f3), ("kappa", f3), ("macro_f1", f3), ("home_share_mae_pp", f2),
            ("home_share_bias_pp", lambda x: f"{x:.2f}"), ("hourly_tvd", f3), ("changes_error_pct", lambda x: f"{x:.1f}")]
    shown = [v for _, vs in groups for v in vs]
    best = {}
    for col, fmt in cols:
        sense = GSA_A_DIRECTIONS[col]
        vals = s.loc[shown, col]
        key = vals if sense else vals.abs()
        top = key.idxmax() if sense > 0 else key.idxmin()
        target = fmt(abs(vals[top])) if sense == 0 else fmt(vals[top])
        best[col] = {v for v in shown if (fmt(abs(vals[v])) if sense == 0 else fmt(vals[v])) == target}
    lines.insert(0, r"\sisetup{detect-weight=true}")
    for gi, (group, variants) in enumerate(groups):
        if gi:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{8}}{{l}}{{{group}}} \\")
        for v in variants:
            r = s.loc[v]
            cells = [("\\bfseries " if v in best[col] else "") + fmt(r[col]) for col, fmt in cols]
            lines.append(f"\\quad {tex_label(FULL_LABEL[v])} & " + " & ".join(cells) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_variants.tex").write_text("\n".join(lines) + "\n")
    return observed


def table_countries(d, out):
    """The seven evaluation metrics of the main specification for each held-out country, with the
    accuracy of the all-at-home baseline for reference."""
    o = metric_means(d)[1]
    m = o[o.variant == MAIN].set_index("country")
    base = o[o.variant == "home"].set_index("country").accuracy
    lines = [r"\begin{tabular}{l r S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] "
             r"S[table-format=1.3] S[table-format=1.2] S[table-format=-1.2] S[table-format=1.3] S[table-format=-2.1]}",
             r"\toprule",
             r"Country & {Diaries} & {\makecell{All at home\\accuracy}} & {Accuracy} & {$\kappa$} & {Macro-F1} & "
             r"{\makecell{Home share\\error (pp)}} & {\makecell{Home share\\bias (pp)}} & "
             r"{\makecell{Hourly\\TVD}} & {\makecell{Change\\error (\%)}} \\",
             r"\midrule"]
    for coding, title in (("full", "Full place coding (donors)"), ("partial", "Partial place coding (scored only)")):
        if coding == "partial":
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{10}}{{l}}{{{title}}} \\")
        for c in sorted(m.index[m.coding == coding]):
            r = m.loc[c]
            lines.append(f"\\quad {c} & {int(r.diaries):,} & {f3(base[c])} & {f3(r.accuracy)} & {f3(r.kappa)} & "
                         f"{f3(r.macro_f1)} & {f2(r.home_share_mae_pp)} & {r.home_share_bias_pp:.2f} & "
                         f"{f3(r.hourly_tvd)} & {r.changes_error_pct:.1f} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_countries.tex").write_text("\n".join(lines) + "\n")


def table_germany(d, out):
    """Hours per day by place for German groups. With version 3 outputs, a second column gives
    the range of 90% of the combinations of the global sensitivity analysis for all respondents,
    and from version 4.1 a third column the same range over the combinations whose method choices
    are Pareto-optimal in part A; with version 2 outputs, the range implied by the 13 donors'
    held-out errors (the German value minus the largest and minus the smallest error)."""
    day = d["de_day"].pivot(index="place", columns="group", values="hours_per_day")
    n = d["de_slot"].groupby("group").diaries.first()
    cols = ["all", "weekday", "weekend", "age 10-17", "age 18-64", "age 65+"]
    heads = ["All", "Weekdays", "Weekend", "10--17", "18--64", "65+"]
    hours_err = d["transfer"].get("hours")
    gsa = d["gsa_germany_uncertainty"]
    if gsa is not None:
        gsa = gsa.set_index("output")
    best = d.get("gsa_germany_uncertainty_pareto")
    if best is not None:
        best = best.set_index("output")
    rng_col = hours_err is not None or gsa is not None
    n_rng = (1 if rng_col else 0) + (1 if best is not None else 0)
    spec = "l S[table-format=2.2] " + "c " * n_rng + " ".join(["S[table-format=2.2]"] * (len(cols) - 1))
    if best is not None:
        top = (r" & \multicolumn{3}{c}{All respondents} & \multicolumn{2}{c}{Day type} & "
               r"\multicolumn{3}{c}{Age (years)} \\", r"\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}",
               r"Location & {Imputed} & {\makecell{Range,\\all}} & {\makecell{Range,\\Pareto}} & "
               + " & ".join("{" + h + "}" for h in heads[1:]) + r" \\")
    elif rng_col:
        top = (r" & \multicolumn{2}{c}{All respondents} & \multicolumn{2}{c}{Day type} & "
               r"\multicolumn{3}{c}{Age (years)} \\", r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-8}",
               "Location & {Imputed} & {Range} & " + " & ".join("{" + h + "}" for h in heads[1:]) + r" \\")
    else:
        top = (r" & & \multicolumn{2}{c}{Day type} & \multicolumn{3}{c}{Age (years)} \\",
               r"\cmidrule(lr){3-4} \cmidrule(lr){5-7}",
               "Location & {All} & " + " & ".join("{" + h + "}" for h in heads[1:]) + r" \\")
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", *top, r"\midrule"]

    def rng(frame, p):
        return f"{frame.loc[p, RANGE_LO]:.2f}--{frame.loc[p, RANGE_HI]:.2f}"

    for p in stack_of(day.index):
        cells = [f2(day.loc[p, "all"])]
        recorded = p in ("travelling", "transport")
        if rng_col:
            if recorded:
                cells.append("recorded")
            elif gsa is not None and p in gsa.index:
                cells.append(rng(gsa, p))
            elif hours_err is not None and p in hours_err.index:
                lo = max(day.loc[p, "all"] - hours_err.loc[p, "err_max"], 0)
                hi = day.loc[p, "all"] - hours_err.loc[p, "err_min"]
                cells.append(f"{lo:.2f}--{hi:.2f}")
            else:
                cells.append("")
        if best is not None:
            cells.append("recorded" if recorded else (rng(best, p) if p in best.index else ""))
        cells += [f2(day.loc[p, c]) for c in cols[1:]]
        lines.append(p[0].upper() + p[1:] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    diaries = ["{" + f"{int(n[c]):,}".replace(",", "{,}") + "}" for c in cols]
    for _ in range(n_rng):
        diaries.insert(1, "")
    lines.append("Diaries & " + " & ".join(diaries) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_germany.tex").write_text("\n".join(lines) + "\n")


def table_gsa(idx, factors, outputs, path, ot=None):
    """Sobol' indices in percent of the variance of each output, in two panels: (a) first-order
    indices and the share due to interactions (100 minus their sum), (b) total indices; with ot
    (one row per output and factor, column ot_index), (c) the optimal-transport indices of the same
    outputs, in percent. One row per output and one column per factor."""
    if idx is None:
        return
    k = len(factors)
    fmt = "S[table-format=3.1]"
    lines = [r"\begin{tabular}{l " + " ".join([fmt] * (k + 1)) + "}",
             r"\toprule",
             "Output & " + " & ".join("{" + lab + "}" for _, lab in factors) + r" & {\makecell{Inter-\\actions}} \\",
             r"\midrule"]

    def cell(v):
        return "{--}" if not np.isfinite(v) else f"{100 * min(max(v, 0.0), 1.0):.1f}"

    rows_a, rows_b, rows_c = [], [], []
    for out_key, lab in outputs:
        sub = idx[idx.output == out_key].set_index("factor")
        if sub.empty:
            continue
        first = [cell(float(sub.at[key, "first_order"])) if key in sub.index else "" for key, _ in factors]
        inter = cell(float(sub.at["interactions", "first_order"])) if "interactions" in sub.index else ""
        total = [cell(float(sub.at[key, "total"])) if key in sub.index else "" for key, _ in factors]
        rows_a.append(r"\quad " + lab + " & " + " & ".join(first + [inter]) + r" \\")
        rows_b.append(r"\quad " + lab + " & " + " & ".join(total + [""]) + r" \\")
        if ot is not None:
            o = ot[ot.output == out_key].set_index("factor").ot_index
            cells = [cell(float(o[key])) if key in o.index else "" for key, _ in factors]
            rows_c.append(r"\quad " + lab + " & " + " & ".join(cells + [""]) + r" \\")
    lines += [rf"\multicolumn{{{k + 2}}}{{l}}{{(a) First-order indices $S_i$}} \\"] + rows_a
    lines += [r"\addlinespace", rf"\multicolumn{{{k + 2}}}{{l}}{{(b) Total indices $S_{{Ti}}$}} \\"] + rows_b
    if rows_c:
        lines += [r"\addlinespace", rf"\multicolumn{{{k + 2}}}{{l}}{{(c) Optimal-transport indices $\iota_i$}} \\"] + rows_c
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def assignment(cost):
    """Minimum-cost perfect matching of a square cost matrix (Hungarian algorithm with potentials,
    O(n^3)). Returns col, with row i matched to column col[i]."""
    cost = np.asarray(cost, float)
    n = cost.shape[0]
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, int), np.zeros(n + 1, int)       # p[j]: row matched to column j (1-based)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(n + 1, np.inf), np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col = np.empty(n, int)
    col[p[1:] - 1] = np.arange(n)
    return col


def w2_squared(P, Q):
    """Squared 2-Wasserstein distance between the uniform distributions on the rows of P and Q,
    with len(P) a multiple of len(Q) (each row of Q then carries the mass of len(P)/len(Q) rows)."""
    Qr = np.repeat(Q, len(P) // len(Q), axis=0)
    C = ((P[:, None, :] - Qr[None, :, :]) ** 2).sum(2)
    return float(C[np.arange(len(P)), assignment(C)].mean())


def w2sq_exact(ys, yc):
    """Squared 2-Wasserstein distance between the uniform distributions on the sorted samples ys
    (size n) and yc (size m), for any two sizes. Both quantile functions are step functions, with
    steps at i/n and k/m; on every piece of the merged grid both are constant, so the integral over
    (0, 1) of their squared difference is a finite sum. Equals POT's ot.wasserstein_1d(ys, yc, p=2)
    and, when n is a multiple of m, the mean squared difference between ys and yc with each value
    of yc repeated n/m times. Used for every optimal-transport index of a single output, instead of
    pairing the values by nearest rank, which is exact only for some class sizes."""
    ys, yc = np.asarray(ys, float), np.asarray(yc, float)
    n, m = len(ys), len(yc)
    grid = np.union1d(np.arange(1, n + 1) / n, np.arange(1, m + 1) / m)     # right ends of the pieces
    width = np.diff(np.concatenate([[0.0], grid]))
    mid = grid - width / 2
    a = ys[np.minimum((mid * n).astype(int), n - 1)]
    b = yc[np.minimum((mid * m).astype(int), m - 1)]
    return float(np.sum(width * (a - b) ** 2))


def ot_univariate(m, metric, factors):
    """Optimal-transport index of one metric for each factor, iota_i = E[W2^2(P_Y, P_{Y|X_i})] /
    (2 Var Y), with one class per level, weighted by its share of the combinations, and the variance
    of the combinations themselves (ddof=0). Then iota_i = S_i/2 + R_i exactly, with R_i >= 0 due to
    changes in the spread and shape of the distribution, iota_i = 1 when X_i determines Y, and
    S_i/2 <= iota_i <= S_Ti for independent factors. Exact for the enumerated factorial."""
    y = m[metric].to_numpy(float)
    ys, n, var = np.sort(y), len(y), y.var()
    out = {}
    for f in factors:
        w = 0.0
        for _, sub in m.groupby(f):
            q = np.sort(sub[metric].to_numpy(float))
            w += len(q) / n * w2sq_exact(ys, q)
        out[f] = w / (2 * var) if var > 0 else np.nan
    return out


def ot_by_metric(d):
    """Optimal-transport indices of each metric of part A (one row per metric and factor)."""
    m = d.get("gsa_loco_combinations")
    if m is None and d.get("gsa_loco") is not None:
        m = d["gsa_loco"].groupby([f for f, _ in GSA_A_FACTORS])[list(GSA_A_DIRECTIONS)].mean().reset_index()
    if m is None:
        return None
    rows = []
    for metric in GSA_A_DIRECTIONS:
        for f, v in ot_univariate(m, metric, [f for f, _ in GSA_A_FACTORS]).items():
            rows.append(dict(output=metric, factor=f, ot_index=v))
    return pd.DataFrame(rows)


def ot_joint(d):
    """Optimal-transport index of the seven evaluation metrics of part A taken jointly,
    iota_i = E[W2^2(P_Y, P_{Y|X_i})] / (2 tr Cov(Y)), exact by optimal assignment (w2_squared),
    under five ways of putting the metrics on a common scale: standardised, ranked and
    standardised, scaled by their range, standardised with the change error as the logarithm of
    the ratio of imputed to recorded changes, and standardised without kappa and macro-F1. The
    metrics have no common scale, so the paper reports no joint index, only how its value and
    ranking of the factors depend on the scale (derived_numbers.csv)."""
    m = d.get("gsa_loco_combinations")
    if m is None and d.get("gsa_loco") is not None:
        m = d["gsa_loco"].groupby([f for f, _ in GSA_A_FACTORS])[list(GSA_A_DIRECTIONS)].mean().reset_index()
    if m is None:
        return None
    m = m.reset_index(drop=True)
    Y = m[list(GSA_A_DIRECTIONS)].astype(float)

    def std(frame):
        a = frame.to_numpy(float)
        return (a - a.mean(0)) / a.std(0)

    raw = Y.to_numpy(float)
    scalings = {"standardised": std(Y),
                "ranks": std(Y.rank()),
                "range": (raw - raw.mean(0)) / (raw.max(0) - raw.min(0)),
                "log_change_ratio": std(Y.assign(changes_error_pct=np.log1p(Y.changes_error_pct / 100))),
                "five_metrics": std(Y.drop(columns=["kappa", "macro_f1"]))}
    rows = []
    for name, Z in scalings.items():
        denom = 2 * (Z ** 2).sum(1).mean()
        for f, _ in GSA_A_FACTORS:
            w = [w2_squared(Z, Z[sub.index.to_numpy()]) for _, sub in m.groupby(f)]
            rows.append(dict(scaling=name, factor=f, ot_index=np.mean(w) / denom))
    return pd.DataFrame(rows)


def table_sensitivity(d, out):
    """Hours per day for all German respondents under the sensitivity runs (version 2)."""
    s = d["germany_sensitivity"]
    if s is None:
        return
    s = s[s.group == "all"].pivot(index="place", columns="scenario", values="hours_per_day")
    cols = ["main (fixed donor pool)", "no in-transit draws", "code 13 missing for home activities", "both"]
    heads = [r"\makecell{Main\\specification}", r"\makecell{No in-transit\\draws}",
             r"\makecell{Code 13 missing,\\home activities}", r"\makecell{Both}"]
    lines = [r"\begin{tabular}{l " + " ".join(["S[table-format=2.2]"] * len(cols)) + "}",
             r"\toprule",
             "Location & " + " & ".join("{" + h + "}" for h in heads) + r" \\",
             r"\midrule"]
    for p in STACK_V3:
        lines.append(p[0].upper() + p[1:] + " & " + " & ".join(f2(s.loc[p, c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_sensitivity.tex").write_text("\n".join(lines) + "\n")


def table_calibration(d, out):
    """Coverage of the recorded share at home by slot in the held-out countries by plain 95%
    intervals from repeated imputations, with the fixed donor pool and with random donor weights."""
    c = d["loco_calibration_summary"]
    if c is None:
        return
    lines = [r"\begin{tabular}{l S[table-format=1.2] S[table-format=2.1]}",
             r"\toprule",
             r"Spread across imputations & {Coverage} & {\makecell{Width\\(pp)}} \\",
             r"\midrule"]
    c = c.assign(g=c.gamma.astype(float)).sort_values("g", ascending=False)
    for _, r in c.iterrows():
        label = "Fixed donor pool (draws only)" if np.isinf(r.g) else rf"Donor weights, $\gamma = {r.g:g}$"
        lines.append(f"{label} & {r.coverage_home_plain:.2f} & {r.width_home_plain_pp:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_calibration.tex").write_text("\n".join(lines) + "\n")


def derived_numbers(d, out):
    """Numbers derived here (not in the pipeline outputs) that the text quotes."""
    t, rows = d["transfer"], []
    rows.append(dict(name="home_share_halfwidth_pp", value=100 * t["home_q"]))
    rep = d["loco_repeated_imputations_summary"]
    if rep is not None:                   # version 4.0: plain 95% intervals from repeated imputations
        for _, r in rep.iterrows():
            tag = "all" if r.countries.startswith("all") else "donors"
            for q in ("coverage_home", "coverage_hours", "coverage_changes", "width_home_pp"):
                rows.append(dict(name=f"repeated_{q}_{tag}", value=r[q]))
    cal = d["loco_calibration_summary"]
    if cal is not None:                   # versions 2 to 3.1: the fixed donor pool row
        r = cal[np.isinf(cal.gamma.astype(float))].iloc[0]
        rows.append(dict(name="repeated_coverage_home_all", value=r.coverage_home_plain))
        rows.append(dict(name="repeated_width_home_pp_all", value=r.width_home_plain_pp))
    rows.append(dict(name="home_share_halfwidth_cv_coverage", value=t["home_cv"]))
    if "hours" in t:
        day = d["de_day"].pivot(index="place", columns="group", values="hours_per_day")["all"]
        for p, r in t["hours"].iterrows():
            rows.append(dict(name=f"germany_hours_{p}_low", value=max(day[p] - r.err_max, 0)))
            rows.append(dict(name=f"germany_hours_{p}_high", value=day[p] - r.err_min))
            rows.append(dict(name=f"donor_error_{p}_min", value=r.err_min))
            rows.append(dict(name=f"donor_error_{p}_max", value=r.err_max))
    ch = d["germany_changes_per_diary"]
    if ch is not None:
        x = float(ch.loc[ch.group == "all", "changes"].iloc[0])
        rows.append(dict(name="germany_changes_all", value=x))
        rows.append(dict(name="germany_changes_low", value=x - t["changes"]["err_max"]))
        rows.append(dict(name="germany_changes_high", value=x - t["changes"]["err_min"]))
    u = d["gsa_germany_uncertainty"]
    if u is not None:                     # version 3: distribution over the combinations
        for _, r in u.iterrows():
            for q in ("main_imputed", "q05", "q50", "q95", "min", "max"):
                rows.append(dict(name=f"gsa_germany_{r.output}_{q}", value=r[q]))
    uh = d["gsa_germany_uncertainty_home_by_slot"]
    if uh is not None:
        width = 100 * (group_rows(uh, "all")[RANGE_HI] - group_rows(uh, "all")[RANGE_LO])
        rows.append(dict(name="gsa_home_share_range_width_pp_mean", value=width.mean()))
        rows.append(dict(name="gsa_home_share_range_width_pp_max", value=width.max()))
        for g in ("all", "weekday", "weekend"):          # range at the minimum of the main imputation
            u = group_rows(uh, g)
            if len(u):
                r = u.loc[u.main_imputed.idxmin()]
                for q in ("main_imputed", RANGE_LO, RANGE_HI):
                    rows.append(dict(name=f"gsa_home_share_{g}_minimum_{q}_pct", value=100 * r[q]))
                rows.append(dict(name=f"gsa_home_share_{g}_minimum_slot_clock", value=r.clock))
    for name, idx in (("gsa_loco", d["gsa_loco_indices"]), ("gsa_germany", d["gsa_germany_indices"])):
        if idx is not None:
            for _, r in idx.iterrows():
                rows.append(dict(name=f"{name}_{r.output}_{r.factor}_first_order", value=r.first_order))
                if np.isfinite(r.total):
                    rows.append(dict(name=f"{name}_{r.output}_{r.factor}_total", value=r.total))
    m = d.get("gsa_loco_combinations")
    if m is not None and "pareto_optimal" in m.columns:      # version 4.1
        best = m[m.pareto_optimal]
        rows.append(dict(name="pareto_optimal_combinations", value=len(best)))
        for (unit, assign), k in best.groupby(["unit", "assignment"]).size().items():
            rows.append(dict(name=f"pareto_optimal_{unit}_{assign}", value=k))
        for metric, sense in GSA_A_DIRECTIONS.items():
            v = m[metric] if sense else m[metric].abs()
            i = v.idxmax() if sense > 0 else v.idxmin()
            rows.append(dict(name=f"best_{metric}", value=m.loc[i, metric]))
            rows.append(dict(name=f"best_{metric}_combination",
                             value=" ".join(f"{f}={m.loc[i, f]}" for f, _ in GSA_A_FACTORS)))
    up = d.get("gsa_germany_uncertainty_pareto")
    if up is not None:
        for _, r in up.iterrows():
            for q in ("q05", "q50", "q95", "min", "max"):
                rows.append(dict(name=f"gsa_germany_pareto_{r.output}_{q}", value=r[q]))
    uhp = d.get("gsa_germany_uncertainty_home_by_slot_pareto")
    if uhp is not None:
        for g in ("all", "weekday", "weekend"):
            u = group_rows(uhp, g)
            if len(u):
                r = u.loc[u.main_imputed.idxmin()]
                for q in (RANGE_LO, RANGE_HI):
                    rows.append(dict(name=f"gsa_home_share_pareto_{g}_minimum_{q}_pct", value=100 * r[q]))
        w = 100 * (group_rows(uhp, "all")[RANGE_HI] - group_rows(uhp, "all")[RANGE_LO])
        rows.append(dict(name="gsa_home_share_pareto_range_width_pp_mean", value=w.mean()))
    comb, lc = d.get("gsa_germany_combinations"), d.get("gsa_loco_combinations")
    if comb is not None and lc is not None and "unit" in comb.columns and "pareto_optimal" in lc.columns:
        # Version 4.1. The combinations whose method choices are Pareto-optimal in part A form a
        # complete factorial of those methods (as one factor), code 13, the draws and the transfer
        # error, so exact first-order and total indices follow from conditional means.
        factors = [f for f, _ in GSA_A_FACTORS]
        key = lambda frame: frame[factors].astype({"alpha": float})
        best = key(lc.loc[lc.pareto_optimal.astype(bool)]).drop_duplicates()
        sub = comb.assign(**key(comb)).merge(best, on=factors, how="inner")
        sub["method"] = sub[factors].astype(str).agg("|".join, axis=1)
        within = ["method", "code_13", "draws", "transfer_error"]
        outputs = [o for o in comb.columns if o in PLACES + ["changes per diary"]]
        for out_key in outputs:
            v = sub[out_key].var(ddof=0)
            if v > 1e-15:
                for f in within:
                    rest = [g for g in within if g != f]
                    first = sub.groupby(f)[out_key].transform("mean").var(ddof=0) / v
                    total = 1 - sub.groupby(rest)[out_key].transform("mean").var(ddof=0) / v
                    rows.append(dict(name=f"pareto_subset_{out_key}_{f}_first_order", value=first))
                    rows.append(dict(name=f"pareto_subset_{out_key}_{f}_total", value=total))
            vv = comb[out_key].var(ddof=0)
            if vv > 1e-15:     # closed indices of groups of factors over all combinations
                for group, name in ((factors, "methods"), (["alpha", "age", "day_type"], "alpha_age_day"),
                                    (["unit", "time", "assignment"], "unit_time_assignment")):
                    rows.append(dict(name=f"gsa_germany_{out_key}_closed_{name}",
                                     value=comb.groupby(group)[out_key].transform("mean").var(ddof=0) / vv))
        rows.append(dict(name="pareto_subset_combinations", value=len(sub)))
        rows.append(dict(name="pareto_subset_methods", value=sub["method"].nunique()))
    oj = d.get("ot_joint")
    if oj is not None:       # joint index of the seven metrics under five scalings (not reported as such)
        for _, r in oj.iterrows():
            rows.append(dict(name=f"ot_joint_{r.scaling}_{r.factor}", value=r.ot_index))
        for f, _ in GSA_A_FACTORS:
            v = oj[oj.factor == f].ot_index
            rows.append(dict(name=f"ot_joint_{f}_min", value=v.min()))
            rows.append(dict(name=f"ot_joint_{f}_max", value=v.max()))
        orders = {tuple(g.sort_values("ot_index", ascending=False).factor) for _, g in oj.groupby("scaling")}
        rows.append(dict(name="ot_joint_orders_over_scalings", value=" / ".join(" > ".join(o) for o in sorted(orders))))
    lc = d.get("gsa_loco_combinations")
    if lc is not None:       # correlations between the metrics over the method combinations
        c = lc[list(GSA_A_DIRECTIONS)].astype(float).corr()
        keys = list(GSA_A_DIRECTIONS)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                rows.append(dict(name=f"metric_corr_{a}_{b}", value=c.at[a, b]))
    u = ot_by_metric(d)
    if u is not None and d.get("gsa_loco_indices") is not None:
        si = d["gsa_loco_indices"].set_index(["output", "factor"])
        for _, r in u.iterrows():
            rows.append(dict(name=f"ot_{r.output}_{r.factor}", value=r.ot_index))
        for metric in GSA_A_DIRECTIONS:       # do the OT index and the total index order the factors alike?
            ou = u[u.output == metric].set_index("factor").ot_index
            tot = si.xs(metric).total.reindex(ou.index)
            first = si.xs(metric).first_order.reindex(ou.index)
            same_total = list(ou.sort_values(ascending=False).index) == list(tot.sort_values(ascending=False).index)
            same_first = list(ou.sort_values(ascending=False).index) == list(first.sort_values(ascending=False).index)
            rows.append(dict(name=f"ot_{metric}_same_order_as_total", value=same_total))
            rows.append(dict(name=f"ot_{metric}_same_order_as_first_order", value=same_first))
    ih = d["gsa_germany_indices_home_by_slot"]
    if ih is not None:
        ih = group_rows(ih, "all")
        tr = ih[ih.factor == "transfer_error"]
        rows.append(dict(name="gsa_home_by_slot_transfer_first_order_min", value=tr.first_order.min()))
        rows.append(dict(name="gsa_home_by_slot_transfer_first_order_mean", value=tr.first_order.mean()))
    pd.DataFrame(rows).to_csv(out / "derived_numbers.csv", index=False)


GENERATED = re.compile(r"(?P<start>% <<< generated: (?P<name>[\w]+) >>>\n)(?P<body>.*?)(?P<end>% <<< end of (?P=name) >>>)",
                       re.S)


def sync_paper(paper, tables):
    """Copy each generated table into the single-file manuscript, between its two marker lines."""
    text = paper.read_text()
    done = []

    def put(m):
        f = tables / f"{m.group('name')}.tex"
        if not f.is_file():
            return m.group(0)
        done.append(m.group("name"))
        return m.group("start") + f.read_text() + m.group("end")

    new = GENERATED.sub(put, text)
    if new != text:
        paper.write_text(new)
    return done


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    d = load(sys.argv[1])
    fig_variants(d, out / "figures")
    fig_home_error(d, out / "figures")
    fig_places(d, out / "figures")
    fig_germany_tempogram(d, out / "figures")
    fig_germany_home(d, out / "figures")
    fig_gsa_methods(d, out / "figures")
    fig_gsa_germany(d, out / "figures")
    fig_gsa_home(d, out / "figures")
    fig_pareto(d, out / "figures")
    table_variants(d, out / "tables")
    table_countries(d, out / "tables")
    table_germany(d, out / "tables")
    table_sensitivity(d, out / "tables")
    table_calibration(d, out / "tables")
    table_gsa(d["gsa_loco_indices"], GSA_A_TABLE_FACTORS, GSA_A_TABLE_ROWS, out / "tables" / "tab_gsa_methods.tex",
              ot=ot_by_metric(d))
    d["ot_joint"] = ot_joint(d)
    table_gsa(d["gsa_germany_indices"], b_factors(d["gsa_germany_indices"], table=True), GSA_B_TABLE_ROWS,
              out / "tables" / "tab_gsa_germany.tex")
    derived_numbers(d, out / "tables")
    if (out / "paper.tex").is_file():
        synced = sync_paper(out / "paper.tex", out / "tables")
        print(f"tables copied into paper.tex: {', '.join(synced) or 'none'}")
    print("figures and tables written to", out)


if __name__ == "__main__":
    main()
