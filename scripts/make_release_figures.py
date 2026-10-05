"""Regenerate aggregate figures/tables; never read images or patient-level data."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "score_confidence": "Cancer score alone",
    "score_disagreement": "View disagreement",
    "score_uncertainty": "Between-member variance",
    "score_confidence_disagreement": "Cancer score + disagreement",
    "score_confidence_uncertainty": "Cancer score + variance",
    "score_combined": "Combined score",
}
COLORS = ["#222222", "#CC79A7", "#009E73", "#D55E00", "#E69F00", "#0072B2"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs(root, result):
    """Cross-check independent saved draw rows and completed-result invariants."""
    verification = json.loads((root / "validation/fixed_gmic_final_result_verification_20261001.json").read_text())
    for name in ["fixed_gmic_final_results_20261001.json", "fixed_gmic_bootstrap_draws_20261001.csv"]:
        assert digest(root / "validation" / name) == verification["files"][name]["sha256"], name
    p = result["primary"]
    assert result["status"] == "final_fixed_model_evaluation_completed"
    assert p["candidate"]["false_negatives_flagged"] == 2
    assert p["baseline"]["false_negatives_flagged"] == 3
    assert all(p[k]["reviewed_breasts"] == 415 and p[k]["false_negatives"] == 39 for k in ["candidate", "baseline"])
    with (root / "validation/fixed_gmic_bootstrap_draws_20261001.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4000
    # Schema is read from the original saved output, not a new bootstrap run.
    schemes = {}
    for scheme, record in result["bootstrap"].items():
        selected = [r for r in rows if r["mode"] == scheme]
        values = [float(r["difference"]) for r in selected]
        assert len(values) == record["defined"] == 2000
        assert len({r["draw"] for r in selected}) == 2000
        assert all(r["status"] == "ok" for r in selected)
        interval = np.quantile(values, [0.025, 0.975])
        assert np.allclose(interval, record["ci95"], rtol=0, atol=1e-14)
        schemes[scheme] = {"draws": len(values), "ci95_reproduced": interval.tolist()}
    for row in result["review"]:
        assert row["reviewed_breasts"] == int(np.floor(row["budget"] * row["negative_breasts"]))
        assert np.isclose(row["fn_capture_fraction"], row["false_negatives_flagged"] / row["false_negatives"])
    return {"status": "passed", "bootstrap": schemes, "review_rows_checked": len(result["review"])}


def save(fig, directory, stem):
    for ext in ["png", "pdf", "svg"]:
        options = {"dpi": 300} if ext == "png" else {}
        if ext == "pdf":
            options["metadata"] = {"CreationDate": None, "ModDate": None}
        if ext == "svg":
            options["metadata"] = {"Date": None}
        fig.savefig(directory / f"{stem}.{ext}", bbox_inches="tight", **options)
    plt.close(fig)


def cohort_figure(result, config, directory):
    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    dev, test = config["roles"]["development"], config["roles"]["holdout"]
    pooled = next(r for r in result["classification"] if r["group"] == "pooled")

    def box(x, y, w, h, title, body, color="#EDF2F6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10", facecolor=color, edgecolor="#718096", linewidth=0.8))
        ax.text(x+w/2, y+h-0.18, title, ha="center", va="top", weight="bold", fontsize=10)
        ax.text(x+w/2, y+h-0.54, body, ha="center", va="top", fontsize=9, linespacing=1.15)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#718096", linewidth=1.2))

    box(2.2, 8.1, 5.6, 1.2, "Audited RSNA cohort", "11,913 patients · 23,826 breasts\n54,668 standard-view images; 38 other images excluded")
    box(.2, 5.8, 4.1, 1.45, "Development", f"{dev['patients']:,} patients · {dev['breasts']:,} breasts\n{dev['positive_breasts']} positive breasts\nCalibration, cutoff and score references")
    box(5.1, 5.8, 4.7, 1.45, "Patient-separated holdout", f"{test['patients']:,} patients · {test['breasts']:,} breasts\n{test['positive_breasts']} positive breasts · 43,787 images\nFive computation shards; one evaluation")
    arrow((4.0, 8.05), (2.25, 7.30)); arrow((6.0, 8.05), (7.45, 7.30))
    box(4.65, 3.35, 2.4, 1.5, "AI-positive", f"{pooled['tp']+pooled['fp']:,} breasts\n{pooled['tp']} true positives\n{pooled['fp']:,} false positives")
    box(7.35, 3.35, 2.4, 1.5, "AI-negative", f"{pooled['tn']+pooled['fn']:,} breasts\n{pooled['fn']} false negatives\n{pooled['tn']:,} true negatives", "#E8F2F8")
    arrow((6.7, 5.75), (5.85, 4.90)); arrow((8.2, 5.75), (8.55, 4.90))
    box(4.55, .85, 5.2, 1.45, "Primary review comparison", "5% global budget = 415 breasts per method\nCombined: 2/39 misses flagged\nCancer score alone: 3/39 misses flagged", "#E8F2F8")
    arrow((8.55, 3.30), (8.55, 2.35))
    ax.text(.2, 4.5, "Five fixed external GMIC models\nNo RSNA weight training\nNo breast exclusions", fontsize=10, va="top", linespacing=1.6)
    fig.suptitle("Study flow and evaluation denominators", x=.05, ha="left", weight="bold", fontsize=15)
    ax.text(.2, .15, "Review flagging was measured retrospectively; human-reader recovery was not measured.", fontsize=9, color="#475569")
    save(fig, directory, "figure_1_study_flow")


def primary_figure(result, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.07, right=.98, top=.78, bottom=.24, wspace=.48)
    a, b = axes
    primary = result["primary"]
    records = [primary["baseline"], primary["candidate"]]
    values = [r["fn_capture_fraction"]*100 for r in records]
    a.bar([0, 1], values, color=[COLORS[0], COLORS[-1]], width=.58)
    for i, (r, v) in enumerate(zip(records, values)):
        a.text(i, v+.4, f"{r['false_negatives_flagged']}/39\n{v:.2f}%", ha="center", fontsize=11)
    a.set(xticks=[0, 1], xticklabels=["Cancer score\nalone", "Combined\nscore"], ylim=(0, 11.4), ylabel="Missed positive breasts flagged (%)")
    a.set_title("A   Equal capacity: 415 breasts each", loc="left", fontsize=11, pad=13)
    difference = primary["capture_difference"]*100
    for y, scheme, color in [(1, "conditional", COLORS[-1]), (0, "calibration_inclusive", "#64748B")]:
        lo, hi = np.array(result["bootstrap"][scheme]["ci95"])*100
        b.errorbar(difference, y, xerr=[[difference-lo], [hi-difference]], fmt="o", color=color, capsize=5, linewidth=2)
        b.text(difference, y+.23, f"{difference:.2f} [{lo:.2f}, {hi:+.2f}]", ha="center", fontsize=10)
    b.axvline(0, color="#94A3B8", linestyle="--", linewidth=1)
    b.set(xlim=(-19, 13), ylim=(-.6, 1.7), yticks=[1, 0], yticklabels=["Fixed calibration\n(primary)", "Calibration refit\n(secondary)"], xlabel="Combined minus cancer score alone\nCapture difference (percentage points)")
    b.set_title("B   Paired patient-bootstrap 95% intervals", loc="left", fontsize=11, pad=13)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
    a.grid(axis="y", color="#E2E8F0", zorder=0); a.set_axisbelow(True)
    fig.suptitle("Primary comparison at 5% review capacity", x=.055, ha="left", weight="bold", fontsize=15)
    fig.text(.055, .055, "39 AI-missed positive breasts among 8,315 AI-negative breasts. 2,000 draws per interval.\nThe intervals cross zero; incremental benefit is inconclusive.", fontsize=10, color="#475569")
    save(fig, directory, "figure_2_primary_comparison")


def curve_figure(result, directory):
    fig, ax = plt.subplots(figsize=(9.7, 5.7))
    fig.subplots_adjust(left=.09, right=.64, top=.84, bottom=.20)
    markers = ["o", "s", "^", "D", "v", "o"]
    for i, method in enumerate(LABELS):
        rows = sorted((r for r in result["review"] if r["method"] == method), key=lambda r: r["budget"])
        x = [r["actual_review_fraction"]*100 for r in rows]
        y = [r["fn_capture_fraction"]*100 for r in rows]
        ax.plot(x, y, label=LABELS[method], color=COLORS[i], marker=markers[i], markersize=3.4,
                linewidth=2.2 if i in (0, 5) else 1.3, linestyle="-" if i in (0, 5) else "--")
    ax.plot([0, 20], [0, 20], label="Random-selection expectation", color="#94A3B8", linestyle=":", linewidth=1.8)
    ax.axvline(415/8315*100, color="#94A3B8", linestyle="--", linewidth=1)
    ax.text(5.2, 47, "Primary\n5% budget", fontsize=9, color="#475569")
    ax.set(xlim=(0, 20), ylim=(0, 52), xticks=[0, 5, 10, 15, 20], ylabel="Missed positive breasts flagged (%)", xlabel="AI-negative breasts selected for review (%)")
    ax.grid(color="#E2E8F0", linewidth=.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False, fontsize=9, labelspacing=1.1)
    fig.suptitle("Review capacity and false-negative capture", x=.065, ha="left", weight="bold", fontsize=15)
    fig.text(.065, .04, "All six frozen methods; 39 missed positive breasts. Lines connect the 21 evaluated budgets.\nSelections use one global budget; plotted x positions reflect floor rounding. No pointwise intervals are shown.", fontsize=9, color="#475569")
    save(fig, directory, "figure_3_capture_curves")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def write_tables(result, config, directory):
    text = ["# Frozen aggregate tables", "Generated from the completed aggregate result and frozen configuration. No new model evaluation or inference is performed."]
    text += ["## 1. Cohort", table(["Group", "Patients", "Breasts", "Positive breasts"], [[group.title(), f"{v['patients']:,}", f"{v['breasts']:,}", v['positive_breasts']] for group, v in config["roles"].items() if isinstance(v, dict)])]
    text += ["## 2. Classification", table(["Group", "Breasts", "Positive", "TP", "FN", "TN", "FP", "Sensitivity", "Specificity", "ROC-AUC", "AP"], [[r['group'], r['n_breasts'], r['n_positive'], r['tp'], r['fn'], r['tn'], r['fp'], f"{100*r['sensitivity']:.2f}%" if r['sensitivity'] is not None else 'NA', f"{100*r['specificity']:.2f}%" if r['specificity'] is not None else 'NA', f"{r['roc_auc']:.5f}" if r['roc_auc'] is not None else 'NA', f"{r['average_precision']:.5f}" if r['average_precision'] is not None else 'NA'] for r in result['classification']])]
    text += ["## 3. Review at the four prespecified budgets", table(["Method", "Budget", "Selected breasts", "Misses flagged / all misses", "Capture", "Yield"], [[LABELS[r['method']], f"{100*r['budget']:g}%", r['reviewed_breasts'], f"{r['false_negatives_flagged']}/{r['false_negatives']}", f"{100*r['fn_capture_fraction']:.2f}%", f"{100*r['review_yield']:.2f}%"] for r in result['review'] if r['budget'] in config['review']['report_budgets']])]
    difference = 100*result['primary']['capture_difference']
    text += ["## 4. Primary difference and uncertainty", table(["Bootstrap scheme", "Difference (percentage points)", "95% interval", "Valid draws"], [[scheme, f"{difference:.2f}", f"[{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:+.2f}]", r['defined']] for scheme, r in result['bootstrap'].items()]), "Difference = combined minus cancer score alone. Draws resample patients within site, retain both breasts and use the same holdout draws for paired comparisons. An interval crossing zero does not establish equivalence."]
    text += ["## 5. Subgroups at 5% global review capacity", table(["Group", "Method", "AI-negative breasts", "Misses", "Selected in group", "Misses flagged", "Capture"], [[r['group'], LABELS[r['method']], r['negative_breasts'], r['false_negatives'], r['reviewed_breasts'], r['false_negatives_flagged'], f"{100*r['fn_capture_fraction']:.2f}%" if r['fn_capture_fraction'] is not None else 'NA'] for r in result['subgroup_review'] if r['global_budget'] == .05]), "Selection is global; no subgroup receives its own 5% budget. Density groups are available only at site 1; site 2 has no recorded density. These sparse subgroup comparisons are descriptive."]
    text += ["## 6. Probability quality", table(["Group", "Score", "Brier", "Log loss"], [[r['group'], r['score'], f"{r['brier']:.6f}", f"{r['log_loss']:.6f}"] for r in result['calibration']]), "Calibration was fitted only on development patients. The constant comparator uses development prevalence. Lower values are better; study-sample performance does not establish population calibration."]
    (directory / 'TABLES.md').write_text('\n\n'.join(text)+'\n')
    # Exact machine-readable tables retain all budgets, subgroup records and bins.
    for key in ['classification', 'review', 'subgroup_review', 'calibration', 'calibration_bins', 'selection_overlap']:
        (directory / f'table_{key}.json').write_text(json.dumps(result[key], indent=2)+'\n')
    (directory / 'CAPTIONS.md').write_text('''# Figure captions

**Figure 1. Study flow.** Original audited cohort, patient-separated development and holdout groups, frozen classification and the primary review comparison. Of 54,706 original images, 38 nonstandard-view images were excluded from model input; no breasts were excluded. Both breasts of a patient remained in the same group. Five inference shards implement one holdout analysis, not cross-validation.

**Figure 2. Primary comparison.** Each method selected 415 of 8,315 AI-negative breasts (nominal 5% capacity; realized 4.991%). The combined score flagged 2 of the 39 cancer-positive misses; cancer score alone flagged 3. The capture difference was −2.56 percentage points. The primary fixed-calibration paired patient-bootstrap 95% interval was [−13.79, +8.11] points; the secondary development-calibration-refit interval was [−14.29, +8.00] points. Each scheme used 2,000 valid draws. Resampling preserved site patient counts and both breasts, and review capacity was recomputed globally in each draw. The comparison does not establish added benefit, equivalence or clinical recovery.

**Figure 3. Review-capacity curves.** All six frozen methods across 21 budgets from 0% to 20%, evaluated on the same 39 AI-missed cancer-positive breasts. Points are joined for readability; no intermediate budgets or pointwise confidence intervals were estimated. The random-selection line is an expectation, not a simulated or observed comparator. The primary comparison remains the frozen 5% budget; other budgets are secondary. Cancer score is the calibrated malignant score; disagreement is absolute CC–MLO difference; variance is between-member population variance. Combined methods average the relevant development-reference percentile scores.

These figures describe retrospective flagging against dataset labels. Human-reader recovery, clinical outcomes and interval-cancer detection were not measured. Source: validation/fixed_gmic_final_results_20261001.json and fixed_gmic_final_config.json; all figure/table generation was performed after the final outcomes were opened.
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'reports'
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads((root / 'validation/fixed_gmic_final_results_20261001.json').read_text())
    config = json.loads((root / 'fixed_gmic_final_config.json').read_text())
    verification = check_inputs(root, result)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'fixed-gmic-20261001'})
    cohort_figure(result, config, output)
    primary_figure(result, output)
    curve_figure(result, output)
    write_tables(result, config, output)
    (output / 'figure_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print('Verified aggregate inputs; generated 3 figures in PDF, SVG and PNG plus 6 table sections.')


if __name__ == '__main__':
    main()
