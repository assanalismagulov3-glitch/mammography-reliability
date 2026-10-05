"""Calibration and review-budget evaluation, with patient-cluster uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .audit import BREAST_KEY, require


def checked_probability(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    require(p.ndim == 1 and len(p) > 0, "Expected a nonempty probability vector.")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities.")
    return p


def checked_labels(y, length: int) -> np.ndarray:
    y = np.asarray(y)
    require(y.ndim == 1 and len(y) == length and np.isin(y, [0, 1]).all(), "Expected matching binary labels.")
    return y.astype(int)


@dataclass
class MonotonePlatt:
    """Positive-slope logistic calibration; fit only on calibration patients."""
    slope: float = 1.0
    intercept: float = 0.0

    def fit(self, p, y):
        p = checked_probability(p)
        y = checked_labels(y, len(p))
        require(len(np.unique(y)) == 2, "Calibration needs both classes.")
        z = logit(np.clip(p, 1e-7, 1 - 1e-7))

        def objective(params):
            q = np.exp(params[0]) * z + params[1]
            return float(np.mean(np.logaddexp(0, q) - y * q) + 1e-6 * np.sum(np.square(params)))

        fitted = minimize(objective, [0.0, 0.0], method="L-BFGS-B", bounds=[(-4, 4), (-20, 20)])
        require(fitted.success and np.isfinite(fitted.fun), f"Calibration did not converge: {fitted.message}")
        self.slope, self.intercept = float(np.exp(fitted.x[0])), float(fitted.x[1])
        return self

    def predict(self, p):
        p = checked_probability(p)
        return expit(self.slope * logit(np.clip(p, 1e-7, 1 - 1e-7)) + self.intercept)


def sensitivity_threshold(p, y, target: float = 0.90) -> float:
    p = checked_probability(p)
    y = checked_labels(y, len(p))
    require(0 < target <= 1, "Sensitivity target must be in (0, 1].")
    positives = np.sort(p[y == 1])[::-1]
    require(len(positives) > 0, "No positive calibration breasts.")
    # Prediction is positive for p >= threshold. Tied scores are all positive.
    return float(positives[math.ceil(target * len(positives)) - 1])


def classification_metrics(y, p, threshold: float) -> dict:
    p = checked_probability(p)
    y = checked_labels(y, len(p))
    require(np.isfinite(threshold) and 0 <= threshold <= 1, "Invalid fixed threshold.")
    positive = p >= threshold
    tp, fn = int(((y == 1) & positive).sum()), int(((y == 1) & ~positive).sum())
    tn, fp = int(((y == 0) & ~positive).sum()), int(((y == 0) & positive).sum())
    safe_ratio = lambda a, b: float(a / b) if b else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(float((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()))
              for b in range(10) if (bins == b).any())
    both = len(np.unique(y)) == 2
    return {"n_breasts": len(y), "n_positive": int(y.sum()), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": safe_ratio(tp, tp + fn), "specificity": safe_ratio(tn, tn + fp),
            "ppv_in_study_sample": safe_ratio(tp, tp + fp), "npv_in_study_sample": safe_ratio(tn, tn + fn),
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece_10_equal_width_bins": ece, "threshold": float(threshold)}


def ensemble_features(predictions: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    keys = BREAST_KEY + ["seed"]
    required = set(keys + ["z_cc", "z_mlo"])
    require(required.issubset(predictions), "Prediction schema is incomplete.")
    require(not predictions[keys].duplicated().any(), "Duplicate breast/member predictions.")
    require(len(expected_seeds) >= 2 and len(set(expected_seeds)) == len(expected_seeds), "Ensemble needs distinct members.")
    require(set(predictions["seed"].unique()) == set(expected_seeds), "Unexpected or missing ensemble seed.")
    require((predictions.groupby(BREAST_KEY)["seed"].nunique() == len(expected_seeds)).all(), "Incomplete ensemble for a breast.")
    require(np.isfinite(predictions[["z_cc", "z_mlo"]].to_numpy()).all(), "Nonfinite model predictions.")
    tmp = predictions.copy()
    tmp["p_member"] = expit((tmp["z_cc"] + tmp["z_mlo"]) / 2)
    tmp["p_cc"] = expit(tmp["z_cc"])
    tmp["p_mlo"] = expit(tmp["z_mlo"])
    group = tmp.groupby(BREAST_KEY)
    result = group.agg(p_raw=("p_member", "mean"), p_cc=("p_cc", "mean"), p_mlo=("p_mlo", "mean"))
    result["ensemble_variance"] = group["p_member"].var(ddof=0)
    result["view_disagreement"] = abs(result["p_cc"] - result["p_mlo"])
    return result.reset_index()


def reference_percentile(reference, values) -> np.ndarray:
    """Midrank empirical CDF learned from calibration negatives only."""
    ref = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    require(len(ref) > 0 and np.isfinite(ref).all() and np.isfinite(values).all(), "Invalid percentile reference.")
    return (np.searchsorted(ref, values, side="left") + np.searchsorted(ref, values, side="right")) / (2 * len(ref))


def add_review_scores(calibration: pd.DataFrame, target: pd.DataFrame, threshold: float) -> pd.DataFrame:
    reference = calibration.loc[calibration["p_calibrated"] < threshold]
    require(len(reference) > 0, "No calibration negatives for score normalization.")
    result = target.copy()
    result["score_confidence"] = result["p_calibrated"]
    result["score_disagreement"] = result["view_disagreement"]
    result["score_uncertainty"] = result["ensemble_variance"]
    normalized = []
    for col in ["p_calibrated", "view_disagreement", "ensemble_variance"]:
        normalized.append(reference_percentile(reference[col], result[col]))
    result["score_confidence_disagreement"] = (normalized[0] + normalized[1]) / 2
    result["score_confidence_uncertainty"] = (normalized[0] + normalized[2]) / 2
    result["score_combined"] = sum(normalized) / 3
    result["predicted_positive"] = result["p_calibrated"] >= threshold
    return result


def review_selection(frame: pd.DataFrame, score_column: str, budget: float = 0.05, seed: int = 20260909) -> pd.Series:
    require(0 <= budget <= 1, "Review budget must be between zero and one.")
    require(frame.index.is_unique, "Reset the dataframe index before selecting review candidates.")
    require(not frame[BREAST_KEY].duplicated().any(), "Review input must contain unique breasts.")
    require(frame["predicted_positive"].isin([True, False]).all(), "Missing decision flags.")
    checked_labels(frame["cancer"], len(frame))
    require(np.isfinite(frame[score_column]).all(), "Nonfinite review scores.")
    negative = frame.loc[~frame["predicted_positive"].astype(bool)].copy()
    # Tie ordering uses only identifiers and a fixed seed, never labels or table order.
    negative["tie"] = [hashlib.sha256(f"{seed}:{p}:{l}".encode()).hexdigest()
                       for p, l in negative[BREAST_KEY].itertuples(index=False, name=None)]
    negative = negative.sort_values([score_column, "tie"], ascending=[False, True])
    k = math.floor(budget * len(negative))
    selection = pd.Series(False, index=frame.index)
    selection.loc[negative.iloc[:k].index] = True
    return selection


def review_at_budget(frame: pd.DataFrame, score_column: str, budget: float = 0.05, seed: int = 20260909) -> dict:
    selected = review_selection(frame, score_column, budget, seed)
    negative = frame.loc[~frame["predicted_positive"].astype(bool)]
    k = int(selected.sum())
    fn = int(negative["cancer"].sum())
    caught = int(frame.loc[selected, "cancer"].sum())
    return {"method": score_column, "budget": budget, "negative_breasts": len(negative),
            "reviewed_breasts": k, "false_negatives": fn, "false_negatives_flagged": caught,
            "fn_capture_fraction": float(caught / fn) if fn else None,
            "review_yield": float(caught / k) if k else None,
            "actual_review_fraction": float(k / len(negative)) if len(negative) else 0.0,
            "random_expected_fn_capture": float(k / len(negative)) if len(negative) and fn else None,
            "interpretation": "flagging potential review candidates; no human review or cancer recovery was measured"}


def paired_cluster_bootstrap(frame: pd.DataFrame, method: str = "score_combined", baseline: str = "score_confidence",
                             budget: float = 0.05, repeats: int = 2000, seed: int = 20260909) -> dict:
    """Patient bootstrap of pooled capture difference, allocating budget per fold.

    Multiplicity weights represent sampled patient clusters, including both breasts.
    All fitted scores/thresholds stay fixed. A replicate with zero total false
    negatives is undefined. This does not include model-retraining uncertainty.
    """
    require(repeats > 0, "Bootstrap repetitions must be positive.")
    require(0 <= budget <= 1, "Review budget must be between zero and one.")
    require(not frame[BREAST_KEY].duplicated().any(), "Expected unique held-out breasts.")
    require(frame.groupby("patient_id")["outer_fold"].nunique().max() == 1, "Patient spans held-out folds.")
    rng = np.random.default_rng(seed)
    fold_groups = []
    for _, fold in frame.groupby("outer_fold"):
        # Validation and point-estimate routines use the same fixed tie ordering.
        review_selection(fold, method, budget, seed)
        review_selection(fold, baseline, budget, seed)
        patients = np.sort(fold["patient_id"].unique())
        negative = fold.loc[~fold["predicted_positive"].astype(bool)]
        group_idx = np.searchsorted(patients, negative["patient_id"].to_numpy())
        labels = negative["cancer"].to_numpy(dtype=int)
        ties = np.array([hashlib.sha256(f"{seed}:{p}:{l}".encode()).hexdigest()
                         for p, l in negative[BREAST_KEY].itertuples(index=False, name=None)])
        orders = [np.lexsort((ties, -negative[col].to_numpy(dtype=float))) for col in [method, baseline]]
        fold_groups.append((len(patients), group_idx, labels, orders))
    estimates, undefined = [], 0
    for _ in range(repeats):
        total_fn, total_difference = 0, 0
        for n_patients, group_idx, labels, orders in fold_groups:
            multiplicity = rng.multinomial(n_patients, np.full(n_patients, 1 / n_patients))
            weight = multiplicity[group_idx]
            total_fn += int(weight @ labels)
            k = math.floor(budget * weight.sum())
            caught = []
            for order in orders:
                ordered_weight = weight[order]
                preceding = np.cumsum(ordered_weight) - ordered_weight
                selected_weight = np.minimum(ordered_weight, np.maximum(0, k - preceding))
                caught.append(int(selected_weight @ labels[order]))
            total_difference += caught[0] - caught[1]
        if total_fn == 0:
            undefined += 1
        else:
            estimates.append(float(total_difference / total_fn))
    point_difference, point_fn = 0, 0
    for _, fold in frame.groupby("outer_fold"):
        a = review_at_budget(fold, method, budget, seed)
        b = review_at_budget(fold, baseline, budget, seed)
        point_difference += a["false_negatives_flagged"] - b["false_negatives_flagged"]
        point_fn += a["false_negatives"]
    result = {"method": method, "baseline": baseline, "budget": budget,
              "estimand": "pooled FN capture difference with the review budget applied separately in each held-out fold",
              "point_estimate": float(point_difference / point_fn) if point_fn else None,
              "valid_bootstrap_replicates": len(estimates), "undefined_replicates": undefined,
              "requested_replicates": repeats, "ci95": None,
              "uncertainty_scope": "conditional on fitted models, calibration, and threshold; not model retraining"}
    if len(estimates) >= max(20, math.ceil(repeats * 0.8)) and result["point_estimate"] is not None:
        result["ci95"] = [float(x) for x in np.quantile(estimates, [0.025, 0.975])]
    return result
