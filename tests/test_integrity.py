"""Artificial fixtures test software behavior; none are clinical results."""
import numpy as np
import pandas as pd
import pytest

from mammo_reliability.audit import audit_metadata, make_patient_folds, partition_outer_fold
from mammo_reliability.evaluation import (MonotonePlatt, add_review_scores, classification_metrics,
    ensemble_features, paired_cluster_bootstrap, reference_percentile, review_at_budget,
    review_selection, sensitivity_threshold)
from mammo_reliability.imaging import check_cross_patient_duplicates, letterbox, normalize_pixels


def artificial_metadata(n_patients=80):
    records = []
    for patient in range(n_patients):
        site = 1 + (patient % 2)
        cancer_patient = (patient // 2) % 2
        for side in ["L", "R"]:
            for view in ["CC", "MLO"]:
                records.append({"site_id": site, "patient_id": patient, "image_id": len(records),
                                "laterality": side, "view": view,
                                "cancer": int(cancer_patient and side == "L"),
                                "density": "C" if site == 1 else np.nan})
    return pd.DataFrame(records)


def test_patient_roles_never_separate_breasts_and_are_reproducible():
    breasts, _, _ = audit_metadata(artificial_metadata())
    patients = make_patient_folds(breasts)
    shuffled = make_patient_folds(breasts.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(patients, shuffled)
    for fold in range(5):
        roles = partition_outer_fold(patients, fold)
        combined = breasts.merge(roles[["patient_id", "role"]], on="patient_id", validate="many_to_one")
        assert combined.groupby("patient_id")["role"].nunique().max() == 1
        assert set(roles.loc[roles["role"] == "test", "patient_id"]) == set(patients.loc[patients["outer_fold"] == fold, "patient_id"])
        assert set(roles["role"]) == {"fit", "tune", "calibrate", "test"}


@pytest.mark.parametrize("problem", ["conflicting_cancer", "duplicate_image", "bad_label", "patient_crosses_site"])
def test_unsafe_metadata_stops(problem):
    frame = artificial_metadata()
    if problem == "conflicting_cancer":
        frame.loc[0, "cancer"] = 1
    elif problem == "duplicate_image":
        frame.loc[0, "image_id"] = frame.loc[1, "image_id"]
    elif problem == "bad_label":
        frame.loc[0, "cancer"] = np.nan
    else:
        frame.loc[0, "site_id"] = 2
    with pytest.raises(ValueError):
        audit_metadata(frame)


def test_density_missing_partial_and_conflict_remain_distinct():
    frame = artificial_metadata()
    frame.loc[0, "density"] = np.nan
    frame.loc[2, "density"] = "D"
    breasts, summary, _ = audit_metadata(frame)
    partial = breasts.query("patient_id == 0 and laterality == 'L'").iloc[0]
    conflict = breasts.query("patient_id == 0 and laterality == 'R'").iloc[0]
    assert partial.density_partly_missing and partial.density == "C"
    assert conflict.density_conflict and pd.isna(conflict.density)
    assert breasts.loc[breasts.site_id == 2, "density"].isna().all()
    assert summary["density_conflicts"] == 1


def test_extra_images_do_not_increase_breast_or_cancer_counts():
    frame = artificial_metadata()
    extra = frame.iloc[[8]].copy()
    extra["image_id"] = 99999
    b1, s1, _ = audit_metadata(frame)
    b2, s2, _ = audit_metadata(pd.concat([frame, extra], ignore_index=True))
    assert len(b1) == len(b2) and s1["positive_breasts"] == s2["positive_breasts"]
    assert s2["breasts_with_extra_images"] == 1


def test_sensitivity_threshold_ties_and_boundary():
    p, y = np.array([0.9, 0.7, 0.7, 0.2, 0.4]), np.array([1, 1, 1, 1, 0])
    t = sensitivity_threshold(p, y, 0.75)
    assert t == 0.7
    metrics = classification_metrics(y, p, t)
    assert metrics["tp"] == 3 and metrics["fn"] == 1
    assert metrics["sensitivity"] == 0.75
    with pytest.raises(ValueError):
        sensitivity_threshold([0.2], [0])


def test_calibration_corrects_constant_overconfidence_and_preserves_order():
    p = np.full(100, 0.8)
    y = np.r_[np.ones(10), np.zeros(90)]
    model = MonotonePlatt().fit(p, y)
    calibrated = model.predict(p)
    assert np.allclose(calibrated, 0.1, atol=0.003)
    assert np.all(np.diff(model.predict([0.1, 0.2, 0.4, 0.8])) > 0)
    assert model.slope > 0


def artificial_scores():
    return pd.DataFrame({"patient_id": [1, 1, 2, 2, 3, 3, 4, 4], "laterality": ["L", "R"] * 4,
                         "cancer": [1, 0, 0, 0, 1, 0, 0, 0], "predicted_positive": [False] * 8,
                         "score_confidence": [0.2, 0.3, 0.4, 0.8, 0.1, 0.5, 0.9, 0.7],
                         "score_combined": [0.99, 0.3, 0.4, 0.8, 0.98, 0.5, 0.9, 0.7],
                         "outer_fold": [0] * 8})


def test_review_budget_and_false_negative_denominator():
    frame = artificial_scores()
    result = review_at_budget(frame, "score_combined", 0.25)
    assert result["reviewed_breasts"] == 2
    assert result["false_negatives"] == 2 and result["fn_capture_fraction"] == 1.0
    zero = frame.copy()
    zero["cancer"] = 0
    assert review_at_budget(zero, "score_combined")["fn_capture_fraction"] is None
    assert review_at_budget(frame, "score_combined", 0.10)["reviewed_breasts"] == 0


def test_review_rejects_ambiguous_duplicate_index():
    frame = artificial_scores()
    frame.index = [0] * len(frame)
    with pytest.raises(ValueError, match="index"):
        review_selection(frame, "score_combined")


def test_tie_selection_ignores_labels_and_row_order():
    frame = artificial_scores()
    frame["score_combined"] = 1.0
    original = set(frame.loc[review_selection(frame, "score_combined", 0.5), ["patient_id", "laterality"]].itertuples(index=False, name=None))
    changed = frame.sample(frac=1, random_state=9).copy()
    changed["cancer"] = 1 - changed["cancer"]
    selected = set(changed.loc[review_selection(changed, "score_combined", 0.5), ["patient_id", "laterality"]].itertuples(index=False, name=None))
    assert original == selected


def test_member_ensemble_completeness_is_enforced():
    prediction = pd.DataFrame({"patient_id": [1, 1], "laterality": ["L", "L"],
                               "seed": [10, 20], "z_cc": [0, 2], "z_mlo": [0, 0]})
    values = ensemble_features(prediction, [10, 20])
    assert values.loc[0, "ensemble_variance"] > 0 and values.loc[0, "view_disagreement"] > 0
    with pytest.raises(ValueError):
        ensemble_features(prediction.iloc[:1], [10, 20])


def test_reference_normalization_never_uses_test_labels():
    calibration = pd.DataFrame({"p_calibrated": [0.1, 0.2, 0.9], "view_disagreement": [0.0, 0.1, 0.5],
                                "ensemble_variance": [0.0, 0.02, 0.2]})
    test = calibration.copy()
    test["cancer"] = [0, 0, 1]
    one = add_review_scores(calibration, test, 0.5)
    test["cancer"] = [1, 1, 0]
    two = add_review_scores(calibration, test, 0.5)
    pd.testing.assert_series_equal(one["score_combined"], two["score_combined"])
    assert reference_percentile([1, 1], [0, 1, 2]).tolist() == [0.0, 0.5, 1.0]


def test_cluster_bootstrap_is_paired_and_reports_undefined_draws():
    frame = artificial_scores()
    result = paired_cluster_bootstrap(frame, budget=0.25, repeats=100, seed=18)
    assert result["point_estimate"] == 1.0
    assert result["valid_bootstrap_replicates"] + result["undefined_replicates"] == 100
    identical = paired_cluster_bootstrap(frame, method="score_confidence", budget=0.25, repeats=100, seed=18)
    assert identical["point_estimate"] == 0 and identical["ci95"] == [0.0, 0.0]


def test_padding_and_monochrome_inversion():
    raw = np.array([[0, 10], [20, 30]], dtype=np.uint16)
    normal = normalize_pixels(raw, raw, "MONOCHROME2", padding_value=0)
    inverted = normalize_pixels(raw, raw, "MONOCHROME1", padding_value=0)
    assert normal[0, 0] == inverted[0, 0] == 0
    assert normal[0, 1] == 0 and normal[1, 1] == 1
    assert inverted[0, 1] == 1 and inverted[1, 1] == 0
    with pytest.raises(ValueError):
        normalize_pixels(raw, np.ones_like(raw), "MONOCHROME2")


def test_letterbox_keeps_all_edges():
    image = np.ones((4, 8), dtype=np.float32)
    output = letterbox(image, 8, 8)
    assert output.shape == (8, 8)
    assert output[2:6, :].min() == 1
    assert output[:2].max() == output[6:].max() == 0


def test_identical_pixels_under_different_patients_stop_training():
    same_patient = pd.DataFrame({"patient_id": [1, 1], "decoded_pixel_sha256": ["abc", "abc"]})
    check_cross_patient_duplicates(same_patient)
    same_patient.loc[1, "patient_id"] = 2
    with pytest.raises(ValueError):
        check_cross_patient_duplicates(same_patient)
