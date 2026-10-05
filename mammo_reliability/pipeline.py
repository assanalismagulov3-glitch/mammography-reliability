"""Command-line workflow. Audits and training records stay in a private directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import (BREAST_KEY, STANDARD_VIEWS, audit_metadata, environment, file_sha256,
                    locate_dataset, require, write_json)
from .evaluation import (MonotonePlatt, add_review_scores, classification_metrics, ensemble_features,
                         paired_cluster_bootstrap, review_at_budget, review_selection, sensitivity_threshold)
from .study import (freeze_protocol, make_study_patients, pilot_roles,
                    require_frozen_protocol, study_outer_roles)


def initialize(data_root: Path, run_root: Path, config: dict):
    images = pd.read_csv(data_root / "train.csv")
    breasts, summary, tables = audit_metadata(images)
    patients = make_study_patients(breasts, config["n_outer_folds"], config["split_seed"],
                                   config["pilot_fraction"], config["pilot_tune_fraction"])
    run_root.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "metadata_sha256": file_sha256(data_root / "train.csv")}
    identity_path = run_root / "experiment.json"
    if identity_path.exists():
        require(json.loads(identity_path.read_text()) == identity, "Different source/config already exists in this directory.")
    else:
        write_json(identity_path, identity)
    (run_root / "aggregates").mkdir(exist_ok=True)
    write_json(run_root / "aggregates" / "data_audit.json", summary)
    write_json(run_root / "environment.json", environment())
    for name, table in tables.items():
        table.to_csv(run_root / "aggregates" / f"{name}.csv", index=False)
    breasts.to_csv(run_root / "breasts_private.csv", index=False)
    images.loc[images.view.isin(STANDARD_VIEWS), ['patient_id', 'image_id', 'laterality', 'view']].sort_values(
        'image_id').to_csv(run_root / 'standard_images_private.csv', index=False)
    patients.to_csv(run_root / "patients_private.csv", index=False)
    pilot_folder = run_root / 'pilot'
    pilot_folder.mkdir(exist_ok=True)
    pilot_roles(patients).to_csv(pilot_folder / 'roles_private.csv', index=False)
    patients.groupby(['cohort', 'site_id', 'patient_cancer']).size().rename('patients').to_csv(
        run_root / 'aggregates' / 'cohort_patient_counts.csv')
    breasts.merge(patients[['patient_id', 'cohort', 'pilot_role']], on='patient_id', validate='many_to_one').groupby(
        ['cohort', 'pilot_role', 'site_id', 'cancer']).size().rename('breasts').to_csv(
            run_root / 'aggregates' / 'cohort_breast_counts.csv')
    for fold in range(config["n_outer_folds"]):
        folder = run_root / f"fold_{fold}"
        folder.mkdir(exist_ok=True)
        roles = study_outer_roles(patients, fold, config["split_seed"])
        roles.to_csv(folder / "roles_private.csv", index=False)
        roles.groupby(["role", "site_id", "patient_cancer"]).size().rename("patients").to_csv(folder / "partition_counts.csv")
    return images, breasts, patients, summary


def require_pixel_audit(run_root: Path, config: dict):
    from .imaging import check_cross_patient_duplicates
    root = run_root / "preprocessed"
    complete = json.loads((root / "pixel_audit_complete.json").read_text())
    settings = json.loads((root / "preprocessing.json").read_text())
    identity = json.loads((run_root / "experiment.json").read_text())
    require(complete["metadata_sha256"] == identity["metadata_sha256"], "Pixel audit uses another dataset.")
    require(settings["height"] == config["model"]["height"] and settings["width"] == config["model"]["width"], "Preprocessing resolution differs from model.")
    require(file_sha256(root / "manifest.csv") == complete["manifest_sha256"], "Pixel manifest changed after audit.")
    manifest = pd.read_csv(root / "manifest.csv")
    require(len(manifest) == complete["n_images"], "Pixel audit count mismatch.")
    expected = pd.read_csv(run_root / 'standard_images_private.csv')
    columns = ['patient_id', 'image_id', 'laterality', 'view']
    require(manifest.image_id.is_unique and
            manifest[columns].sort_values('image_id').reset_index(drop=True).equals(expected[columns]),
            'Pixel audit does not cover the complete study image cohort.')
    check_cross_patient_duplicates(manifest)
    return manifest


def fit_fold(run_root: Path, breasts: pd.DataFrame, config: dict, fold: int):
    from .training import train_member
    require_frozen_protocol(run_root, config)
    manifest = require_pixel_audit(run_root, config)
    folder = run_root / f"fold_{fold}"
    roles = pd.read_csv(folder / "roles_private.csv")
    metadata_hash = json.loads((run_root / "experiment.json").read_text())["metadata_sha256"]
    for seed in config["ensemble_seeds"]:
        train_member(manifest, breasts, roles, config["model"], seed, folder / f"seed_{seed}", metadata_hash)


def evaluate_fold(run_root: Path, breasts: pd.DataFrame, config: dict, fold: int):
    from .training import predict_members, source_fingerprint
    require_frozen_protocol(run_root, config)
    manifest = require_pixel_audit(run_root, config)
    folder = run_root / f"fold_{fold}"
    roles = pd.read_csv(folder / "roles_private.csv")
    label_table = breasts.merge(roles[["patient_id", "role"]], on="patient_id", validate="many_to_one")
    frames = {}
    for role in ["calibrate", "test"]:
        subset = label_table.loc[label_table["role"] == role].copy()
        members = predict_members(manifest, subset, config["model"], config["ensemble_seeds"], folder)
        require(len(members) == len(subset) * len(config["ensemble_seeds"]), "Incomplete breast/member predictions.")
        members.to_csv(folder / f"{role}_member_predictions_private.csv", index=False)
        frames[role] = ensemble_features(members, config["ensemble_seeds"]).merge(subset, on=BREAST_KEY, validate="one_to_one")
    cal, test = frames["calibrate"], frames["test"]
    calibrator = MonotonePlatt().fit(cal["p_raw"], cal["cancer"])
    for frame in [cal, test]:
        frame["p_calibrated"] = calibrator.predict(frame["p_raw"])
    threshold = sensitivity_threshold(cal["p_calibrated"], cal["cancer"], config["sensitivity_target"])
    scored = add_review_scores(cal, test, threshold)
    scored["outer_fold"] = fold
    scored["threshold"] = threshold
    methods = [name for name in scored if name.startswith("score_")]
    review_rows = []
    for budget in config["review_budgets"]:
        for method in methods:
            review_rows.append({"outer_fold": fold, **review_at_budget(scored, method, budget, config["split_seed"])})
    metrics = {"outer_fold": fold, "source_sha256": source_fingerprint(),
               "calibration": {"slope": calibrator.slope, "intercept": calibrator.intercept,
                               "n_breasts": len(cal), "n_positive": int(cal["cancer"].sum())},
               "target_calibration_sensitivity": config["sensitivity_target"],
               "calibration_operating_point": classification_metrics(cal["cancer"], cal["p_calibrated"], threshold),
               "held_out_test": classification_metrics(scored["cancer"], scored["p_calibrated"], threshold)}
    pd.DataFrame(review_rows).to_csv(folder / "review_results.csv", index=False)
    subgroup_rows = []
    # Subgroups inherit the review selections from the full test fold.
    # Their budget is NOT reset, which would describe a different policy.
    for method in methods:
        selection = review_selection(scored, method, config["primary_review_budget"], config["split_seed"])
        grouped = scored.copy()
        grouped["selected"] = selection
        grouped["density_group"] = grouped["density_group"].fillna("Missing/conflicting")
        for group_columns in [["site_id"], ["site_id", "density_group"], ["site_id", "has_extra_images"]]:
            for group_name, sub in grouped.groupby(group_columns):
                fn = int(((sub["cancer"] == 1) & ~sub["predicted_positive"]).sum())
                flagged = int(((sub["cancer"] == 1) & sub["selected"]).sum())
                subgroup_rows.append({"outer_fold": fold, "group_by": "+".join(group_columns),
                                      "group": str(group_name), "method": method, "breasts": len(sub),
                                      "positive_breasts": int(sub["cancer"].sum()), "false_negatives": fn,
                                      "reviewed": int(sub["selected"].sum()), "false_negatives_flagged": flagged,
                                      "fn_capture_fraction": flagged / fn if fn else None})
    pd.DataFrame(subgroup_rows).to_csv(folder / "subgroup_review_results.csv", index=False)
    scored.to_csv(folder / "test_scored_private.csv", index=False)
    write_json(folder / "evaluation.json", metrics)
    return metrics


def summarize(run_root: Path, breasts: pd.DataFrame, config: dict) -> dict:
    require_frozen_protocol(run_root, config)
    patients = pd.read_csv(run_root / 'patients_private.csv')
    breasts = breasts.loc[breasts.patient_id.isin(patients.loc[patients.cohort == 'evaluation', 'patient_id'])]
    frames, reviews, checks = [], [], []
    for fold in range(config["n_outer_folds"]):
        folder = run_root / f"fold_{fold}"
        frames.append(pd.read_csv(folder / "test_scored_private.csv"))
        reviews.append(pd.read_csv(folder / "review_results.csv"))
        checks.append(json.loads((folder / "evaluation.json").read_text()))
    all_test = pd.concat(frames, ignore_index=True)
    require(not all_test[BREAST_KEY].duplicated().any(), "A breast occurs in multiple held-out folds.")
    require(set(map(tuple, all_test[BREAST_KEY].to_numpy())) == set(map(tuple, breasts[BREAST_KEY].to_numpy())), "Cross-validation omitted or added breasts.")
    require(len({item["source_sha256"] for item in checks}) == 1, "Code changed between evaluated folds.")
    comparison = paired_cluster_bootstrap(all_test, budget=config["primary_review_budget"],
                                          repeats=config["bootstrap_repeats"], seed=config["split_seed"])
    pd.concat(reviews, ignore_index=True).to_csv(run_root / "aggregates" / "all_fold_review_results.csv", index=False)
    write_json(run_root / "aggregates" / "primary_comparison.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "prepare", "pilot", "freeze", "train", "evaluate", "summarize"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("/kaggle/working/mammo_private_v0_2"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument('--trial', default='baseline_1024_v1')
    parser.add_argument('--pilot-model-config', type=Path)
    parser.add_argument('--pilot-cache-root', type=Path)
    parser.add_argument('--development-review', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_root = args.data_root or locate_dataset()
    images, breasts, _, summary = initialize(data_root, args.run_root, config)
    require(0 <= args.fold < config["n_outer_folds"], "Invalid fold index.")
    if args.stage == "audit":
        print(json.dumps(summary, indent=2))
    elif args.stage == "prepare":
        from .imaging import prepare_images
        prepare_images(images, data_root, args.run_root / "preprocessed", config["model"]["height"],
                       config["model"]["width"], config["cache_images"])
    elif args.stage == "train":
        fit_fold(args.run_root, breasts, config, args.fold)
    elif args.stage == 'pilot':
        from .pilot import run_pilot
        overrides = json.loads(args.pilot_model_config.read_text()) if args.pilot_model_config else {}
        print(json.dumps(run_pilot(args.run_root, breasts, config, args.trial, overrides, args.pilot_cache_root), indent=2))
    elif args.stage == 'freeze':
        require(args.development_review is not None, 'Provide the concrete development review JSON.')
        print(json.dumps(freeze_protocol(args.run_root, config, args.development_review), indent=2))
    elif args.stage == "evaluate":
        print(json.dumps(evaluate_fold(args.run_root, breasts, config, args.fold), indent=2))
    else:
        print(json.dumps(summarize(args.run_root, breasts, config), indent=2))


if __name__ == "__main__":
    main()
