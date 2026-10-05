"""Validate original RSNA metadata without silently changing the cohort."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

BREAST_KEY = ["patient_id", "laterality"]
STANDARD_VIEWS = ["CC", "MLO"]
REQUIRED = ["site_id", "patient_id", "image_id", "laterality", "view", "cancer", "density"]
OUTCOME_ONLY = ["cancer", "biopsy", "invasive", "BIRADS", "difficult_negative_case"]


def require(condition: bool, message: str) -> None:
    """Unlike assert, this check is not disabled by Python's -O option."""
    if not bool(condition):
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict:
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "Pillow", "pydicom",
                 "torch", "torchvision", "pylibjpeg", "pylibjpeg-openjpeg"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def locate_dataset() -> Path:
    candidates = [Path("/kaggle/input/competitions/rsna-breast-cancer-detection"),
                  Path("/kaggle/input/rsna-breast-cancer-detection")]
    for root in candidates:
        if (root / "train.csv").is_file():
            return root
    raise FileNotFoundError("Attach the official RSNA competition dataset or provide its local root explicitly.")


def audit_metadata(images: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    require(len(images) > 0, "Metadata is empty.")
    require(set(REQUIRED).issubset(images.columns), f"Missing columns: {set(REQUIRED) - set(images.columns)}")
    for col in ["site_id", "patient_id", "image_id", "laterality", "view"]:
        require(images[col].notna().all(), f"Missing identifying information in {col}.")
    for col in ["site_id", "patient_id", "image_id"]:
        values = pd.to_numeric(images[col], errors="coerce")
        require((values.notna() & np.isfinite(values) & (values >= 0) & (values == np.floor(values))).all(),
                f"{col} must contain nonnegative integer identifiers.")
    require(images["image_id"].is_unique, "Repeated image IDs require investigation.")
    require(images["laterality"].isin(["L", "R"]).all(), "Unexpected laterality.")
    require(images["cancer"].isin([0, 1]).all(), "Missing or nonbinary cancer labels.")
    require((images.groupby("patient_id")["site_id"].nunique() == 1).all(), "A patient occurs at multiple sites.")
    grouped = images.groupby(BREAST_KEY, sort=True)
    require((grouped["cancer"].nunique(dropna=False) == 1).all(), "Cancer labels conflict within a breast.")
    observed_density = images["density"].dropna()
    require(observed_density.isin(list("ABCD")).all(), "Unexpected density category; inspect the source values.")

    breasts = grouped.agg(site_id=("site_id", "first"), cancer=("cancer", "first"),
                          n_images=("image_id", "size"), density=("density", "first"),
                          n_density_labels=("density", "count"),
                          n_density_values=("density", "nunique"))
    views = pd.crosstab(index=[images[c] for c in BREAST_KEY], columns=images["view"])
    views = views.reindex(index=breasts.index, columns=STANDARD_VIEWS, fill_value=0)
    breasts["n_cc"] = views["CC"]
    breasts["n_mlo"] = views["MLO"]
    breasts["has_both_views"] = (breasts["n_cc"] > 0) & (breasts["n_mlo"] > 0)
    breasts["density_missing"] = breasts["n_density_labels"] == 0
    breasts["density_partly_missing"] = ((breasts["n_density_labels"] > 0) &
                                           (breasts["n_density_labels"] < breasts["n_images"]))
    breasts["density_conflict"] = breasts["n_density_values"] > 1
    # The first available value is retained only if all available values agree.
    breasts.loc[breasts["density_conflict"], "density"] = np.nan
    breasts["density_group"] = breasts["density"].map({"A": "A/B", "B": "A/B", "C": "C/D", "D": "C/D"})
    breasts["has_extra_images"] = breasts["n_images"] > 2
    if "age" in images:
        breasts["age"] = grouped["age"].first()
        breasts["age_conflict"] = grouped["age"].nunique() > 1
        breasts.loc[breasts["age_conflict"], "age"] = np.nan
    breasts = breasts.reset_index()

    extra = breasts.groupby(["site_id", "cancer"], as_index=False).agg(
        breast_count=("cancer", "size"), breasts_with_extra_images=("has_extra_images", "sum"),
        extra_image_fraction=("has_extra_images", "mean"))
    density = breasts.groupby("site_id", as_index=False).agg(
        breast_count=("cancer", "size"), all_missing=("density_missing", "sum"),
        partly_missing=("density_partly_missing", "sum"), conflicts=("density_conflict", "sum"))
    tables = {
        "site_cancer": breasts.groupby(["site_id", "cancer"]).size().rename("breast_count").reset_index(),
        "extra_images": extra, "density_availability": density,
        "density_cancer": breasts.assign(density=breasts["density"].fillna("Missing/conflicting"))
            .groupby(["site_id", "density", "cancer"]).size().rename("breast_count").reset_index(),
        "image_views": images["view"].value_counts(dropna=False).rename_axis("view").rename("image_count").reset_index(),
        "missing_columns": images.isna().sum().rename_axis("column").rename("missing_image_rows").reset_index(),
    }
    summary = {
        "source": "computed_from_supplied_metadata", "image_rows": int(len(images)),
        "patients": int(images["patient_id"].nunique()), "breasts": int(len(breasts)),
        "positive_breasts": int(breasts["cancer"].sum()),
        "positive_patients": int(breasts.groupby("patient_id")["cancer"].max().sum()),
        "positive_breast_fraction": float(breasts["cancer"].mean()),
        "breasts_missing_standard_view": int((~breasts["has_both_views"]).sum()),
        "breasts_with_extra_images": int(breasts["has_extra_images"].sum()),
        "density_conflicts": int(breasts["density_conflict"].sum()),
        "pixel_duplicates_checked": False,
    }
    return breasts, summary, tables


def make_patient_folds(breasts: pd.DataFrame, n_splits: int = 5, seed: int = 20260909) -> pd.DataFrame:
    require(breasts[BREAST_KEY].duplicated().sum() == 0, "Expected one row per breast.")
    require(breasts["has_both_views"].all(), "Resolve breasts missing CC or MLO before defining the paired cohort.")
    patients = breasts.groupby("patient_id", as_index=False).agg(site_id=("site_id", "first"),
                                                               patient_cancer=("cancer", "max"))
    strata = patients["site_id"].astype(str) + ":" + patients["patient_cancer"].astype(str)
    require(strata.value_counts().min() >= n_splits, "Insufficient patients per site/cancer stratum for these folds.")
    patients["outer_fold"] = -1
    for fold, (_, test_idx) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(patients, strata)):
        patients.loc[test_idx, "outer_fold"] = fold
    require((patients["outer_fold"] >= 0).all(), "Some patients lack a fold.")
    return patients


def partition_outer_fold(patients: pd.DataFrame, fold: int, seed: int = 20260909) -> pd.DataFrame:
    require(patients["patient_id"].is_unique, "Split input must contain unique patients.")
    require(fold in patients["outer_fold"].unique(), "Unknown outer fold.")
    test = patients["outer_fold"] == fold
    dev = patients.loc[~test].copy()
    strata = dev["site_id"].astype(str) + ":" + dev["patient_cancer"].astype(str)
    # Fractions of the complete cohort: fit 60%, tune 10%, calibrate 10%, test 20%.
    fit_idx, held_idx = train_test_split(dev.index, test_size=0.25, stratify=strata, random_state=seed + fold)
    held = dev.loc[held_idx]
    held_strata = held["site_id"].astype(str) + ":" + held["patient_cancer"].astype(str)
    tune_idx, cal_idx = train_test_split(held.index, test_size=0.5, stratify=held_strata, random_state=seed + 100 + fold)
    result = patients.copy()
    result["role"] = "test"
    result.loc[fit_idx, "role"] = "fit"
    result.loc[tune_idx, "role"] = "tune"
    result.loc[cal_idx, "role"] = "calibrate"
    require((result.loc[test, "role"] == "test").all(), "Outer test patients leaked into development.")
    require((result.loc[~test, "role"] != "test").all(), "Development patient role was not assigned.")
    for role, frame in result.groupby("role"):
        require(set(frame["patient_cancer"].unique()) == {0, 1}, f"Role {role} lacks a class.")
    return result
