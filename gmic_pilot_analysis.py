"""Explicit probability pooling for fixed external GMIC development models.

This is separate from the original logit-pooling ResNet pipeline. It does not
calibrate, choose a threshold, select a member, or estimate a final endpoint.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from mammo_reliability.audit import BREAST_KEY, require

MEMBERS = list(range(1, 6))
PREDICTION_COLUMNS = [f'm{m}_{target}' for m in MEMBERS for target in ['benign', 'malignant']]


def aggregate_predictions(images, predictions):
    """Average repeats within view, then give CC and MLO equal weight."""
    require(images.image_id.is_unique and predictions.image_id.is_unique, 'Duplicate image identifiers.')
    require(set(images.image_id) == set(predictions.image_id), 'Incomplete or unexpected prediction coverage.')
    require(set(PREDICTION_COLUMNS).issubset(predictions), 'Missing model output columns.')
    values = predictions[PREDICTION_COLUMNS].to_numpy(dtype=float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), 'Invalid predictions.')
    require(images.view.isin(['CC', 'MLO']).all(), 'Nonstandard view in the pilot.')
    require(images.role.isin(['fit', 'tune']).all(), 'A nondevelopment role entered the pilot.')
    require(images.cancer.isin([0, 1]).all(), 'Invalid cancer labels.')
    groups = images.groupby(BREAST_KEY)
    require((groups.view.nunique() == 2).all(), 'A breast lacks a standard view.')
    for column in ['cancer', 'site_id', 'role']:
        require((groups[column].nunique(dropna=False) == 1).all(), f'Conflicting {column}.')
    require((images.groupby('patient_id').role.nunique() == 1).all(), 'Patient crosses pilot roles.')
    require(not set(PREDICTION_COLUMNS).intersection(images.columns), 'Predictions already in metadata.')
    joined = images.merge(predictions[['image_id'] + PREDICTION_COLUMNS], on='image_id', validate='one_to_one')
    malignant = [f'm{m}_malignant' for m in MEMBERS]
    views = joined.groupby(BREAST_KEY + ['view'])[malignant].mean()
    cc, mlo = views.xs('CC', level='view'), views.xs('MLO', level='view')
    require(cc.index.equals(mlo.index), 'CC and MLO breast identities do not match.')
    result = groups.agg(site_id=('site_id', 'first'), role=('role', 'first'),
                        cancer=('cancer', 'first'), n_images=('image_id', 'size'))
    for member, column in zip(MEMBERS, malignant):
        result[f'm{member}_cc'] = cc[column]
        result[f'm{member}_mlo'] = mlo[column]
        result[f'm{member}_breast'] = (cc[column] + mlo[column]) / 2
    members = result[[f'm{m}_breast' for m in MEMBERS]].to_numpy()
    result['p_raw'] = members.mean(axis=1)
    result['p_cc'] = cc.mean(axis=1)
    result['p_mlo'] = mlo.mean(axis=1)
    result['view_disagreement'] = abs(result.p_cc - result.p_mlo)
    result['ensemble_variance'] = members.var(axis=1, ddof=0)
    require(np.isfinite(result.p_raw).all(), 'Missing pooled predictions.')
    return result.reset_index()


def rank_metrics(frame, column):
    y, p = frame.cancer.to_numpy(), frame[column].to_numpy()
    return {'breasts': len(frame), 'positive_breasts': int(y.sum()),
            'prevalence': float(y.mean()),
            'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}


def summarize_development(breasts):
    rows = []
    for role, group in breasts.groupby('role', sort=True):
        # All fixed members are reported; no best-member selection is performed.
        for name, column in [('ensemble', 'p_raw')] + [(f'model_{m}', f'm{m}_breast') for m in MEMBERS]:
            rows.append({'role': role, 'model': name, 'site': 'pooled', **rank_metrics(group, column)})
        for site, subset in group.groupby('site_id'):
            rows.append({'role': role, 'model': 'ensemble', 'site': int(site), **rank_metrics(subset, 'p_raw')})
    return {'purpose': 'adaptive_reserved_development_only', 'model_fitted_on_rsna': False,
            'threshold_selected': False, 'calibration_fitted': False,
            'final_test_results_opened': False, 'metrics': rows,
            'interpretation': 'Released RSNA labels; development estimates after prior adaptive trials. Not final performance.'}
