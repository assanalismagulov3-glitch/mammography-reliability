"""Sampling helpers for an isolated development experiment; no torch dependency."""
import numpy as np


def balanced_epoch_indices(labels, group_size, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError('A binary fitting cohort with both classes is required.')
    if group_size < 2 or group_size % 2:
        raise ValueError('The accumulation group must be positive and even.')
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    groups = []
    for _ in range((len(labels) + group_size - 1) // group_size):
        group = np.concatenate([rng.choice(positive, group_size // 2, replace=True),
                                rng.choice(negative, group_size // 2, replace=True)])
        rng.shuffle(group)
        groups.append(group)
    return np.concatenate(groups).astype(np.int64)


def objective_scale(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Both classes are required.')
    # Equal-class BCE times this constant has the same expected objective as
    # natural sampling with positive weight n_negative / n_positive.
    return float(2 * (labels == 0).mean())
