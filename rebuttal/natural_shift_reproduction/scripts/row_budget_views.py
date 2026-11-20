"""Moderate context TYPE counts with expanded context ROW budgets.

Shared: five appendix roles. Specific: those five plus only three domain-aware
roles. Each type is evaluated at 256/512/1024/2048 maximum context rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances

from natural_components import (CONTINUOUS, SHARED_NAMES, distances, geometry,
                                mixed_domain_geometry, sample_rows)

BUDGETS = [256, 512, 1024, 2048]
SPECIFIC_EXTRA_NAMES = ["target_matched_bootstrap", "domain_ratio_bootstrap", "continuous_transport"]


def prepare_views(X, y, Q, features, seed=0):
    Z, V = geometry(X, Q)
    anchors = np.quantile(V, [.1, .3, .5, .7, .9], axis=0)
    near = pairwise_distances(Z, anchors, metric="sqeuclidean").min(1)
    target_distance = distances(Z, V)
    outlying = (Z**2).mean(1)
    d = pairwise_distances(Z, metric="sqeuclidean")
    np.fill_diagonal(d, np.inf)
    neighbors = np.argpartition(d, 9, axis=1)[:, :10]
    disagreement = 1. - np.maximum((y[neighbors] == 0).mean(1), (y[neighbors] == 1).mean(1))
    scores = {"target_neighbor": (near, .8),
              "representative": (target_distance+.35*outlying, .7),
              "locally_stable": (disagreement+.10*target_distance, .8),
              "low_conflict": (disagreement+.20*outlying, .75)}
    retained = {}
    for name, (score, fraction) in scores.items():
        ids = list(np.argsort(score, kind="stable")[:int(np.ceil(len(y)*fraction))])
        for label in np.unique(y):
            if not np.any(y[ids] == label):
                ids[-1] = int(np.flatnonzero(y == label)[np.argmin(score[y == label])])
        retained[name] = np.asarray(ids)
    centered, yc = X-X.mean(0), y-y.mean()
    association = np.abs((centered*yc[:, None]).sum(0)) / np.maximum(
        np.sqrt((centered**2).sum(0)*(yc**2).sum()), 1e-10)
    columns = np.delete(np.arange(X.shape[1]), int(np.argmax(association)))
    base_order = np.random.default_rng(seed).permutation(len(y))
    # Domain-aware extras use categorical one-hot geometry, never arithmetic
    # transport of category identifiers or state-specific geographic codes.
    mixed_X, mixed_Q = mixed_domain_geometry(X, Q[:1024], features)
    target = distances(mixed_X, mixed_Q)
    positive = target[target > 1e-9]
    bandwidth = max(float(np.median(positive)), 1e-6) if len(positive) else 1.
    match_weight = np.exp(np.clip(-target/bandwidth, -30., 0.))
    domain_y = np.r_[np.zeros(len(X)), np.ones(len(mixed_Q))]
    domain = LogisticRegression(C=.1, max_iter=300, random_state=seed).fit(np.r_[mixed_X, mixed_Q], domain_y)
    p = np.clip(domain.predict_proba(mixed_X)[:, 1], .01, .99)
    ratio = np.minimum(p/(1.-p)*len(X)/len(mixed_Q), 10.)**.5
    numeric = [i for i, name in enumerate(features) if name in CONTINUOUS]
    transported = X.copy()
    if numeric:
        sm, tm = np.median(X[:, numeric], 0), np.median(Q[:, numeric], 0)
        ss = np.maximum(np.percentile(X[:, numeric], 75, axis=0)-np.percentile(X[:, numeric], 25, axis=0), 1.)
        ts = np.maximum(np.percentile(Q[:, numeric], 75, axis=0)-np.percentile(Q[:, numeric], 25, axis=0), 1.)
        transported[:, numeric] = .5*X[:, numeric]+.5*((X[:, numeric]-sm)/ss*ts+tm)

    def materialize(budget):
        for name in SHARED_NAMES:
            if name == "feature_subspace":
                ids = base_order[:budget]
                yield name, X[ids][:, columns], y[ids], Q[:, columns], {
                    "removed_column": int(np.argmax(association)), "source_row_ids": ids}
                continue
            eligible = retained[name]
            count = min(budget, len(eligible))
            rng = np.random.default_rng(seed)
            mandatory = [int(rng.choice(eligible[y[eligible] == label])) for label in np.unique(y)]
            remainder = np.setdiff1d(eligible, mandatory)
            ids = np.r_[mandatory, rng.choice(remainder, count-len(mandatory), replace=False)]
            rng.shuffle(ids)
            yield name, X[ids], y[ids], Q, {"retained_pool_fraction": scores[name][1], "source_row_ids": ids}
        ids = sample_rows(y, match_weight, count=budget, offset=seed)
        yield "target_matched_bootstrap", X[ids], y[ids], Q, {"target_distance_exponent": 1., "source_row_ids": ids}
        ids = sample_rows(y, ratio, count=budget, offset=seed)
        yield "domain_ratio_bootstrap", X[ids], y[ids], Q, {"ratio_exponent": .5, "ratio_clip": 10., "source_row_ids": ids}
        ids = base_order[:budget]
        yield "continuous_transport", transported[ids], y[ids], Q, {
            "transport_strength": .5, "continuous_columns": numeric, "source_row_ids": ids}
    return materialize
