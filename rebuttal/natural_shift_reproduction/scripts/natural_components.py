"""Reconstructed TabACE contexts and the paper's classification fusion rule.

All constructors accept source labels and unlabeled query features only.
Continuous-column transport does not interpolate categorical codes.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder, RobustScaler

CONTINUOUS = {"AGEP", "WKHP", "POVPIP"}
SHARED_NAMES = ["target_neighbor", "representative", "locally_stable",
                "feature_subspace", "low_conflict"]


def normalize(p):
    a = np.clip(np.asarray(p, dtype=np.float64), 1e-10, 1.)
    return a / a.sum(-1, keepdims=True)


def geometry(X, Q):
    center = np.median(X, axis=0)
    scale = np.maximum(np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0), 1.)
    return np.clip((X - center) / scale, -10, 10), np.clip((Q - center) / scale, -10, 10)


def distances(Z, V):
    d = pairwise_distances(Z, V[:1024], metric="sqeuclidean")
    k = min(10, d.shape[1])
    return np.partition(d, k - 1, axis=1)[:, :k].mean(1)


def subset(y, score, fraction):
    order = np.argsort(score, kind="stable")
    n = max(len(np.unique(y)), int(np.ceil(len(y) * fraction)))
    ids = list(order[:n])
    for label in np.unique(y):
        if not np.any(y[ids] == label):
            candidate = next(int(i) for i in order if y[i] == label)
            for j in range(len(ids) - 1, -1, -1):
                if np.sum(y[ids] == y[ids[j]]) > 1:
                    ids[j] = candidate
                    break
    return np.asarray(ids)


def shared_views(X, y, Q):
    Z, V = geometry(X, Q)
    anchors = np.quantile(V, [.1, .3, .5, .7, .9], axis=0)
    near = pairwise_distances(Z, anchors, metric="sqeuclidean").min(1)
    target = distances(Z, V)
    outlying = (Z ** 2).mean(1)
    d = pairwise_distances(Z, metric="sqeuclidean")
    np.fill_diagonal(d, np.inf)
    k = min(10, len(y) - 1)
    neighbors = np.argpartition(d, k - 1, axis=1)[:, :k]
    consensus = np.maximum((y[neighbors] == 0).mean(1), (y[neighbors] == 1).mean(1))
    disagreement = 1. - consensus
    result = []
    for name, score, fraction in [
        ("target_neighbor", near, .8),
        ("representative", target + .35 * outlying, .7),
        ("locally_stable", disagreement + .10 * target, .8),
        ("low_conflict", disagreement + .20 * outlying, .75),
    ]:
        ids = subset(y, score, fraction)
        result.append((name, X[ids], y[ids], Q, {"fraction": fraction}))
    centered = X - X.mean(0)
    yc = y - y.mean()
    association = np.abs((centered * yc[:, None]).sum(0)) / np.maximum(
        np.sqrt((centered ** 2).sum(0) * (yc ** 2).sum()), 1e-10)
    drop = int(np.argmax(association))
    columns = np.delete(np.arange(X.shape[1]), drop)
    result.insert(3, ("feature_subspace", X[:, columns], y, Q[:, columns],
                      {"removed_column": drop}))
    assert [view[0] for view in result] == SHARED_NAMES
    return result


def sample_rows(y, weights, count=256, offset=0):
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    weights /= weights.sum()
    rng = np.random.default_rng(offset)
    # Preserve observed classes but otherwise retain the proposed sampling prior.
    mandatory = [int(rng.choice(np.flatnonzero(y == c))) for c in np.unique(y)]
    ids = np.r_[mandatory, rng.choice(len(y), count - len(mandatory), replace=True, p=weights)]
    rng.shuffle(ids)
    return ids


def mixed_domain_geometry(X, Q, features):
    numeric = [i for i, name in enumerate(features) if name in CONTINUOUS]
    categorical = [i for i in range(len(features)) if i not in numeric]
    parts_source, parts_query = [], []
    if numeric:
        scaler = RobustScaler().fit(X[:, numeric])
        parts_source.append(np.clip(scaler.transform(X[:, numeric]), -10, 10))
        parts_query.append(np.clip(scaler.transform(Q[:, numeric]), -10, 10))
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
        parts_source.append(encoder.fit_transform(X[:, categorical]))
        parts_query.append(encoder.transform(Q[:, categorical]))
    return np.concatenate(parts_source, 1), np.concatenate(parts_query, 1)


def specific_views(X, y, Q, pool_X, pool_y, features, scenario):
    """Domain-aware candidates; family is known, target labels remain unavailable.

    There are no genuine within-year timestamps in these ACS records. Thus the
    temporal reconstruction uses cross-year support/feature alignment, never
    treats arbitrary CSV row order as recency.
    """
    Z, V = mixed_domain_geometry(pool_X, Q[:1024], features)
    target_d = distances(Z, V)
    bandwidth = max(float(np.median(target_d[target_d > 1e-9])), 1e-6)
    result = []
    for exponent in [.25, .5, 1., 2.]:
        weights = np.exp(np.clip(-exponent * target_d / bandwidth, -30, 0))
        ids = sample_rows(pool_y, weights, offset=0)
        result.append((f"target_match_{exponent:g}", pool_X[ids], pool_y[ids], Q,
                       {"target_distance_exponent": exponent, "uses_larger_source_pool": True}))
    labels = np.r_[np.zeros(len(Z)), np.ones(len(V))]
    domain = LogisticRegression(C=.1, max_iter=300, random_state=0).fit(np.r_[Z, V], labels)
    prob = np.clip(domain.predict_proba(Z)[:, 1], .01, .99)
    ratio = prob / (1. - prob) * len(Z) / len(V)
    for exponent in [.25, .5, 1.]:
        ids = sample_rows(pool_y, np.minimum(ratio, 10.) ** exponent, offset=0)
        result.append((f"domain_ratio_{exponent:g}", pool_X[ids], pool_y[ids], Q,
                       {"ratio_exponent": exponent, "ratio_clip": 10., "uses_larger_source_pool": True}))
    for member in range(3):
        ids = sample_rows(pool_y, np.ones(len(pool_y)), offset=member + 1)
        result.append((f"source_resample_{member}", pool_X[ids], pool_y[ids], Q,
                       {"uses_larger_source_pool": True, "member_rng_offset": member + 1}))
    numeric = [i for i, name in enumerate(features) if name in CONTINUOUS]
    if numeric:
        src_m = np.median(X[:, numeric], 0)
        qry_m = np.median(Q[:, numeric], 0)
        src_s = np.maximum(np.percentile(X[:, numeric], 75, axis=0) - np.percentile(X[:, numeric], 25, axis=0), 1.)
        qry_s = np.maximum(np.percentile(Q[:, numeric], 75, axis=0) - np.percentile(Q[:, numeric], 25, axis=0), 1.)
        for strength in [.25, .5, 1.]:
            view = X.copy()
            full = (X[:, numeric] - src_m) / src_s * qry_s + qry_m
            view[:, numeric] = (1. - strength) * X[:, numeric] + strength * full
            result.append((f"continuous_transport_{strength:g}", view, y, Q,
                           {"transport_strength": strength, "continuous_columns": numeric}))
    # State-specific geographic codes cannot be transported as real numbers.
    for drop_names in [["PUMA", "POWPUMA", "ST"], ["OCCP", "POBP"]]:
        columns = np.array([i for i, name in enumerate(features) if name not in drop_names])
        if len(columns) == len(features) or not len(columns):
            continue
        result.append(("drop_" + "_".join(drop_names), X[:, columns], y, Q[:, columns],
                       {"removed_feature_names": drop_names}))
    return result


def evidence(base, bank):
    base, bank = normalize(base), normalize(bank)
    Q = 1. + (bank * np.log(bank)).sum(-1) / np.log(base.shape[1])
    S = 1. - .5 * np.abs(bank - bank.mean(0)[None]).sum(-1)
    U = -(base * np.log(base)).sum(-1) / np.log(base.shape[1])
    return Q.T, S.T, U


def fit_policy(cells, regularization):
    """Fit paper Eq. 3--7 using disjoint source-side calibration labels only."""
    count = cells[0]["bank"].shape[0]
    arrays = {key: [] for key in ["Q", "S", "U", "members", "base", "weights"]}
    for cell in cells:
        y, b, bank = cell["y"], cell["base"], cell["bank"]
        Q, S, U = evidence(b, bank)
        arrays["Q"].append(Q); arrays["S"].append(S); arrays["U"].append(U)
        arrays["members"].append(bank[:, np.arange(len(y)), y].T)
        arrays["base"].append(b[np.arange(len(y)), y])
        arrays["weights"].append(np.full(len(y), 1. / (len(cells) * len(y))))
    data = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float64) for k, v in arrays.items()}
    initial = np.r_[np.zeros(count), np.ones(count * 2), .5]
    def objective(parameters):
        p = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
        w = torch.softmax(p[:count] + data["Q"] * p[count:2*count] + data["S"] * p[2*count:3*count], 1)
        aggregate = (w * data["members"]).sum(1)
        support = (w * data["Q"] * data["S"]).sum(1)
        alpha = p[-1] * data["U"] * support
        prediction = (1-alpha) * data["base"] + alpha * aggregate
        loss = -(data["weights"] * prediction.clamp(1e-10, 1).log()).sum()
        loss = loss + regularization * ((p[:count] ** 2).mean() + ((p[count:3*count]-1.) ** 2).mean())
        loss.backward()
        return float(loss.detach()), p.grad.detach().numpy()
    answer = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(-6, 6)] * count + [(0, 8)] * (2*count) + [(0, 1)],
                      options={"maxiter": 200, "ftol": 1e-9})
    return answer.x, {"regularization": regularization, "success": bool(answer.success),
                      "iterations": int(answer.nit), "loss": float(answer.fun), "message": str(answer.message)}


def apply_policy(base, bank, parameters):
    Q, S, U = evidence(base, bank)
    count = len(bank)
    logits = parameters[:count] + Q * parameters[count:2*count] + S * parameters[2*count:3*count]
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    aggregate = np.einsum("nr,rnc->nc", weights, bank)
    alpha = parameters[-1] * U * (weights * Q * S).sum(1)
    return normalize((1-alpha[:, None]) * base + alpha[:, None] * aggregate), alpha
