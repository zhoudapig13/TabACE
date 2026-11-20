"""Query-cross-fitted pseudo-label contexts without true query labels.

The teacher is a fixed mean/median of raw TFM predictions, never a fusion head
trained on query labels. A query row is excluded from its own pseudo context.
This is an explicit transductive constructor extension, not original Shared.
"""
import numpy as np
from run_tabicl import normalize


def source_rows(y, task, count, rng):
    groups = [np.flatnonzero(y == label) for label in np.unique(y)] if task == 'classification' else [
        g for g in np.array_split(np.argsort(y, kind='stable'), min(8, len(y))) if len(g)]
    mandatory = np.array([rng.choice(g) for g in groups], dtype=int)
    assert count >= len(mandatory)
    remaining = np.setdiff1d(np.arange(len(y)), mandatory)
    chosen = np.r_[mandatory, rng.choice(remaining, size=count-len(mandatory), replace=False)]
    return np.sort(chosen.astype(int))


def contexts(X_source, y_source, X_query, predictions, task):
    """Return two branch plans and the outcome-free teacher control.

    Each plan has up to four query folds. Fold labels are source/pseudo labels
    only; no query outcome argument exists. Context row count stays <=source N.
    """
    X, y, Q, P = map(np.asarray, (X_source, y_source, X_query, predictions))
    assert task in ('classification', 'regression') and P.shape[1] == len(Q)
    assert len(X) == len(y) and len(y) <= 256 and len(Q) > 0
    if task == 'classification':
        teacher = normalize(P.mean(axis=0))
        pseudo = teacher.argmax(axis=1).astype(y.dtype)
        confidence = 1+(teacher*np.log(np.clip(teacher, 1e-12, 1.))).sum(axis=1)/np.log(max(2, teacher.shape[1]))
        minimum_source_rows = len(np.unique(y))
    else:
        teacher = np.clip(np.median(P, axis=0), float(y.min()), float(y.max()))
        pseudo = teacher.astype(y.dtype)
        confidence = -P.std(axis=0)/max(float(y.std()), 1e-12)
        minimum_source_rows = min(8, len(y))
    folds = np.array_split(np.random.default_rng(0).permutation(len(Q)), min(4, len(Q)))
    plans = []
    for variant, fraction in enumerate((.25, .5)):
        branch = []
        for fold, held in enumerate(folds):
            available = np.setdiff1d(np.arange(len(Q)), held)
            count = min(int(np.ceil(fraction*len(y))), len(available), len(y)-minimum_source_rows)
            target_ids = available[np.argsort(-confidence[available], kind='stable')[:count]]
            rng = np.random.default_rng(np.random.SeedSequence([0, variant, fold]))
            ids = source_rows(y, task, len(y)-count, rng)
            A, b = np.vstack([X[ids], Q[target_ids]]), np.r_[y[ids], pseudo[target_ids]]
            assert len(A) == len(b) == len(y)
            assert not set(target_ids) & set(held)
            if task == 'classification':
                assert set(y) <= set(b)
            branch.append({'X': A.astype(np.float32), 'y': b, 'query_rows': held,
                'source_rows': ids, 'pseudo_query_rows': target_ids,
                'pseudo_labels': pseudo[target_ids], 'no_pseudo_fallback': count == 0})
        assert sorted(np.concatenate([p['query_rows'] for p in branch])) == list(range(len(Q)))
        plans.append(branch)
    return plans, teacher
