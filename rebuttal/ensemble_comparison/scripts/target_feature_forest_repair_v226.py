"""Unlabeled-query conditional feature repair, not original Shared.

Each pretext response is one INPUT FEATURE, never a task/query label. Its
entire encoding group is excluded from its predictors. Feature cross-validation
controls whether a conditional correction is permitted. Source labels are not
accepted by either API. Two strengths are fixed before downstream evaluation.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

MAX_RESPONSES = 24
MAX_PREDICTORS = 256
TREES = 48
MIN_RELIABILITY = .1


def encoded(A, state):
    parts = []
    if len(state['numeric']):
        parts.append(np.clip((A[:, state['numeric']]-state['center'])/state['scale'], -10., 10.))
    if len(state['categorical']):
        parts.append(state['encoder'].transform(A[:, state['categorical']]))
    out = np.concatenate(parts, axis=1).astype(np.float32)
    assert np.isfinite(out).all()
    return out


def estimator(n, categorical, seed):
    cls = ExtraTreesClassifier if categorical else ExtraTreesRegressor
    return cls(n_estimators=TREES, max_depth=8, min_samples_leaf=max(3, int(.02*n)),
        max_features=.7, random_state=seed, n_jobs=1)


def fit_target(Q, numeric_columns):
    Q = np.asarray(Q, dtype=np.float64)
    assert len(Q) >= 3 and Q.shape[1] >= 1 and np.isfinite(Q).all()
    numeric = np.asarray(numeric_columns, dtype=int)
    categorical = np.setdiff1d(np.arange(Q.shape[1]), numeric)
    center = np.median(Q[:, numeric], axis=0)
    scale = np.percentile(Q[:, numeric], 75, axis=0)-np.percentile(Q[:, numeric], 25, axis=0)
    scale = np.where(scale > 1e-8, scale, np.maximum(Q[:, numeric].std(0), 1e-6))
    state = {'numeric': numeric, 'categorical': categorical, 'center': center, 'scale': scale}
    groups = {int(j): [i] for i, j in enumerate(numeric)}
    offset = len(numeric)
    if len(categorical):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        encoder.fit(Q[:, categorical])
        state['encoder'] = encoder
        for j, values in zip(categorical, encoder.categories_):
            groups[int(j)] = list(range(offset, offset+len(values)))
            offset += len(values)
    state['groups'] = groups
    design = encoded(Q, state)
    candidates = [j for j in range(Q.shape[1]) if np.unique(Q[:, j]).size > 1 and
        (j in numeric or np.unique(Q[:, j]).size <= 32)]
    if len(candidates) > MAX_RESPONSES:
        candidates = sorted(np.random.default_rng(226).choice(candidates, MAX_RESPONSES, replace=False).tolist())
    splits = list(KFold(3, shuffle=True, random_state=0).split(Q))
    state['models'] = []
    diagnostics = []
    for j in candidates:
        own = set(groups[j])
        columns = np.asarray([k for k in range(design.shape[1]) if k not in own], dtype=int)
        if len(columns) > MAX_PREDICTORS:
            columns = np.sort(np.random.default_rng(226+j).choice(columns, MAX_PREDICTORS, replace=False))
        if not len(columns):
            diagnostics.append({'column': j, 'skip': 'no_other_feature'})
            continue
        assert not own.intersection(columns.tolist())
        is_cat = j in categorical
        A = design[:, columns]
        if is_cat:
            values, response = np.unique(Q[:, j], return_inverse=True)
            K = len(values)
            oof, prior = np.zeros((len(Q), K)), np.zeros((len(Q), K))
        else:
            position = int(np.flatnonzero(numeric == j)[0])
            response = (Q[:, j]-center[position])/scale[position]
            oof, prior = np.empty(len(Q)), np.empty(len(Q))
        for fold, (train, valid) in enumerate(splits):
            model = estimator(len(train), is_cat, fold)
            model.fit(A[train], response[train])
            if is_cat:
                oof[np.ix_(valid, model.classes_.astype(int))] = model.predict_proba(A[valid])
                prior[valid] = np.bincount(response[train], minlength=K)/len(train)
            else:
                oof[valid] = model.predict(A[valid])
                prior[valid] = response[train].mean()
        if is_cat:
            truth = np.eye(K)[response]
            error, baseline = np.mean(np.sum((oof-truth)**2, 1)), np.mean(np.sum((prior-truth)**2, 1))
        else:
            error, baseline = np.mean((oof-response)**2), np.mean((prior-response)**2)
        reliability = float(np.clip(1-error/max(baseline, 1e-12), 0., 1.))
        info = {'column': j, 'categorical': bool(is_cat), 'input_encoded_columns': columns.tolist(),
            'own_encoding_columns': groups[j], 'CV_feature_prediction_loss': float(error),
            'CV_feature_prior_loss': float(baseline), 'reliability': reliability,
            'used': bool(reliability >= MIN_RELIABILITY)}
        if reliability >= MIN_RELIABILITY:
            model = estimator(len(Q), is_cat, 0)
            model.fit(A, response)
            record = {'column': j, 'columns': columns, 'model': model, 'categorical': is_cat,
                'reliability': reliability}
            if is_cat:
                record['values'] = values
            else:
                low, high = np.quantile(response-oof, [.025, .975])
                margin = .2*max(high-low, .1)
                record.update(position=position, low=float(low-margin), high=float(high+margin))
                info.update(residual_low=record['low'], residual_high=record['high'])
            state['models'].append(record)
        diagnostics.append(info)
    low, high = np.quantile(Q[:, numeric], [.005, .995], axis=0)
    state.update(low=low-.25*scale, high=high+.25*scale)
    state['info'] = {'responses_considered': diagnostics,
        'feature_CV_splits': [{'train': a.tolist(), 'valid': b.tolist()} for a, b in splits],
        'max_feature_responses': MAX_RESPONSES, 'max_encoded_predictors': MAX_PREDICTORS,
        'trees': TREES, 'max_depth': 8, 'max_features': .7,
        'min_leaf': 'max(3,floor(.02*training_rows))', 'minimum_reliability': MIN_RELIABILITY,
        'task_labels_used': False, 'own_feature_excluded_from_predictors': True,
        'numeric_residual_quantiles': [.025, .975], 'numeric_residual_margin': .2,
        'numeric_marginal_quantiles': [.005, .995], 'numeric_marginal_margin_IQR': .25,
        'categorical_confidence_mild_full': [.9, .8], 'categorical_observed_probability_max': .1,
        'query_input_features_used_transductively': True}
    return state


def repair_source(X_source, state):
    X = np.asarray(X_source, dtype=np.float64)
    bounded = X.copy()
    bounded[:, state['numeric']] = np.clip(bounded[:, state['numeric']], state['low'], state['high'])
    A = encoded(bounded, state)
    result = []
    for gain in (.5, 1.):
        proposal = bounded.copy()
        cat_edits = 0
        for r in state['models']:
            j = r['column']
            if r['categorical']:
                probability = r['model'].predict_proba(A[:, r['columns']])
                choices = r['model'].classes_.astype(int)
                best = probability.argmax(1)
                observed = np.zeros(len(X))
                for k, code in enumerate(choices):
                    mask = X[:, j] == r['values'][code]
                    observed[mask] = probability[mask, k]
                mask = (probability.max(1) >= (.9 if gain == .5 else .8)) & (observed < .1)
                proposal[mask, j] = r['values'][choices[best[mask]]]
                cat_edits += int(mask.sum())
            else:
                p = r['position']
                expected = r['model'].predict(A[:, r['columns']])
                raw = (X[:, j]-state['center'][p])/state['scale'][p]
                corrected = expected+np.clip(raw-expected, r['low'], r['high'])
                corrected = np.clip(corrected*state['scale'][p]+state['center'][p], state['low'][p], state['high'][p])
                proposal[:, j] = (1-r['reliability'])*bounded[:, j]+r['reliability']*corrected
        output = X.copy()
        output[:, state['numeric']] = (1-gain)*X[:, state['numeric']]+gain*proposal[:, state['numeric']]
        # Categorical values are replaced by valid observed codes, never interpolated.
        output[:, state['categorical']] = proposal[:, state['categorical']]
        output = output.astype(np.float32)
        assert np.isfinite(output).all()
        result.append({'X': output, 'info': {'rule': 'target_feature_forest_repair', 'numeric_gain': gain,
            'modified_cells': int(np.sum(output != np.asarray(X_source, dtype=np.float32))),
            'categorical_edits': cat_edits, 'task_labels_used': False,
            'query_features_modified': False}})
    return result
