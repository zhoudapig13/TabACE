"""Six unregistered mechanisms; no target labels enter any operator.

These are new empirical generators within broad statistical shift classes,
not claims of new mutually exclusive shift categories. Constants are fixed
before evaluating the frozen Shared policies.
"""
import numpy as np

FAMILIES = {
    'cell_heavy_tail': {'group': 'perturbation', 'numeric_minimum': 1,
        'description': 'source numeric cells receive sparse truncated Student-t(2) measurement errors'},
    'column_desynchronization': {'group': 'perturbation', 'numeric_minimum': 0,
        'description': 'independent within-subset column permutations preserve every column marginal but break row correspondence'},
    'feature_quantization': {'group': 'perturbation', 'numeric_minimum': 1,
        'description': 'source numeric measurements are rounded to a severity-scaled source-IQR grid'},
    'nonlinear_sensor': {'group': 'shift', 'numeric_minimum': 1,
        'description': 'non-affine monotone signed-log observation map'},
    'cross_feature_rotation': {'group': 'shift', 'numeric_minimum': 2,
        'description': 'pairwise Givens rotations mix distinct standardized numeric coordinates'},
    'interaction_sampling': {'group': 'shift', 'numeric_minimum': 0,
        'description': 'complete source rows are sampled with X-only pair-interaction weights'},
}
STRENGTHS = (20, 50, 80, 100)
PARAMETERS = {
    'heavy_tail_cell_probability_at_max': .1,
    'heavy_tail_df': 2., 'heavy_tail_truncation': 20., 'heavy_tail_scale': 2.,
    'desynchronization_feature_fraction': .5, 'desynchronization_row_fraction_at_max': .5,
    'quantization_step_iqr_at_max': 1.,
    'rotation_angle_at_max': float(np.pi / 4),
    'interaction_pairs_cap': 8, 'interaction_sampling_log_weight': 3.,
    'numeric_scale': 'IQR, falling back to source std when IQR is zero',
    'observation_shift_side': 'target X for classification; source X for regression',
}


def coordinates(X, numeric_columns):
    numeric_columns = np.asarray(numeric_columns, dtype=int)
    sd = X.std(axis=0, dtype=np.float64)
    return numeric_columns[sd[numeric_columns] > 1e-8]


def applicable(X, numeric_columns, family):
    count = len(coordinates(X, numeric_columns))
    if count < FAMILIES[family]['numeric_minimum']:
        return False, 'insufficient_nonconstant_numeric_features'
    if family == 'interaction_sampling' and np.sum(X.std(axis=0) > 1e-8) < 2:
        return False, 'fewer_than_two_nonconstant_coordinates'
    if family == 'column_desynchronization' and not np.any(X.std(axis=0) > 1e-8):
        return False, 'no_nonconstant_coordinates'
    return True, None


def source_statistics(X):
    center = np.median(X, axis=0).astype(np.float64)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    sd = X.std(axis=0, dtype=np.float64)
    scale = np.where(iqr > 1e-8, iqr, np.where(sd > 1e-8, sd, 1.))
    return center, scale


def generate(X_source, y_source, X_target, numeric_columns, task, family, rho, task_id=0):
    """Return source X/y, target X, and provenance; no target-y argument exists."""
    assert task in ('classification', 'regression')
    assert family in FAMILIES and rho in (0,) + STRENGTHS
    X = np.asarray(X_source, dtype=np.float32).copy()
    y = np.asarray(y_source).copy()
    Q = np.asarray(X_target, dtype=np.float32).copy()
    info = {'family': family, 'rho': int(rho), 'generator_receives_target_labels': False,
        'source_label_values_modified': False, 'source_rows_resampled': False,
        'categorical_values_numerically_interpolated': False}
    if rho == 0:
        return X, y, Q, info
    ok, why = applicable(X, numeric_columns, family)
    if not ok:
        raise ValueError((family, why))
    index = list(FAMILIES).index(family)
    rng = np.random.default_rng(np.random.SeedSequence([0, int(task_id), 117, index]))
    s = rho / 100.
    numeric = coordinates(X, numeric_columns)
    center, scale = source_statistics(X)
    if family == 'cell_heavy_tail':
        u = rng.random((len(X), len(numeric)))
        noise = np.clip(rng.standard_t(2., size=u.shape), -20., 20.)
        mask = u < .1*s
        X[:, numeric] += (mask * noise * (2.*scale[numeric])).astype(np.float32)
        info['affected_cells'] = int(mask.sum())
    elif family == 'column_desynchronization':
        available = np.flatnonzero(X.std(axis=0) > 1e-8)
        columns = rng.permutation(available)[:max(1, int(np.ceil(.5*len(available))))]
        ids = rng.permutation(len(X))[:int(np.ceil(.5*s*len(X)))]
        for j in columns:
            permutation = np.random.default_rng(np.random.SeedSequence([0, int(task_id), 117, 1, int(j)])).permutation(len(ids))
            X[ids, j] = X[ids[permutation], j].copy()
        info.update(affected_columns=columns.tolist(), affected_row_positions=ids.tolist(),
            column_marginals_preserved=True)
    elif family == 'feature_quantization':
        step = s*scale[numeric]
        X[:, numeric] = (center[numeric] + np.rint((X[:, numeric]-center[numeric])/step)*step).astype(np.float32)
    elif family in ('nonlinear_sensor', 'cross_feature_rotation'):
        observed = Q if task == 'classification' else X
        if family == 'nonlinear_sensor':
            z = (observed[:, numeric]-center[numeric])/scale[numeric]
            warped = np.sign(z)*np.log1p(s*np.abs(z))/s
            observed[:, numeric] = (center[numeric]+scale[numeric]*warped).astype(np.float32)
        else:
            order = rng.permutation(numeric)
            pairs = order[:2*(len(order)//2)].reshape(-1, 2)
            a, b = pairs[:, 0], pairs[:, 1]
            za = (observed[:, a]-center[a])/scale[a]
            zb = (observed[:, b]-center[b])/scale[b]
            angle = s*np.pi/4
            observed[:, a] = (center[a]+scale[a]*(np.cos(angle)*za-np.sin(angle)*zb)).astype(np.float32)
            observed[:, b] = (center[b]+scale[b]*(np.sin(angle)*za+np.cos(angle)*zb)).astype(np.float32)
            info.update(rotation_pairs=pairs.tolist(), angle_radians=float(angle))
        info['changed_observation_block'] = 'target' if task == 'classification' else 'source'
    else:
        available = np.flatnonzero(X.std(axis=0) > 1e-8)
        order = rng.permutation(available)
        pairs = order[:min(16, 2*(len(order)//2))].reshape(-1, 2)
        z = np.tanh((X-center)/scale)
        score = (z[:, pairs[:, 0]]*z[:, pairs[:, 1]]).mean(axis=1)
        logits = 3.*s*score
        weights = np.exp(logits-logits.max())
        weights /= weights.sum()
        # Same uniforms across severities; finite empirical resampling uses no
        # label-conditioned anchors, oversampling, or target-row statistics.
        u = rng.random(len(X))
        ids = np.minimum(np.searchsorted(np.cumsum(weights), u), len(X)-1)
        X, y = X[ids].copy(), y[ids].copy()
        info.update(source_rows_resampled=True, selected_source_positions=ids.tolist(),
            interaction_pairs=pairs.tolist(), sampling_uses_labels=False)
    assert np.isfinite(X).all() and np.isfinite(Q).all()
    return X, y, Q, info
