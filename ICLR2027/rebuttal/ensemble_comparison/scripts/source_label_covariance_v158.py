"""Source-label-assisted feature posterior; explicit new context adaptation.

Fits a low-rank Gaussian measurement-error model using observed source X/y
and unlabeled target X. No target outcomes or clean source counterfactuals.
Its assumptions are approximate, so it is evaluated rather than presumed safe.
"""
import numpy as np


def psd(matrix, floor=1e-6):
    eigen, vectors = np.linalg.eigh((matrix+matrix.T)/2)
    return (vectors*np.maximum(eigen, floor))@vectors.T


def repair(X_source, y_source, X_target, numeric_columns, task):
    X, Q = np.asarray(X_source, dtype=np.float32), np.asarray(X_target, dtype=np.float64)
    y = np.asarray(y_source)
    numeric = np.asarray(numeric_columns, dtype=int)
    output = X.copy()
    if not len(numeric):
        return output, {'numeric_features': 0, 'modified_cells': 0,
            'source_task_labels_used': True, 'target_task_labels_used': False}
    T = Q[:, numeric]
    center = T.mean(axis=0)
    sd = T.std(axis=0)
    scale = np.percentile(T, 75, axis=0)-np.percentile(T, 25, axis=0)
    scale = np.where(scale > 1e-8, scale, np.maximum(sd, 1e-6))
    active = sd > 1e-8
    if (~active).any():
        output[:, numeric[~active]] = center[~active]
    rank, noise_trace = 0, 0.
    if active.any():
        Zq = ((T-center)/scale)[:, active]
        Zs = ((np.asarray(X[:, numeric], dtype=np.float64)-center)/scale)[:, active]
        low, high = Zq.min(axis=0)-.5, Zq.max(axis=0)+.5
        bounded = np.clip(Zs, low, high)
        _, singular, basis = np.linalg.svd(Zq, full_matrices=False)
        variance = singular**2/max(len(Q), 1)
        valid = variance > max(float(variance.max())*1e-4, 1e-8)
        rank = min(int(valid.sum()), 32)
        if rank:
            basis, variance = basis[:rank], variance[:rank]
            root = np.sqrt(variance)
            latent = (bounded@basis.T)/root
            if task == 'classification':
                labels = np.eye(len(np.unique(y)))[y.astype(int)]
                prior = labels.mean(axis=0)
                encoded = (labels-prior)/np.maximum(np.sqrt(prior*(1-prior)), .1)
            else:
                encoded = ((y-y.mean())/max(float(y.std()), 1e-12))[:, None]
            observed = latent-latent.mean(axis=0)
            joint = np.column_stack([observed, encoded])
            covariance = joint.T@joint/max(len(y), 1)
            noise = psd(covariance[:rank, :rank]-np.eye(rank), floor=0.)
            signal = covariance.copy()
            signal[:rank, :rank] = np.eye(rank)
            signal = psd(signal)
            total = signal.copy()
            total[:rank, :rank] += noise
            total += .05*np.eye(len(total))
            cross = signal[:rank]
            posterior = joint@np.linalg.solve(total, cross.T)
            raw_projection = (latent*root)@basis
            repaired = (posterior*root)@basis+(bounded-raw_projection)
            repaired = np.clip(.25*bounded+.75*repaired, low, high)
            output[:, numeric[active]] = repaired*scale[active]+center[active]
            noise_trace = float(np.trace(noise))
        else:
            output[:, numeric[active]] = bounded*scale[active]+center[active]
    categorical = sorted(set(range(X.shape[1]))-set(numeric))
    np.testing.assert_array_equal(output[:, categorical], X[:, categorical])
    assert np.isfinite(output).all()
    return output, {'numeric_features': len(numeric), 'active_numeric_features': int(active.sum()),
        'target_latent_rank': rank, 'latent_rank_cap': 32,
        'minimum_target_component_variance_fraction': 1e-4,
        'estimated_latent_noise_trace': noise_trace, 'posterior_ridge': .05,
        'posterior_blend': .75, 'marginal_support_margin_standardized': .5,
        'source_task_labels_used': True, 'source_label_values_unchanged': True,
        'target_features_used_transductively': True, 'target_task_labels_used': False,
        'clean_source_counterfactual_used': False, 'categorical_columns_unchanged': True,
        'modified_cells': int(np.sum(output != X))}
