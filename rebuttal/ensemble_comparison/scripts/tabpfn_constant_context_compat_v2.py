"""Input-only prior fallback when a PFN sub-context loses all features.

This matches the Base wrapper's constant-context semantics. It is activated
only by the explicit all-constant preprocessing error, never by a test score.
The installed PFN and archived official extension sources are left unchanged.
"""
from functools import wraps
import json
import os

import numpy as np

from prepare_data import protocol
from tabpfn_official_benchmark_v2 import OUT

SCOPE = None


def event(estimator, X, y):
    folder = OUT / 'constant_context_compat_events'; folder.mkdir(exist_ok=True)
    record = {'reason': 'all features removed by sub-context preprocessing',
        'selection_uses_query_labels_or_scores': False, 'source_X_sha256': protocol.array_sha256(np.asarray(X)),
        'source_y_sha256': protocol.array_sha256(np.asarray(y)), 'source_rows': len(y),
        'source_features': np.asarray(X).shape[1], 'model_type': type(estimator).__name__, 'outer_scope': SCOPE}
    with (folder / f'{os.getpid()}.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')


def install():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.errors import TabPFNValidationError
    for cls in [TabPFNClassifier, TabPFNRegressor]:
        if getattr(cls, '_constant_context_compat_v2', False):
            continue
        original_fit, original_predict = cls.fit, cls.predict
        def fit_wrapper(function):
            @wraps(function)
            def fit(self, X, y, *args, **kwargs):
                self._constant_context_compat_y = None
                try:
                    return function(self, X, y, *args, **kwargs)
                except TabPFNValidationError as exc:
                    if 'All features are constant' not in str(exc):
                        raise
                    self._constant_context_compat_y = np.asarray(y).copy()
                    if isinstance(self, TabPFNClassifier):
                        self.classes_ = np.unique(y)
                    event(self, X, y)
                    return self
            return fit
        def predict_wrapper(function):
            @wraps(function)
            def predict(self, X, *args, **kwargs):
                y = getattr(self, '_constant_context_compat_y', None)
                if y is None:
                    return function(self, X, *args, **kwargs)
                if isinstance(self, TabPFNClassifier):
                    labels, counts = np.unique(y, return_counts=True)
                    return np.full(len(X), labels[int(np.argmax(counts))])
                output = kwargs.get('output_type', 'mean')
                mean = np.full(len(X), float(np.mean(y)))
                median = np.full(len(X), float(np.median(y)))
                quantiles = [np.full(len(X), float(np.quantile(y, q))) for q in kwargs.get('quantiles', [.1, .5, .9])]
                if output == 'mean':
                    return mean
                if output == 'median':
                    return median
                if output == 'quantiles':
                    return quantiles
                if output == 'main':
                    return {'mean': mean, 'median': median, 'mode': mean, 'quantiles': quantiles}
                raise NotImplementedError(f'Constant-context analytical fallback for output_type={output}')
            return predict
        cls.fit = fit_wrapper(original_fit)
        cls.predict = predict_wrapper(original_predict)
        if cls is TabPFNClassifier:
            original_proba = cls.predict_proba
            def proba_wrapper(function):
                @wraps(function)
                def predict_proba(self, X, *args, **kwargs):
                    y = getattr(self, '_constant_context_compat_y', None)
                    if y is None:
                        return function(self, X, *args, **kwargs)
                    labels, counts = np.unique(y, return_counts=True)
                    return np.broadcast_to(counts / counts.sum(), (len(X), len(labels))).copy()
                return predict_proba
            cls.predict_proba = proba_wrapper(original_proba)
        cls._constant_context_compat_v2 = True


def scope_prediction(function):
    @wraps(function)
    def wrapped(task, method, X, y, Q, classes):
        global SCOPE
        SCOPE = {'task': task, 'method': method, 'outer_source_X_sha256': protocol.array_sha256(X),
                 'outer_source_y_sha256': protocol.array_sha256(y)}
        try:
            return function(task, method, X, y, Q, classes)
        finally:
            SCOPE = None
    return wrapped
