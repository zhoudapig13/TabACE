"""Official TabICL Native-1 with one explicit preprocessing adaptation."""
from run_tabicl import Predictor, TabICLClassifier, TabICLRegressor, CHECKPOINTS


def create(task, device='cuda:0'):
    predictor = Predictor(task, device)
    cls = TabICLClassifier if task == 'classification' else TabICLRegressor
    name = 'classifier' if task == 'classification' else 'regressor'
    predictor.models[1] = cls(n_estimators=1, norm_methods='quantile',
        device=device, model_path=str(CHECKPOINTS/f'tabicl-{name}-v2-20260212.ckpt'),
        allow_auto_download=False, n_jobs=1, batch_size=1, random_state=0)
    return predictor
