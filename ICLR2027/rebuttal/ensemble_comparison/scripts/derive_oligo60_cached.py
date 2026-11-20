"""Reuse the first 60 actual cached candidates on unchanged regression inputs."""
import json
import numpy as np
from prepare_data import RUN,atomic_json
from run_tabicl import metrics


def main():
    for source in sorted((RUN/'results/oligo_extended/tabicl/five/regression').glob('*/*.npz')):
        if '.partial' in source.name or '.tmp' in source.name:continue
        out=RUN/'results/oligo_selected60/tabicl/five/regression'/source.parent.name/source.name
        if out.with_suffix('.json').exists():continue
        m=json.loads(source.with_suffix('.json').read_text())
        with np.load(source) as f:
            bank=f['bank'][:60];iqr=f['iqr'][:60];y=f['y_target'];seconds=f['candidate_seconds'][:60]
        assert len(bank)==60
        selected=np.argsort(iqr.mean(1),kind='stable')[:6]
        predictions={'OligoICP_IQR60_keep6_adapted':bank[selected].mean(0),'Bootstrap_mean60_extended_budget':bank.mean(0)}
        observed={name:metrics(y,p,'regression',m['scale'])|{'candidate_evaluations':60,'seconds':float(seconds.sum())} for name,p in predictions.items()}
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,y_target=y,**{'prediction__'+k:v for k,v in predictions.items()});tmp.replace(out)
        atomic_json(out.with_suffix('.json'),m|{'methods':observed,'candidate_pool_size':60,'retained_members':6,
            'selected_candidates':selected.tolist(),'candidate_predictions_reused_from':str(source.relative_to(RUN)),
            'budget_group':'60 candidate forwards; six retained; extra selection cost reported'})
    print('CACHED_OLIGO60_COMPLETE',flush=True)


if __name__=='__main__':main()
