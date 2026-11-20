"""Use larger held-out development cohorts; preserve the64 target fitting patients."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import tune_medical_development as main
import tune_medical_local as local
from tune_known_domain_medical import DomainWorker
from tune_medical_logit import LogitWorker


class ExpandedMixin:
    def __init__(self,task,gpu,representation):
        super().__init__(task,gpu,representation)
        pack=self.pack
        unused=np.setdiff1d(np.arange(len(pack['y_dev_fit'])),
                            np.r_[self.target_context_indices,self.target_calibration_indices])
        assert np.array_equal(pack['X_ood_dev'],np.r_[pack['X_dev_val'],pack['X_dev_fit'][unused]])
        assert np.array_equal(pack['y_ood_dev'],np.r_[pack['y_dev_val'],pack['y_dev_fit'][unused]])
        self.val_y=pack['y_ood_dev'];self.id_validation_y=pack['y_id_dev']
        self.queries=np.r_[pack['X_cal'],pack['X_dev_fit'][self.target_calibration_indices],
                           pack['X_id_dev'],pack['X_ood_dev'],pack['X_id'],pack['X_ood']]
        sizes=[len(self.cal_y),len(self.id_validation_y),len(self.val_y),len(pack['X_id']),len(pack['X_ood'])]
        self.query_slices={k:slice(a,b) for k,a,b in zip(
            ['cal','id_development','ood_development','id_test','ood_test'],
            np.cumsum([0]+sizes[:-1]),np.cumsum(sizes))}
        self.slices=dict(cal=self.query_slices['cal'],validation=self.query_slices['ood_development'],
                         id=self.query_slices['id_test'],ood=self.query_slices['ood_test'])
        self.id_validation=np.arange(self.query_slices['id_development'].start,self.query_slices['id_development'].stop)
        self.source_calibration_fit=np.arange(self.n_source_cal)
        self.calibration_fit=np.arange(len(self.cal_y))
        assert not set(self.calibration_fit).intersection(self.id_validation)

    def development(self,p,threshold):
        return dict(id=float(np.mean((p[self.id_validation]>=threshold)==self.id_validation_y)),
                    ood=float(np.mean((p[self.slices['validation']]>=threshold)==self.val_y)))

    def supplement_snapshot(self):
        path=main.ROOT/'results'/self.results_tag/self.task/'frozen_selection.json'
        d=json.loads(path.read_text())
        d.update(external_id_development=True,query_layout={k:[int(s.start),int(s.stop)] for k,s in self.query_slices.items()},
                 development_truth_keys=dict(id='y_id_dev',ood='y_ood_dev'),
                 source_calibration_full_pool_restored=True,target_fitting_budget_unchanged=64,
                 source_development_patients_disjoint_from_source_training_calibration_and_tests=True,
                 target_extra_validation_excludes_all64_fitting_patients=True)
        path.write_text(json.dumps(d,indent=2))


class ExpandedProbability(ExpandedMixin,DomainWorker):pass
class ExpandedLogit(ExpandedMixin,LogitWorker):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--stage',choices=['probability','logit'],required=True);args=parser.parse_args()
    main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):
        if args.stage=='probability':
            worker=ExpandedProbability(args.task,args.gpu,'raw')
            worker.results_tag='expanded_medical_probability_v18'
            worker.predictions_filename='expanded_probability_predictions.npz'
            worker.protocol_path=main.ROOT/'configs/expanded_medical_development_v18.json'
            worker.run_domain()
        else:
            worker=ExpandedLogit(args.task,args.gpu,'raw')
            worker.parent_results_tag='expanded_medical_probability_v18'
            worker.parent_predictions_filename='expanded_probability_predictions.npz'
            worker.results_tag='expanded_medical_logit_v19'
            worker.predictions_filename='expanded_logit_predictions.npz'
            worker.logit_protocol_path=main.ROOT/'configs/expanded_medical_logit_v19.json'
            worker.run_logit()
        worker.supplement_snapshot()
