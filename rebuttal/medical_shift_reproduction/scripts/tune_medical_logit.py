"""Additional logit-pooling adaptation; frozen Base and unchanged clinical labels."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
import tune_medical_development as main
import tune_medical_local as local
from tune_known_domain_medical import DomainWorker


def logit_probability(anchor,bank,policy):
    m=len(bank);p,q,s,u=exp.features(anchor,bank,m)
    q=.25+.75*q;u=policy['uncertainty_floor']+(1-policy['uncertainty_floor'])*u
    c=np.asarray(policy['parameters'])
    logits=c[:m]+q*c[m:2*m]+s*c[2*m:3*m]
    w=np.exp(logits-logits.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    alpha=policy['alpha_cap']*np.clip(3*c[-1]*u*(w*q*s).sum(1),0,1)
    return expit((1-alpha)*logit(np.clip(anchor,1e-6,1-1e-6))+alpha*(w*logit(np.clip(p,1e-6,1-1e-6))).sum(1))


def fit_logit(anchor,bank,y,weight,reg,cap,ufloor):
    m=len(bank);p,q,s,u=exp.features(anchor,bank,m)
    q=.25+.75*q;u=ufloor+(1-ufloor)*u
    arrays=[q,s,u,logit(np.clip(anchor,1e-6,1-1e-6)),logit(np.clip(p,1e-6,1-1e-6)),y,weight/weight.sum()]
    Q,S,U,B,L,Y,Wt=[torch.as_tensor(a,dtype=torch.float64) for a in arrays]
    init=np.r_[np.zeros(m),np.ones(2*m),.5]
    def objective(x):
        c=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        w=torch.softmax(c[:m]+Q*c[m:2*m]+S*c[2*m:3*m],dim=1)
        alpha=cap*(3*c[-1]*U*(w*Q*S).sum(1)).clamp(0,1)
        mixture=(1-alpha)*B+alpha*(w*L).sum(1)
        loss=(Wt*torch.nn.functional.binary_cross_entropy_with_logits(mixture,Y,reduction='none')).sum()
        loss=loss+reg*((c[:m]-c[:m].mean()).square().mean()+(c[m:3*m]-1).square().mean())
        loss.backward();return float(loss.detach()),c.grad.detach().numpy()
    result=minimize(objective,init,jac=True,method='L-BFGS-B',
                    bounds=[(-6,6)]*m+[(0,6)]*(2*m)+[(0,1)],options={'maxiter':100,'ftol':1e-9})
    return dict(kind='logit_pooling',parameters=result.x.tolist(),regularization=reg,alpha_cap=cap,
                uncertainty_floor=ufloor,optimizer_success=bool(result.success))


class LogitWorker(DomainWorker):
    def search_logit(self,anchor,bank,fallback,config,threshold,domain):
        best=fallback;conf=config.copy();value=self.domain_score(best,threshold,domain);trials=[]
        fit=self.calibration_fit
        for reg in [1.,.1]:
            for share in ([0.,.25] if domain=='id' else [.5,.75]):
                for cap in [.15,.25]:
                    for ufloor in [0.,.25]:
                        policy=fit_logit(anchor[fit],bank[:,fit],self.cal_y[fit],self.fit_weights(share),reg,cap,ufloor)
                        p=logit_probability(anchor,bank,policy);score=self.domain_score(p,threshold,domain)
                        trials.append(dict(regularization=reg,target_calibration_share=share,alpha_cap=cap,
                                           uncertainty_floor=ufloor,development_accuracy=score))
                        if score>value+1e-12:
                            best=p;value=score;conf=dict(policy=dict(policy,threshold=threshold),
                                target_calibration_share=share,adapter='logit-pooling development variant')
        return best,conf,value,trials

    def run_logit(self):
        parent=json.loads((main.ROOT/'results'/getattr(self,'parent_results_tag','known_domain_medical_v16')/self.task/'frozen_selection.json').read_text())
        with np.load(main.DATA/self.task/getattr(self,'parent_predictions_filename','known_domain_raw_predictions.npz')) as z:old={k:z[k] for k in z.files}
        configurations={};predictions={};scores={m:{} for m in ['Base','Shared','Specific']};trials=[]
        xc=self.X-self.X.mean(0);yc=self.y-self.y.mean()
        corr=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        ranked=np.argsort(-corr,kind='stable')
        numeric=json.loads((main.ROOT/self.meta['original_inputs']/'metadata.json').read_text())['numeric_features']
        phenotype=np.array([i for i,c in enumerate(numeric) if c.endswith('_last') or c in ['Age','Gender','HospAdmTime']],int) if self.task.startswith('sepsis') else np.sort(ranked[:64])
        self.cache_dir=main.DATA/self.task/'member_cache_known_domain_raw'
        for domain in ['id','ood']:
            self.reference_domain(domain)
            previous=parent['configurations'][domain]
            base=old[domain+'_Base'];base_conf=previous['Base'];threshold=base_conf['policy']['threshold']
            shared=old[domain+'_Shared'];sc=previous['Shared'].copy();shared_score=self.domain_score(shared,threshold,domain)
            shared_bank=self.bank(sc.get('context_fraction',base_conf['context_fraction']),sc.get('sampling_strength',.5))
            for fraction in [0.,1.]:
                for strength in [.5,1.]:
                    bank=self.bank(fraction,strength)
                    p,conf,value,choices=self.search_logit(base,bank,shared,sc,threshold,domain)
                    trials.append(dict(domain=domain,stage='Shared',fraction=fraction,strength=strength,choices=choices))
                    if value>shared_score+1e-12:
                        shared=p;sc=dict(conf,context_fraction=fraction,sampling_strength=strength);shared_score=value;shared_bank=bank
            if self.domain_score(old[domain+'_Specific'],threshold,domain)>=shared_score:
                specific=old[domain+'_Specific'];pc=previous['Specific'].copy()
            else:specific=shared;pc=sc.copy()
            specific_score=self.domain_score(specific,threshold,domain)
            for vi,columns in enumerate([np.sort(ranked[:16]),np.sort(ranked[:64]),phenotype]):
                for size in [32,64,128,256]:
                    nt=self.nt
                    try:
                        if domain=='id':self.nt=0
                        near,_=self.local(columns,size)
                    finally:self.nt=nt
                    a=self.predict(self.rows(base_conf['context_fraction'],np.ones(len(self.y)),201),columns)
                    b=self.predict(self.rows(sc.get('context_fraction',base_conf['context_fraction']),np.ones(len(self.y)),202),phenotype)
                    bank=np.r_[shared_bank,np.array([near,a,b])]
                    p,conf,value,choices=self.search_logit(shared,bank,specific,pc,threshold,domain)
                    trials.append(dict(domain=domain,stage='Specific',view=vi,size=size,choices=choices))
                    if value>specific_score+1e-12:
                        specific=p;pc=dict(conf,view=vi,columns=columns.tolist(),local_context_size=size,anchor='selected logit Shared')
                        specific_score=value
            configurations[domain]=dict(Base=base_conf,Shared=sc,Specific=pc)
            for m,p in [('Base',base),('Shared',shared),('Specific',specific)]:
                predictions[domain+'_'+m]=p;scores[m][domain]=self.domain_score(p,threshold,domain)
            print('LOGIT_DEVELOPMENT',self.task,domain,{m:s[domain] for m,s in scores.items()},flush=True)
        report=dict(parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_labels_opened=False,
                      configurations=configurations,development_accuracy=scores,logit_trials=trials,
                      protocol=json.loads(getattr(self,'logit_protocol_path',main.ROOT/'configs/known_domain_logit_v17.json').read_text()),
                      method_status='DAGGER known-domain logit fusion adaptation; not original probability interpolation',
                      probability_update_cap_not_applicable=True)
        exp.write_json(main.ROOT/'results'/getattr(self,'results_tag','known_domain_logit_v17')/self.task/'frozen_selection.json',report)
        np.savez_compressed(main.DATA/self.task/getattr(self,'predictions_filename','known_domain_logit_predictions.npz'),**predictions)
        print('LOGIT_FROZEN',self.task,scores,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if args.task=='readmission_other_to_ER':
        main.DATA=main.ROOT/'data/processed/target_development_readmission_cc_v5';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):LogitWorker(args.task,args.gpu,'raw').run_logit()
