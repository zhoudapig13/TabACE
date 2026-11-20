"""Ordinary-accuracy fine-tuning for the active heavy-tail classification task.

This is a labeled development adaptation, not frozen Shared. Training excludes
all held-out evaluation rows and the target-adaptation validation labels.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tabicl import FinetunedTabICLClassifier
from tabicl._finetune.base import ValidationMetrics
import shared_finetune_label_support_v303 as compatible


class AccuracyClassifier(compatible.parent.CaptureMixin,FinetunedTabICLClassifier):
    @property
    def _metric_name(self):return 'target_and_source_accuracy_mean'

    def _metric_improved(self,current,best):
        if current>best+self.min_delta:return True
        return abs(current-best)<1e-12 and self._last_nll_<self._best_nll_-1e-6

    def _save_checkpoint(self,**kwargs):
        super()._save_checkpoint(**kwargs)
        if kwargs['is_best'] and not hasattr(self,'_best_metric_'):self._best_nll_=self._last_nll_

    def _compute_batch_loss(self,batch,model):
        self.loss_calls_+=1;losses=[];limit=int(self.model_.max_classes)
        for i in range(len(batch.X)):
            yc,yq=batch.y_train[i].long(),batch.y_query[i].long()
            labels=torch.unique(yc)
            if len(labels)>limit:
                present=torch.unique(yq);present=present[torch.isin(present,labels)]
                first=present[(self.loss_calls_-1)%len(present)]
                rest=labels[labels!=first];order=torch.randperm(len(rest),device=rest.device)
                labels=torch.sort(torch.cat([first[None],rest[order[:limit-1]]])).values
            context=torch.isin(yc,labels);query=torch.isin(yq,labels);assert query.any()
            train=torch.searchsorted(labels,yc[context]);target=torch.searchsorted(labels,yq[query])
            X=torch.cat([batch.X[i,:batch.train_size][context],batch.X[i,batch.train_size:][query]])[None]
            logits=model(X,train.float()[None])[...,:len(labels)]
            # No inverse-class-frequency weighting: each query contributes once.
            losses.append(F.cross_entropy(logits.reshape(-1,len(labels)),target))
        return torch.stack(losses).mean()

    def _run_validation(self,inner,X_train,y_train,X_val,y_val):
        root=getattr(getattr(inner,'model_',None),'icl_predictor',None)
        if root is not None and hasattr(root,'root'):root.root=None
        inner.fit(self.context_X_,self.context_y_)
        p=inner.predict_proba(X_val);classes=np.asarray(inner.classes_)
        pred=classes[p.argmax(axis=1)];mapping={int(k):i for i,k in enumerate(classes)}
        labels=np.array([mapping[int(k)] for k in y_val])
        nll=-np.log(np.clip(p[np.arange(len(y_val)),labels],1e-12,1.))
        n=self.source_validation_count_;assert 0<n<len(y_val)
        source_acc=float(accuracy_score(y_val[:n],pred[:n]));target_acc=float(accuracy_score(y_val[n:],pred[n:]))
        self._last_nll_=.5*float(nll[:n].mean()+nll[n:].mean())
        if not self.validation_trace_:self._best_nll_=self._last_nll_
        primary=.5*(source_acc+target_acc)
        result=ValidationMetrics(primary=primary,secondary={'source_accuracy':source_acc,'target_accuracy':target_acc,
            'source_log_loss':float(nll[:n].mean()),'target_log_loss':float(nll[n:].mean()),'equal_domain_log_loss':self._last_nll_})
        self.validation_trace_.append({'primary':primary,'secondary':result.secondary,'context_rows':len(self.context_y_)})
        return result
