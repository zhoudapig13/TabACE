"""Develop an extra Specific fusion on frozen Native-8 controls.

The added, previously fitted experts use Native-1 or a supervised classifier;
this is an explicitly mixed-compute medical adaptation, not the equal Native-8
control and not the original TabACE implementation. Selection uses development
labels only; final test truth is never loaded here.
"""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from scipy.special import expit, logit
from run_public_shift import ROOT, write_json

TASKS = ['sepsis_A_to_B', 'sepsis_B_to_A', 'readmission_other_to_ER']
ALPHAS = [.10, .20, .30]


def main():
    protocol = dict(seed=0, selection='Development accuracy only; strict improvement, Shared fallback',
        alphas=ALPHAS, parent='medical_common_native8_v24',
        fixed_base_shared=True, target_fitting_labels=64,
        target_labels_split=dict(context=32, calibration=32),
        additional_development_validation_labels=True,
        added_experts='Previously frozen v22 TFM context (Native-1) and supervised auxiliary',
        mixed_compute_explicit=True, original_tabace_reproduction=False,
        previous_test_cohorts_observed=True)
    write_json(ROOT/'configs/medical_specific_native8_pool_v25.json', protocol)
    for task in TASKS:
        data = ROOT/'data/processed/expanded_medical_development_v18'/task
        parent_path = ROOT/'results/medical_common_native8_v24'/task/'frozen_selection.json'
        parent = json.loads(parent_path.read_text())
        expert_path = ROOT/'results/medical_specific_simple_v22'/task/'frozen_selection.json'
        expert_config = json.loads(expert_path.read_text())
        with np.load(data/'inputs.npz') as z:
            truth_dev = {d: z['y_'+d+'_dev'] for d in ['id', 'ood']}
        with np.load(data/'medical_common_native8_predictions.npz') as z:
            controls = {k: z[k] for k in z.files}
        with np.load(data/'specific_simple_predictions.npz') as z:
            experts = {k: z[k] for k in z.files}
        output = {}; configs = {}; trials = []
        scores = {m: {} for m in ['Base', 'Shared', 'Specific_N8P']}
        for domain in ['id', 'ood']:
            threshold = parent['configurations'][domain]['Base']['policy']['threshold']
            sl = slice(*parent['query_layout'][domain+'_development'])
            def score(p):
                return float(np.mean((p[sl] >= threshold) == truth_dev[domain]))
            shared = controls[domain+'_Shared']
            best = shared.copy(); best_score = score(best)
            config = dict(kind='Shared fallback', threshold=threshold)
            centered = logit(np.clip(shared, 1e-6, 1-1e-6))-logit(threshold)
            candidates = [
                ('Native8_Specific', controls[domain+'_Specific'], threshold),
                ('TFM_context_Native1', experts[domain+'_TFM_expert'],
                 expert_config['pooling_expert_configurations'][domain]['TFM_context']['threshold']),
                ('Supervised_auxiliary', experts[domain+'_Auxiliary'],
                 expert_config['pooling_expert_configurations'][domain]['Supervised_auxiliary']['threshold'])]
            for name, expert, tau in candidates:
                e = logit(np.clip(expert, 1e-6, 1-1e-6))-logit(tau)
                for alpha in ALPHAS:
                    p = expit((1-alpha)*centered+alpha*e+logit(threshold))
                    value = score(p)
                    trials.append(dict(domain=domain, expert=name, alpha=alpha, development_accuracy=value))
                    if value > best_score+1e-12:
                        best=p; best_score=value
                        config=dict(expert=name, alpha=alpha, threshold=threshold, expert_threshold=tau)
            configs[domain] = config
            for method, p in [('Base', controls[domain+'_Base']), ('Shared', shared), ('Specific_N8P', best)]:
                output[domain+'_'+method] = p
                scores[method][domain] = score(p)
        report = dict(parent)
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            protocol=protocol, development_accuracy=scores, base_shared_unchanged=True,
            test_labels_opened=False, added_specific_configurations=configs, trials=trials,
            native_estimator_count_matched_across_all_methods=False,
            native8_controls_with_additional_native1_or_supervised_expert=True,
            source_snapshots_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [parent_path, expert_path]},
            method_status='Specific-N8P DAGGER mixed-expert medical development adaptation')
        write_json(ROOT/'results/medical_specific_native8_pool_v25'/task/'frozen_selection.json', report)
        np.savez_compressed(data/'specific_native8_pool_predictions.npz', **output)
        print(task, scores, configs, flush=True)


if __name__ == '__main__':
    main()
