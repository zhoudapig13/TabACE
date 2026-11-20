"""Fixed 1.96-SE sensitivity control on the existing development predictions.

This is a conservative development variant, informed by prior development
outcomes. Residual coefficients are computed from source validation only, but
the overall experiment must NOT be represented as an independent test or as
recovery of the paper's original implementation.
"""
import json
from pathlib import Path

import numpy as np

from prepare_data import atomic_json
from run_tabicl import metrics
import uncertainty_regularization_v82 as prior

OUT = prior.RUN / "results/tabace_conservative_development_v83"
MULTIPLIER = 1.96


def run():
    assert not (OUT / "plan.json").exists(), "Preserve the declared variant; do not overwrite it."
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "plan.json", dict(status="FROZEN_BEFORE_VARIANT_QUERY_EVALUATION", seed=0,
        context_cap=256, SE_multiplier=MULTIPLIER,
        criterion="smallest residual on fixed grid within best source MSE + 1.96*SE(best)",
        source_trials_frozen_in=str(prior.OUT), original_one_SE_control_preserved=True,
        query_labels_used_to_compute_residual_coefficients=False,
        prior_development_query_outcomes_informed_method_design=True,
        reference_alignment_motivated_development=True,
        original_paper_reproduction=False, independent_test_claim=False,
        paper_confirmation_evidence=False, extra_clean_TRAIN_supervision=True,
        Base_and_benchmark_unchanged=True, no_metric_editing=True,
        script_sha256=prior.parent.old.sha(__file__), Markdown_written=False))
    for task, block in [("classification","clean"),("classification","five"),("classification","seven"),("regression","clean")]:
        dest=OUT/"predictions/256"/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.symlink_to(prior.parent.OUT/"predictions/256"/task/block,target_is_directory=True)
    all_policies={}
    for block in ["five","seven"]:
        path=prior.OUT/"policies/regression"/block/"frozen_policy.json"
        old=json.loads(path.read_text());policies={}
        for key,p in old["policies"].items():
            threshold=p["best_source_MSE"]+MULTIPLIER*p["best_SE"]
            chosen=next(t for t in p["trials"] if t["source_MSE"]<=threshold+1e-12)
            policies[key]=p|dict(chosen_blend=chosen["blend"],threshold=threshold,SE_multiplier=MULTIPLIER)
        freeze=OUT/"policies/regression"/block/"frozen_policy.json"
        atomic_json(freeze,dict(status="FROZEN_BEFORE_QUERY",policies=policies,bank_capacity=old["bank_capacity"],
            source_policy=str(path),source_policy_sha256=prior.parent.old.sha(path),
            query_labels_used_for_coefficient_computation=False,independent_test_claim=False))
        all_policies[block]=policies
    atomic_json(OUT/"all_policies_frozen.json",dict(status="FROZEN_BEFORE_QUERY",policies=all_policies))
    for block in ["five","seven"]:
        policies=all_policies[block]
        capacity=json.loads((prior.OUT/"policies/regression"/block/"frozen_policy.json").read_text())["bank_capacity"]
        freeze=OUT/"policies/regression"/block/"frozen_policy.json"
        for path in sorted((prior.parent.OUT/"predictions/256/regression"/block).glob("*/*.npz")):
            info=json.loads(path.with_suffix(".json").read_text());tid=path.parent.name;family=info["family"]
            linear_path=prior.parent.old.OUT/capacity/"predictions/256/regression"/block/tid/path.name
            with np.load(path) as z:base,shared,truth=z["Base"],z["Shared"],z["y_target"];nonlinear=z["Specific"]
            with np.load(linear_path) as z:
                assert np.array_equal(z["Base"],base) and np.array_equal(z["y_target"],truth)
                linear=z["Specific"]
            anchor_path=linear_path;extension_path=path
            if block=="five":anchor=linear;extension=nonlinear;policy=policies["global"]
            else:
                meta_path=prior.parent.old.PARENT/"predictions/256/regression/seven"/tid/path.with_suffix(".json").name
                old_info=json.loads(meta_path.read_text());anchor_path=Path(old_info["candidate_path"])
                with np.load(anchor_path) as z:
                    assert np.array_equal(z["Base"],base) and np.array_equal(z["y_target"],truth)
                    anchor=z["Specific"]
                extension=linear;extension_path=linear_path;policy=policies[family]
            a=policy["chosen_blend"];specific=(1-a)*anchor+a*extension
            out=OUT/"predictions/256/regression"/block/tid/path.name
            prior.parent.old.cls.source.adaptation.raw.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
            info.update(adaptation_version="v83",SE_multiplier=MULTIPLIER,selected_blend=a,
                policy_path=str(freeze),policy_sha256=prior.parent.old.sha(freeze),
                anchor_prediction=str(anchor_path),anchor_prediction_sha256=prior.parent.old.sha(anchor_path),
                extension_prediction=str(extension_path),extension_prediction_sha256=prior.parent.old.sha(extension_path),
                prior_development_query_outcomes_informed_method_design=True,
                original_paper_reproduction=False,independent_test_claim=False,
                methods={n:metrics(truth,p,"regression",info["scale"]) for n,p in [("Base",base),("Shared",shared),("Specific",specific)]})
            atomic_json(out.with_suffix(".json"),info)
        print("CONSERVATIVE_QUERY_COMPLETE",block,{k:p["chosen_blend"] for k,p in policies.items()},flush=True)
    previous=prior.OUT
    try:prior.OUT=OUT;prior.report()
    finally:prior.OUT=previous
    atomic_json(OUT/"execution.json",dict(status="COMPLETE_PENDING_AUDIT",seed=0,SE_multiplier=MULTIPLIER))


if __name__=="__main__":run()
