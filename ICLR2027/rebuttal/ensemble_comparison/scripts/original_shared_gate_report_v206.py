"""Report v204 with a metadata-only clarification of retained constructors.

The sealed runner inherited an experiment-log template flag. Correct that
flag during export; predictions, policies, metrics and statistics are untouched.
"""
import original_shared_gate_calibration_v204 as run


def main():
    write = run.atomic_json
    def clarified(path, value):
        if path.name == 'experiment_log.json':
            value = value | {'Analysis': value['Analysis'] | {
                'explicit_constructor_adaptation': False,
                'original_five_constructor_roles_and_fusion_formula_retained': True,
                'only_gate_parameters_and_internal_estimator_budget_recalibrated': True}}
        write(path, value)
    run.atomic_json = clarified
    run.report()
    write(run.OUT/'metadata_scope_clarification.json', {
        'reason': 'Inherited experiment-log template flag did not describe the retained original bank',
        'corrected_field': 'experiment_log.Analysis.explicit_constructor_adaptation=False',
        'predictions_and_policies_and_metrics_unchanged_by_this_wrapper': True,
        'sealed_fit_and_report_runner_sha256': run.original.sha(run.__file__),
        'metadata_export_wrapper_sha256': run.original.sha(__file__)})


if __name__ == '__main__':
    main()
