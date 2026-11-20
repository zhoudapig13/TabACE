"""Explain which context bank each comparison used and verify both references."""
from pathlib import Path
import csv
import datetime
import json

RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parents[1]
RESULTS=RUN/'results'
BLOCKS=[('five_corrected','classification',1512,1800,5),
        ('five','regression',735,875,5),
        ('seven_reimplemented','classification',1015,1225,7),
        ('seven_reimplemented','regression',957,1155,7)]


def main():
    audit=json.loads((RESULTS/'raw_result_audit.json').read_text())
    assert audit['status']=='PASS'
    coverage=[]
    for backbone in ['tabicl','tabpfn']:
        for track,task,shared_count,specific_count,families in BLOCKS:
            for variant,folder,expected in [('shared','reference',shared_count),('specific','reference_specific',specific_count)]:
                checked=next(c for c in audit['coverage'] if c['root']==folder+'/'+backbone and c['track']==track and c['task']==task)
                assert checked['complete'] and checked['completed']==expected
                paths=list((RESULTS/folder/backbone/track/task).glob('*/*.json'))
                assert len(paths)==expected
                sample=json.loads(paths[0].read_text())
                if 'constructors_receive_family' in sample:
                    assert sample['constructors_receive_family']==(variant=='specific')
                    assert sample['constructors_receive_strength'] is False
                else:
                    assert variant=='shared' and sample['constructors_receive_family_or_strength'] is False
                pattern=(f'fold_*_tabace_{backbone}_reimplemented.json' if variant=='shared'
                         else f'fold_*_tabace_specific_{backbone}_*_reimplemented.json')
                fits=list((RESULTS/'combiners'/track/task).glob(pattern))
                assert len(fits)==(5 if variant=='shared' else families*5)
                for path in fits:
                    fitted=json.loads(path.read_text())
                    assert fitted['status']=='COMPLETE' and fitted['optimization']['optimizer_success']
                coverage.append({'backbone':backbone,'variant':variant,'track':track,'task':task,
                    'raw_conditions':len(paths),'oof_policies':len(fits),
                    'receives_family':variant=='specific','receives_strength':False,'complete':True})
    with (RESULTS/'main_table_current_protocol.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['method'] in
              ['TabACE_Shared_R5_reimplemented','TabACE_Specific_R5_reimplemented']]
    with (RESULTS/'shared_specific_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    provenance=[
        {'method':'Native ensemble','contexts':'canonical source context, native transformations','tabace_bank_used':False,'family_visible':False},
        {'method':'Random / GES / Stacking','contexts':'generic bootstrap contexts; GES/Stacking reuse their prediction bank','tabace_bank_used':False,'family_visible':False},
        {'method':'DENSE','contexts':'class-covering groups and full-size bootstrap variants','tabace_bank_used':False,'family_visible':False},
        {'method':'OligoICP adapted','contexts':'6/40/400 generic bootstrap candidates with IQR selection','tabace_bank_used':False,'family_visible':False},
        {'method':'PHE','contexts':'canonical source context, six custom configurations, source holdout calibration','tabace_bank_used':False,'family_visible':False},
        {'method':'TabFM-style / divide-and-conquer','contexts':'source-fitted feature views / subspaces','tabace_bank_used':False,'family_visible':False},
        {'method':'RF-PFN / BETA component / MICP component','contexts':'own forest bootstrap / generic bootstrap / source clustering','tabace_bank_used':False,'family_visible':False},
        {'method':'TabACE-Shared reimplemented','contexts':'five fixed Shared registry roles, pooled OOF policy','tabace_bank_used':True,'family_visible':False},
        {'method':'TabACE-Specific reimplemented','contexts':'five family-dependent branches, family-specific OOF policy','tabace_bank_used':True,'family_visible':True}]
    result={'updated_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'status':'PASS','coverage':coverage,'context_provenance':provenance,
        'all_ensemble_rules_applied_to_identical_TabACE_banks':False,
        'note':'Standalone algorithm comparisons and both TabACE references are complete. Applying every fusion rule to identical Shared/Specific banks would be a separate controlled experiment.'}
    (RESULTS/'context_usage_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    # The user-facing Markdown intentionally contains macro-average tables only.
    print(json.dumps({'status':'PASS','coverage_blocks':len(coverage),'comparison_rows':len(rows)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
