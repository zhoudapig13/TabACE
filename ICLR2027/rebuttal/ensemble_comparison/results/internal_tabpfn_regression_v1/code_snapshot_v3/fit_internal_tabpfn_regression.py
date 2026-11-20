"""Reuse the declared nested dataset-OOF protocol with a separate result root."""
import argparse
from internal_tabpfn_regression import OUT
import fit_internal_regression_ensembles as fitter

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five','seven'],required=True);p.add_argument('--fold',type=int,required=True)
    a=p.parse_args();assert a.fold in range(5)
    fitter.OUT=OUT
    fitter.fit(a.track,a.fold)
