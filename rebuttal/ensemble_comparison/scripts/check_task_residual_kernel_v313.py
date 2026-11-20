"""Check analytical LOO against explicit refitting, including multi-output."""
import numpy as np
import task_residual_kernel_v313 as kernel


def main():
    rng=np.random.default_rng(0)
    F=rng.normal(size=(9,5));d=kernel.distances(F,F)
    K=np.exp(-d/np.median(d[d>1e-12]))
    for width in (1,4):
        p=rng.normal(size=(9,width));target=rng.normal(size=(9,width));r=target-p
        for penalty in (.01,.1,1.,10.):
            inverse=np.linalg.inv(K+penalty*np.eye(len(F)))
            loo=p+r-(inverse@r)/np.diag(inverse)[:,None]
            brute=[]
            for i in range(len(F)):
                keep=np.arange(len(F))!=i
                alpha=np.linalg.solve(K[np.ix_(keep,keep)]+penalty*np.eye(sum(keep)),r[keep])
                brute.append(p[i]+K[i,keep]@alpha)
            np.testing.assert_allclose(loo,np.array(brute),atol=1e-11,rtol=1e-10)
    for task in ('classification','regression'):
        if task=='classification':
            p=np.tile([.2,.3,.5],(9,1));y=np.array([0,1,2]*3)
        else:p=np.zeros(9);y=rng.normal(size=9)
        model=kernel.fit(F,p,y,task)
        output=kernel.predict(F[:2],p[:2],F,model,task)
        assert np.isfinite(output).all() and len(output)==2
        assert len(model['trials'])==12
        if task=='classification':np.testing.assert_allclose(output.sum(axis=1),1)
    print('PASS: exact LOO equals 72 explicit refits; both task APIs finite')


if __name__=='__main__':main()
