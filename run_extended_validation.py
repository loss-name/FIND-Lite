"""Stage-three unseen-function and baseline-sensitivity validation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from find_lite import find_lite

NAMES=('zakharov','griewank','levy','schwefel','rotated_rastrigin')
def rotation(d): return np.linalg.qr(np.random.default_rng(12031+d).normal(size=(d,d)))[0]
def zakharov(x):
    a=np.sum(x*x); b=np.sum(.5*np.arange(1,x.size+1)*x); return float(a+b*b+b**4)
def griewank(x): return float(np.sum(x*x)/4000-np.prod(np.cos(x/np.sqrt(np.arange(1,x.size+1))))+1)
def levy(x):
    w=1+(x-1)/4
    return float(np.sin(np.pi*w[0])**2+np.sum((w[:-1]-1)**2*(1+10*np.sin(np.pi*w[:-1]+1)**2))+(w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])**2))
def schwefel(x): return float(418.9829*x.size-np.sum(x*np.sin(np.sqrt(np.abs(x)))))
def rastrigin(x): return float(10*x.size+np.sum(x*x-10*np.cos(2*np.pi*x)))
def make(name,d):
    if name=='zakharov': return zakharov,[(-5.,10.)]*d
    if name=='griewank': return griewank,[(-600.,600.)]*d
    if name=='levy': return levy,[(-10.,10.)]*d
    if name=='schwefel': return schwefel,[(-500.,500.)]*d
    q=rotation(d); return lambda x:rastrigin(q@x),[(-5.12,5.12)]*d
def ev(f,p): return np.asarray([float(f(x)) for x in p])
def de(f,bounds,budget,seed,F,CR):
    rng=np.random.default_rng(seed); lo,hi=np.asarray(bounds,float).T; d=len(bounds); n=2*d; pop=rng.uniform(lo,hi,(n,d)); val=ev(f,pop); calls=n
    while calls+n<=budget:
        for i in range(n):
            rest=np.delete(np.arange(n),i); a,b,c=rng.choice(rest,3,False); mutant=np.clip(pop[a]+F*(pop[b]-pop[c]),lo,hi); mask=rng.random(d)<CR; mask[rng.integers(d)]=True; t=np.where(mask,mutant,pop[i]); v=float(f(t)); calls+=1
            if v<=val[i]: pop[i],val[i]=t,v
    return {'best_f':float(val.min()),'evaluations':calls}
def pso(f,bounds,budget,seed,w,c1,c2):
    rng=np.random.default_rng(seed); lo,hi=np.asarray(bounds,float).T; d=len(bounds); n=2*d; pos=rng.uniform(lo,hi,(n,d)); vel=rng.uniform(-(hi-lo),hi-lo,(n,d))*.1; val=ev(f,pos); calls=n; personal=pos.copy(); pv=val.copy(); g=personal[int(pv.argmin())].copy(); gv=float(pv.min())
    while calls+n<=budget:
        vel=w*vel+c1*rng.random((n,d))*(personal-pos)+c2*rng.random((n,d))*(g-pos); pos=np.clip(pos+vel,lo,hi); val=ev(f,pos); calls+=n; ok=val<pv; personal[ok],pv[ok]=pos[ok],val[ok]; j=int(pv.argmin())
        if pv[j]<gv:g,gv=personal[j].copy(),float(pv[j])
    return {'best_f':gv,'evaluations':calls}
METHODS={
 'FIND_Lite':lambda f,b,n,s:find_lite(f,b,n,s),
 'DE_default':lambda f,b,n,s:de(f,b,n,s,.5,.9),
 'DE_alt_F08_CR05':lambda f,b,n,s:de(f,b,n,s,.8,.5),
 'PSO_default':lambda f,b,n,s:pso(f,b,n,s,.7298,1.49618,1.49618),
 'PSO_alt_w07_c15':lambda f,b,n,s:pso(f,b,n,s,.7,1.5,1.5),
}
def stat(x):
    x=np.asarray(x,float); return {'median':float(np.median(x)),'q1':float(np.quantile(x,.25)),'q3':float(np.quantile(x,.75)),'mean':float(x.mean()),'std':float(x.std()),'values':x.tolist()}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dims',nargs='+',type=int,default=[10,30]); p.add_argument('--functions',nargs='+',choices=NAMES,default=NAMES); p.add_argument('--runs',type=int,default=20); p.add_argument('--budget-factor',type=int,default=1000); p.add_argument('--seed-start',type=int,default=8000); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); out={'protocol':{'stage':3,'dims':a.dims,'functions':a.functions,'runs':a.runs,'seeds':[a.seed_start,a.seed_start+a.runs-1],'budget':f'{a.budget_factor}d','methods':list(METHODS)},'results':{}}
    for d in a.dims:
        for name in a.functions:
            f,b=make(name,d); case={}
            for m,run in METHODS.items():
                vals=[]
                for seed in range(a.seed_start,a.seed_start+a.runs):
                    row=run(f,b,a.budget_factor*d,seed)
                    if row['evaluations']>a.budget_factor*d:raise RuntimeError('budget violation')
                    vals.append(row['best_f'])
                case[m]=stat(vals)
            out['results'].setdefault(str(d),{})[name]=case
            print(f'd={d} {name}: '+' | '.join(f'{m}={case[m]["median"]:.5g}' for m in METHODS),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':main()
