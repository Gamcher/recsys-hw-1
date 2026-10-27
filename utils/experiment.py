"""Reproducible full-catalog experiment. Test is opened only after selection."""
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import pandas as pd
import scipy
import torch
from models.item_knn import ItemKNN
from models.als import ExplicitALS
from models.ease import EASE
from models.fm import FM
from models.custom_als import CustomALS
from models.common import interactions
from utils.evaluation import cohort, evaluate, topk

SEED = 42
GRIDS = {
    'ItemKNN': {'neighbors':[40,100], 'shrinkage':[0,20]},
    'ExplicitALS': {'rank':[8,16], 'regularization':[1.,10.], 'iterations':[8], 'seed':[SEED]},
    'EASE': {'regularization':[10.,100.,1000.]},
    'FM': {'rank':[8,16], 'regularization':[1e-4,1e-3], 'learning_rate':[.01],
           'epochs':[8], 'seed':[SEED]},
}
CUSTOM_GRID = [{'alpha':.25,'tau_days':180}, {'alpha':.5,'tau_days':365},
               {'alpha':1.,'tau_days':365}]
CLASSES = {'ItemKNN':ItemKNN,'ExplicitALS':ExplicitALS,'EASE':EASE,'FM':FM,'CustomALS':CustomALS}


class Popularity:
    def fit(self, frame, n_users, n_items, metadata=None):
        self.counts = np.asarray(interactions(frame,n_users,n_items,positive=True).sum(axis=0)).ravel()
        return self

    def score(self, users):
        return np.broadcast_to(self.counts,(len(users),len(self.counts)))


def combinations(grid):
    return [dict(zip(grid,values)) for values in itertools.product(*grid.values())]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_results(output_dir='results'):
    """Reject interrupted runs and tables modified after successful completion."""
    out = Path(output_dir)
    status = json.loads((out/'status.json').read_text())
    expected = {'protocol.json','selected.json','validation.csv','test.csv','recommendations.json'}
    if status.get('state') != 'completed' or set(status.get('artifact_sha256', {})) != expected:
        raise ValueError('Experiment is incomplete; old tables are not current results')
    for name, digest in status['artifact_sha256'].items():
        if sha256(out/name) != digest:
            raise ValueError(f'Result checksum mismatch: {name}')
    return status


def run(data_dir='data', output_dir='results'):
    """Always recompute selection; no silent reuse of stale result files."""
    root = Path(data_dir)/'processed'
    out = Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    run_id = uuid.uuid4().hex
    status_path = out/'status.json'
    def write_status(value):
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(status_path)
    write_status({'run_id':run_id,'state':'running'})
    train = pd.read_csv(root/'train.csv.gz',keep_default_na=False)
    validation = pd.read_csv(root/'validation.csv.gz',keep_default_na=False)
    metadata = pd.read_json(root/'item_metadata.jsonl',lines=True)
    n_users,n_items = int(train.user_idx.max()+1),int(train.item_idx.max()+1)
    truth = cohort(validation,train,n_items)
    spec = {'seed':SEED,'primary_metric':'NDCG@10','secondary_metric':'Recall@10',
            'relevance':'rating >= 4','K':10,'refit_train_validation':False,
            'validation_grid':GRIDS,'custom_target_grid':CUSTOM_GRID,
            'custom_solver':'rank/regularization/iterations from best validation ExplicitALS',
            'versions':{'python':platform.python_version(),'numpy':np.__version__,
                        'pandas':pd.__version__,'scipy':scipy.__version__,'torch':torch.__version__},
            'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),
            'train_sha256':sha256(root/'train.csv.gz'),
            'validation_sha256':sha256(root/'validation.csv.gz'),
            'metadata_sha256':sha256(root/'item_metadata.jsonl'),
            'validation_users':len(truth),
            'validation_cohort_sha256':hashlib.sha256(json.dumps(truth,default=sorted,sort_keys=True).encode()).hexdigest()}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2))
    rows, best_models, selected = [],{},{}
    for name in ['ItemKNN','ExplicitALS','EASE','FM','CustomALS']:
        configs = combinations(GRIDS[name]) if name!='CustomALS' else [
            {**selected['ExplicitALS'],**target} for target in CUSTOM_GRID]
        best = -np.inf
        for params in configs:
            start = time.perf_counter()
            model = CLASSES[name](**params).fit(train,n_users,n_items,metadata)
            fit_seconds = time.perf_counter()-start
            metrics = evaluate(model,truth,train,n_users,n_items)
            row = {'model':name,'params':json.dumps(params,sort_keys=True),**metrics,
                   'fit_seconds':fit_seconds,'loss_history':json.dumps(getattr(model,'loss_history',[]))}
            rows.append(row)
            pd.DataFrame(rows).to_csv(out/'validation.csv',index=False)
            print('VALIDATION',name,params,metrics,'fit seconds',round(fit_seconds,2),flush=True)
            if metrics['NDCG@10'] > best:
                best,best_models[name],selected[name] = metrics['NDCG@10'],model,params
            del model
            gc.collect()
    # ID-only ablation uses exactly the chosen FM optimizer settings, not test selection.
    start=time.perf_counter()
    ablation = FM(**selected['FM'],include_metadata=False).fit(train,n_users,n_items,metadata)
    ablation_fit_seconds=time.perf_counter()-start
    ablation_metrics=evaluate(ablation,truth,train,n_users,n_items)
    rows.append({'model':'FM_ID_only_ablation','params':json.dumps({**selected['FM'],'include_metadata':False}),
                 **ablation_metrics,'fit_seconds':ablation_fit_seconds,
                 'loss_history':json.dumps(ablation.loss_history)})
    best_models['FM_ID_only_ablation']=ablation
    selected['FM_ID_only_ablation']={**selected['FM'],'include_metadata':False}
    popularity=Popularity().fit(train,n_users,n_items)
    best_models['Popularity_baseline']=popularity
    selected['Popularity_baseline']={}
    rows.append({'model':'Popularity_baseline','params':'{}',**evaluate(popularity,truth,train,n_users,n_items),
                 'fit_seconds':0.,'loss_history':'[]'})
    pd.DataFrame(rows).to_csv(out/'validation.csv',index=False)
    # Freeze settings before any final test access. No model is retrained.
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    test = pd.read_csv(root/'test.csv.gz',keep_default_na=False)
    known = pd.concat([train,validation],ignore_index=True)
    test_truth = cohort(test,known,n_items)
    spec.update(test_users=len(test_truth),test_sha256=sha256(root/'test.csv.gz'),
                test_cohort_sha256=hashlib.sha256(json.dumps(test_truth,default=sorted,sort_keys=True).encode()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(spec,indent=2))
    final=[]
    for name,model in best_models.items():
        metrics=evaluate(model,test_truth,known,n_users,n_items)
        final.append({'model':name,**metrics,'params':json.dumps(selected.get(name,{}),sort_keys=True)})
        print('TEST',name,metrics,flush=True)
    pd.DataFrame(final).to_csv(out/'test.csv',index=False)
    # A reproducible example of personalized top-10, not an extra evaluation sample.
    example_users=np.array(sorted(test_truth)[:3])
    known_matrix=interactions(known,n_users,n_items)
    examples={name:topk(model.score(example_users),known_matrix[example_users]).tolist()
              for name,model in best_models.items()}
    (out/'recommendations.json').write_text(json.dumps({'user_idx':example_users.tolist(),'top10_item_idx':examples},indent=2))
    artifacts = ['protocol.json','selected.json','validation.csv','test.csv','recommendations.json']
    write_status({'run_id':run_id,'state':'completed',
                  'artifact_sha256':{name:sha256(out/name) for name in artifacts}})
    print('Experiment complete; results:',out,flush=True)
    return pd.DataFrame(final)


if __name__=='__main__':
    run()
