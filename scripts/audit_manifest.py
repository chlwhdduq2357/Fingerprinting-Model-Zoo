"""Audit the actual local original population and its provenance contracts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
from collections import Counter
from itertools import combinations
from model_zoo.core import ROOT, models, read_json, sha256, write_json
from model_zoo import pair_relation

required='model_id group lineage_id parent_id architecture architecture_family source_repository source_url checkpoint_url original_filename local_checkpoint_path sha256 framework framework_version checkpoint_format num_parameters input_size normalization_mean normalization_std cifar10_test_accuracy_reported cifar10_test_accuracy_verified training_seed optimizer learning_rate weight_decay augmentation training_recipe notes download_date'.split()
rows=models()
assert len({r['model_id'] for r in rows})==len(rows)
assert len({r['lineage_id'] for r in rows})==len(rows)
assert all(set(required)<=set(r) for r in rows)
assert all(r['parent_id'] is None and r['group'] in ('A','B','B2') and r['dataset']=='CIFAR-10' for r in rows)
assert all(r['status']=='verified' for r in rows), 'Some candidates are not usable'
for r in rows:
    path=(ROOT/r['local_checkpoint_path']).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert sha256(path)==r['sha256']
for field in ['sha256','state_dict_sha256','parameter_sha256']:
    assert all(r.get(field) for r in rows)
    assert len({r[field] for r in rows})==len(rows),field
B=[r for r in rows if r['group']=='B']
B2=[r for r in rows if r['group']=='B2']
assert len({r['training_seed'] for r in B})==len(B)
assert len({r['training_run_id'] for r in B})==len(B)
assert len({r['topology_id'] for r in B})==1
for r in B:
    params=read_json(ROOT/'metadata/evidence'/r['model_id']/'params.json')
    assert params['seed']==r['training_seed']
    assert params['training::checkpoint_dir'] is None
    assert params['model::type']=='Resnet18' and params['model::o_dim']==10
    logs=[__import__('json').loads(line) for line in (ROOT/'metadata/evidence'/r['model_id']/'result.jsonl').read_text().splitlines()]
    logged=next(x for x in logs if x['training_iteration']==r['checkpoint_iteration'])
    assert abs(r['cifar10_test_accuracy_reported']-100*logged['test_acc'])<1e-8
assert len(B2)==30
assert len({r['topology_id'] for r in B2})==1
assert len({r['hyperparameter_signature'] for r in B2})==30
assert {r['training_seed'] for r in B2}=={0}
assert all(r['loader_spec']=={'adapter':'sharpness_preact_resnet18','state_key':'last'} for r in B2)
assert all(r['training_recipe']['epochs']==200 and r['training_recipe']['batch_size']==128 for r in B2)
assert all(r['training_recipe']['max_learning_rate']==r['learning_rate'] for r in B2)
with (ROOT/'metadata/models.csv').open(encoding='utf-8-sig',newline='') as f:
    csvrows=list(csv.DictReader(f))
assert [r['model_id'] for r in csvrows]==[r['model_id'] for r in rows]
relations=[pair_relation(a,b) for a,b in combinations(rows,2)]
assert not any(r['same_lineage'] for r in relations)
expected_hard_negatives=len(B)*(len(B)-1)//2+len(B2)*(len(B2)-1)//2
assert sum(r['hard_negative'] for r in relations)==expected_hard_negatives
result=dict(passed=True,models=len(rows),groups=dict(Counter(r['group'] for r in rows)),
            unique_lineages=len(rows),pairs=len(relations),
            hard_negative_pairs=sum(r['hard_negative'] for r in relations),
            different_architecture_pairs=sum(r['different_architecture'] for r in relations),
            distinct_B_seeds=len({r['training_seed'] for r in B}),
            distinct_B2_hyperparameter_signatures=len({r['hyperparameter_signature'] for r in B2}),
            B2_seed_values=sorted({r['training_seed'] for r in B2}))
write_json(ROOT/'reports/manifest_audit.json',result)
print(result)
