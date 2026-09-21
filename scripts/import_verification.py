"""Import full results from --no-update only when they match current checkpoint hashes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_zoo.core import ROOT, models, save_models, read_json

rows=models()
count=0
for row in rows:
    smoke_path=ROOT/'reports/verification'/(row['model_id']+'_smoke.json')
    if smoke_path.exists():
        smoke=read_json(smoke_path)
        if (not smoke.get('passed') or smoke.get('full_test_set') or smoke.get('preprocessing')!='native'
            or smoke.get('checkpoint_sha256') not in (None, row['sha256'])):
            raise ValueError(f'Invalid/stale result: {smoke_path}')
        row['verification']=smoke
    path=ROOT/'reports/verification'/(row['model_id']+'_full.json')
    if not path.exists():continue
    result=read_json(path)
    if (not result.get('passed') or result.get('samples')!=10000 or not result.get('full_test_set')
        or result.get('preprocessing')!='native' or result.get('checkpoint_sha256')!=row['sha256']):
        raise ValueError(f'Invalid/stale result: {path}')
    row['cifar10_test_accuracy_verified']=result['accuracy_percent']
    row['full_verification']=result
    reported=row['cifar10_test_accuracy_reported']
    if reported is not None:
        gap=result['accuracy_percent']-reported
        row['reported_verified_gap_pp']=round(gap,6)
        row['accuracy_warning']=f'Reported/full-test gap {gap:+.2f} percentage points' if abs(gap)>2 else None
    count+=1
save_models(rows)
print('Imported',count,'matching full-test results')
