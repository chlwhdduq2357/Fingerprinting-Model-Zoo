import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gc
import hashlib
import time
from collections import defaultdict
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
from model_zoo.core import ROOT, models, save_models, sha256, tensor_hash, write_json
from model_zoo.loaders.unified import Classifier, build_network, load_model, read_state

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=['A', 'B', 'B2', 'C'])
    p.add_argument('--model-id', nargs='+')
    p.add_argument('--full', action='store_true', help='Evaluate all 10,000 test images; otherwise deterministic smoke subset')
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--no-update', action='store_true', help='Write per-model result files only; leave shared manifest unchanged')
    p.add_argument('--data-root', type=Path, default=ROOT.parent.parent / 'work' / 'data')
    a = p.parse_args()
    if not 1 <= a.samples <= 10000 or a.batch_size < 1 or a.threads < 1:
        p.error('Invalid samples, batch size or threads')
    torch.set_num_threads(a.threads)
    dataset = CIFAR10(root=str(a.data_root), train=False, transform=ToTensor(), download=True)
    # Fixed seed avoids relying on the original test file's ordering. No model-dependent selection.
    indices = list(range(10000)) if a.full else torch.randperm(10000, generator=torch.Generator().manual_seed(20260907))[:a.samples].tolist()
    data = DataLoader(Subset(dataset, indices), batch_size=a.batch_size, shuffle=False, num_workers=0)
    rows = models()
    by_id = {r['model_id']: r for r in rows}
    if a.model_id and set(a.model_id) - set(by_id):
        p.error('Unknown model ID')
    failures = []
    for row in rows:
        if a.group and row['group'] != a.group or a.model_id and row['model_id'] not in a.model_id:
            continue
        mid = row['model_id']
        started = time.monotonic()
        try:
            path = ROOT / row['local_checkpoint_path']
            if not path.is_file() or not row.get('sha256') or sha256(path) != row['sha256']:
                raise ValueError('Missing checkpoint or SHA256 mismatch')
            actual_device = 'cpu' if row.get('checkpoint_format') == 'torchscript_int8' else a.device
            if row.get('checkpoint_format') == 'torchscript_int8':
                model = load_model(mid, device='cpu', check_hash=False)
                network = model.network
                state = None
            else:
                state = read_state(path, row)
                if not all(torch.isfinite(t).all() for t in state.values()):
                    raise ValueError('Non-finite checkpoint tensor')
                row['state_dict_sha256'] = tensor_hash(state)
                network = build_network(row)
                network.load_state_dict(state, strict=True)
                model = Classifier(network, row).to(actual_device).eval()
                row['num_parameters'] = sum(t.numel() for t in model.network.parameters())
                row['num_trainable_parameters'] = sum(t.numel() for t in model.network.parameters() if t.requires_grad)
                row['num_nontrainable_parameters'] = row['num_parameters']-row['num_trainable_parameters']
                row['parameter_sha256'] = tensor_hash(dict(model.network.named_parameters()))
            row['framework_version'] = torch.__version__
            correct = total = 0
            with torch.inference_mode():
                for images, labels in data:
                    out = model(images.to(actual_device))
                    if out.shape != (len(images), 10) or not torch.isfinite(out).all():
                        raise ValueError('Invalid output dimensions or nonfinite logits')
                    correct += (out.argmax(1).cpu() == labels).sum().item()
                    total += len(labels)
            acc = 100 * correct / total
            if a.full:
                row['cifar10_test_accuracy_verified'] = acc
            else:
                row['cifar10_smoke_accuracy'] = acc
            result = dict(model_id=mid,passed=True,samples=total,correct=correct,accuracy_percent=acc,
                          checkpoint_sha256=row['sha256'],
                          full_test_set=a.full,subset_seed=None if a.full else 20260907,
                          indices_sha256=hashlib.sha256(str(indices).encode()).hexdigest(),
                          preprocessing='native',device=actual_device,torch_version=torch.__version__,
                          seconds=round(time.monotonic()-started,3),date=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            if a.full:
                row['full_verification'] = result
            else:
                row['verification'] = result
            row['status'] = 'verified'
            if a.full and row['cifar10_test_accuracy_reported'] is not None:
                gap = acc-row['cifar10_test_accuracy_reported']
                row['reported_verified_gap_pp'] = round(gap,6)
                row['accuracy_warning'] = f'Reported/full-test gap {gap:+.2f} percentage points' if abs(gap)>2 else None
            write_json(ROOT/'reports/verification'/(mid+('_full.json' if a.full else '_smoke.json')),result)
            print(mid,f'{correct}/{total} ({acc:.2f}%)',f'{result["seconds"]}s',flush=True)
            del model, state, network
            gc.collect()
        except Exception as e:
            row['status']='failed'
            row['verification_error']=str(e)
            failures.append(mid)
            print(mid,'FAILED',repr(e),flush=True)
        if not a.no_update:
            save_models(rows)
    duplicates = []
    for field in ['sha256', 'state_dict_sha256', 'parameter_sha256']:
        buckets = defaultdict(list)
        for row in rows:
            if row.get(field):
                buckets[row[field]].append(row['model_id'])
        duplicates.extend(dict(kind=field,model_ids=ids) for ids in buckets.values() if len(ids)>1)
    # Duplicate candidates are excluded from the usable population; never counted as independent.
    for dup in duplicates:
        for mid in dup['model_ids'][1:]:
            by_id[mid]['status']='duplicate'
    if not a.no_update:
        write_json(ROOT/'reports/duplicate_check.json',dict(duplicates=duplicates,checked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   models_with_file_hash=sum(bool(r.get('sha256')) for r in rows),
                   models_with_state_hash=sum(bool(r.get('state_dict_sha256')) for r in rows),
                   models_with_parameter_hash=sum(bool(r.get('parameter_sha256')) for r in rows)))
        save_models(rows)
    if failures or duplicates:
        raise SystemExit(1)

if __name__=='__main__':
    main()
