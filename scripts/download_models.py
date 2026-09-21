import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_zoo.core import models, save_models, download_one

def main():
    p = argparse.ArgumentParser(description='Download only manifest-selected CIFAR-10 checkpoints; no full zoo archives.')
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument('--group', choices=['A', 'B', 'B2'])
    s.add_argument('--all', action='store_true')
    s.add_argument('--model-id', nargs='+')
    p.add_argument('--workers', type=int, default=1, help='1-4 bounded parallel checkpoint requests')
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        p.error('workers must be 1..4')
    rows = models()
    if a.model_id and set(a.model_id) - {r['model_id'] for r in rows}:
        p.error('Unknown model ID')
    failures = []
    chosen = [r for r in rows if a.all or r['group'] == a.group or a.model_id and r['model_id'] in a.model_id]
    # Workers use private rows. Only the main thread writes the shared manifest.
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = {pool.submit(download_one, dict(r)): r for r in chosen}
        for future in as_completed(jobs):
            row = jobs[future]
            try:
                row.update(future.result())
                print(row['model_id'], row['status'], flush=True)
            except Exception as e:
                failures.append(row['model_id'])
                row['download_error'] = str(e)
                print(row['model_id'], 'FAILED', e, flush=True)
            save_models(rows)
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
