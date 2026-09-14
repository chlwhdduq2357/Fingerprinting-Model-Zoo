#!/usr/bin/env python3
"""C 120개를 데이터셋 없이 hash, strict load, 단일 입력으로 빠르게 검사한다."""

import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_zoo import load_model, pair_relation
from model_zoo.core import ROOT, models, sha256, write_json


def main() -> None:
    started = time.perf_counter()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = models()
    by_id = {row["model_id"]: row for row in rows}
    children = sorted(
        (row for row in rows if row.get("group") == "C"),
        key=lambda row: int(row["model_id"][1:]),
    )
    expected_ids = [f"C{number:03d}" for number in range(1, 121)]
    assert [row["model_id"] for row in children] == expected_ids
    assert Counter(row["transform_type"] for row in children) == {
        "ft5": 30, "prune50": 30, "ptq_int8": 30, "prune20": 30
    }

    expected_files = set()
    probe = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(20260914))
    for index, row in enumerate(children, 1):
        number = int(row["model_id"][1:])
        parent_id = f"A{((number - 1) // 4) + 1:03d}"
        transform = ("ft5", "prune50", "ptq_int8", "prune20")[(number - 1) % 4]
        suffix = ".pt" if transform == "ptq_int8" else ".pth"
        filename = f"{row['model_id']}_{parent_id}_{transform}{suffix}"
        assert row["parent_id"] == parent_id and row["transform_type"] == transform
        assert Path(row["local_checkpoint_path"]).name == filename
        assert pair_relation(parent_id, row)["same_lineage"]
        path = ROOT / row["local_checkpoint_path"]
        assert path.is_file() and sha256(path) == row["sha256"]
        expected_files.add(filename)

        # load_model은 일반 checkpoint에는 strict state_dict 로딩을 적용하고,
        # INT8 자식에는 검증된 TorchScript CPU 로더를 적용한다.
        model = load_model(row["model_id"], device="cpu", check_hash=False)
        with torch.inference_mode():
            logits = model(probe)
        assert tuple(logits.shape) == (1, 10) and torch.isfinite(logits).all()
        del model, logits
        gc.collect()
        if index % 10 == 0:
            print(f"{index:3d}/120 로딩 및 추론 통과", flush=True)

    actual_files = {path.name for path in (ROOT / "checkpoints/C").iterdir() if path.is_file()}
    assert actual_files == expected_files, "metadata에 없거나 누락된 C 파일이 있습니다."
    seconds = round(time.perf_counter() - started, 3)
    report = {
        "passed": True, "models": len(children),
        "transforms": dict(Counter(row["transform_type"] for row in children)),
        "checkpoint_bytes": sum((ROOT / row["local_checkpoint_path"]).stat().st_size for row in children),
        "checks": ["filename/parent/transform", "sha256", "strict load", "finite [1,10] logits"],
        "seconds": seconds, "torch_version": torch.__version__,
    }
    write_json(ROOT / "reports/c_quick_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
