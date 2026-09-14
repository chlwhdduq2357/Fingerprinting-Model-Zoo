#!/usr/bin/env python3
"""데이터셋이나 모델 로딩 없이 로컬 checkpoint의 SHA256만 빠르게 검사한다."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_zoo.core import ROOT, models, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", nargs="+", choices=["A", "B", "B2", "C"])
    parser.add_argument("--model-id", nargs="+")
    args = parser.parse_args()

    rows = models()
    known = {row["model_id"] for row in rows}
    if args.model_id and (unknown := set(args.model_id) - known):
        parser.error("Unknown model ID: " + ", ".join(sorted(unknown)))

    chosen = [
        row for row in rows
        if (not args.group or row["group"] in args.group)
        and (not args.model_id or row["model_id"] in args.model_id)
    ]
    failures = []
    total_bytes = 0
    for row in chosen:
        path = ROOT / row["local_checkpoint_path"]
        if not path.is_file():
            failures.append(f"{row['model_id']}: missing {path}")
            continue
        total_bytes += path.stat().st_size
        actual = sha256(path)
        if not row.get("sha256") or actual != row["sha256"]:
            failures.append(f"{row['model_id']}: SHA256 mismatch")

    print(f"검사 완료: {len(chosen)}개, {total_bytes / 1024**3:.3f} GiB")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("모든 checkpoint의 SHA256이 metadata와 일치합니다.")


if __name__ == "__main__":
    main()
