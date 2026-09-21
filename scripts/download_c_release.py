#!/usr/bin/env python3
"""GitHub Release에서 C checkpoint archive를 받아 검증하고 안전하게 배치한다."""

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_zoo.core import ROOT, models, sha256

EXPECTED_C_MODELS = 128


def all_ready(rows: list[dict]) -> bool:
    return all(
        (path := ROOT / row["local_checkpoint_path"]).is_file()
        and sha256(path) == row["sha256"]
        for row in rows
    )


def download(url: str, destination: Path, expected_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "CIFAR10-Model-Zoo/1.0"})
    part = destination.with_suffix(destination.suffix + ".part")
    received = 0
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            if received > expected_bytes:
                raise RuntimeError("Release asset이 metadata의 예상 크기를 초과했습니다.")
            if received // (128 * 1024**2) != (received - len(chunk)) // (128 * 1024**2):
                print(f"downloaded {received / 1024**3:.2f} GiB", flush=True)
    part.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/c_release.json").read_text(encoding="utf-8"))
    rows = sorted(
        (row for row in models() if row.get("group") == "C"),
        key=lambda row: int(row["model_id"][1:]),
    )
    expected_ids = [f"C{number:03d}" for number in range(1, EXPECTED_C_MODELS + 1)]
    if [row["model_id"] for row in rows] != expected_ids:
        raise SystemExit(f"metadata에 C001-C{EXPECTED_C_MODELS:03d}가 연속적으로 등록되어 있지 않습니다.")
    if all_ready(rows):
        print(f"C001-C{EXPECTED_C_MODELS:03d}이 이미 존재하며 모든 SHA256이 일치합니다.")
        return

    cache = ROOT / "work" / config["asset_name"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file() or cache.stat().st_size != config["bytes"] or sha256(cache) != config["sha256"]:
        cache.unlink(missing_ok=True)
        print(f"GitHub Release 다운로드: {config['download_url']}", flush=True)
        download(config["download_url"], cache, config["bytes"])
    if cache.stat().st_size != config["bytes"] or sha256(cache) != config["sha256"]:
        raise SystemExit("Release archive의 크기 또는 SHA256이 일치하지 않습니다.")

    expected = {PurePosixPath(row["local_checkpoint_path"]).as_posix(): row for row in rows}
    with zipfile.ZipFile(cache) as archive:
        members = {info.filename for info in archive.infolist() if not info.is_dir()}
        if members != set(expected) | {"C_SHA256SUMS.txt"}:
            raise SystemExit("Release archive의 파일 목록이 metadata와 일치하지 않습니다.")
        for index, (member, row) in enumerate(expected.items(), 1):
            destination = ROOT / member
            if destination.is_file() and sha256(destination) == row["sha256"]:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            part = destination.with_suffix(destination.suffix + ".part")
            with archive.open(member) as source, part.open("wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            if sha256(part) != row["sha256"]:
                part.unlink(missing_ok=True)
                raise SystemExit(f"압축 해제 후 SHA256 불일치: {row['model_id']}")
            part.replace(destination)
            if index % 10 == 0:
                print(f"{index}/{len(rows)} extracted", flush=True)

    if not all_ready(rows):
        raise SystemExit("C checkpoint 최종 SHA256 검증에 실패했습니다.")
    if not args.keep_archive:
        cache.unlink(missing_ok=True)
    print(f"C001-C{EXPECTED_C_MODELS:03d} 다운로드, 배치와 SHA256 검증을 완료했습니다.")


if __name__ == "__main__":
    main()
