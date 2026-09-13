#!/usr/bin/env python3
"""Model Zoo의 선택, 로딩, 호출 방법을 한 번에 보여 주는 실행형 예제.

이 파일은 정밀 정확도 평가용 테스트가 아니다. CIFAR-10을 내려받지 않고 고정된
합성 이미지를 사용하여 loader API, metadata 필터링, 출력 형식과 모델 간 관계를
몇 분 안에 확인하는 smoke test다.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


# Windows PowerShell에서 Python의 기본 code page와 log 수집기의 encoding이 다를 때
# 한국어 설명이 깨지지 않도록 UTF-8 출력을 명시한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


# 이 파일을 repository root에서 직접 실행해도 ``import model_zoo``가 되도록
# outputs/ 디렉터리를 Python module 검색 경로에 추가한다. 패키지를 editable
# install한 환경에서는 이 코드가 없어도 되지만, 첫 사용자의 실행 편의를 위한 것이다.
ZOO_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ZOO_DIR.parent
if str(OUTPUTS_DIR) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_DIR))

from model_zoo import load_model, pair_relation  # noqa: E402
from model_zoo.core import ROOT, models  # noqa: E402


# CIFAR-10 class index의 표준 순서다. load_model()의 반환값은 probability가 아닌
# raw logits이므로 argmax index를 이 이름으로 변환해서 사람이 읽기 쉽게 표시한다.
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# 기본 실행에서 다루는 모델이다. A에서는 서로 다른 family를 고르고, B/B2에서는
# 같은 topology로 독립 학습된 두 모델씩 골라 hard-negative 사례도 함께 보여 준다.
DEFAULT_MODEL_IDS = (
    "A001",       # NIN
    "A002",       # ResNet20
    "A013",       # DenseNet
    "A027",       # MobileNetV2
    "A029",       # ShuffleNetV2
    "B001", "B002",       # 같은 topology/recipe, 서로 다른 seed
    "B2_001", "B2_002",   # 같은 topology/seed, 서로 다른 hyperparameter
)


def usable_accuracy(row: dict[str, Any]) -> float | None:
    """선택/표시에 쓸 수 있는 가장 강한 accuracy 근거를 반환한다.

    전체 test set 검증값을 우선하고, 없으면 source reported accuracy, 마지막으로
    작은 부분집합의 smoke accuracy를 쓴다. smoke 값은 full accuracy가 아니므로
    연구 결과를 보고할 때 세 필드를 반드시 구분해야 한다.
    """
    for key in (
        "cifar10_test_accuracy_verified",
        "cifar10_test_accuracy_reported",
        "cifar10_smoke_accuracy",
    ):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def select_models(
    rows: Iterable[dict[str, Any]],
    *,
    group: str | None = None,
    family: str | None = None,
    topology_id: str | None = None,
    min_accuracy: float | None = None,
    status: str = "verified",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """metadata만 사용해 모델을 선택하는 작은 재사용 함수.

    문자열 비교는 대소문자를 무시한다. ``topology_id``는 architecture 이름보다
    엄격한 구조 식별자이므로, 동일 architecture hard-negative를 고를 때 사용한다.
    이 함수는 checkpoint를 메모리에 올리지 않아 90개 전체를 대상으로도 즉시 끝난다.
    """
    selected: list[dict[str, Any]] = []
    for row in rows:
        if status and row.get("status") != status:
            continue
        if group and str(row.get("group", "")).casefold() != group.casefold():
            continue
        if family and str(row.get("architecture_family", "")).casefold() != family.casefold():
            continue
        if topology_id and row.get("topology_id") != topology_id:
            continue
        accuracy = usable_accuracy(row)
        if min_accuracy is not None and (accuracy is None or accuracy < min_accuracy):
            continue
        selected.append(row)
    selected.sort(key=lambda item: item["model_id"])
    return selected[:limit] if limit is not None else selected


def fixed_probe_batch(batch_size: int, seed: int, device: str) -> torch.Tensor:
    """모든 모델에 똑같이 입력할 재현 가능한 RGB 합성 이미지를 만든다.

    값 범위는 [0, 1]이며 아직 normalize하지 않았다. native/common 전처리는
    load_model()이 반환하는 wrapper 안에서 적용된다. 실제 fingerprint 연구에서는
    이 부분을 CIFAR-10 또는 설계한 query set으로 교체하면 된다.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.rand(batch_size, 3, 32, 32, generator=generator)
    return images.to(device)


def describe_catalog(rows: list[dict[str, Any]]) -> None:
    """checkpoint를 로드하지 않고 zoo의 구성과 선택 예시를 출력한다."""
    print("\n[1] metadata 탐색과 모델 선택")
    print("- group별 개수:", dict(sorted(Counter(r["group"] for r in rows).items())))
    print("- architecture family 수:", len({r["architecture_family"] for r in rows}))

    # 예시 1: A에서 family가 ResNet인 모델 중 확인 가능한 accuracy가 90% 이상인 것.
    resnets = select_models(rows, group="A", family="ResNet", min_accuracy=90.0)
    print("- A / ResNet / accuracy>=90 후보:", [r["model_id"] for r in resnets])

    # 예시 2: B의 첫 모델과 topology_id가 완전히 같은 모델을 선택한다. 이름이 비슷한
    # 모델보다 실제 topology_id가 같은 모델을 고르는 것이 hard-negative 구성에 안전하다.
    b_reference = next(r for r in rows if r["model_id"] == "B001")
    same_topology = select_models(rows, topology_id=b_reference["topology_id"], limit=5)
    print("- B001과 같은 topology의 앞 5개:", [r["model_id"] for r in same_topology])

    # 예시 3: B2 training_recipe는 구조화된 dict다. 원하는 hyperparameter 조건을
    # metadata 단계에서 직접 골라 checkpoint 로딩 대상을 줄일 수 있다.
    b2_with_sam = [
        r for r in select_models(rows, group="B2")
        if float((r.get("training_recipe") or {}).get("sam_rho") or 0.0) > 0.0
    ]
    print("- B2 중 SAM을 사용한 모델 수:", len(b2_with_sam))
    print("  예시:", [r["model_id"] for r in b2_with_sam[:5]])


def describe_relations() -> None:
    """forensic attribution 평가에 중요한 모델 pair 관계를 보여 준다."""
    print("\n[2] 모델 pair 관계")
    examples = (
        ("B001", "B002", "같은 구조·recipe, 다른 seed"),
        ("B2_001", "B2_002", "같은 구조·seed, 다른 hyperparameter"),
        ("B001", "B2_001", "같은 ResNet family, 다른 block topology"),
        ("A001", "A013", "서로 다른 architecture"),
    )
    for first, second, note in examples:
        print(f"- {first} ↔ {second} ({note})")
        print(" ", pair_relation(first, second))


def run_one_model(
    model_id: str,
    images: torch.Tensor,
    *,
    device: str,
    check_hash: bool,
) -> torch.Tensor:
    """한 모델을 native preprocessing으로 호출하고 출력 계약을 검사한다."""
    model = load_model(model_id, device=device, preprocessing="native", check_hash=check_hash)
    assert not model.training, f"{model_id}: model.eval() 상태가 아닙니다."

    # inference_mode는 gradient graph를 만들지 않으므로 빠르고 메모리 사용량이 작다.
    with torch.inference_mode():
        logits = model(images)

    expected_shape = (images.shape[0], 10)
    assert tuple(logits.shape) == expected_shape, (
        f"{model_id}: 예상 출력 {expected_shape}, 실제 출력 {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), f"{model_id}: logits에 NaN/Inf가 있습니다."

    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities[0].max(dim=0)
    metadata = model.metadata
    print(
        f"- {model_id:<6} | {metadata['architecture_family']:<13} "
        f"| logits={tuple(logits.shape)} "
        f"| 첫 입력 예측={CIFAR10_CLASSES[prediction.item()]:<10} "
        f"({confidence.item():.3f})"
    )

    # 큰 network 객체는 즉시 해제하여 여러 architecture를 CPU에서도 순서대로
    # 시험할 때 peak memory가 불필요하게 커지지 않도록 한다.
    result = logits.detach().cpu()
    del model, logits, probabilities
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def compare_preprocessing(images: torch.Tensor, device: str, check_hash: bool) -> None:
    """같은 checkpoint를 native/common/none 조건으로 호출하는 방법을 보인다."""
    print("\n[4] 전처리 조건 선택 (A002 한 모델)")
    settings = {
        "native": {},
        "common": {
            "common_mean": [0.4914, 0.4822, 0.4465],
            "common_std": [0.2023, 0.1994, 0.2010],
        },
        "none": {},
    }
    outputs: dict[str, torch.Tensor] = {}
    for preprocessing, kwargs in settings.items():
        model = load_model(
            "A002", device=device, preprocessing=preprocessing,
            check_hash=check_hash, **kwargs,
        )
        with torch.inference_mode():
            outputs[preprocessing] = model(images[:1]).cpu()
        del model
    for name, output in outputs.items():
        print(f"- {name:<6}: predicted={CIFAR10_CLASSES[output.argmax(1).item()]}, "
              f"logit_norm={output.norm().item():.3f}")
    print("  같은 weights라도 preprocessing 조건이 바뀌면 응답이 달라질 수 있습니다.")


def compare_response_fingerprints(logits_by_id: dict[str, torch.Tensor]) -> None:
    """고정 query에 대한 logits를 가장 단순한 응답 fingerprint로 비교한다.

    이는 연구용 fingerprint algorithm이 아니라 loader 사용 예시다. 동일 입력 batch의
    logits를 1차원으로 펴고 cosine similarity를 계산해 서로 다른 독립 모델의 응답이
    실제로 같지 않음을 빠르게 확인한다.
    """
    print("\n[5] 간단한 출력 fingerprint 비교")
    pairs = (("B001", "B002"), ("B2_001", "B2_002"), ("A001", "A013"))
    for first, second in pairs:
        if first not in logits_by_id or second not in logits_by_id:
            continue
        a = logits_by_id[first].flatten().float()
        b = logits_by_id[second].flatten().float()
        cosine = F.cosine_similarity(a, b, dim=0).item()
        mean_abs_diff = (a - b).abs().mean().item()
        print(f"- {first} ↔ {second}: cosine={cosine:+.4f}, mean|Δlogit|={mean_abs_diff:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Model Zoo의 선택·로딩·호출 예제를 빠르게 실행합니다."
    )
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODEL_IDS), metavar="MODEL_ID",
        help="실제로 호출할 model_id 목록 (기본: 여러 A/B/B2 대표 모델)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="합성 query batch 크기")
    parser.add_argument("--seed", type=int, default=20260907, help="합성 query 생성 seed")
    parser.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread 수")
    parser.add_argument(
        "--skip-hash", action="store_true",
        help="반복 실험 시 파일 SHA256 검사를 생략 (기본값은 안전하게 검사)",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="metadata 선택과 pair 관계만 출력하고 checkpoint는 로드하지 않음",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size는 1 이상이어야 합니다.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다.")
    torch.set_num_threads(max(1, args.threads))

    started = time.perf_counter()
    rows = models()
    by_id = {row["model_id"]: row for row in rows}
    unknown = [model_id for model_id in args.models if model_id not in by_id]
    if unknown:
        raise SystemExit(f"알 수 없는 model_id: {', '.join(unknown)}")

    describe_catalog(rows)
    describe_relations()
    if args.list_only:
        print(f"\n완료: checkpoint를 로드하지 않았습니다. ({time.perf_counter()-started:.2f}초)")
        return

    # 파일이 빠진 경우 torch.load의 긴 traceback 대신 재현 가능한 다운로드 명령을 안내한다.
    missing = [
        model_id for model_id in args.models
        if not (ROOT / by_id[model_id]["local_checkpoint_path"]).is_file()
    ]
    if missing:
        joined = " ".join(missing)
        raise SystemExit(
            f"checkpoint가 없습니다: {joined}\n"
            f"먼저 실행: python scripts/download_models.py --model-id {joined}"
        )

    print("\n[3] 다양한 모델의 통일된 호출")
    images = fixed_probe_batch(args.batch_size, args.seed, args.device)
    logits_by_id: dict[str, torch.Tensor] = {}
    for model_id in args.models:
        logits_by_id[model_id] = run_one_model(
            model_id, images, device=args.device, check_hash=not args.skip_hash,
        )

    compare_preprocessing(images, args.device, check_hash=not args.skip_hash)
    compare_response_fingerprints(logits_by_id)
    print(f"\n모든 예제와 assertion을 통과했습니다. 총 실행 시간: {time.perf_counter()-started:.2f}초")


if __name__ == "__main__":
    main()
