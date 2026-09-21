#!/usr/bin/env python3
"""Model Zoo의 핵심 기능만 실행해 보는 짧은 예제."""

import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

# Windows에서도 한국어 출력이 깨지지 않도록 표준 출력을 UTF-8로 맞춘다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# editable install 전에도 repository root에서 바로 실행할 수 있게 경로를 추가한다.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model_zoo import load_model, pair_relation
from model_zoo.core import models


def main() -> None:
    # 1) metadata만 읽으면 checkpoint를 메모리에 올리지 않고 모델을 선택할 수 있다.
    rows = models()
    print("그룹별 모델 수:", dict(Counter(row["group"] for row in rows)))
    a_resnets = [
        row["model_id"] for row in rows
        if row["group"] == "A" and row["architecture_family"] == "ResNet"
    ]
    print("A의 ResNet 모델:", a_resnets)

    # 2) 이름만 지정하면 architecture 생성, checkpoint 로딩, native normalization,
    # eval 설정까지 끝난 동일 인터페이스의 모델을 얻는다.
    parent = load_model("A002", device="cpu")
    child = load_model("C005", device="cpu")       # A002를 5 epochs fine-tuning
    quantized = load_model("C007", device="cpu")   # A002의 static INT8 모델은 CPU 전용

    # 3) 입력은 normalize하지 않은 RGB float tensor [N,3,32,32], 값 범위 [0,1]이다.
    # 여기서는 실행을 짧게 유지하기 위해 실제 사진 대신 이미지 한 장을 만든다.
    image = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        parent_logits = parent(image)
        child_logits = child(image)
        int8_logits = quantized(image)
    print("출력 크기:", parent_logits.shape)  # CIFAR-10이므로 [1,10]
    print("A002 예측 class index:", parent_logits.argmax(dim=1).item())

    # 4) 같은 입력에 대한 logits cosine similarity를 간단히 비교할 수 있다.
    print("A002-C005 logits cosine:", F.cosine_similarity(parent_logits, child_logits).item())
    print("A002-C007 logits cosine:", F.cosine_similarity(parent_logits, int8_logits).item())

    # 5) lineage/topology 관계는 forensic 실험의 pair label로 바로 사용할 수 있다.
    print("A002-C005:", pair_relation("A002", "C005"))  # 같은 lineage
    print("B001-B002:", pair_relation("B001", "B002"))  # 같은 구조, 독립 학습
    print("A001-A002:", pair_relation("A001", "A002"))  # 서로 다른 구조


if __name__ == "__main__":
    main()
