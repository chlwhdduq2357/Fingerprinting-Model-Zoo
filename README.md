# CIFAR-10 Forensic Model Attribution Zoo

여러 CIFAR-10 분류 모델의 fingerprint를 비교해 **architecture**, **독립 학습**, **동일 lineage의 변형**을 구분하기 위한 Model Zoo이다. 모든 모델은 `model_id`로 선택하고 같은 입력·출력 인터페이스로 호출할 수 있다.

현재 metadata에는 총 **210개**가 등록되어 있다.

| 그룹 | 개수 | 역할 |
|---|---:|---|
| A | 30 | architecture가 다양한 독립 원본 |
| B | 30 | 같은 ResNet18 구조 및 hyperparameter 학습 설정, 서로 다른 seed로 학습 |
| B2 | 30 | 같은 PreActResNet18 구조 및 seed, 서로 다른 hyperparameter 설정으로 학습 |
| C | 120 | A 30개에서 각각 만든 네 가지 변형 |

## 1. 프로젝트 구조

```text
Fingerprinting-Model-Zoo/
├── checkpoints/
│   ├── A/                    # architecture-diverse originals 30개
│   ├── B/                    # seed가 다른 ResNet18 originals 30개
│   ├── B2/                   # hyperparameter가 다른 PreActResNet18 originals 30개
│   └── C/                    # A의 descendants 120개
├── configs/                  # 모델 선택과 B2 source 설정
├── loaders/
│   └── unified.py            # load_model(), native preprocessing, pair_relation()
├── metadata/
│   ├── models.json           # 전체 모델 manifest와 lineage 정보
│   ├── models.csv            # 표 형식의 동일 metadata
│   ├── sources.json          # 출처 repository, revision, 다운로드 방식
│   ├── pairs.csv             # 모델 pair 관계를 내보낸 결과
│   └── evidence/             # source snapshot, 학습 log와 독립성 근거
├── reports/                  # 구축·검증 결과
├── runs/                     # C fine-tuning 실행 log가 있을 때 저장되는 위치
├── scripts/
│   ├── download_models.py    # A/B/B2 선택 다운로드
│   ├── verify_hashes.py      # dataset 없이 SHA256 검사
│   ├── verify_c_quick.py     # C의 빠른 load/inference 검사
│   ├── verify_models.py      # CIFAR-10 sample 또는 전체 정확도 검사
│   ├── list_models.py        # 등록 모델 목록
│   └── export_pairs.py       # pair label 생성
├── tests/                    # loader와 변환 contract 단위 테스트
├── vendor/                   # 외부 실행 없이 재현하는 최소 architecture 정의
├── example.py                # 핵심 API의 짧은 실행 예제
├── COLAB_C_FAST.ipynb        # C 생성용 Colab notebook
├── pyproject.toml
└── requirements-lock.txt
```

Checkpoint는 용량 때문에 Git에 포함되지 않고 `.gitignore` 처리된다. `metadata/`, `vendor/`, loader와 script는 Git으로 보존한다.

## 2. A/B/B2/C 모델군

### A: architecture-diverse originals

NIN, ResNet, PreResNet, SE-ResNet, PyramidNet, DenseNet, WideResNet, RoR, Shake-Shake, MobileNetV2, ShuffleNetV2, RepVGG 등 **18개 family의 30개 공개 pretrained CIFAR-10 모델**이다. 각 모델은 별도의 original lineage이며 서로 다른 architecture population을 구성한다.

### B: same topology, different seed

3×3 stem과 `[2,2,2,2]` BasicBlock을 사용하는 CIFAR-10용 **post-activation ResNet18 30개**이다. 학습 recipe는 같고 random seed와 initialization이 서로 다른 독립 run이다. B 내부 pair는 `same_architecture=True`, `same_lineage=False`인 hard negative이다.

### B2: same topology, different hyperparameters

폭 64의 **PreActResNet18 30개**이다. seed는 0으로 고정하고 maximum learning rate, SAM 설정, augmentation을 달리한 독립 run을 선택했다. B와 B2는 ResNet family를 공유하지만 block ordering이 달라 서로 같은 topology로 취급하지 않는다.

### C: same-lineage descendants

A001부터 A030까지 각 원본에 아래 변형을 하나씩 적용한 총 120개이다. 자식은 부모의 `lineage_id`를 그대로 상속하며 `parent_id`에 직접 부모 A 모델을 기록한다.

| 순서 | 변형 | 설명 |
|---:|---|---|
| 1 | `ft5` | 원본 checkpoint에서 CIFAR-10으로 5 epochs fine-tuning |
| 2 | `prune50` | Conv2d/Linear weight의 global unstructured L1 pruning 50% |
| 3 | `ptq_int8` | calibration 1,024장을 사용한 static post-training INT8 quantization |
| 4 | `prune20` | 같은 방식의 weight pruning 20% |

예를 들어 A002의 자식은 C005–C008이다. `ptq_int8` 파일은 CPU quantized operator를 포함한 TorchScript이며, 나머지는 PyTorch `state_dict`이다.

## 3. 처음 설치하고 사용하는 파이프라인

### 3.1 Clone과 환경 설치

```bash
git clone https://github.com/chlwhdduq2357/Fingerprinting-Model-Zoo.git
cd Fingerprinting-Model-Zoo
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

`requirements-lock.txt`는 검증에 사용한 CPU 환경을 고정한다. CUDA나 ROCm을 쓸 때는 먼저 해당 GPU의 공식 지원표에 맞는 PyTorch를 설치하고 `pip install --no-deps -e .`로 이 프로젝트만 연결한다.

### 3.2 A/B/B2 다운로드

```bash
python scripts/download_models.py --group A --workers 2
python scripts/download_models.py --group B --workers 2
python scripts/download_models.py --group B2 --workers 2

# 한 번에 받거나 일부 ID만 받을 수도 있다.
python scripts/download_models.py --all --workers 2
python scripts/download_models.py --model-id A002 B001 B2_001
```

Downloader는 manifest에서 선택한 checkpoint만 받고, 이미 존재하는 파일은 SHA256이 일치할 때 건너뛴다. 대형 Model Zoo archive 전체를 내려받는 fallback은 사용하지 않으며 파일당 512 MiB, source당 19 GB 제한을 둔다. 공유 metadata를 갱신하므로 downloader 두 개를 동시에 실행하지 않는다.

### 3.3 Hash 검증

```bash
# 다운로드한 원본 90개만 검사한다. CIFAR-10 dataset은 필요 없다.
python scripts/verify_hashes.py --group A B B2

# 특정 모델 또는 현재 로컬에 있는 모든 그룹도 검사할 수 있다.
python scripts/verify_hashes.py --model-id A002 B001 C005
python scripts/verify_hashes.py
```

명령이 `모든 checkpoint의 SHA256이 metadata와 일치합니다.`로 끝나야 한다. 누락 또는 hash 불일치가 있으면 exit code 1을 반환한다.

### 3.4 C 배치와 빠른 검증

C는 공개 원본 다운로드 대상이 아니라 이 프로젝트에서 생성한 derivative이므로 별도 보관본의 `C` 폴더를 `checkpoints/C/`에 배치한다. 현재 `metadata/models.json`에는 C001–C120의 hash와 lineage가 등록되어 있다.

```bash
python scripts/verify_c_quick.py
```

이 검사는 CIFAR-10을 다운로드하거나 정확도를 재계산하지 않는다. C 120개의 ID·부모·변형·파일명, SHA256, strict parameter loading, 한 장의 합성 입력에 대한 finite `[1,10]` logits를 확인한다. 현재 CPU 검증 환경에서는 약 **41초**가 걸렸다. 결과는 `reports/c_quick_verification.json`에 저장된다.

## 4. 모델 선택, 호출과 비교

가장 짧은 실행 예제는 다음 명령으로 확인한다.

```bash
python example.py
```

기본 API는 다음과 같다.

```python
import torch
from model_zoo import load_model, pair_relation

# 입력은 아직 normalize하지 않은 RGB float [N,3,32,32], 범위 [0,1]이다.
images = torch.rand(4, 3, 32, 32)
model = load_model("A002", device="cpu")
with torch.inference_mode():
    logits = model(images)                 # shape: [4,10]

print(pair_relation("A002", "C005"))    # 동일 lineage의 parent/child
print(pair_relation("B001", "B002"))    # 동일 topology의 독립 학습 모델
```

`load_model()`은 ID에 맞는 architecture를 생성하고, SHA256과 strict loading을 확인하고, `.eval()`을 적용한다. 각 source의 native normalization도 wrapper 안에서 자동 적용한다. 따라서 입력을 미리 normalize하지 않는다.

Static INT8 자식은 CPU에서 호출한다.

```python
quantized = load_model("C007", device="cpu")
```

## 5. 추가 설명

### Lineage와 pair label

- A/B/B2의 서로 다른 모델은 모두 서로 다른 `lineage_id`를 가진다.
- C는 A 부모의 `lineage_id`를 상속한다.
- `topology_id`가 같고 lineage가 다르면 `hard_negative=True`이다.
- 부모와 C 자식은 변형 후 topology 표현이 달라질 수 있어도 `same_lineage=True`이다.

세 평가 population은 `different architecture / independent`, `same architecture / independent training`, `same lineage / transformed descendant`로 나뉜다. `python scripts/export_pairs.py`로 현재 metadata에 대한 pair table을 다시 만들 수 있다.

### 전처리와 출력

모든 wrapper 입력은 RGB `float` tensor `[N,3,32,32]`, 값 범위 `[0,1]`이며 출력은 CIFAR-10 표준 class 순서의 raw logits `[N,10]`이다. 기본 `preprocessing="native"`는 모델별 mean/std를 사용한다. 실험용 공통 전처리는 `preprocessing="common"`과 mean/std를 명시하고, 이미 전처리한 입력은 `preprocessing="none"`을 사용한다.

### 검증 수준

`verify_hashes.py`는 파일 무결성만, `verify_c_quick.py`는 C의 구조·로딩·단일 입력까지 검사한다. 실제 CIFAR-10 sample 또는 전체 10,000장 accuracy 검증은 별도 명령이다.

```bash
python scripts/verify_models.py --model-id A002 --samples 256
python scripts/verify_models.py --model-id A002 --full
```

부분집합의 `cifar10_smoke_accuracy`와 전체 test set의 `cifar10_test_accuracy_verified`는 서로 다른 값이다. 확인할 수 없는 seed, hyperparameter와 reported accuracy는 추측하지 않고 `null`로 둔다.

### 출처와 재현성

모델별 repository, revision, direct URL 또는 archive member, SHA256, native preprocessing과 독립 학습 근거는 `metadata/models.json`, `metadata/sources.json`, `metadata/evidence/`에 있다. 외부 repository의 setup script나 remote Python 코드를 실행하지 않고 `vendor/`의 검토된 최소 정의를 사용한다. A/B/B2 초기 구축 결과는 `reports/model_zoo_report.md`, 현재 C 검증 결과는 `reports/c_quick_verification.json`을 참고한다.
