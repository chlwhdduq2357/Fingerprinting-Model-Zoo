# CIFAR-10 forensic attribution Model Zoo

이 프로젝트는 모델 fingerprint가 architecture와 model lineage 중 무엇을 구분하는지 평가하기 위한 A/B/B2 원본 집단입니다. 실제 확보 수와 검증 결과는 **reports/model_zoo_report.md**와 **reports/summary.json**을 확인하세요.

- A: architecture가 다양한 pretrained CIFAR-10 originals 30개, 18 families.
- B: 동일 CIFAR ResNet18 topology를 서로 다른 seed로 독립 학습한 30개 run.
- B2: 동일 CIFAR PreActResNet18 topology와 seed=0에서 lr_max, SAM, augmentation을 달리한 30개 run.
- C는 생성하지 않았습니다. 학습이나 weight 변환도 수행하지 않았습니다.

B와 B2는 모두 ResNet18 family이지만 block ordering이 다릅니다. B는 post-activation BasicBlock이고 B2 공개 배포의 `model=resnet18` 구현은 pre-activation BasicBlock입니다. 따라서 B2 내부 435 pair만 same-architecture hard negative이며, B-B2 pair는 same-family/different-topology입니다.

B는 공개 zoo의 **training_iteration 25** checkpoint입니다. 종료 iteration 50 checkpoint가 아니며, 같은 run의 여러 epoch를 독립 모델로 세지 않습니다. 각 checkpoint에 대응하는 reported accuracy를 사용합니다.

B2는 공개된 epoch-200 원본 container를 그대로 보존합니다. 파일마다 `last`, `best`, `swa_last`, `swa_best`가 있으며 unified loader는 source 평가 코드와 동일하게 `last`를 사용합니다. 개별 파일의 reported accuracy는 공개되지 않아 null로 두고, 30개 smoke test와 층화된 6개 full test 결과를 별도로 기록합니다.

## 설치

이번 CPU 환경은 **Python 3.13.12**로 검증했습니다. 정확한 버전은 `requirements-lock.txt`, `metadata/environment.json`에 기록되어 있습니다. lock 파일을 그대로 사용할 때 Python 3.13을 사용하세요. 소스 API는 Python 3.10 이상을 대상으로 하지만 구버전 Python의 dependency 조합은 별도로 검증해야 합니다. GPU를 사용할 때도 동일 native preprocessing을 사용하세요.

```powershell
git clone https://github.com/chlwhdduq2357/Fingerprinting-Model-Zoo.git
cd Fingerprinting-Model-Zoo
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Linux/macOS에서는 virtualenv의 `bin/python`/`bin/activate` 경로를 사용하세요. `-e .` 설치를 유지하고 repository 전체를 보존하세요. metadata/checkpoints가 필요한 로컬 연구 프로젝트이며 wheel만 배포하는 용도가 아닙니다.

`requirements-lock.txt`는 CPU 재현 환경이며 PyTorch CPU index를 명시합니다. AMD ROCm이나 CUDA GPU를 사용할 때는 이 lock 파일로 PyTorch를 덮어쓰지 말고, GPU와 OS의 공식 지원표에 맞는 별도 Python/PyTorch 환경을 구성하세요. 데스크톱에서 C 단계를 수행할 전체 지시는 `DESKTOP_CODEX_PROMPT.md`에 있습니다.

## Colab에서 C의 빠른 변형부터 생성

[COLAB_C_FAST.ipynb](COLAB_C_FAST.ipynb)은 C 단계 중 계산이 빠른 두 변형을 먼저 생성한다.

- `FT5`: parent checkpoint에서 CIFAR-10으로 5 epochs fine-tuning
- `PRUNE50`: Conv2d/Linear weight의 global unstructured L1 pruning 50%, recovery 학습 없음

고정된 A parent 10개에서 각각 두 모델을 만들어 `C001/C002`, `C005/C006`, …,
`C037/C038`의 총 20개를 생성한다. Adversarial fine-tuning과 quantization 슬롯은 건드리지
않는다. 노트북은 전체 90개 checkpoint 대신 필요한 A parent 10개만 선택 다운로드하고,
각 모델 완료 직후 checkpoint와 metadata를 Google Drive에 저장한다. 세션이 종료되면 같은
셀을 다시 실행하며, hash와 config가 일치하는 완료 결과는 자동으로 건너뛴다.

[Google Colab에서 바로 열기](https://colab.research.google.com/github/chlwhdduq2357/Fingerprinting-Model-Zoo/blob/main/COLAB_C_FAST.ipynb)

로컬이나 다른 CUDA 환경에서는 같은 작업을 다음과 같이 실행할 수 있다.

```bash
python scripts/download_models.py --model-id A001 A002 A003 A006 A008 A010 A013 A015 A027 A029 --workers 2
python scripts/generate_c_fast.py --model-id A002 --device cuda --amp
python scripts/generate_c_fast.py --device cuda --amp
```

생성 결과는 `checkpoints/C/`, epoch log는 `runs/C/`, 실행 요약은
`reports/c_fast_run.json`에 저장된다. `metadata/models.json`과 `models.csv`에도 C row를
등록하므로 완료 후 기존 `load_model("C005")` 인터페이스로 바로 호출할 수 있다.

외부 repository clone, torch.hub 실행, 외부 setup script 실행 없이 inference할 수 있습니다. A의 최소 정의는 `vendor/`에 있고, B/B2 adapter는 각 source의 CIFAR ResNet topology와 일치하도록 프로젝트 내부에 고정했습니다. 원본 정의 snapshot과 라이선스 고지를 보존했습니다.

## 다운로드

checkpoint는 GitHub에 포함하지 않습니다. 아래 명령은 repository root에서 실행하며, 이미 hash가 일치하는 파일은 네트워크 요청 없이 건너뜁니다. 전체 90개 checkpoint의 예상 크기는 약 6.72 GiB입니다.

```bash
python scripts/download_models.py --group A
python scripts/download_models.py --group B
python scripts/download_models.py --group B2
python scripts/download_models.py --group B --workers 3
python scripts/download_models.py --all
python scripts/download_models.py --model-id A002 B001
python scripts/list_models.py
```

manifest의 **선택된 checkpoint만** 받습니다. B의 ZIP은 저장된 central-directory member index, TAR는 byte offset/length를 사용합니다. HTTP Range에 대해 206과 정확한 Content-Range를 반환하지 않으면 즉시 실패하며 전체 archive 다운로드로 전환하지 않습니다. ZIP local header 이름/CRC와 최종 SHA256을 검사합니다. 소스가 파일을 바꾸어 offset이 무효화되어도 SHA256 불일치로 거부합니다.

단일 응답/추출 파일 상한은 512MiB, source별 누적 전송 예산은 19GB입니다. `metadata/transfer_ledger.json`은 실패 요청에 대한 보수적 예약량도 포함할 수 있습니다. 429/일시적 서버 오류는 최대 5회 backoff 후 실패 기록을 남깁니다. 실행 실패 뒤 같은 명령으로 재개할 수 있습니다. `download_error`는 감사 기록으로 유지됩니다.

`download_models.py`와 `verify_models.py`는 manifest를 갱신하므로 **동시에 여러 프로세스로 실행하지 마세요**. 모델 inference는 독립 프로세스에서 병렬 실행할 수 있습니다.

## 모델 불러오기

editable 설치 후 어느 디렉터리에서든:

```python
import torch
from model_zoo import load_model, pair_relation

# images는 RGB float32 [N,3,32,32], 각 값은 [0,1]. 미리 normalize하지 않습니다.
images = torch.rand(8, 3, 32, 32)
model = load_model("A002")       # strict loading, SHA256 검사, native preprocessing, eval
with torch.inference_mode():
    logits = model(images)       # [8,10], raw logits

print(pair_relation("B001", "B002"))
# same_lineage=False, same_architecture=True, same_family=True,
# different_architecture=False, hard_negative=True

print(pair_relation("B2_001", "B2_002"))  # B2 내부: hard_negative=True
print(pair_relation("B001", "B2_001"))    # post-act vs pre-act: same_family=True, same_architecture=False
```

Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

```python
native = load_model("B001", device="cpu", preprocessing="native")
common = load_model("B001", preprocessing="common",
                    common_mean=[0.4914, 0.4822, 0.4465],
                    common_std=[0.2023, 0.1994, 0.2010])
raw_network = load_model("B001", preprocessing="none")
# none: caller가 이미 전처리한 tensor를 그대로 network에 전달.
```

## 빠른 실행 예제

`test.py`는 전체 프로젝트를 처음 보는 사용자를 위한 실행형 예제입니다. CIFAR-10을
다운로드하지 않고 고정된 합성 입력을 사용해 metadata 필터링, A/B/B2 모델 선택,
unified loader 호출, native/common/none 전처리, pair relation과 간단한 logits 비교를
수 분 안에 보여 줍니다. 자세한 한국어 주석이 있어 각 부분을 작은 실험 코드로
복사해 사용할 수 있습니다.

```bash
# 여러 family와 A/B/B2 대표 모델을 모두 호출하는 기본 예제
python test.py

# checkpoint를 열지 않고 model 선택과 pair 관계만 확인
python test.py --list-only

# 원하는 모델만 골라 호출하거나 GPU 사용
python test.py --models A001 B001 B2_001
python test.py --device cuda --models A002 A013

# 이미 무결성을 검증한 파일로 반복 실험할 때만 hash 계산 생략
python test.py --skip-hash --batch-size 4
```

이 repository의 root에서 `python test.py`로 실행할 수 있습니다.

native는 A/B/B2에 등록된 source별 normalization을 적용합니다. wrapper 내부에서는 resize나 추가 augmentation을 하지 않습니다. 입력 float dtype/shape는 검사하며 [0,1] 범위는 caller 계약입니다. `none`에는 범위를 벗어나는 normalized tensor도 입력할 수 있습니다. fingerprint 실험에는 `.eval()`을 유지하세요.

## 검증

```bash
# 기본: 공식 CIFAR-10 test set에서 동일한 256개 index로 smoke test
python scripts/verify_models.py
python scripts/verify_models.py --group B --samples 256
python scripts/verify_models.py --group B2 --samples 256
python scripts/verify_models.py --model-id A002 B001 --full

# 90개 × test images 10,000개: CPU에서는 오래 걸릴 수 있음
python scripts/verify_models.py --full --batch-size 32 --threads 4
# GPU가 구성되어 있다면 --device cuda 사용 가능

python scripts/export_pairs.py
python scripts/audit_manifest.py
python scripts/report.py
python -m unittest discover -s tests -v
```

공식 CIFAR-10 데이터는 기본적으로 프로젝트의 `work/data`에 저장됩니다. 기존 dataset을 쓰려면 `--data-root 경로`를 지정하세요. 데이터 다운로드 및 원본 배포 파일의 integrity 검사는 torchvision CIFAR10 구현을 이용합니다.

각 checkpoint마다 존재/SHA256, weights_only=True, architecture 생성, strict state_dict load, finite parameters/buffers, eval, 실제 CIFAR-10 batch, output dimension 10, finite logits를 검사합니다. 파일 hash와 전체 state_dict/parameter hash로 중복을 찾습니다. 중복은 `duplicate` 상태로 제외하며 자동 원본 삭제는 하지 않습니다.

- `cifar10_test_accuracy_verified`: **전체 10,000개**를 검증했을 때만 채워집니다. 단위는 %입니다.
- `cifar10_smoke_accuracy`: 부분집합 accuracy. full accuracy로 해석하지 않습니다.
- `reports/verification/*_full.json`, `*_smoke.json`: 각 검증 실행 결과.
- full reported/verified 차이가 2 percentage points를 초과하면 warning을 기록합니다.
- smoke test는 구조/전처리의 큰 오류를 찾을 수 있지만 모델 성능의 완전한 재현 증명은 아닙니다.

## 구조와 metadata

```text
model_zoo/
  checkpoints/A/                 # 원본 배포 checkpoint
  checkpoints/B/
  checkpoints/B2/
  metadata/models.json           # source, lineage, preprocessing, hashes, verification
  metadata/models.csv
  metadata/sources.json          # repository, revision, strategy
  metadata/evidence/             # params, logs, source snapshots, archive indexes
  metadata/pairs.csv             # verified population의 모든 unordered pair
  configs/selection.json
  loaders/                      # unified input/output adapter
  vendor/                       # 최소 model definitions 및 라이선스
  scripts/                      # downloader, verifier, listing, report
  tests/
  reports/model_zoo_report.md
```

필수 metadata 필드는 요청된 `model_id`, `group`, `lineage_id`, `parent_id`, `architecture`, `architecture_family`, source URLs, filename/path/hash, framework/version/format, parameter count, input/normalization, reported/verified accuracy, seed/optimizer/lr/weight decay/augmentation/recipe, notes/download date를 포함합니다. 알 수 없는 값은 null이며 CSV에서는 빈 칸입니다. list/dict는 CSV 셀 안의 JSON 문자열입니다.

추가 필드:

- `topology_id`: 구체적 연산 구조를 식별합니다. 같은 family나 같은 이름만으로 topology를 합치지 않습니다.
- `training_run_id`, `independence_evidence`: B/B2 독립 training run의 출처 증거입니다.
- `hyperparameter_signature`: B2의 lr/SAM/augmentation 조합을 비교하는 고유 서명입니다.
- `download_spec`: direct file, ZIP member 또는 TAR offset/length 및 archive 크기입니다.
- `state_dict_sha256`: tensor 이름/shape/dtype/값, BatchNorm buffer를 포함합니다.
- `parameter_sha256`: named_parameters만 비교합니다.
- `num_parameters`: 고정 mask를 포함한 등록 Parameter 전체. `num_trainable_parameters`는 requires_grad=True만 집계합니다. XDenseNet은 두 값이 다릅니다.
- `framework_version`: 이번 검증 runtime. 원래 학습 버전과 혼동하지 않습니다.
- `status`: candidate → downloaded → verified. failed/duplicate는 usable population에서 제외합니다.

파일 hash 차이는 독립 학습의 충분조건이 아닙니다. B는 서로 다른 seed/run provenance, B2는 서로 다른 public training file/run과 hyperparameter signature를 요구합니다. B2는 의도적으로 seed=0을 고정했습니다. A는 다른 topology의 공개 original checkpoint를 사용하며 해당 모든 training run의 seed를 복원한 것은 아닙니다. osmr의 Gluon→PyTorch 포맷 변환본을 별도 lineage로 이중 집계하지 않았습니다.

현재 서로 다른 A/B/B2 원본은 모두 다른 lineage입니다. self-pair의 same_lineage는 True입니다. future C의 경우 parent_id를 직접 부모로, lineage_id를 원본에서 상속합니다. C 등록 시 동일 source/weights provenance를 검사하고 original population과 별도로 관리하세요.

## 모델 추가

1. CIFAR-10으로 실제 학습된 공개 checkpoint와 학습/run 근거를 확보합니다. seed나 accuracy를 추정하지 않습니다.
2. `models.json` row를 추가하고 새 original이면 고유 lineage_id와 parent_id=null을 부여합니다. topology가 같다고 기존 lineage를 재사용하지 않습니다.
3. 정확한 source commit/release, checkpoint URL/hash, native preprocessing 증거와 independent training 근거를 저장합니다.
4. `loaders/unified.py`의 adapter 또는 `vendor/`에 최소 정의를 추가합니다. 라이선스를 보존합니다. 자동 remote-code 실행은 사용하지 않습니다.
5. `download_spec`을 설정하고 특정 `--model-id`로 다운로드/검증합니다. 19GB source 상한을 우회하는 전체 archive 다운로드를 추가하지 않습니다.
6. hash/parameter duplicate를 제거한 뒤 `export_pairs.py`, `report.py`를 재실행합니다. 서로 다른 구현을 같은 topology_id로 합치기 전 실제 연산 구조를 확인합니다.

source snapshot과 정확한 사용 파일 정보는 로컬에 있지만 외부 checkpoint 서버의 영구 가용성을 보장하지는 않습니다. 장기 보존에는 이 디렉터리 전체의 백업이 필요합니다.
