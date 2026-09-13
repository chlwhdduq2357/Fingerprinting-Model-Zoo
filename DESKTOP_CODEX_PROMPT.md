# Desktop Codex handoff prompt: C descendants

아래 프롬프트 전체를 데스크톱의 Codex 새 작업에 붙여 넣는다. Codex에서 이 repository를
프로젝트로 연 상태에서 실행한다.

---

나는 CIFAR-10 image classifier의 model fingerprinting을 이용한 forensic model
attribution 연구를 진행하고 있다. 이 repository에는 이미 다음 원본 population이 있다.

- A: architecture-diverse pretrained originals 30개
- B: 같은 CIFAR ResNet18 topology와 같은 recipe에서 seed만 다른 originals 30개
- B2: 같은 PreActResNet18 topology와 seed=0에서 hyperparameter가 다른 originals 30개
- A/B/B2의 90개 모델은 모두 서로 다른 independent lineage이다.

이번 작업의 목표는 **C 단계 descendant 40개를 실제로 생성하고 검증하는 것**이다.
기존 A/B/B2 파일, ID, hash, provenance와 검증 결과는 보존한다. 합리적인 구현 선택은
스스로 결정하고 작업을 끝까지 진행하되, 아래 실험 정의는 임의로 바꾸지 않는다.

## 1. 먼저 환경과 repository를 감사하라

1. 현재 OS, CPU, RAM, GPU의 정확한 이름과 VRAM, driver를 확인한다.
2. `README.md`, `metadata/models.json`, `metadata/sources.json`,
   `reports/model_zoo_report.md`, `loaders/unified.py`, downloader/verifier/tests를 읽는다.
3. `git status`와 현재 branch를 확인한다. 기존 변경을 덮어쓰거나 삭제하지 않는다.
4. GPU는 AMD Radeon RX 9060 XT로 예상하지만 실제 장치를 명령으로 확인한다.
5. Windows라면 AMD가 해당 GPU에 공식 제공하는 PyTorch/ROCm 조합을 우선 사용한다.
   이 repository의 `requirements-lock.txt`는 CPU 재현 환경이므로 GPU 환경에 그대로
   설치하지 않는다. 별도 virtual environment를 만든다.
6. 가능하면 Python 3.12와 AMD 공식 PyTorch 2.8 + ROCm 6.4.4 Windows wheel을 사용한다.
   Linux라면 RX 9060 XT를 공식 지원하는 ROCm/PyTorch 조합을 사용한다.
7. `torch.cuda.is_available()`, device name, HIP/ROCm version, 간단한 tensor 연산 및
   A002 forward/backward로 GPU가 실제 사용되는지 확인한다. PyTorch ROCm에서도 device
   문자열은 일반적으로 `cuda`를 사용한다.
8. GPU 환경 구성이 실패하면 원인과 실행 명령/output을 기록하고 해결을 시도한다.
   무심코 CPU에서 장시간 학습을 시작하지 않는다.

## 2. 원본 checkpoint를 복원하고 검증하라

GitHub에는 대형 checkpoint가 들어 있지 않다. repository root에서 제공된 downloader를
사용한다. 한 source의 거대 archive 전체를 받지 말고 기존 selective downloader의
상한과 Range 검사를 유지한다.

```bash
python scripts/download_models.py --all
python scripts/audit_manifest.py
python test.py --list-only
python test.py --models A002 B001 B2_001 --device cuda
```

90개 checkpoint의 예상 합계는 약 6.72 GiB다. 다운로드를 재실행할 때 hash가 일치하는
파일은 건너뛰어야 한다. 다운로드 후 최소한 A/B/B2 전체의 파일 존재와 SHA256을 검사하고,
대표 모델을 GPU에서 smoke inference한다.

## 3. C parent를 정확히 다음 10개로 고정하라

| Parent | Architecture |
|---|---|
| A001 | NIN |
| A002 | ResNet20 |
| A003 | ResNet56 |
| A006 | PreResNet56 |
| A008 | SE-ResNet56 |
| A010 | PyramidNet110-a48 |
| A013 | DenseNet100-k12-BC |
| A015 | WRN16-10 |
| A027 | MobileNetV2-x1.0 |
| A029 | ShuffleNetV2-x1.0 |

먼저 A002에서 전체 pipeline을 pilot으로 실행하고, 저장·재로딩·lineage·정확도 검증이
통과한 뒤 나머지 9개 parent로 확장한다.

## 4. parent마다 다음 네 descendant를 생성하라

각 parent당 정확히 4개, 총 40개를 만든다. ID는 parent 순서대로 네 개씩 배정한다.

네 변형의 단일 근거 논문은 Peng et al., CVPR 2022, *Fingerprinting Deep Neural Networks
Globally via Universal Adversarial Perturbations*이다. 이 논문은 stolen/piracy model에 대한
post-modification robustness 실험에서 fine-tuning, weight pruning, FP32→INT8 quantization,
adversarial training을 모두 평가한다. 논문은 pruning rate 0.2–0.6, FP32→INT8 변환,
그리고 DeepFool로 매 iteration 128개 adversarial example을 생성하는 최대 270회의
adversarial-training iteration을 명시한다.
<https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Fingerprinting_Deep_Neural_Networks_Globally_via_Universal_Adversarial_Perturbations_CVPR_2022_paper.pdf>

따라서 이번 C는 다음 순서의 네 축으로 고정한다.

1. **Fine-tuning**
2. **Weight pruning**
3. **Adversarial fine-tuning**
4. **Quantization**

네 변형 모두 parent checkpoint 자체에서 출발하여 그 weight 또는 parameter 표현을 직접
변경한다. 별도로 초기화한 student를 학습하는 knowledge distillation과 model extraction은
포함하지 않는다. Weight noise도 별도 checkpoint 없이 fingerprint 평가 시점에 적용할 수
있으므로 포함하지 않는다.

- `C001`–`C004`: A001
- `C005`–`C008`: A002
- `C009`–`C012`: A003
- `C013`–`C016`: A006
- `C017`–`C020`: A008
- `C021`–`C024`: A010
- `C025`–`C028`: A013
- `C029`–`C032`: A015
- `C033`–`C036`: A027
- `C037`–`C040`: A029

각 묶음 안의 순서는 `FT5`, `PRUNE50`, `ADV_FT270`, `PTQ_INT8`이다. 예를 들어
`C001`–`C004`는 각각 A001의 네 변형이고, `C005`–`C008`은 각각 A002의 네 변형이다.

### a. FT5

- CIFAR-10 train 50,000장
- parent의 native preprocessing 유지
- augmentation: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`
- optimizer: SGD, momentum 0.9, weight decay 5e-4
- initial learning rate: 1e-3
- scheduler: cosine decay over exactly 5 epochs
- batch size: 기본 128. VRAM 부족 시에만 줄이고 metadata에 실제 값을 기록
- seed: `31000 + parent A 번호` (예: A002는 31002)
- AMP mixed precision 사용 가능. 실제 사용 여부와 dtype을 metadata에 기록
- parent weights에서 시작하고 classifier를 포함한 모든 trainable parameter를 update
- epoch별 loss, train accuracy, validation/test accuracy, elapsed time, device를 JSONL로 기록
- 매 epoch atomic checkpoint를 저장하여 중단 후 재개 가능하게 구현
- 최종 descendant는 epoch 5 weights

Test set은 optimizer/scheduler 결정에 사용하지 않는다. 별도 validation split을 만들지
않는 단순 5-epoch lineage 변환이므로 epoch 중 test accuracy는 관찰용으로만 기록하고,
best checkpoint 선택에는 사용하지 않는다.

### b. PRUNE50

- `Conv2d`와 `Linear`의 weight 전체를 대상으로 global unstructured L1 magnitude pruning
- 전체 대상 weight element의 정확히 50%를 zero로 만든다.
- bias, BatchNorm affine parameter, running statistics는 pruning하지 않는다.
- Peng et al.이 평가한 pruning rate 0.2–0.6의 중앙 조건으로 0.5를 고정한다.
- **pruning 후 recovery fine-tuning을 수행하지 않는다.** Fine-tuning과 weight pruning을
  독립된 변형 축으로 유지하기 위한 조건이다.
- seed가 개입하는 학습 변형은 아니지만 실행 재현성을 위해 seed를
  `33000 + parent A 번호`로 기록한다.
- 완료 후 PyTorch pruning reparameterization을 제거하여 일반 state_dict로 저장한다.
- pruning 직후 accuracy, 전체/layer별 sparsity와 zero count를 기록한다.
- topology와 tensor shape는 parent와 동일하게 유지한다.

### c. ADV_FT270

- 논문에서 `Adversarial Training`이라고 부르는 조건을, pretrained parent에서 시작한다는
  점을 명확히 하기 위해 이 프로젝트에서는 `Adversarial fine-tuning`으로 기록한다.
- parent weights에서 시작하고 classifier를 포함한 모든 trainable parameter를 update한다.
- 최대 **270 adversarial fine-tuning iterations**를 수행한다.
- 각 iteration마다 CIFAR-10 train sample 128개에서 현재 descendant를 대상으로 DeepFool
  adversarial example을 생성하고, 생성한 adversarial batch의 정답 label로 cross-entropy
  update를 한 번 수행한다. 논문의 “128 adversarial examples as new datapoints per iteration”
  조건을 따른다.
- parent의 native preprocessing을 정확히 유지한다. attack budget, DeepFool overshoot,
  최대 attack step 등 논문에 명시되지 않은 구현값은 사용한 library의 고정 default를
  machine-readable config와 metadata에 기록하고 모든 parent에 동일하게 적용한다.
- optimizer: SGD, momentum 0.9, weight decay 5e-4, initial learning rate 1e-3
- scheduler: 270 optimizer steps에 대한 cosine decay
- seed: `35000 + parent A 번호`
- adversarial example 생성 성공률, perturbation L2/Linf 통계, clean accuracy,
  adversarial-batch accuracy, loss, elapsed time을 주기적으로 JSONL에 기록한다.
- 270회가 짧아 epoch checkpoint보다 iteration 30회마다 atomic checkpoint를 저장하고,
  중단 시 마지막 완료 iteration에서 재개할 수 있게 한다.
- 최종 descendant는 iteration 270 weights다.

### d. PTQ_INT8

- fingerprinting 논문에서 사용하는 FP32→INT8 compression 조건을 재현하는
  **post-training static quantization**이다. gradient 학습과 QAT는 수행하지 않는다.
- weight와 activation을 모두 INT8 대상으로 하며 입력과 최종 logits interface는 float를
  유지한다. 단순 weight rounding만 적용한 모델을 PTQ_INT8이라고 부르지 않는다.
- CIFAR-10 train set에서 seed 32000을 사용해 고정한 1,024개 sample로 calibration한다.
  test set은 calibration에 사용하지 않는다.
- Conv/Linear weight는 가능한 경우 symmetric per-channel quantization을, activation은
  calibrated affine per-tensor quantization을 사용한다. backend, observer, qscheme,
  dtype, scale/zero-point와 calibration-index hash를 기록한다.
- quantized operator는 AMD GPU가 아니라 지원되는 CPU quantization backend에서 검증해도
  된다. native preprocessing과 `[N,10]` float logits 계약은 유지한다.
- custom topology 때문에 표준 PyTorch PTQ graph conversion이 실패하면 quantize/dequantize
  boundary를 명시적으로 삽입하는 adapter까지 시도한다. 그래도 실제 weight+activation
  INT8 경로를 구성할 수 없으면 weight-only 결과로 조용히 대체하지 말고 해당 후보를
  `failed`로 기록한다. 모델 수를 맞추기 위해 의미가 다른 변형을 같은 이름으로 넣지 않는다.
- FP32 parent 대비 serialized size, quantized layer 수, latency, accuracy delta를 기록한다.

## 5. lineage와 metadata 규칙

각 C row는 최소한 다음을 만족해야 한다.

- `group = "C"`
- `parent_id = 해당 A model_id`
- `lineage_id = parent A의 lineage_id` 그대로 상속
- `architecture`, `architecture_family`, `topology_id`, input/native normalization과
  class order는 parent에서 상속
- `transform_type`: `ft5`, `prune50`, `adv_ft270`, `ptq_int8` 중 하나
- `lineage_mechanism`: 네 변형 모두 `parameter_inheritance`
- 정확한 transform config, seed, device, software versions, 시작 parent SHA256
- local checkpoint path, file SHA256, canonical state_dict SHA256, parameter SHA256
- parameter 수, nonzero 수/sparsity, 생성 시각, 검증 상태
- `source_repository/source_url`은 parent provenance를 보존하면서 local derivative임을 명시
- 알 수 없는 값은 추측하지 말고 null

C는 새로운 original이 아니다. C끼리 parent가 같으면 `same_lineage=True`이며, parent A와
그 C도 `same_lineage=True`다. 다른 parent의 C는 같은 architecture라도
`same_lineage=False`다. `pair_relation()`과 pair export가 이 규칙을 정확히 반영하도록
확장한다.

## 6. checkpoint 및 loader

- `checkpoints/C/` 아래에 저장한다.
- A/B/B2 checkpoint를 overwrite하지 않는다.
- C checkpoint는 가능한 범위에서 순수 tensor state_dict로 저장하고 pickle object 전체를
  저장하지 않는다. PTQ packed parameter처럼 별도 표현이 필요하면 포맷과 안전한 loading
  방법을 문서화한다.
- load 시 `weights_only=True`, strict state_dict loading과 SHA256 검사를 유지한다.
- 기존 `load_model(model_id)` 인터페이스로 C001–C040도 동일하게 호출되어야 한다.
- 모든 모델 입력은 float `[N,3,32,32]`이며 native wrapper가 해당 normalization을 적용한다.
- FT/ADV_FT 학습 및 adversarial example 생성 시 wrapper 내부 normalization이 중복 적용되지
  않도록 data pipeline과 gradient 경로를 감사한다.

## 7. 검증

모든 C 모델에 대해 다음을 수행한다.

1. 파일 존재 및 SHA256
2. architecture 생성과 strict checkpoint load
3. `.eval()` inference
4. output `[N,10]`
5. parameter/buffer와 logits의 NaN/Inf 검사
6. 동일한 고정 256-sample CIFAR-10 smoke test
7. parameter duplicate 및 exact duplicate 검사
8. parent와 다른 state_dict인지 확인
9. parent/descendant 및 descendant/descendant pair relation assertion

40개 모두 smoke test가 통과한 후 전체 CIFAR-10 10,000장 accuracy를 실행한다. 시간이
길면 FT5 10개를 먼저 full verify하고 나머지는 transformation별로 순차 진행하되,
최종 작업을 끝내기 전 40개 모두 full verification을 완료한다. Reported accuracy와
verified accuracy를 구분하고, parent 대비 accuracy delta를 기록한다.

다음과 같은 자동화 테스트를 추가한다.

- 변환의 결정성
- PRUNE50 실제 sparsity 허용 오차와 pruning reparameterization 제거
- ADV_FT270 parent-weight 상속, adversarial example 생성, iteration/seed 결정성
- PTQ_INT8 calibration 결정성, quantized weight/activation 경로와 float I/O 계약
- lineage 상속 및 hard-negative 관계
- checkpoint round trip과 unified loader
- A/B/B2 metadata가 변하지 않았다는 regression 검사

## 8. 실행 안전성과 재현성

- 먼저 예상 disk usage를 계산한다. 사용 가능한 공간을 확인한다.
- 한 번에 한 모델만 GPU에 올려 peak memory를 제한한다.
- CUDA/ROCm OOM이면 batch size를 절반으로 줄이고 실제 값을 기록한 뒤 재개한다.
- 체크포인트는 임시 파일에 저장 후 atomic rename한다.
- 이미 hash와 config가 일치하는 결과는 건너뛰어 idempotent하게 만든다.
- SIGINT/오류 후 마지막 완료 epoch 또는 iteration에서 재개할 수 있게 한다.
- 외부 repository의 임의 setup script나 remote code를 실행하지 않는다.
- 대형 source archive 전체를 자동 다운로드하지 않는다.
- C checkpoint를 일반 Git history에 추가하지 않는다. `.gitignore`를 유지한다.
- 장시간 작업 전에 A002 pilot의 실제 images/sec를 측정하여 전체 ETA를 출력한다.

## 9. 문서와 최종 산출물

다음을 구현하거나 갱신한다.

- reproducible C generation/training CLI
- configs/C의 machine-readable transform configuration
- unified C loader
- metadata JSON/CSV
- pair relation CSV
- C verification 결과
- duplicate audit
- README 사용법
- `reports/c_stage_report.md`

보고서에는 다음을 포함한다.

- 성공한 C 수와 transformation별 수
- parent 10개와 descendant 매핑
- 실제 GPU/driver/ROCm/PyTorch/Python 환경
- 모델별 학습/변환 시간과 전체 wall time
- parent/descendant accuracy 및 delta
- pruning sparsity, adversarial fine-tuning 공격·perturbation 통계,
  PTQ backend/size/latency
- 실패와 재시도, OOM 및 batch-size 변경
- 총 checkpoint 용량
- lineage/pair population 통계

모든 코드와 metadata/report 변경은 테스트 후 commit한다. 대형 checkpoint, dataset, virtual
environment는 commit하지 않는다. 원격 push 권한이 이미 구성되어 있으면 source와 작은
metadata/report만 현재 GitHub repository에 push한다. 인증이나 OS 재부팅처럼 사용자
개입이 실제로 필요한 지점이 오면, 완료한 작업과 정확한 다음 한 단계만 명확히 요청한다.
그 외에는 불필요한 질문으로 작업을 중단하지 말고 목표를 끝까지 수행한다.

최종 응답에는 실제 device 사용 증거, C 40개 성공 수, 총시간, 모델별 accuracy delta,
모든 검증 결과, 실패/제약, commit hash를 요약하라.

---
