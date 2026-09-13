import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from model_zoo.core import ROOT, models, read_json, write_json
from model_zoo import pair_relation

def fmt(value):
    return 'unknown' if value is None else f'{value:.2f}' if isinstance(value,float) else str(value)

def main():
    rows=models();ok=[r for r in rows if r['status']=='verified']
    A=[r for r in ok if r['group']=='A'];B=[r for r in ok if r['group']=='B'];B2=[r for r in ok if r['group']=='B2']
    checkpoint_bytes=sum((ROOT/r['local_checkpoint_path']).stat().st_size for r in rows if (ROOT/r['local_checkpoint_path']).exists())
    zoo_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    work_path=ROOT.parent.parent/'work'
    work_bytes=sum(p.stat().st_size for p in work_path.rglob('*') if p.is_file()) if work_path.exists() else 0
    families=Counter(r['architecture_family'] for r in ok)
    source_bytes=defaultdict(int)
    for r in rows:
        path=ROOT/r['local_checkpoint_path']
        if path.exists():source_bytes[r['source_id']]+=path.stat().st_size
    hard=sum(pair_relation(a,b)['hard_negative'] for a,b in combinations(ok,2))
    full=[r for r in ok if r['cifar10_test_accuracy_verified'] is not None]
    summary=dict(A=len(A),B=len(B),B2=len(B2),total=len(ok),checkpoint_bytes=checkpoint_bytes,zoo_directory_bytes=zoo_bytes,
                 auxiliary_work_bytes=work_bytes,zoo_plus_work_bytes=zoo_bytes+work_bytes,
                 checkpoint_GiB=checkpoint_bytes/1024**3,architecture_families=len(families),
                 family_distribution=dict(families),hard_negative_pairs=hard,
                 full_accuracy_verified=len(full),source_checkpoint_bytes=dict(source_bytes))
    write_json(ROOT/'reports/summary.json',summary)
    lines=['# CIFAR-10 Forensic Model Attribution Zoo — A/B/B2 구축 보고서','',
           '## Summary','',f'- A: **{len(A)}**, B: **{len(B)}**, B2: **{len(B2)}**, 합계: **{len(ok)}** verified checkpoints.',
           f'- Checkpoint 용량: **{checkpoint_bytes:,} bytes ({checkpoint_bytes/1024**3:.3f} GiB)**.',
           f'- Zoo 디렉터리 용량(생성 시점): {zoo_bytes:,} bytes. Python 환경과 CIFAR-10 데이터는 별도 work/에 위치한다.',
           f'- work/(환경, 데이터, 조사 자료): {work_bytes:,} bytes. Zoo + work: {(zoo_bytes+work_bytes)/1024**3:.3f} GiB. 파일 논리 크기 합계이며 외부 pip cache는 제외한다.',
           f'- Architecture family: **{len(families)}**. 독립 lineage: **{len({r["lineage_id"] for r in ok})}**.',
           f'- Same-topology independent hard-negative pairs: **{hard}**.',
           '- 신규 학습, fine-tuning, pruning, quantization, noise, C derivative 생성 없음.',
           '- 모든 usable 모델은 SHA256, strict state_dict load, eval, CIFAR-10 실제 sample, [N,10] logits, finite tensor 검사를 통과해야 집계됨.','',
           '## A table','', '| ID | Architecture | Family | Source | Reported Acc % | Full Verified Acc % | Params |',
           '|---|---|---|---|---:|---:|---:|']
    for r in A:lines.append(f'| {r["model_id"]} | {r["architecture"]} | {r["architecture_family"]} | {r["source_id"]} | {fmt(r["cifar10_test_accuracy_reported"])} | {fmt(r["cifar10_test_accuracy_verified"])} | {r["num_parameters"]:,} |')
    lines+=['','Params는 등록된 Parameter 전체이다. A014 XDenseNet에는 학습하지 않는 고정 connectivity mask도 Parameter로 등록되어 있어 total=1,319,338, trainable=690,346이다. 이는 이번에 pruning을 적용한 모델이 아니라 공개 XDenseNet 원래 정의이다.','', '## B table','', '| ID | Architecture | Training difference | Source | Reported Acc % | Full Verified Acc % |', '|---|---|---|---|---:|---:|']
    for r in B:lines.append(f'| {r["model_id"]} | {r["architecture"]} | independent seed={r["training_seed"]}, iteration={r["checkpoint_iteration"]} | ModelZoos | {fmt(r["cifar10_test_accuracy_reported"])} | {fmt(r["cifar10_test_accuracy_verified"])} |')
    lines+=['','## B2 table','', '| ID | Architecture | Hyperparameter difference | Source | Reported Acc % | Full Verified Acc % |', '|---|---|---|---|---:|---:|']
    for r in B2:
        recipe=r['training_recipe']
        lines.append(f'| {r["model_id"]} | {r["architecture"]} | lr_max={recipe["max_learning_rate"]:g}, SAM={recipe["sam_rho"]:g}, aug={recipe["standard_augmentation"]} | ICML 2023 sharpness zoo | {fmt(r["cifar10_test_accuracy_reported"])} | {fmt(r["cifar10_test_accuracy_verified"])} |')
    lines+=['','## Family distribution','', '| Family | A | B | B2 | Total |','|---|---:|---:|---:|---:|']
    for fam,n in sorted(families.items()):lines.append(f'| {fam} | {sum(r["architecture_family"]==fam for r in A)} | {sum(r["architecture_family"]==fam for r in B)} | {sum(r["architecture_family"]==fam for r in B2)} | {n} |')
    lines+=['','## B hard-negative structure','',
      f'ResNet18-CIFAR-3x3-no-maxpool: {len(B)} independent runs. B 내부 pair는 {len(B)*(len(B)-1)//2}개이다. 3×3 stride-1 stem, maxpool 없음, BasicBlock [2,2,2,2], classifier 10을 공유한다.',
      '', '샘플 ZIP의 25개 run과 원본 tar의 추가 5개 run을 선택했다. 모두 서로 다른 seed이며, 동일 seed의 다른 epoch는 집계하지 않는다. 공개 generator는 seed grid 1..1000, kaiming_uniform initialization, resume=False, reuse_actors=False이다. 각 params.json과 result.jsonl을 보존했다.',
      '', '**B는 iteration 25의 중간 학습 checkpoint**이다. 종료 iteration 50의 accuracy를 가져오지 않았다. 이 구성은 충분히 학습된 독립 run을 제공하지만 최종 수렴 모델 집단과 같지는 않다. A와 B 간 성능/학습 기간 차이는 추후 분석에서 통제해야 한다.',
      '', 'B recipe: SGD, lr=0.05, momentum=0.9, weight_decay=0.0005, OneCycleLR, batch=256. 원본 train 50,000개 중 40,000 학습/10,000 validation(split seed 42), 공식 test 10,000개. 독립성 근거는 seed/run provenance이며, hash 차이만으로 독립 학습을 증명한다고 간주하지 않는다.',
      '', '## B2 hard-negative structure','',
      f'PreActResNet18-CIFAR-width64: {len(B2)} independent hyperparameter runs, 내부 hard-negative pair {len(B2)*(len(B2)-1)//2}개. lr_max 30개 값, SAM rho 0/0.05/0.1, augmentation off/on을 균형 있게 선택했다. 모든 파일은 seed=0이므로 initialization 변화가 아니라 training hyperparameter 변화만 통제한다.',
      '', 'B2는 공개 배포에서 `model=resnet18`로 명명되지만 실제 source implementation은 pre-activation block이다. 따라서 topology_id를 B의 post-activation ResNet18과 다르게 기록했다. B-B2 cross pair는 same_family=True, same_architecture=False이며 동일 topology 실험으로 해석하면 안 된다.',
      '', 'B2 recipe: SGD, 200 epochs, batch=128, width=64, cyclic schedule, weight_decay=0.0, label noise=0. 각 원본 container의 `last` state_dict를 로드하며 best/swa_last/swa_best state는 원본 파일 안에 보존한다.',
      '', '## Verification','',f'전체 10,000 test images accuracy 재계산: {len(full)}/{len(ok)} 모델. 그 외 full verified accuracy는 null이다. Smoke accuracy를 full test accuracy 칸에 넣지 않는다.',
      '', '| ID | Smoke Acc % | Smoke N | Full Acc % | Warning |','|---|---:|---:|---:|---|']
    for r in ok:
        sp=ROOT/'reports/verification'/(r['model_id']+'_smoke.json');sm=read_json(sp) if sp.exists() else {}
        lines.append(f'| {r["model_id"]} | {fmt(r.get("cifar10_smoke_accuracy"))} | {sm.get("samples","unknown")} | {fmt(r["cifar10_test_accuracy_verified"])} | {r.get("accuracy_warning") or ""} |')
    lines+=['','소규모 sample은 고정 seed 20260907로 test index를 선택한다. 모델 선택에 accuracy를 사용하지 않았다. 256개 smoke accuracy는 통계적 불확실성이 커서 보고된 full accuracy와 직접 동일시하면 안 된다.',
      '', 'Native/common/none preprocessing, lineage/topology 구분, B2 network contract, hash의 순서 독립성 및 buffer 민감도, Range 무시/잘못된 offset 차단, source 용량 예산 차단의 7개 contract test를 통과했다. 별도 manifest audit는 90개 checkpoint hash, 30개 B seed/run 증거, 30개 B2 hyperparameter signature, JSON/CSV ID 일치, 4,005개 pair의 관계를 검사한다.',
      '', '## Download failures and source strategies','',
      '1. osmr/imgclsmob: 30개 중 A 23개를 개별 release ZIP으로 확보. 예전 pytorch/README.md 경로는 404였으며 osmr/pytorchcv README로 이동한 사실을 확인했다.',
      '2. chenyaofo/pytorch-cifar-models: A 7개 개별 state_dict 확보. 각 training log의 native mean/std와 pretrained=False 기록을 보존했다.',
      '3. ModelZoos selective download: ZIP central-directory 조회 후 선택 member byte range만 수신. API 다중 요청 중 HTTP 429 발생; 이미 확보한 metadata를 재사용하고 member별 단일 range 및 순차 요청/backoff로 수정했다.',
      '4. PhaseTransitionModelZoo: README에서 Proton Drive ZIP 샘플 배포와 192개 width/batch-size grid 확인. 동일 family라도 width가 다르면 topology가 다르므로 그대로 same-architecture로 묶을 수 없다. ModelZoos에서 목표가 확보되어 해당 archive는 다운로드하지 않았다.',
      '5. ICML 2023 sharpness zoo: 50개 CIFAR-10 `model=resnet18` 파일 중 30개를 개별 Google Drive URL로 확보. 폴더 전체 archive는 받지 않았다. Source code 확인 결과 실제 구현은 PreActResNet18이므로 별도 topology_id로 등록했다.',
      '6. ICLR 2026 folding zoo: optimizer/lr/weight-decay/L1/RandAugment/SAM/scheduler를 바꾼 792개 post-activation ResNet18을 확인했으나 checkpoint 다운로드가 공개되지 않아 채택하지 않았다.',
      '7. FLStore CIFAR10_resnet18.zip: 2.2GB로 용량 제한 이내지만 50 federated rounds의 client updates라 독립 original lineage 조건을 충족하지 않아 채택하지 않았다.',
      '', '현재 후보별 오류 기록:', '']
    errors=[r for r in rows if r.get('download_error') or r.get('verification_error')]
    lines += [f'- {r["model_id"]}: {r.get("download_error") or r.get("verification_error")}; current status={r["status"]}' for r in errors] or ['- 최종 다운로드/로드 실패 모델 없음.']
    lines+=['','## Duplicate check','', '파일 SHA256, state_dict SHA256(이름/shape/dtype/값/BN buffer), parameter SHA256를 각각 검사한다. 결과:','', '```json',json.dumps(read_json(ROOT/'reports/duplicate_check.json'),ensure_ascii=False,indent=2),'```',
      '', '## Unknown metadata','',
      '- A osmr 모델의 개별 seed, optimizer, learning rate, weight decay는 해당 checkpoint별 확실한 증거를 확보하지 못한 경우 null이다. 현재 소스의 기본 recipe를 과거 모든 모델의 실제 recipe로 추정하지 않았다.',
      '- chenyaofo 모델은 각 저장 log의 optimizer/lr/weight decay/normalization을 기록했다. 확인되지 않은 seed와 augmentation은 null이다.',
      '- B2의 source filename은 seed, lr_max, weight decay, SAM, augmentation, epoch, batch size를 기록한다. SGD는 논문의 고정 recipe로 확인했지만 momentum은 확인하지 못해 training_recipe.momentum=null이다. 개별 reported accuracy는 배포 폴더에 없어 null이다.',
      '- framework_version은 이번 검증 runtime이다. 원래 학습 framework 버전은 training_framework_version이 있는 경우에만 확인된 값이다.',
      '- A 모델은 서로 다른 공개 topology checkpoint라는 근거로 독립 originals로 등록했다. 모든 원본 학습의 데이터/난수 이력까지 복원한 것은 아니다.',
      '', '## Native preprocessing and relations','',
      '입력은 float RGB [N,3,32,32], [0,1]. A는 source mean/std, B는 mean=[125.3,123.0,113.9]/255 및 std=[63.0,62.1,66.7]/255, B2는 mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010]을 적용한다. 출력은 CIFAR-10 표준 class 순서의 raw logits. `common` 모드는 mean/std를 명시해야 하며 `none`은 이미 전처리된 입력을 위한 선택이다.',
      '', '`topology_id`는 구체 구현의 구조를 구분한다. 이름에 ResNet이 들어간다고 same_architecture로 판정하지 않는다. distinct A/B/B2 모델 pair는 모두 same_lineage=False이며 self-pair는 True이다. 향후 C는 원본 lineage_id를 상속하고 parent_id를 직접 부모 model_id로 기록해야 한다.',
      '', '## Disk usage','', '| Source | Checkpoint bytes | GiB |','|---|---:|---:|']
    for source,n in source_bytes.items():lines.append(f'| {source} | {n:,} | {n/1024**3:.3f} |')
    lines+=['','| Model | Bytes | Local checkpoint |','|---|---:|---|']
    for r in rows:lines.append(f'| {r["model_id"]} | {r.get("checkpoint_bytes",0):,} | {r["local_checkpoint_path"]} |')
    lines+=['','원본 tar 크기는 2,328,714,690,560 bytes이지만 전체 파일을 저장/다운로드하지 않았다. tar 헤더 탐색은 약 4.8MB, 이후 필요한 checkpoint 5개 및 params/log만 Range로 받았다. ModelZoos ZIP도 전체 5.1GB를 받지 않았다. B2는 30개 direct file만 받았다. `metadata/transfer_ledger.json`은 downloader의 source별 누적 보수적 전송 예산이며 초기 조사 traffic은 별도다.',
      '', '## C parent recommendations (아직 생성하지 않음)','',
      '| ID | Architecture | 선정 이유 |','|---|---|---|',
      '| A001 | NIN | residual 없는 convolution baseline |',
      '| A002 | ResNet20 | 작고 빠른 residual baseline |',
      '| A003 | ResNet56 | 깊이에 따른 변환 영향 비교 |',
      '| A006 | PreResNet56 | preactivation 비교 |',
      '| A008 | SE-ResNet56 | channel attention 비교 |',
      '| A010 | PyramidNet110-a48 | 점진적 channel 증가 |',
      '| A013 | DenseNet100-k12-BC | dense connectivity 비교 |',
      '| A015 | WRN16-10 | width 중심 residual baseline |',
      '| A027 | MobileNetV2-x1.0 | depthwise/inverted residual |',
      '| A029 | ShuffleNetV2-x1.0 | channel split/shuffle 비교 |',
      '', '선정 기준은 구조 대표성, 저장량, 구현 접근성이다. RepVGG는 topology를 보존할 실험과 deployment reparameterization을 구분해야 하므로 첫 10개에서 제외했다.',
      '', ('추천 A 부모 10개는 모두 공식 test set 10,000장 전체 accuracy 검증을 마쳤다.' if all(next(r for r in rows if r['model_id']==mid)['cifar10_test_accuracy_verified'] is not None for mid in ['A001','A002','A003','A006','A008','A010','A013','A015','A027','A029']) else '추천 부모 중 full accuracy가 null인 모델은 C 생성 전에 전체 검증을 수행해야 한다.'),
      '', '## Sources','',
      '- [osmr/imgclsmob](https://github.com/osmr/imgclsmob)', '- [osmr/pytorchcv](https://github.com/osmr/pytorchcv)',
      '- [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)', '- [ModelZoos/ModelZooDataset](https://github.com/ModelZoos/ModelZooDataset)',
      '- [Model Zoo samples — Zenodo](https://zenodo.org/records/13144018)', '- [PhaseTransitionModelZoo](https://github.com/ModelZoos/PhaseTransitionModelZoo)',
      '- [ICML 2023 sharpness-vs-generalization](https://github.com/tml-epfl/sharpness-vs-generalization)', '- [CIFAR-10 model files](https://drive.google.com/drive/folders/1dwfb2Iqw6BTMi57SeG44aDebuu-TBzo5)',
      '- [ICLR 2026 Cut Less, Fold More](https://arxiv.org/abs/2602.18116)',
      '', '정확한 commit/release, direct URL, archive member/byte offset, hash, source snapshot은 metadata/models.json, sources.json, evidence에 저장되어 있다.']
    (ROOT/'reports/model_zoo_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
