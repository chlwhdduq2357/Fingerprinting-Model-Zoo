# CIFAR-10 Forensic Model Attribution Zoo — A/B/B2 구축 보고서

## Summary

- A: **30**, B: **30**, B2: **30**, 합계: **90** verified checkpoints.
- Checkpoint 용량: **7,217,273,212 bytes (6.722 GiB)**.
- Zoo 디렉터리 용량(생성 시점): 7,232,173,859 bytes. Python 환경과 CIFAR-10 데이터는 별도 work/에 위치한다.
- work/(환경, 데이터, 조사 자료): 1,124,145,701 bytes. Zoo + work: 7.782 GiB. 파일 논리 크기 합계이며 외부 pip cache는 제외한다.
- Architecture family: **18**. 독립 lineage: **90**.
- Same-topology independent hard-negative pairs: **870**.
- 신규 학습, fine-tuning, pruning, quantization, noise, C derivative 생성 없음.
- 모든 usable 모델은 SHA256, strict state_dict load, eval, CIFAR-10 실제 sample, [N,10] logits, finite tensor 검사를 통과해야 집계됨.

## A table

| ID | Architecture | Family | Source | Reported Acc % | Full Verified Acc % | Params |
|---|---|---|---|---:|---:|---:|
| A001 | nin | NIN | osmr | 92.57 | 92.57 | 966,986 |
| A002 | resnet20 | ResNet | osmr | 94.03 | 94.03 | 272,474 |
| A003 | resnet56 | ResNet | osmr | 95.48 | 95.48 | 855,770 |
| A004 | resnet110 | ResNet | osmr | 96.31 | unknown | 1,730,714 |
| A005 | preresnet20 | PreResNet | osmr | 93.49 | unknown | 272,282 |
| A006 | preresnet56 | PreResNet | osmr | 95.51 | 95.51 | 855,578 |
| A007 | seresnet20 | SE-ResNet | osmr | 93.99 | unknown | 274,847 |
| A008 | seresnet56 | SE-ResNet | osmr | 95.87 | 95.87 | 862,889 |
| A009 | sepreresnet56 | SE-PreResNet | osmr | 95.49 | unknown | 862,601 |
| A010 | pyramidnet110_a48 | PyramidNet | osmr | 96.28 | 96.28 | 1,772,706 |
| A011 | pyramidnet110_a84 | PyramidNet | osmr | 97.02 | unknown | 3,904,446 |
| A012 | densenet40_k12 | DenseNet | osmr | 94.39 | unknown | 599,050 |
| A013 | densenet100_k12_bc | DenseNet | osmr | 95.84 | 95.84 | 769,162 |
| A014 | xdensenet40_2_k24_bc | XDenseNet | osmr | 94.69 | unknown | 1,319,338 |
| A015 | wrn16_10 | WideResNet | osmr | 97.07 | 97.07 | 17,116,634 |
| A016 | wrn28_10 | WideResNet | osmr | 97.61 | unknown | 36,479,194 |
| A017 | ror3_56 | RoR | osmr | 94.57 | unknown | 762,746 |
| A018 | ror3_110 | RoR | osmr | 95.65 | unknown | 1,637,690 |
| A019 | rir | RiR | osmr | 96.72 | unknown | 9,492,980 |
| A020 | shakeshakeresnet20_2x16d | Shake-Shake | osmr | 94.85 | unknown | 541,082 |
| A021 | shakeshakeresnet26_2x32d | Shake-Shake | osmr | 96.83 | unknown | 2,923,162 |
| A022 | diaresnet56 | DIA-ResNet | osmr | 94.95 | unknown | 870,162 |
| A023 | diapreresnet56 | DIA-PreResNet | osmr | 95.17 | unknown | 869,970 |
| A024 | vgg11_bn | VGG | chenyaofo | 92.79 | unknown | 9,756,426 |
| A025 | vgg16_bn | VGG | chenyaofo | 94.16 | unknown | 15,253,578 |
| A026 | mobilenetv2_x0_5 | MobileNetV2 | chenyaofo | 92.88 | unknown | 700,490 |
| A027 | mobilenetv2_x1_0 | MobileNetV2 | chenyaofo | 93.79 | 94.05 | 2,236,682 |
| A028 | shufflenetv2_x0_5 | ShuffleNetV2 | chenyaofo | 90.13 | unknown | 352,042 |
| A029 | shufflenetv2_x1_0 | ShuffleNetV2 | chenyaofo | 92.98 | 93.30 | 1,263,854 |
| A030 | repvgg_a0 | RepVGG | chenyaofo | 94.39 | unknown | 7,840,874 |

Params는 등록된 Parameter 전체이다. A014 XDenseNet에는 학습하지 않는 고정 connectivity mask도 Parameter로 등록되어 있어 total=1,319,338, trainable=690,346이다. 이는 이번에 pruning을 적용한 모델이 아니라 공개 XDenseNet 원래 정의이다.

## B table

| ID | Architecture | Training difference | Source | Reported Acc % | Full Verified Acc % |
|---|---|---|---|---:|---:|
| B001 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=6, iteration=25 | ModelZoos | 87.36 | 87.36 |
| B002 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=92, iteration=25 | ModelZoos | 86.43 | unknown |
| B003 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=95, iteration=25 | ModelZoos | 85.09 | unknown |
| B004 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=107, iteration=25 | ModelZoos | 86.64 | unknown |
| B005 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=159, iteration=25 | ModelZoos | 84.93 | unknown |
| B006 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=185, iteration=25 | ModelZoos | 84.29 | unknown |
| B007 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=320, iteration=25 | ModelZoos | 82.86 | unknown |
| B008 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=375, iteration=25 | ModelZoos | 82.89 | unknown |
| B009 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=385, iteration=25 | ModelZoos | 86.65 | unknown |
| B010 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=412, iteration=25 | ModelZoos | 86.82 | unknown |
| B011 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=482, iteration=25 | ModelZoos | 85.94 | unknown |
| B012 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=493, iteration=25 | ModelZoos | 86.37 | unknown |
| B013 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=511, iteration=25 | ModelZoos | 86.69 | unknown |
| B014 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=580, iteration=25 | ModelZoos | 81.50 | unknown |
| B015 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=640, iteration=25 | ModelZoos | 82.19 | unknown |
| B016 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=662, iteration=25 | ModelZoos | 86.49 | unknown |
| B017 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=731, iteration=25 | ModelZoos | 84.32 | unknown |
| B018 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=749, iteration=25 | ModelZoos | 86.73 | unknown |
| B019 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=812, iteration=25 | ModelZoos | 85.11 | unknown |
| B020 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=836, iteration=25 | ModelZoos | 84.98 | unknown |
| B021 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=881, iteration=25 | ModelZoos | 85.00 | unknown |
| B022 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=894, iteration=25 | ModelZoos | 86.49 | unknown |
| B023 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=941, iteration=25 | ModelZoos | 86.70 | unknown |
| B024 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=962, iteration=25 | ModelZoos | 86.55 | unknown |
| B025 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=985, iteration=25 | ModelZoos | 86.95 | unknown |
| B026 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=441, iteration=25 | ModelZoos | 83.32 | 83.32 |
| B027 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=647, iteration=25 | ModelZoos | 86.54 | unknown |
| B028 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=213, iteration=25 | ModelZoos | 86.81 | unknown |
| B029 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=619, iteration=25 | ModelZoos | 84.09 | unknown |
| B030 | ResNet18-CIFAR-3x3-no-maxpool | independent seed=719, iteration=25 | ModelZoos | 84.44 | unknown |

## B2 table

| ID | Architecture | Hyperparameter difference | Source | Reported Acc % | Full Verified Acc % |
|---|---|---|---|---:|---:|
| B2_001 | PreActResNet18-CIFAR-3x3 | lr_max=0.0761962, SAM=0, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_002 | PreActResNet18-CIFAR-3x3 | lr_max=0.206771, SAM=0, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_003 | PreActResNet18-CIFAR-3x3 | lr_max=0.272861, SAM=0, aug=False | ICML 2023 sharpness zoo | unknown | 85.91 |
| B2_004 | PreActResNet18-CIFAR-3x3 | lr_max=0.558993, SAM=0, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_005 | PreActResNet18-CIFAR-3x3 | lr_max=1.42443, SAM=0, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_006 | PreActResNet18-CIFAR-3x3 | lr_max=0.111799, SAM=0, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_007 | PreActResNet18-CIFAR-3x3 | lr_max=0.271494, SAM=0, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_008 | PreActResNet18-CIFAR-3x3 | lr_max=0.387734, SAM=0, aug=True | ICML 2023 sharpness zoo | unknown | 95.50 |
| B2_009 | PreActResNet18-CIFAR-3x3 | lr_max=0.568488, SAM=0, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_010 | PreActResNet18-CIFAR-3x3 | lr_max=0.960321, SAM=0, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_011 | PreActResNet18-CIFAR-3x3 | lr_max=0.0551324, SAM=0.05, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_012 | PreActResNet18-CIFAR-3x3 | lr_max=0.218742, SAM=0.05, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_013 | PreActResNet18-CIFAR-3x3 | lr_max=0.292359, SAM=0.05, aug=False | ICML 2023 sharpness zoo | unknown | 88.72 |
| B2_014 | PreActResNet18-CIFAR-3x3 | lr_max=0.357789, SAM=0.05, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_015 | PreActResNet18-CIFAR-3x3 | lr_max=0.812511, SAM=0.05, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_016 | PreActResNet18-CIFAR-3x3 | lr_max=0.0599879, SAM=0.05, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_017 | PreActResNet18-CIFAR-3x3 | lr_max=0.0633788, SAM=0.05, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_018 | PreActResNet18-CIFAR-3x3 | lr_max=0.299438, SAM=0.05, aug=True | ICML 2023 sharpness zoo | unknown | 95.53 |
| B2_019 | PreActResNet18-CIFAR-3x3 | lr_max=0.463377, SAM=0.05, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_020 | PreActResNet18-CIFAR-3x3 | lr_max=1.34708, SAM=0.05, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_021 | PreActResNet18-CIFAR-3x3 | lr_max=0.0504226, SAM=0.1, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_022 | PreActResNet18-CIFAR-3x3 | lr_max=0.114267, SAM=0.1, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_023 | PreActResNet18-CIFAR-3x3 | lr_max=0.167475, SAM=0.1, aug=False | ICML 2023 sharpness zoo | unknown | 89.06 |
| B2_024 | PreActResNet18-CIFAR-3x3 | lr_max=0.361512, SAM=0.1, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_025 | PreActResNet18-CIFAR-3x3 | lr_max=0.786361, SAM=0.1, aug=False | ICML 2023 sharpness zoo | unknown | unknown |
| B2_026 | PreActResNet18-CIFAR-3x3 | lr_max=0.082242, SAM=0.1, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_027 | PreActResNet18-CIFAR-3x3 | lr_max=0.289074, SAM=0.1, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_028 | PreActResNet18-CIFAR-3x3 | lr_max=0.360359, SAM=0.1, aug=True | ICML 2023 sharpness zoo | unknown | 95.74 |
| B2_029 | PreActResNet18-CIFAR-3x3 | lr_max=0.366764, SAM=0.1, aug=True | ICML 2023 sharpness zoo | unknown | unknown |
| B2_030 | PreActResNet18-CIFAR-3x3 | lr_max=0.970862, SAM=0.1, aug=True | ICML 2023 sharpness zoo | unknown | unknown |

## Family distribution

| Family | A | B | B2 | Total |
|---|---:|---:|---:|---:|
| DIA-PreResNet | 1 | 0 | 0 | 1 |
| DIA-ResNet | 1 | 0 | 0 | 1 |
| DenseNet | 2 | 0 | 0 | 2 |
| MobileNetV2 | 2 | 0 | 0 | 2 |
| NIN | 1 | 0 | 0 | 1 |
| PreResNet | 2 | 0 | 0 | 2 |
| PyramidNet | 2 | 0 | 0 | 2 |
| RepVGG | 1 | 0 | 0 | 1 |
| ResNet | 3 | 30 | 30 | 63 |
| RiR | 1 | 0 | 0 | 1 |
| RoR | 2 | 0 | 0 | 2 |
| SE-PreResNet | 1 | 0 | 0 | 1 |
| SE-ResNet | 2 | 0 | 0 | 2 |
| Shake-Shake | 2 | 0 | 0 | 2 |
| ShuffleNetV2 | 2 | 0 | 0 | 2 |
| VGG | 2 | 0 | 0 | 2 |
| WideResNet | 2 | 0 | 0 | 2 |
| XDenseNet | 1 | 0 | 0 | 1 |

## B hard-negative structure

ResNet18-CIFAR-3x3-no-maxpool: 30 independent runs. B 내부 pair는 435개이다. 3×3 stride-1 stem, maxpool 없음, BasicBlock [2,2,2,2], classifier 10을 공유한다.

샘플 ZIP의 25개 run과 원본 tar의 추가 5개 run을 선택했다. 모두 서로 다른 seed이며, 동일 seed의 다른 epoch는 집계하지 않는다. 공개 generator는 seed grid 1..1000, kaiming_uniform initialization, resume=False, reuse_actors=False이다. 각 params.json과 result.jsonl을 보존했다.

**B는 iteration 25의 중간 학습 checkpoint**이다. 종료 iteration 50의 accuracy를 가져오지 않았다. 이 구성은 충분히 학습된 독립 run을 제공하지만 최종 수렴 모델 집단과 같지는 않다. A와 B 간 성능/학습 기간 차이는 추후 분석에서 통제해야 한다.

B recipe: SGD, lr=0.05, momentum=0.9, weight_decay=0.0005, OneCycleLR, batch=256. 원본 train 50,000개 중 40,000 학습/10,000 validation(split seed 42), 공식 test 10,000개. 독립성 근거는 seed/run provenance이며, hash 차이만으로 독립 학습을 증명한다고 간주하지 않는다.

## B2 hard-negative structure

PreActResNet18-CIFAR-width64: 30 independent hyperparameter runs, 내부 hard-negative pair 435개. lr_max 30개 값, SAM rho 0/0.05/0.1, augmentation off/on을 균형 있게 선택했다. 모든 파일은 seed=0이므로 initialization 변화가 아니라 training hyperparameter 변화만 통제한다.

B2는 공개 배포에서 `model=resnet18`로 명명되지만 실제 source implementation은 pre-activation block이다. 따라서 topology_id를 B의 post-activation ResNet18과 다르게 기록했다. B-B2 cross pair는 same_family=True, same_architecture=False이며 동일 topology 실험으로 해석하면 안 된다.

B2 recipe: SGD, 200 epochs, batch=128, width=64, cyclic schedule, weight_decay=0.0, label noise=0. 각 원본 container의 `last` state_dict를 로드하며 best/swa_last/swa_best state는 원본 파일 안에 보존한다.

## Verification

전체 10,000 test images accuracy 재계산: 18/90 모델. 그 외 full verified accuracy는 null이다. Smoke accuracy를 full test accuracy 칸에 넣지 않는다.

| ID | Smoke Acc % | Smoke N | Full Acc % | Warning |
|---|---:|---:|---:|---|
| A001 | 89.06 | 256 | 92.57 |  |
| A002 | 92.97 | 256 | 94.03 |  |
| A003 | 94.14 | 256 | 95.48 |  |
| A004 | 94.14 | 256 | unknown |  |
| A005 | 92.97 | 256 | unknown |  |
| A006 | 94.14 | 256 | 95.51 |  |
| A007 | 91.80 | 256 | unknown |  |
| A008 | 95.70 | 256 | 95.87 |  |
| A009 | 95.31 | 256 | unknown |  |
| A010 | 96.09 | 256 | 96.28 |  |
| A011 | 96.09 | 256 | unknown |  |
| A012 | 90.23 | 256 | unknown |  |
| A013 | 94.14 | 256 | 95.84 |  |
| A014 | 92.58 | 256 | unknown |  |
| A015 | 96.88 | 256 | 97.07 |  |
| A016 | 98.83 | 256 | unknown |  |
| A017 | 94.14 | 256 | unknown |  |
| A018 | 94.53 | 256 | unknown |  |
| A019 | 97.66 | 256 | unknown |  |
| A020 | 93.36 | 256 | unknown |  |
| A021 | 96.09 | 256 | unknown |  |
| A022 | 94.53 | 256 | unknown |  |
| A023 | 94.14 | 256 | unknown |  |
| A024 | 88.28 | 256 | unknown |  |
| A025 | 91.41 | 256 | unknown |  |
| A026 | 91.80 | 256 | unknown |  |
| A027 | 93.36 | 256 | 94.05 |  |
| A028 | 91.80 | 256 | unknown |  |
| A029 | 91.41 | 256 | 93.30 |  |
| A030 | 91.41 | 256 | unknown |  |
| B001 | 87.11 | 256 | 87.36 |  |
| B002 | 84.38 | 256 | unknown |  |
| B003 | 82.03 | 256 | unknown |  |
| B004 | 87.11 | 256 | unknown |  |
| B005 | 85.16 | 256 | unknown |  |
| B006 | 79.69 | 256 | unknown |  |
| B007 | 82.03 | 256 | unknown |  |
| B008 | 84.77 | 256 | unknown |  |
| B009 | 86.33 | 256 | unknown |  |
| B010 | 83.98 | 256 | unknown |  |
| B011 | 85.16 | 256 | unknown |  |
| B012 | 83.20 | 256 | unknown |  |
| B013 | 87.11 | 256 | unknown |  |
| B014 | 76.56 | 256 | unknown |  |
| B015 | 82.42 | 256 | unknown |  |
| B016 | 83.98 | 256 | unknown |  |
| B017 | 82.81 | 256 | unknown |  |
| B018 | 82.81 | 256 | unknown |  |
| B019 | 85.94 | 256 | unknown |  |
| B020 | 80.47 | 256 | unknown |  |
| B021 | 81.64 | 256 | unknown |  |
| B022 | 83.59 | 256 | unknown |  |
| B023 | 87.11 | 256 | unknown |  |
| B024 | 85.16 | 256 | unknown |  |
| B025 | 84.38 | 256 | unknown |  |
| B026 | 78.12 | 256 | 83.32 |  |
| B027 | 83.59 | 256 | unknown |  |
| B028 | 85.55 | 256 | unknown |  |
| B029 | 80.47 | 256 | unknown |  |
| B030 | 83.59 | 256 | unknown |  |
| B2_001 | 82.03 | 256 | unknown |  |
| B2_002 | 85.55 | 256 | unknown |  |
| B2_003 | 81.25 | 256 | 85.91 |  |
| B2_004 | 84.38 | 256 | unknown |  |
| B2_005 | 84.77 | 256 | unknown |  |
| B2_006 | 93.75 | 256 | unknown |  |
| B2_007 | 95.31 | 256 | unknown |  |
| B2_008 | 95.70 | 256 | 95.50 |  |
| B2_009 | 95.31 | 256 | unknown |  |
| B2_010 | 93.75 | 256 | unknown |  |
| B2_011 | 83.20 | 256 | unknown |  |
| B2_012 | 83.59 | 256 | unknown |  |
| B2_013 | 82.81 | 256 | 88.72 |  |
| B2_014 | 85.94 | 256 | unknown |  |
| B2_015 | 84.77 | 256 | unknown |  |
| B2_016 | 93.36 | 256 | unknown |  |
| B2_017 | 93.75 | 256 | unknown |  |
| B2_018 | 96.48 | 256 | 95.53 |  |
| B2_019 | 93.36 | 256 | unknown |  |
| B2_020 | 94.14 | 256 | unknown |  |
| B2_021 | 83.98 | 256 | unknown |  |
| B2_022 | 85.16 | 256 | unknown |  |
| B2_023 | 83.98 | 256 | 89.06 |  |
| B2_024 | 87.11 | 256 | unknown |  |
| B2_025 | 85.55 | 256 | unknown |  |
| B2_026 | 93.36 | 256 | unknown |  |
| B2_027 | 94.14 | 256 | unknown |  |
| B2_028 | 93.36 | 256 | 95.74 |  |
| B2_029 | 93.36 | 256 | unknown |  |
| B2_030 | 94.14 | 256 | unknown |  |

소규모 sample은 고정 seed 20260907로 test index를 선택한다. 모델 선택에 accuracy를 사용하지 않았다. 256개 smoke accuracy는 통계적 불확실성이 커서 보고된 full accuracy와 직접 동일시하면 안 된다.

Native/common/none preprocessing, lineage/topology 구분, B2 network contract, hash의 순서 독립성 및 buffer 민감도, Range 무시/잘못된 offset 차단, source 용량 예산 차단의 7개 contract test를 통과했다. 별도 manifest audit는 90개 checkpoint hash, 30개 B seed/run 증거, 30개 B2 hyperparameter signature, JSON/CSV ID 일치, 4,005개 pair의 관계를 검사한다.

## Download failures and source strategies

1. osmr/imgclsmob: 30개 중 A 23개를 개별 release ZIP으로 확보. 예전 pytorch/README.md 경로는 404였으며 osmr/pytorchcv README로 이동한 사실을 확인했다.
2. chenyaofo/pytorch-cifar-models: A 7개 개별 state_dict 확보. 각 training log의 native mean/std와 pretrained=False 기록을 보존했다.
3. ModelZoos selective download: ZIP central-directory 조회 후 선택 member byte range만 수신. API 다중 요청 중 HTTP 429 발생; 이미 확보한 metadata를 재사용하고 member별 단일 range 및 순차 요청/backoff로 수정했다.
4. PhaseTransitionModelZoo: README에서 Proton Drive ZIP 샘플 배포와 192개 width/batch-size grid 확인. 동일 family라도 width가 다르면 topology가 다르므로 그대로 same-architecture로 묶을 수 없다. ModelZoos에서 목표가 확보되어 해당 archive는 다운로드하지 않았다.
5. ICML 2023 sharpness zoo: 50개 CIFAR-10 `model=resnet18` 파일 중 30개를 개별 Google Drive URL로 확보. 폴더 전체 archive는 받지 않았다. Source code 확인 결과 실제 구현은 PreActResNet18이므로 별도 topology_id로 등록했다.
6. ICLR 2026 folding zoo: optimizer/lr/weight-decay/L1/RandAugment/SAM/scheduler를 바꾼 792개 post-activation ResNet18을 확인했으나 checkpoint 다운로드가 공개되지 않아 채택하지 않았다.
7. FLStore CIFAR10_resnet18.zip: 2.2GB로 용량 제한 이내지만 50 federated rounds의 client updates라 독립 original lineage 조건을 충족하지 않아 채택하지 않았다.

현재 후보별 오류 기록:

- 최종 다운로드/로드 실패 모델 없음.

## Duplicate check

파일 SHA256, state_dict SHA256(이름/shape/dtype/값/BN buffer), parameter SHA256를 각각 검사한다. 결과:

```json
{
  "duplicates": [],
  "checked_at": "2026-09-07T06:07:46Z",
  "models_with_file_hash": 90,
  "models_with_state_hash": 90,
  "models_with_parameter_hash": 90
}
```

## Unknown metadata

- A osmr 모델의 개별 seed, optimizer, learning rate, weight decay는 해당 checkpoint별 확실한 증거를 확보하지 못한 경우 null이다. 현재 소스의 기본 recipe를 과거 모든 모델의 실제 recipe로 추정하지 않았다.
- chenyaofo 모델은 각 저장 log의 optimizer/lr/weight decay/normalization을 기록했다. 확인되지 않은 seed와 augmentation은 null이다.
- B2의 source filename은 seed, lr_max, weight decay, SAM, augmentation, epoch, batch size를 기록한다. SGD는 논문의 고정 recipe로 확인했지만 momentum은 확인하지 못해 training_recipe.momentum=null이다. 개별 reported accuracy는 배포 폴더에 없어 null이다.
- framework_version은 이번 검증 runtime이다. 원래 학습 framework 버전은 training_framework_version이 있는 경우에만 확인된 값이다.
- A 모델은 서로 다른 공개 topology checkpoint라는 근거로 독립 originals로 등록했다. 모든 원본 학습의 데이터/난수 이력까지 복원한 것은 아니다.

## Native preprocessing and relations

입력은 float RGB [N,3,32,32], [0,1]. A는 source mean/std, B는 mean=[125.3,123.0,113.9]/255 및 std=[63.0,62.1,66.7]/255, B2는 mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010]을 적용한다. 출력은 CIFAR-10 표준 class 순서의 raw logits. `common` 모드는 mean/std를 명시해야 하며 `none`은 이미 전처리된 입력을 위한 선택이다.

`topology_id`는 구체 구현의 구조를 구분한다. 이름에 ResNet이 들어간다고 same_architecture로 판정하지 않는다. distinct A/B/B2 모델 pair는 모두 same_lineage=False이며 self-pair는 True이다. 향후 C는 원본 lineage_id를 상속하고 parent_id를 직접 부모 model_id로 기록해야 한다.

## Disk usage

| Source | Checkpoint bytes | GiB |
|---|---:|---:|
| osmr | 347,453,498 | 0.324 |
| chenyaofo | 150,696,444 | 0.140 |
| modelzoos | 1,343,215,110 | 1.251 |
| sharpness_cifar10 | 5,375,908,160 | 5.007 |

| Model | Bytes | Local checkpoint |
|---|---:|---|
| A001 | 3,871,814 | checkpoints/A/A001_nin_cifar10-0743-795b0824.pth |
| A002 | 1,120,753 | checkpoints/A/A002_resnet20_cifar10-0597-9b0024ac.pth |
| A003 | 3,507,568 | checkpoints/A/A003_resnet56_cifar10-0452-628c42a2.pth |
| A004 | 7,089,118 | checkpoints/A/A004_resnet110_cifar10-0369-4d6ca1fc.pth |
| A005 | 1,117,532 | checkpoints/A/A005_preresnet20_cifar10-0651-76cec68d.pth |
| A006 | 3,504,515 | checkpoints/A/A006_preresnet56_cifar10-0449-e9124fcf.pth |
| A007 | 1,139,690 | checkpoints/A/A007_seresnet20_cifar10-0601-935d8943.pth |
| A008 | 3,564,383 | checkpoints/A/A008_seresnet56_cifar10-0413-b61c1439.pth |
| A009 | 3,561,270 | checkpoints/A/A009_sepreresnet56_cifar10-0451-fc23e153.pth |
| A010 | 7,312,929 | checkpoints/A/A010_pyramidnet110_a48_cifar10-0372-eb185645.pth |
| A011 | 15,865,327 | checkpoints/A/A011_pyramidnet110_a84_cifar10-0298-7b835a3c.pth |
| A012 | 2,485,866 | checkpoints/A/A012_densenet40_k12_cifar10-0561-8b8e8194.pth |
| A013 | 3,284,562 | checkpoints/A/A013_densenet100_k12_bc_cifar10-0416-b9232829.pth |
| A014 | 5,367,253 | checkpoints/A/A014_xdensenet40_2_k24_bc_cifar10-0531-b91a9dc3.pth |
| A015 | 68,518,601 | checkpoints/A/A015_wrn16_10_cifar10-0293-ce810d8a.pth |
| A016 | 146,019,059 | checkpoints/A/A016_wrn28_10_cifar10-0239-fe97dcd6.pth |
| A017 | 3,142,669 | checkpoints/A/A017_ror3_56_cifar10-0543-44f0f47d.pth |
| A018 | 6,734,683 | checkpoints/A/A018_ror3_110_cifar10-0435-fb2a2b04.pth |
| A019 | 38,054,435 | checkpoints/A/A019_rir_cifar10-0328-414c3e60.pth |
| A020 | 2,222,854 | checkpoints/A/A020_shakeshakeresnet20_2x16d_cifar10-0515-ef71ec0d.pth |
| A021 | 11,786,456 | checkpoints/A/A021_shakeshakeresnet26_2x32d_cifar10-0317-ecd1f833.pth |
| A022 | 4,092,647 | checkpoints/A/A022_diaresnet56_cifar10-0505-8ac86804.pth |
| A023 | 4,089,514 | checkpoints/A/A023_diapreresnet56_cifar10-0483-41cae958.pth |
| A024 | 39,068,509 | checkpoints/A/A024_cifar10_vgg11_bn-eaeebf42.pt |
| A025 | 61,080,472 | checkpoints/A/A025_cifar10_vgg16_bn-6ee7ea24.pt |
| A026 | 2,986,233 | checkpoints/A/A026_cifar10_mobilenetv2_x0_5-ca14ced9.pt |
| A027 | 9,193,273 | checkpoints/A/A027_cifar10_mobilenetv2_x1_0-fe6a5b48.pt |
| A028 | 1,554,833 | checkpoints/A/A028_cifar10_shufflenetv2_x0_5-1308b4e9.pt |
| A029 | 5,230,673 | checkpoints/A/A029_cifar10_shufflenetv2_x1_0-98807be3.pt |
| A030 | 31,582,451 | checkpoints/A/A030_cifar10_repvgg_a0-ef08a50e.pt |
| B001 | 44,773,837 | checkpoints/B/B001_resnet18_seed6_iteration25.pt |
| B002 | 44,773,837 | checkpoints/B/B002_resnet18_seed92_iteration25.pt |
| B003 | 44,773,837 | checkpoints/B/B003_resnet18_seed95_iteration25.pt |
| B004 | 44,773,837 | checkpoints/B/B004_resnet18_seed107_iteration25.pt |
| B005 | 44,773,837 | checkpoints/B/B005_resnet18_seed159_iteration25.pt |
| B006 | 44,773,837 | checkpoints/B/B006_resnet18_seed185_iteration25.pt |
| B007 | 44,773,837 | checkpoints/B/B007_resnet18_seed320_iteration25.pt |
| B008 | 44,773,837 | checkpoints/B/B008_resnet18_seed375_iteration25.pt |
| B009 | 44,773,837 | checkpoints/B/B009_resnet18_seed385_iteration25.pt |
| B010 | 44,773,837 | checkpoints/B/B010_resnet18_seed412_iteration25.pt |
| B011 | 44,773,837 | checkpoints/B/B011_resnet18_seed482_iteration25.pt |
| B012 | 44,773,837 | checkpoints/B/B012_resnet18_seed493_iteration25.pt |
| B013 | 44,773,837 | checkpoints/B/B013_resnet18_seed511_iteration25.pt |
| B014 | 44,773,837 | checkpoints/B/B014_resnet18_seed580_iteration25.pt |
| B015 | 44,773,837 | checkpoints/B/B015_resnet18_seed640_iteration25.pt |
| B016 | 44,773,837 | checkpoints/B/B016_resnet18_seed662_iteration25.pt |
| B017 | 44,773,837 | checkpoints/B/B017_resnet18_seed731_iteration25.pt |
| B018 | 44,773,837 | checkpoints/B/B018_resnet18_seed749_iteration25.pt |
| B019 | 44,773,837 | checkpoints/B/B019_resnet18_seed812_iteration25.pt |
| B020 | 44,773,837 | checkpoints/B/B020_resnet18_seed836_iteration25.pt |
| B021 | 44,773,837 | checkpoints/B/B021_resnet18_seed881_iteration25.pt |
| B022 | 44,773,837 | checkpoints/B/B022_resnet18_seed894_iteration25.pt |
| B023 | 44,773,837 | checkpoints/B/B023_resnet18_seed941_iteration25.pt |
| B024 | 44,773,837 | checkpoints/B/B024_resnet18_seed962_iteration25.pt |
| B025 | 44,773,837 | checkpoints/B/B025_resnet18_seed985_iteration25.pt |
| B026 | 44,773,837 | checkpoints/B/B026_resnet18_seed441_iteration25.pt |
| B027 | 44,773,837 | checkpoints/B/B027_resnet18_seed647_iteration25.pt |
| B028 | 44,773,837 | checkpoints/B/B028_resnet18_seed213_iteration25.pt |
| B029 | 44,773,837 | checkpoints/B/B029_resnet18_seed619_iteration25.pt |
| B030 | 44,773,837 | checkpoints/B/B030_resnet18_seed719_iteration25.pt |
| B2_001 | 179,197,297 | checkpoints/B2/B2_001_preact_resnet18_lr0.0761962_sam0_aug0.pth |
| B2_002 | 179,197,297 | checkpoints/B2/B2_002_preact_resnet18_lr0.206771_sam0_aug0.pth |
| B2_003 | 179,197,297 | checkpoints/B2/B2_003_preact_resnet18_lr0.272861_sam0_aug0.pth |
| B2_004 | 179,197,297 | checkpoints/B2/B2_004_preact_resnet18_lr0.558993_sam0_aug0.pth |
| B2_005 | 179,196,437 | checkpoints/B2/B2_005_preact_resnet18_lr1.42443_sam0_aug0.pth |
| B2_006 | 179,196,437 | checkpoints/B2/B2_006_preact_resnet18_lr0.111799_sam0_aug1.pth |
| B2_007 | 179,196,437 | checkpoints/B2/B2_007_preact_resnet18_lr0.271494_sam0_aug1.pth |
| B2_008 | 179,196,437 | checkpoints/B2/B2_008_preact_resnet18_lr0.387734_sam0_aug1.pth |
| B2_009 | 179,196,437 | checkpoints/B2/B2_009_preact_resnet18_lr0.568488_sam0_aug1.pth |
| B2_010 | 179,196,437 | checkpoints/B2/B2_010_preact_resnet18_lr0.960321_sam0_aug1.pth |
| B2_011 | 179,197,727 | checkpoints/B2/B2_011_preact_resnet18_lr0.0551324_sam0.05_aug0.pth |
| B2_012 | 179,197,727 | checkpoints/B2/B2_012_preact_resnet18_lr0.218742_sam0.05_aug0.pth |
| B2_013 | 179,197,727 | checkpoints/B2/B2_013_preact_resnet18_lr0.292359_sam0.05_aug0.pth |
| B2_014 | 179,197,727 | checkpoints/B2/B2_014_preact_resnet18_lr0.357789_sam0.05_aug0.pth |
| B2_015 | 179,197,297 | checkpoints/B2/B2_015_preact_resnet18_lr0.812511_sam0.05_aug0.pth |
| B2_016 | 179,196,867 | checkpoints/B2/B2_016_preact_resnet18_lr0.0599879_sam0.05_aug1.pth |
| B2_017 | 179,196,867 | checkpoints/B2/B2_017_preact_resnet18_lr0.0633788_sam0.05_aug1.pth |
| B2_018 | 179,196,867 | checkpoints/B2/B2_018_preact_resnet18_lr0.299438_sam0.05_aug1.pth |
| B2_019 | 179,196,867 | checkpoints/B2/B2_019_preact_resnet18_lr0.463377_sam0.05_aug1.pth |
| B2_020 | 179,196,867 | checkpoints/B2/B2_020_preact_resnet18_lr1.34708_sam0.05_aug1.pth |
| B2_021 | 179,197,297 | checkpoints/B2/B2_021_preact_resnet18_lr0.0504226_sam0.1_aug0.pth |
| B2_022 | 179,196,867 | checkpoints/B2/B2_022_preact_resnet18_lr0.114267_sam0.1_aug0.pth |
| B2_023 | 179,197,297 | checkpoints/B2/B2_023_preact_resnet18_lr0.167475_sam0.1_aug0.pth |
| B2_024 | 179,197,297 | checkpoints/B2/B2_024_preact_resnet18_lr0.361512_sam0.1_aug0.pth |
| B2_025 | 179,197,297 | checkpoints/B2/B2_025_preact_resnet18_lr0.786361_sam0.1_aug0.pth |
| B2_026 | 179,196,007 | checkpoints/B2/B2_026_preact_resnet18_lr0.082242_sam0.1_aug1.pth |
| B2_027 | 179,196,437 | checkpoints/B2/B2_027_preact_resnet18_lr0.289074_sam0.1_aug1.pth |
| B2_028 | 179,196,437 | checkpoints/B2/B2_028_preact_resnet18_lr0.360359_sam0.1_aug1.pth |
| B2_029 | 179,196,437 | checkpoints/B2/B2_029_preact_resnet18_lr0.366764_sam0.1_aug1.pth |
| B2_030 | 179,196,437 | checkpoints/B2/B2_030_preact_resnet18_lr0.970862_sam0.1_aug1.pth |

원본 tar 크기는 2,328,714,690,560 bytes이지만 전체 파일을 저장/다운로드하지 않았다. tar 헤더 탐색은 약 4.8MB, 이후 필요한 checkpoint 5개 및 params/log만 Range로 받았다. ModelZoos ZIP도 전체 5.1GB를 받지 않았다. B2는 30개 direct file만 받았다. `metadata/transfer_ledger.json`은 downloader의 source별 누적 보수적 전송 예산이며 초기 조사 traffic은 별도다.

## C parent recommendations (아직 생성하지 않음)

| ID | Architecture | 선정 이유 |
|---|---|---|
| A001 | NIN | residual 없는 convolution baseline |
| A002 | ResNet20 | 작고 빠른 residual baseline |
| A003 | ResNet56 | 깊이에 따른 변환 영향 비교 |
| A006 | PreResNet56 | preactivation 비교 |
| A008 | SE-ResNet56 | channel attention 비교 |
| A010 | PyramidNet110-a48 | 점진적 channel 증가 |
| A013 | DenseNet100-k12-BC | dense connectivity 비교 |
| A015 | WRN16-10 | width 중심 residual baseline |
| A027 | MobileNetV2-x1.0 | depthwise/inverted residual |
| A029 | ShuffleNetV2-x1.0 | channel split/shuffle 비교 |

선정 기준은 구조 대표성, 저장량, 구현 접근성이다. RepVGG는 topology를 보존할 실험과 deployment reparameterization을 구분해야 하므로 첫 10개에서 제외했다.

추천 A 부모 10개는 모두 공식 test set 10,000장 전체 accuracy 검증을 마쳤다.

## Sources

- [osmr/imgclsmob](https://github.com/osmr/imgclsmob)
- [osmr/pytorchcv](https://github.com/osmr/pytorchcv)
- [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
- [ModelZoos/ModelZooDataset](https://github.com/ModelZoos/ModelZooDataset)
- [Model Zoo samples — Zenodo](https://zenodo.org/records/13144018)
- [PhaseTransitionModelZoo](https://github.com/ModelZoos/PhaseTransitionModelZoo)
- [ICML 2023 sharpness-vs-generalization](https://github.com/tml-epfl/sharpness-vs-generalization)
- [CIFAR-10 model files](https://drive.google.com/drive/folders/1dwfb2Iqw6BTMi57SeG44aDebuu-TBzo5)
- [ICLR 2026 Cut Less, Fold More](https://arxiv.org/abs/2602.18116)

정확한 commit/release, direct URL, archive member/byte offset, hash, source snapshot은 metadata/models.json, sources.json, evidence에 저장되어 있다.
