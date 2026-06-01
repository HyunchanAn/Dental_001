# Implementation Plan: CVM Stage Classification & Landmark Detection Enhancement (Issues #1, #2, #3)

본 계획은 기존 모델의 한계를 극복하고 임상적 유효성을 강화하기 위한 세 가지 주요 이슈(#1, #2, #3)를 해결하는 알고리즘 고도화 로드맵입니다. 특히 이준엽 교수님의 MREIT 역문제(Inverse Problem) 방법론을 응용한 전처리와, CORAL 분류 모델의 오분류 제어 로직을 핵심으로 합니다.

## User Review Required

> [!WARNING]
> **역문제(Inverse Problem) 기반 전처리 구현 수준**
> 원래의 Equipotential Line Method(2002)는 내부 저항 분포를 복원하기 위한 편미분 방정식(PDE) 해법입니다. 본 프로젝트에서는 이를 이미지 도메인에 맞게 변형하여, **라플라시안(Laplacian) 연산과 Anisotropic Diffusion(비등방성 확산) 필터**를 결합한 수학적 에지 보존(Edge-Preserving) 전처리 모듈로 근사(Approximation)하여 적용할 계획입니다. 완전한 역문제 복원 방정식을 풀기 위해서는 추가적인 경계 조건(Boundary Condition) 데이터가 필요한데, 이 방식으로 근사하여 진행해도 괜찮을까요?

> [!IMPORTANT]
> **CS3 vs CS4 비대칭 페널티(Cost-Sensitive Loss) 가중치 설정**
> 임상적으로 가장 치명적인 오분류인 CS3(최대 성장기 직전)과 CS4(직후) 간의 오류를 강하게 처벌하기 위해, 오차 손실(Loss) 계산 시 해당 구간을 넘어가는 오분류에 대해 페널티 가중치를 2~3배 높일 예정입니다. 해당 수치에 대한 특별한 임상적 기준이 있다면 피드백 부탁드립니다.

## Open Questions

- 랜드마크 기반 Fallback ROI 추정 시, 기준이 될 특정 랜드마크(예: Me, Go, Gn 등 하악골 근처 랜드마크) 위치를 기준으로 C2-C4 영역의 오프셋(Offset)과 스케일(Scale)을 정적으로(통계 평균값) 산출하는 방식이 적절할까요? 아니면 간단한 선형 회귀 맵핑 모델을 추가로 학습시키는 것이 좋을까요? (일단 통계적 정적 오프셋 계산 방식으로 제안합니다.)

---

## Proposed Changes

### 1. 전처리 모듈 (Inverse Problem Approximation)

#### [NEW] `src/cvm/inverse_filter.py`
- 이미지의 포텐셜 필드를 모사하여 등전위선을 구하는 로직으로 Anisotropic Diffusion 및 Gradient Preserving 필터를 구현합니다.
- `apply_equipotential_filter(image)` 함수를 통해 원본 영상의 노이즈를 억제하면서 경추 피질골(Cortical Bone)의 함몰도(Concavity)와 에지를 선명하게 보존합니다.

### 2. 추론 엔진 고도화 (Fallback ROI & Confidence Filter)

#### [MODIFY] `tools/inference.py`
- **Inverse Filter 연동:** YOLO 예측 전 또는 분류기 입력 전에 `apply_equipotential_filter`를 통과시키도록 파이프라인 수정.
- **Fallback ROI 로직 구현:** YOLO(`results[0].boxes`)가 검출되지 않을 경우, `src.landmark.model`의 UNetHeatmapModel을 호출하여 하악골 하연(Gonion, Menton) 좌표를 획득하고, 이를 기준으로 통계적 상대 좌표를 역산(Inverse Mapping)하여 C2-C4 영역 BBox를 강제로 추정합니다.
- **Confidence Score 안전장치 (Uncertainty Rejection):** `proba_to_label`에서 변환된 확률값(sigmoid)의 분산이나 마진이 임계치(Threshold) 미만일 경우 `Uncertain` 상태를 반환하여 임상적 리스크 방지.

### 3. CVM 모델 학습 고도화 (Cost-Sensitive CORAL)

#### [MODIFY] `src/cvm/model_mil.py` (또는 신규 손실 함수)
- 기존 Binary Cross Entropy의 합산 방식이었던 CORAL 로스에 **Cost-Sensitive Weighting** 매트릭스를 적용.
- CS3과 CS4 사이를 가로지르는 예측 오차에 대해 `cost_weight = 3.0`을 부여하는 커스텀 로스 `CostSensitiveCoralLoss` 구현.
- `train_cvm.py` (존재할 경우) 또는 모델 아키텍처 내 손실 함수 계산부에 반영.

## Verification Plan

### Automated Tests
- `tests/test_inverse_filter.py`: 역문제 필터 적용 전후의 이미지 그레디언트(Gradient) 총합이 보존되면서 노이즈 분산이 줄어드는지 수치적 검증.
- `tests/test_fallback_roi.py`: 의도적으로 YOLO가 실패하도록 노이즈가 심한 영상을 주입한 뒤, 랜드마크 기반 Fallback 로직이 ROI 좌표를 정상 반환하는지 검증.
- `tests/test_cost_sensitive_loss.py`: CS3 라벨을 CS4로 예측했을 때의 손실값이 CS1을 CS2로 예측했을 때보다 2배 이상 높게 산출되는지 단위 테스트 수행.

### Manual Verification
- `docs/assets/inference_results.png` 시각화를 통해, 이전 버전에서 오분류되었던 모호한 CS3/CS4 데이터가 올바르게 분류되거나 'Uncertain'으로 보류되는지 육안 확인.
- YOLO 실패 데이터에 대해 그려진 Fallback ROI BBox가 실제 C2-C4 위치를 어느 정도 커버하는지 시각적 확인.
