# Visual Assets & Visualization History

이 문서는 프로젝트 개발 및 모델 훈련 과정에서 생성된 주요 시각화 이미지들의 의미와 용도를 기술합니다.

## 1. 랜드마크 탐지 관련 (Landmark Detection)

### [`interim_result.jpg`](file:///e:/Github/Automatic-Cephalometric-Landmark-Detection-and-CVM-Stage-Classification/interim_result.jpg)
- **생성 시점**: 랜드마크 히트맵 모델(ResNet-50 Heatmap V2) 훈련 과정 중 (2026-02-10)
- **의미**: 훈련 200 에포크 완주 후 최종 도달한 **MRE 4.25 px**의 정밀도를 시각적으로 증명하는 파일입니다.
- **분석**: 29개 랜드마크가 해부학적 구조(Sella, Nasion 등)에 자석처럼 정확히 안착한 모습을 확인할 수 있으며, 기존 회귀 모델의 뭉침 현상을 완전히 극복했음을 보여줍니다.

### [`debug_dataset_output.jpg`](file:///e:/Github/Automatic-Cephalometric-Landmark-Detection-and-CVM-Stage-Classification/debug_dataset_output.jpg)
- **생성 시점**: 히트맵 데이터셋 로직 개발 및 검증 단계
- **의미**: 입력 이미지, 좌표 기반 랜드마크, 그리고 생성된 가우시안 히트맵이 공간적으로 완벽하게 일치하는지 확인하기 위한 디버깅용 이미지입니다.
- **분석**: 데이터 전처리 파이프라인의 무결성을 보장하는 기술적 증거입니다.

### [`docs/assets/inference_results.png`](file:///e:/Github/Automatic-Cephalometric-Landmark-Detection-and-CVM-Stage-Classification/docs/assets/inference_results.png)
- **생성 시점**: 프로젝트 초기 랜드마크-CVM 통합 모델 테스트 단계
- **의미**: 랜드마크 예측 좌표와 CVM 단계 분류 결과가 한 장의 이미지에 오버레이되어 나타나는 최종 분석 리포트 양식의 시초입니다.

## 2. CVM 단계 분류 관련 (CVM Stage Classification)

### [`docs/assets/cvm_premium_visualization.png`](file:///e:/Github/Automatic-Cephalometric-Landmark-Detection-and-CVM-Stage-Classification/docs/assets/cvm_premium_visualization.png)
- **생성 시점**: CVM V2(768px) 고해상도 학습 및 Streamlit UI 디자인 고도화 단계
- **의미**: 경추(C2, C3, C4) 영역을 정밀 탐지(Detection)하고, 각 노드의 특징을 분석하여 최종 성숙 단계를 도출하는 과정을 프리미엄 UI 스타일로 구성한 예시입니다.
- **분석**: 단순 분류를 넘어 의료 전문가에게 '근거 있는 분석'을 시각적으로 제공하는 인터페이스의 지향점을 보여줍니다.

---
*연구 및 재현성을 위해 위 이미지들은 보존되며, 향후 논문 작성이나 기술 보고 시 근거 자료로 활용될 수 있습니다.*
