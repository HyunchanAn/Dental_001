# 자동 두부 계측 랜드마크 탐지 및 CVM 단계 분류 프로젝트

이 프로젝트는 Aariz 데이터셋을 사용하여 **자동 두부 계측 랜드마크 탐지**와 **경추골 성숙도(CVM) 단계 분류**를 수행하는 인공지능 모델을 개발하는 프로젝트입니다.

## 🚀 최종 성과

### 1. 랜드마크 탐지 (Landmark Detection)
- **성능:** **MRE (평균 반경 오차) 4.28 px** 달성
- **모델:** `HeatmapModel` (ResNet-50 백본)
- **상태:** 최적화 완료

### 2. CVM 단계 분류 (CVM Stage Classification) - **개발 성공**
- **방식:** Two-Stage Pipeline (Detection + Classification)
- **성능:** **Quadratic Weighted Kappa 0.63** 달성
- **모델:** YOLOv8 (Detector) + EfficientNet-B0 CORAL (Classifier)
- **특징:** 기존 랜드마크 기반 방식의 한계를 극복하고, ROI 추출을 통한 정밀 판독 시스템 구축

<p align="center">
  <img src="inference_results.png" width="800">
</p>
<p align="center">통합 분석 파이프라인(inference.py) 실행 결과 예시</p>

---

## 📂 프로젝트 구조 (Project Structure)

본 프로젝트는 유지보수와 전문적인 개발 환경을 위해 다음과 같이 체계적으로 구조화되어 있습니다.

- **`src/`**: 핵심 소스 코드. 모델 정의, 데이터셋 로드 로직 등 (백엔드 엔진)
- **`scripts/`**: 실행용 스크립트. 데이터 전처리, YOLO/Classifier 학습 (작업 공정)
- **`tools/`**: 실무용 엔드포인트 도구. 통합 인퍼런스(`inference.py`), 라벨링 툴 (최종 제품)
- **`analysis/`**: 데이터 분석 및 시각화 도구. 분포 분석, Attention 분석, 고급 시각화
- **`docs/`**: 프로젝트 기록관. 개발 일지(`development_log.txt`), 전략 기획서, 메모
- **`checkpoints/`**: 학습된 모델 가중치 파일 (.pth, .pt) 저장소
- **`archive/`**: 과거 실험 코드 및 참고용 구버전 스크립트 보관소

---

## 🛠️ 주요 기능 및 사용법

### 1. 통합 분석 (Inference)
이미지 한 장으로 경추 검출과 CVM 단계 판독을 동시에 수행합니다.
```bash
python tools/inference.py
```
- 테스트 세트에서 랜덤으로 샘플을 추출하여 시각화 결과(`inference_results.png`)를 생성합니다.

### 2. CVM 단계 분류 학습 파이프라인
프로젝트는 크게 세 단계로 수동 라벨링부터 학습까지 이어집니다.

1. **ROI 라벨링**: YOLO 학습을 위한 경추 영역 박싱 (180장 완료)
   ```bash
   python tools/labeling_tool.py
   ```
2. **Detector 학습**: YOLOv8를 이용한 경추 검출기 학습
   ```bash
   python scripts/train_yolo.py
   ```
3. **Classifier 학습**: 검출된 ROI를 이용한 EfficientNet 분류기 학습
   ```bash
   python scripts/prepare_classification_data.py
   python scripts/train_classifier.py
   ```

---

## 📂 프로젝트 구조 및 기록

- **상세 개발 지표**: `training_log/` 폴더 내 CSV 파일 참조
- **상세 개발 일지**: `development_log.txt`에 모든 실험 과정과 Plan 2 성공기가 상세히 기록되어 있습니다.
- **모델 파일**:
  - 랜드마크: `checkpoints/best_model.pth`
  - 경추 검출: `yolo_runs/cvm_detector/weights/best.pt`
  - CVM 분류: `checkpoints/best_cvm_classifier.pth`
- **정리된 코드**: `archive/` 폴더에 과거 실패했던 실험 코드(MIL 등)가 보관되어 있습니다.

## ⚙️ 환경 설정
```bash
pip install -r requirements.txt
# 추가 라이브러리 (YOLOv8 등)
pip install ultralytics
```

---
*이 프로젝트는 Apple Silicon (M2 Pro) 환경에서 MPS 가속을 사용하여 최적화되었습니다.*
