# CephAI Pro: Medical Data Privacy & Regulatory Compliance Guidelines

본 문서는 CephAI Pro 진단 엔진이 의료 정보 데이터 취급 시 국내외 법적 규제(대한민국 보건복지부, 미국 HIPAA, 유럽 GDPR)를 어떻게 준수하는지 설명하고, 환자의 개인정보 유출을 원천 방지하기 위해 설계된 기술적 보호 장치를 명세합니다.

---

## 1. Regulatory Context

의료 영상 데이터(Lateral Cephalogram X-ray)는 민감한 생체 정보 및 비정형 개인정보에 해당하므로, 데이터 수집·처리·추론 전 과정에서 철저한 비식별화 처리가 요구됩니다.

### 1.1 대한민국 보건복지부 & 개인정보보호위원회: 보건의료 데이터 활용 가이드라인
- <b>비정형 데이터 가명처리 기준</b>: 비정형 데이터(의료 영상 이미지, 텍스트 등)는 성명, 환자 번호, 생년월일 등의 직접 식별자뿐만 아니라 이미지 메타데이터 내에 존재하는 간접 식별자까지 소거하여 추가 정보 없이는 특정 개인을 식별할 수 없도록 조치해야 합니다.
- <b>익명 정보의 지위</b>: 기술적, 시간적 수단을 동원해도 더 이상 특정 개인을 재식별할 수 없는 상태(익명 정보)에 도달할 경우, 이는 개인정보보호법의 규제 대상에서 제외되며 안전하고 폭넓은 의료 연구에 활용할 수 있습니다.

### 1.2 미국 HIPAA (Health Insurance Portability and Accountability Act)
- <b>Safe Harbor 방법론 준수</b>: 성명, 주소, 모든 종류의 구체적 날짜, 전화번호, 이메일, 환자 고유 식별 번호(Patient ID) 등 18가지 종류의 개인 건강 정보(PHI, Protected Health Information) 식별자를 완전히 소거해야 안전한 비식별 정보로 인정받을 수 있습니다.

### 1.3 유럽 GDPR (General Data Protection Regulation)
- <b>Anonymisation vs Pseudonymisation</b>: 데이터의 단순 가명화(Pseudonymisation)를 넘어 추가 정보의 결합을 통한 재식별 위험이 전혀 없는 완전한 익명화(Anonymisation) 상태를 실현하여 정보 주체의 권리를 보장하고 국외 이전 및 시스템 연동 시의 법적 리스크를 해소해야 합니다.

---

## 2. Technical Compliance Measures in CephAI Pro

CephAI Pro는 위 규정들에 명시된 요구사항을 충족하고 환자 정보를 보호하기 위해 아래의 핵심 3대 보안 기술 아키텍처를 진단 파이프라인에 내장하였습니다.

### 2.1 Metadata De-identification Pipeline (메타데이터 물리적 소거)
- <b>기술적 메커니즘</b>: API(/predict) 서버에 이미지를 업로드할 때, Python 메모리 레벨에서 <code>cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)</code>를 사용하여 이미지의 순수 픽셀 데이터(Numeric Array)만을 디코딩합니다.
- <b>효과</b>: 이 방식은 DICOM 헤더 정보, EXIF 태그, 촬영 기기 고유 번호, 날짜, 환자 메타데이터 등 바이너리 파일 헤더에 숨겨진 모든 간접 식별 변수를 100% 물리적으로 즉시 소거(Strip)합니다. 오직 연산에 필요한 가시적 픽셀 매트릭스만 추출되므로, 메타데이터 변조를 통한 재식별이 절대 불가능합니다.

### 2.2 Zero-Storage Policy (무보존 메모리 추론)
- <b>기술적 메커니즘</b>: 업로드된 원본 X-ray 이미지 파일 및 메타데이터는 서버의 물리적/가상 디스크 디렉토리에 영구 저장(Persistence)되지 않습니다.
- <b>효과</b>: 모든 이미지 디코딩, 전처리, UNet 및 EfficientNet 연산은 CPU/GPU 가속 장치 메모리(RAM/VRAM) 상에서 비동기로 수행됩니다. 추론 결과를 취득한 직후 가비지 컬렉터(Garbage Collector)에 의해 메모리 점유가 해제되며 잔재 데이터를 물리적으로 소거하므로, 서버 침입이나 스토리지 유출 사고 발생 시에도 누출될 환자 데이터 자체가 존재하지 않는 완전 무보존 환경을 구현합니다.

### 2.3 Anonymized Diagnostic Output (비식별 진단 출력)
- <b>기술적 메커니즘</b>: API가 반환하는 최종 JSON 응답값에는 환자의 성명, 병원 번호 등 개인식별 변수가 완전히 배제되며, 랜드마크의 해부학적 약어(예: "A", "ANS", "Me")와 그에 매핑되는 픽셀 좌표값(x, y), 그리고 추론 시간(latency_ms) 등 순수 분석 지표만을 포함합니다.
- <b>효과</b>: 외부 연동 PACS나 클라이언트 시스템이 전달받는 진단서는 완벽한 탈식별 상태를 갖추어, 국제 표준 규제 준수성을 완벽히 만족합니다.
