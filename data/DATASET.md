# Part 2-2 데이터셋 설명

## 개요
ViT (Vision Transformer) 전이학습 및 YOLOv8 객체 탐지를 위한 제조 결함 이미지 데이터.

## 포함 샘플 파일

### MachineVision/
| 파일 | 설명 |
|------|------|
| `sample.png` | ViT 입문 노트북용 샘플 이미지 (패치 분할 시각화) |

### defect_detection/ (YOLO 형식)
| 폴더 | 포함 수 | 원본 전체 수 | 설명 |
|------|---------|-------------|------|
| `images/train/` | 5장 | 350+ | 학습용 결함 이미지 |
| `images/val/` | 3장 | 100+ | 검증용 결함 이미지 |
| `images/test/` | 3장 | 50+ | 테스트용 결함 이미지 |
| `labels/train/` | 5개 | 350+ | YOLO 형식 바운딩 박스 라벨 |
| `labels/val/` | 3개 | 100+ | 검증용 라벨 |
| `labels/test/` | 3개 | 50+ | 테스트용 라벨 |
| `data.yaml` | 1개 | - | 데이터셋 설정 (클래스 정의) |

**총 포함**: 24 파일 (Git 저장소 용량 관리를 위해 최소 샘플만 포함)

### YOLO 라벨 형식
```
<class_id> <x_center> <y_center> <width> <height>
```
- 좌표: 이미지 크기 대비 정규화 (0~1)
- 클래스: 0=scratch, 1=contamination, 2=crack

### data.yaml 구조
```yaml
names: [scratch, contamination, crack]
nc: 3
path: ../data/defect_detection
train: images/train
val: images/val
test: images/test
```

## KAMP 원본 데이터셋 (전체)

**출처**: [KAMP (Korea AI Manufacturing Platform)](https://www.kamp-ai.kr/)

### 제조 결함 객체 탐지 데이터
- **데이터셋명**: 금속 표면 결함 검출 데이터 (바운딩 박스 버전)
- **전체 규모**: ~5,000장 + 라벨, ~3GB
- **클래스**: scratch, contamination, crack (3 클래스)
- **이미지 사양**: 640×640 px, RGB, JPG
- **라벨 형식**: YOLO v5/v8 호환 (txt)
- **분할**: train 70% / val 20% / test 10%

### ViT 전이학습용 데이터
- **데이터셋명**: 제조 이미지 분류 데이터
- **전체 규모**: ~4,000장, ~2GB
- **클래스**: 정상/불량 이진 분류 또는 다중 결함 분류
- **이미지 사양**: 224×224 px, RGB

### 다운로드 방법
1. https://www.kamp-ai.kr/ 접속
2. 결함 검출 데이터셋 신청 (바운딩 박스 버전)
3. `../../dataset/part2-2/` 경로에 배치
4. data.yaml의 path를 실제 경로로 수정

> 노트북은 KAMP 실데이터가 없으면 자동으로 샘플 데이터(11장)로 학습을 시연합니다.
