# Part 2-2: 생산최적화 (Production Optimization)

> **v12 Enhanced**: ViT 전이학습 + YOLOv8 객체 탐지 + 최신 비전 AI

---

## 🎯 학습 목표

- ✅ Vision Transformer (ViT) 이해
- ✅ ViT 전이학습으로 불량 분류
- ✅ YOLOv8 Fine-tuning
- ✅ 실시간 불량 탐지 시스템
- ✅ 모델 배포 기초

---

## 📚 실습 구성

| 순서 | 실습 | 파일 | 소요 시간 | 난이도 |
|:----:|------|------|:---------:|:------:|
| 1 | ViT 기초 | `01_vit_introduction.ipynb` | 30분 | ⭐⭐ |
| 2 | ViT 전이학습 | `02_vit_transfer_learning.ipynb` | 60분 | ⭐⭐⭐ |
| 3 | YOLOv8 Fine-tuning | `03_yolov8_finetuning.ipynb` | 60분 | ⭐⭐⭐ |

**총 소요 시간**: 약 2.5시간

---

## 🚀 시작하기

### 1️⃣ 환경 설정

```bash
# Part 2-2 폴더로 이동
cd practice-v12-enhanced/part2-2

# PyTorch 설치 (CUDA 지원)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# HuggingFace & Ultralytics
pip install transformers datasets accelerate ultralytics
```

### 2️⃣ GPU 확인

```python
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"GPU 이름: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU만 사용'}")
```

---

## 📊 사용 모델

### Vision Transformer (ViT)

| 항목 | 내용 |
|------|------|
| **모델** | `google/vit-base-patch16-224` |
| **용도** | 이미지 분류 |
| **파라미터** | 86M |
| **입력 크기** | 224×224 |

### YOLOv8

| 항목 | 내용 |
|------|------|
| **모델** | `yolov8n.pt` (nano) |
| **용도** | 객체 탐지 |
| **속도** | ~100 FPS (GPU) |
| **정확도** | mAP 37.3 |

---

## 🔧 실습 상세 내용

### 실습 1: ViT 기초 (30분)

**학습 내용**:
- Transformer vs CNN 비교
- ViT 아키텍처 이해
- 사전학습 모델 로드
- 간단한 이미지 분류

**주요 코드**:
```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# 모델 & 프로세서 로드
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 이미지 분류
image = Image.open('defect.jpg')
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# 예측
predicted_class = logits.argmax(-1).item()
print(f"예측 클래스: {model.config.id2label[predicted_class]}")
```

### 실습 2: ViT 전이학습 (60분)

**학습 내용**:
- 커스텀 데이터셋 준비
- ViT Fine-tuning
- 학습 모니터링
- 성능 평가 및 시각화

**주요 코드**:
```python
from transformers import ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("imagefolder", data_dir="./defect_images")

# ViT 모델 (헤드 교체)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=4,  # 양품, 불량A, 불량B, 불량C
    ignore_mismatched_sizes=True
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./vit_defect_model",
    per_device_train_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# 학습 시작
trainer.train()
```

### 실습 3: YOLOv8 Fine-tuning (60분)

**학습 내용**:
- YOLO 데이터 포맷 이해
- YOLOv8 커스텀 학습
- 실시간 불량 탐지
- 성능 메트릭 분석

**주요 코드**:
```python
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# Fine-tuning
results = model.train(
    data='defect_dataset.yaml',  # 데이터셋 설정
    epochs=100,
    imgsz=640,
    batch=16,
    name='defect_detector',
    pretrained=True,
)

# 추론
results = model.predict(
    source='test_images/',
    conf=0.5,
    save=True,
    show_labels=True,
    show_conf=True,
)

# 검증
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

---

## 💡 학습 팁

### GPU 메모리 절약

```python
# Mixed Precision Training
from transformers import TrainingArguments

training_args = TrainingArguments(
    fp16=True,  # 16-bit 학습
    gradient_accumulation_steps=2,  # Gradient 누적
    per_device_train_batch_size=8,  # Batch 크기 조정
)
```

### 데이터 증강

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

---

## 📚 참고 자료

### 논문
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [YOLOv8](https://docs.ultralytics.com/)

### 코드 & 문서
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)

---

## 🎓 학습 체크리스트

- [ ] ViT 모델의 구조를 이해했다
- [ ] ViT로 커스텀 데이터셋을 Fine-tuning했다
- [ ] YOLOv8로 불량 탐지 모델을 학습했다
- [ ] 실시간 추론을 수행하고 성능을 평가했다

---

*제조AI 교육 v12 Enhanced | Part 2-2 | 2025.02*
