# DETR 실험 자동화 스크립트

GPU 병렬 처리를 지원하는 DETR 평가 자동화 스크립트 모음입니다.

## 📋 개요

이 스크립트들은 FACET 데이터셋을 사용하여 다양한 조건(skin type, lighting, darkness)에서 DETR 모델의 성능을 평가하기 위해 작성되었습니다.

## 🚀 주요 특징

- ✅ **GPU 병렬 처리**: `torch.distributed.launch`를 사용한 멀티 GPU 지원
- ✅ **Python 직접 실행**: Shell script 없이 Python 파일만으로 실행 가능
- ✅ **자동 결과 수집**: JSON 및 CSV 형식으로 결과 자동 저장
- ✅ **유연한 설정**: 커맨드라인 인자를 통한 세밀한 제어

## 📁 스크립트 구성

### 1. `run_skin_evaluations.py`
모든 skin type (skin1-skin10)에 대해 평가를 수행합니다.

**사용 예시:**
```bash
# 단일 GPU
python run_skin_evaluations.py \
    --batch_size 16 \
    --coco_path /path/to/COCO \
    --output_dir ./results/skin_eval

# 멀티 GPU (4개)
python run_skin_evaluations.py \
    --batch_size 16 \
    --num_gpus 4 \
    --coco_path /path/to/COCO \
    --output_dir ./results/skin_eval \
    --no_aux_loss
```

**주요 인자:**
- `--skin_start`: 시작 skin type 번호 (기본값: 1)
- `--skin_end`: 종료 skin type 번호 (기본값: 10)
- `--num_gpus`: 사용할 GPU 개수 (기본값: 1)
- `--batch_size`: GPU당 배치 크기 (기본값: 16)

### 2. `run_lighting_evaluations.py`
Lighting 조건(well, dimly)별로 모든 skin type을 평가합니다.

**사용 예시:**
```bash
# Well과 Dimly 조건 모두 평가
python run_lighting_evaluations.py \
    --batch_size 16 \
    --num_gpus 2 \
    --coco_path /path/to/COCO \
    --output_dir ./results/lighting_eval

# Well 조건만 평가
python run_lighting_evaluations.py \
    --batch_size 16 \
    --lighting_conditions well \
    --coco_path /path/to/COCO
```

**주요 인자:**
- `--lighting_conditions`: 평가할 조건 리스트 (기본값: well, dimly)

### 3. `run_darkness_evaluations.py`
Darkness 값(0.1, 0.5, 1.0)별로 모든 skin type을 평가합니다.

**사용 예시:**
```bash
# 기본 darkness 값들로 평가
python run_darkness_evaluations.py \
    --batch_size 16 \
    --num_gpus 2 \
    --coco_path /path/to/COCO \
    --output_dir ./results/darkness_eval

# 커스텀 darkness 값들로 평가
python run_darkness_evaluations.py \
    --batch_size 16 \
    --darkness_values 0.1 0.3 0.5 0.7 1.0 \
    --coco_path /path/to/COCO
```

**주요 인자:**
- `--darkness_values`: 평가할 darkness 값 리스트 (기본값: 0.1 0.5 1.0)

### 4. `run_all_experiments.py`
모든 실험을 한 번에 실행합니다.

**사용 예시:**
```bash
# 모든 실험 실행
python run_all_experiments.py \
    --batch_size 16 \
    --num_gpus 4 \
    --coco_path /path/to/COCO \
    --output_dir ./results/all_experiments \
    --run_all

# 특정 실험만 선택적으로 실행
python run_all_experiments.py \
    --batch_size 16 \
    --run_skin \
    --run_lighting \
    --coco_path /path/to/COCO
```

**주요 인자:**
- `--run_all`: 모든 실험 실행
- `--run_skin`: Skin type 실험만 실행
- `--run_lighting`: Lighting 실험만 실행
- `--run_darkness`: Darkness 실험만 실행

### 5. `collect_results.py`
실험 결과를 수집하고 분석합니다.

**사용 예시:**
```bash
# 결과 수집 및 분석
python collect_results.py \
    --results_dir ./results \
    --output_file ./results/analysis_summary.json \
    --export_csv
```

**주요 인자:**
- `--results_dir`: 결과가 저장된 디렉토리
- `--export_csv`: CSV 파일로도 내보내기

## 💡 사용 가이드

### 기본 워크플로우

1. **단일 실험 실행**
   ```bash
   python run_skin_evaluations.py \
       --batch_size 16 \
       --num_gpus 2 \
       --coco_path /path/to/COCO \
       --output_dir ./results/skin_eval
   ```

2. **모든 실험 실행**
   ```bash
   python run_all_experiments.py \
       --batch_size 16 \
       --num_gpus 4 \
       --coco_path /path/to/COCO \
       --run_all
   ```

3. **결과 수집 및 분석**
   ```bash
   python collect_results.py \
       --results_dir ./results \
       --export_csv
   ```

### GPU 설정

스크립트는 자동으로 GPU 병렬 처리를 설정합니다:

- **단일 GPU** (`--num_gpus 1`): 기본 실행
- **멀티 GPU** (`--num_gpus N`): `torch.distributed.launch` 자동 사용

환경 변수 설정이 필요 없으며, 스크립트가 자동으로 처리합니다.

### 출력 구조

```
results/
├── run_skin_evaluations/
│   ├── evaluation_summary.json
│   ├── skin1/
│   ├── skin2/
│   └── ...
├── run_lighting_evaluations/
│   ├── lighting_evaluation_summary.json
│   ├── well/
│   │   ├── skin1/
│   │   └── ...
│   └── dimly/
│       ├── skin1/
│       └── ...
├── run_darkness_evaluations/
│   ├── darkness_evaluation_summary.json
│   ├── darkness_0.1/
│   ├── darkness_0.5/
│   └── darkness_1.0/
├── analysis_summary.json
└── csv/
    ├── skin_evaluations.csv
    ├── lighting_evaluations.csv
    └── darkness_evaluations.csv
```

## 🔧 고급 설정

### 커스텀 모델 사용

```bash
python run_skin_evaluations.py \
    --resume /path/to/your/checkpoint.pth \
    --batch_size 8 \
    --coco_path /path/to/COCO
```

### 특정 skin type 범위만 평가

```bash
python run_skin_evaluations.py \
    --skin_start 3 \
    --skin_end 7 \
    --batch_size 16 \
    --coco_path /path/to/COCO
```

### 배치 크기 조정 (메모리 부족 시)

```bash
python run_all_experiments.py \
    --batch_size 8 \
    --num_gpus 2 \
    --coco_path /path/to/COCO
```

## 📊 결과 분석

`collect_results.py`는 다음 정보를 제공합니다:

- **메트릭**: AP, AP50, AP75, AR, mAR
- **통계**: 평균, 최소, 최대값
- **형식**: JSON, CSV

### JSON 출력 예시

```json
{
  "results": {
    "skin_evaluations": {
      "skin1": {"AP": 0.42, "AR": 0.55, ...},
      "skin2": {"AP": 0.41, "AR": 0.54, ...}
    }
  },
  "statistics": {
    "skin_evaluations": {
      "AP_mean": 0.415,
      "AP_min": 0.40,
      "AP_max": 0.43
    }
  }
}
```

## 🐛 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
python run_skin_evaluations.py --batch_size 8

# 또는 GPU 수 늘리기
python run_skin_evaluations.py --num_gpus 4 --batch_size 4
```

### 특정 실험만 재실행
```bash
# 실패한 실험만 다시 실행
python run_lighting_evaluations.py \
    --lighting_conditions dimly \
    --skin_start 5 \
    --skin_end 10
```

## 📝 기존 Shell Script와의 비교

### 기존 방식 (Shell Script)
```bash
bash run_all_skins.sh  # GPU 병렬 처리 없음
```

### 새로운 방식 (Python)
```bash
# GPU 병렬 처리 자동 지원
python run_skin_evaluations.py --num_gpus 4
```

**장점:**
- ✅ GPU 병렬 처리 자동 설정
- ✅ 결과 자동 수집 및 저장
- ✅ 유연한 설정 옵션
- ✅ 에러 처리 및 재시작 지원
- ✅ 크로스 플랫폼 호환성

## 🔗 관련 파일

- `main.py`: DETR 메인 학습/평가 스크립트
- `engine.py`: 학습/평가 엔진
- `AGENTS.md`: 프로젝트 가이드라인
- `EMBEDDING_GUIDE.md`: 임베딩 추출 가이드

## 📄 라이선스

이 스크립트들은 DETR 프로젝트의 라이선스를 따릅니다.
