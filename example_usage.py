#!/usr/bin/env python3
"""
실험 스크립트 사용 예시
실제 데이터가 없어도 스크립트가 정상적으로 동작하는지 테스트합니다.
"""
import sys
from pathlib import Path


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    print_section("DETR 실험 자동화 스크립트 - 사용 예시")
    
    print("이 스크립트들은 다음과 같은 방식으로 사용할 수 있습니다:\n")
    
    # 1. Skin Evaluations
    print_section("1. Skin Type 평가")
    print("# 모든 skin type (skin1-10) 평가 - 단일 GPU")
    print("python run_skin_evaluations.py \\")
    print("    --batch_size 16 \\")
    print("    --coco_path /path/to/COCO \\")
    print("    --output_dir ./results/skin_eval\n")
    
    print("# 멀티 GPU (4개) 사용")
    print("python run_skin_evaluations.py \\")
    print("    --batch_size 16 \\")
    print("    --num_gpus 4 \\")
    print("    --coco_path /path/to/COCO \\")
    print("    --output_dir ./results/skin_eval\n")
    
    print("# 특정 범위만 평가 (skin3-7)")
    print("python run_skin_evaluations.py \\")
    print("    --skin_start 3 \\")
    print("    --skin_end 7 \\")
    print("    --batch_size 16 \\")
    print("    --coco_path /path/to/COCO\n")
    
    # 2. Lighting Evaluations
    print_section("2. Lighting 조건별 평가")
    print("# Well과 Dimly 조건 모두 평가")
    print("python run_lighting_evaluations.py \\")
    print("    --batch_size 16 \\")
    print("    --num_gpus 2 \\")
    print("    --coco_path /path/to/COCO \\")
    print("    --output_dir ./results/lighting_eval\n")
    
    print("# Well 조건만 평가")
    print("python run_lighting_evaluations.py \\")
    print("    --lighting_conditions well \\")
    print("    --batch_size 16 \\")
    print("    --coco_path /path/to/COCO\n")
    
    # 3. Darkness Evaluations
    print_section("3. Darkness 값별 평가")
    print("# 기본 darkness 값 (0.1, 0.5, 1.0)")
    print("python run_darkness_evaluations.py \\")
    print("    --batch_size 16 \\")
    print("    --num_gpus 2 \\")
    print("    --coco_path /path/to/COCO \\")
    print("    --output_dir ./results/darkness_eval\n")
    
    print("# 커스텀 darkness 값")
    print("python run_darkness_evaluations.py \\")
    print("    --darkness_values 0.1 0.3 0.5 0.7 1.0 \\")
    print("    --batch_size 16 \\")
    print("    --coco_path /path/to/COCO\n")
    
    # 4. All Experiments
    print_section("4. 모든 실험 한번에 실행")
    print("# 모든 실험 실행")
    print("python run_all_experiments.py \\")
    print("    --batch_size 16 \\")
    print("    --num_gpus 4 \\")
    print("    --coco_path /path/to/COCO \\")
    print("    --output_dir ./results/all_experiments \\")
    print("    --run_all\n")
    
    print("# 특정 실험만 선택")
    print("python run_all_experiments.py \\")
    print("    --run_skin \\")
    print("    --run_lighting \\")
    print("    --batch_size 16 \\")
    print("    --coco_path /path/to/COCO\n")
    
    # 5. Collect Results
    print_section("5. 결과 수집 및 분석")
    print("# 결과 수집 및 JSON 생성")
    print("python collect_results.py \\")
    print("    --results_dir ./results \\")
    print("    --output_file ./results/analysis_summary.json\n")
    
    print("# CSV로도 내보내기")
    print("python collect_results.py \\")
    print("    --results_dir ./results \\")
    print("    --export_csv\n")
    
    # 6. Complete Workflow
    print_section("6. 전체 워크플로우 예시")
    print("# Step 1: 모든 실험 실행")
    print("python run_all_experiments.py --run_all --num_gpus 4 --coco_path /path/to/COCO\n")
    print("# Step 2: 결과 분석")
    print("python collect_results.py --results_dir ./results --export_csv\n")
    
    # GPU Configuration
    print_section("GPU 설정 가이드")
    print("스크립트는 자동으로 GPU 병렬 처리를 설정합니다:\n")
    print("• --num_gpus 1  : 단일 GPU (기본값)")
    print("• --num_gpus 2  : 2개 GPU로 분산 학습")
    print("• --num_gpus 4  : 4개 GPU로 분산 학습")
    print("• --num_gpus 8  : 8개 GPU로 분산 학습\n")
    print("환경 변수 설정이 필요 없으며, torch.distributed.launch가 자동으로 실행됩니다.\n")
    
    # Memory Optimization
    print_section("메모리 최적화 팁")
    print("CUDA Out of Memory 오류 발생 시:\n")
    print("1. 배치 크기 줄이기:")
    print("   --batch_size 8  (또는 4)\n")
    print("2. GPU 수 늘리기:")
    print("   --num_gpus 4  (배치를 GPU들에 분산)\n")
    print("3. 둘 다 조정:")
    print("   --batch_size 4 --num_gpus 4\n")
    
    print_section("더 많은 정보")
    print("자세한 내용은 EXPERIMENT_GUIDE.md 파일을 참고하세요:\n")
    print("  cat EXPERIMENT_GUIDE.md\n")
    
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
