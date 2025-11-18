#!/usr/bin/env python3
"""
모든 실험 통합 실행 스크립트 - GPU 병렬 처리 지원
Skin types, lighting conditions, darkness values에 대한 모든 평가를 수행합니다.
"""
import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('Run all DETR evaluations', add_help=False)
    
    # Model parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--resume', default='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                        type=str, help='Resume from checkpoint')
    parser.add_argument('--coco_path', default='/home/dohyeong/Desktop/COCO',
                        type=str, help='Path to COCO dataset')
    parser.add_argument('--output_dir', default='./results/all_experiments',
                        type=str, help='Path to save results')
    
    # Evaluation parameters
    parser.add_argument('--skin_start', default=1, type=int,
                        help='Start skin type number')
    parser.add_argument('--skin_end', default=10, type=int,
                        help='End skin type number')
    
    # Experiment selection
    parser.add_argument('--run_skin', action='store_true',
                        help='Run skin type evaluations')
    parser.add_argument('--run_lighting', action='store_true',
                        help='Run lighting condition evaluations')
    parser.add_argument('--run_darkness', action='store_true',
                        help='Run darkness value evaluations')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all experiments')
    
    # Distributed parameters
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='Number of GPUs to use')
    
    # Additional parameters
    parser.add_argument('--no_aux_loss', action='store_true',
                        help='Disable auxiliary decoding losses')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of data loading workers')
    
    return parser


def run_experiment(script_name, args, extra_args=None):
    """개별 실험 스크립트 실행"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}")
    print(f"{'='*80}\n")
    
    # 기본 인자 구성
    cmd = [
        sys.executable, script_name,
        '--batch_size', str(args.batch_size),
        '--resume', args.resume,
        '--coco_path', args.coco_path,
        '--output_dir', str(Path(args.output_dir) / Path(script_name).stem),
        '--skin_start', str(args.skin_start),
        '--skin_end', str(args.skin_end),
        '--num_gpus', str(args.num_gpus),
        '--num_workers', str(args.num_workers),
    ]
    
    if args.no_aux_loss:
        cmd.append('--no_aux_loss')
    
    # 추가 인자 병합
    if extra_args:
        cmd.extend(extra_args)
    
    # 실험 실행
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Completed {script_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed {script_name}", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        return False


def main(args):
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 실행 정보 저장
    run_info = {
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'experiments_run': [],
        'experiments_results': {}
    }
    
    # 실험 실행 여부 결정
    run_skin = args.run_skin or args.run_all
    run_lighting = args.run_lighting or args.run_all
    run_darkness = args.run_darkness or args.run_all
    
    # 아무것도 선택되지 않은 경우 모두 실행
    if not (run_skin or run_lighting or run_darkness):
        print("No specific experiments selected. Running all experiments by default.")
        run_skin = run_lighting = run_darkness = True
    
    total_experiments = 0
    successful_experiments = 0
    
    # 1. Skin Type 평가
    if run_skin:
        total_experiments += 1
        run_info['experiments_run'].append('skin_evaluations')
        success = run_experiment('run_skin_evaluations.py', args)
        run_info['experiments_results']['skin_evaluations'] = success
        if success:
            successful_experiments += 1
    
    # 2. Lighting Condition 평가
    if run_lighting:
        total_experiments += 1
        run_info['experiments_run'].append('lighting_evaluations')
        success = run_experiment('run_lighting_evaluations.py', args)
        run_info['experiments_results']['lighting_evaluations'] = success
        if success:
            successful_experiments += 1
    
    # 3. Darkness Value 평가
    if run_darkness:
        total_experiments += 1
        run_info['experiments_run'].append('darkness_evaluations')
        success = run_experiment('run_darkness_evaluations.py', args)
        run_info['experiments_results']['darkness_evaluations'] = success
        if success:
            successful_experiments += 1
    
    # 결과 요약 저장
    run_info['end_time'] = datetime.now().isoformat()
    
    summary_file = output_dir / 'all_experiments_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # 최종 결과 출력
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS SUMMARY")
    print(f"{'='*80}")
    print(f"Experiments run: {', '.join(run_info['experiments_run'])}")
    print(f"Successful: {successful_experiments}/{total_experiments}")
    print(f"\nResults saved to: {summary_file}")
    print(f"{'='*80}\n")
    
    # 개별 결과 출력
    for exp_name, success in run_info['experiments_results'].items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {exp_name}: {status}")
    
    print(f"\n{'='*80}\n")
    
    return 0 if successful_experiments == total_experiments else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run all DETR evaluations', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    sys.exit(main(args))
