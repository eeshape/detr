#!/usr/bin/env python3
"""
Darkness 값별 평가 스크립트 - GPU 병렬 처리 지원
다양한 darkness 값(0.1, 0.5, 1.0)에 대해 모든 skin type 평가를 수행합니다.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('Run DETR evaluation for darkness values', add_help=False)
    
    # Model parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--resume', default='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                        type=str, help='Resume from checkpoint')
    parser.add_argument('--coco_path', default='/home/dohyeong/Desktop/COCO',
                        type=str, help='Path to COCO dataset')
    parser.add_argument('--output_dir', default='./results/darkness_evaluation',
                        type=str, help='Path to save results')
    
    # Evaluation parameters
    parser.add_argument('--skin_start', default=1, type=int,
                        help='Start skin type number')
    parser.add_argument('--skin_end', default=10, type=int,
                        help='End skin type number')
    parser.add_argument('--darkness_values', nargs='+', type=float,
                        default=[0.1, 0.5, 1.0],
                        help='Darkness values to evaluate')
    
    # Distributed parameters
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='Number of GPUs to use')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    
    # Additional parameters
    parser.add_argument('--no_aux_loss', action='store_true',
                        help='Disable auxiliary decoding losses')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of data loading workers')
    
    return parser


def run_evaluation(skin_type, darkness_value, args):
    """단일 skin type과 darkness 값에 대해 평가 실행"""
    print(f"\n{'='*60}")
    print(f"Evaluating {skin_type} with darkness={darkness_value}")
    print(f"{'='*60}\n")
    
    # 출력 디렉토리 설정
    darkness_dir = f"darkness_{darkness_value:.1f}"
    output_subdir = Path(args.output_dir) / darkness_dir / skin_type
    
    # main.py에 전달할 인자 구성
    cmd = [
        sys.executable, 'main.py',
        '--batch_size', str(args.batch_size),
        '--eval',
        '--resume', args.resume,
        '--coco_path', args.coco_path,
        '--skin', skin_type,
        '--darkness', str(darkness_value),
        '--output_dir', str(output_subdir),
        '--num_workers', str(args.num_workers),
    ]
    
    if args.no_aux_loss:
        cmd.append('--no_aux_loss')
    
    # GPU 병렬 처리 설정
    if args.num_gpus > 1:
        launch_cmd = [
            sys.executable, '-m', 'torch.distributed.launch',
            f'--nproc_per_node={args.num_gpus}',
            '--use_env'
        ] + cmd[1:]
        cmd = launch_cmd
    
    # 평가 실행
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}", file=sys.stderr)
        
        print(f"\n✓ Completed: {skin_type} - darkness={darkness_value}\n")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {skin_type} - darkness={darkness_value}", file=sys.stderr)
        print(f"Error: {e.stderr}", file=sys.stderr)
        return False, str(e)


def main(args):
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 실행 정보 저장
    run_info = {
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'results': {}
    }
    
    # 모든 조합에 대해 평가 실행
    total_evaluations = 0
    successful_evaluations = 0
    
    for darkness_value in args.darkness_values:
        darkness_key = f"darkness_{darkness_value:.1f}"
        run_info['results'][darkness_key] = {}
        
        for i in range(args.skin_start, args.skin_end + 1):
            skin_type = f'skin{i}'
            total_evaluations += 1
            
            success, output = run_evaluation(skin_type, darkness_value, args)
            run_info['results'][darkness_key][skin_type] = {
                'success': success,
                'output': output
            }
            
            if success:
                successful_evaluations += 1
    
    # 결과 요약 저장
    run_info['end_time'] = datetime.now().isoformat()
    
    summary_file = output_dir / 'darkness_evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("Darkness Evaluation Summary")
    print(f"{'='*60}")
    print(f"Darkness values: {', '.join(map(str, args.darkness_values))}")
    print(f"Skin types: {args.skin_start}-{args.skin_end}")
    print(f"Successful: {successful_evaluations}/{total_evaluations}")
    print(f"Results saved to: {summary_file}")
    print(f"{'='*60}\n")
    
    return 0 if successful_evaluations == total_evaluations else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run DETR evaluation for darkness values', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    sys.exit(main(args))
