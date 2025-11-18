#!/usr/bin/env python3
"""
Skin Type 평가 스크립트 - GPU 병렬 처리 지원
모든 skin type (skin1-skin10)에 대해 DETR 모델 평가를 수행합니다.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('Run DETR evaluation for all skin types', add_help=False)
    
    # Model parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--resume', default='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                        type=str, help='Resume from checkpoint')
    parser.add_argument('--coco_path', default='/home/dohyeong/Desktop/COCO',
                        type=str, help='Path to COCO dataset')
    parser.add_argument('--output_dir', default='./results/skin_evaluation',
                        type=str, help='Path to save results')
    
    # Evaluation parameters
    parser.add_argument('--skin_start', default=1, type=int,
                        help='Start skin type number')
    parser.add_argument('--skin_end', default=10, type=int,
                        help='End skin type number')
    parser.add_argument('--light', default=None, type=str,
                        help='Lighting condition (well, dimly, or None for all)')
    
    # Distributed parameters
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='Number of GPUs to use')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    
    # Additional parameters to pass to main.py
    parser.add_argument('--no_aux_loss', action='store_true',
                        help='Disable auxiliary decoding losses')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of data loading workers')
    
    return parser


def run_evaluation(skin_type, args):
    """단일 skin type에 대해 평가 실행"""
    print(f"\n{'='*60}")
    print(f"Starting evaluation for {skin_type}")
    print(f"{'='*60}\n")
    
    # main.py에 전달할 인자 구성
    cmd = [
        sys.executable, 'main.py',
        '--batch_size', str(args.batch_size),
        '--eval',
        '--resume', args.resume,
        '--coco_path', args.coco_path,
        '--skin', skin_type,
        '--output_dir', str(Path(args.output_dir) / skin_type),
        '--num_workers', str(args.num_workers),
    ]
    
    if args.no_aux_loss:
        cmd.append('--no_aux_loss')
    
    if args.light:
        cmd.extend(['--light', args.light])
    
    # GPU 병렬 처리 설정
    if args.num_gpus > 1:
        # torch.distributed.launch 사용
        launch_cmd = [
            sys.executable, '-m', 'torch.distributed.launch',
            f'--nproc_per_node={args.num_gpus}',
            '--use_env'
        ] + cmd[1:]  # python을 제외한 나머지
        cmd = launch_cmd
    
    # 평가 실행
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}", file=sys.stderr)
        
        print(f"\n✓ Completed evaluation for {skin_type}\n")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed evaluation for {skin_type}", file=sys.stderr)
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
    
    # 모든 skin type에 대해 평가 실행
    results = {}
    for i in range(args.skin_start, args.skin_end + 1):
        skin_type = f'skin{i}'
        success, output = run_evaluation(skin_type, args)
        results[skin_type] = {
            'success': success,
            'output': output
        }
    
    # 결과 요약 저장
    run_info['end_time'] = datetime.now().isoformat()
    run_info['results'] = results
    
    summary_file = output_dir / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"Successful: {successful}/{total}")
    print(f"Results saved to: {summary_file}")
    print(f"{'='*60}\n")
    
    return 0 if successful == total else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run DETR evaluation for all skin types', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    sys.exit(main(args))
