#!/usr/bin/env python3
"""
실험 결과 수집 및 분석 스크립트
평가 결과를 수집하고 시각화 데이터를 생성합니다.
"""
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import re


def get_args_parser():
    parser = argparse.ArgumentParser('Collect and analyze evaluation results', add_help=False)
    
    parser.add_argument('--results_dir', default='./results',
                        type=str, help='Directory containing evaluation results')
    parser.add_argument('--output_file', default='./results/analysis_summary.json',
                        type=str, help='Output file for analysis')
    parser.add_argument('--export_csv', action='store_true',
                        help='Export results to CSV files')
    
    return parser


def extract_metrics_from_output(output_text: str) -> Dict[str, float]:
    """출력 텍스트에서 메트릭 추출"""
    metrics = {}
    
    # AP, AR 등의 메트릭 추출 패턴
    patterns = {
        'AP': r'Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95.*?\]\s*=\s*([\d.]+)',
        'AP50': r'Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50.*?\]\s*=\s*([\d.]+)',
        'AP75': r'Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75.*?\]\s*=\s*([\d.]+)',
        'AR': r'Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95.*?maxDets=\s*100\s*\]\s*=\s*([\d.]+)',
        'mAR': r'Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95.*?maxDets=\s*100\s*\]\s*=\s*([\d.]+)',
    }
    
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match:
            metrics[metric_name] = float(match.group(1))
    
    return metrics


def collect_results(results_dir: Path) -> Dict[str, Any]:
    """결과 디렉토리에서 모든 평가 결과 수집"""
    all_results = {
        'skin_evaluations': {},
        'lighting_evaluations': {},
        'darkness_evaluations': {}
    }
    
    # Skin evaluations
    skin_dir = results_dir / 'run_skin_evaluations'
    if skin_dir.exists():
        summary_file = skin_dir / 'evaluation_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                for skin_type, result in data.get('results', {}).items():
                    if result['success']:
                        metrics = extract_metrics_from_output(result.get('output', ''))
                        all_results['skin_evaluations'][skin_type] = metrics
    
    # Lighting evaluations
    lighting_dir = results_dir / 'run_lighting_evaluations'
    if lighting_dir.exists():
        summary_file = lighting_dir / 'lighting_evaluation_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                for lighting, skin_results in data.get('results', {}).items():
                    all_results['lighting_evaluations'][lighting] = {}
                    for skin_type, result in skin_results.items():
                        if result['success']:
                            metrics = extract_metrics_from_output(result.get('output', ''))
                            all_results['lighting_evaluations'][lighting][skin_type] = metrics
    
    # Darkness evaluations
    darkness_dir = results_dir / 'run_darkness_evaluations'
    if darkness_dir.exists():
        summary_file = darkness_dir / 'darkness_evaluation_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                for darkness, skin_results in data.get('results', {}).items():
                    all_results['darkness_evaluations'][darkness] = {}
                    for skin_type, result in skin_results.items():
                        if result['success']:
                            metrics = extract_metrics_from_output(result.get('output', ''))
                            all_results['darkness_evaluations'][darkness][skin_type] = metrics
    
    return all_results


def export_to_csv(results: Dict[str, Any], output_dir: Path):
    """결과를 CSV 파일로 내보내기"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Skin evaluations CSV
    if results['skin_evaluations']:
        csv_file = output_dir / 'skin_evaluations.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 헤더
            header = ['skin_type']
            if results['skin_evaluations']:
                sample_metrics = list(results['skin_evaluations'].values())[0]
                header.extend(sample_metrics.keys())
            writer.writerow(header)
            
            # 데이터
            for skin_type in sorted(results['skin_evaluations'].keys()):
                metrics = results['skin_evaluations'][skin_type]
                row = [skin_type] + [metrics.get(k, '') for k in header[1:]]
                writer.writerow(row)
        print(f"Exported: {csv_file}")
    
    # Lighting evaluations CSV
    if results['lighting_evaluations']:
        csv_file = output_dir / 'lighting_evaluations.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 헤더
            header = ['lighting', 'skin_type']
            sample_data = None
            for lighting_data in results['lighting_evaluations'].values():
                if lighting_data:
                    sample_data = list(lighting_data.values())[0]
                    break
            if sample_data:
                header.extend(sample_data.keys())
            writer.writerow(header)
            
            # 데이터
            for lighting in sorted(results['lighting_evaluations'].keys()):
                for skin_type in sorted(results['lighting_evaluations'][lighting].keys()):
                    metrics = results['lighting_evaluations'][lighting][skin_type]
                    row = [lighting, skin_type] + [metrics.get(k, '') for k in header[2:]]
                    writer.writerow(row)
        print(f"Exported: {csv_file}")
    
    # Darkness evaluations CSV
    if results['darkness_evaluations']:
        csv_file = output_dir / 'darkness_evaluations.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # 헤더
            header = ['darkness', 'skin_type']
            sample_data = None
            for darkness_data in results['darkness_evaluations'].values():
                if darkness_data:
                    sample_data = list(darkness_data.values())[0]
                    break
            if sample_data:
                header.extend(sample_data.keys())
            writer.writerow(header)
            
            # 데이터
            for darkness in sorted(results['darkness_evaluations'].keys()):
                for skin_type in sorted(results['darkness_evaluations'][darkness].keys()):
                    metrics = results['darkness_evaluations'][darkness][skin_type]
                    row = [darkness, skin_type] + [metrics.get(k, '') for k in header[2:]]
                    writer.writerow(row)
        print(f"Exported: {csv_file}")


def calculate_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """통계 계산"""
    stats = {}
    
    # Skin type별 평균
    if results['skin_evaluations']:
        skin_stats = {}
        for metric in ['AP', 'AP50', 'AP75', 'AR', 'mAR']:
            values = [m.get(metric, 0) for m in results['skin_evaluations'].values() if metric in m]
            if values:
                skin_stats[f'{metric}_mean'] = sum(values) / len(values)
                skin_stats[f'{metric}_min'] = min(values)
                skin_stats[f'{metric}_max'] = max(values)
        stats['skin_evaluations'] = skin_stats
    
    # Lighting 조건별 평균
    if results['lighting_evaluations']:
        lighting_stats = {}
        for lighting, skin_data in results['lighting_evaluations'].items():
            lighting_stats[lighting] = {}
            for metric in ['AP', 'AP50', 'AP75', 'AR', 'mAR']:
                values = [m.get(metric, 0) for m in skin_data.values() if metric in m]
                if values:
                    lighting_stats[lighting][f'{metric}_mean'] = sum(values) / len(values)
        stats['lighting_evaluations'] = lighting_stats
    
    # Darkness별 평균
    if results['darkness_evaluations']:
        darkness_stats = {}
        for darkness, skin_data in results['darkness_evaluations'].items():
            darkness_stats[darkness] = {}
            for metric in ['AP', 'AP50', 'AP75', 'AR', 'mAR']:
                values = [m.get(metric, 0) for m in skin_data.values() if metric in m]
                if values:
                    darkness_stats[darkness][f'{metric}_mean'] = sum(values) / len(values)
        stats['darkness_evaluations'] = darkness_stats
    
    return stats


def main(args):
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print("Collecting evaluation results...")
    results = collect_results(results_dir)
    
    print("Calculating statistics...")
    stats = calculate_statistics(results)
    
    # 분석 결과 저장
    analysis = {
        'results': results,
        'statistics': stats
    }
    
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nAnalysis saved to: {output_file}")
    
    # CSV 내보내기
    if args.export_csv:
        print("\nExporting to CSV...")
        export_to_csv(results, output_file.parent / 'csv')
    
    # 요약 출력
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if stats.get('skin_evaluations'):
        print("\nSkin Type Evaluations:")
        for metric, value in stats['skin_evaluations'].items():
            print(f"  {metric}: {value:.4f}")
    
    if stats.get('lighting_evaluations'):
        print("\nLighting Condition Evaluations:")
        for lighting, metrics in stats['lighting_evaluations'].items():
            print(f"  {lighting}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    if stats.get('darkness_evaluations'):
        print("\nDarkness Value Evaluations:")
        for darkness, metrics in stats['darkness_evaluations'].items():
            print(f"  {darkness}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    print(f"{'='*60}\n")
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collect and analyze evaluation results', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    exit(main(args))
