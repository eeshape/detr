#!/usr/bin/env python3
"""
실험 스크립트 통합 테스트
스크립트들이 올바르게 import되고 기본 기능이 작동하는지 확인합니다.
"""
import sys
import os
from pathlib import Path


def test_imports():
    """모든 스크립트가 정상적으로 import 되는지 테스트"""
    print("Testing imports...")
    
    try:
        import run_skin_evaluations
        print("✓ run_skin_evaluations.py imports successfully")
    except Exception as e:
        print(f"✗ run_skin_evaluations.py import failed: {e}")
        return False
    
    try:
        import run_lighting_evaluations
        print("✓ run_lighting_evaluations.py imports successfully")
    except Exception as e:
        print(f"✗ run_lighting_evaluations.py import failed: {e}")
        return False
    
    try:
        import run_darkness_evaluations
        print("✓ run_darkness_evaluations.py imports successfully")
    except Exception as e:
        print(f"✗ run_darkness_evaluations.py import failed: {e}")
        return False
    
    try:
        import run_all_experiments
        print("✓ run_all_experiments.py imports successfully")
    except Exception as e:
        print(f"✗ run_all_experiments.py import failed: {e}")
        return False
    
    try:
        import collect_results
        print("✓ collect_results.py imports successfully")
    except Exception as e:
        print(f"✗ collect_results.py import failed: {e}")
        return False
    
    return True


def test_argument_parsers():
    """각 스크립트의 argument parser가 정상적으로 작동하는지 테스트"""
    print("\nTesting argument parsers...")
    
    scripts = [
        'run_skin_evaluations',
        'run_lighting_evaluations',
        'run_darkness_evaluations',
        'run_all_experiments',
        'collect_results'
    ]
    
    for script_name in scripts:
        try:
            module = __import__(script_name)
            parser = module.get_args_parser()
            print(f"✓ {script_name}.py argument parser works")
        except Exception as e:
            print(f"✗ {script_name}.py argument parser failed: {e}")
            return False
    
    return True


def test_file_permissions():
    """실행 권한이 올바르게 설정되어 있는지 테스트"""
    print("\nTesting file permissions...")
    
    scripts = [
        'run_skin_evaluations.py',
        'run_lighting_evaluations.py',
        'run_darkness_evaluations.py',
        'run_all_experiments.py',
        'collect_results.py',
        'example_usage.py'
    ]
    
    for script in scripts:
        path = Path(script)
        if not path.exists():
            print(f"✗ {script} does not exist")
            return False
        
        if os.access(path, os.X_OK):
            print(f"✓ {script} is executable")
        else:
            print(f"⚠ {script} is not executable (but may still work)")
    
    return True


def test_documentation():
    """문서 파일이 존재하는지 테스트"""
    print("\nTesting documentation...")
    
    doc_file = Path('EXPERIMENT_GUIDE.md')
    if doc_file.exists():
        print(f"✓ EXPERIMENT_GUIDE.md exists ({doc_file.stat().st_size} bytes)")
        return True
    else:
        print("✗ EXPERIMENT_GUIDE.md not found")
        return False


def main():
    print("="*70)
    print("  실험 스크립트 통합 테스트")
    print("="*70)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test argument parsers
    results.append(("Argument Parsers", test_argument_parsers()))
    
    # Test file permissions
    results.append(("File Permissions", test_file_permissions()))
    
    # Test documentation
    results.append(("Documentation", test_documentation()))
    
    # Summary
    print("\n" + "="*70)
    print("  테스트 결과 요약")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\n총 {passed}/{total} 테스트 통과")
    print("="*70 + "\n")
    
    if passed == total:
        print("✓ 모든 테스트가 통과했습니다!")
        print("\n사용 예시를 보려면 다음 명령을 실행하세요:")
        print("  python example_usage.py")
        return 0
    else:
        print("✗ 일부 테스트가 실패했습니다.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
