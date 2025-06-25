#!/usr/bin/env python3
"""
DiPPeR 검증 자동화 스크립트
학습된 모델의 종합적 성능 평가
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time

def run_command(cmd, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"명령어: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"경고: {result.stderr}")
        
        elapsed = time.time() - start_time
        print(f"✅ 완료 ({elapsed:.1f}초)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 실행 실패: {e}")
        print(f"stderr: {e.stderr}")
        return False

def check_requirements():
    """필수 요구사항 확인"""
    print("🔍 필수 요구사항 확인 중...")
    
    required_files = [
        'robot_simuator_dippeR.py',
        'train_dipperp.py',
        'evaluate_dipperp.py',
        'evaluate_integration.py',
        'Circulation1.xml'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 누락된 파일들: {missing_files}")
        return False
    
    # Python 패키지 확인
    required_packages = ['torch', 'numpy', 'matplotlib', 'pandas', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 누락된 패키지들: {missing_packages}")
        print("다음 명령으로 설치하세요: pip install " + ' '.join(missing_packages))
        return False
    
    print("✅ 모든 요구사항 충족")
    return True

def main():
    parser = argparse.ArgumentParser(description='DiPPeR 종합 검증 실행')
    parser.add_argument('--model_path', required=True, help='학습된 DiPPeR 모델 경로')
    parser.add_argument('--xml_file', default='scenarios/Circulation1.xml', help='시뮬레이션 XML 파일')
    parser.add_argument('--scenarios', type=int, default=50, help='테스트 시나리오 수')
    parser.add_argument('--episodes', type=int, default=20, help='통합 테스트 에피소드 수')
    parser.add_argument('--skip_model_eval', action='store_true', help='모델 자체 평가 건너뛰기')
    parser.add_argument('--skip_integration', action='store_true', help='통합 평가 건너뛰기')
    
    args = parser.parse_args()
    
    # 요구사항 확인
    if not check_requirements():
        print("❌ 요구사항을 충족하지 못했습니다.")
        sys.exit(1)
    
    # 모델 파일 확인
    if not Path(args.model_path).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model_path}")
        sys.exit(1)
    
    print(f"\n🎯 DiPPeR 종합 검증 시작")
    print(f"모델: {args.model_path}")
    print(f"XML: {args.xml_file}")
    print(f"시나리오: {args.scenarios}개")
    print(f"에피소드: {args.episodes}개")
    
    # 결과 디렉토리 생성
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    
    success_count = 0
    total_tests = 0
    
    # 1. DiPPeR 모델 자체 검증
    if not args.skip_model_eval:
        total_tests += 1
        print(f"\n🔬 1단계: DiPPeR 모델 자체 검증")
        cmd = [
            sys.executable, 'evaluate_dipperp.py',
            '--model_path', args.model_path,
            '--xml_file', args.xml_file,
            '--num_scenarios', str(args.scenarios),
            '--save_dir', str(results_dir / 'model_evaluation')
        ]
        
        if run_command(cmd, "DiPPeR 모델 경로 생성 능력 평가"):
            success_count += 1
            print("📊 결과:")
            print(f"  - 시각적 비교: {results_dir}/model_evaluation/scenario_*.png")
            print(f"  - 정량적 분석: {results_dir}/model_evaluation/evaluation_report.json")
    
    # 2. 통합 시스템 검증
    if not args.skip_integration:
        total_tests += 1
        print(f"\n🚀 2단계: 통합 시스템 검증")
        cmd = [
            sys.executable, 'evaluate_integration.py',
            '--model_path', args.model_path,
            '--xml_file', args.xml_file,
            '--num_episodes', str(args.episodes),
            '--max_steps', '300',
            '--save_dir', str(results_dir / 'integration_evaluation')
        ]
        
        if run_command(cmd, "DiPPeR vs A* 통합 시스템 성능 비교"):
            success_count += 1
            print("📊 결과:")
            print(f"  - 성능 비교: {results_dir}/integration_evaluation/integration_report.json")
    
    # 3. 종합 리포트 생성
    total_tests += 1
    print(f"\n📋 3단계: 종합 리포트 생성")
    
    try:
        generate_summary_report(results_dir, args)
        success_count += 1
        print("✅ 종합 리포트 생성 완료")
    except Exception as e:
        print(f"❌ 종합 리포트 생성 실패: {e}")
    
    # 최종 결과
    print(f"\n{'='*80}")
    print(f"🏁 DiPPeR 검증 완료")
    print(f"{'='*80}")
    print(f"성공한 테스트: {success_count}/{total_tests}")
    print(f"결과 위치: {results_dir.absolute()}")
    
    if success_count == total_tests:
        print("🎉 모든 검증이 성공적으로 완료되었습니다!")
        print("\n📈 주요 결과 파일:")
        
        # 결과 파일 목록
        result_files = [
            ("모델 성능 리포트", "model_evaluation/evaluation_report.json"),
            ("통합 성능 리포트", "integration_evaluation/integration_report.json"),
            ("종합 요약", "summary_report.md")
        ]
        
        for desc, path in result_files:
            full_path = results_dir / path
            if full_path.exists():
                print(f"  - {desc}: {full_path}")
    else:
        print(f"⚠️  일부 테스트가 실패했습니다. ({total_tests - success_count}개 실패)")
        sys.exit(1)

def generate_summary_report(results_dir, args):
    """종합 요약 리포트 생성"""
    import json
    
    summary_path = results_dir / "summary_report.md"
    
    # 기본 정보
    report = f"""# DiPPeR 검증 결과 요약

## 검증 개요
- **모델 경로**: {args.model_path}
- **XML 파일**: {args.xml_file}
- **테스트 시나리오**: {args.scenarios}개
- **통합 에피소드**: {args.episodes}개
- **검증 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 검증 결과

"""
    
    # 모델 평가 결과
    model_report_path = results_dir / "model_evaluation" / "evaluation_report.json"
    if model_report_path.exists():
        try:
            with open(model_report_path, 'r') as f:
                model_data = json.load(f)
            
            overview = model_data.get('overview', {})
            performance = model_data.get('performance_comparison', {})
            
            report += f"""### 1. DiPPeR 모델 자체 성능

#### 기본 지표
- **DiPPeR 성공률**: {overview.get('dipperp_success_rate', 0):.2%}
- **A* 성공률**: {overview.get('astar_success_rate', 0):.2%}
- **평균 속도 향상**: {overview.get('average_speedup', 0):.2f}x

#### 성능 개선
- **비용 개선**: {performance.get('cost_improvement', 0):.2%}
- **길이 효율성**: {performance.get('length_efficiency', 0):.2%}
- **부드러움 개선**: {performance.get('smoothness_improvement', 0):.2%}
- **시간 개선**: {performance.get('time_improvement', 0):.2%}

"""
        except Exception as e:
            report += f"### 1. 모델 평가 결과 로드 실패: {e}\n\n"
    
    # 통합 평가 결과
    integration_report_path = results_dir / "integration_evaluation" / "integration_report.json"
    if integration_report_path.exists():
        try:
            with open(integration_report_path, 'r') as f:
                integration_data = json.load(f)
            
            overall = integration_data.get('overall_performance', {})
            
            report += f"""### 2. 통합 시스템 성능

#### End-to-End 성능 비교
- **DiPPeR 성공률**: {overall.get('dipperp_success_rate', 0):.2%}
- **A* 성공률**: {overall.get('astar_success_rate', 0):.2%}
- **DiPPeR 평균 충돌**: {overall.get('dipperp_avg_collision', 0):.2f}
- **A* 평균 충돌**: {overall.get('astar_avg_collision', 0):.2f}
- **DiPPeR 평균 시간**: {overall.get('dipperp_avg_time', 0):.2f}초
- **A* 평균 시간**: {overall.get('astar_avg_time', 0):.2f}초
- **DiPPeR 경로 효율성**: {overall.get('dipperp_path_efficiency', 0):.2f}
- **A* 경로 효율성**: {overall.get('astar_path_efficiency', 0):.2f}

#### 개선 지표
- **성공률 차이**: {(overall.get('dipperp_success_rate', 0) - overall.get('astar_success_rate', 0)) * 100:+.1f}%
- **충돌 감소율**: {(overall.get('astar_avg_collision', 0) - overall.get('dipperp_avg_collision', 0)) / max(overall.get('astar_avg_collision', 1e-6), 1e-6) * 100:+.1f}%

"""
        except Exception as e:
            report += f"### 2. 통합 평가 결과 로드 실패: {e}\n\n"
    
    report += f"""## 결론

DiPPeR 시스템의 검증이 완료되었습니다. 상세한 분석 결과는 개별 리포트 파일을 참조하세요.

### 주요 파일
- 모델 평가: `model_evaluation/evaluation_report.json`
- 통합 평가: `integration_evaluation/integration_report.json`
- 시각적 결과: `model_evaluation/scenario_*.png`

### 추천 사항
1. 성공률이 낮다면 더 많은 학습 데이터 수집 고려
2. 충돌률이 높다면 안전성 검증 강화 필요
3. 계획 시간이 느리다면 모델 경량화 검토

---
*검증 완료: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 종합 요약 리포트: {summary_path}")

if __name__ == "__main__":
    main() 