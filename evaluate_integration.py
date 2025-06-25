#!/usr/bin/env python3
"""
DiPPeR 통합 시스템 검증 스크립트
End-to-End 시뮬레이션 성능 평가
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd

from robot_simuator_dippeR import RobotSimulatorDiPPeR

class IntegrationValidator:
    """DiPPeR 통합 시스템 검증 클래스"""
    
    def __init__(self, model_path, xml_file):
        self.model_path = model_path
        self.xml_file = xml_file
        
        # 결과 저장용
        self.results = {
            'dipperp_system': [],
            'astar_system': []
        }
    
    def run_comparative_evaluation(self, num_episodes=50, max_steps=500):
        """DiPPeR vs A* 시스템 비교 평가"""
        print(f"🚀 통합 시스템 비교 평가 시작 ({num_episodes}개 에피소드)")
        
        # 테스트 시나리오 생성
        scenarios = self._generate_test_scenarios(num_episodes)
        
        # DiPPeR 시스템 평가
        print("\n🔮 DiPPeR 기반 시스템 평가...")
        dipperp_results = self._evaluate_system(scenarios, use_dipperp=True, max_steps=max_steps)
        
        # A* 시스템 평가
        print("\n⭐ A* 기반 시스템 평가...")
        astar_results = self._evaluate_system(scenarios, use_dipperp=False, max_steps=max_steps)
        
        # 결과 저장
        self.results['dipperp_system'] = dipperp_results
        self.results['astar_system'] = astar_results
        
        return dipperp_results, astar_results
    
    def _generate_test_scenarios(self, num_scenarios):
        """다양한 테스트 시나리오 생성"""
        scenarios = []
        
        # 난이도별 시나리오
        difficulty_configs = [
            {'name': 'easy', 'num_agents': 5, 'agent_speed': 0.5, 'count': num_scenarios//3},
            {'name': 'medium', 'num_agents': 10, 'agent_speed': 0.8, 'count': num_scenarios//3},
            {'name': 'hard', 'num_agents': 15, 'agent_speed': 1.0, 'count': num_scenarios - 2*(num_scenarios//3)}
        ]
        
        scenario_id = 0
        for config in difficulty_configs:
            for _ in range(config['count']):
                start_pos = self._generate_safe_position()
                goal_pos = self._generate_safe_position()
                
                # 최소 거리 보장
                while np.linalg.norm(np.array(start_pos) - np.array(goal_pos)) < 3.0:
                    goal_pos = self._generate_safe_position()
                
                scenarios.append({
                    'id': scenario_id,
                    'difficulty': config['name'],
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'num_agents': config['num_agents'],
                    'agent_speed': config['agent_speed']
                })
                scenario_id += 1
        
        return scenarios
    
    def _generate_safe_position(self):
        """안전한 위치 생성"""
        safe_zones = [
            (-4.5, -0.5, -4.5, 1.5),   # 왼쪽 위
            (-4.5, -0.5, -2.0, -1.5),  # 왼쪽 아래
            (2.5, 3.5, -4.5, -0.5),    # 오른쪽 아래
            (2.5, 3.5, 1.0, 2.0),      # 오른쪽 위
            (-0.5, 1.5, 3.0, 4.5)      # 위쪽 중앙
        ]
        
        for _ in range(50):
            zone = safe_zones[np.random.randint(len(safe_zones))]
            pos = [np.random.uniform(zone[0], zone[1]), 
                   np.random.uniform(zone[2], zone[3])]
            # 간단한 안전성 체크 (실제 시뮬레이터 없이)
            if not self._is_in_obstacle(pos):
                return pos
        
        return [0.0, 0.0]  # 폴백
    
    def _is_in_obstacle(self, pos):
        """간단한 장애물 체크"""
        # Circulation1.xml의 장애물 영역 (대략적)
        obstacles = [
            (1.5, 2.5, 0.0, 1.0),     # 장애물 1
            (-3.0, -1.5, -3.0, -2.0)  # 장애물 2
        ]
        
        for obs in obstacles:
            if obs[0] <= pos[0] <= obs[1] and obs[2] <= pos[1] <= obs[3]:
                return True
        return False
    
    def _evaluate_system(self, scenarios, use_dipperp, max_steps):
        """시스템 성능 평가"""
        results = []
        
        system_name = "DiPPeR" if use_dipperp else "A*"
        
        for scenario in tqdm(scenarios, desc=f"{system_name} 평가"):
            # 시뮬레이터 초기화
            if use_dipperp and Path(self.model_path).exists():
                simulator = RobotSimulatorDiPPeR(self.xml_file, model_path=self.model_path)
                simulator.use_dipperp = True
            else:
                simulator = RobotSimulatorDiPPeR(self.xml_file, model_path=None)
                simulator.use_dipperp = False
            
            # 시나리오 설정
            simulator.robot_pos = scenario['start_pos'].copy()
            simulator.target_pos = scenario['goal_pos'].copy()
            
            # 평가 실행
            episode_result = self._run_single_episode(
                simulator, scenario, max_steps, use_dipperp
            )
            
            results.append(episode_result)
        
        return results
    
    def _run_single_episode(self, simulator, scenario, max_steps, use_dipperp):
        """단일 에피소드 실행 및 평가"""
        start_time = time.time()
        
        # 성능 지표 초기화
        metrics = {
            'scenario_id': scenario['id'],
            'difficulty': scenario['difficulty'],
            'system_type': 'DiPPeR' if use_dipperp else 'A*',
            'success': False,
            'collision_count': 0,
            'total_steps': 0,
            'total_time': 0,
            'total_distance': 0,
            'average_speed': 0,
            'path_efficiency': 0,
            'planning_time_total': 0,
            'planning_calls': 0,
            'stuck_episodes': 0
        }
        
        # 초기 위치 기록
        prev_pos = simulator.robot_pos.copy()
        path_points = [prev_pos.copy()]
        planning_times = []
        
        # 시뮬레이션 실행
        for step in range(max_steps):
            step_start = time.time()
            
            # 한 스텝 실행
            simulator.update()
            
            # 위치 기록
            current_pos = simulator.robot_pos.copy()
            path_points.append(current_pos)
            
            # 거리 계산
            step_distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
            metrics['total_distance'] += step_distance
            prev_pos = current_pos
            
            # 목표 도달 체크
            goal_distance = np.linalg.norm(np.array(current_pos) - np.array(scenario['goal_pos']))
            if goal_distance < 0.5:  # 목표 도달
                metrics['success'] = True
                break
            
            # 정체 감지 (최근 10스텝 동안 0.1m 미만 이동)
            if step >= 10:
                recent_movement = np.linalg.norm(
                    np.array(path_points[-1]) - np.array(path_points[-10])
                )
                if recent_movement < 0.1:
                    metrics['stuck_episodes'] += 1
                    if metrics['stuck_episodes'] > 5:  # 연속 정체
                        break
            
            metrics['total_steps'] = step + 1
        
        # 최종 지표 계산
        metrics['total_time'] = time.time() - start_time
        metrics['average_speed'] = metrics['total_distance'] / max(metrics['total_time'], 1e-6)
        
        # 경로 효율성 (직선 거리 대비 실제 이동 거리)
        straight_distance = np.linalg.norm(
            np.array(scenario['goal_pos']) - np.array(scenario['start_pos'])
        )
        metrics['path_efficiency'] = straight_distance / max(metrics['total_distance'], 1e-6)
        
        return metrics
    
    def generate_comparison_report(self, save_path="integration_report.json"):
        """비교 분석 리포트 생성"""
        print("📊 통합 시스템 비교 리포트 생성 중...")
        
        dipperp_df = pd.DataFrame(self.results['dipperp_system'])
        astar_df = pd.DataFrame(self.results['astar_system'])
        
        # 전체 성능 비교
        comparison = {
            'overall_performance': {
                'dipperp_success_rate': dipperp_df['success'].mean(),
                'astar_success_rate': astar_df['success'].mean(),
                'dipperp_avg_collision': dipperp_df['collision_count'].mean(),
                'astar_avg_collision': astar_df['collision_count'].mean(),
                'dipperp_avg_time': dipperp_df['total_time'].mean(),
                'astar_avg_time': astar_df['total_time'].mean(),
                'dipperp_avg_speed': dipperp_df['average_speed'].mean(),
                'astar_avg_speed': astar_df['average_speed'].mean(),
                'dipperp_path_efficiency': dipperp_df['path_efficiency'].mean(),
                'astar_path_efficiency': astar_df['path_efficiency'].mean()
            }
        }
        
        # 난이도별 비교
        comparison['difficulty_analysis'] = {}
        for difficulty in ['easy', 'medium', 'hard']:
            dipperp_subset = dipperp_df[dipperp_df['difficulty'] == difficulty]
            astar_subset = astar_df[astar_df['difficulty'] == difficulty]
            
            if len(dipperp_subset) > 0 and len(astar_subset) > 0:
                comparison['difficulty_analysis'][difficulty] = {
                    'dipperp_success_rate': dipperp_subset['success'].mean(),
                    'astar_success_rate': astar_subset['success'].mean(),
                    'dipperp_avg_collision': dipperp_subset['collision_count'].mean(),
                    'astar_avg_collision': astar_subset['collision_count'].mean(),
                    'dipperp_avg_time': dipperp_subset['total_time'].mean(),
                    'astar_avg_time': astar_subset['total_time'].mean()
                }
        
        # 상세 결과 포함
        comparison['detailed_results'] = {
            'dipperp_system': self.results['dipperp_system'],
            'astar_system': self.results['astar_system']
        }
        
        # JSON 저장
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # 콘솔 출력
        self._print_comparison_summary(comparison)
        
        print(f"✅ 상세 리포트 저장: {save_path}")
        return comparison
    
    def _print_comparison_summary(self, comparison):
        """비교 결과 요약 출력"""
        overall = comparison['overall_performance']
        
        print("\n" + "="*80)
        print("🏆 DiPPeR vs A* 통합 시스템 성능 비교")
        print("="*80)
        
        print(f"📈 전체 성능 비교:")
        print(f"  성공률:      DiPPeR {overall['dipperp_success_rate']:.2%} vs A* {overall['astar_success_rate']:.2%}")
        print(f"  평균 충돌:   DiPPeR {overall['dipperp_avg_collision']:.2f} vs A* {overall['astar_avg_collision']:.2f}")
        print(f"  평균 시간:   DiPPeR {overall['dipperp_avg_time']:.2f}s vs A* {overall['astar_avg_time']:.2f}s")
        print(f"  평균 속도:   DiPPeR {overall['dipperp_avg_speed']:.2f} vs A* {overall['astar_avg_speed']:.2f}")
        print(f"  경로 효율성: DiPPeR {overall['dipperp_path_efficiency']:.2f} vs A* {overall['astar_path_efficiency']:.2f}")
        
        # 개선 지표
        success_improvement = (overall['dipperp_success_rate'] - overall['astar_success_rate']) * 100
        collision_reduction = (overall['astar_avg_collision'] - overall['dipperp_avg_collision']) / max(overall['astar_avg_collision'], 1e-6) * 100
        
        print(f"\n🎯 핵심 개선 지표:")
        print(f"  성공률 개선:   {success_improvement:+.1f}%")
        print(f"  충돌 감소:     {collision_reduction:+.1f}%")
        
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description='DiPPeR 통합 시스템 검증')
    parser.add_argument('--model_path', required=True, help='학습된 DiPPeR 모델 경로')
    parser.add_argument('--xml_file', default='Circulation1.xml', help='시뮬레이션 XML 파일')
    parser.add_argument('--num_episodes', type=int, default=30, help='테스트 에피소드 수')
    parser.add_argument('--max_steps', type=int, default=500, help='에피소드당 최대 스텝')
    parser.add_argument('--save_dir', default='integration_results', help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 검증 실행
    validator = IntegrationValidator(args.model_path, args.xml_file)
    
    # 비교 평가
    dipperp_results, astar_results = validator.run_comparative_evaluation(
        args.num_episodes, args.max_steps
    )
    
    # 리포트 생성
    Path(args.save_dir).mkdir(exist_ok=True)
    validator.generate_comparison_report(f"{args.save_dir}/integration_report.json")

if __name__ == "__main__":
    main() 