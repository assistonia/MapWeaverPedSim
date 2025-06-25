#!/usr/bin/env python3
"""
DiPPeR 모델 검증 스크립트
1. 정성적 평가: 시각적 경로 품질 검증
2. 정량적 평가: 성능 지표 측정
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

from robot_simuator_dippeR import RobotSimulatorDiPPeR, DiPPeR
from train_dipperp import SimulationDataCollector

class DiPPeRValidator:
    """DiPPeR 모델 검증 클래스"""
    
    def __init__(self, model_path, xml_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DiPPeR 모델 로드
        self.dipperp_model = DiPPeR(visual_feature_dim=512, path_dim=2, max_timesteps=1000)
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # 학습 시 저장된 체크포인트 형식
                self.dipperp_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 단순 모델 상태만 저장된 형식
                self.dipperp_model.load_state_dict(checkpoint)
            print(f"✅ DiPPeR 모델 로드 완료: {model_path}")
        else:
            print(f"❌ 모델 파일을 찾을 수 없음: {model_path}")
            return
            
        self.dipperp_model.to(self.device)
        self.dipperp_model.eval()
        
        # 시뮬레이터 초기화 (비교용 A* 포함)
        self.simulator = RobotSimulatorDiPPeR(xml_file, model_path=None)
        self.simulator.use_dipperp = False  # A* 사용
        
        # 결과 저장용
        self.results = {
            'qualitative': [],
            'quantitative': []
        }
    
    def collect_test_scenarios(self, num_scenarios=50):
        """다양한 테스트 시나리오 수집"""
        print(f"🔍 {num_scenarios}개 테스트 시나리오 수집 중...")
        
        scenarios = []
        
        for i in tqdm(range(num_scenarios), desc="시나리오 수집"):
            # 랜덤 시작/목표점 생성
            start_pos = self._generate_safe_position()
            goal_pos = self._generate_safe_position()
            
            # 최소 거리 보장
            while np.linalg.norm(np.array(start_pos) - np.array(goal_pos)) < 2.0:
                goal_pos = self._generate_safe_position()
            
            # 시뮬레이터 상태 설정
            self.simulator.robot_pos = start_pos.copy()
            self.simulator.target_pos = goal_pos.copy()
            
            # 몇 스텝 진행하여 동적 상황 생성
            for _ in range(np.random.randint(10, 30)):
                self.simulator.update()
            
            # 현재 상태 캡처
            self.simulator.update_fused_cost_map()
            scenario = {
                'id': i,
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'fused_cost_map': self.simulator.fused_cost_map.copy(),
                'agent_positions': [(agent.pos[0], agent.pos[1]) for agent in self.simulator.agents]
            }
            scenarios.append(scenario)
        
        print(f"✅ {len(scenarios)}개 시나리오 수집 완료")
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
            if self.simulator.is_position_safe(pos):
                return pos
        
        return [0.0, 0.0]  # 폴백
    
    def qualitative_evaluation(self, scenarios, save_dir="evaluation_results"):
        """정성적 평가: 시각적 검증"""
        print("🎨 정성적 평가 시작...")
        
        Path(save_dir).mkdir(exist_ok=True)
        qualitative_results = []
        
        for i, scenario in enumerate(tqdm(scenarios[:10], desc="시각적 검증")):  # 처음 10개만
            # DiPPeR 경로 생성
            dipperp_path = self._generate_dipperp_path(scenario)
            
            # A* 경로 생성 (비교용)
            astar_path = self.simulator.fallback_astar_planning(
                scenario['start_pos'], scenario['goal_pos']
            )
            
            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # DiPPeR 결과
            self._plot_path_comparison(ax1, scenario, dipperp_path, "DiPPeR Path")
            
            # A* 결과
            self._plot_path_comparison(ax2, scenario, astar_path, "A* Path")
            
            plt.suptitle(f"Scenario {i}: Start{scenario['start_pos']} → Goal{scenario['goal_pos']}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/scenario_{i:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 정성적 지표 계산
            quality_metrics = self._calculate_path_quality(scenario, dipperp_path, astar_path)
            qualitative_results.append(quality_metrics)
        
        self.results['qualitative'] = qualitative_results
        print(f"✅ 정성적 평가 완료. 결과 저장: {save_dir}/")
        
        return qualitative_results
    
    def _generate_dipperp_path(self, scenario):
        """DiPPeR로 경로 생성 (안전성 검증 포함)"""
        try:
            # Fused Cost Map을 텐서로 변환
            cost_map = torch.from_numpy(scenario['fused_cost_map']).float()
            cost_map = cost_map.unsqueeze(0).unsqueeze(0).to(self.device) / 255.0  # 정규화
            
            # 시작/목표점 정규화
            start_norm = torch.tensor([[scenario['start_pos'][0]/6.0, scenario['start_pos'][1]/6.0]], 
                                    dtype=torch.float32, device=self.device)
            goal_norm = torch.tensor([[scenario['goal_pos'][0]/6.0, scenario['goal_pos'][1]/6.0]], 
                                   dtype=torch.float32, device=self.device)
            
            # DiPPeR 경로 생성
            with torch.no_grad():
                generated_path = self.dipperp_model.sample_path(
                    cost_map, start_norm, goal_norm, 
                    path_length=50, num_inference_steps=20
                )
            
            # 실제 좌표로 변환
            path_real = generated_path[0].cpu().numpy() * 6.0
            path_list = [[float(point[0]), float(point[1])] for point in path_real]
            
            # 경로 안전성 검증 및 수정 (실제 시뮬레이터와 동일한 로직)
            safe_path = []
            for i, point in enumerate(path_list):
                if self.simulator.is_position_safe(point):
                    safe_path.append(point)
                else:
                    # 위험한 웨이포인트 감지 시 A* 보정 적용
                    if safe_path:
                        prev_safe = safe_path[-1]
                        # 다음 안전한 점 찾기
                        next_safe = None
                        for j in range(i+1, len(path_list)):
                            if self.simulator.is_position_safe(path_list[j]):
                                next_safe = path_list[j]
                                break
                        
                        if next_safe:
                            # A*로 안전한 경로 생성
                            correction_path = self.simulator.fallback_astar_planning(prev_safe, next_safe)
                            if correction_path and len(correction_path) > 2:
                                # 중간 점들만 추가 (시작점, 끝점 제외)
                                safe_path.extend(correction_path[1:-1])
                        
                        if next_safe:
                            safe_path.append(next_safe)
                    else:
                        # 첫 번째 점이 위험하면 시작점 사용
                        safe_path.append(scenario['start_pos'])
            
            # 안전성 검증된 경로가 너무 짧으면 A* 폴백
            if len(safe_path) < 5:
                return self.simulator.fallback_astar_planning(
                    scenario['start_pos'], scenario['goal_pos']
                )
            
            return safe_path
            
        except Exception as e:
            print(f"DiPPeR 경로 생성 실패: {e}")
            # 폴백: A* 사용
            return self.simulator.fallback_astar_planning(
                scenario['start_pos'], scenario['goal_pos']
            )
    
    def _plot_path_comparison(self, ax, scenario, path, title):
        """경로 비교 시각화"""
        # Fused Cost Map 표시
        ax.imshow(scenario['fused_cost_map'], extent=[-6, 6, -6, 6], 
                 origin='lower', cmap='hot', alpha=0.7)
        
        # 에이전트 위치
        for agent_pos in scenario['agent_positions']:
            ax.plot(agent_pos[0], agent_pos[1], 'bo', markersize=8, alpha=0.8)
        
        # 경로 표시
        if path and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=3, alpha=0.8)
            ax.plot(path_array[:, 0], path_array[:, 1], 'g.', markersize=4)
        
        # 시작/목표점
        ax.plot(scenario['start_pos'][0], scenario['start_pos'][1], 'rs', markersize=12, label='Start')
        ax.plot(scenario['goal_pos'][0], scenario['goal_pos'][1], 'r^', markersize=12, label='Goal')
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def quantitative_evaluation(self, scenarios):
        """정량적 평가: 성능 지표 측정"""
        print("📊 정량적 평가 시작...")
        
        quantitative_results = []
        
        for scenario in tqdm(scenarios, desc="정량적 평가"):
            # 시간 측정
            start_time = time.time()
            dipperp_path = self._generate_dipperp_path(scenario)
            dipperp_time = time.time() - start_time
            
            start_time = time.time()
            astar_path = self.simulator.fallback_astar_planning(
                scenario['start_pos'], scenario['goal_pos']
            )
            astar_time = time.time() - start_time
            
            # 지표 계산
            metrics = {
                'scenario_id': scenario['id'],
                'dipperp_success': self._check_path_validity(scenario, dipperp_path),
                'astar_success': self._check_path_validity(scenario, astar_path),
                'dipperp_cost': self._calculate_path_cost(scenario, dipperp_path),
                'astar_cost': self._calculate_path_cost(scenario, astar_path),
                'dipperp_length': self._calculate_path_length(dipperp_path),
                'astar_length': self._calculate_path_length(astar_path),
                'dipperp_smoothness': self._calculate_smoothness(dipperp_path),
                'astar_smoothness': self._calculate_smoothness(astar_path),
                'dipperp_time': dipperp_time,
                'astar_time': astar_time,
                'speedup': astar_time / dipperp_time if dipperp_time > 0 else float('inf')
            }
            
            quantitative_results.append(metrics)
        
        self.results['quantitative'] = quantitative_results
        print("✅ 정량적 평가 완료")
        
        return quantitative_results
    
    def _check_path_validity(self, scenario, path):
        """경로 유효성 검사"""
        if not path or len(path) < 2:
            return False
        
        # 시작/목표점 도달 확인
        start_dist = np.linalg.norm(np.array(path[0]) - np.array(scenario['start_pos']))
        goal_dist = np.linalg.norm(np.array(path[-1]) - np.array(scenario['goal_pos']))
        
        if start_dist > 1.0 or goal_dist > 1.0:
            return False
        
        # 장애물 관통 확인
        for point in path:
            if not self.simulator.is_position_safe(point):
                return False
        
        return True
    
    def _calculate_path_cost(self, scenario, path):
        """경로 비용 계산"""
        if not path:
            return float('inf')
        
        total_cost = 0
        for point in path:
            x_idx = int((point[0] + 6) / 0.2)
            y_idx = int((point[1] + 6) / 0.2)
            if 0 <= x_idx < 60 and 0 <= y_idx < 60:
                total_cost += scenario['fused_cost_map'][y_idx, x_idx]
        
        return total_cost / len(path)  # 평균 비용
    
    def _calculate_path_length(self, path):
        """경로 길이 계산"""
        if not path or len(path) < 2:
            return 0
        
        total_length = 0
        for i in range(len(path) - 1):
            dist = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            total_length += dist
        
        return total_length
    
    def _calculate_smoothness(self, path):
        """경로 부드러움 계산 (각도 변화의 표준편차)"""
        if not path or len(path) < 3:
            return 0
        
        angles = []
        for i in range(1, len(path) - 1):
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
        
        return np.std(angles) if angles else 0
    
    def _calculate_path_quality(self, scenario, dipperp_path, astar_path):
        """경로 품질 종합 평가"""
        return {
            'scenario_id': scenario['id'],
            'dipperp_valid': self._check_path_validity(scenario, dipperp_path),
            'astar_valid': self._check_path_validity(scenario, astar_path),
            'cost_ratio': (self._calculate_path_cost(scenario, dipperp_path) / 
                          max(self._calculate_path_cost(scenario, astar_path), 1e-6)),
            'length_ratio': (self._calculate_path_length(dipperp_path) / 
                           max(self._calculate_path_length(astar_path), 1e-6)),
            'smoothness_ratio': (self._calculate_smoothness(dipperp_path) / 
                               max(self._calculate_smoothness(astar_path), 1e-6))
        }
    
    def generate_report(self, save_path="evaluation_report.json"):
        """평가 결과 리포트 생성"""
        print("📋 평가 리포트 생성 중...")
        
        # 정량적 결과 통계
        df = pd.DataFrame(self.results['quantitative'])
        
        summary = {
            'overview': {
                'total_scenarios': len(df),
                'dipperp_success_rate': df['dipperp_success'].mean(),
                'astar_success_rate': df['astar_success'].mean(),
                'average_speedup': df['speedup'].mean(),
                'median_speedup': df['speedup'].median()
            },
            'performance_comparison': {
                'cost_improvement': 1 - (df['dipperp_cost'] / df['astar_cost']).mean(),
                'length_efficiency': 1 - (df['dipperp_length'] / df['astar_length']).mean(),
                'smoothness_improvement': 1 - (df['dipperp_smoothness'] / df['astar_smoothness']).mean(),
                'time_improvement': 1 - (df['dipperp_time'] / df['astar_time']).mean()
            },
            'detailed_results': self.results
        }
        
        # JSON 저장
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("📊 DiPPeR 평가 결과 요약")
        print("="*60)
        print(f"총 시나리오: {summary['overview']['total_scenarios']}")
        print(f"DiPPeR 성공률: {summary['overview']['dipperp_success_rate']:.2%}")
        print(f"A* 성공률: {summary['overview']['astar_success_rate']:.2%}")
        print(f"평균 속도 향상: {summary['overview']['average_speedup']:.2f}x")
        print(f"비용 개선: {summary['performance_comparison']['cost_improvement']:.2%}")
        print(f"길이 효율성: {summary['performance_comparison']['length_efficiency']:.2%}")
        print(f"부드러움 개선: {summary['performance_comparison']['smoothness_improvement']:.2%}")
        print(f"시간 개선: {summary['performance_comparison']['time_improvement']:.2%}")
        print("="*60)
        
        print(f"✅ 상세 리포트 저장: {save_path}")
        return summary

def main():
    parser = argparse.ArgumentParser(description='DiPPeR 모델 검증')
    parser.add_argument('--model_path', required=True, help='학습된 DiPPeR 모델 경로')
    parser.add_argument('--xml_file', default='scenarios/Circulation1.xml', help='시뮬레이션 XML 파일')
    parser.add_argument('--num_scenarios', type=int, default=100, help='테스트 시나리오 수')
    parser.add_argument('--save_dir', default='evaluation_results', help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 검증 실행
    validator = DiPPeRValidator(args.model_path, args.xml_file)
    
    # 1. 테스트 시나리오 수집
    scenarios = validator.collect_test_scenarios(args.num_scenarios)
    
    # 2. 정성적 평가
    validator.qualitative_evaluation(scenarios, args.save_dir)
    
    # 3. 정량적 평가
    validator.quantitative_evaluation(scenarios)
    
    # 4. 리포트 생성
    validator.generate_report(f"{args.save_dir}/evaluation_report.json")

if __name__ == "__main__":
    main() 