#!/usr/bin/env python3
"""
CDIF Training Results Visualization & Validation
- 학습 결과 시각적 검증
- 실제 이동 경로 확인
- 벽 뚫기 방지 검사
- 안전성 검증
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 서버용
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import cv2
from tqdm import tqdm

# 프로젝트 모듈
from cdif_model import CDIFModel, DDPMScheduler
from train_cdif import CDIFConfig
from robot_simulator_cgip import RobotSimulatorCGIP

class CDIFInference:
    """CDIF 추론기"""
    
    def __init__(self, model_path: str, config: CDIFConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = CDIFModel(
            max_waypoints=config.max_waypoints,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        ).to(self.device)
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule
        ).to(self.device)
        
        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ CDIF 모델 로드 완료: {model_path}")
        print(f"🎯 최고 성능: {checkpoint.get('best_val_loss', 'N/A')}")
    
    def generate_waypoints(self, 
                          cost_map: np.ndarray,
                          start_pos: List[float],
                          goal_pos: List[float],
                          num_inference_steps: int = 50) -> Tuple[np.ndarray, int]:
        """전략적 웨이포인트 생성"""
        
        with torch.no_grad():
            # 데이터 전처리
            cost_map_tensor = torch.from_numpy(cost_map).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 60, 60]
            start_pos_tensor = torch.from_numpy(np.array(start_pos) / 6.0).float().unsqueeze(0).to(self.device)  # [1, 2]
            goal_pos_tensor = torch.from_numpy(np.array(goal_pos) / 6.0).float().unsqueeze(0).to(self.device)  # [1, 2]
            
            # 초기 잡음 생성
            waypoints_shape = (1, self.config.max_waypoints, 2)
            waypoints = torch.randn(waypoints_shape, device=self.device)
            
            # 시작점과 끝점 고정 (inpainting)
            waypoints[0, 0] = start_pos_tensor[0]  # 시작점
            waypoints[0, -1] = goal_pos_tensor[0]  # 끝점
            
            # DDPM 역방향 과정
            timesteps = torch.linspace(
                self.config.num_train_timesteps - 1, 0, 
                num_inference_steps, dtype=torch.long, device=self.device
            )
            
            for i, t in enumerate(timesteps):
                t_batch = t.unsqueeze(0)  # [1]
                
                # 잡음 예측
                predicted_noise, num_waypoints_prob = self.model(
                    cost_map_tensor, waypoints, t_batch, start_pos_tensor, goal_pos_tensor
                )
                
                # 디노이징
                alpha_t = self.scheduler.alphas_cumprod[t]
                alpha_t_prev = self.scheduler.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)
                
                beta_t = 1 - alpha_t / alpha_t_prev
                waypoints = (waypoints - beta_t * predicted_noise) / torch.sqrt(alpha_t / alpha_t_prev)
                
                # 잡음 추가 (마지막 단계 제외)
                if i < len(timesteps) - 1:
                    noise = torch.randn_like(waypoints)
                    waypoints = waypoints + torch.sqrt(beta_t) * noise
                
                # 시작점과 끝점 다시 고정
                waypoints[0, 0] = start_pos_tensor[0]
                waypoints[0, -1] = goal_pos_tensor[0]
            
            # 웨이포인트 수 예측
            num_waypoints = torch.argmax(num_waypoints_prob, dim=1).item() + 1
            
            # 좌표 복원 (-1~1 → -6~6)
            waypoints = waypoints[0].cpu().numpy() * 6.0
            
            return waypoints[:num_waypoints], num_waypoints

class CDIFValidator:
    """CDIF 검증기"""
    
    def __init__(self, scenario_file='scenarios/Circulation1.xml'):
        self.simulator = RobotSimulatorCGIP(scenario_file)
        self.grid_size = 0.2
        
    def validate_waypoints(self, waypoints: np.ndarray) -> Dict[str, bool]:
        """웨이포인트 안전성 검증"""
        results = {
            'all_safe': True,
            'wall_collision': False,
            'obstacle_collision': False,
            'connectivity': True,
            'path_exists': True
        }
        
        # 1. 각 웨이포인트 안전성 체크
        for i, waypoint in enumerate(waypoints):
            if not self.simulator.is_position_safe(waypoint):
                results['all_safe'] = False
                
                # 장애물 유형 확인
                x_idx = int((waypoint[0] + 6) / self.grid_size)
                y_idx = int((waypoint[1] + 6) / self.grid_size)
                
                if 0 <= x_idx < 60 and 0 <= y_idx < 60:
                    if self.simulator.grid[y_idx, x_idx] == 1:  # 벽
                        results['wall_collision'] = True
                    elif self.simulator.fused_cost_map[y_idx, x_idx] > 0.5:  # 장애물
                        results['obstacle_collision'] = True
        
        # 2. 연결성 체크
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            
            path = self.simulator.fallback_astar_planning(start, goal)
            if not path:
                results['connectivity'] = False
                results['path_exists'] = False
                break
        
        return results
    
    def check_wall_penetration(self, waypoints: np.ndarray) -> List[Tuple[int, int]]:
        """벽 관통 체크"""
        penetrations = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # 직선 경로상의 점들 체크
            num_checks = int(np.linalg.norm(end - start) / 0.1) + 1
            for j in range(num_checks):
                t = j / max(1, num_checks - 1)
                point = start + t * (end - start)
                
                x_idx = int((point[0] + 6) / self.grid_size)
                y_idx = int((point[1] + 6) / self.grid_size)
                
                if 0 <= x_idx < 60 and 0 <= y_idx < 60:
                    if self.simulator.grid[y_idx, x_idx] == 1:  # 벽
                        penetrations.append((i, i+1))
                        break
        
        return penetrations

class CDIFVisualizer:
    """CDIF 시각화기"""
    
    def __init__(self, output_dir='visualization_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.simulator = RobotSimulatorCGIP('scenarios/Circulation1.xml')
        
    def visualize_generation_process(self, 
                                   cost_map: np.ndarray,
                                   start_pos: List[float],
                                   goal_pos: List[float],
                                   inference: CDIFInference,
                                   save_path: str):
        """웨이포인트 생성 과정 시각화"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('CDIF Waypoint Generation Process', fontsize=16)
        
        # 1. 원본 비용맵
        ax = axes[0, 0]
        im1 = ax.imshow(cost_map, cmap='viridis', origin='lower', extent=[-6, 6, -6, 6])
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
        ax.set_title('Social Cost Map')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax)
        
        # 2. CGIP A* 경로 (비교용)
        ax = axes[0, 1]
        ax.imshow(cost_map, cmap='viridis', origin='lower', extent=[-6, 6, -6, 6], alpha=0.7)
        
        astar_path = self.simulator.fallback_astar_planning(start_pos, goal_pos)
        if astar_path:
            path_x = [p[0] for p in astar_path]
            path_y = [p[1] for p in astar_path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='A* Path')
        
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
        ax.set_title('CGIP A* Reference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. CDIF 생성 웨이포인트
        ax = axes[0, 2]
        ax.imshow(cost_map, cmap='viridis', origin='lower', extent=[-6, 6, -6, 6], alpha=0.7)
        
        waypoints, num_waypoints = inference.generate_waypoints(cost_map, start_pos, goal_pos)
        
        # 웨이포인트 연결
        if len(waypoints) > 1:
            wp_x = waypoints[:, 0]
            wp_y = waypoints[:, 1]
            ax.plot(wp_x, wp_y, 'r--', linewidth=2, alpha=0.8, label='CDIF Waypoints')
            ax.scatter(wp_x[1:-1], wp_y[1:-1], c='orange', s=100, zorder=5, label='Strategic Points')
        
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
        ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
        ax.set_title(f'CDIF Generated ({num_waypoints} points)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 안전성 검증
        validator = CDIFValidator()
        validation_results = validator.validate_waypoints(waypoints)
        penetrations = validator.check_wall_penetration(waypoints)
        
        ax = axes[1, 0]
        ax.imshow(self.simulator.grid, cmap='gray', origin='lower', extent=[-6, 6, -6, 6])
        
        # 웨이포인트 표시
        if len(waypoints) > 1:
            wp_x = waypoints[:, 0]
            wp_y = waypoints[:, 1]
            ax.plot(wp_x, wp_y, 'r-', linewidth=3, alpha=0.8)
            ax.scatter(wp_x, wp_y, c='red', s=100, zorder=5)
        
        # 관통 지점 표시
        for start_idx, end_idx in penetrations:
            ax.plot([waypoints[start_idx, 0], waypoints[end_idx, 0]], 
                   [waypoints[start_idx, 1], waypoints[end_idx, 1]], 
                   'r-', linewidth=5, alpha=0.8, label='Wall Penetration!')
        
        ax.set_title('Safety Validation')
        if penetrations:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. A* 연결 확인
        ax = axes[1, 1]
        ax.imshow(cost_map, cmap='viridis', origin='lower', extent=[-6, 6, -6, 6], alpha=0.7)
        
        # 각 웨이포인트 간 A* 경로 확인
        all_connected = True
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            path = self.simulator.fallback_astar_planning(start, goal)
            
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                ax.plot(path_x, path_y, 'g-', alpha=0.6, linewidth=1)
            else:
                all_connected = False
                ax.plot([start[0], goal[0]], [start[1], goal[1]], 'r-', linewidth=3, alpha=0.8)
        
        ax.scatter(waypoints[:, 0], waypoints[:, 1], c='blue', s=100, zorder=5)
        ax.set_title(f'A* Connectivity: {"✅" if all_connected else "❌"}')
        ax.grid(True, alpha=0.3)
        
        # 6. 검증 결과 요약
        ax = axes[1, 2]
        ax.axis('off')
        
        results_text = f"""
CDIF Validation Results:

🎯 Generated Waypoints: {num_waypoints}
{'✅' if validation_results['all_safe'] else '❌'} All Safe: {validation_results['all_safe']}
{'✅' if not validation_results['wall_collision'] else '❌'} No Wall Collision: {not validation_results['wall_collision']}
{'✅' if not validation_results['obstacle_collision'] else '❌'} No Obstacle Collision: {not validation_results['obstacle_collision']}
{'✅' if validation_results['connectivity'] else '❌'} Connectivity: {validation_results['connectivity']}
{'✅' if not penetrations else '❌'} No Penetrations: {len(penetrations) == 0}

🔍 Wall Penetrations: {len(penetrations)}
🛣️  A* Path Length: {len(astar_path) if astar_path else 0}
🎯 CDIF Efficiency: {len(astar_path) / num_waypoints if astar_path and num_waypoints > 0 else 'N/A'}
        """
        
        ax.text(0.1, 0.9, results_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return validation_results, penetrations
    
    def batch_validation(self, 
                        inference: CDIFInference,
                        num_tests: int = 20,
                        save_dir: str = None):
        """배치 검증"""
        
        if save_dir is None:
            save_dir = self.output_dir / 'batch_validation'
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"🔍 CDIF 배치 검증 시작: {num_tests}개 테스트")
        
        # 안전한 위치 생성을 위한 영역
        safe_zones = [
            (-5.5, -5.5, -0.5, -0.5),
            (-5.5, 0.5, -0.5, 5.5),
            (0.5, -5.5, 5.5, -0.5),
            (0.5, 0.5, 5.5, 5.5),
            (-2.0, -2.0, 2.0, 2.0)
        ]
        
        results_summary = {
            'total_tests': 0,
            'safe_count': 0,
            'wall_collisions': 0,
            'obstacle_collisions': 0,
            'connectivity_failures': 0,
            'penetrations': 0,
            'avg_waypoints': 0,
            'efficiency_scores': []
        }
        
        validator = CDIFValidator()
        
        for test_idx in tqdm(range(num_tests), desc="배치 검증"):
            # 랜덤 시작/목표점 생성
            start_pos = self._generate_safe_position(safe_zones)
            goal_pos = self._generate_safe_position(safe_zones)
            
            if start_pos is None or goal_pos is None:
                continue
            
            # 거리 체크
            dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            if dist < 3.0:
                continue
            
            # CDIF 웨이포인트 생성
            waypoints, num_waypoints = inference.generate_waypoints(
                self.simulator.fused_cost_map, start_pos, goal_pos
            )
            
            # 검증
            validation_results = validator.validate_waypoints(waypoints)
            penetrations = validator.check_wall_penetration(waypoints)
            
            # 통계 업데이트
            results_summary['total_tests'] += 1
            if validation_results['all_safe']:
                results_summary['safe_count'] += 1
            if validation_results['wall_collision']:
                results_summary['wall_collisions'] += 1
            if validation_results['obstacle_collision']:
                results_summary['obstacle_collisions'] += 1
            if not validation_results['connectivity']:
                results_summary['connectivity_failures'] += 1
            if penetrations:
                results_summary['penetrations'] += 1
            
            results_summary['avg_waypoints'] += num_waypoints
            
            # 효율성 계산
            astar_path = self.simulator.fallback_astar_planning(start_pos, goal_pos)
            if astar_path and num_waypoints > 0:
                efficiency = len(astar_path) / num_waypoints
                results_summary['efficiency_scores'].append(efficiency)
            
            # 실패 케이스 시각화
            if not validation_results['all_safe'] or penetrations:
                save_path = save_dir / f'failure_case_{test_idx:03d}.png'
                self.visualize_generation_process(
                    self.simulator.fused_cost_map, start_pos, goal_pos, 
                    inference, save_path
                )
        
        # 결과 정리
        if results_summary['total_tests'] > 0:
            results_summary['avg_waypoints'] /= results_summary['total_tests']
            results_summary['success_rate'] = results_summary['safe_count'] / results_summary['total_tests']
            results_summary['avg_efficiency'] = np.mean(results_summary['efficiency_scores']) if results_summary['efficiency_scores'] else 0
        
        # 결과 저장
        with open(save_dir / 'validation_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # 결과 출력
        print(f"\n📊 CDIF 배치 검증 결과:")
        print(f"✅ 성공률: {results_summary['success_rate']:.1%}")
        print(f"🎯 평균 웨이포인트 수: {results_summary['avg_waypoints']:.1f}")
        print(f"🚀 평균 효율성: {results_summary['avg_efficiency']:.1f}x")
        print(f"🚫 벽 충돌: {results_summary['wall_collisions']}")
        print(f"🚧 장애물 충돌: {results_summary['obstacle_collisions']}")
        print(f"🔗 연결성 실패: {results_summary['connectivity_failures']}")
        print(f"⚠️  벽 관통: {results_summary['penetrations']}")
        
        return results_summary
    
    def _generate_safe_position(self, safe_zones):
        """안전한 위치 생성"""
        import random
        
        for _ in range(50):
            zone = random.choice(safe_zones)
            x = random.uniform(zone[0], zone[2])
            y = random.uniform(zone[1], zone[3])
            
            if self.simulator.is_position_safe([x, y]):
                return [x, y]
        
        return None

def main():
    parser = argparse.ArgumentParser(description='CDIF Training Results Validation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained CDIF model')
    parser.add_argument('--output_dir', type=str, default='validation_results', help='Output directory')
    parser.add_argument('--num_tests', type=int, default=20, help='Number of validation tests')
    parser.add_argument('--single_test', action='store_true', help='Run single test with visualization')
    args = parser.parse_args()
    
    # 설정 로드
    config = CDIFConfig()
    
    print("🔍 CDIF 학습 결과 검증 시작!")
    
    # 추론기 초기화
    inference = CDIFInference(args.model_path, config)
    
    # 시각화기 초기화
    visualizer = CDIFVisualizer(args.output_dir)
    
    if args.single_test:
        # 단일 테스트
        print("🎯 단일 테스트 실행...")
        
        # 테스트 케이스
        start_pos = [-4.0, -4.0]
        goal_pos = [4.0, 4.0]
        
        save_path = Path(args.output_dir) / 'single_test_result.png'
        validation_results, penetrations = visualizer.visualize_generation_process(
            visualizer.simulator.fused_cost_map, start_pos, goal_pos, 
            inference, save_path
        )
        
        print(f"💾 결과 저장: {save_path}")
        print(f"✅ 검증 결과: {validation_results}")
        if penetrations:
            print(f"⚠️  벽 관통 감지: {penetrations}")
    
    else:
        # 배치 검증
        results_summary = visualizer.batch_validation(
            inference, num_tests=args.num_tests, 
            save_dir=Path(args.output_dir) / 'batch_validation'
        )
        
        print(f"\n💾 상세 결과 저장: {Path(args.output_dir) / 'batch_validation'}")

if __name__ == "__main__":
    main() 