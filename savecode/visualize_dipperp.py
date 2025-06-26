#!/usr/bin/env python3
"""
DiPPeR 모델 시각화 도구
실시간으로 DiPPeR vs A* 경로를 비교 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from robot_simuator_dippeR import RobotSimulatorDiPPeR
import time

class DiPPeRVisualizer:
    def __init__(self, model_path, xml_file):
        self.simulator = RobotSimulatorDiPPeR(xml_file, model_path)
        self.xml_file = xml_file
        
    def visualize_single_scenario(self, start_pos=None, goal_pos=None):
        """단일 시나리오 시각화"""
        print("🎯 DiPPeR vs A* 경로 비교 시각화")
        
        # 안전한 위치 생성
        if start_pos is None:
            start_pos = self._generate_safe_position()
        if goal_pos is None:
            goal_pos = self._generate_safe_position()
            
        print(f"시작점: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        print(f"목표점: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        
        # 시나리오 생성
        scenario = {
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'agent_positions': [[np.random.uniform(-3, 3), np.random.uniform(-3, 3)] for _ in range(5)],
            'fused_cost_map': self.simulator.fused_cost_map.copy()
        }
        
        # DiPPeR 경로 생성 (시간 측정)
        print("🤖 DiPPeR 경로 생성 중...")
        start_time = time.time()
        dipperp_path = self._generate_dipperp_path(scenario)
        dipperp_time = time.time() - start_time
        
        # A* 경로 생성 (시간 측정)
        print("⭐ A* 경로 생성 중...")
        start_time = time.time()
        astar_path = self.simulator.fallback_astar_planning(start_pos, goal_pos)
        astar_time = time.time() - start_time
        
        # 시각화
        self._plot_comparison(scenario, dipperp_path, astar_path, dipperp_time, astar_time)
        
    def _generate_safe_position(self):
        """안전한 위치 생성"""
        safe_zones = [
            [(-5, -1), (-3, 1)],   # 왼쪽 위
            [(3, -1), (5, 1)],     # 오른쪽 위  
            [(-5, -5), (-3, -3)],  # 왼쪽 아래
            [(3, -5), (5, -3)],    # 오른쪽 아래
            [(-1, 3), (1, 5)]      # 중앙 위
        ]
        
        for _ in range(100):  # 최대 100번 시도
            zone = np.random.choice(len(safe_zones))
            x_min, y_min = safe_zones[zone][0]
            x_max, y_max = safe_zones[zone][1]
            
            pos = [
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ]
            
            if self.simulator.is_position_safe(pos):
                return pos
                
        # 폴백: 기본 안전 위치
        return [0.0, 4.0]
    
    def _generate_dipperp_path(self, scenario):
        """DiPPeR로 경로 생성"""
        try:
            return self.simulator.dipperp_path_planning(
                scenario['start_pos'], scenario['goal_pos']
            )
        except Exception as e:
            print(f"DiPPeR 오류: {e}")
            return self.simulator.fallback_astar_planning(
                scenario['start_pos'], scenario['goal_pos']
            )
    
    def _plot_comparison(self, scenario, dipperp_path, astar_path, dipperp_time, astar_time):
        """비교 시각화"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. DiPPeR 경로
        self._plot_single_path(ax1, scenario, dipperp_path, 
                              f'DiPPeR 경로 ({dipperp_time:.3f}초)', 'red')
        
        # 2. A* 경로  
        self._plot_single_path(ax2, scenario, astar_path,
                              f'A* 경로 ({astar_time:.3f}초)', 'blue')
        
        # 3. 오버레이 비교
        self._plot_overlay(ax3, scenario, dipperp_path, astar_path)
        
        # 성능 정보 표시
        self._add_performance_info(fig, dipperp_path, astar_path, dipperp_time, astar_time)
        
        plt.tight_layout()
        plt.show()
        
    def _plot_single_path(self, ax, scenario, path, title, color):
        """단일 경로 시각화"""
        # Fused Cost Map 배경
        ax.imshow(scenario['fused_cost_map'], extent=[-6, 6, -6, 6], 
                 origin='lower', cmap='YlOrRd', alpha=0.6)
        
        # 장애물
        for obs in self.simulator.obstacles:
            ax.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=3)
        
        # 에이전트
        for agent_pos in scenario['agent_positions']:
            ax.plot(agent_pos[0], agent_pos[1], 'go', markersize=6, alpha=0.7)
        
        # 경로
        if path and len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   color=color, linewidth=3, alpha=0.8, label='경로')
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   'o', color=color, markersize=4, alpha=0.6)
        
        # 시작/목표점
        ax.plot(scenario['start_pos'][0], scenario['start_pos'][1], 
               'rs', markersize=12, label='시작')
        ax.plot(scenario['goal_pos'][0], scenario['goal_pos'][1], 
               'r^', markersize=12, label='목표')
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_overlay(self, ax, scenario, dipperp_path, astar_path):
        """오버레이 비교"""
        # 배경
        ax.imshow(scenario['fused_cost_map'], extent=[-6, 6, -6, 6], 
                 origin='lower', cmap='YlOrRd', alpha=0.4)
        
        # 장애물
        for obs in self.simulator.obstacles:
            ax.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=3)
        
        # DiPPeR 경로
        if dipperp_path and len(dipperp_path) > 1:
            path_array = np.array(dipperp_path)
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   'r-', linewidth=3, alpha=0.8, label='DiPPeR')
        
        # A* 경로
        if astar_path and len(astar_path) > 1:
            path_array = np.array(astar_path)
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   'b--', linewidth=2, alpha=0.8, label='A*')
        
        # 시작/목표점
        ax.plot(scenario['start_pos'][0], scenario['start_pos'][1], 
               'rs', markersize=12, label='시작')
        ax.plot(scenario['goal_pos'][0], scenario['goal_pos'][1], 
               'r^', markersize=12, label='목표')
        
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title('DiPPeR vs A* 비교', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _add_performance_info(self, fig, dipperp_path, astar_path, dipperp_time, astar_time):
        """성능 정보 추가"""
        info_text = f"""
성능 비교:
• DiPPeR 시간: {dipperp_time:.3f}초
• A* 시간: {astar_time:.3f}초  
• 속도 비율: {astar_time/dipperp_time:.2f}x

• DiPPeR 길이: {self._calculate_length(dipperp_path):.2f}m
• A* 길이: {self._calculate_length(astar_path):.2f}m
• 길이 비율: {self._calculate_length(dipperp_path)/max(self._calculate_length(astar_path), 0.1):.2f}x
        """
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def _calculate_length(self, path):
        """경로 길이 계산"""
        if not path or len(path) < 2:
            return 0
        
        total_length = 0
        for i in range(len(path) - 1):
            dist = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            total_length += dist
        return total_length
    
    def interactive_demo(self):
        """인터랙티브 데모"""
        print("🎮 DiPPeR 인터랙티브 시각화 데모")
        print("Enter를 누르면 새로운 랜덤 시나리오 생성")
        print("'q'를 입력하면 종료")
        
        while True:
            user_input = input("\n새 시나리오 생성 (Enter) 또는 종료 (q): ").strip().lower()
            
            if user_input == 'q':
                print("👋 시각화 종료")
                break
                
            self.visualize_single_scenario()
            
    def custom_scenario(self, start_x, start_y, goal_x, goal_y):
        """사용자 정의 시나리오"""
        start_pos = [float(start_x), float(start_y)]
        goal_pos = [float(goal_x), float(goal_y)]
        
        print(f"🎯 사용자 정의 시나리오")
        self.visualize_single_scenario(start_pos, goal_pos)

def main():
    parser = argparse.ArgumentParser(description='DiPPeR 시각화 도구')
    parser.add_argument('--model_path', default='models/dipperp_fast_best.pth',
                       help='DiPPeR 모델 경로')
    parser.add_argument('--xml_file', default='scenarios/Circulation1.xml',
                       help='시나리오 XML 파일')
    parser.add_argument('--mode', choices=['single', 'interactive', 'custom'], 
                       default='single', help='시각화 모드')
    parser.add_argument('--start_x', type=float, help='시작점 X')
    parser.add_argument('--start_y', type=float, help='시작점 Y')  
    parser.add_argument('--goal_x', type=float, help='목표점 X')
    parser.add_argument('--goal_y', type=float, help='목표점 Y')
    
    args = parser.parse_args()
    
    # 시각화 도구 초기화
    visualizer = DiPPeRVisualizer(args.model_path, args.xml_file)
    
    if args.mode == 'single':
        visualizer.visualize_single_scenario()
    elif args.mode == 'interactive':
        visualizer.interactive_demo()
    elif args.mode == 'custom':
        if all([args.start_x, args.start_y, args.goal_x, args.goal_y]):
            visualizer.custom_scenario(args.start_x, args.start_y, args.goal_x, args.goal_y)
        else:
            print("❌ 사용자 정의 모드는 --start_x, --start_y, --goal_x, --goal_y 필요")

if __name__ == "__main__":
    main() 