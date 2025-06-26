#!/usr/bin/env python3
"""CDIF 시각화 테스트"""

import numpy as np
import matplotlib.pyplot as plt
from robot_simulator_cgip import RobotSimulator

def test_visualization():
    """시각화 테스트"""
    print("🔍 시각화 테스트 시작...")
    
    # 시뮬레이터 초기화
    sim = RobotSimulator('scenarios/Circulation1.xml')
    
    print(f"Grid shape: {sim.grid.shape}")
    print(f"Grid range: {sim.grid.min()} ~ {sim.grid.max()}")
    print(f"Obstacles: {sim.grid.sum()} cells")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 장애물 맵
    ax = axes[0]
    im1 = ax.imshow(sim.grid, cmap='gray', origin='lower', extent=[-6, 6, -6, 6])
    ax.set_title('Obstacle Map')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax)
    
    # 테스트 점들 추가
    start_pos = [-4, -4]
    goal_pos = [4, 4]
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
    ax.legend()
    
    # 2. A* 경로 테스트
    ax = axes[1]
    ax.imshow(sim.grid, cmap='gray', origin='lower', extent=[-6, 6, -6, 6], alpha=0.7)
    
    # A* 경로 계산
    path = sim.a_star(start_pos, goal_pos)
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label=f'A* Path ({len(path)} points)')
        print(f"✅ A* 경로 생성 성공: {len(path)}개 점")
    else:
        print("❌ A* 경로 생성 실패")
    
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
    ax.set_title('A* Path Planning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("💾 테스트 이미지 저장: test_visualization.png")

if __name__ == "__main__":
    test_visualization() 