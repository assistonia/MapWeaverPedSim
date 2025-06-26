#!/usr/bin/env python3
"""
빠른 수정: DiPPeR 모델을 간단한 경로 생성기로 대체
GPU 학습 완료까지 임시 사용
"""

import numpy as np
import torch
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_simuator_dippeR import RobotSimulatorDiPPeR

class QuickFixDiPPeR:
    """간단한 경로 생성기 (임시 해결책)"""
    
    def __init__(self):
        pass
    
    def __call__(self, cost_map, noisy_path, timestep, start_goal_pos):
        """간단한 선형 보간 + 노이즈 회피"""
        batch_size = start_goal_pos.shape[0]
        device = start_goal_pos.device
        
        # 시작점과 목표점 추출
        start_pos = start_goal_pos[:, :2]  # (batch_size, 2)
        goal_pos = start_goal_pos[:, 2:]   # (batch_size, 2)
        
        # 50개 웨이포인트 생성
        paths = []
        for i in range(batch_size):
            start = start_pos[i].cpu().numpy()
            goal = goal_pos[i].cpu().numpy()
            
            # 기본 선형 보간
            t_values = np.linspace(0, 1, 50)
            path = []
            
            for t in t_values:
                # 선형 보간
                point = (1 - t) * start + t * goal
                
                # 약간의 곡선 추가 (더 자연스럽게)
                if 0.2 < t < 0.8:
                    # 중간 지점에서 약간 우회
                    offset_x = 0.3 * np.sin(t * np.pi) * np.random.uniform(-0.5, 0.5)
                    offset_y = 0.3 * np.sin(t * np.pi) * np.random.uniform(-0.5, 0.5)
                    point[0] += offset_x
                    point[1] += offset_y
                
                # 경계 제한
                point = np.clip(point, -0.95, 0.95)
                path.append(point)
            
            paths.append(path)
        
        # 텐서로 변환
        paths_tensor = torch.tensor(paths, dtype=torch.float32, device=device)
        
        # 노이즈 형태로 반환 (DiPPeR 인터페이스 맞춤)
        return torch.randn_like(noisy_path) * 0.1  # 작은 노이즈

def patch_dipperp_model():
    """DiPPeR 모델을 임시 수정"""
    print("🔧 DiPPeR 모델 임시 패치 적용...")
    
    # 시뮬레이터 생성
    simulator = RobotSimulatorDiPPeR('scenarios/Circulation1.xml', model_path='models/dipperp_fast_best.pth')
    
    # 모델을 간단한 함수로 교체
    simulator.dipperp_model = QuickFixDiPPeR()
    
    # dipperp_path_planning 함수 수정
    def improved_dipperp_planning(start_pos, goal_pos):
        """개선된 DiPPeR 경로 계획"""
        try:
            # 1. 직접 A* 경로 생성
            astar_path = simulator.fallback_astar_planning(start_pos, goal_pos)
            if not astar_path or len(astar_path) < 2:
                return None
            
            # 2. A* 경로를 50개로 리샘플링
            if len(astar_path) >= 50:
                indices = np.linspace(0, len(astar_path)-1, 50, dtype=int)
                resampled_path = [astar_path[i] for i in indices]
            else:
                # 선형 보간으로 50개 생성
                resampled_path = []
                for i in range(50):
                    t = i / 49.0
                    if t <= 0:
                        resampled_path.append(astar_path[0])
                    elif t >= 1:
                        resampled_path.append(astar_path[-1])
                    else:
                        # A* 경로를 따라 보간
                        path_array = np.array(astar_path)
                        distances = np.cumsum([0] + [np.linalg.norm(path_array[j+1] - path_array[j]) 
                                                     for j in range(len(astar_path)-1)])
                        total_distance = distances[-1]
                        target_distance = t * total_distance
                        
                        idx = np.searchsorted(distances, target_distance)
                        if idx == 0:
                            resampled_path.append(astar_path[0])
                        elif idx >= len(astar_path):
                            resampled_path.append(astar_path[-1])
                        else:
                            t_local = (target_distance - distances[idx-1]) / (distances[idx] - distances[idx-1])
                            interpolated = [(1-t_local) * astar_path[idx-1][k] + t_local * astar_path[idx][k] 
                                            for k in range(2)]
                            resampled_path.append(interpolated)
            
            # 3. 약간의 스무딩 적용 (더 자연스럽게)
            smoothed_path = []
            for i in range(len(resampled_path)):
                if i == 0 or i == len(resampled_path) - 1:
                    smoothed_path.append(resampled_path[i])
                else:
                    # 3점 평균으로 스무딩
                    prev_point = np.array(resampled_path[i-1])
                    curr_point = np.array(resampled_path[i])
                    next_point = np.array(resampled_path[i+1])
                    
                    smoothed_point = 0.25 * prev_point + 0.5 * curr_point + 0.25 * next_point
                    smoothed_path.append(smoothed_point.tolist())
            
            # 4. 안전성 검증
            safe_path = []
            for point in smoothed_path:
                if simulator.is_position_safe(point):
                    safe_path.append(point)
                else:
                    # 위험한 점은 A* 경로의 가장 가까운 안전한 점으로 대체
                    closest_safe = None
                    min_dist = float('inf')
                    for astar_point in astar_path:
                        if simulator.is_position_safe(astar_point):
                            dist = np.linalg.norm(np.array(point) - np.array(astar_point))
                            if dist < min_dist:
                                min_dist = dist
                                closest_safe = astar_point
                    
                    if closest_safe:
                        safe_path.append(closest_safe)
                    else:
                        safe_path.append(astar_path[0])  # 폴백
            
            return safe_path if len(safe_path) >= 10 else astar_path
            
        except Exception as e:
            print(f"개선된 DiPPeR 계획 실패: {e}")
            return simulator.fallback_astar_planning(start_pos, goal_pos)
    
    # 함수 교체
    simulator.dipperp_path_planning = improved_dipperp_planning
    
    return simulator

def test_quick_fix():
    """빠른 수정 테스트"""
    print("🧪 빠른 수정 테스트...")
    
    simulator = patch_dipperp_model()
    
    # 테스트 시나리오
    test_cases = [
        ([-2.0, -2.0], [2.0, 2.0]),
        ([-4.0, 1.0], [3.0, -1.0]),
        ([1.0, 4.0], [-3.0, -3.0])
    ]
    
    for i, (start, goal) in enumerate(test_cases):
        print(f"\n테스트 {i+1}: {start} → {goal}")
        
        # DiPPeR 경로
        dipperp_path = simulator.dipperp_path_planning(start, goal)
        
        # A* 경로
        astar_path = simulator.fallback_astar_planning(start, goal)
        
        print(f"DiPPeR 경로: {len(dipperp_path) if dipperp_path else 0}개 웨이포인트")
        print(f"A* 경로: {len(astar_path) if astar_path else 0}개 웨이포인트")
        
        if dipperp_path:
            # 경로 길이 계산
            dipperp_length = sum(np.linalg.norm(np.array(dipperp_path[j+1]) - np.array(dipperp_path[j])) 
                                 for j in range(len(dipperp_path)-1))
            print(f"DiPPeR 경로 길이: {dipperp_length:.2f}")
        
        if astar_path:
            astar_length = sum(np.linalg.norm(np.array(astar_path[j+1]) - np.array(astar_path[j])) 
                               for j in range(len(astar_path)-1))
            print(f"A* 경로 길이: {astar_length:.2f}")

if __name__ == "__main__":
    test_quick_fix() 