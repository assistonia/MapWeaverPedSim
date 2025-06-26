#!/usr/bin/env python3
"""
전략적 웨이포인트 기반 DiPPeR 학습
- 병목지점 중심 경로 계획
- 최소한의 핵심 웨이포인트 (5-10개)
- 로컬 에이전트와의 역할 분담
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import random
import cv2
from scipy.ndimage import binary_dilation, distance_transform_edt

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_simuator_dippeR import RobotSimulatorDiPPeR

class StrategicWaypointExtractor:
    """전략적 웨이포인트 추출기"""
    
    def __init__(self, grid_size=0.2):
        self.grid_size = grid_size
        
    def extract_strategic_waypoints(self, start_pos, goal_pos, cost_map, max_waypoints=8):
        """병목지점 기반 전략적 웨이포인트 추출"""
        
        # 1. 병목지점 탐지
        bottlenecks = self._detect_bottlenecks(cost_map)
        
        # 2. A* 기본 경로
        simulator = RobotSimulatorDiPPeR('scenarios/Circulation1.xml', model_path=None)
        astar_path = simulator.fallback_astar_planning(start_pos, goal_pos)
        
        if not astar_path or len(astar_path) < 2:
            return [start_pos, goal_pos]
        
        # 3. 경로 상의 중요 지점 식별
        critical_points = self._identify_critical_points(astar_path, bottlenecks, cost_map)
        
        # 4. 전략적 웨이포인트 선택
        strategic_waypoints = self._select_strategic_waypoints(
            start_pos, goal_pos, critical_points, max_waypoints
        )
        
        return strategic_waypoints
    
    def _detect_bottlenecks(self, cost_map):
        """병목지점 탐지"""
        # 장애물 맵 (높은 비용 = 장애물)
        obstacle_map = (cost_map > 0.5).astype(np.uint8)
        
        # 자유 공간
        free_space = 1 - obstacle_map
        
        # 거리 변환 (장애물로부터의 거리)
        distance_map = distance_transform_edt(free_space)
        
        # 병목지점: 거리가 작지만 0이 아닌 지점 (좁은 통로)
        bottleneck_threshold = 3.0  # 0.6m 이하의 좁은 통로
        bottlenecks = (distance_map > 0) & (distance_map < bottleneck_threshold)
        
        return bottlenecks
    
    def _identify_critical_points(self, astar_path, bottlenecks, cost_map):
        """경로 상의 중요 지점 식별"""
        critical_points = []
        
        for i, point in enumerate(astar_path):
            # 격자 좌표로 변환
            x_idx = int((point[0] + 6) / self.grid_size)
            y_idx = int((point[1] + 6) / self.grid_size)
            
            if 0 <= x_idx < 60 and 0 <= y_idx < 60:
                # 병목지점 근처인지 확인
                if bottlenecks[y_idx, x_idx]:
                    critical_points.append({
                        'point': point,
                        'index': i,
                        'type': 'bottleneck',
                        'importance': 1.0
                    })
                
                # 방향 급변 지점
                if i > 0 and i < len(astar_path) - 1:
                    v1 = np.array(astar_path[i]) - np.array(astar_path[i-1])
                    v2 = np.array(astar_path[i+1]) - np.array(astar_path[i])
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        
                        # 45도 이상 방향 변경
                        if angle > np.pi / 4:
                            critical_points.append({
                                'point': point,
                                'index': i,
                                'type': 'turn',
                                'importance': angle / np.pi
                            })
                
                # 높은 비용 지역 통과
                if cost_map[y_idx, x_idx] > 0.3:
                    critical_points.append({
                        'point': point,
                        'index': i,
                        'type': 'high_cost',
                        'importance': cost_map[y_idx, x_idx]
                    })
        
        return critical_points
    
    def _select_strategic_waypoints(self, start_pos, goal_pos, critical_points, max_waypoints):
        """전략적 웨이포인트 선택"""
        waypoints = [start_pos]
        
        if not critical_points:
            waypoints.append(goal_pos)
            return waypoints
        
        # 중요도 순으로 정렬
        critical_points.sort(key=lambda x: x['importance'], reverse=True)
        
        # 경로 순서 유지하면서 선택
        selected_indices = set()
        for cp in critical_points:
            if len(selected_indices) >= max_waypoints - 2:  # 시작/끝점 제외
                break
            selected_indices.add(cp['index'])
        
        # 인덱스 순으로 정렬하여 경로 순서 유지
        selected_indices = sorted(selected_indices)
        
        # 웨이포인트 추가
        for idx in selected_indices:
            for cp in critical_points:
                if cp['index'] == idx:
                    waypoints.append(cp['point'])
                    break
        
        waypoints.append(goal_pos)
        
        # 중복 제거 및 거리 체크
        final_waypoints = [waypoints[0]]
        for i in range(1, len(waypoints)):
            dist = np.linalg.norm(np.array(waypoints[i]) - np.array(final_waypoints[-1]))
            if dist > 1.0:  # 1m 이상 떨어진 점만 추가
                final_waypoints.append(waypoints[i])
        
        # 마지막 점이 목표점이 아니면 추가
        if np.linalg.norm(np.array(final_waypoints[-1]) - np.array(goal_pos)) > 0.5:
            final_waypoints.append(goal_pos)
        
        return final_waypoints

class StrategicDataset(Dataset):
    """전략적 웨이포인트 데이터셋"""
    
    def __init__(self, data_dir='training_data_strategic'):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._load_data()
        
    def _load_data(self):
        """데이터 로드"""
        data_files = list(self.data_dir.glob('*.json'))
        
        for file_path in tqdm(data_files, desc="데이터 로딩"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.samples.extend(data)
                    else:
                        self.samples.append(data)
            except Exception as e:
                print(f"파일 로딩 실패 {file_path}: {e}")
        
        print(f"총 {len(self.samples)}개 샘플 로드 완료")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 텐서 변환
        cost_map = torch.from_numpy(np.array(sample['cost_map'])).float().unsqueeze(0)
        start_pos = torch.from_numpy(np.array(sample['start_pos'])).float()
        goal_pos = torch.from_numpy(np.array(sample['goal_pos'])).float()
        
        # 가변 길이 웨이포인트를 고정 길이로 패딩
        waypoints = sample['strategic_waypoints']
        max_waypoints = 10
        
        if len(waypoints) > max_waypoints:
            waypoints = waypoints[:max_waypoints]
        else:
            # 패딩: 마지막 점으로 채움
            while len(waypoints) < max_waypoints:
                waypoints.append(waypoints[-1])
        
        waypoints_tensor = torch.from_numpy(np.array(waypoints)).float()
        num_valid = torch.tensor(len(sample['strategic_waypoints']), dtype=torch.long)
        
        return cost_map, start_pos, goal_pos, waypoints_tensor, num_valid

class StrategicDiPPeR(nn.Module):
    """전략적 웨이포인트 생성 모델"""
    
    def __init__(self, max_waypoints=10):
        super().__init__()
        self.max_waypoints = max_waypoints
        
        # CNN 백본 (비용 맵 처리)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 60x60 -> 30x30
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 30x30 -> 15x15
            
            nn.AdaptiveAvgPool2d(8),  # 15x15 -> 8x8
            nn.Flatten()
        )
        
        # 시작/목표점 인코더
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 융합 및 웨이포인트 생성
        cnn_out_size = 256 * 8 * 8
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 웨이포인트 수 예측
        self.num_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, max_waypoints),
            nn.Softmax(dim=-1)
        )
        
        # 웨이포인트 좌표 예측
        self.waypoint_predictor = nn.Sequential(
            nn.Linear(256, max_waypoints * 2),
            nn.Tanh()  # -1 ~ 1 범위
        )
    
    def forward(self, cost_map, start_pos, goal_pos):
        # 비용 맵 특징 추출
        cnn_features = self.cnn(cost_map)
        
        # 시작/목표점 인코딩
        start_goal = torch.cat([start_pos, goal_pos], dim=-1)
        pos_features = self.pos_encoder(start_goal)
        
        # 특징 융합
        fused = torch.cat([cnn_features, pos_features], dim=-1)
        fused_features = self.fusion(fused)
        
        # 웨이포인트 수 예측
        num_waypoints_prob = self.num_predictor(fused_features)
        
        # 웨이포인트 좌표 예측
        waypoints_flat = self.waypoint_predictor(fused_features)
        waypoints = waypoints_flat.view(-1, self.max_waypoints, 2)
        
        return waypoints, num_waypoints_prob

class StrategicDataCollector:
    """전략적 웨이포인트 데이터 수집기"""
    
    def __init__(self, xml_file, output_dir='training_data_strategic'):
        self.simulator = RobotSimulatorDiPPeR(xml_file, model_path=None)
        self.simulator.use_dipperp = False
        self.extractor = StrategicWaypointExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def collect_strategic_data(self, target_samples=10000):
        """전략적 웨이포인트 데이터 수집"""
        print(f"🎯 목표: {target_samples:,}개 전략적 경로 수집")
        
        collected_data = []
        batch_size = 500
        batch_count = 0
        
        pbar = tqdm(total=target_samples, desc="전략적 데이터 수집")
        
        while len(collected_data) < target_samples:
            # 랜덤 시작/목표점
            start_pos = self._generate_safe_position()
            goal_pos = self._generate_safe_position()
            
            if start_pos is None or goal_pos is None:
                continue
            
            # 거리 체크
            dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            if dist < 3.0:  # 최소 3m 이상
                continue
            
            # 전략적 웨이포인트 추출
            strategic_waypoints = self.extractor.extract_strategic_waypoints(
                start_pos, goal_pos, self.simulator.fused_cost_map
            )
            
            if len(strategic_waypoints) >= 2:
                data_sample = {
                    'cost_map': self.simulator.fused_cost_map.copy(),
                    'start_pos': np.array([start_pos[0]/6.0, start_pos[1]/6.0]),  # 정규화
                    'goal_pos': np.array([goal_pos[0]/6.0, goal_pos[1]/6.0]),    # 정규화
                    'strategic_waypoints': np.array([[p[0]/6.0, p[1]/6.0] for p in strategic_waypoints]),
                    'num_waypoints': len(strategic_waypoints)
                }
                
                collected_data.append(data_sample)
                pbar.update(1)
            
            # 배치 저장
            if len(collected_data) >= batch_size:
                self._save_batch(collected_data[:batch_size], batch_count)
                collected_data = collected_data[batch_size:]
                batch_count += 1
        
        # 남은 데이터 저장
        if collected_data:
            self._save_batch(collected_data, batch_count)
        
        pbar.close()
        print(f"✅ 전략적 데이터 수집 완료: {batch_count + 1}개 배치 파일")
    
    def _generate_safe_position(self):
        """안전한 위치 생성"""
        safe_zones = [
            (-5.5, -5.5, -0.5, -0.5),
            (-5.5, 0.5, -0.5, 5.5),
            (0.5, -5.5, 5.5, -0.5),
            (0.5, 0.5, 5.5, 5.5),
            (-2.0, -2.0, 2.0, 2.0)
        ]
        
        for _ in range(50):
            zone = random.choice(safe_zones)
            x = random.uniform(zone[0], zone[2])
            y = random.uniform(zone[1], zone[3])
            
            if self.simulator.is_position_safe([x, y]):
                return [x, y]
        
        return None
    
    def _save_batch(self, data, batch_id):
        """배치 데이터 저장"""
        filename = self.output_dir / f"strategic_batch_{batch_id:04d}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)
        print(f"전략적 배치 저장: {filename} ({len(data)}개 샘플)")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def train_strategic_dipperp():
    """전략적 DiPPeR 학습"""
    print("=== 전략적 웨이포인트 DiPPeR 학습 ===")
    
    # 1. 데이터 수집
    print("\n1. 전략적 데이터 수집...")
    collector = StrategicDataCollector('scenarios/Circulation1.xml')
    collector.collect_strategic_data(target_samples=5000)
    
    # 2. 데이터셋 로드
    print("\n2. 데이터셋 로드...")
    dataset = StrategicDataset('training_data_strategic')
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"학습 데이터: {len(train_dataset):,}개")
    print(f"검증 데이터: {len(val_dataset):,}개")
    
    # 3. 모델 및 옵티마이저
    print("\n3. 전략적 모델 생성...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StrategicDiPPeR(max_waypoints=10).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # 손실 함수
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # 4. 학습
    print("\n4. 전략적 학습 시작...")
    os.makedirs('models', exist_ok=True)
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        # 학습
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            cost_maps, start_pos, goal_pos, waypoints_gt, num_valid = batch
            cost_maps = cost_maps.to(device)
            start_pos = start_pos.to(device)
            goal_pos = goal_pos.to(device)
            waypoints_gt = waypoints_gt.to(device)
            num_valid = num_valid.to(device)
            
            optimizer.zero_grad()
            
            # 예측
            waypoints_pred, num_pred = model(cost_maps, start_pos, goal_pos)
            
            # 손실 계산
            waypoint_loss = mse_loss(waypoints_pred, waypoints_gt)
            
            # 웨이포인트 수 예측 손실
            num_targets = torch.zeros(num_valid.size(0), 10, device=device)
            for i, n in enumerate(num_valid):
                if n < 10:
                    num_targets[i, n-1] = 1.0  # 인덱스는 0부터
            
            num_loss = ce_loss(num_pred, num_targets.argmax(dim=1))
            
            total_loss = waypoint_loss + 0.1 * num_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cost_maps, start_pos, goal_pos, waypoints_gt, num_valid = batch
                cost_maps = cost_maps.to(device)
                start_pos = start_pos.to(device)
                goal_pos = goal_pos.to(device)
                waypoints_gt = waypoints_gt.to(device)
                num_valid = num_valid.to(device)
                
                waypoints_pred, num_pred = model(cost_maps, start_pos, goal_pos)
                
                waypoint_loss = mse_loss(waypoints_pred, waypoints_gt)
                num_targets = torch.zeros(num_valid.size(0), 10, device=device)
                for i, n in enumerate(num_valid):
                    if n < 10:
                        num_targets[i, n-1] = 1.0
                
                num_loss = ce_loss(num_pred, num_targets.argmax(dim=1))
                total_loss = waypoint_loss + 0.1 * num_loss
                
                val_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/strategic_dipperp_best.pth')
            print(f"🎯 새로운 최고 성능: {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    print("✅ 전략적 DiPPeR 학습 완료!")

if __name__ == "__main__":
    train_strategic_dipperp() 