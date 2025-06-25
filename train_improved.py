#!/usr/bin/env python3
"""
개선된 DiPPeR 학습 코드
- 명확한 장애물 구조의 맵들 사용
- 다양한 학습 데이터 생성
- 더 나은 모델 구조 및 학습 전략
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
import random
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from robot_simulator import RobotSimulator

class ImprovedPathDataset(Dataset):
    def __init__(self, data_dir='training_data_improved'):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._load_data()
    
    def _load_data(self):
        if not self.data_dir.exists():
            print(f"데이터 디렉토리 {self.data_dir}가 존재하지 않습니다.")
            return
            
        data_files = list(self.data_dir.glob('*.npz'))
        print(f"총 {len(data_files)}개의 데이터 파일 발견")
        
        for data_file in tqdm(data_files, desc="데이터 로딩"):
            try:
                data = np.load(data_file)
                required_keys = ['cost_map', 'start_pos', 'goal_pos', 'path']
                if not all(key in data for key in required_keys):
                    continue
                
                sample = {
                    'cost_map': torch.FloatTensor(data['cost_map']),
                    'start_pos': torch.FloatTensor(data['start_pos']),
                    'goal_pos': torch.FloatTensor(data['goal_pos']),
                    'path': torch.FloatTensor(data['path'])
                }
                
                if self._validate_sample(sample):
                    self.samples.append(sample)
                    
            except Exception as e:
                continue
        
        print(f"총 {len(self.samples)}개의 유효한 샘플 로드됨")
    
    def _validate_sample(self, sample):
        try:
            if sample['cost_map'].shape != (120, 120):
                return False
            if sample['start_pos'].shape != (2,):
                return False
            if sample['goal_pos'].shape != (2,):
                return False
            if sample['path'].shape != (50, 2):
                return False
            if torch.any(torch.isnan(sample['path'])) or torch.any(torch.isinf(sample['path'])):
                return False
            return True
        except:
            return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class ImprovedDiPPeRModel(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=256, num_waypoints=50):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # 더 강력한 CNN 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 120x120 -> 60x60
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 60x60 -> 30x30
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 30x30 -> 15x15
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # 더 깊은 MLP 디코더
        feature_dim = 256 * 8 * 8
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, num_waypoints * 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cost_map, start_pos, goal_pos):
        batch_size = cost_map.size(0)
        
        # 시작점과 목표점을 맵에 추가
        start_channel = torch.zeros_like(cost_map).unsqueeze(1)  # (B, 1, 120, 120)
        goal_channel = torch.zeros_like(cost_map).unsqueeze(1)   # (B, 1, 120, 120)
        
        for i in range(batch_size):
            start_x = int((start_pos[i, 0] + 6) * 119 / 12)
            start_y = int((start_pos[i, 1] + 6) * 119 / 12)
            goal_x = int((goal_pos[i, 0] + 6) * 119 / 12)
            goal_y = int((goal_pos[i, 1] + 6) * 119 / 12)
            
            start_x = max(0, min(119, start_x))
            start_y = max(0, min(119, start_y))
            goal_x = max(0, min(119, goal_x))
            goal_y = max(0, min(119, goal_y))
            
            # 가우시안 분포로 점 표시
            y_coords, x_coords = torch.meshgrid(torch.arange(120), torch.arange(120), indexing='ij')
            
            start_dist = ((x_coords - start_x) ** 2 + (y_coords - start_y) ** 2).float()
            goal_dist = ((x_coords - goal_x) ** 2 + (y_coords - goal_y) ** 2).float()
            
            start_channel[i, 0] = torch.exp(-start_dist / 8)
            goal_channel[i, 0] = torch.exp(-goal_dist / 8)
        
        # 3채널 입력 생성
        input_tensor = torch.cat([
            cost_map.unsqueeze(1),  # (B, 1, 120, 120)
            start_channel,          # (B, 1, 120, 120)
            goal_channel            # (B, 1, 120, 120)
        ], dim=1)  # (B, 3, 120, 120)
        
        # CNN 인코딩
        features = self.encoder(input_tensor)
        features = features.view(batch_size, -1)
        
        # 위치 정보 추가
        pos_info = torch.cat([start_pos, goal_pos], dim=1)
        combined_features = torch.cat([features, pos_info], dim=1)
        
        # MLP 디코딩
        path_flat = self.decoder(combined_features)
        path = path_flat.view(batch_size, self.num_waypoints, 2)
        
        return path

def collect_improved_data(target_samples=8000):
    """개선된 학습 데이터 수집"""
    output_dir = Path('training_data_improved')
    output_dir.mkdir(exist_ok=True)
    
    # 수정된 시나리오들 사용
    scenarios = [
        'scenarios/Circulation1.xml',
        'scenarios/Circulation2.xml', 
        'scenarios/Congestion1.xml',
        'scenarios/Congestion2.xml'
    ]
    
    # 안전 구역 정의 (명확한 장애물 구조 기반)
    safe_zones = [
        (-4.5, -4.0, 1.0, 3.0),    # 왼쪽 위
        (-4.5, -4.0, -4.0, -2.0),  # 왼쪽 아래
        (-1.0, 1.0, -4.0, -3.0),   # 중앙 아래
        (-1.0, 1.0, 2.0, 4.0),     # 중앙 위
        (2.5, 3.5, -4.0, 2.0)      # 오른쪽
    ]
    
    sample_count = 0
    
    print(f"목표: {target_samples}개의 학습 데이터 수집")
    pbar = tqdm(total=target_samples, desc="데이터 수집")
    
    attempts = 0
    max_attempts = target_samples * 5
    
    while sample_count < target_samples and attempts < max_attempts:
        attempts += 1
        
        try:
            scenario = random.choice(scenarios)
            simulator = RobotSimulator(scenario, use_dipperp=False)
            
            # 안전한 시작/목표점 생성
            start_pos, goal_pos = generate_safe_positions(safe_zones, simulator)
            if start_pos is None:
                continue
            
            # A* 경로 계획
            path = simulator.astar_path_planning(start_pos, goal_pos)
            if path is None or len(path) < 10:
                continue
            
            # 50개 웨이포인트로 보간
            waypoints = interpolate_path(path, 50)
            if waypoints is None:
                continue
            
            # 비용 맵 생성
            cost_map = simulator.create_cost_map()
            
            # 데이터 저장
            filename = f"sample_{sample_count:06d}.npz"
            filepath = output_dir / filename
            
            np.savez_compressed(
                filepath,
                cost_map=cost_map,
                start_pos=start_pos,
                goal_pos=goal_pos,
                path=waypoints
            )
            
            sample_count += 1
            pbar.update(1)
            
        except Exception as e:
            continue
    
    pbar.close()
    print(f"총 {sample_count}개의 샘플 수집 완료")

def generate_safe_positions(safe_zones, simulator):
    """안전한 시작/목표점 생성"""
    max_attempts = 50
    for _ in range(max_attempts):
        # 80% 확률로 안전 구역에서 선택
        if random.random() < 0.8:
            zone = random.choice(safe_zones)
            start_x = random.uniform(zone[0], zone[1])
            start_y = random.uniform(zone[2], zone[3])
            
            zone = random.choice(safe_zones)
            goal_x = random.uniform(zone[0], zone[1])
            goal_y = random.uniform(zone[2], zone[3])
        else:
            start_x = random.uniform(-5.5, 5.5)
            start_y = random.uniform(-5.0, 5.0)
            goal_x = random.uniform(-5.5, 5.5)
            goal_y = random.uniform(-5.0, 5.0)
        
        start_pos = np.array([start_x, start_y])
        goal_pos = np.array([goal_x, goal_y])
        
        # 거리 체크
        if np.linalg.norm(goal_pos - start_pos) < 2.0:
            continue
        
        # 장애물 체크
        if (simulator._is_in_obstacle(start_pos) or 
            simulator._is_in_obstacle(goal_pos)):
            continue
        
        # 연결성 체크
        path = simulator.astar_path_planning(start_pos, goal_pos)
        if path is not None and len(path) >= 5:
            return start_pos, goal_pos
    
    return None, None

def interpolate_path(path, target_points):
    """경로를 지정된 개수의 점으로 보간"""
    if len(path) < 2:
        return None
    
    path_array = np.array(path)
    
    # 경로 길이 계산
    distances = np.cumsum([0] + [np.linalg.norm(path_array[i+1] - path_array[i]) 
                               for i in range(len(path_array)-1)])
    total_distance = distances[-1]
    
    if total_distance < 1e-6:
        return None
    
    # 균등한 간격으로 보간
    target_distances = np.linspace(0, total_distance, target_points)
    
    # 선형 보간
    interpolated_x = np.interp(target_distances, distances, path_array[:, 0])
    interpolated_y = np.interp(target_distances, distances, path_array[:, 1])
    
    return np.column_stack([interpolated_x, interpolated_y])

def train_improved_model():
    """개선된 모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터셋 로딩
    dataset = ImprovedPathDataset('training_data_improved')
    if len(dataset) == 0:
        print("학습 데이터가 없습니다!")
        return
    
    # 학습/검증 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 모델 생성
    model = ImprovedDiPPeRModel().to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 학습 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("학습 시작...")
    for epoch in range(100):
        # 학습
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            cost_map = batch['cost_map'].to(device)
            start_pos = batch['start_pos'].to(device)
            goal_pos = batch['goal_pos'].to(device)
            target_path = batch['path'].to(device)
            
            optimizer.zero_grad()
            predicted_path = model(cost_map, start_pos, goal_pos)
            loss = criterion(predicted_path, target_path)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                cost_map = batch['cost_map'].to(device)
                start_pos = batch['start_pos'].to(device)
                goal_pos = batch['goal_pos'].to(device)
                target_path = batch['path'].to(device)
                
                predicted_path = model(cost_map, start_pos, goal_pos)
                loss = criterion(predicted_path, target_path)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss - 0.005:
            improvement = (best_val_loss - val_loss) / best_val_loss * 100
            print(f"검증 손실 개선: {improvement:.2f}% -> 모델 저장")
            
            best_val_loss = val_loss
            patience_counter = 0
            
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'models/dipperp_improved.pth')
        else:
            patience_counter += 1
            print(f"개선 없음 ({patience_counter}/{patience})")
        
        # Early Stopping
        if patience_counter >= patience:
            print(f"Early Stopping at epoch {epoch+1}")
            break
    
    # 학습 그래프 저장
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("학습 완료!")
    print(f"최고 검증 손실: {best_val_loss:.6f}")

def main():
    print("=== 개선된 DiPPeR 학습 시스템 ===")
    
    # 1. 데이터 수집
    print("1. 학습 데이터 수집...")
    collect_improved_data(target_samples=8000)
    
    # 2. 모델 학습
    print("2. 모델 학습...")
    train_improved_model()
    
    print("완료!")

if __name__ == "__main__":
    main() 