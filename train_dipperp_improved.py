#!/usr/bin/env python3
"""
개선된 DiPPeR 학습 코드
- 다양한 학습 데이터 (기본 A*, 사회적 비용 강화, 우회 경로)
- 더 나은 모델 구조 및 학습 전략
- 안정적인 학습 및 검증 시스템
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import random
from collections import deque
import gc

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_simulator import RobotSimulator

class ImprovedPathDataset(Dataset):
    """개선된 경로 데이터셋"""
    
    def __init__(self, data_dir='training_data'):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._load_data()
    
    def _load_data(self):
        """학습 데이터 로드"""
        if not self.data_dir.exists():
            print(f"데이터 디렉토리 {self.data_dir}가 존재하지 않습니다.")
            return
            
        data_files = list(self.data_dir.glob('*.npz'))
        print(f"총 {len(data_files)}개의 데이터 파일 발견")
        
        for data_file in tqdm(data_files, desc="데이터 로딩"):
            try:
                data = np.load(data_file)
                
                # 필수 키 확인
                required_keys = ['cost_map', 'start_pos', 'goal_pos', 'path']
                if not all(key in data for key in required_keys):
                    continue
                
                sample = {
                    'cost_map': torch.FloatTensor(data['cost_map']),
                    'start_pos': torch.FloatTensor(data['start_pos']),
                    'goal_pos': torch.FloatTensor(data['goal_pos']),
                    'path': torch.FloatTensor(data['path'])
                }
                
                # 데이터 유효성 검사
                if self._validate_sample(sample):
                    self.samples.append(sample)
                    
            except Exception as e:
                print(f"파일 {data_file} 로딩 실패: {e}")
                continue
        
        print(f"총 {len(self.samples)}개의 유효한 샘플 로드됨")
    
    def _validate_sample(self, sample):
        """샘플 유효성 검사"""
        try:
            # 텐서 크기 확인
            if sample['cost_map'].shape != (120, 120):
                return False
            if sample['start_pos'].shape != (2,):
                return False
            if sample['goal_pos'].shape != (2,):
                return False
            if sample['path'].shape != (50, 2):
                return False
            
            # 값 범위 확인
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
    """개선된 DiPPeR 모델"""
    
    def __init__(self, input_channels=3, hidden_dim=256, num_waypoints=50):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        # CNN 인코더 (더 깊고 강력한 구조)
        self.encoder = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 120x120 -> 60x60
            
            # 두 번째 블록
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 60x60 -> 30x30
            
            # 세 번째 블록
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 30x30 -> 15x15
            
            # 네 번째 블록
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # 고정 크기로 변환
        )
        
        # 특징 차원 계산
        feature_dim = 256 * 8 * 8  # 16384
        
        # MLP 디코더 (더 깊고 강력한 구조)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + 4, hidden_dim * 2),  # +4 for start/goal pos
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
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
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
        start_channel = torch.zeros_like(cost_map[:, :1])
        goal_channel = torch.zeros_like(cost_map[:, :1])
        
        for i in range(batch_size):
            # 좌표를 그리드 인덱스로 변환 ([-6, 6] -> [0, 119])
            start_x = int((start_pos[i, 0] + 6) * 119 / 12)
            start_y = int((start_pos[i, 1] + 6) * 119 / 12)
            goal_x = int((goal_pos[i, 0] + 6) * 119 / 12)
            goal_y = int((goal_pos[i, 1] + 6) * 119 / 12)
            
            # 범위 체크
            start_x = max(0, min(119, start_x))
            start_y = max(0, min(119, start_y))
            goal_x = max(0, min(119, goal_x))
            goal_y = max(0, min(119, goal_y))
            
            # 가우시안 분포로 점 표시 (더 부드러운 표현)
            y_coords, x_coords = torch.meshgrid(torch.arange(120), torch.arange(120), indexing='ij')
            
            start_dist = ((x_coords - start_x) ** 2 + (y_coords - start_y) ** 2).float()
            goal_dist = ((x_coords - goal_x) ** 2 + (y_coords - goal_y) ** 2).float()
            
            start_channel[i, 0] = torch.exp(-start_dist / 8)  # 시그마=2
            goal_channel[i, 0] = torch.exp(-goal_dist / 8)
        
        # 3채널 입력 생성 (cost_map, start_channel, goal_channel)
        input_tensor = torch.cat([
            cost_map.unsqueeze(1),
            start_channel,
            goal_channel
        ], dim=1)
        
        # CNN 인코딩
        features = self.encoder(input_tensor)
        features = features.view(batch_size, -1)
        
        # 시작점과 목표점 정보 추가
        pos_info = torch.cat([start_pos, goal_pos], dim=1)
        combined_features = torch.cat([features, pos_info], dim=1)
        
        # MLP 디코딩
        path_flat = self.decoder(combined_features)
        path = path_flat.view(batch_size, self.num_waypoints, 2)
        
        return path

class DataCollector:
    """개선된 데이터 수집기"""
    
    def __init__(self, output_dir='training_data_improved'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 시뮬레이터 초기화
        self.scenarios = [
            'scenarios/Circulation1.xml',
            'scenarios/Circulation2.xml', 
            'scenarios/Congestion1.xml',
            'scenarios/Congestion2.xml'
        ]
        
        self.sample_count = 0
        
    def collect_diverse_data(self, target_samples=8000):
        """다양한 학습 데이터 수집"""
        print(f"목표: {target_samples}개의 다양한 학습 데이터 수집")
        
        samples_per_type = target_samples // 3
        
        # 1. 기본 A* 경로 (50%)
        print("1. 기본 A* 경로 수집...")
        self._collect_basic_paths(samples_per_type)
        
        # 2. 사회적 비용 강화 A* 경로 (30%) 
        print("2. 사회적 비용 강화 경로 수집...")
        self._collect_social_paths(samples_per_type)
        
        # 3. 우회 경로 (20%)
        print("3. 우회 경로 수집...")
        self._collect_detour_paths(target_samples - 2 * samples_per_type)
        
        print(f"총 {self.sample_count}개의 샘플 수집 완료")
    
    def _collect_basic_paths(self, target_count):
        """기본 A* 경로 수집"""
        collected = 0
        attempts = 0
        max_attempts = target_count * 3
        
        pbar = tqdm(total=target_count, desc="기본 A* 경로")
        
        while collected < target_count and attempts < max_attempts:
            attempts += 1
            
            try:
                scenario = random.choice(self.scenarios)
                simulator = RobotSimulator(scenario, use_dipperp=False)
                
                # 안전한 시작/목표점 생성
                start_pos, goal_pos = self._generate_safe_positions(simulator)
                if start_pos is None:
                    continue
                
                # A* 경로 계획
                path = simulator.astar_path_planning(start_pos, goal_pos)
                if path is None or len(path) < 10:
                    continue
                
                # 50개 웨이포인트로 보간
                waypoints = self._interpolate_path(path, 50)
                
                # 비용 맵 생성
                cost_map = simulator.create_cost_map()
                
                # 데이터 저장
                self._save_sample(cost_map, start_pos, goal_pos, waypoints, 'basic')
                collected += 1
                pbar.update(1)
                
            except Exception as e:
                continue
        
        pbar.close()
    
    def _collect_social_paths(self, target_count):
        """사회적 비용 강화 A* 경로 수집"""
        collected = 0
        attempts = 0
        max_attempts = target_count * 3
        
        pbar = tqdm(total=target_count, desc="사회적 비용 강화")
        
        while collected < target_count and attempts < max_attempts:
            attempts += 1
            
            try:
                scenario = random.choice(self.scenarios)
                simulator = RobotSimulator(scenario, use_dipperp=False)
                
                # 안전한 시작/목표점 생성
                start_pos, goal_pos = self._generate_safe_positions(simulator)
                if start_pos is None:
                    continue
                
                # 사회적 비용 강화 맵 생성
                social_cost_map = self._create_social_cost_map(simulator)
                
                # 강화된 비용 맵으로 A* 경로 계획
                path = simulator.astar_path_planning(start_pos, goal_pos, social_cost_map)
                if path is None or len(path) < 10:
                    continue
                
                # 50개 웨이포인트로 보간
                waypoints = self._interpolate_path(path, 50)
                
                # 원본 비용 맵 사용 (학습 시에는 원본 맵 사용)
                cost_map = simulator.create_cost_map()
                
                # 데이터 저장
                self._save_sample(cost_map, start_pos, goal_pos, waypoints, 'social')
                collected += 1
                pbar.update(1)
                
            except Exception as e:
                continue
        
        pbar.close()
    
    def _collect_detour_paths(self, target_count):
        """우회 경로 수집"""
        collected = 0
        attempts = 0
        max_attempts = target_count * 3
        
        pbar = tqdm(total=target_count, desc="우회 경로")
        
        while collected < target_count and attempts < max_attempts:
            attempts += 1
            
            try:
                scenario = random.choice(self.scenarios)
                simulator = RobotSimulator(scenario, use_dipperp=False)
                
                # 안전한 시작/목표점 생성
                start_pos, goal_pos = self._generate_safe_positions(simulator)
                if start_pos is None:
                    continue
                
                # 중간점을 거쳐가는 우회 경로 생성
                detour_path = self._create_detour_path(simulator, start_pos, goal_pos)
                if detour_path is None or len(detour_path) < 15:
                    continue
                
                # 50개 웨이포인트로 보간
                waypoints = self._interpolate_path(detour_path, 50)
                
                # 비용 맵 생성
                cost_map = simulator.create_cost_map()
                
                # 데이터 저장
                self._save_sample(cost_map, start_pos, goal_pos, waypoints, 'detour')
                collected += 1
                pbar.update(1)
                
            except Exception as e:
                continue
        
        pbar.close()
    
    def _generate_safe_positions(self, simulator):
        """안전한 시작/목표점 생성"""
        # Circulation1.xml 기준 안전 구역 정의
        safe_zones = [
            (-4.5, -4.0, 1.0, 3.0),    # 왼쪽 위
            (-4.5, -4.0, -4.0, -2.0),  # 왼쪽 아래
            (-1.0, 1.0, -4.0, -3.0),   # 중앙 아래
            (-1.0, 1.0, 2.0, 4.0),     # 중앙 위
            (2.5, 3.5, -4.0, 2.0)      # 오른쪽
        ]
        
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
                # 20% 확률로 전체 영역에서 선택
                start_x = random.uniform(-5.5, 5.5)
                start_y = random.uniform(-5.0, 5.0)
                goal_x = random.uniform(-5.5, 5.5)
                goal_y = random.uniform(-5.0, 5.0)
            
            start_pos = np.array([start_x, start_y])
            goal_pos = np.array([goal_x, goal_y])
            
            # 거리 체크 (너무 가깝지 않게)
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
    
    def _create_social_cost_map(self, simulator):
        """사회적 비용이 강화된 맵 생성"""
        cost_map = simulator.create_cost_map()
        
        # 장애물 주변에 더 높은 비용 부여
        enhanced_map = cost_map.copy()
        
        # 장애물 근처 비용 증가
        obstacle_mask = cost_map > 0.5
        
        # 확장 커널 (장애물 주변 영역)
        from scipy import ndimage
        expanded_obstacles = ndimage.binary_dilation(obstacle_mask, iterations=3)
        
        # 장애물 주변에 높은 비용 부여
        enhanced_map[expanded_obstacles & ~obstacle_mask] = 0.8
        
        return enhanced_map
    
    def _create_detour_path(self, simulator, start_pos, goal_pos):
        """중간점을 거쳐가는 우회 경로 생성"""
        # 안전한 중간점 생성
        safe_zones = [
            (-4.5, -4.0, 1.0, 3.0),
            (-4.5, -4.0, -4.0, -2.0),
            (-1.0, 1.0, -4.0, -3.0),
            (-1.0, 1.0, 2.0, 4.0),
            (2.5, 3.5, -4.0, 2.0)
        ]
        
        for _ in range(10):
            zone = random.choice(safe_zones)
            mid_x = random.uniform(zone[0], zone[1])
            mid_y = random.uniform(zone[2], zone[3])
            mid_pos = np.array([mid_x, mid_y])
            
            if simulator._is_in_obstacle(mid_pos):
                continue
            
            # 시작점 -> 중간점 -> 목표점 경로 생성
            path1 = simulator.astar_path_planning(start_pos, mid_pos)
            path2 = simulator.astar_path_planning(mid_pos, goal_pos)
            
            if path1 is not None and path2 is not None:
                # 두 경로 연결
                full_path = path1 + path2[1:]  # 중복 제거
                return full_path
        
        return None
    
    def _interpolate_path(self, path, target_points):
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
    
    def _save_sample(self, cost_map, start_pos, goal_pos, waypoints, data_type):
        """샘플 저장"""
        filename = f"sample_{self.sample_count:06d}_{data_type}.npz"
        filepath = self.output_dir / filename
        
        np.savez_compressed(
            filepath,
            cost_map=cost_map,
            start_pos=start_pos,
            goal_pos=goal_pos,
            path=waypoints,
            data_type=data_type
        )
        
        self.sample_count += 1

class ImprovedTrainer:
    """개선된 학습기"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 손실 함수 (SmoothL1Loss는 아웃라이어에 더 강함)
        self.criterion = nn.SmoothL1Loss()
        
        # 옵티마이저 (더 낮은 학습률)
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=5e-5, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 스케줄러 (Cosine Annealing with Warm Restarts)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Early Stopping
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        self.min_improvement = 0.005  # 0.5% 이상 개선 필요
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            cost_map = batch['cost_map'].to(self.device)
            start_pos = batch['start_pos'].to(self.device)
            goal_pos = batch['goal_pos'].to(self.device)
            target_path = batch['path'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_path = self.model(cost_map, start_pos, goal_pos)
            
            # Loss 계산
            loss = self.criterion(predicted_path, target_path)
            
            # Backward pass
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        """검증"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                cost_map = batch['cost_map'].to(self.device)
                start_pos = batch['start_pos'].to(self.device)
                goal_pos = batch['goal_pos'].to(self.device)
                target_path = batch['path'].to(self.device)
                
                predicted_path = self.model(cost_map, start_pos, goal_pos)
                loss = self.criterion(predicted_path, target_path)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs=100):
        """전체 학습 과정"""
        print(f"학습 시작: {num_epochs} 에포크")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 학습
            train_loss = self.train_epoch()
            
            # 검증
            val_loss = self.validate()
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 기록 저장
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # 최고 성능 모델 저장
            if val_loss < self.best_val_loss - self.min_improvement:
                improvement = (self.best_val_loss - val_loss) / self.best_val_loss * 100
                print(f"검증 손실 개선: {improvement:.2f}% -> 모델 저장")
                
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 모델 저장
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, 'models/dipperp_best_improved.pth')
                
            else:
                self.patience_counter += 1
                print(f"개선 없음 ({self.patience_counter}/{self.patience})")
            
            # Early Stopping
            if self.patience_counter >= self.patience:
                print(f"Early Stopping at epoch {epoch+1}")
                break
            
            # 메모리 정리
            if epoch % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 학습 완료 후 그래프 그리기
        self._plot_training_history()
        
        print("학습 완료!")
        print(f"최고 검증 손실: {self.best_val_loss:.6f}")
    
    def _plot_training_history(self):
        """학습 기록 시각화"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """메인 함수"""
    print("=== 개선된 DiPPeR 학습 시작 ===")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 1. 데이터 수집
    print("\n1. 다양한 학습 데이터 수집...")
    collector = DataCollector()
    collector.collect_diverse_data(target_samples=8000)
    
    # 2. 데이터셋 생성
    print("\n2. 데이터셋 로딩...")
    dataset = ImprovedPathDataset('training_data_improved')
    
    if len(dataset) == 0:
        print("학습 데이터가 없습니다!")
        return
    
    # 학습/검증 분할 (80:20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 3. 모델 생성
    print("\n3. 개선된 모델 생성...")
    model = ImprovedDiPPeRModel().to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터: {total_params:,}")
    print(f"학습 가능한 파라미터: {trainable_params:,}")
    
    # 4. 학습
    print("\n4. 학습 시작...")
    trainer = ImprovedTrainer(model, train_loader, val_loader, device)
    trainer.train(num_epochs=100)
    
    print("\n=== 학습 완료 ===")

if __name__ == "__main__":
    main() 