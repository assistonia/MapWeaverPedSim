#!/usr/bin/env python3
"""
GPU 서버용 최적화된 DiPPeR 학습 코드
- 대용량 배치 처리
- 멀티 GPU 지원
- 고성능 데이터로더
- 메모리 최적화
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # 서버용 (GUI 없음)
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import random
import gc
import time
from torch.nn.parallel import DataParallel

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_simuator_dippeR import DiPPeR, RobotSimulatorDiPPeR

class GPUOptimizedDataset(Dataset):
    """GPU 최적화된 데이터셋"""
    
    def __init__(self, data_dir='training_data_gpu'):
        self.data_dir = Path(data_dir)
        self.samples = []
        self._load_data()
        
        # GPU 메모리 최적화를 위한 텐서 변환
        print("🚀 GPU 최적화를 위한 텐서 사전 변환...")
        self._preprocess_for_gpu()
    
    def _load_data(self):
        """다양한 데이터 소스에서 로드"""
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
    
    def _preprocess_for_gpu(self):
        """GPU 처리를 위한 사전 변환"""
        processed_samples = []
        
        for sample in tqdm(self.samples, desc="GPU 최적화"):
            try:
                # 텐서 변환 및 정규화
                cost_map = torch.from_numpy(np.array(sample['cost_map'])).float()
                start_pos = torch.from_numpy(np.array(sample['start_pos'])).float()
                goal_pos = torch.from_numpy(np.array(sample['goal_pos'])).float()
                path = torch.from_numpy(np.array(sample['path'])).float()
                
                processed_samples.append({
                    'cost_map': cost_map,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'path': path,
                    'path_type': sample.get('path_type', 'unknown')
                })
            except Exception as e:
                print(f"샘플 변환 실패: {e}")
                continue
        
        self.samples = processed_samples
        print(f"GPU 최적화 완료: {len(self.samples)}개 샘플")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미 텐서로 변환된 데이터 반환
        cost_map = sample['cost_map'].unsqueeze(0)  # (1, 60, 60)
        start_pos = sample['start_pos']  # (2,)
        goal_pos = sample['goal_pos']  # (2,)
        path = sample['path']  # (path_length, 2)
        
        return cost_map, start_pos, goal_pos, path

class GPUDataCollector:
    """GPU 서버용 대용량 데이터 수집기"""
    
    def __init__(self, xml_file, output_dir='training_data_gpu'):
        self.simulator = RobotSimulatorDiPPeR(xml_file, model_path=None)
        self.simulator.use_dipperp = False  # A* 사용
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def collect_massive_data(self, target_samples=50000):
        """대용량 데이터 수집 (GPU 서버용)"""
        print(f"🎯 목표: {target_samples:,}개 샘플 수집")
        
        collected_data = []
        batch_size = 1000  # 1000개씩 배치로 저장
        batch_count = 0
        
        pbar = tqdm(total=target_samples, desc="데이터 수집")
        
        while len(collected_data) < target_samples:
            # 랜덤 시작/목표점 생성
            start_pos = self._generate_safe_position()
            goal_pos = self._generate_safe_position()
            
            if start_pos is None or goal_pos is None:
                continue
            
            # 거리 체크 (너무 가깝지 않게)
            dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            if dist < 2.0:
                continue
            
            # 다양한 경로 타입 생성
            episode_data = self._collect_episode_data(start_pos, goal_pos)
            collected_data.extend(episode_data)
            
            pbar.update(len(episode_data))
            
            # 배치 저장 (메모리 관리)
            if len(collected_data) >= batch_size:
                self._save_batch(collected_data[:batch_size], batch_count)
                collected_data = collected_data[batch_size:]
                batch_count += 1
                
                # 메모리 정리
                gc.collect()
        
        # 남은 데이터 저장
        if collected_data:
            self._save_batch(collected_data, batch_count)
        
        pbar.close()
        print(f"✅ 데이터 수집 완료: {batch_count + 1}개 배치 파일")
    
    def _generate_safe_position(self):
        """안전한 위치 생성"""
        # 안전 구역 정의
        safe_zones = [
            (-5.5, -5.5, -0.5, -0.5),  # 좌하단
            (-5.5, 0.5, -0.5, 5.5),    # 좌상단
            (0.5, -5.5, 5.5, -0.5),    # 우하단
            (0.5, 0.5, 5.5, 5.5),      # 우상단
            (-2.0, -2.0, 2.0, 2.0)     # 중앙
        ]
        
        for _ in range(50):
            zone = random.choice(safe_zones)
            x = random.uniform(zone[0], zone[2])
            y = random.uniform(zone[1], zone[3])
            
            if self.simulator.is_position_safe([x, y]):
                return [x, y]
        
        return None
    
    def _collect_episode_data(self, start_pos, goal_pos):
        """한 에피소드에서 다양한 경로 데이터 수집"""
        episode_data = []
        
        # 1. 기본 A* 경로 (40%)
        if random.random() < 0.4:
            path = self.simulator.fallback_astar_planning(start_pos, goal_pos)
            if path and len(path) > 5:
                data = self._create_data_sample(start_pos, goal_pos, path, "astar")
                if data:
                    episode_data.append(data)
        
        # 2. 사회적 비용 강화 경로 (35%)
        if random.random() < 0.35:
            path = self._generate_social_aware_path(start_pos, goal_pos)
            if path and len(path) > 5:
                data = self._create_data_sample(start_pos, goal_pos, path, "social")
                if data:
                    episode_data.append(data)
        
        # 3. 우회 경로 (25%)
        if random.random() < 0.25:
            path = self._generate_detour_path(start_pos, goal_pos)
            if path and len(path) > 5:
                data = self._create_data_sample(start_pos, goal_pos, path, "detour")
                if data:
                    episode_data.append(data)
        
        return episode_data
    
    def _create_data_sample(self, start_pos, goal_pos, path, path_type):
        """데이터 샘플 생성"""
        try:
            # 경로를 50개로 리샘플링
            if len(path) >= 50:
                indices = np.linspace(0, len(path)-1, 50, dtype=int)
                resampled_path = [path[i] for i in indices]
            else:
                resampled_path = self._interpolate_path(path, 50)
            
            # 안전성 검증
            for point in resampled_path:
                if not self.simulator.is_position_safe(point):
                    return None
            
            # 데이터 샘플 생성
            return {
                'cost_map': self.simulator.fused_cost_map.copy(),
                'start_pos': np.array([start_pos[0]/6.0, start_pos[1]/6.0]),  # 정규화
                'goal_pos': np.array([goal_pos[0]/6.0, goal_pos[1]/6.0]),    # 정규화
                'path': np.array([[p[0]/6.0, p[1]/6.0] for p in resampled_path]),  # 정규화
                'path_type': path_type
            }
        except Exception as e:
            return None
    
    def _interpolate_path(self, path, target_length):
        """경로 선형 보간"""
        if len(path) < 2:
            return path
        
        path_array = np.array(path)
        distances = np.cumsum([0] + [np.linalg.norm(path_array[i+1] - path_array[i]) 
                                     for i in range(len(path)-1)])
        total_distance = distances[-1]
        
        target_distances = np.linspace(0, total_distance, target_length)
        resampled_path = []
        
        for target_dist in target_distances:
            idx = np.searchsorted(distances, target_dist)
            if idx == 0:
                resampled_path.append(path[0])
            elif idx >= len(path):
                resampled_path.append(path[-1])
            else:
                t = (target_dist - distances[idx-1]) / (distances[idx] - distances[idx-1])
                interpolated = [(1-t) * path[idx-1][j] + t * path[idx][j] for j in range(2)]
                resampled_path.append(interpolated)
        
        return resampled_path
    
    def _generate_social_aware_path(self, start_pos, goal_pos):
        """사회적 비용 강화 경로"""
        original_map = self.simulator.fused_cost_map.copy()
        
        # 에이전트 주변 비용 강화
        for agent in self.simulator.agents:
            x_idx = int((agent.pos[0] + 6) / self.simulator.grid_size)
            y_idx = int((agent.pos[1] + 6) / self.simulator.grid_size)
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x_idx + dx, y_idx + dy
                    if 0 <= nx < 60 and 0 <= ny < 60:
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance <= 3:
                            penalty = 0.8 * (1 - distance/3)
                            self.simulator.fused_cost_map[ny, nx] = min(
                                self.simulator.fused_cost_map[ny, nx] + penalty, 0.95
                            )
        
        path = self.simulator.fallback_astar_planning(start_pos, goal_pos)
        self.simulator.fused_cost_map = original_map
        return path
    
    def _generate_detour_path(self, start_pos, goal_pos):
        """우회 경로 생성"""
        mid_x = (start_pos[0] + goal_pos[0]) / 2
        mid_y = (start_pos[1] + goal_pos[1]) / 2
        
        for _ in range(10):
            waypoint = [
                mid_x + random.uniform(-3.0, 3.0),
                mid_y + random.uniform(-3.0, 3.0)
            ]
            
            if self.simulator.is_position_safe(waypoint):
                path1 = self.simulator.fallback_astar_planning(start_pos, waypoint)
                path2 = self.simulator.fallback_astar_planning(waypoint, goal_pos)
                
                if path1 and path2 and len(path1) > 1 and len(path2) > 1:
                    return path1 + path2[1:]
        
        return self.simulator.fallback_astar_planning(start_pos, goal_pos)
    
    def _save_batch(self, data, batch_id):
        """배치 데이터 저장"""
        filename = self.output_dir / f"batch_{batch_id:04d}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder)
        print(f"배치 저장: {filename} ({len(data)}개 샘플)")

class NumpyEncoder(json.JSONEncoder):
    """NumPy 배열을 JSON으로 인코딩"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class GPUDiPPeRTrainer:
    """GPU 최적화된 DiPPeR 학습기"""
    
    def __init__(self, model, device='cuda'):
        self.device = torch.device(device)
        
        # 멀티 GPU 지원
        if torch.cuda.device_count() > 1:
            print(f"🚀 {torch.cuda.device_count()}개 GPU 감지 - DataParallel 사용")
            self.model = DataParallel(model)
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # GPU 최적화된 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-4,  # GPU에서 더 높은 학습률
            weight_decay=1e-6,
            eps=1e-8
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=5e-4,
            epochs=200,
            steps_per_epoch=100,  # 추정값
            pct_start=0.1
        )
        
        # 혼합 정밀도 학습 (GPU 메모리 절약)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 20
        
        # GPU 최적화
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    def train_step(self, batch):
        """GPU 최적화된 학습 스텝"""
        cost_maps, start_pos, goal_pos, paths = batch
        batch_size = cost_maps.shape[0]
        
        # GPU로 이동 (non_blocking=True로 성능 향상)
        cost_maps = cost_maps.to(self.device, non_blocking=True)
        start_pos = start_pos.to(self.device, non_blocking=True)
        goal_pos = goal_pos.to(self.device, non_blocking=True)
        paths = paths.to(self.device, non_blocking=True)
        
        # 혼합 정밀도 학습
        with torch.cuda.amp.autocast():
            # 랜덤 타임스텝
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
            
            # 노이즈 추가
            noise = torch.randn_like(paths)
            
            # DiPPeR 모델에서 alphas_cumprod 가져오기
            if hasattr(self.model, 'module'):
                alphas_cumprod = self.model.module.alphas_cumprod
            else:
                alphas_cumprod = self.model.alphas_cumprod
            
            alpha_cumprod = alphas_cumprod[timesteps].view(-1, 1, 1)
            noisy_paths = torch.sqrt(alpha_cumprod) * paths + torch.sqrt(1 - alpha_cumprod) * noise
            
            # 조건 입력
            start_goal_pos = torch.cat([start_pos, goal_pos], dim=-1)
            
            # 노이즈 예측
            predicted_noise = self.model(cost_maps, noisy_paths, timesteps, start_goal_pos)
            
            # 손실 계산
            loss = nn.MSELoss()(predicted_noise, noise)
        
        # 역전파 (혼합 정밀도)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """에포크 학습"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="학습 중")
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            
            pbar.set_postfix({'Loss': f'{loss:.6f}'})
            
            # 스케줄러 업데이트
            self.scheduler.step()
        
        return total_loss / len(dataloader)
    
    def save_model(self, path, epoch=None, loss=None):
        """모델 저장"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        
        torch.save(checkpoint, path)
        print(f"모델 저장: {path}")

def main():
    """GPU 서버용 메인 함수"""
    print("=== GPU 최적화 DiPPeR 학습 ===")
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다!")
        return
    
    device = torch.device('cuda')
    print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 1. 대용량 데이터 수집
    print("\n1. 대용량 데이터 수집...")
    collector = GPUDataCollector('scenarios/Circulation1.xml')
    collector.collect_massive_data(target_samples=50000)
    
    # 2. 데이터셋 로드
    print("\n2. GPU 최적화 데이터셋 로드...")
    dataset = GPUOptimizedDataset('training_data_gpu')
    
    # 학습/검증 분할
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 고성능 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # GPU용 대용량 배치
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"학습 데이터: {len(train_dataset):,}개")
    print(f"검증 데이터: {len(val_dataset):,}개")
    
    # 3. 모델 및 학습기 생성
    print("\n3. GPU 최적화 모델 생성...")
    model = DiPPeR(visual_feature_dim=512, path_dim=2, max_timesteps=1000)
    trainer = GPUDiPPeRTrainer(model, device='cuda')
    
    # 4. 학습 시작
    print("\n4. GPU 고속 학습 시작...")
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(200):
        # 학습
        train_loss = trainer.train_epoch(train_loader)
        
        # 검증 (간단히)
        trainer.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                cost_maps, start_pos, goal_pos, paths = batch
                cost_maps = cost_maps.to(device, non_blocking=True)
                start_pos = start_pos.to(device, non_blocking=True)
                goal_pos = goal_pos.to(device, non_blocking=True)
                paths = paths.to(device, non_blocking=True)
                
                batch_size = cost_maps.shape[0]
                timesteps = torch.randint(0, 1000, (batch_size,), device=device)
                noise = torch.randn_like(paths)
                
                if hasattr(trainer.model, 'module'):
                    alphas_cumprod = trainer.model.module.alphas_cumprod
                else:
                    alphas_cumprod = trainer.model.alphas_cumprod
                
                alpha_cumprod = alphas_cumprod[timesteps].view(-1, 1, 1)
                noisy_paths = torch.sqrt(alpha_cumprod) * paths + torch.sqrt(1 - alpha_cumprod) * noise
                start_goal_pos = torch.cat([start_pos, goal_pos], dim=-1)
                
                predicted_noise = trainer.model(cost_maps, noisy_paths, timesteps, start_goal_pos)
                loss = nn.MSELoss()(predicted_noise, noise)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/200: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # 최고 성능 모델 저장
        if val_loss < trainer.best_loss:
            trainer.best_loss = val_loss
            trainer.patience_counter = 0
            trainer.save_model(f'models/dipperp_gpu_best.pth', epoch, val_loss)
            print(f"🎯 새로운 최고 성능: {val_loss:.6f}")
        else:
            trainer.patience_counter += 1
        
        # Early stopping
        if trainer.patience_counter >= trainer.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            trainer.save_model(f'models/dipperp_gpu_epoch_{epoch+1}.pth', epoch, val_loss)
    
    print("✅ GPU 학습 완료!")

if __name__ == "__main__":
    main() 