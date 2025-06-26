#!/usr/bin/env python3
"""
CDIF (CCTV-Diffusion) Training Pipeline
- GPU A6000 서버 최적화
- 사회적 비용맵 기반 전략적 웨이포인트 학습
- 실시간 학습 모니터링 및 검증
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # 서버용 백엔드
import matplotlib.pyplot as plt
from tqdm import tqdm

# 프로젝트 모듈
from cdif_model import CDIFModel, DDPMScheduler
from robot_simulator_cgip import RobotSimulator

@dataclass
class CDIFConfig:
    """CDIF 학습 설정"""
    # 모델 설정
    max_waypoints: int = 8
    feature_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 6
    
    # 학습 설정
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Diffusion 설정
    num_train_timesteps: int = 1000
    beta_schedule: str = "cosine"
    
    # 데이터 설정
    target_samples: int = 10000
    train_ratio: float = 0.9
    
    # GPU 설정
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    
    # 체크포인트 설정
    save_every: int = 5
    validate_every: int = 2
    early_stopping_patience: int = 15

class StrategicWaypointExtractor:
    """전략적 웨이포인트 추출기"""
    
    def __init__(self, grid_size=0.2):
        self.grid_size = grid_size
        
    def extract_strategic_waypoints(self, start_pos, goal_pos, cost_map, max_waypoints=6):
        """병목지점 기반 전략적 웨이포인트 추출"""
        
        # 1. A* 기본 경로 생성
        simulator = RobotSimulator('scenarios/Circulation1.xml')
        astar_path = simulator.a_star(start_pos, goal_pos)
        
        if not astar_path or len(astar_path) < 2:
            return [start_pos, goal_pos]
        
        # 2. 경로를 균등하게 샘플링하여 전략적 웨이포인트 생성
        path_length = len(astar_path)
        if path_length <= max_waypoints:
            return astar_path
        
        # 3. 균등 간격으로 웨이포인트 선택
        indices = np.linspace(0, path_length - 1, max_waypoints, dtype=int)
        strategic_waypoints = [astar_path[i] for i in indices]
        
        # 4. 시작점과 끝점 보장
        strategic_waypoints[0] = start_pos
        strategic_waypoints[-1] = goal_pos
        
        return strategic_waypoints

class CDIFDataset(Dataset):
    """CDIF 학습 데이터셋"""
    
    def __init__(self, data_samples: List[Dict], config: CDIFConfig):
        self.samples = data_samples
        self.config = config
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 데이터 로드 (JSON에서 로드된 리스트를 NumPy로 변환)
        cost_map = torch.from_numpy(np.array(sample['cost_map'])).float().unsqueeze(0)  # [1, 60, 60]
        start_pos = torch.from_numpy(np.array(sample['start_pos'])).float()  # [2]
        goal_pos = torch.from_numpy(np.array(sample['goal_pos'])).float()  # [2]
        waypoints = np.array(sample['strategic_waypoints'])  # Convert to numpy array
        
        # 웨이포인트를 고정 길이로 패딩
        max_waypoints = self.config.max_waypoints
        num_waypoints = len(waypoints)
        
        if num_waypoints > max_waypoints:
            # 너무 많으면 잘라내기
            waypoints = waypoints[:max_waypoints]
            num_waypoints = max_waypoints
        else:
            # 부족하면 마지막 점으로 패딩
            waypoints_padded = waypoints.tolist()
            while len(waypoints_padded) < max_waypoints:
                waypoints_padded.append(waypoints_padded[-1])
            waypoints = np.array(waypoints_padded)
        
        waypoints_tensor = torch.from_numpy(waypoints).float()  # [max_waypoints, 2]
        num_waypoints_tensor = torch.tensor(num_waypoints, dtype=torch.long)
        
        return {
            'cost_map': cost_map,
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'waypoints': waypoints_tensor,
            'num_waypoints': num_waypoints_tensor
        }

class CDIFDataCollector:
    """CDIF 데이터 수집기"""
    
    def __init__(self, config: CDIFConfig, output_dir='training_data_cdif'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 시뮬레이터 초기화
        self.simulator = RobotSimulator('scenarios/Circulation1.xml')
        self.extractor = StrategicWaypointExtractor()
        
    def collect_data(self) -> List[Dict]:
        """전략적 웨이포인트 데이터 수집"""
        print(f"🎯 목표: {self.config.target_samples:,}개 CDIF 데이터 수집")
        
        collected_data = []
        
        # 안전한 위치 생성을 위한 영역 정의
        safe_zones = [
            (-5.5, -5.5, -0.5, -0.5),  # 좌하
            (-5.5, 0.5, -0.5, 5.5),    # 좌상
            (0.5, -5.5, 5.5, -0.5),    # 우하
            (0.5, 0.5, 5.5, 5.5),      # 우상
            (-2.0, -2.0, 2.0, 2.0)     # 중앙
        ]
        
        pbar = tqdm(total=self.config.target_samples, desc="CDIF 데이터 수집")
        
        while len(collected_data) < self.config.target_samples:
            # 랜덤 시작/목표점 생성
            start_pos = self._generate_safe_position(safe_zones)
            goal_pos = self._generate_safe_position(safe_zones)
            
            if start_pos is None or goal_pos is None:
                continue
            
            # 거리 체크 (최소 3m 이상)
            dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            if dist < 3.0:
                continue
            
            # 전략적 웨이포인트 추출
            strategic_waypoints = self.extractor.extract_strategic_waypoints(
                start_pos, goal_pos, self.simulator.fused_cost_map, 
                max_waypoints=self.config.max_waypoints
            )
            
            if len(strategic_waypoints) >= 2:
                # 데이터 정규화 (-6~6 → -1~1)
                data_sample = {
                    'cost_map': self.simulator.fused_cost_map.copy(),
                    'start_pos': np.array([start_pos[0]/6.0, start_pos[1]/6.0]),
                    'goal_pos': np.array([goal_pos[0]/6.0, goal_pos[1]/6.0]),
                    'strategic_waypoints': np.array([[p[0]/6.0, p[1]/6.0] for p in strategic_waypoints]),
                    'num_waypoints': len(strategic_waypoints)
                }
                
                collected_data.append(data_sample)
                pbar.update(1)
        
        pbar.close()
        
        # 데이터 저장
        save_path = self.output_dir / 'cdif_training_data.json'
        with open(save_path, 'w') as f:
            json.dump(collected_data, f, cls=NumpyEncoder, indent=2)
        
        print(f"✅ CDIF 데이터 수집 완료: {len(collected_data):,}개 샘플")
        print(f"💾 저장 위치: {save_path}")
        
        return collected_data
    
    def _generate_safe_position(self, safe_zones):
        """안전한 위치 생성"""
        for _ in range(50):
            zone = random.choice(safe_zones)
            x = random.uniform(zone[0], zone[2])
            y = random.uniform(zone[1], zone[3])
            
            # 그리드 기반 안전성 체크
            x_idx = int((x + 6) / self.simulator.grid_size)
            y_idx = int((y + 6) / self.simulator.grid_size)
            
            if (0 <= x_idx < 60 and 0 <= y_idx < 60 and 
                self.simulator.grid[y_idx, x_idx] == 0):
                return [x, y]
        
        return None

class NumpyEncoder(json.JSONEncoder):
    """NumPy 배열을 JSON으로 인코딩"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class CDIFTrainer:
    """CDIF 학습기"""
    
    def __init__(self, config: CDIFConfig, output_dir='models'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 사용 디바이스: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 모델 및 스케줄러 초기화
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
        
        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 학습률 스케줄러
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Mixed Precision 설정
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=f'runs/cdif_{int(time.time())}')
        
        # 학습 상태
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print(f"📊 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """CDIF 모델 학습"""
        print("🎓 CDIF 학습 시작!")
        
        # 데이터셋 생성
        train_dataset = CDIFDataset(train_data, self.config)
        val_dataset = CDIFDataset(val_data, self.config)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"📚 학습 데이터: {len(train_dataset):,}개")
        print(f"📖 검증 데이터: {len(val_dataset):,}개")
        
        # 학습 루프
        for epoch in range(self.config.num_epochs):
            # 학습
            train_loss = self._train_epoch(train_loader, epoch)
            
            # 검증
            if epoch % self.config.validate_every == 0:
                val_loss = self._validate_epoch(val_loader, epoch)
                
                # 최고 성능 모델 저장
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"🎯 새로운 최고 성능! Val Loss: {val_loss:.6f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"⏰ Early stopping at epoch {epoch+1}")
                    break
            
            # 정기적 체크포인트 저장
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)
            
            # 학습률 스케줄링
            self.lr_scheduler.step()
        
        print("✅ CDIF 학습 완료!")
        self.writer.close()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, batch in enumerate(pbar):
            # 데이터 GPU로 이동
            cost_map = batch['cost_map'].to(self.device)  # [B, 1, 60, 60]
            start_pos = batch['start_pos'].to(self.device)  # [B, 2]
            goal_pos = batch['goal_pos'].to(self.device)  # [B, 2]
            waypoints = batch['waypoints'].to(self.device)  # [B, max_waypoints, 2]
            num_waypoints = batch['num_waypoints'].to(self.device)  # [B]
            
            batch_size = cost_map.shape[0]
            
            # 랜덤 타임스텝 생성
            timesteps = torch.randint(
                0, self.config.num_train_timesteps, 
                (batch_size,), device=self.device
            )
            
            # 잡음 생성
            noise = torch.randn_like(waypoints)
            
            # 잡음 추가
            noisy_waypoints = self.scheduler.add_noise(waypoints, noise, timesteps)
            
            # 순전파
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                predicted_noise, num_waypoints_prob = self.model(
                    cost_map, noisy_waypoints, timesteps, start_pos, goal_pos
                )
                
                # 손실 계산
                noise_loss = F.mse_loss(predicted_noise, noise)
                
                # 웨이포인트 수 예측 손실
                num_targets = F.one_hot(num_waypoints - 1, num_classes=self.config.max_waypoints).float()
                num_loss = F.cross_entropy(num_waypoints_prob, num_targets.argmax(dim=1))
                
                total_loss_batch = noise_loss + 0.1 * num_loss
            
            # 역전파
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # 손실 누적
            total_loss += total_loss_batch.item()
            
            # 진행률 업데이트
            pbar.set_postfix({
                'Loss': f"{total_loss_batch.item():.6f}",
                'Noise': f"{noise_loss.item():.6f}",
                'Num': f"{num_loss.item():.6f}"
            })
            
            # TensorBoard 로깅
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', total_loss_batch.item(), self.global_step)
                self.writer.add_scalar('Train/NoiseLoss', noise_loss.item(), self.global_step)
                self.writer.add_scalar('Train/NumLoss', num_loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                # 데이터 GPU로 이동
                cost_map = batch['cost_map'].to(self.device)
                start_pos = batch['start_pos'].to(self.device)
                goal_pos = batch['goal_pos'].to(self.device)
                waypoints = batch['waypoints'].to(self.device)
                num_waypoints = batch['num_waypoints'].to(self.device)
                
                batch_size = cost_map.shape[0]
                
                # 랜덤 타임스텝
                timesteps = torch.randint(
                    0, self.config.num_train_timesteps,
                    (batch_size,), device=self.device
                )
                
                # 잡음 추가
                noise = torch.randn_like(waypoints)
                noisy_waypoints = self.scheduler.add_noise(waypoints, noise, timesteps)
                
                # 순전파
                predicted_noise, num_waypoints_prob = self.model(
                    cost_map, noisy_waypoints, timesteps, start_pos, goal_pos
                )
                
                # 손실 계산
                noise_loss = F.mse_loss(predicted_noise, noise)
                num_targets = F.one_hot(num_waypoints - 1, num_classes=self.config.max_waypoints).float()
                num_loss = F.cross_entropy(num_waypoints_prob, num_targets.argmax(dim=1))
                
                total_loss_batch = noise_loss + 0.1 * num_loss
                total_loss += total_loss_batch.item()
        
        avg_loss = total_loss / len(val_loader)
        
        # TensorBoard 로깅
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        print(f"📊 Epoch {epoch+1} - Val Loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        if is_best:
            save_path = self.output_dir / 'cdif_best.pth'
            torch.save(checkpoint, save_path)
            print(f"💾 최고 성능 모델 저장: {save_path}")
        
        # 정기적 체크포인트
        save_path = self.output_dir / f'cdif_epoch_{epoch+1}.pth'
        torch.save(checkpoint, save_path)

def main():
    parser = argparse.ArgumentParser(description='CDIF Training')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--data_only', action='store_true', help='Only collect data')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # 설정 로드
    config = CDIFConfig()
    
    print("🚀 CDIF 학습 파이프라인 시작!")
    print(f"⚙️  설정: {config}")
    
    # 1. 데이터 수집
    collector = CDIFDataCollector(config)
    
    # 기존 데이터 확인
    data_path = collector.output_dir / 'cdif_training_data.json'
    if data_path.exists():
        print(f"📂 기존 데이터 로드: {data_path}")
        with open(data_path, 'r') as f:
            all_data = json.load(f)
    else:
        print("📊 새로운 데이터 수집 시작...")
        all_data = collector.collect_data()
    
    if args.data_only:
        print("✅ 데이터 수집만 완료!")
        return
    
    # 2. 데이터 분할
    random.shuffle(all_data)
    split_idx = int(len(all_data) * config.train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"📚 학습/검증 데이터 분할: {len(train_data)}/{len(val_data)}")
    
    # 3. 학습
    trainer = CDIFTrainer(config)
    
    if args.resume:
        print(f"🔄 체크포인트에서 재개: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.global_step = checkpoint['global_step']
    
    trainer.train(train_data, val_data)

if __name__ == "__main__":
    main() 