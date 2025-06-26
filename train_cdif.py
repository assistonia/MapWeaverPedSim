#!/usr/bin/env python3
"""
CDIF: CCTV-informed Diffusion Training Pipeline

핵심 혁신:
- 편재하는 CCTV 인프라의 전략적 활용을 통한 제로-비용 센서 확장
- 정적 장애물과 동적 사회적 요소를 단일 표현 공간으로 통합하는 환경 모델링
- 확률적 다중 경로 후보 동시 생성을 위한 다중 모달 학습
- 실시간 적응형 내비게이션을 위한 계층적 융합 구조

차별화 포인트:
- vs CGIP: 결정적 단일 해 → 확률적 다중 경로 생성
- vs DiPPeR-Legged: 단순 장애물 회피 → 사회적 맥락 인식
- 기존 인프라 활용 + 첨단 생성 모델링 기법 융합
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
    """CDIF 학습 설정 - 사회적 맥락 인식 다중 경로 생성"""
    # 모델 설정
    max_waypoints: int = 8
    feature_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 6
    num_path_modes: int = 3  # 경로 모드: 0=직접, 1=사회적, 2=우회
    
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

class SocialContextWaypointExtractor:
    """사회적 맥락 인식 웨이포인트 추출기 - 다중 모달 경로 생성"""
    
    def __init__(self, grid_size=0.2):
        self.grid_size = grid_size
        # 경로 스타일 정의
        self.path_styles = {
            0: "direct",     # 직접 경로 (최단거리)
            1: "social",     # 사회적 경로 (사람 회피)
            2: "detour"      # 우회 경로 (안전 우선)
        }
        
    def extract_multimodal_waypoints(self, start_pos, goal_pos, simulator, max_waypoints=6):
        """다중 모달 웨이포인트 추출 - 사회적 맥락별 다양한 경로 스타일"""
        
        # 기본 A* 경로 (직접 경로)
        direct_path = simulator.a_star(start_pos, goal_pos)
        if not direct_path or len(direct_path) < 2:
            return {0: [start_pos, goal_pos], 1: [start_pos, goal_pos], 2: [start_pos, goal_pos]}
        
        # 다중 모달 경로 생성
        multimodal_paths = {}
        
        # 모드 0: 직접 경로 (최단거리)
        multimodal_paths[0] = self._sample_waypoints(direct_path, max_waypoints)
        
        # 모드 1: 사회적 경로 (사람 밀도 회피)
        social_path = self._generate_social_aware_path(start_pos, goal_pos, simulator)
        multimodal_paths[1] = self._sample_waypoints(social_path, max_waypoints)
        
        # 모드 2: 우회 경로 (안전 우선)
        detour_path = self._generate_detour_path(start_pos, goal_pos, simulator)
        multimodal_paths[2] = self._sample_waypoints(detour_path, max_waypoints)
        
        return multimodal_paths
    
    def _sample_waypoints(self, path, max_waypoints):
        """경로에서 웨이포인트 균등 샘플링"""
        if len(path) <= max_waypoints:
            return path
        
        indices = np.linspace(0, len(path) - 1, max_waypoints, dtype=int)
        sampled_waypoints = [path[i] for i in indices]
        return sampled_waypoints
    
    def _generate_social_aware_path(self, start_pos, goal_pos, simulator):
        """사회적 인식 경로 생성 (Individual Space 영역 회피)"""
        # 사회적 비용맵 기반 A* 경로 계획
        return self._social_cost_astar(start_pos, goal_pos, simulator)
    
    def _generate_detour_path(self, start_pos, goal_pos, simulator):
        """우회 경로 생성 (장애물 마진 증가)"""
        # 장애물 주변 마진을 증가시킨 A* 경로 계획
        return self._safe_margin_astar(start_pos, goal_pos, simulator)
    
    def _social_cost_astar(self, start_pos, goal_pos, simulator):
        """사회적 비용을 고려한 A* 경로 계획"""
        start = (int((start_pos[0] + 6) / simulator.grid_size), int((start_pos[1] + 6) / simulator.grid_size))
        goal = (int((goal_pos[0] + 6) / simulator.grid_size), int((goal_pos[1] + 6) / simulator.grid_size))
        
        # 60x60 그리드 기반 동적 장애물 맵 생성
        dynamic_grid = simulator.grid.copy()
        
        # 에이전트를 동적 장애물로 추가 (60x60 그리드)
        for agent in simulator.agents:
            x_idx = int((agent.pos[0] + 6) / simulator.grid_size)
            y_idx = int((agent.pos[1] + 6) / simulator.grid_size)
            radius_idx = int(agent.radius / simulator.grid_size)
            
            for i in range(-radius_idx, radius_idx + 1):
                for j in range(-radius_idx, radius_idx + 1):
                    if 0 <= x_idx + i < 60 and 0 <= y_idx + j < 60:
                        if i*i + j*j <= radius_idx*radius_idx:
                            dynamic_grid[y_idx + j, x_idx + i] = 1
        
        # 사회적 비용이 높은 영역을 장애물로 처리
        social_map = np.zeros((60, 60))
        for agent in simulator.agents:
            if not agent.finished and simulator.is_in_cctv_coverage(agent.pos):
                for i in range(60):
                    for j in range(60):
                        x = (j * simulator.grid_size) - 6
                        y = (i * simulator.grid_size) - 6
                        is_value = agent.calculate_individual_space([x, y])
                        social_map[i, j] = max(social_map[i, j], is_value)
        
        # 사회적 비용이 높은 영역을 장애물로 처리 (임계값: 0.3)
        social_obstacle_mask = social_map > 0.3
        dynamic_grid = np.logical_or(dynamic_grid, social_obstacle_mask).astype(int)
        
        return self._astar_with_cost_map(start, goal, dynamic_grid, simulator)
    
    def _safe_margin_astar(self, start_pos, goal_pos, simulator):
        """안전 마진을 증가시킨 A* 경로 계획"""
        start = (int((start_pos[0] + 6) / simulator.grid_size), int((start_pos[1] + 6) / simulator.grid_size))
        goal = (int((goal_pos[0] + 6) / simulator.grid_size), int((goal_pos[1] + 6) / simulator.grid_size))
        
        # 장애물 마진 확장
        expanded_grid = self._expand_obstacles(simulator.grid, margin=3)
        
        # 60x60 그리드 기반 동적 장애물 추가
        for agent in simulator.agents:
            x_idx = int((agent.pos[0] + 6) / simulator.grid_size)
            y_idx = int((agent.pos[1] + 6) / simulator.grid_size)
            radius_idx = int(agent.radius / simulator.grid_size)
            
            for i in range(-radius_idx, radius_idx + 1):
                for j in range(-radius_idx, radius_idx + 1):
                    if 0 <= x_idx + i < 60 and 0 <= y_idx + j < 60:
                        if i*i + j*j <= radius_idx*radius_idx:
                            expanded_grid[y_idx + j, x_idx + i] = 1
        
        return self._astar_with_cost_map(start, goal, expanded_grid, simulator)
    
    def _expand_obstacles(self, grid, margin=2):
        """장애물 주변에 마진 추가 (수동 구현)"""
        expanded = grid.copy()
        h, w = grid.shape
        
        for i in range(h):
            for j in range(w):
                if grid[i, j] == 1:  # 장애물인 경우
                    # 주변 마진 영역에 장애물 표시
                    for di in range(-margin, margin + 1):
                        for dj in range(-margin, margin + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                expanded[ni, nj] = 1
        return expanded
    
    def _astar_with_cost_map(self, start, goal, cost_grid, simulator):
        """비용맵을 사용한 A* 경로 계획"""
        from heapq import heappush, heappop
        
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self._get_neighbors_with_cost(current, cost_grid):
                new_cost = cost_so_far[current] + self._movement_cost(current, next_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + simulator.heuristic(goal, next_pos)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # 경로 재구성
        path = []
        current = goal
        while current is not None:
            x = current[0] * simulator.grid_size - 6
            y = current[1] * simulator.grid_size - 6
            path.append([x, y])
            current = came_from.get(current)
        path.reverse()
        return path if path else [start_pos, goal_pos]
    
    def _get_neighbors_with_cost(self, pos, cost_grid):
        """비용맵 기반 이웃 노드 탐색"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
            new_x = x + dx
            new_y = y + dy
            if (0 <= new_x < cost_grid.shape[1] and 0 <= new_y < cost_grid.shape[0] and 
                cost_grid[new_y, new_x] == 0):
                neighbors.append((new_x, new_y))
        return neighbors
    
    def _movement_cost(self, current, next_pos):
        """이동 비용 계산 (대각선 이동 고려)"""
        dx = abs(next_pos[0] - current[0])
        dy = abs(next_pos[1] - current[1])
        return 1.414 if (dx + dy) == 2 else 1.0

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
        integrated_cost_map = torch.from_numpy(np.array(sample['integrated_cost_map'])).float()  # [3, 60, 60]
        start_pos = torch.from_numpy(np.array(sample['start_pos'])).float()  # [2]
        goal_pos = torch.from_numpy(np.array(sample['goal_pos'])).float()  # [2]
        waypoints = np.array(sample['strategic_waypoints'])  # Convert to numpy array
        path_mode = sample.get('path_mode', 0)  # 경로 모드 (기본값: 0)
        
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
        path_mode_tensor = torch.tensor(path_mode, dtype=torch.long)
        
        return {
            'integrated_cost_map': integrated_cost_map,  # 3채널 통합 비용맵
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'waypoints': waypoints_tensor,
            'num_waypoints': num_waypoints_tensor,
            'path_mode': path_mode_tensor
        }

class CDIFDataCollector:
    """CDIF 데이터 수집기"""
    
    def __init__(self, config: CDIFConfig, output_dir='training_data_cdif'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 시뮬레이터 초기화 (사회적 맥락을 위해 Congestion1.xml 사용)
        self.simulator = RobotSimulator('scenarios/Congestion1.xml')
        self.extractor = SocialContextWaypointExtractor()
        
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
            
            # 🚨 사회적 맥락 생성을 위해 시뮬레이션 업데이트
            # 에이전트들을 업데이트하여 사회적 비용맵 생성
            for agent in self.simulator.agents:
                agent.update(self.simulator.agents, self.simulator.obstacles)
            self.simulator.update()  # Individual Space 맵 업데이트
            
            # 다중 모달 웨이포인트 추출 (사회적 맥락 고려)
            multimodal_waypoints = self.extractor.extract_multimodal_waypoints(
                start_pos, goal_pos, self.simulator, 
                max_waypoints=self.config.max_waypoints
            )
            
            # 랜덤하게 하나의 모드 선택 (학습 데이터 다양성)
            selected_mode = random.randint(0, 2)
            strategic_waypoints = multimodal_waypoints[selected_mode]
            
            if len(strategic_waypoints) >= 2:
                # 통합 비용맵 생성 (3채널: 정적 + 사회적 + 흐름)
                integrated_cost_map = self._create_integrated_cost_map()
                
                # 데이터 정규화 (-6~6 → -1~1)
                data_sample = {
                    'integrated_cost_map': integrated_cost_map,  # [3, 60, 60]
                    'start_pos': np.array([start_pos[0]/6.0, start_pos[1]/6.0]),
                    'goal_pos': np.array([goal_pos[0]/6.0, goal_pos[1]/6.0]),
                    'strategic_waypoints': np.array([[p[0]/6.0, p[1]/6.0] for p in strategic_waypoints]),
                    'num_waypoints': len(strategic_waypoints),
                    'path_mode': selected_mode  # 경로 모드 정보 추가
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
    
    def _create_integrated_cost_map(self):
        """통합 비용맵 생성 (3채널: 정적 + 사회적 + 흐름) - 60x60 그리드"""
        grid_size = 60
        
        # 채널 0: 정적 장애물 맵 (CGIP 방식 - 이미 60x60)
        static_map = self.simulator.grid.copy().astype(np.float32)
        
        # 채널 1: 사회적 비용맵 (Individual Space 맵 생성)
        social_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # 각 에이전트에 대해 Individual Space 계산
        for agent in self.simulator.agents:
            if not agent.finished and self.simulator.is_in_cctv_coverage(agent.pos):
                # 그리드의 각 셀에 대해 Individual Space 값 계산
                for i in range(grid_size):
                    for j in range(grid_size):
                        # 그리드 좌표를 실제 좌표로 변환
                        x = (j * self.simulator.grid_size) - 6
                        y = (i * self.simulator.grid_size) - 6
                        
                        # Individual Space 값 계산
                        is_value = agent.calculate_individual_space([x, y])
                        social_map[i, j] = max(social_map[i, j], is_value)
        
        # 채널 2: 보행자 흐름맵 (속도 정보)
        flow_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        for agent in self.simulator.agents:
            if hasattr(agent, 'pos') and hasattr(agent, 'velocity'):
                x, y = agent.pos
                vx, vy = agent.velocity
                
                # 60x60 그리드 좌표로 변환
                x_idx = int((x + 6) / self.simulator.grid_size)
                y_idx = int((y + 6) / self.simulator.grid_size)
                
                if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                    # 속도 크기를 흐름 강도로 사용
                    flow_intensity = min(np.sqrt(vx*vx + vy*vy) / 2.0, 1.0)
                    flow_map[y_idx, x_idx] = max(flow_map[y_idx, x_idx], flow_intensity)
        
        # 3채널 통합 [3, 60, 60]
        integrated_map = np.stack([static_map, social_map, flow_map], axis=0)
        return integrated_map
    
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
        
        # 모델 및 스케줄러 초기화 (다중 경로 모드 지원)
        self.model = CDIFModel(
            max_waypoints=config.max_waypoints,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_path_modes=config.num_path_modes
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
            # 데이터 GPU로 이동 (3채널 통합 비용맵)
            integrated_cost_map = batch['integrated_cost_map'].to(self.device)  # [B, 3, 60, 60]
            start_pos = batch['start_pos'].to(self.device)  # [B, 2]
            goal_pos = batch['goal_pos'].to(self.device)  # [B, 2]
            waypoints = batch['waypoints'].to(self.device)  # [B, max_waypoints, 2]
            num_waypoints = batch['num_waypoints'].to(self.device)  # [B]
            path_mode = batch['path_mode'].to(self.device)  # [B]
            
            batch_size = integrated_cost_map.shape[0]
            
            # 랜덤 타임스텝 생성
            timesteps = torch.randint(
                0, self.config.num_train_timesteps, 
                (batch_size,), device=self.device
            )
            
            # 잡음 생성
            noise = torch.randn_like(waypoints)
            
            # 잡음 추가
            noisy_waypoints = self.scheduler.add_noise(waypoints, noise, timesteps)
            
            # 순전파 (사회적 맥락 인식 다중 모달)
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                predicted_noise, mode_probs, num_waypoints_prob = self.model(
                    integrated_cost_map, noisy_waypoints, timesteps, start_pos, goal_pos, path_mode
                )
                
                # 손실 계산
                noise_loss = F.mse_loss(predicted_noise, noise)
                
                # 경로 모드 예측 손실 (사회적 맥락 인식)
                mode_targets = F.one_hot(path_mode, num_classes=self.config.num_path_modes).float()
                mode_loss = F.cross_entropy(mode_probs, mode_targets.argmax(dim=1))
                
                # 웨이포인트 수 예측 손실
                num_targets = F.one_hot(num_waypoints - 1, num_classes=self.config.max_waypoints).float()
                num_loss = F.cross_entropy(num_waypoints_prob, num_targets.argmax(dim=1))
                
                # 총 손실 (잡음 + 모드 + 웨이포인트 수)
                total_loss_batch = noise_loss + 0.2 * mode_loss + 0.1 * num_loss
            
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
                'Mode': f"{mode_loss.item():.6f}",
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
                integrated_cost_map = batch['integrated_cost_map'].to(self.device)
                start_pos = batch['start_pos'].to(self.device)
                goal_pos = batch['goal_pos'].to(self.device)
                waypoints = batch['waypoints'].to(self.device)
                num_waypoints = batch['num_waypoints'].to(self.device)
                path_mode = batch['path_mode'].to(self.device)
                
                batch_size = integrated_cost_map.shape[0]
                
                # 랜덤 타임스텝
                timesteps = torch.randint(
                    0, self.config.num_train_timesteps,
                    (batch_size,), device=self.device
                )
                
                # 잡음 추가
                noise = torch.randn_like(waypoints)
                noisy_waypoints = self.scheduler.add_noise(waypoints, noise, timesteps)
                
                # 순전파
                predicted_noise, mode_probs, num_waypoints_prob = self.model(
                    integrated_cost_map, noisy_waypoints, timesteps, start_pos, goal_pos, path_mode
                )
                
                # 손실 계산
                noise_loss = F.mse_loss(predicted_noise, noise)
                
                mode_targets = F.one_hot(path_mode, num_classes=self.config.num_path_modes).float()
                mode_loss = F.cross_entropy(mode_probs, mode_targets.argmax(dim=1))
                
                num_targets = F.one_hot(num_waypoints - 1, num_classes=self.config.max_waypoints).float()
                num_loss = F.cross_entropy(num_waypoints_prob, num_targets.argmax(dim=1))
                
                total_loss_batch = noise_loss + 0.2 * mode_loss + 0.1 * num_loss
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