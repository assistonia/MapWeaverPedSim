#!/usr/bin/env python3
"""
CDIF (CCTV-Diffusion) Model
- CCTV 데이터 기반 사회적 비용맵 조건부 Diffusion 모델
- 전략적 웨이포인트 생성 (2-4개)
- GPU A6000 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math

class ResNetVisualEncoder(nn.Module):
    """ResNet 기반 시각 특징 추출기"""
    
    def __init__(self, input_channels=1, feature_dim=256):
        super().__init__()
        
        # ResNet-18 스타일 백본
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet 블록들
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 글로벌 평균 풀링
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 특징 압축
        self.feature_proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # 첫 번째 블록 (stride 적용)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 나머지 블록들
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, 1, 60, 60] 비용맵
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_proj(x)
        
        return x

class BasicBlock(nn.Module):
    """ResNet Basic Block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal Position Embedding for Diffusion Timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CDIFDiffusionNet(nn.Module):
    """CDIF Diffusion 잡음 예측 네트워크"""
    
    def __init__(self, 
                 max_waypoints=8,
                 feature_dim=256,
                 hidden_dim=512,
                 num_layers=6):
        super().__init__()
        
        self.max_waypoints = max_waypoints
        self.feature_dim = feature_dim
        
        # 시간 임베딩
        self.time_embedding = SinusoidalPositionEmbedding(feature_dim)
        
        # 위치 임베딩 (시작점, 목표점)
        self.pos_embedding = nn.Sequential(
            nn.Linear(4, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.ReLU()
        )
        
        # 웨이포인트 임베딩
        self.waypoint_embedding = nn.Sequential(
            nn.Linear(max_waypoints * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 조건부 융합 레이어들
        self.fusion_layers = nn.ModuleList([
            FusionLayer(feature_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 최종 출력 레이어
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_waypoints * 2)
        )
        
        # 웨이포인트 수 예측
        self.num_waypoints_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_waypoints),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, 
                noisy_waypoints: torch.Tensor,
                timesteps: torch.Tensor,
                visual_features: torch.Tensor,
                start_pos: torch.Tensor,
                goal_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_waypoints: [B, max_waypoints, 2] 잡음이 추가된 웨이포인트
            timesteps: [B] 확산 시간 단계
            visual_features: [B, feature_dim] 비용맵 시각 특징
            start_pos: [B, 2] 시작 위치
            goal_pos: [B, 2] 목표 위치
        
        Returns:
            predicted_noise: [B, max_waypoints, 2] 예측된 잡음
            num_waypoints_prob: [B, max_waypoints] 웨이포인트 수 확률
        """
        batch_size = noisy_waypoints.shape[0]
        
        # 시간 임베딩
        time_emb = self.time_embedding(timesteps)  # [B, feature_dim]
        
        # 위치 임베딩
        start_goal = torch.cat([start_pos, goal_pos], dim=-1)  # [B, 4]
        pos_emb = self.pos_embedding(start_goal)  # [B, feature_dim]
        
        # 웨이포인트 임베딩
        waypoints_flat = noisy_waypoints.view(batch_size, -1)  # [B, max_waypoints*2]
        waypoint_emb = self.waypoint_embedding(waypoints_flat)  # [B, feature_dim]
        
        # 조건부 특징 융합
        x = waypoint_emb
        for layer in self.fusion_layers:
            x = layer(x, visual_features, time_emb, pos_emb)
        
        # 잡음 예측
        predicted_noise = self.output_proj(x)  # [B, max_waypoints*2]
        predicted_noise = predicted_noise.view(batch_size, self.max_waypoints, 2)
        
        # 웨이포인트 수 예측
        num_waypoints_prob = self.num_waypoints_head(x)  # [B, max_waypoints]
        
        return predicted_noise, num_waypoints_prob

class FusionLayer(nn.Module):
    """조건부 특징 융합 레이어"""
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # 셀프 어텐션
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # 조건부 변조 (FiLM)
        self.visual_modulation = nn.Linear(feature_dim, feature_dim * 2)
        self.time_modulation = nn.Linear(feature_dim, feature_dim * 2)
        self.pos_modulation = nn.Linear(feature_dim, feature_dim * 2)
        
        # 피드포워드
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, x, visual_features, time_emb, pos_emb):
        # 정규화
        x_norm = self.norm1(x)
        
        # 조건부 변조 (FiLM)
        visual_scale, visual_shift = self.visual_modulation(visual_features).chunk(2, dim=-1)
        time_scale, time_shift = self.time_modulation(time_emb).chunk(2, dim=-1)
        pos_scale, pos_shift = self.pos_modulation(pos_emb).chunk(2, dim=-1)
        
        # 종합 변조
        scale = visual_scale + time_scale + pos_scale
        shift = visual_shift + time_shift + pos_shift
        x_modulated = x_norm * (1 + scale) + shift
        
        # 셀프 어텐션 (배치 차원 확장)
        x_expanded = x_modulated.unsqueeze(1)  # [B, 1, feature_dim]
        attn_out, _ = self.self_attn(x_expanded, x_expanded, x_expanded)
        attn_out = attn_out.squeeze(1)  # [B, feature_dim]
        
        # 잔차 연결
        x = x + attn_out
        
        # 피드포워드
        x = x + self.ff(self.norm2(x))
        
        return x

class CDIFModel(nn.Module):
    """CDIF 통합 모델"""
    
    def __init__(self, 
                 max_waypoints=8,
                 feature_dim=256,
                 hidden_dim=512,
                 num_layers=6):
        super().__init__()
        
        self.max_waypoints = max_waypoints
        
        # 시각 인코더
        self.visual_encoder = ResNetVisualEncoder(
            input_channels=1, 
            feature_dim=feature_dim
        )
        
        # Diffusion 네트워크
        self.diffusion_net = CDIFDiffusionNet(
            max_waypoints=max_waypoints,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
    def forward(self, 
                cost_map: torch.Tensor,
                noisy_waypoints: torch.Tensor,
                timesteps: torch.Tensor,
                start_pos: torch.Tensor,
                goal_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cost_map: [B, 1, 60, 60] 사회적 비용맵
            noisy_waypoints: [B, max_waypoints, 2] 잡음이 추가된 웨이포인트
            timesteps: [B] 확산 시간 단계
            start_pos: [B, 2] 시작 위치 (정규화됨)
            goal_pos: [B, 2] 목표 위치 (정규화됨)
        
        Returns:
            predicted_noise: [B, max_waypoints, 2] 예측된 잡음
            num_waypoints_prob: [B, max_waypoints] 웨이포인트 수 확률
        """
        # 시각 특징 추출
        visual_features = self.visual_encoder(cost_map)
        
        # Diffusion 잡음 예측
        predicted_noise, num_waypoints_prob = self.diffusion_net(
            noisy_waypoints=noisy_waypoints,
            timesteps=timesteps,
            visual_features=visual_features,
            start_pos=start_pos,
            goal_pos=goal_pos
        )
        
        return predicted_noise, num_waypoints_prob

class DDPMScheduler:
    """DDPM 스케줄러 (GPU 최적화)"""
    
    def __init__(self, 
                 num_train_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        self.num_train_timesteps = num_train_timesteps
        
        # 베타 스케줄
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        
        # 알파 계산
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 샘플링을 위한 계수들
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """코사인 베타 스케줄"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original_samples, noise, timesteps):
        """원본 샘플에 잡음 추가"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # 브로드캐스팅을 위한 차원 조정
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def to(self, device):
        """스케줄러를 특정 디바이스로 이동"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

if __name__ == "__main__":
    # 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 생성
    model = CDIFModel(max_waypoints=8).to(device)
    scheduler = DDPMScheduler().to(device)
    
    # 테스트 데이터
    batch_size = 4
    cost_map = torch.randn(batch_size, 1, 60, 60).to(device)
    noisy_waypoints = torch.randn(batch_size, 8, 2).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    start_pos = torch.randn(batch_size, 2).to(device)
    goal_pos = torch.randn(batch_size, 2).to(device)
    
    # 순전파 테스트
    with torch.no_grad():
        predicted_noise, num_waypoints_prob = model(
            cost_map, noisy_waypoints, timesteps, start_pos, goal_pos
        )
    
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print(f"Num waypoints prob shape: {num_waypoints_prob.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ CDIF 모델 테스트 완료!") 