#!/usr/bin/env python3
"""
CDIF: CCTV-informed Diffusion-based Path Planning for Social Context-Aware Robot Navigation

핵심 혁신:
1. 편재하는 CCTV 인프라의 전략적 활용을 통한 제로-비용 센서 확장
2. 정적 장애물과 동적 사회적 요소를 단일 표현 공간으로 통합하는 환경 모델링
3. ResNet 기반 시각 특징 추출기와 조건부 확산 생성 모델을 통한 다중 경로 합성
4. 확률적 글로벌 계획과 반응적 로컬 제어의 계층적 융합

차별화 포인트:
- vs CGIP: 결정적 → 확률적 다중 경로 생성
- vs DiPPeR-Legged: 장애물 회피 → 사회적 맥락 인식
- 실시간 적응형 내비게이션 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math

class IntegratedCostMapEncoder(nn.Module):
    """통합 비용 지도 인코더 - 정적 장애물과 동적 사회적 요소를 단일 표현 공간으로 통합"""
    
    def __init__(self, input_channels=3, feature_dim=256):  # 3채널: 정적맵 + 사회적비용 + 흐름맵
        super().__init__()
        
        # 다중 채널 특징 추출 (편재하는 CCTV 인프라 활용)
        self.static_conv = nn.Conv2d(1, 32, 3, padding=1)  # 정적 장애물
        self.social_conv = nn.Conv2d(1, 32, 3, padding=1)  # 사회적 비용
        self.flow_conv = nn.Conv2d(1, 32, 3, padding=1)    # 보행자 흐름
        
        # 통합 특징 융합
        self.fusion_conv = nn.Conv2d(96, 64, 1)  # 3*32 = 96 채널 융합
        
        # ResNet-18 스타일 백본 (통합된 특징 처리)
        self.conv1 = nn.Conv2d(64, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet 블록들 (사회적 맥락 인식을 위한 깊은 특징 추출)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 공간 인식 풀링 (위치 정보 보존)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))  # 2x2 → 4x4로 확장
        
        # 특징 압축 (사회적 맥락 정보 보존)
        self.feature_proj = nn.Sequential(
            nn.Linear(512 * 16, feature_dim * 2),  # 4x4 = 16
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU()
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # 첫 번째 블록 (stride 적용)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 나머지 블록들
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, integrated_cost_map):
        # integrated_cost_map: [B, 3, 60, 60] - 정적맵 + 사회적비용 + 흐름맵
        
        # 채널별 특징 추출 (편재하는 CCTV 인프라의 전략적 활용)
        static_features = self.relu(self.static_conv(integrated_cost_map[:, 0:1]))    # 정적 장애물
        social_features = self.relu(self.social_conv(integrated_cost_map[:, 1:2]))    # 사회적 비용
        flow_features = self.relu(self.flow_conv(integrated_cost_map[:, 2:3]))        # 보행자 흐름
        
        # 다중 모달 특징 융합 (정적 + 동적 사회적 요소 통합)
        fused_features = torch.cat([static_features, social_features, flow_features], dim=1)
        x = self.relu(self.fusion_conv(fused_features))
        
        # ResNet 백본을 통한 깊은 사회적 맥락 인식
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 공간 정보 보존 풀링
        x = self.spatial_pool(x)  # [B, 512, 4, 4]
        x = torch.flatten(x, 1)   # [B, 512*16]
        
        # 사회적 맥락이 통합된 특징 벡터
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

class SocialContextAwareDiffusionNet(nn.Module):
    """사회적 맥락 인식 확산 네트워크 - 다중 경로 후보 동시 생성"""
    
    def __init__(self, 
                 max_waypoints=8,
                 feature_dim=256,
                 hidden_dim=512,
                 num_layers=6,
                 num_path_modes=3):  # 다중 모달 경로 생성
        super().__init__()
        
        self.max_waypoints = max_waypoints
        self.feature_dim = feature_dim
        self.num_path_modes = num_path_modes
        
        # 시간 임베딩 (확산 과정 제어)
        self.time_embedding = SinusoidalPositionEmbedding(feature_dim)
        
        # 위치 임베딩 (시작점, 목표점)
        self.pos_embedding = nn.Sequential(
            nn.Linear(4, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.ReLU()
        )
        
        # 웨이포인트 임베딩 (다중 경로 표현)
        self.waypoint_embedding = nn.Sequential(
            nn.Linear(max_waypoints * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 사회적 맥락 모달리티 임베딩 (다양한 경로 스타일)
        self.mode_embedding = nn.Embedding(num_path_modes, feature_dim)
        
        # 조건부 융합 레이어들 (사회적 맥락 통합)
        self.fusion_layers = nn.ModuleList([
            SocialContextFusionLayer(feature_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # 다중 모달 경로 생성 헤드
        self.multi_modal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, max_waypoints * 2)
            ) for _ in range(num_path_modes)
        ])
        
        # 경로 모드 선택기 (상황별 최적 경로 스타일 결정)
        self.mode_selector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_path_modes),
            nn.Softmax(dim=-1)
        )
        
        # 웨이포인트 수 예측 (적응적 경로 길이)
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
                goal_pos: torch.Tensor,
                path_mode: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_waypoints: [B, max_waypoints, 2] 잡음이 추가된 웨이포인트
            timesteps: [B] 확산 시간 단계
            visual_features: [B, feature_dim] 통합 비용맵 시각 특징
            start_pos: [B, 2] 시작 위치
            goal_pos: [B, 2] 목표 위치
            path_mode: [B] 경로 모드 (선택적)
        
        Returns:
            predicted_noise: [B, max_waypoints, 2] 예측된 잡음
            mode_probs: [B, num_path_modes] 경로 모드 확률
            num_waypoints_prob: [B, max_waypoints] 웨이포인트 수 확률
        """
        batch_size = noisy_waypoints.shape[0]
        
        # 시간 임베딩 (확산 과정 제어)
        time_emb = self.time_embedding(timesteps)  # [B, feature_dim]
        
        # 위치 임베딩 (시작/목표점 맥락)
        start_goal = torch.cat([start_pos, goal_pos], dim=-1)  # [B, 4]
        pos_emb = self.pos_embedding(start_goal)  # [B, feature_dim]
        
        # 웨이포인트 임베딩
        waypoints_flat = noisy_waypoints.view(batch_size, -1)  # [B, max_waypoints*2]
        waypoint_emb = self.waypoint_embedding(waypoints_flat)  # [B, feature_dim]
        
        # 경로 모드 임베딩 (다양한 사회적 맥락 스타일)
        if path_mode is None:
            # 학습 시: 모든 모드의 평균 사용
            mode_emb = self.mode_embedding.weight.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        else:
            mode_emb = self.mode_embedding(path_mode)
        
        # 사회적 맥락 인식 특징 융합
        x = waypoint_emb + mode_emb  # 모달리티 조건부 임베딩
        for layer in self.fusion_layers:
            x = layer(x, visual_features, time_emb, pos_emb)
        
        # 경로 모드 확률 예측 (상황별 최적 스타일 결정)
        mode_probs = self.mode_selector(x)  # [B, num_path_modes]
        
        # 다중 모달 잡음 예측 (확률적 다중 경로 생성)
        if path_mode is None:
            # 학습 시: 가중 평균 사용
            multi_noise = []
            for head in self.multi_modal_heads:
                noise_pred = head(x)  # [B, max_waypoints*2]
                multi_noise.append(noise_pred)
            
            multi_noise = torch.stack(multi_noise, dim=1)  # [B, num_modes, max_waypoints*2]
            # 모드 확률로 가중 평균
            predicted_noise = torch.sum(multi_noise * mode_probs.unsqueeze(-1), dim=1)
        else:
            # 추론 시: 특정 모드 사용
            selected_heads = []
            for i, head in enumerate(self.multi_modal_heads):
                selected_heads.append(head(x))
            
            multi_noise = torch.stack(selected_heads, dim=1)  # [B, num_modes, max_waypoints*2]
            # 원-핫 인코딩으로 특정 모드 선택
            mode_onehot = F.one_hot(path_mode, num_classes=self.num_path_modes).float()
            predicted_noise = torch.sum(multi_noise * mode_onehot.unsqueeze(-1), dim=1)
        
        predicted_noise = predicted_noise.view(batch_size, self.max_waypoints, 2)
        
        # 적응적 웨이포인트 수 예측
        num_waypoints_prob = self.num_waypoints_head(x)  # [B, max_waypoints]
        
        return predicted_noise, mode_probs, num_waypoints_prob

class SocialContextFusionLayer(nn.Module):
    """사회적 맥락 인식 융합 레이어 - 다중 조건부 정보 통합"""
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # 셀프 어텐션
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # 사회적 맥락 조건부 변조 (FiLM - Feature-wise Linear Modulation)
        self.visual_modulation = nn.Linear(feature_dim, feature_dim * 2)  # 통합 비용맵 변조
        self.time_modulation = nn.Linear(feature_dim, feature_dim * 2)    # 확산 시간 변조
        self.pos_modulation = nn.Linear(feature_dim, feature_dim * 2)     # 위치 맥락 변조
        
        # 크로스 어텐션 (CCTV 기반 글로벌 맥락 인식)
        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        
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
        
        # 사회적 맥락 조건부 변조 (다중 모달 정보 통합)
        visual_scale, visual_shift = self.visual_modulation(visual_features).chunk(2, dim=-1)
        time_scale, time_shift = self.time_modulation(time_emb).chunk(2, dim=-1)
        pos_scale, pos_shift = self.pos_modulation(pos_emb).chunk(2, dim=-1)
        
        # 종합 변조 (정적 + 동적 사회적 맥락)
        scale = visual_scale + time_scale + pos_scale
        shift = visual_shift + time_shift + pos_shift
        x_modulated = x_norm * (1 + scale) + shift
        
        # 셀프 어텐션 (경로 내부 일관성)
        x_expanded = x_modulated.unsqueeze(1)  # [B, 1, feature_dim]
        self_attn_out, _ = self.self_attn(x_expanded, x_expanded, x_expanded)
        self_attn_out = self_attn_out.squeeze(1)  # [B, feature_dim]
        
        # 크로스 어텐션 (CCTV 글로벌 맥락과의 상호작용)
        visual_expanded = visual_features.unsqueeze(1)  # [B, 1, feature_dim]
        x_for_cross = x_modulated.unsqueeze(1)
        cross_attn_out, _ = self.cross_attn(x_for_cross, visual_expanded, visual_expanded)
        cross_attn_out = cross_attn_out.squeeze(1)  # [B, feature_dim]
        
        # 다중 어텐션 융합 (로컬 + 글로벌 맥락)
        attn_out = self_attn_out + 0.5 * cross_attn_out
        
        # 잔차 연결
        x = x + attn_out
        
        # 피드포워드 (사회적 맥락 정제)
        x = x + self.ff(self.norm2(x))
        
        return x

class CDIFModel(nn.Module):
    """CDIF: CCTV-informed Diffusion 통합 모델
    
    편재하는 CCTV 인프라를 활용한 사회적 맥락 인식 경로 생성
    - 정적 장애물 + 동적 사회적 요소 통합 모델링
    - 확률적 다중 경로 후보 동시 생성
    - 실시간 적응형 내비게이션 지원
    """
    
    def __init__(self, 
                 max_waypoints=8,
                 feature_dim=256,
                 hidden_dim=512,
                 num_layers=6,
                 num_path_modes=3):
        super().__init__()
        
        self.max_waypoints = max_waypoints
        self.num_path_modes = num_path_modes
        
        # 통합 비용맵 인코더 (편재하는 CCTV 인프라 활용)
        self.cost_map_encoder = IntegratedCostMapEncoder(
            input_channels=3,  # 정적맵 + 사회적비용 + 흐름맵
            feature_dim=feature_dim
        )
        
        # 사회적 맥락 인식 확산 네트워크
        self.diffusion_net = SocialContextAwareDiffusionNet(
            max_waypoints=max_waypoints,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_path_modes=num_path_modes
        )
        
    def forward(self, 
                integrated_cost_map: torch.Tensor,
                noisy_waypoints: torch.Tensor,
                timesteps: torch.Tensor,
                start_pos: torch.Tensor,
                goal_pos: torch.Tensor,
                path_mode: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CDIF 순전파: 사회적 맥락 인식 다중 경로 생성
        
        Args:
            integrated_cost_map: [B, 3, 60, 60] 통합 비용맵 (정적 + 사회적 + 흐름)
            noisy_waypoints: [B, max_waypoints, 2] 잡음이 추가된 웨이포인트
            timesteps: [B] 확산 시간 단계
            start_pos: [B, 2] 시작 위치 (정규화됨)
            goal_pos: [B, 2] 목표 위치 (정규화됨)
            path_mode: [B] 경로 모드 (0: 직접, 1: 사회적, 2: 우회)
        
        Returns:
            predicted_noise: [B, max_waypoints, 2] 예측된 잡음
            mode_probs: [B, num_path_modes] 경로 모드 확률
            num_waypoints_prob: [B, max_waypoints] 웨이포인트 수 확률
        """
        # 통합 비용맵 특징 추출 (편재하는 CCTV 인프라 활용)
        visual_features = self.cost_map_encoder(integrated_cost_map)
        
        # 사회적 맥락 인식 확산 과정
        predicted_noise, mode_probs, num_waypoints_prob = self.diffusion_net(
            noisy_waypoints=noisy_waypoints,
            timesteps=timesteps,
            visual_features=visual_features,
            start_pos=start_pos,
            goal_pos=goal_pos,
            path_mode=path_mode
        )
        
        return predicted_noise, mode_probs, num_waypoints_prob

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
    
    # CDIF 모델 생성 (다중 경로 모드)
    model = CDIFModel(max_waypoints=8, num_path_modes=3).to(device)
    scheduler = DDPMScheduler().to(device)
    
    # 테스트 데이터 (통합 비용맵)
    batch_size = 4
    integrated_cost_map = torch.randn(batch_size, 3, 60, 60).to(device)  # 3채널
    noisy_waypoints = torch.randn(batch_size, 8, 2).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    start_pos = torch.randn(batch_size, 2).to(device)
    goal_pos = torch.randn(batch_size, 2).to(device)
    path_mode = torch.randint(0, 3, (batch_size,)).to(device)  # 0: 직접, 1: 사회적, 2: 우회
    
    # 순전파 테스트 (다중 모달 출력)
    with torch.no_grad():
        predicted_noise, mode_probs, num_waypoints_prob = model(
            integrated_cost_map, noisy_waypoints, timesteps, start_pos, goal_pos, path_mode
        )
    
    print(f"🎯 CDIF 모델 테스트 결과:")
    print(f"  - Predicted noise shape: {predicted_noise.shape}")
    print(f"  - Mode probabilities shape: {mode_probs.shape}")
    print(f"  - Num waypoints prob shape: {num_waypoints_prob.shape}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 경로 모드별 확률: {mode_probs[0].cpu().numpy()}")
    print("✅ CDIF: 사회적 맥락 인식 다중 경로 생성 모델 테스트 완료!") 