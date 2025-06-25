import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm
import json

# DiPPeR 모델 클래스들 (robot_simuator_dippeR.py에서 가져옴)
from robot_simuator_dippeR import Agent, ResNetEncoder, NoisePredictor, DiPPeR, RobotSimulatorDiPPeR

class SimulationDataset(Dataset):
    """시뮬레이션 환경에서 생성된 합성 데이터셋"""
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 데이터 형태: (cost_map, start_pos, goal_pos, path)
        cost_map = torch.from_numpy(item['cost_map']).float().unsqueeze(0)  # (1, 60, 60)
        start_pos = torch.from_numpy(item['start_pos']).float()  # (2,)
        goal_pos = torch.from_numpy(item['goal_pos']).float()  # (2,)
        path = torch.from_numpy(item['path']).float()  # (path_length, 2)
        
        return cost_map, start_pos, goal_pos, path

class SimulationDataCollector:
    """시뮬레이션 환경에서 합성 데이터 수집"""
    def __init__(self, xml_file, visualize=False):
        self.simulator = RobotSimulatorDiPPeR(xml_file, model_path=None)
        self.visualize = visualize
        self.collected_data = []
        
        # A* 폴백만 사용하도록 설정 (DiPPeR 비활성화)
        self.simulator.use_dipperp = False
        
    def collect_data_episode(self, start_pos, goal_pos, max_steps=200):
        """한 에피소드에서 데이터 수집 - 다양한 경로 스타일 포함"""
        print(f"데이터 수집: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) → ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        
        # 시뮬레이터 초기화
        self.simulator.robot_pos = start_pos.copy()
        self.simulator.target_pos = goal_pos.copy()
        self.simulator.stuck_count = 0
        self.simulator.timestep_counter = 0
        
        # 시각화 창 정리 (메모리 누수 방지)
        if hasattr(self.simulator, 'fig') and self.simulator.fig is not None:
            plt.close(self.simulator.fig)
            self.simulator.fig = None
        
        episode_data = []
        stuck_counter = 0
        last_pos = None
        
        for step in range(max_steps):
            # 현재 상태 업데이트
            self.simulator.update()
            
            # 정체 상태 체크 (더 엄격하게)
            current_pos = self.simulator.robot_pos.copy()
            current_pos_tuple = (round(current_pos[0], 1), round(current_pos[1], 1))  # 0.1m 정밀도
            
            if last_pos == current_pos_tuple:
                stuck_counter += 1
                if stuck_counter > 20:  # 20스텝 동안 같은 위치에 있으면 중단 (더 빠르게)
                    print(f"로봇 정체 상태 감지. 에피소드 중단. 스텝: {step}, 수집된 데이터: {len(episode_data)}개")
                    break
            else:
                stuck_counter = 0
            last_pos = current_pos_tuple
            
            # 다양한 경로 스타일 생성
            if self.simulator.target_pos is not None:
                paths_to_collect = []
                
                # 1. 기본 A* 경로 (30% 확률 - 감소)
                if np.random.random() < 0.3:
                    astar_path = self.simulator.fallback_astar_planning(current_pos, goal_pos)
                    if astar_path and len(astar_path) > 2:
                        paths_to_collect.append(("astar", astar_path))
                
                # 2. 사회적 비용 강화 A* 경로 (40% 확률 - 증가)
                if np.random.random() < 0.4:
                    social_path = self.generate_social_aware_path(current_pos, goal_pos)
                    if social_path and len(social_path) > 2:
                        paths_to_collect.append(("social", social_path))
                
                # 3. 우회 경로 (30% 확률 - 증가) - 중간점을 거쳐가는 경로
                if np.random.random() < 0.3:
                    detour_path = self.generate_detour_path(current_pos, goal_pos)
                    if detour_path and len(detour_path) > 2:
                        paths_to_collect.append(("detour", detour_path))
                
                # 수집된 경로들을 학습 데이터로 변환
                for path_type, path in paths_to_collect:
                    # 경로를 고정 길이로 맞춤 (50개)
                    path_length = 50
                    if len(path) >= path_length:
                        # 다운샘플링
                        indices = np.linspace(0, len(path)-1, path_length, dtype=int)
                        resampled_path = [path[i] for i in indices]
                    else:
                        # 업샘플링 (선형 보간)
                        resampled_path = self.interpolate_path(path, path_length)
                    
                    # 경로 안전성 검증 (장애물 관통 체크)
                    path_is_safe = True
                    for point in resampled_path:
                        if not self.simulator.is_position_safe(point):
                            path_is_safe = False
                            break
                    
                    # 안전한 경로만 학습 데이터로 사용
                    if path_is_safe:
                        # 데이터 저장
                        data_item = {
                            'cost_map': self.simulator.fused_cost_map.copy(),
                            'start_pos': np.array([current_pos[0]/6.0, current_pos[1]/6.0]),  # 정규화
                            'goal_pos': np.array([goal_pos[0]/6.0, goal_pos[1]/6.0]),  # 정규화
                            'path': np.array([[p[0]/6.0, p[1]/6.0] for p in resampled_path]),  # 정규화
                            'path_type': path_type  # 경로 타입 추가
                        }
                        episode_data.append(data_item)
                        print(f"경로 수집: {path_type} ({len(resampled_path)}개 웨이포인트)")
            
            # 시각화
            if self.visualize and step % 10 == 0:
                self.simulator.visualize()
                time.sleep(0.01)
            
            # 목표 도달 체크
            dist_to_goal = np.sqrt((current_pos[0] - goal_pos[0])**2 + (current_pos[1] - goal_pos[1])**2)
            if dist_to_goal < 0.3:
                print(f"목표 도달! 스텝: {step}, 수집된 데이터: {len(episode_data)}개")
                break
        
        self.collected_data.extend(episode_data)
        return len(episode_data)
    
    def interpolate_path(self, path, target_length):
        """경로를 선형 보간으로 리샘플링"""
        if len(path) < 2:
            return path
        
        # 경로 길이 계산
        path_array = np.array(path)
        distances = np.cumsum([0] + [np.linalg.norm(path_array[i+1] - path_array[i]) 
                                     for i in range(len(path)-1)])
        total_distance = distances[-1]
        
        # 균등 간격으로 리샘플링
        target_distances = np.linspace(0, total_distance, target_length)
        resampled_path = []
        
        for target_dist in target_distances:
            # 가장 가까운 구간 찾기
            idx = np.searchsorted(distances, target_dist)
            if idx == 0:
                resampled_path.append(path[0])
            elif idx >= len(path):
                resampled_path.append(path[-1])
            else:
                # 선형 보간
                t = (target_dist - distances[idx-1]) / (distances[idx] - distances[idx-1])
                interpolated = [(1-t) * path[idx-1][j] + t * path[idx][j] for j in range(2)]
                resampled_path.append(interpolated)
        
        return resampled_path
    
    def generate_social_aware_path(self, start_pos, goal_pos):
        """사회적 비용을 더 강하게 고려한 경로 생성"""
        # 임시로 사회적 비용 가중치를 높여서 A* 실행
        original_fused_map = self.simulator.fused_cost_map.copy()
        
        # 에이전트 주변 비용을 더 높게 설정
        for agent in self.simulator.agents:
            agent_pos = agent.pos
            x_idx = int((agent_pos[0] + 6) / self.simulator.grid_size)
            y_idx = int((agent_pos[1] + 6) / self.simulator.grid_size)
            
            # 에이전트 주변 3x3 영역에 높은 비용 부여
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x_idx + dx, y_idx + dy
                    if 0 <= nx < 60 and 0 <= ny < 60:
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance <= 2:
                            social_penalty = 0.9 * (1 - distance/2)  # 거리에 따른 페널티
                            self.simulator.fused_cost_map[ny, nx] = min(
                                self.simulator.fused_cost_map[ny, nx] + social_penalty, 0.95
                            )
        
        # 수정된 코스트 맵으로 A* 실행
        path = self.simulator.fallback_astar_planning(start_pos, goal_pos)
        
        # 원래 코스트 맵 복원
        self.simulator.fused_cost_map = original_fused_map
        
        return path
    
    def generate_detour_path(self, start_pos, goal_pos):
        """중간점을 거쳐가는 우회 경로 생성"""
        # 시작점과 목표점 사이의 중점 근처에 랜덤 중간점 생성
        mid_x = (start_pos[0] + goal_pos[0]) / 2
        mid_y = (start_pos[1] + goal_pos[1]) / 2
        
        # 중점에서 랜덤하게 오프셋 추가
        offset_range = 2.0
        for attempt in range(10):
            waypoint = [
                mid_x + np.random.uniform(-offset_range, offset_range),
                mid_y + np.random.uniform(-offset_range, offset_range)
            ]
            
            if self.simulator.is_position_safe(waypoint):
                # 시작점 → 중간점 → 목표점 경로 생성
                path1 = self.simulator.fallback_astar_planning(start_pos, waypoint)
                path2 = self.simulator.fallback_astar_planning(waypoint, goal_pos)
                
                if path1 and path2 and len(path1) > 1 and len(path2) > 1:
                    # 두 경로 연결 (중복 제거)
                    combined_path = path1 + path2[1:]
                    return combined_path
        
        # 우회 경로 생성 실패 시 기본 A* 경로 반환
        return self.simulator.fallback_astar_planning(start_pos, goal_pos)
    
    def collect_random_episodes(self, num_episodes=100):
        """랜덤한 시작/목표점으로 에피소드 수집"""
        total_data = 0
        
        for episode in tqdm(range(num_episodes), desc="데이터 수집"):
            # 안전한 랜덤 위치 생성 함수 (더 넓은 안전 구역에서 생성)
            def generate_safe_position(max_attempts=50):
                # 먼저 안전한 구역에서 시도
                safe_zones = [
                    (-4.5, -0.5, -4.5, 1.5),  # 왼쪽 위 구역
                    (-4.5, -0.5, -2.0, -1.5), # 왼쪽 아래 구역  
                    (2.5, 3.5, -4.5, -0.5),   # 오른쪽 아래 구역
                    (2.5, 3.5, 1.0, 2.0),     # 오른쪽 위 구역
                    (-0.5, 1.5, 3.0, 4.5)     # 위쪽 중앙 구역
                ]
                
                for _ in range(max_attempts):
                    # 안전 구역 중 하나 선택
                    if np.random.random() < 0.8:  # 80% 확률로 안전 구역에서 선택
                        zone = safe_zones[np.random.randint(len(safe_zones))]
                        pos = [np.random.uniform(zone[0], zone[1]), 
                               np.random.uniform(zone[2], zone[3])]
                    else:  # 20% 확률로 전체 영역에서 선택
                        pos = [np.random.uniform(-4.8, 4.8), np.random.uniform(-4.8, 4.8)]
                    
                    if self.simulator.is_position_safe(pos):
                        return pos
                
                # 최악의 경우 검증된 안전한 위치들 중 하나 선택
                safe_positions = [[-3.0, 0.0], [3.0, -2.0], [0.0, 3.5], [-2.5, -3.0], [3.0, 1.5]]
                for safe_pos in safe_positions:
                    if self.simulator.is_position_safe(safe_pos):
                        print(f"안전한 기본 위치 사용: {safe_pos}")
                        return safe_pos
                
                print(f"모든 안전한 위치 생성 실패, 원점 사용")
                return [0.0, 0.0]
            
            # 연결 가능한 시작점과 목표점 생성
            max_pair_attempts = 10
            valid_pair_found = False
            
            for pair_attempt in range(max_pair_attempts):
                start_pos = generate_safe_position()
                goal_pos = generate_safe_position()
                
                # 시작점과 목표점이 너무 가까우면 다시 생성
                min_distance = 2.0
                distance_attempts = 0
                while (np.linalg.norm(np.array(start_pos) - np.array(goal_pos)) < min_distance and 
                       distance_attempts < 10):
                    goal_pos = generate_safe_position()
                    distance_attempts += 1
                
                # A* 경로 계획으로 연결성 체크
                test_path = self.simulator.fallback_astar_planning(start_pos, goal_pos)
                
                if test_path and len(test_path) > 1:
                    valid_pair_found = True
                    print(f"유효한 경로 발견: {start_pos} → {goal_pos} ({len(test_path)}개 웨이포인트)")
                    break
                else:
                    print(f"경로 연결 실패 (시도 {pair_attempt+1}/{max_pair_attempts}): {start_pos} → {goal_pos}")
            
            if not valid_pair_found:
                print(f"에피소드 {episode}: 연결 가능한 경로를 찾을 수 없음. 건너뛰기.")
                continue
            
            # 데이터 수집
            collected = self.collect_data_episode(start_pos, goal_pos)
            total_data += collected
            
            if episode % 10 == 0:
                print(f"에피소드 {episode}/{num_episodes}, 총 데이터: {total_data}개")
        
        print(f"데이터 수집 완료! 총 {total_data}개 데이터")
        return self.collected_data

class DiPPeRTrainer:
    """DiPPeR 모델 학습"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # GPU 사용 시 학습률 조정 (성능 개선을 위해 학습률 대폭 감소)
        if device.type == 'cuda':
            lr = 1e-5  # GPU에서도 낮은 학습률로 안정적 학습
            print(f"🚀 GPU 학습률 (개선): {lr}")
        else:
            lr = 5e-6  # CPU에서는 더 낮은 학습률
            print(f"💻 CPU 학습률 (개선): {lr}")
            
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 50  # Early stopping patience 대폭 증가 (더 긴 학습)
        
        # GPU 메모리 최적화
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # 성능 최적화
            torch.backends.cudnn.deterministic = False  # 성능 우선
            print("CUDNN 최적화 활성화")
        
        self.first_batch = False  # 첫 번째 배치 체크용
        
    def train_step(self, batch):
        cost_maps, start_pos, goal_pos, paths = batch
        batch_size = cost_maps.shape[0]
        
        # GPU로 이동 (non_blocking=True로 성능 향상)
        cost_maps = cost_maps.to(self.device, non_blocking=True)
        start_pos = start_pos.to(self.device, non_blocking=True)
        goal_pos = goal_pos.to(self.device, non_blocking=True)
        paths = paths.to(self.device, non_blocking=True)
        
        # GPU 메모리 사용량 체크 (첫 번째 배치에서만)
        if hasattr(self, 'first_batch') and not self.first_batch:
            if self.device.type == 'cuda':
                print(f"배치 처리 중 GPU 메모리: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
            self.first_batch = True
        
        # 랜덤 타임스텝 선택 (더 다양한 범위)
        timesteps = torch.randint(0, self.model.max_timesteps, (batch_size,), device=self.device)
        
        # 노이즈 추가
        noise = torch.randn_like(paths)
        alpha_cumprod = self.model.alphas_cumprod[timesteps].view(-1, 1, 1)
        noisy_paths = torch.sqrt(alpha_cumprod) * paths + torch.sqrt(1 - alpha_cumprod) * noise
        
        # 시작점과 목표점 조건
        start_goal_pos = torch.cat([start_pos, goal_pos], dim=-1)
        
        # 노이즈 예측
        predicted_noise = self.model(cost_maps, noisy_paths, timesteps, start_goal_pos)
        
        # 손실 계산 (MSE Loss로 변경 - 더 정확한 학습)
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 그래디언트 클리핑 강화
        self.optimizer.step()
        
        # GPU 메모리 정리 (매 10번째 배치마다)
        if self.device.type == 'cuda' and hasattr(self, 'batch_count'):
            self.batch_count += 1
            if self.batch_count % 10 == 0:
                torch.cuda.empty_cache()
        elif self.device.type == 'cuda':
            self.batch_count = 1
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            # 실시간 손실 표시
            current_avg_loss = total_loss / num_batches
            pbar.set_postfix({'Loss': f'{current_avg_loss:.6f}', 'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})
        
        avg_loss = total_loss / num_batches
        self.scheduler.step()
        
        # Early stopping 체크 (더 엄격하게)
        if avg_loss < self.best_loss * 0.999:  # 0.1% 이상 개선되어야 함 (더 엄격)
            self.best_loss = avg_loss
            self.patience_counter = 0
            return avg_loss, True  # 개선됨
        else:
            self.patience_counter += 1
            return avg_loss, False  # 개선 안됨
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        print(f"모델 저장 완료: {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"모델 로드 완료: {path}")

def main():
    parser = argparse.ArgumentParser(description='DiPPeR 모델 학습')
    parser.add_argument('--xml_file', default='scenarios/Congestion1.xml', help='시뮬레이션 XML 파일')
    parser.add_argument('--num_episodes', type=int, default=1000, help='데이터 수집 에피소드 수 (대폭 증가)')
    parser.add_argument('--epochs', type=int, default=500, help='학습 에포크 수 (대폭 증가)')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기 (GPU 메모리에 따라 조정)')
    parser.add_argument('--visualize', action='store_true', help='데이터 수집 시 시각화')
    parser.add_argument('--save_data', help='수집된 데이터 저장 경로')
    parser.add_argument('--load_data', help='저장된 데이터 로드 경로')
    parser.add_argument('--model_save_path', default='models/dipperp_model.pth', help='모델 저장 경로')
    
    args = parser.parse_args()
    
    # GPU 사용 가능하면 GPU, 아니면 CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # 명시적으로 GPU 0번 지정
        torch.cuda.set_device(0)  # GPU 0번으로 설정
        print(f"🚀 GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"현재 GPU 디바이스: {torch.cuda.current_device()}")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        print(f"GPU 메모리 정리 완료")
    else:
        device = torch.device('cpu')
        print("💻 GPU 없음, CPU로 학습")
    
    # 데이터 수집 또는 로드
    if args.load_data and os.path.exists(args.load_data):
        print(f"데이터 로드: {args.load_data}")
        with open(args.load_data, 'r') as f:
            data_list = json.load(f)
        # JSON에서 numpy 배열로 변환
        for item in data_list:
            item['cost_map'] = np.array(item['cost_map'])
            item['start_pos'] = np.array(item['start_pos'])
            item['goal_pos'] = np.array(item['goal_pos'])
            item['path'] = np.array(item['path'])
    else:
        print("새로운 데이터 수집 시작...")
        collector = SimulationDataCollector(args.xml_file, visualize=args.visualize)
        data_list = collector.collect_random_episodes(args.num_episodes)
        
        # 데이터 저장
        if args.save_data:
            # numpy 배열을 JSON 직렬화 가능하게 변환
            json_data = []
            for item in data_list:
                json_item = {
                    'cost_map': item['cost_map'].tolist(),
                    'start_pos': item['start_pos'].tolist(),
                    'goal_pos': item['goal_pos'].tolist(),
                    'path': item['path'].tolist()
                }
                json_data.append(json_item)
            
            with open(args.save_data, 'w') as f:
                json.dump(json_data, f)
            print(f"데이터 저장 완료: {args.save_data}")
    
    print(f"총 데이터 개수: {len(data_list)}")
    
    # 데이터셋 및 데이터로더 생성
    dataset = SimulationDataset(data_list)
    
    # CPU 코어 수에 따른 num_workers 자동 조정
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count())
    
    # GPU 사용 시 배치 크기 자동 조정
    if device.type == 'cuda':
        # GPU 메모리에 따른 배치 크기 조정
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 8:
            recommended_batch_size = 16
        elif gpu_memory_gb >= 4:
            recommended_batch_size = 12
        else:
            recommended_batch_size = 8
        
        if args.batch_size == 8:  # 기본값인 경우만 조정
            args.batch_size = recommended_batch_size
            print(f"🚀 GPU 메모리 {gpu_memory_gb:.1f}GB 감지: 배치 크기를 {args.batch_size}로 자동 조정")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"📊 배치 크기: {args.batch_size}, 워커 수: {num_workers}")
    
    # 모델 초기화
    model = DiPPeR(visual_feature_dim=512, path_dim=2, max_timesteps=1000)
    print(f"모델 생성 완료")
    
    # 모델을 GPU로 이동
    model = model.to(device)
    print(f"모델을 {device}로 이동 완료")
    
    # GPU 메모리 사용량 확인
    if device.type == 'cuda':
        print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    
    trainer = DiPPeRTrainer(model, device)
    
    # 학습
    print("학습 시작...")
    best_model_path = f"models/{args.model_save_path.split('/')[-1].split('.')[0]}_best.pth"
    
    for epoch in range(args.epochs):
        avg_loss, improved = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, Best: {trainer.best_loss:.6f}")
        
        # 최고 성능 모델 저장
        if improved:
            trainer.save_model(best_model_path)
            print(f"🎯 새로운 최고 성능 모델 저장: {best_model_path}")
        
        # 정기 모델 저장 (10 에포크마다)
        if (epoch + 1) % 10 == 0:
            save_path = f"models/{args.model_save_path.split('/')[-1].split('.')[0]}_epoch_{epoch+1}.pth"
            trainer.save_model(save_path)
        
        # Early stopping 체크
        if trainer.patience_counter >= trainer.patience:
            print(f"⏹️ Early stopping: {trainer.patience} 에포크 동안 개선 없음")
            break
    
    # 최종 모델 저장
    trainer.save_model(args.model_save_path)
    print("학습 완료!")
    print(f"최고 성능 모델: {best_model_path} (Loss: {trainer.best_loss:.6f})")

if __name__ == "__main__":
    main() 