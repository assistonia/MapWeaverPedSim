#!/usr/bin/env python3
"""
빠른 DiPPeR 학습 코드 - 병렬 데이터 수집
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
import json
from tqdm import tqdm
import time

# CUDA 멀티프로세싱 문제 해결
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

from robot_simuator_dippeR import DiPPeR, RobotSimulatorDiPPeR

def collect_single_episode(args):
    """단일 에피소드 데이터 수집 (멀티프로세싱용)"""
    xml_file, episode_id, episodes_per_process = args
    
    try:
        # 각 프로세스마다 별도 시뮬레이터 생성 (CPU만 사용)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 자식 프로세스에서 GPU 사용 금지
        simulator = RobotSimulatorDiPPeR(xml_file, model_path=None)
        simulator.use_dipperp = False  # A* 폴백만 사용
        
        episode_data = []
        
        for i in range(episodes_per_process):
            # 안전한 시작/목표점 생성
            start_pos, goal_pos = generate_safe_positions(simulator)
            if start_pos is None:
                continue
                
            # A* 경로 계획
            path = simulator.fallback_astar_planning(start_pos, goal_pos)
            if path is None or len(path) < 10:
                continue
            
            # 50개 웨이포인트로 보간
            waypoints = interpolate_path(path, 50)
            if waypoints is None:
                continue
            
            # 안전성 검증
            if not all(simulator.is_position_safe(wp) for wp in waypoints):
                continue
            
            # 데이터 저장
            data_item = {
                'cost_map': simulator.fused_cost_map.copy(),
                'start_pos': np.array([start_pos[0]/6.0, start_pos[1]/6.0]),
                'goal_pos': np.array([goal_pos[0]/6.0, goal_pos[1]/6.0]),
                'path': np.array([[wp[0]/6.0, wp[1]/6.0] for wp in waypoints])
            }
            episode_data.append(data_item)
            
        print(f"프로세스 {episode_id}: {len(episode_data)}개 데이터 수집 완료")
        return episode_data
        
    except Exception as e:
        print(f"프로세스 {episode_id} 오류: {e}")
        return []

def generate_safe_positions(simulator):
    """안전한 시작/목표점 생성"""
    safe_zones = [
        (-4.5, -0.5, -4.5, 1.5),  # 왼쪽 위
        (-4.5, -0.5, -2.0, -1.5), # 왼쪽 아래
        (2.5, 3.5, -4.5, -0.5),   # 오른쪽 아래
        (2.5, 3.5, 1.0, 2.0),     # 오른쪽 위
        (-0.5, 1.5, 3.0, 4.5)     # 위쪽 중앙
    ]
    
    for _ in range(20):  # 시도 횟수 줄임
        # 안전 구역에서 선택
        zone1 = safe_zones[np.random.randint(len(safe_zones))]
        zone2 = safe_zones[np.random.randint(len(safe_zones))]
        
        start_pos = [np.random.uniform(zone1[0], zone1[1]), 
                    np.random.uniform(zone1[2], zone1[3])]
        goal_pos = [np.random.uniform(zone2[0], zone2[1]), 
                   np.random.uniform(zone2[2], zone2[3])]
        
        # 거리 체크
        if np.linalg.norm(np.array(goal_pos) - np.array(start_pos)) < 2.0:
            continue
            
        # 안전성 체크
        if (simulator.is_position_safe(start_pos) and 
            simulator.is_position_safe(goal_pos)):
            return start_pos, goal_pos
    
    return None, None

def interpolate_path(path, target_points):
    """경로 보간"""
    if len(path) < 2:
        return None
    
    path_array = np.array(path)
    distances = np.cumsum([0] + [np.linalg.norm(path_array[i+1] - path_array[i]) 
                                for i in range(len(path_array)-1)])
    total_distance = distances[-1]
    
    if total_distance < 1e-6:
        return None
    
    target_distances = np.linspace(0, total_distance, target_points)
    interpolated_x = np.interp(target_distances, distances, path_array[:, 0])
    interpolated_y = np.interp(target_distances, distances, path_array[:, 1])
    
    return np.column_stack([interpolated_x, interpolated_y])

def collect_parallel_data(xml_file, total_episodes=2000, num_processes=8):
    """병렬로 데이터 수집"""
    episodes_per_process = total_episodes // num_processes
    
    print(f"🚀 {num_processes}개 프로세스로 총 {total_episodes}개 에피소드 수집 시작")
    print(f"각 프로세스당 {episodes_per_process}개 에피소드")
    
    # 프로세스 인자 준비
    process_args = [(xml_file, i, episodes_per_process) 
                   for i in range(num_processes)]
    
    all_data = []
    
    # 병렬 실행 (spawn 방식)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 작업 제출
        futures = [executor.submit(collect_single_episode, args) 
                  for args in process_args]
        
        # 진행률 표시
        for future in tqdm(as_completed(futures), total=num_processes, 
                          desc="병렬 데이터 수집"):
            episode_data = future.result()
            all_data.extend(episode_data)
    
    print(f"✅ 총 {len(all_data)}개 데이터 수집 완료!")
    return all_data

class FastDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        cost_map = torch.from_numpy(item['cost_map']).float().unsqueeze(0)
        start_pos = torch.from_numpy(item['start_pos']).float()
        goal_pos = torch.from_numpy(item['goal_pos']).float()
        path = torch.from_numpy(item['path']).float()
        return cost_map, start_pos, goal_pos, path

def train_fast_dipperp(data_list, device='cuda', epochs=100):
    """빠른 DiPPeR 학습"""
    dataset = FastDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, 
                           num_workers=4, pin_memory=True)
    
    model = DiPPeR(visual_feature_dim=512, path_dim=2, max_timesteps=1000).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    print(f"🎯 {device}에서 학습 시작: {len(dataset)}개 데이터, {epochs} 에포크")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            cost_maps, start_pos, goal_pos, paths = batch
            batch_size = cost_maps.shape[0]
            
            # GPU로 이동
            cost_maps = cost_maps.to(device, non_blocking=True)
            start_pos = start_pos.to(device, non_blocking=True)
            goal_pos = goal_pos.to(device, non_blocking=True)
            paths = paths.to(device, non_blocking=True)
            
            # Diffusion 학습
            timesteps = torch.randint(0, model.max_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(paths)
            alpha_cumprod = model.alphas_cumprod[timesteps].view(-1, 1, 1)
            noisy_paths = torch.sqrt(alpha_cumprod) * paths + torch.sqrt(1 - alpha_cumprod) * noise
            
            start_goal_pos = torch.cat([start_pos, goal_pos], dim=-1)
            predicted_noise = model(cost_maps, noisy_paths, timesteps, start_goal_pos)
            
            loss = nn.MSELoss()(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, 'models/dipperp_fast_best.pth')
            print(f"🎯 새로운 최고 성능 모델 저장: {avg_loss:.6f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='빠른 DiPPeR 학습')
    parser.add_argument('--xml_file', default='scenarios/Circulation1.xml')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    # GPU 확인
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("💻 CPU 사용")
    
    # 병렬 데이터 수집
    start_time = time.time()
    data_list = collect_parallel_data(args.xml_file, args.episodes, args.processes)
    collect_time = time.time() - start_time
    
    print(f"⏱️ 데이터 수집 시간: {collect_time:.1f}초")
    if len(data_list) > 0:
        print(f"📊 초당 {len(data_list)/collect_time:.1f}개 데이터 수집")
    
    if len(data_list) == 0:
        print("❌ 수집된 데이터가 없습니다!")
        return
    
    # 빠른 학습
    os.makedirs('models', exist_ok=True)
    model = train_fast_dipperp(data_list, device, args.epochs)
    
    print("✅ 학습 완료!")

if __name__ == "__main__":
    main() 