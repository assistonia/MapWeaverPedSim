import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

class Agent:
    def __init__(self, x, y, waypoints):
        self.pos = [float(x), float(y)]
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.speed = 0.1
        self.radius = 0.3
        self.path = []
        self.grid_size = 0.1
        self.grid = np.zeros((120, 120))
        self.stuck_count = 0
        self.last_pos = [float(x), float(y)]
        self.finished = False
        self.waypoint_reached = False
        self.waypoint_threshold = 0.2
        self.velocity = [0.0, 0.0]

    def create_grid(self, obstacles, agents):
        self.grid = np.zeros((120, 120))
        # 장애물 그리기
        for obs in obstacles:
            x1, y1 = obs[0]
            x2, y2 = obs[1]
            x1_idx = int((x1 + 6) / self.grid_size)
            y1_idx = int((y1 + 6) / self.grid_size)
            x2_idx = int((x2 + 6) / self.grid_size)
            y2_idx = int((y2 + 6) / self.grid_size)
            self.grid[min(y1_idx, y2_idx):max(y1_idx, y2_idx)+1, 
                     min(x1_idx, x2_idx):max(x1_idx, x2_idx)+1] = 1
        
        # 다른 에이전트를 장애물로 추가
        for agent in agents:
            if agent != self:
                x_idx = int((agent.pos[0] + 6) / self.grid_size)
                y_idx = int((agent.pos[1] + 6) / self.grid_size)
                radius_idx = int(agent.radius * 0.5 / self.grid_size)
                for i in range(-radius_idx, radius_idx + 1):
                    for j in range(-radius_idx, radius_idx + 1):
                        if 0 <= y_idx + i < 120 and 0 <= x_idx + j < 120:
                            if i*i + j*j <= radius_idx*radius_idx:
                                self.grid[y_idx + i, x_idx + j] = 1

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < 120 and 0 <= new_y < 120 and self.grid[new_y, new_x] == 0:
                neighbors.append((new_x, new_y))
        return neighbors

    def a_star(self, start, goal):
        start_idx = (int((start[0] + 6) / self.grid_size), int((start[1] + 6) / self.grid_size))
        goal_idx = (int((goal[0] + 6) / self.grid_size), int((goal[1] + 6) / self.grid_size))
        
        open_set = {start_idx}
        closed_set = set()
        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: self.heuristic(start_idx, goal_idx)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal_idx:
                path = []
                while current in came_from:
                    x = (current[0] * self.grid_size) - 6
                    y = (current[1] * self.grid_size) - 6
                    path.append([x, y])
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal_idx)
        
        return None

    def calculate_individual_space(self, pos):
        v = np.linalg.norm(self.velocity)
        
        sigma_h = max(2 * v, 0.5)
        sigma_r = 0.5 * sigma_h
        sigma_s = (2/3) * sigma_h
        
        if v > 0:
            theta = np.arctan2(self.velocity[1], self.velocity[0])
        else:
            theta = 0
        
        A = (np.cos(theta)**2)/(2*sigma_h**2) + (np.sin(theta)**2)/(2*sigma_s**2)
        B = (np.sin(2*theta))/(4*sigma_h**2) - (np.sin(2*theta))/(4*sigma_s**2)
        C = (np.sin(theta)**2)/(2*sigma_h**2) + (np.cos(theta)**2)/(2*sigma_s**2)
        
        dx = pos[0] - self.pos[0]
        dy = pos[1] - self.pos[1]
        IS = np.exp(-(A*dx**2 + 2*B*dx*dy + C*dy**2))
        
        return IS

    def update(self, agents, obstacles):
        # finished 체크 제거 - 항상 활성 상태

        if self.current_waypoint < len(self.waypoints):
            target = self.waypoints[self.current_waypoint]
            
            self.create_grid(obstacles, agents)
            
            if np.sqrt((self.pos[0] - self.last_pos[0])**2 + (self.pos[1] - self.last_pos[1])**2) < self.speed * 0.1:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
            
            dist_to_waypoint = np.sqrt((self.pos[0] - target[0])**2 + (self.pos[1] - target[1])**2)
            if dist_to_waypoint < self.waypoint_threshold and not self.waypoint_reached:
                self.waypoint_reached = True
                self.current_waypoint += 1
                self.path = None
                self.stuck_count = 0
                
                # 순환 경로: 마지막 웨이포인트에 도달하면 처음으로 돌아감
                if self.current_waypoint >= len(self.waypoints):
                    self.current_waypoint = 0  # 처음 웨이포인트로 리셋
                    print(f"에이전트 순환: 웨이포인트 0으로 이동 (위치: {self.pos[0]:.2f}, {self.pos[1]:.2f})")
                    # finished = True 제거 - 계속 움직임
            
            if not self.path or len(self.path) < 2 or self.stuck_count > 10 or self.waypoint_reached:
                if self.current_waypoint < len(self.waypoints):
                    self.path = self.a_star(self.pos, self.waypoints[self.current_waypoint])
                    self.waypoint_reached = False
                    self.stuck_count = 0
            
            if self.path and len(self.path) > 1:
                next_point = self.path[1]
                dx = next_point[0] - self.pos[0]
                dy = next_point[1] - self.pos[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0:
                    self.pos[0] += (dx/dist) * self.speed
                    self.pos[1] += (dy/dist) * self.speed
                
                if dist < self.speed:
                    self.path.pop(0)
            
            self.velocity = [
                self.pos[0] - self.last_pos[0],
                self.pos[1] - self.last_pos[1]
            ]
            
            self.last_pos = self.pos.copy()

    def visualize(self):
        # finished 상태가 없으므로 항상 파란색 (활성 상태)
        return 'blue'

# DiPPeR 관련 클래스들
class ResNetEncoder(nn.Module):
    """Fused Cost Map을 Feature Vector로 변환하는 Visual Encoder"""
    def __init__(self, input_channels=1, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-18 기본 블록들
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
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
        x = self.fc(x)
        
        return x

class NoisePredictor(nn.Module):
    """DiPPeR의 핵심 Noise Prediction Network"""
    def __init__(self, visual_feature_dim=512, path_dim=2, max_timesteps=1000):
        super().__init__()
        self.path_dim = path_dim
        self.max_timesteps = max_timesteps
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Position embedding (시작점, 목표점)
        self.pos_embed = nn.Sequential(
            nn.Linear(4, 128),  # start_x, start_y, goal_x, goal_y
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Path encoder (현재 노이즈가 섞인 경로)
        self.path_encoder = nn.Sequential(
            nn.Linear(path_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Feature fusion
        total_feature_dim = visual_feature_dim + 128 + 128 + 256  # visual + time + pos + path
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, path_dim)  # 예측할 노이즈 차원
        )
        
    def forward(self, visual_features, noisy_path, timestep, start_goal_pos):
        # Timestep embedding
        t_embed = self.time_embed(timestep.float().unsqueeze(-1))
        
        # Position embedding
        pos_embed = self.pos_embed(start_goal_pos)
        
        # Path encoding (각 path point를 개별적으로 처리)
        batch_size, path_length, _ = noisy_path.shape
        path_flat = noisy_path.view(-1, self.path_dim)
        path_encoded = self.path_encoder(path_flat)
        path_encoded = path_encoded.view(batch_size, path_length, -1)
        path_encoded = torch.mean(path_encoded, dim=1)  # Path sequence를 평균으로 요약
        
        # Feature concatenation
        combined_features = torch.cat([
            visual_features, 
            t_embed, 
            pos_embed, 
            path_encoded
        ], dim=-1)
        
        # Noise prediction
        predicted_noise = self.fusion(combined_features)
        
        return predicted_noise.unsqueeze(1).expand(-1, path_length, -1)

class DiPPeR(nn.Module):
    """DiPPeR Diffusion Model for Path Planning"""
    def __init__(self, visual_feature_dim=512, path_dim=2, max_timesteps=1000):
        super().__init__()
        self.visual_encoder = ResNetEncoder(input_channels=1, feature_dim=visual_feature_dim)
        self.noise_predictor = NoisePredictor(visual_feature_dim, path_dim, max_timesteps)
        self.max_timesteps = max_timesteps
        
        # Diffusion parameters
        self.register_buffer('betas', self._cosine_beta_schedule(max_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, cost_map, path, timestep, start_goal_pos):
        # Visual encoding
        visual_features = self.visual_encoder(cost_map)
        
        # Noise prediction
        predicted_noise = self.noise_predictor(visual_features, path, timestep, start_goal_pos)
        
        return predicted_noise
    
    @torch.no_grad()
    def sample_path(self, cost_map, start_pos, goal_pos, path_length=100, num_inference_steps=50):
        """DiPPeR inference: 노이즈에서 경로 생성"""
        device = cost_map.device
        batch_size = cost_map.shape[0]
        
        # Random noise로 시작
        path = torch.randn(batch_size, path_length, 2, device=device)
        
        # 시작점과 목표점 고정 (Inpainting)
        path[:, 0, :] = start_pos
        path[:, -1, :] = goal_pos
        
        # Start/Goal position for conditioning
        start_goal = torch.cat([start_pos, goal_pos], dim=-1)
        
        # Reverse diffusion process
        timesteps = torch.linspace(self.max_timesteps-1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # Noise prediction
            predicted_noise = self.forward(cost_map, path, t_batch, start_goal)
            
            # Denoising step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            # DDPM sampling formula
            pred_original = (path - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            if t > 0:
                noise = torch.randn_like(path)
                path = torch.sqrt(alpha_t_prev) * pred_original + torch.sqrt(1 - alpha_t_prev) * noise
            else:
                path = pred_original
            
            # 시작점과 목표점 다시 고정
            path[:, 0, :] = start_pos
            path[:, -1, :] = goal_pos
        
        return path

class RobotSimulatorDiPPeR:
    def __init__(self, xml_file, model_path=None):
        self.xml_file = xml_file  # xml_file 속성 추가
        self.obstacles = []
        self.grid_size = 0.2
        self.load_obstacles(xml_file)
        self.robot_pos = [3, -4]
        self.robot_vel = [0, 0]
        self.target_pos = None
        self.current_path = []
        self.robot_speed = 0.1
        self.create_grid()
        self.agents = []
        self.load_agents(xml_file)
        self.stuck_count = 0
        self.last_pos = [3, -4]
        self.timestep_counter = 0
        
        # CCTV 커버리지 영역
        self.cctv_coverage = [
            {'name': 'CCTV1', 'points': [(-5.6, -1.6), (-0.15, -5.2)], 'color': 'lightgreen'},
            {'name': 'CCTV2', 'points': [(-1.5, 2.0), (-0.15, -5.2)], 'color': 'gold'},
            {'name': 'CCTV3', 'points': [(2.4, 1.51), (3.8, -5.21)], 'color': 'lightcoral'}
        ]
        
        # Cost maps
        self.is_map = np.zeros((60, 60))
        self.fused_cost_map = np.zeros((60, 60))
        
        # CGIP parameters
        self.gamma1 = 0.5
        self.is_threshold = np.exp(-4)
        
        # DiPPeR 모델 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dipperp_model = DiPPeR(visual_feature_dim=512, path_dim=2, max_timesteps=1000)
        self.use_dipperp = True  # DiPPeR 사용 여부 (학습 시 False로 설정 가능)
        
        if model_path:
            # 모델 파일 경로 처리
            if not os.path.exists(model_path) and not model_path.startswith('models/'):
                model_path = f'models/{model_path}'
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    # 학습 시 저장된 체크포인트 형식
                    self.dipperp_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # 단순 모델 상태만 저장된 형식
                    self.dipperp_model.load_state_dict(checkpoint)
                print(f"DiPPeR 모델 로드 완료: {model_path}")
            except Exception as e:
                print(f"DiPPeR 모델 로드 실패: {e}, 랜덤 초기화 사용")
        else:
            print("DiPPeR 모델 랜덤 초기화 (학습되지 않은 상태)")
            
        self.dipperp_model.to(self.device)
        self.dipperp_model.eval()
        
        # 시각화 초기화
        plt.ion()
        self.fig1 = plt.figure(1, figsize=(8, 8))
        self.fig2 = plt.figure(2, figsize=(8, 8))
        self.fig3 = plt.figure(3, figsize=(8, 8))

    def load_agents(self, xml_file):
        # XML 파일 경로 처리
        if not os.path.exists(xml_file) and not xml_file.startswith('scenarios/'):
            xml_file = f'scenarios/{xml_file}'
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        waypoints = {}
        for wp in root.findall('waypoint'):
            wp_id = wp.get('id')
            x = float(wp.get('x'))
            y = float(wp.get('y'))
            waypoints[wp_id] = (x, y)
        
        for agent in root.findall('agent'):
            x = float(agent.get('x'))
            y = float(agent.get('y'))
            waypoint_list = []
            for wp in agent.findall('addwaypoint'):
                wp_id = wp.get('id')
                waypoint_list.append(waypoints[wp_id])
            self.agents.append(Agent(x, y, waypoint_list))

    def load_obstacles(self, xml_file):
        # XML 파일 경로 처리
        if not os.path.exists(xml_file) and not xml_file.startswith('scenarios/'):
            xml_file = f'scenarios/{xml_file}'
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obstacle in root.findall('obstacle'):
            x1 = float(obstacle.get('x1'))
            y1 = float(obstacle.get('y1'))
            x2 = float(obstacle.get('x2'))
            y2 = float(obstacle.get('y2'))
            self.obstacles.append([(x1, y1), (x2, y2)])
    
    def create_grid(self):
        self.grid = np.zeros((60, 60))
        
        # 내부 장애물 직사각형들을 직접 정의 (XML 분석 결과)
        rectangles = [
            # 첫 번째 직사각형: (-0.06, -2.54) ~ (2.19, 0.21)
            {"x1": -0.06, "y1": -2.54, "x2": 2.19, "y2": 0.21},
            # 두 번째 직사각형: (-3.61, -1.34) ~ (-1.61, 2.66)  
            {"x1": -3.61, "y1": -1.34, "x2": -1.61, "y2": 2.66}
        ]
        
        # 직사각형들을 그리드에 채우기
        for rect in rectangles:
            x1, y1 = rect["x1"], rect["y1"]
            x2, y2 = rect["x2"], rect["y2"]
            
            # 좌표를 그리드 인덱스로 변환
            x1_idx = int((x1 + 6) / self.grid_size)
            y1_idx = int((y1 + 6) / self.grid_size)
            x2_idx = int((x2 + 6) / self.grid_size)
            y2_idx = int((y2 + 6) / self.grid_size)
            
            # 직사각형 영역 전체를 장애물로 설정
            min_x, max_x = min(x1_idx, x2_idx), max(x1_idx, x2_idx)
            min_y, max_y = min(y1_idx, y2_idx), max(y1_idx, y2_idx)
            
            for y in range(max(0, min_y), min(60, max_y + 1)):
                for x in range(max(0, min_x), min(60, max_x + 1)):
                    self.grid[y, x] = 1
        
        # 외곽 경계도 추가 (맵 테두리 - 두껍게)
        self.grid[0:2, :] = 1      # 아래쪽 경계 (2픽셀 두께)
        self.grid[-2:, :] = 1      # 위쪽 경계 (2픽셀 두께)
        self.grid[:, 0:2] = 1      # 왼쪽 경계 (2픽셀 두께)  
        self.grid[:, -2:] = 1      # 오른쪽 경계 (2픽셀 두께)
    


    def fused_cost_map_to_image(self):
        """Fused Cost Map을 DiPPeR 입력용 이미지로 변환"""
        # 60x60 Fused Cost Map을 0-255 범위로 정규화
        cost_map_normalized = np.clip(self.fused_cost_map * 255, 0, 255).astype(np.uint8)
        
        # PyTorch tensor로 변환 (1, 1, 60, 60)
        cost_map_tensor = torch.from_numpy(cost_map_normalized).float().unsqueeze(0).unsqueeze(0) / 255.0
        cost_map_tensor = cost_map_tensor.to(self.device)
        
        return cost_map_tensor

    def calculate_intention_alignment(self, agent):
        if self.target_pos is None:
            return 0
        
        v_r_to_g = np.array([
            self.target_pos[0] - self.robot_pos[0],
            self.target_pos[1] - self.robot_pos[1]
        ])
        
        v_h = np.array([
            agent.pos[0] - agent.last_pos[0],
            agent.pos[1] - agent.last_pos[1]
        ])
        v_r = np.array(self.robot_vel)
        v_r_to_h = v_h - v_r
        
        if np.linalg.norm(v_r_to_g) == 0 or np.linalg.norm(v_r_to_h) == 0:
            return 0
        
        tau = np.dot(v_r_to_g, v_r_to_h) / (np.linalg.norm(v_r_to_g) * np.linalg.norm(v_r_to_h))
        return tau

    def calculate_social_cost(self, pos):
        total_social_cost = 0
        
        for agent in self.agents:
            if self.is_in_cctv_coverage(agent.pos):
                dist = np.sqrt((pos[0] - agent.pos[0])**2 + (pos[1] - agent.pos[1])**2)
                
                if dist <= 3.0:
                    IS = agent.calculate_individual_space(pos)
                    
                    if IS > self.is_threshold:
                        tau = self.calculate_intention_alignment(agent)
                        social_cost = self.gamma1 * (1 - tau)
                        total_social_cost += social_cost
        
        return total_social_cost

    def update_fused_cost_map(self):
        """CGIP의 Fused Cost Map 업데이트 (장애물 + 사회적 비용)"""
        # 1. 기본 장애물 맵으로 시작 (더 강한 장애물 신호)
        self.fused_cost_map = self.grid.copy()
        
        # 장애물 주변에 높은 비용 영역 추가 (DiPPeR가 장애물을 더 잘 인식하도록)
        obstacle_penalty_map = np.zeros((60, 60))
        for y in range(60):
            for x in range(60):
                if self.grid[y, x] >= 1.0:  # 장애물인 경우
                    # 주변 3x3 영역에 높은 비용 부여
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < 60 and 0 <= nx < 60 and self.grid[ny, nx] < 1.0:
                                distance = np.sqrt(dx*dx + dy*dy)
                                penalty = max(0, 0.8 - distance * 0.2)  # 거리에 따른 페널티
                                obstacle_penalty_map[ny, nx] = max(obstacle_penalty_map[ny, nx], penalty)
        
        # 장애물 페널티 적용
        self.fused_cost_map = np.maximum(self.fused_cost_map, obstacle_penalty_map)
        
        # 2. 에이전트들의 사회적 비용 추가
        for agent in self.agents:
            if self.is_in_cctv_coverage(agent.pos):
                agent_x_idx = int((agent.pos[0] + 6) / self.grid_size)
                agent_y_idx = int((agent.pos[1] + 6) / self.grid_size)
                
                radius_cells = int(3.0 / self.grid_size)
                for i in range(-radius_cells, radius_cells + 1):
                    for j in range(-radius_cells, radius_cells + 1):
                        x_idx = agent_x_idx + j
                        y_idx = agent_y_idx + i
                        
                        if 0 <= x_idx < 60 and 0 <= y_idx < 60:
                            # 장애물이 아닌 곳에만 사회적 비용 추가
                            if self.fused_cost_map[y_idx, x_idx] < 1.0:  # 장애물(1.0)이 아닌 경우
                                x = (x_idx * self.grid_size) - 6
                                y = (y_idx * self.grid_size) - 6
                                social_cost = self.calculate_social_cost([x, y])
                                # 기존 비용과 사회적 비용 중 큰 값 사용
                                combined_cost = max(self.fused_cost_map[y_idx, x_idx], min(social_cost, 0.9))
                                self.fused_cost_map[y_idx, x_idx] = combined_cost

    def dipperp_path_planning(self, start, goal):
        """DiPPeR로 경로 계획 (A* 대체)"""
        # DiPPeR 비활성화 시 A* 사용 (학습 중에는 A* 사용)
        if not self.use_dipperp or not hasattr(self, 'dipperp_model') or self.dipperp_model is None:
            return self.fallback_astar_planning(start, goal)
            
        print(f"\nDiPPeR 경로 계획 시작:")
        print(f"시작점: ({start[0]:.2f}, {start[1]:.2f})")
        print(f"목표점: ({goal[0]:.2f}, {goal[1]:.2f})")
        
        try:
            # Fused Cost Map을 이미지로 변환
            cost_map_image = self.fused_cost_map_to_image()
            
            # 시작점과 목표점을 정규화된 좌표로 변환 (-6~6 → -1~1)
            start_normalized = torch.tensor([[start[0]/6.0, start[1]/6.0]], dtype=torch.float32, device=self.device)
            goal_normalized = torch.tensor([[goal[0]/6.0, goal[1]/6.0]], dtype=torch.float32, device=self.device)
            
            # DiPPeR로 경로 생성
            with torch.no_grad():
                generated_path = self.dipperp_model.sample_path(
                    cost_map_image, 
                    start_normalized, 
                    goal_normalized, 
                    path_length=50,  # 경로 길이
                    num_inference_steps=20  # 추론 스텝 수 (빠른 실행을 위해 줄임)
                )
            
            # 정규화된 좌표를 실제 좌표로 변환
            path_real = generated_path[0].cpu().numpy() * 6.0  # (-1~1) → (-6~6)
            
            # 경로를 리스트 형태로 변환
            path_list = [[float(point[0]), float(point[1])] for point in path_real]
            
            # 경로 안전성 검증 및 수정
            safe_path = []
            for i, point in enumerate(path_list):
                if self.is_position_safe(point):
                    safe_path.append(point)
                else:
                    print(f"위험한 웨이포인트 감지: {point} -> A* 보정 적용")
                    # 이전 안전한 점에서 다음 안전한 점으로 A* 경로 생성
                    if safe_path:
                        prev_safe = safe_path[-1]
                        # 다음 안전한 점 찾기
                        next_safe = None
                        for j in range(i+1, len(path_list)):
                            if self.is_position_safe(path_list[j]):
                                next_safe = path_list[j]
                                break
                        
                        if next_safe:
                            # A*로 안전한 경로 생성
                            correction_path = self.fallback_astar_planning(prev_safe, next_safe)
                            if correction_path and len(correction_path) > 2:
                                # 중간 점들만 추가 (시작점, 끝점 제외)
                                safe_path.extend(correction_path[1:-1])
                        
                        safe_path.append(next_safe if next_safe else point)
                    else:
                        safe_path.append(point)  # 첫 번째 점이면 그대로 추가
            
            # 안전성 검증된 경로가 너무 짧으면 폴백
            if len(safe_path) < 5:
                print("DiPPeR 경로가 너무 짧음. A* 폴백 사용")
                return self.fallback_astar_planning(start, goal)
            
            print(f"DiPPeR 경로 생성 완료: {len(safe_path)}개 웨이포인트 (안전성 검증됨)")
            return safe_path
            
        except Exception as e:
            print(f"DiPPeR 경로 계획 실패: {e}")
            print("폴백: A* 알고리즘 사용")
            # 폴백: A* 알고리즘으로 장애물 고려한 경로 생성
            return self.fallback_astar_planning(start, goal)

    def fallback_astar_planning(self, start, goal):
        """폴백 A* 경로 계획 (장애물 고려)"""
        print(f"폴백 A* 경로 계획: ({start[0]:.2f}, {start[1]:.2f}) → ({goal[0]:.2f}, {goal[1]:.2f})")
        
        start_idx = (int((start[0] + 6) / self.grid_size), int((start[1] + 6) / self.grid_size))
        goal_idx = (int((goal[0] + 6) / self.grid_size), int((goal[1] + 6) / self.grid_size))
        
        # 유효성 검사
        if not (0 <= start_idx[0] < 60 and 0 <= start_idx[1] < 60 and 
                0 <= goal_idx[0] < 60 and 0 <= goal_idx[1] < 60):
            print("시작점 또는 목표점이 맵 범위를 벗어났습니다.")
            return [[start[0], start[1]], [goal[0], goal[1]]]
        
        # 장애물 체크
        if self.fused_cost_map[start_idx[1], start_idx[0]] >= 1.0 or self.fused_cost_map[goal_idx[1], goal_idx[0]] >= 1.0:
            print("시작점 또는 목표점이 장애물 위에 있습니다.")
            return [[start[0], start[1]], [goal[0], goal[1]]]
        
        # A* 알고리즘
        open_set = {start_idx}
        closed_set = set()
        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: self.heuristic_2d(start_idx, goal_idx)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal_idx:
                # 경로 재구성
                path = []
                while current in came_from:
                    x = (current[0] * self.grid_size) - 6
                    y = (current[1] * self.grid_size) - 6
                    path.append([x, y])
                    current = came_from[current]
                path.append([start[0], start[1]])
                path.reverse()
                print(f"A* 경로 생성 완료: {len(path)}개 웨이포인트")
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            # 8방향 이동
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (neighbor in closed_set or 
                    not (0 <= neighbor[0] < 60 and 0 <= neighbor[1] < 60) or
                    self.fused_cost_map[neighbor[1], neighbor[0]] >= 1.0):  # 장애물 피하기
                    continue
                
                # 사회적 비용 고려
                social_cost = self.fused_cost_map[neighbor[1], neighbor[0]]
                move_cost = np.sqrt(dx*dx + dy*dy) + social_cost * 2.0  # 사회적 비용에 가중치 부여
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic_2d(neighbor, goal_idx)
        
        print("A* 경로를 찾을 수 없습니다. 직선 경로 사용.")
        return [[start[0], start[1]], [goal[0], goal[1]]]
    
    def heuristic_2d(self, a, b):
        """2D 휴리스틱 함수"""
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def move_robot(self, target_pos):
        self.target_pos = target_pos
        self.update_fused_cost_map()
        self.current_path = self.dipperp_path_planning(self.robot_pos, target_pos)  # A* 대신 DiPPeR 사용
        if not self.current_path:
            print("경로를 찾을 수 없습니다.")
            return False
        return True

    def update_robot(self):
        if self.target_pos is None:
            return

        if np.sqrt((self.robot_pos[0] - self.last_pos[0])**2 + (self.robot_pos[1] - self.last_pos[1])**2) < self.robot_speed * 0.1:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        self.update_fused_cost_map()

        # 경로 재계획 조건
        need_replan = False
        
        if not self.current_path or len(self.current_path) < 2:
            need_replan = True
        elif self.stuck_count > 10:
            need_replan = True
        elif len(self.current_path) > 1:
            next_point = self.current_path[1]
            x_idx = int((next_point[0] + 6) / self.grid_size)
            y_idx = int((next_point[1] + 6) / self.grid_size)
            
            if (0 <= x_idx < 60 and 0 <= y_idx < 60 and 
                (self.fused_cost_map[y_idx, x_idx] > 0.8 or
                 self.calculate_social_cost(next_point) > 0.8)):
                need_replan = True
        
        # DiPPeR 경로 재계획
        if need_replan:
            print(f"\n타임스텝 {self.timestep_counter}: DiPPeR 경로 재계산")
            self.current_path = self.dipperp_path_planning(self.robot_pos, self.target_pos)
            self.stuck_count = 0
        
        if not self.current_path:
            print("경로를 찾을 수 없습니다.")
            return

        # 로봇 이동 (장애물 체크 포함)
        next_point = self.current_path[1] if len(self.current_path) > 1 else self.current_path[0]
        dx = next_point[0] - self.robot_pos[0]
        dy = next_point[1] - self.robot_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < self.robot_speed:
            # 목표점이 안전한지 체크
            if self.is_position_safe(next_point):
                self.robot_pos = [next_point[0], next_point[1]]
                if len(self.current_path) > 1:
                    self.current_path.pop(0)
            else:
                print(f"목표점 ({next_point[0]:.2f}, {next_point[1]:.2f})이 안전하지 않음")
        else:
            # 다음 위치 계산
            new_x = self.robot_pos[0] + (dx/dist) * self.robot_speed
            new_y = self.robot_pos[1] + (dy/dist) * self.robot_speed
            new_pos = [new_x, new_y]
            
            # 새 위치가 안전한지 체크
            if self.is_position_safe(new_pos):
                self.robot_pos = new_pos
            else:
                print(f"새 위치 ({new_x:.2f}, {new_y:.2f})이 안전하지 않음. 이동 중단.")
                # 경로 재계획 강제 실행
                self.stuck_count = 11
        
        self.robot_vel = [
            self.robot_pos[0] - self.last_pos[0],
            self.robot_pos[1] - self.last_pos[1]
        ]
        
        self.last_pos = self.robot_pos.copy()

    def is_position_safe(self, pos):
        """위치가 안전한지 체크 (맵 경계 + 장애물)"""
        x, y = pos[0], pos[1]
        
        # 맵 경계 체크 (여유분 0.2m)
        if not (-5.8 <= x <= 5.8 and -5.8 <= y <= 5.8):
            return False
        
        # 그리드가 초기화되지 않았다면 초기화
        if not hasattr(self, 'grid') or self.grid is None:
            self.create_grid()
        
        # 그리드 인덱스 변환
        x_idx = int((x + 6) / self.grid_size)
        y_idx = int((y + 6) / self.grid_size)
        
        # 그리드 범위 체크
        if not (0 <= x_idx < 60 and 0 <= y_idx < 60):
            return False
        
        # 그리드에서 장애물 체크 (1.0은 장애물)
        if self.grid[y_idx, x_idx] >= 1.0:
            return False
        
        # 추가 안전 마진 체크 (주변 셀도 확인)
        margin_cells = 1  # 1셀 마진 (약 0.2m)
        for i in range(-margin_cells, margin_cells + 1):
            for j in range(-margin_cells, margin_cells + 1):
                check_x = x_idx + j
                check_y = y_idx + i
                if (0 <= check_x < 60 and 0 <= check_y < 60 and 
                    self.grid[check_y, check_x] >= 1.0):
                    return False
        
        return True

    def is_in_cctv_coverage(self, pos):
        for cctv in self.cctv_coverage:
            x1, y1 = cctv['points'][0]
            x2, y2 = cctv['points'][1]
            if (min(x1, x2) <= pos[0] <= max(x1, x2) and 
                min(y1, y2) <= pos[1] <= max(y1, y2)):
                return True
        return False
    
    def update(self):
        self.is_map = np.zeros((60, 60))
        
        for agent in self.agents:
            if self.is_in_cctv_coverage(agent.pos):
                velocity = [agent.pos[0] - agent.last_pos[0], 
                          agent.pos[1] - agent.last_pos[1]]
            agent.update(self.agents, self.obstacles)
        
        self.update_robot()
        self.timestep_counter += 1
    
    def visualize(self, path=None):
        # 메인 시뮬레이션 창
        plt.figure(1)
        plt.clf()
        
        # CCTV 커버리지 영역
        for cctv in self.cctv_coverage:
            x1, y1 = cctv['points'][0]
            x2, y2 = cctv['points'][1]
            plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], color=cctv['color'], alpha=0.3)
            plt.text((x1 + x2)/2, (y1 + y2)/2, cctv['name'], ha='center', va='center')
        
        # 장애물 (더 두껍게 표시)
        for obs in self.obstacles:
            plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=4)
        
        # 에이전트
        for agent in self.agents:
            color = agent.visualize()
            circle = plt.Circle(agent.pos, agent.radius, color=color, alpha=0.5)
            plt.gca().add_patch(circle)
        
        # 로봇
        plt.plot(self.robot_pos[0], self.robot_pos[1], 'ro', markersize=10)
        
        # DiPPeR 생성 경로
        if self.current_path:
            path_x = [p[0] for p in self.current_path]
            path_y = [p[1] for p in self.current_path]
            plt.plot(path_x, path_y, 'r--', alpha=0.7, linewidth=2, label='DiPPeR Path')
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title('Robot Simulation with DiPPeR')
        plt.legend()
        
        # Fused Cost Map 창
        plt.figure(2)
        plt.clf()
        
        plt.imshow(self.fused_cost_map, extent=[-6, 6, -6, 6], origin='lower', 
                  cmap='YlOrRd', alpha=0.7)
        
        for cctv in self.cctv_coverage:
            x1, y1 = cctv['points'][0]
            x2, y2 = cctv['points'][1]
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'k-', linewidth=2)
            plt.text((x1 + x2)/2, (y1 + y2)/2, cctv['name'], ha='center', va='center', 
                    color='black', bbox=dict(facecolor='white', alpha=0.7))
        
        for obs in self.obstacles:
            plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2)
        
        plt.colorbar(label='Fused Cost Value')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title('Fused Cost Map (DiPPeR Input)')
        
        # Individual Space 시각화 창 (Figure 3)
        plt.figure(3)
        plt.clf()
        
        # Individual Space 계산 및 시각화
        x_range = np.linspace(-6, 6, 60)
        y_range = np.linspace(-6, 6, 60)
        X, Y = np.meshgrid(x_range, y_range)
        IS_map = np.zeros((60, 60))
        
        for agent in self.agents:
            for i in range(60):
                    for j in range(60):
                        pos = [X[i, j], Y[i, j]]
                        IS_value = agent.calculate_individual_space(pos)
                        IS_map[i, j] = max(IS_map[i, j], IS_value)
        
        plt.imshow(IS_map, extent=[-6, 6, -6, 6], origin='lower', 
                  cmap='Blues', alpha=0.7)
        
        # 장애물과 에이전트 표시
        for obs in self.obstacles:
            plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2)
        
        for agent in self.agents:
            color = 'blue'  # 항상 활성 상태 (파란색)
            circle = plt.Circle(agent.pos, agent.radius, color=color, alpha=0.8)
            plt.gca().add_patch(circle)
        
        # 로봇 위치
        plt.plot(self.robot_pos[0], self.robot_pos[1], 'ro', markersize=10)
        
        plt.colorbar(label='Individual Space Value')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title('Individual Space Visualization')
        
        plt.draw()
        plt.pause(0.01)

def main():
    import sys
    
    # XML 파일 경로 설정
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = "Congestion1.xml"  # 기본값
        print(f"사용법: python robot_simuator_dippeR.py <xml_file_path>")
        print(f"기본 예시 파일 '{xml_file}'로 실행합니다.")
    
    # DiPPeR 모델 경로 (없으면 None)
    model_path = None  # "path/to/trained/dipperp_model.pth"
    
    simulator = RobotSimulatorDiPPeR(xml_file, model_path)
    
    while True:
        try:
            x = float(input("목표 x 좌표를 입력하세요 (-6 ~ 6): "))
            y = float(input("목표 y 좌표를 입력하세요 (-6 ~ 6): "))
            
            if not (-6 <= x <= 6 and -6 <= y <= 6):
                print("좌표는 -6에서 6 사이여야 합니다.")
                continue
                
            simulator.move_robot([x, y])
            
            while True:
                simulator.update()
                simulator.visualize()
                time.sleep(0.1)
                
                if np.sqrt((simulator.robot_pos[0] - x)**2 + (simulator.robot_pos[1] - y)**2) < 0.2:
                    print(f"목표 지점 ({x}, {y})에 도달했습니다!")
                    break
                    
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            plt.close('all')
            break

if __name__ == "__main__":
    main()
