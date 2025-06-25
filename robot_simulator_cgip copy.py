import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import xml.etree.ElementTree as ET
import time

class Agent:
    def __init__(self, x, y, waypoints):
        self.pos = [float(x), float(y)]
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.speed = 0.1  # 이동 속도
        self.radius = 0.3  # 에이전트 반경
        self.path = []
        self.grid_size = 0.1
        self.grid = np.zeros((120, 120))
        self.stuck_count = 0
        self.last_pos = [float(x), float(y)]
        self.finished = False  # 웨이포인트 완료 여부
        self.waypoint_reached = False  # 웨이포인트 도달 상태
        self.waypoint_threshold = 0.2  # 웨이포인트 도달 판정 거리
        self.velocity = [0.0, 0.0]  # 속도 벡터 추가

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
        
        # 다른 에이전트를 장애물로 추가 (반경을 줄임)
        for agent in agents:
            if agent != self:
                x_idx = int((agent.pos[0] + 6) / self.grid_size)
                y_idx = int((agent.pos[1] + 6) / self.grid_size)
                radius_idx = int(agent.radius * 0.5 / self.grid_size)  # 반경을 절반으로 줄임
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
        # 속도 계산
        v = np.linalg.norm(self.velocity)
        
        # 속도에 따른 파라미터 계산
        sigma_h = max(2 * v, 0.5)
        sigma_r = 0.5 * sigma_h
        sigma_s = (2/3) * sigma_h
        
        # 방향 계산
        if v > 0:
            theta = np.arctan2(self.velocity[1], self.velocity[0])
        else:
            theta = 0
        
        # Eq. (2) 계수 계산
        A = (np.cos(theta)**2)/(2*sigma_h**2) + (np.sin(theta)**2)/(2*sigma_s**2)
        B = (np.sin(2*theta))/(4*sigma_h**2) - (np.sin(2*theta))/(4*sigma_s**2)
        C = (np.sin(theta)**2)/(2*sigma_h**2) + (np.cos(theta)**2)/(2*sigma_s**2)
        
        # Eq. (1) 계산
        dx = pos[0] - self.pos[0]
        dy = pos[1] - self.pos[1]
        IS = np.exp(-(A*dx**2 + 2*B*dx*dy + C*dy**2))
        
        return IS

    def update(self, agents, obstacles):
        if self.finished:
            return

        if self.current_waypoint < len(self.waypoints):
            target = self.waypoints[self.current_waypoint]
            
            # 그리드 업데이트 및 경로 재계획
            self.create_grid(obstacles, agents)
            
            # 현재 위치가 이전 위치와 거의 같으면 stuck_count 증가
            if np.sqrt((self.pos[0] - self.last_pos[0])**2 + (self.pos[1] - self.last_pos[1])**2) < self.speed * 0.1:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
            
            # 웨이포인트 도달 체크
            dist_to_waypoint = np.sqrt((self.pos[0] - target[0])**2 + (self.pos[1] - target[1])**2)
            if dist_to_waypoint < self.waypoint_threshold and not self.waypoint_reached:
                self.waypoint_reached = True
                self.current_waypoint += 1
                self.path = None
                self.stuck_count = 0
                
                if self.current_waypoint >= len(self.waypoints):
                    self.finished = True
                    print(f"에이전트가 모든 웨이포인트를 완료했습니다. 최종 위치: ({self.pos[0]:.2f}, {self.pos[1]:.2f})")
                    return
            
            # 경로 재계획 조건
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
                
                # 다음 경로 포인트에 도달했는지 확인
                if dist < self.speed:
                    self.path.pop(0)
            
            # 속도 벡터 업데이트
            self.velocity = [
                self.pos[0] - self.last_pos[0],
                self.pos[1] - self.last_pos[1]
            ]
            
            # 현재 위치 저장
            self.last_pos = self.pos.copy()

    def visualize(self):
        if self.finished:
            return 'green'  # 완료된 에이전트는 초록색으로 표시
        return 'blue'  # 진행 중인 에이전트는 파란색으로 표시

class RobotSimulator:
    def __init__(self, xml_file):
        self.obstacles = []
        self.grid_size = 0.2  # 그리드 크기를 0.2m로 증가
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
        
        # CCTV 커버리지 영역 정의
        self.cctv_coverage = [
            {
                'name': 'CCTV1',
                'points': [(-5.6, -1.6), (-0.15, -5.2)],
                'color': 'lightgreen'
            },
            {
                'name': 'CCTV2',
                'points': [(-1.5, 2.0), (-0.15, -5.2)],
                'color': 'gold'
            },
            {
                'name': 'CCTV3',
                'points': [(2.4, 1.51), (3.8, -5.21)],
                'color': 'lightcoral'
            }
        ]
        
        # Individual Space 맵 초기화 (60x60 그리드)
        self.is_map = np.zeros((60, 60))
        self.fused_cost_map = np.zeros((60, 60))
        
        # CIGP 파라미터
        self.gamma1 = 0.5
        self.is_threshold = np.exp(-4)  # Individual Space 임계값
        
        # 시각화를 위한 figure 초기화
        plt.ion()  # 대화형 모드 활성화
        self.fig1 = plt.figure(1, figsize=(8, 8))  # 메인 시뮬레이션 창
        self.fig2 = plt.figure(2, figsize=(8, 8))  # Individual Space 창
        self.fig3 = plt.figure(3, figsize=(8, 8))  # Fused Cost Map 창

    def load_agents(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 웨이포인트 로드
        waypoints = {}
        for wp in root.findall('waypoint'):
            wp_id = wp.get('id')
            x = float(wp.get('x'))
            y = float(wp.get('y'))
            waypoints[wp_id] = (x, y)
        
        # 에이전트 로드
        for agent in root.findall('agent'):
            x = float(agent.get('x'))
            y = float(agent.get('y'))
            waypoint_list = []
            for wp in agent.findall('addwaypoint'):
                wp_id = wp.get('id')
                waypoint_list.append(waypoints[wp_id])
            self.agents.append(Agent(x, y, waypoint_list))

    def load_obstacles(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obstacle in root.findall('obstacle'):
            x1 = float(obstacle.get('x1'))
            y1 = float(obstacle.get('y1'))
            x2 = float(obstacle.get('x2'))
            y2 = float(obstacle.get('y2'))
            self.obstacles.append([(x1, y1), (x2, y2)])
    
    def create_grid(self):
        # 60x60 그리드로 변경
        self.grid = np.zeros((60, 60))
        for obs in self.obstacles:
            x1, y1 = obs[0]
            x2, y2 = obs[1]
            x1_idx = int((x1 + 6) / self.grid_size)
            y1_idx = int((y1 + 6) / self.grid_size)
            x2_idx = int((x2 + 6) / self.grid_size)
            y2_idx = int((y2 + 6) / self.grid_size)
            self.grid[min(y1_idx, y2_idx):max(y1_idx, y2_idx)+1, 
                     min(x1_idx, x2_idx):max(x1_idx, x2_idx)+1] = 1

    def update_dynamic_obstacles(self):
        # 에이전트를 동적 장애물로 추가
        temp_grid = self.grid.copy()
        for agent in self.agents:
            x_idx = int((agent.pos[0] + 6) / self.grid_size)
            y_idx = int((agent.pos[1] + 6) / self.grid_size)
            radius_idx = int(agent.radius / self.grid_size)
            
            for i in range(-radius_idx, radius_idx + 1):
                for j in range(-radius_idx, radius_idx + 1):
                    if 0 <= x_idx + i < 120 and 0 <= y_idx + j < 120:
                        if i*i + j*j <= radius_idx*radius_idx:
                            temp_grid[y_idx + j, x_idx + i] = 1
        return temp_grid

    def get_neighbors(self, pos, dynamic_grid):
        x, y = pos
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < 120 and 0 <= new_y < 120:
                if dynamic_grid[new_y, new_x] != 1:
                    neighbors.append((new_x, new_y))
        return neighbors

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def a_star(self, start, goal):
        start = (int((start[0] + 6) / self.grid_size), int((start[1] + 6) / self.grid_size))
        goal = (int((goal[0] + 6) / self.grid_size), int((goal[1] + 6) / self.grid_size))
        
        dynamic_grid = self.update_dynamic_obstacles()
        
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current, dynamic_grid):
                new_cost = cost_so_far[current] + self.heuristic(current, next_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        path = []
        current = goal
        while current is not None:
            x = current[0] * self.grid_size - 6
            y = current[1] * self.grid_size - 6
            path.append((x, y))
            current = came_from.get(current)
        path.reverse()
        return path

    def calculate_intention_alignment(self, agent):
        # 로봇의 의도 벡터 계산
        if self.target_pos is None:
            return 0
        
        v_r_to_g = np.array([
            self.target_pos[0] - self.robot_pos[0],
            self.target_pos[1] - self.robot_pos[1]
        ])
        
        # 에이전트의 상대 속도 벡터 계산
        v_h = np.array([
            agent.pos[0] - agent.last_pos[0],
            agent.pos[1] - agent.last_pos[1]
        ])
        v_r = np.array(self.robot_vel)
        v_r_to_h = v_h - v_r
        
        # 벡터 크기가 0인 경우 처리
        if np.linalg.norm(v_r_to_g) == 0 or np.linalg.norm(v_r_to_h) == 0:
            return 0
        
        # 의도 정렬 점수 계산 (Eq. 4)
        tau = np.dot(v_r_to_g, v_r_to_h) / (np.linalg.norm(v_r_to_g) * np.linalg.norm(v_r_to_h))
        return tau

    def calculate_social_cost(self, pos):
        total_social_cost = 0
        x_idx = int((pos[0] + 6) / self.grid_size)
        y_idx = int((pos[1] + 6) / self.grid_size)
        
        if not (0 <= x_idx < 60 and 0 <= y_idx < 60):
            return float('inf')
        
        for agent in self.agents:
            if not agent.finished and self.is_in_cctv_coverage(agent.pos):
                # 에이전트와의 거리 계산
                dist = np.sqrt((pos[0] - agent.pos[0])**2 + (pos[1] - agent.pos[1])**2)
                
                # 에이전트의 영향 범위 내에 있는 경우에만 계산 (3m 반경)
                if dist <= 3.0:
                    # Individual Space 계산
                    IS = self.calculate_individual_space(pos, agent)
                    
                    # Individual Space 임계값 체크
                    if IS > self.is_threshold:
                        tau = self.calculate_intention_alignment(agent)
                        social_cost = self.gamma1 * (1 - tau)
                        total_social_cost += social_cost
                        
                        print(f"\nCIGP 판단 과정:")
                        print(f"위치: ({pos[0]:.2f}, {pos[1]:.2f})")
                        print(f"에이전트 위치: ({agent.pos[0]:.2f}, {agent.pos[1]:.2f})")
                        print(f"거리: {dist:.2f}m")
                        print(f"Individual Space 값: {IS:.2f}")
                        print(f"의도 정렬 점수(τ): {tau:.2f}")
                        print(f"사회적 비용(S_IS): {social_cost:.2f}")
        
        return total_social_cost

    def update_fused_cost_map(self):
        # Fused Cost Map 업데이트
        self.fused_cost_map = self.grid.copy()
        
        # 에이전트의 영향 범위 내의 그리드 셀만 계산
        for agent in self.agents:
            if not agent.finished and self.is_in_cctv_coverage(agent.pos):
                agent_x_idx = int((agent.pos[0] + 6) / self.grid_size)
                agent_y_idx = int((agent.pos[1] + 6) / self.grid_size)
                
                # 에이전트 주변 3m 반경 내의 그리드 셀만 계산
                radius_cells = int(3.0 / self.grid_size)
                for i in range(-radius_cells, radius_cells + 1):
                    for j in range(-radius_cells, radius_cells + 1):
                        x_idx = agent_x_idx + j
                        y_idx = agent_y_idx + i
                        
                        if 0 <= x_idx < 60 and 0 <= y_idx < 60:  # 60x60 그리드로 변경
                            x = (x_idx * self.grid_size) - 6
                            y = (y_idx * self.grid_size) - 6
                            if self.fused_cost_map[y_idx, x_idx] == 0:
                                social_cost = self.calculate_social_cost([x, y])
                                self.fused_cost_map[y_idx, x_idx] = min(social_cost, 1.0)

    def a_star_cigp(self, start, goal):
        start_idx = (int((start[0] + 6) / self.grid_size), int((start[1] + 6) / self.grid_size))
        goal_idx = (int((goal[0] + 6) / self.grid_size), int((goal[1] + 6) / self.grid_size))
        
        print(f"\nCIGP 경로 계획 시작:")
        print(f"시작점: ({start[0]:.2f}, {start[1]:.2f})")
        print(f"목표점: ({goal[0]:.2f}, {goal[1]:.2f})")
        
        # 시작점과 목표점이 유효한지 확인
        if not (0 <= start_idx[0] < 120 and 0 <= start_idx[1] < 120 and 
                0 <= goal_idx[0] < 120 and 0 <= goal_idx[1] < 120):
            print("시작점 또는 목표점이 맵 범위를 벗어났습니다.")
            return None
        
        # 시작점이나 목표점이 장애물 위에 있는지 확인
        if self.fused_cost_map[start_idx[1], start_idx[0]] == 1 or self.fused_cost_map[goal_idx[1], goal_idx[0]] == 1:
            print("시작점 또는 목표점이 장애물 위에 있습니다.")
            return None
        
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
                print("\n경로 계획 완료:")
                for i, point in enumerate(path):
                    print(f"웨이포인트 {i+1}: ({point[0]:.2f}, {point[1]:.2f})")
                return path
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current, self.fused_cost_map):
                if neighbor in closed_set:
                    continue
                
                # CIGP 비용 함수 (Eq. 3)
                # 장애물인 경우 무한대 비용 부여
                if self.fused_cost_map[neighbor[1], neighbor[0]] == 1:
                    continue
                
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor) + self.fused_cost_map[neighbor[1], neighbor[0]]
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal_idx)
        
        print("목표 지점까지의 경로를 찾을 수 없습니다.")
        return None

    def move_robot(self, target_pos):
        self.target_pos = target_pos
        self.update_fused_cost_map()  # Fused Cost Map 업데이트
        self.current_path = self.a_star_cigp(self.robot_pos, target_pos)
        if not self.current_path:
            print("경로를 찾을 수 없습니다.")
            return False
        return True

    def update_robot(self):
        if self.target_pos is None:
            return

        # 현재 위치가 이전 위치와 거의 같으면 stuck_count 증가
        if np.sqrt((self.robot_pos[0] - self.last_pos[0])**2 + (self.robot_pos[1] - self.last_pos[1])**2) < self.robot_speed * 0.1:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        # Fused Cost Map 업데이트 (매 타임스텝마다)
        self.update_fused_cost_map()

        # 경로 재계산 조건 수정
        need_replan = False
        
        # 1. 경로가 없거나 너무 짧은 경우
        if not self.current_path or len(self.current_path) < 2:
            need_replan = True
        
        # 2. stuck 상태인 경우
        elif self.stuck_count > 10:
            need_replan = True
        
        # 3. 현재 경로의 다음 웨이포인트가 장애물이나 높은 사회적 비용을 가진 영역에 있는 경우
        elif len(self.current_path) > 1:
            next_point = self.current_path[1]
            x_idx = int((next_point[0] + 6) / self.grid_size)
            y_idx = int((next_point[1] + 6) / self.grid_size)
            
            if (0 <= x_idx < 60 and 0 <= y_idx < 60 and 
                (self.fused_cost_map[y_idx, x_idx] > 0.8 or  # 장애물 또는 높은 사회적 비용
                 self.calculate_social_cost(next_point) > 0.8)):  # 높은 사회적 비용
                need_replan = True
        
        # 경로 재계산이 필요한 경우에만 실행
        if need_replan:
            print(f"\n타임스텝 {self.timestep_counter}: CIGP 경로 재계산")
            self.current_path = self.a_star_cigp(self.robot_pos, self.target_pos)
            self.stuck_count = 0
        
        if not self.current_path:
            print("경로를 찾을 수 없습니다.")
            return

        # 다음 웨이포인트로 이동
        next_point = self.current_path[1] if len(self.current_path) > 1 else self.current_path[0]
        dx = next_point[0] - self.robot_pos[0]
        dy = next_point[1] - self.robot_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < self.robot_speed:
            self.robot_pos = [next_point[0], next_point[1]]
            if len(self.current_path) > 1:
                self.current_path.pop(0)
        else:
            self.robot_pos[0] += (dx/dist) * self.robot_speed
            self.robot_pos[1] += (dy/dist) * self.robot_speed
        
        # 로봇 속도 업데이트
        self.robot_vel = [
            self.robot_pos[0] - self.last_pos[0],
            self.robot_pos[1] - self.last_pos[1]
        ]
        
        # 현재 위치 저장
        self.last_pos = self.robot_pos.copy()

    def calculate_individual_space(self, pos, agent):
        return agent.calculate_individual_space(pos)

    def is_in_cctv_coverage(self, pos):
        for cctv in self.cctv_coverage:
            x1, y1 = cctv['points'][0]
            x2, y2 = cctv['points'][1]
            if (min(x1, x2) <= pos[0] <= max(x1, x2) and 
                min(y1, y2) <= pos[1] <= max(y1, y2)):
                return True
        return False
    
    def update(self):
        # Individual Space 맵 초기화
        self.is_map = np.zeros((60, 60))
        
        # 에이전트 업데이트 및 Individual Space 계산
        for agent in self.agents:
            if not agent.finished and self.is_in_cctv_coverage(agent.pos):
                # 에이전트의 속도 계산 (이전 위치와의 차이)
                velocity = [agent.pos[0] - agent.last_pos[0], 
                          agent.pos[1] - agent.last_pos[1]]
                self.calculate_individual_space(agent.pos, agent)
            agent.update(self.agents, self.obstacles)
        
        # 로봇 업데이트
        self.update_robot()
        
        # 타임스텝 카운터 증가
        self.timestep_counter += 1
    
    def visualize(self, path=None):
        # 메인 시뮬레이션 창
        plt.figure(1)
        plt.clf()
        
        # CCTV 커버리지 영역 그리기
        for cctv in self.cctv_coverage:
            x1, y1 = cctv['points'][0]
            x2, y2 = cctv['points'][1]
            plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], color=cctv['color'], alpha=0.3)
            plt.text((x1 + x2)/2, (y1 + y2)/2, cctv['name'], ha='center', va='center')
        
        # 장애물 그리기
        for obs in self.obstacles:
            plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2)
        
        # 에이전트 그리기
        for agent in self.agents:
            color = agent.visualize()
            circle = plt.Circle(agent.pos, agent.radius, color=color, alpha=0.5)
            plt.gca().add_patch(circle)
        
        # 로봇 위치 표시
        plt.plot(self.robot_pos[0], self.robot_pos[1], 'ro', markersize=10)
        
        # 경로 표시
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b--', alpha=0.5)
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title('Robot Simulation')
        
        # Individual Space 창
        plt.figure(2)
        plt.clf()
        
        # Individual Space 맵 표시
        plt.imshow(self.is_map, extent=[-6, 6, -6, 6], origin='lower', 
                  cmap='YlOrRd', alpha=0.7)
        
        # CCTV 커버리지 영역을 검은색 박스로 표시
        for cctv in self.cctv_coverage:
            x1, y1 = cctv['points'][0]
            x2, y2 = cctv['points'][1]
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'k-', linewidth=2)
            plt.text((x1 + x2)/2, (y1 + y2)/2, cctv['name'], ha='center', va='center', 
                    color='black', bbox=dict(facecolor='white', alpha=0.7))
        
        # 장애물 표시
        for obs in self.obstacles:
            plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2)
        
        plt.colorbar(label='Individual Space Value')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title('Individual Space Map')
        
        # Fused Cost Map 창
        plt.figure(3)
        plt.clf()
        
        # Fused Cost Map 표시
        plt.imshow(self.fused_cost_map, extent=[-6, 6, -6, 6], origin='lower', 
                  cmap='YlOrRd', alpha=0.7)
        
        # CCTV 커버리지 영역을 검은색 박스로 표시
        for cctv in self.cctv_coverage:
            x1, y1 = cctv['points'][0]
            x2, y2 = cctv['points'][1]
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'k-', linewidth=2)
            plt.text((x1 + x2)/2, (y1 + y2)/2, cctv['name'], ha='center', va='center', 
                    color='black', bbox=dict(facecolor='white', alpha=0.7))
        
        # 장애물 표시
        for obs in self.obstacles:
            plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2)
        
        plt.colorbar(label='Fused Cost Value')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title('Fused Cost Map')
        
        plt.draw()
        plt.pause(0.01)

def main():
    simulator = RobotSimulator("Circulation1.xml")
    
    while True:
        try:
            # 사용자로부터 목표 위치 입력 받기
            x = float(input("목표 x 좌표를 입력하세요 (-6 ~ 6): "))
            y = float(input("목표 y 좌표를 입력하세요 (-6 ~ 6): "))
            
            # 좌표 범위 체크
            if not (-6 <= x <= 6 and -6 <= y <= 6):
                print("좌표는 -6에서 6 사이여야 합니다.")
                continue
                
            # 로봇 이동
            simulator.move_robot([x, y])
            
            # 시뮬레이션 루프
            while True:
                simulator.update()
                simulator.visualize()
                time.sleep(0.1)
                
                # 목표에 도달했는지 확인
                if np.sqrt((simulator.robot_pos[0] - x)**2 + (simulator.robot_pos[1] - y)**2) < 0.2:
                    print(f"목표 지점 ({x}, {y})에 도달했습니다!")
                    break
                    
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            plt.close('all')  # 모든 창 닫기
            break

if __name__ == "__main__":
    main() 