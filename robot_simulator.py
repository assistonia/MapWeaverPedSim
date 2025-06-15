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
            
            # 현재 위치 저장
            self.last_pos = self.pos.copy()

    def visualize(self):
        if self.finished:
            return 'green'  # 완료된 에이전트는 초록색으로 표시
        return 'blue'  # 진행 중인 에이전트는 파란색으로 표시

class RobotSimulator:
    def __init__(self, xml_file):
        self.obstacles = []
        self.grid_size = 0.1
        self.load_obstacles(xml_file)
        self.robot_pos = [3, -4]
        self.target_pos = None
        self.current_path = []
        self.robot_speed = 0.1
        self.create_grid()
        self.agents = []
        self.load_agents(xml_file)
        self.stuck_count = 0
        self.last_pos = [3, -4]
        
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
        self.grid = np.zeros((120, 120))
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
            if 0 <= new_x < 120 and 0 <= new_y < 120 and dynamic_grid[new_y, new_x] == 0:
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

    def move_robot(self, target_pos):
        self.target_pos = target_pos
        self.current_path = self.a_star(self.robot_pos, target_pos)
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

        # 경로 재계산 조건 추가
        if not self.current_path or len(self.current_path) < 2 or self.stuck_count > 10:
            self.current_path = self.a_star(self.robot_pos, self.target_pos)
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
        
        # 현재 위치 저장
        self.last_pos = self.robot_pos.copy()

    def update(self):
        # 에이전트 업데이트
        for agent in self.agents:
            agent.update(self.agents, self.obstacles)
        
        # 로봇 업데이트
        self.update_robot()

    def visualize(self, path=None):
        plt.clf()
        # 장애물 그리기
        for obs in self.obstacles:
            plt.plot([obs[0][0], obs[1][0]], [obs[0][1], obs[1][1]], 'k-', linewidth=2)
        
        # 에이전트 그리기
        for agent in self.agents:
            color = agent.visualize()  # 에이전트 상태에 따른 색상
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
        plt.draw()
        plt.pause(0.01)

def main():
    simulator = RobotSimulator("Circulation1.xml")
    
    # 목표 지점 설정 (예: (4, 4))
    simulator.move_robot([4, 4])
    
    # 시뮬레이션 루프
    while True:
        simulator.update()
        simulator.visualize()
        time.sleep(0.1)

if __name__ == "__main__":
    main() 