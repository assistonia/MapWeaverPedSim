import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys
import numpy as np

def draw_scenario_from_xml(xml_file_path):
    """
    XML 파일에서 장애물, 웨이포인트, 에이전트 경로 정보를 읽어 Matplotlib으로 시각화합니다.

    Args:
        xml_file_path (str): 시나리오 정보가 포함된 XML 파일 경로.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"오류: 파일 '{xml_file_path}'을(를) 찾을 수 없습니다.")
        return
    except ET.ParseError:
        print(f"오류: 파일 '{xml_file_path}'을(를) 파싱하는 중 오류가 발생했습니다.")
        return

    obstacles = []
    waypoints = {}
    agents_paths = []

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # Parse obstacles
    for obs_element in root.findall('obstacle'):
        try:
            x1 = float(obs_element.get('x1'))
            y1 = float(obs_element.get('y1'))
            x2 = float(obs_element.get('x2'))
            y2 = float(obs_element.get('y2'))
            obstacles.append(((x1, y1), (x2, y2)))
            min_x, max_x = min(min_x, x1, x2), max(max_x, x1, x2)
            min_y, max_y = min(min_y, y1, y2), max(max_y, y1, y2)
        except (TypeError, ValueError) as e:
            print(f"오류 (장애물): {ET.tostring(obs_element, encoding='unicode').strip()} - {e}")
            continue

    # Parse waypoints
    for wp_element in root.findall('waypoint'):
        try:
            wp_id = wp_element.get('id')
            x = float(wp_element.get('x'))
            y = float(wp_element.get('y'))
            r = float(wp_element.get('r', 0.1)) # Default radius if not specified
            waypoints[wp_id] = {'x': x, 'y': y, 'r': r}
            min_x, max_x = min(min_x, x - r), max(max_x, x + r)
            min_y, max_y = min(min_y, y - r), max(max_y, y + r)
        except (TypeError, ValueError) as e:
            print(f"오류 (웨이포인트): {ET.tostring(wp_element, encoding='unicode').strip()} - {e}")
            continue
    
    # Parse agents and their paths
    for agent_element in root.findall('agent'):
        try:
            agent_x_start = float(agent_element.get('x'))
            agent_y_start = float(agent_element.get('y'))
            num_agents = int(agent_element.get('n', 1))
            # For simplicity, we'll just draw the path for the first agent of a cluster if n > 1
            # and assume its starting point is (agent_x_start, agent_y_start)

            path_points = [(agent_x_start, agent_y_start)]
            min_x, max_x = min(min_x, agent_x_start), max(max_x, agent_x_start)
            min_y, max_y = min(min_y, agent_y_start), max(max_y, agent_y_start)

            for addwp_element in agent_element.findall('addwaypoint'):
                wp_id = addwp_element.get('id')
                if wp_id in waypoints:
                    wp = waypoints[wp_id]
                    path_points.append((wp['x'], wp['y']))
                else:
                    print(f"경고: 에이전트가 참조하는 웨이포인트 '{wp_id}'를 찾을 수 없습니다.")
            
            if len(path_points) > 1:
                agents_paths.append(path_points)

        except (TypeError, ValueError) as e:
            print(f"오류 (에이전트): {ET.tostring(agent_element, encoding='unicode').strip()} - {e}")
            continue

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw obstacles
    for obs in obstacles:
        point1, point2 = obs
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k-', linewidth=2)

    # Draw waypoints
    for wp_id, wp_data in waypoints.items():
        circle = plt.Circle((wp_data['x'], wp_data['y']), wp_data['r'], color='blue', alpha=0.5, label="Waypoint" if wp_id == list(waypoints.keys())[0] else "")
        ax.add_artist(circle)
        ax.text(wp_data['x'], wp_data['y'], wp_id, fontsize=8, ha='center', va='bottom')

    # Draw agent paths
    # Use different colors for different agent paths for better visibility
    colors = plt.cm.viridis(np.linspace(0, 1, len(agents_paths))) if agents_paths else []
    for i, path in enumerate(agents_paths):
        path_x, path_y = zip(*path)
        # Plot path line
        ax.plot(path_x, path_y, marker='o', linestyle='--', markersize=5, color=colors[i] if agents_paths else 'red', label=f"Agent Path {i+1}" if i == 0 else "")
        # Mark start and end of path
        ax.plot(path_x[0], path_y[0], 'X', markersize=10, color=colors[i] if agents_paths else 'red') # Start
        if len(path_x) >1 : ax.plot(path_x[-1], path_y[-1], 'P', markersize=10, color=colors[i] if agents_paths else 'red') # End (Goal)


    ax.set_xlabel("X 좌표")
    ax.set_ylabel("Y 좌표")
    ax.set_title(f"'{xml_file_path}' 시나리오 시각화")
    ax.set_aspect('equal', adjustable='box')
    
    # Add a small margin to the plot limits if any elements were found
    if obstacles or waypoints or agents_paths:
        margin = 1.5
        ax.set_xlim(min_x - margin, max_x + margin)
        ax.set_ylim(min_y - margin, max_y + margin)
    else:
        print("경고: 시각화할 요소(장애물, 웨이포인트, 에이전트 경로)가 XML 파일에 없습니다.")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    # Add legend if there are paths or waypoints
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Filter out duplicate labels for legend
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
        draw_scenario_from_xml(xml_file)
    else:
        print("사용법: python visual.py <xml_file_path>")
        # 기본 파일로 실행 (예시)
        print("기본 예시 파일 'Congestion1.xml'로 실행합니다.")
        default_xml_file = 'Congestion1.xml' # 또는 HNmap_people.xml 등 테스트하고 싶은 파일
        draw_scenario_from_xml(default_xml_file)