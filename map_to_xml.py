import cv2
import numpy as np
import yaml
import xml.etree.ElementTree as ET
from PIL import Image
import math

def world_to_pixel(world_x, world_y, origin_x, origin_y, resolution, img_height):
    '''Converts world coordinates to pixel coordinates.'''
    pixel_x = int((world_x - origin_x) / resolution)
    pixel_y = img_height - 1 - int((world_y - origin_y) / resolution)
    return pixel_x, pixel_y

def pixel_to_world(pixel_x, pixel_y, origin_x, origin_y, resolution, img_height):
    '''Converts pixel coordinates to world coordinates.'''
    world_x = origin_x + pixel_x * resolution
    world_y = origin_y + (img_height - 1 - pixel_y) * resolution
    return world_x, world_y

def create_or_update_xml_map(yaml_path, pgm_path, xml_path, 
                             occupied_thresh_factor=0.65, 
                             free_thresh_factor=0.196,
                             epsilon_factor=0.0015):
    '''
    Reads a PGM map and YAML file, extracts obstacles, and updates/creates an XML scenario file.
    Also adds two test waypoints and an agent moving between them.
    '''
    try:
        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"오류: YAML 파일 '{yaml_path}'을(를) 찾을 수 없습니다.")
        return
    except yaml.YAMLError as e:
        print(f"오류: YAML 파일 '{yaml_path}' 파싱 중 오류: {e}")
        return

    resolution = map_metadata.get('resolution', 0.05)
    origin = map_metadata.get('origin', [0.0, 0.0, 0.0])
    negate = map_metadata.get('negate', 0)
    # occupied_thresh_yaml = map_metadata.get('occupied_thresh', 0.65) # Using factor based on 255 scale
    # free_thresh_yaml = map_metadata.get('free_thresh', 0.196) # Using factor based on 255 scale

    try:
        img_pil = Image.open(pgm_path)
        img_cv = np.array(img_pil)
    except FileNotFoundError:
        print(f"오류: PGM 파일 '{pgm_path}'을(를) 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류: PGM 파일 '{pgm_path}' 로딩 중 오류: {e}")
        return

    if len(img_cv.shape) == 3: # Color image, convert to grayscale
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    img_height, img_width = img_cv.shape

    # Binarize the image based on thresholds
    # PGM values: 0=occupied, 205=unknown, 254=free
    # occupied_thresh_val = int(occupied_thresh_yaml * 255)
    # free_thresh_val = int(free_thresh_yaml * 255)

    # Simplified: directly use common PGM conventions or values from YAML if they are raw pixel values
    # For many PGM maps: < occupied_thresh means occupied (e.g., < 165 if thresh is 0.65*255)
    # This might need adjustment based on the specific PGM encoding
    # Let's assume darker pixels are obstacles.
    # Thresholding logic based on common ROS map_server interpretation:
    # pixels < (free_thresh * 255) are free
    # pixels > (occupied_thresh * 255) are occupied
    # Anything else is unknown
    
    # We want to find occupied regions.
    # If negate is 0, occupied is darker (smaller values).
    # If negate is 1, occupied is lighter (larger values).

    occupied_pixel_value_thresh = int(occupied_thresh_factor * 255) # e.g. 0.65 * 255 = 165.75
    
    if negate == 0: # Occupied is darker
        binary_map = np.where(img_cv < occupied_pixel_value_thresh, 255, 0).astype(np.uint8)
    else: # Occupied is lighter
        binary_map = np.where(img_cv > (255 - occupied_pixel_value_thresh), 255, 0).astype(np.uint8)


    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    new_obstacles = []
    for contour in contours:
        if cv2.contourArea(contour) < 5: # Ignore very small contours (noise)
            continue
        
        # Approximate contour with polygons
        # Epsilon calculation: a percentage of the arc length.
        # Adjust epsilon_factor for more or less simplification.
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

        for i in range(len(approx_polygon)):
            p1_pixel = approx_polygon[i][0]
            p2_pixel = approx_polygon[(i + 1) % len(approx_polygon)][0] # Next point, wrap around

            # Avoid zero-length segments if approximation is too aggressive or for single points
            if np.array_equal(p1_pixel, p2_pixel):
                continue

            # Convert pixel coordinates to world coordinates
            x1_w, y1_w = pixel_to_world(p1_pixel[0], p1_pixel[1], origin[0], origin[1], resolution, img_height)
            x2_w, y2_w = pixel_to_world(p2_pixel[0], p2_pixel[1], origin[0], origin[1], resolution, img_height)
            
            # Round to a reasonable number of decimal places
            x1_w, y1_w = round(x1_w, 3), round(y1_w, 3)
            x2_w, y2_w = round(x2_w, 3), round(y2_w, 3)

            # Avoid duplicate obstacles if they are just reversed (p1->p2 vs p2->p1)
            # This simple check might not catch all cases but helps.
            # A more robust way would be to store canonical representations (e.g., sorted endpoints).
            is_duplicate = False
            for obs_w_coords in new_obstacles:
                if (obs_w_coords == (x1_w, y1_w, x2_w, y2_w)) or \
                   (obs_w_coords == (x2_w, y2_w, x1_w, y1_w)):
                    is_duplicate = True
                    break
            if not is_duplicate:
                 new_obstacles.append((x1_w, y1_w, x2_w, y2_w))


    # Load or create XML tree
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Remove existing obstacles
        for obs_tag in root.findall('obstacle'):
            root.remove(obs_tag)
    except FileNotFoundError:
        print(f"XML 파일 '{xml_path}'을(를) 찾을 수 없어 새로 생성합니다.")
        root = ET.Element("scenario")
        tree = ET.ElementTree(root)
    except ET.ParseError:
        print(f"오류: XML 파일 '{xml_path}' 파싱 중 오류. 기본 구조로 새로 생성합니다.")
        root = ET.Element("scenario")
        tree = ET.ElementTree(root)

    # Add new obstacles
    for x1, y1, x2, y2 in new_obstacles:
        ET.SubElement(root, "obstacle", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2))

    # Add/Update test waypoints and agent
    # Define waypoint IDs and positions (example positions, adjust as needed)
    # Try to find a somewhat central, clear area based on map dimensions for waypoints
    map_center_x_world = origin[0] + (img_width / 2.0) * resolution
    map_center_y_world = origin[1] + (img_height / 2.0) * resolution

    wp_start_id = "test_wp_start_auto"
    wp_end_id = "test_wp_end_auto"
    
    # Example: Place waypoints offset from the map center
    # These coordinates might need adjustment if they fall inside new obstacles
    wp_start_x = round(map_center_x_world - max(1.0, img_width * resolution * 0.1), 2) 
    wp_start_y = round(map_center_y_world, 2)
    wp_end_x = round(map_center_x_world + max(1.0, img_width * resolution * 0.1), 2)
    wp_end_y = round(map_center_y_world, 2)
    wp_radius = "0.5"

    # Remove existing test waypoints/agent if they exist to avoid duplicates
    for wp_tag in root.findall(f".//waypoint[@id='{wp_start_id}']"):
        root.remove(wp_tag)
    for wp_tag in root.findall(f".//waypoint[@id='{wp_end_id}']"):
        root.remove(wp_tag)
    # A bit simplistic for removing agents, assumes agent has these specific waypoints
    # or you can give the agent a specific ID to find and remove.
    # For now, we just add a new one. If multiple runs, multiple test agents will be added.
    # Consider adding a unique ID to the agent for easier removal/update later.

    ET.SubElement(root, "waypoint", id=wp_start_id, x=str(wp_start_x), y=str(wp_start_y), r=wp_radius)
    ET.SubElement(root, "waypoint", id=wp_end_id, x=str(wp_end_x), y=str(wp_end_y), r=wp_radius)

    agent_tag = ET.SubElement(root, "agent", x=str(wp_start_x), y=str(wp_start_y), n="1", dx="0", dy="0", type="0")
    ET.SubElement(agent_tag, "addwaypoint", id=wp_start_id)
    ET.SubElement(agent_tag, "addwaypoint", id=wp_end_id)
    
    # Nicely format XML output
    ET.indent(tree, space="    ")
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    print(f"XML 파일 '{xml_path}'이(가) 업데이트되었습니다. 장애물 {len(new_obstacles)}개 추가, 테스트 웨이포인트 및 에이전트 추가됨.")

if __name__ == '__main__':
    yaml_file = 'HNmap.yaml'
    pgm_file = 'HNmap.pgm'
    xml_target_file = 'HNmap_people.xml' # Will be overwritten

    # These factors might need tuning based on your specific map's characteristics
    # occupied_thresh_factor: For PGM, lower values usually mean occupied (if not negated).
    #                        If PGM has 0 as black (occupied) and 255 as white (free), then
    #                        a pixel is occupied if its value < occupied_thresh_factor * 255.
    #                        Default in ROS map_server for occupied_thresh is 0.65.
    #                        So, pixels with value > 0.65*255 are considered occupied if negate=false
    #                        Wait, map_server docs say:
    #                        occupied_thresh: Pixels with cost p >= occupied_thresh are considered fully occupied.
    #                        free_thresh: Pixels with cost p <= free_thresh are considered fully free.
    #                        If PGM values are 0-255, cost is (255 - PGM_VALUE) / 255.0 for negate=false
    #                        So PGM_VALUE < (1 - occupied_thresh) * 255 means occupied.
    #                        Let's stick to the direct interpretation: for negate=0, smaller PGM value = occupied.
    #                        So if pgm_value < occupied_thresh_factor * 255, it's occupied.
    # epsilon_factor: Smaller means more detailed polygons (more obstacle segments).
    #                 Larger means more simplified polygons (fewer segments).
    create_or_update_xml_map(yaml_file, pgm_file, xml_target_file, 
                             occupied_thresh_factor=0.5, # PGM values < 0.5*255 are obstacles (if not negated)
                             epsilon_factor=0.002)     # Contour approximation precision
    
    print(f"'{xml_target_file}' 생성/업데이트 완료.")
    print(f"이제 'python3 visual.py {xml_target_file}' 로 결과를 시각화할 수 있습니다.") 