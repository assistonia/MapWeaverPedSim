import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import torch
from matplotlib.patches import Rectangle
import random

def load_dataset(data_path):
    """저장된 데이터셋 로드"""
    print(f"데이터셋 로드: {data_path}")
    with open(data_path, 'r') as f:
        data_list = json.load(f)
    
    # JSON에서 numpy 배열로 변환
    for item in data_list:
        item['cost_map'] = np.array(item['cost_map'])
        item['start_pos'] = np.array(item['start_pos'])
        item['goal_pos'] = np.array(item['goal_pos'])
        item['path'] = np.array(item['path'])
    
    print(f"총 데이터 개수: {len(data_list)}")
    return data_list

def visualize_sample(data_item, sample_idx):
    """단일 샘플 시각화"""
    cost_map = data_item['cost_map']
    start_pos = data_item['start_pos'] * 6.0  # 역정규화
    goal_pos = data_item['goal_pos'] * 6.0    # 역정규화
    path = data_item['path'] * 6.0            # 역정규화
    path_type = data_item.get('path_type', 'unknown')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Cost Map 시각화
    ax1.imshow(cost_map, cmap='viridis', origin='lower', extent=[-6, 6, -6, 6])
    ax1.set_title(f'Sample {sample_idx}: Cost Map')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)
    
    # 시작점과 목표점 표시
    ax1.plot(start_pos[0], start_pos[1], 'ro', markersize=10, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'go', markersize=10, label='Goal')
    ax1.legend()
    
    # 2. 경로 시각화
    ax2.imshow(cost_map, cmap='gray', alpha=0.5, origin='lower', extent=[-6, 6, -6, 6])
    ax2.set_title(f'Sample {sample_idx}: Path ({path_type})')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    
    # 경로 그리기
    ax2.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label=f'Path ({path_type})')
    ax2.plot(start_pos[0], start_pos[1], 'ro', markersize=10, label='Start')
    ax2.plot(goal_pos[0], goal_pos[1], 'go', markersize=10, label='Goal')
    
    # 웨이포인트 번호 표시 (일부만)
    for i in range(0, len(path), 10):
        ax2.text(path[i, 0], path[i, 1], str(i), fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax2.legend()
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    
    plt.tight_layout()
    return fig

def analyze_dataset(data_list):
    """데이터셋 통계 분석"""
    print("\n" + "="*50)
    print("📊 데이터셋 분석 결과")
    print("="*50)
    
    # 경로 타입별 통계
    path_types = {}
    path_lengths = []
    
    for item in data_list:
        path_type = item.get('path_type', 'unknown')
        path_types[path_type] = path_types.get(path_type, 0) + 1
        path_lengths.append(len(item['path']))
    
    print(f"총 데이터 개수: {len(data_list)}")
    print(f"평균 경로 길이: {np.mean(path_lengths):.1f} 웨이포인트")
    print(f"경로 길이 범위: {min(path_lengths)} ~ {max(path_lengths)}")
    
    print("\n경로 타입별 분포:")
    for path_type, count in path_types.items():
        percentage = (count / len(data_list)) * 100
        print(f"  {path_type}: {count}개 ({percentage:.1f}%)")
    
    # 시작점/목표점 분포 분석
    start_positions = np.array([item['start_pos'] * 6.0 for item in data_list])
    goal_positions = np.array([item['goal_pos'] * 6.0 for item in data_list])
    
    print(f"\n시작점 분포 (X): {start_positions[:, 0].min():.1f} ~ {start_positions[:, 0].max():.1f}")
    print(f"시작점 분포 (Y): {start_positions[:, 1].min():.1f} ~ {start_positions[:, 1].max():.1f}")
    print(f"목표점 분포 (X): {goal_positions[:, 0].min():.1f} ~ {goal_positions[:, 0].max():.1f}")
    print(f"목표점 분포 (Y): {goal_positions[:, 1].min():.1f} ~ {goal_positions[:, 1].max():.1f}")
    
    return path_types, path_lengths

def visualize_distribution(data_list):
    """데이터 분포 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 경로 타입별 분포
    path_types = {}
    for item in data_list:
        path_type = item.get('path_type', 'unknown')
        path_types[path_type] = path_types.get(path_type, 0) + 1
    
    axes[0, 0].bar(path_types.keys(), path_types.values())
    axes[0, 0].set_title('경로 타입별 분포')
    axes[0, 0].set_ylabel('개수')
    
    # 경로 길이 분포
    path_lengths = [len(item['path']) for item in data_list]
    axes[0, 1].hist(path_lengths, bins=20, alpha=0.7)
    axes[0, 1].set_title('경로 길이 분포')
    axes[0, 1].set_xlabel('웨이포인트 개수')
    axes[0, 1].set_ylabel('빈도')
    
    # Cost Map 평균
    cost_maps = np.array([item['cost_map'] for item in data_list])
    avg_cost_map = np.mean(cost_maps, axis=0)
    im = axes[0, 2].imshow(avg_cost_map, cmap='viridis', origin='lower', extent=[-6, 6, -6, 6])
    axes[0, 2].set_title('평균 Cost Map')
    axes[0, 2].set_xlabel('X (m)')
    axes[0, 2].set_ylabel('Y (m)')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 시작점 분포
    start_positions = np.array([item['start_pos'] * 6.0 for item in data_list])
    axes[1, 0].scatter(start_positions[:, 0], start_positions[:, 1], alpha=0.5, s=10)
    axes[1, 0].set_title('시작점 분포')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-6, 6)
    axes[1, 0].set_ylim(-6, 6)
    
    # 목표점 분포
    goal_positions = np.array([item['goal_pos'] * 6.0 for item in data_list])
    axes[1, 1].scatter(goal_positions[:, 0], goal_positions[:, 1], alpha=0.5, s=10, color='green')
    axes[1, 1].set_title('목표점 분포')
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-6, 6)
    axes[1, 1].set_ylim(-6, 6)
    
    # 경로 타입별 예시 (중첩)
    axes[1, 2].imshow(avg_cost_map, cmap='gray', alpha=0.3, origin='lower', extent=[-6, 6, -6, 6])
    
    colors = {'astar': 'blue', 'social': 'orange', 'detour': 'red', 'unknown': 'purple'}
    for path_type, color in colors.items():
        type_samples = [item for item in data_list if item.get('path_type') == path_type]
        if type_samples:
            sample = random.choice(type_samples)
            path = sample['path'] * 6.0
            axes[1, 2].plot(path[:, 0], path[:, 1], color=color, alpha=0.7, 
                          linewidth=1, label=f'{path_type} 예시')
    
    axes[1, 2].set_title('경로 타입별 예시')
    axes[1, 2].set_xlabel('X (m)')
    axes[1, 2].set_ylabel('Y (m)')
    axes[1, 2].legend()
    axes[1, 2].set_xlim(-6, 6)
    axes[1, 2].set_ylim(-6, 6)
    
    plt.tight_layout()
    return fig

def interactive_viewer(data_list):
    """대화형 데이터 뷰어"""
    print("\n" + "="*50)
    print("🔍 대화형 데이터 뷰어")
    print("="*50)
    print("명령어:")
    print("  숫자: 해당 인덱스 샘플 보기")
    print("  'random': 랜덤 샘플 보기")
    print("  'type [astar/social/detour]': 특정 타입 샘플 보기")
    print("  'stats': 통계 정보 보기")
    print("  'dist': 분포 시각화")
    print("  'quit': 종료")
    print("-"*50)
    
    while True:
        try:
            command = input("\n명령어 입력: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'random':
                idx = random.randint(0, len(data_list) - 1)
                fig = visualize_sample(data_list[idx], idx)
                plt.show()
            elif command.startswith('type '):
                path_type = command.split(' ')[1]
                type_samples = [(i, item) for i, item in enumerate(data_list) 
                              if item.get('path_type') == path_type]
                if type_samples:
                    idx, sample = random.choice(type_samples)
                    fig = visualize_sample(sample, idx)
                    plt.show()
                else:
                    print(f"'{path_type}' 타입의 샘플이 없습니다.")
            elif command == 'stats':
                analyze_dataset(data_list)
            elif command == 'dist':
                fig = visualize_distribution(data_list)
                plt.show()
            elif command.isdigit():
                idx = int(command)
                if 0 <= idx < len(data_list):
                    fig = visualize_sample(data_list[idx], idx)
                    plt.show()
                else:
                    print(f"인덱스 범위 오류. 0 ~ {len(data_list)-1} 사이의 값을 입력하세요.")
            else:
                print("알 수 없는 명령어입니다.")
                
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description='DiPPeR 데이터셋 시각화 도구')
    parser.add_argument('--data_path', default='training_data.json', help='데이터셋 파일 경로')
    parser.add_argument('--mode', choices=['interactive', 'analyze', 'sample'], 
                       default='interactive', help='실행 모드')
    parser.add_argument('--sample_idx', type=int, default=0, help='샘플 모드에서 볼 인덱스')
    parser.add_argument('--num_samples', type=int, default=5, help='분석 모드에서 보여줄 샘플 수')
    
    args = parser.parse_args()
    
    try:
        # 데이터셋 로드
        data_list = load_dataset(args.data_path)
        
        if not data_list:
            print("❌ 데이터셋이 비어있습니다.")
            return
        
        # 통계 분석
        analyze_dataset(data_list)
        
        if args.mode == 'interactive':
            interactive_viewer(data_list)
        elif args.mode == 'analyze':
            print(f"\n📊 {args.num_samples}개 샘플 시각화...")
            for i in range(min(args.num_samples, len(data_list))):
                idx = random.randint(0, len(data_list) - 1)
                fig = visualize_sample(data_list[idx], idx)
                plt.show()
            
            # 분포 시각화
            fig = visualize_distribution(data_list)
            plt.show()
            
        elif args.mode == 'sample':
            if 0 <= args.sample_idx < len(data_list):
                fig = visualize_sample(data_list[args.sample_idx], args.sample_idx)
                plt.show()
            else:
                print(f"❌ 인덱스 오류: 0 ~ {len(data_list)-1} 범위의 값을 입력하세요.")
    
    except FileNotFoundError:
        print(f"❌ 데이터셋 파일을 찾을 수 없습니다: {args.data_path}")
        print("먼저 train_dipperp.py로 데이터를 수집하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main() 