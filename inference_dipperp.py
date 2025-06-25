import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import argparse
import sys
from robot_simuator_dippeR import RobotSimulatorDiPPeR, DiPPeR

class DiPPeRInference:
    """학습된 DiPPeR 모델로 추론만 수행"""
    def __init__(self, xml_file, model_path=None):
        self.simulator = RobotSimulatorDiPPeR(xml_file, model_path)
        print(f"추론 시뮬레이터 초기화 완료: {xml_file}")
        
        if model_path:
            print(f"학습된 모델 사용: {model_path}")
        else:
            print("학습되지 않은 모델 사용 (A* 폴백 모드)")
    
    def run_interactive(self):
        """대화형 모드로 실행"""
        print("\n=== DiPPeR 경로 계획 시뮬레이션 ===")
        print("목표 좌표를 입력하여 로봇을 이동시키세요.")
        print("Ctrl+C로 종료합니다.\n")
        
        while True:
            try:
                x = float(input("목표 x 좌표를 입력하세요 (-6 ~ 6): "))
                y = float(input("목표 y 좌표를 입력하세요 (-6 ~ 6): "))
                
                if not (-6 <= x <= 6 and -6 <= y <= 6):
                    print("좌표는 -6에서 6 사이여야 합니다.")
                    continue
                
                success = self.run_navigation(x, y)
                if success:
                    print(f"✅ 목표 지점 ({x}, {y})에 성공적으로 도달했습니다!\n")
                else:
                    print(f"❌ 목표 지점 ({x}, {y})에 도달하지 못했습니다.\n")
                    
            except ValueError:
                print("올바른 숫자를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                plt.close('all')
                break
    
    def run_navigation(self, x, y, max_steps=1000):
        """특정 목표점으로 네비게이션 실행"""
        # 목표점이 안전한지 체크
        if not self.simulator.is_position_safe([x, y]):
            print(f"목표점 ({x}, {y})이 안전하지 않습니다 (장애물 또는 맵 경계 외부)")
            return False
        
        # 로봇 이동 시작
        self.simulator.move_robot([x, y])
        
        start_time = time.time()
        step_count = 0
        
        while step_count < max_steps:
            # 시뮬레이션 업데이트
            self.simulator.update()
            self.simulator.visualize()
            time.sleep(0.05)
            
            step_count += 1
            
            # 목표 도달 체크
            current_pos = self.simulator.robot_pos
            dist_to_goal = np.sqrt((current_pos[0] - x)**2 + (current_pos[1] - y)**2)
            
            if dist_to_goal < 0.2:
                elapsed_time = time.time() - start_time
                print(f"🎯 목표 도달! 소요 시간: {elapsed_time:.1f}초, 스텝: {step_count}")
                return True
            
            # 진행 상황 출력 (100스텝마다)
            if step_count % 100 == 0:
                print(f"스텝 {step_count}: 현재 위치 ({current_pos[0]:.2f}, {current_pos[1]:.2f}), 거리: {dist_to_goal:.2f}m")
        
        print(f"❌ 최대 스텝 ({max_steps}) 도달. 목표에 도달하지 못했습니다.")
        return False
    
    def run_benchmark(self, num_tests=10):
        """성능 벤치마크 실행"""
        print(f"\n=== DiPPeR 성능 벤치마크 ({num_tests}회 테스트) ===")
        
        success_count = 0
        total_time = 0
        total_steps = 0
        
        for test_num in range(1, num_tests + 1):
            print(f"\n테스트 {test_num}/{num_tests}")
            
            # 랜덤 목표점 생성
            while True:
                x = np.random.uniform(-5.0, 5.0)
                y = np.random.uniform(-5.0, 5.0)
                if self.simulator.is_position_safe([x, y]):
                    break
            
            print(f"목표: ({x:.2f}, {y:.2f})")
            
            # 네비게이션 실행 (시각화 없이)
            start_time = time.time()
            self.simulator.move_robot([x, y])
            
            step_count = 0
            max_steps = 500
            success = False
            
            while step_count < max_steps:
                self.simulator.update()
                step_count += 1
                
                # 목표 도달 체크
                current_pos = self.simulator.robot_pos
                dist_to_goal = np.sqrt((current_pos[0] - x)**2 + (current_pos[1] - y)**2)
                
                if dist_to_goal < 0.2:
                    success = True
                    break
            
            elapsed_time = time.time() - start_time
            
            if success:
                success_count += 1
                total_time += elapsed_time
                total_steps += step_count
                print(f"✅ 성공! 시간: {elapsed_time:.1f}초, 스텝: {step_count}")
            else:
                print(f"❌ 실패 (시간 초과)")
        
        # 결과 출력
        success_rate = success_count / num_tests * 100
        avg_time = total_time / success_count if success_count > 0 else 0
        avg_steps = total_steps / success_count if success_count > 0 else 0
        
        print(f"\n=== 벤치마크 결과 ===")
        print(f"성공률: {success_rate:.1f}% ({success_count}/{num_tests})")
        if success_count > 0:
            print(f"평균 시간: {avg_time:.1f}초")
            print(f"평균 스텝: {avg_steps:.1f}")

def main():
    parser = argparse.ArgumentParser(description='DiPPeR 추론 실행')
    parser.add_argument('--xml_file', default='Congestion1.xml', help='시뮬레이션 XML 파일')
    parser.add_argument('--model_path', help='학습된 DiPPeR 모델 경로')
    parser.add_argument('--mode', choices=['interactive', 'benchmark'], default='interactive', 
                        help='실행 모드: interactive (대화형) 또는 benchmark (성능 테스트)')
    parser.add_argument('--num_tests', type=int, default=10, help='벤치마크 테스트 횟수')
    parser.add_argument('--target_x', type=float, help='단일 목표 x 좌표')
    parser.add_argument('--target_y', type=float, help='단일 목표 y 좌표')
    
    args = parser.parse_args()
    
    # 추론 시뮬레이터 생성
    inference = DiPPeRInference(args.xml_file, args.model_path)
    
    if args.target_x is not None and args.target_y is not None:
        # 단일 목표점 테스트
        print(f"단일 목표점 테스트: ({args.target_x}, {args.target_y})")
        success = inference.run_navigation(args.target_x, args.target_y)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'interactive':
        # 대화형 모드
        inference.run_interactive()
    
    elif args.mode == 'benchmark':
        # 벤치마크 모드
        inference.run_benchmark(args.num_tests)
    
    plt.close('all')

if __name__ == "__main__":
    main() 