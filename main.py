# main.py
import cv2
from pipeline.inference_pipeline import SafetyPipeline

def main():
    pipeline = SafetyPipeline()
    
    # 노트북 기본 웹캠: 0 / 동영상 파일 테스트 시: "videos/test_video.mp4"
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라 또는 비디오 소스를 열 수 없습니다.")
        return

    print("=== Safety-AI 파이프라인 가동 시작 ('q' 키를 누르면 종료) ===")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 파이프라인 추론 및 시각화 프레임 획득
        processed_frame = pipeline.process_frame(frame)

        # 화면 출력
        cv2.imshow("Safety-AI Real-time Monitor", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()