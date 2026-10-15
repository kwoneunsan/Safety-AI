# main.py
import cv2
from pipeline.inference_pipeline import SafetyPipeline


def main():
    # 모델 경로 미지정 시 ultralytics 공식 yolov8n.pt 자동 로드
    pipeline = SafetyPipeline(model_path="yolov8n.pt", conf_thresh=0.35, margin_ratio=0.15)

    # 0번 기본 웹캠 오픈
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] 웹캠을 열 수 없습니다. 카메라 연결 상태를 확인하세요.")
        return

    print("\n=======================================================")
    print(" [Safety-AI] 웹캠 테스트 가동 시작!")
    print(" - 화면 우측: 가상 창호(노란색) 및 15% 완충구역(주황색)")
    print(" - 창문 쪽으로 몸을 숙이거나 이동해보세요.")
    print(" - 'q' 키를 누르면 종료됩니다.")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] 프레임을 가져올 수 없습니다.")
            break

        # 창호 좌표를 None으로 넘기면 웹캠 화면 해상도에 맞춰 자동 생성됩니다.
        processed_frame, _ = pipeline.process_frame(frame, window_boxes=None)

        cv2.imshow("Safety-AI Real-time Webcam Demo", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()