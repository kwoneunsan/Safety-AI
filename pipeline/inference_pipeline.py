# pipeline/inference_pipeline.py
import cv2
from typing import List, Optional
from vision.person_detector import PPEPersonDetector
from geometry.matching import WindowPersonMatcher


class SafetyPipeline:
    def __init__(
        self, 
        model_path: str = "yolov8n.pt", 
        conf_thresh: float = 0.35, 
        margin_ratio: float = 0.15
    ):
        print(f"[SafetyPipeline] 초기화 중... (Model: {model_path})")
        self.detector = PPEPersonDetector(model_path=model_path, conf_thresh=conf_thresh)
        # 기존 15% -> 5% 축소, 최대 확장 폭 15~20px로 제한
        self.matcher = WindowPersonMatcher(margin_ratio=0.05, max_margin_px=15)

    def process_frame(self, frame, window_boxes: Optional[List[List[int]]] = None):
        H, W = frame.shape[:2]

        # 개구부 좌표가 지정되지 않은 경우, 화면 우측에 가상 창호 자동 생성
        if not window_boxes:
            window_boxes = [[int(W * 0.55), int(H * 0.2), int(W * 0.95), int(H * 0.85)]]

        # 1. 작업자 탐지 및 ByteTrack 추적
        persons = self.detector.detect_and_track(frame)
        active_ids = [p.track_id for p in persons if p.track_id != -1]
        self.matcher.clear_lost_tracks(active_ids)

        # 2. 개구부별 위험 버퍼 생성 및 기하 연산
        for w_box in window_boxes:
            buffer_box = self.matcher.expand_window_bbox(w_box, (H, W))

            # 창호 원본(노란색) 및 15% 위험 완충 구역(주황색) 표시
            cv2.rectangle(frame, (w_box[0], w_box[1]), (w_box[2], w_box[3]), (0, 255, 255), 2)
            cv2.rectangle(frame, (buffer_box[0], buffer_box[1]), (buffer_box[2], buffer_box[3]), (0, 140, 255), 1)
            cv2.putText(frame, "Virtual Window", (w_box[0], max(20, w_box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Buffer 15%", (buffer_box[0], max(15, buffer_box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

            # 3. 작업자별 위험도 판정
            for p in persons:
                is_overlapped, overlap_ratio = self.matcher.check_dangerous_overlap(p.bbox, buffer_box)
                speed, is_approaching = self.matcher.calculate_trajectory(p.track_id, p.center, w_box)
                is_leaning = p.is_leaning_or_bending()

                # 위험 상태 결정
                risk_status = "SAFE"
                color = (0, 255, 0)  # 초록색

                if is_overlapped:
                    # 버퍼 안에서 창문 쪽으로 다가가거나 상체를 숙인 경우 즉시 위험
                    if is_approaching or is_leaning or (p.has_helmet is False):
                        risk_status = "DANGER"
                        color = (0, 0, 255)  # 빨간색
                    else:
                        risk_status = "WARNING"
                        color = (0, 165, 255)  # 주황색
                elif is_approaching:
                    risk_status = "CAUTION"
                    color = (0, 255, 255)  # 노란색

                # 시각화 박스 및 신체 중심점
                x1, y1, x2, y2 = p.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, p.center, 5, color, -1)

                # 안전모 텍스트 처리
                if p.has_helmet is True:
                    helmet_str = "Helmet: OK"
                elif p.has_helmet is False:
                    helmet_str = "NO Helmet!"
                else:
                    helmet_str = "PPE: Base-Mode"

                label = f"ID:{p.track_id} | {risk_status} | {helmet_str} | Spd:{speed:.1f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, persons