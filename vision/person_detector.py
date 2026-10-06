# vision/person_detector.py
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_weight="yolov8n.pt", conf=0.35):
        # 모델이 로컬에 없으면 Ultralytics가 자동으로 기본 가중치를 다운로드함
        self.model = YOLO(model_weight)
        self.conf = conf

    def detect_and_track(self, frame):
        """
        classes=[0]: COCO 기준 'person'만 탐지
        tracker="bytetrack.yaml": ByteTrack 실시간 ID 추적 활성화
        persist=True: 프레임 간 ID 연속성 유지
        """
        results = self.model.track(
            frame, 
            classes=[0], 
            conf=self.conf, 
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )

        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, score in zip(boxes, track_ids, confs):
                detections.append({
                    "track_id": int(track_id),
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "conf": float(score)
                })

        return detections