# pipeline/inference_pipeline.py
import cv2
from vision.person_detector import PersonDetector
from geometry.person_box import PersonBox
from geometry.risk_zone import RiskZoneManager
from risk.risk_engine import RiskEngine

class SafetyPipeline:
    def __init__(self):
        self.detector = PersonDetector(model_weight="yolov8n.pt")
        self.box_geom = PersonBox()
        self.zone_mgr = RiskZoneManager()
        self.risk_eng = RiskEngine()

    def process_frame(self, frame):
        # 1. 다중 사람 탐지 및 ID 트래킹
        detections = self.detector.detect_and_track(frame)
        
        # 2. 위험구역 배경에 그리기
        frame = self.zone_mgr.draw_zones(frame)

        # 3. 각 검출 객체별 공간 기하 연산 및 위험도 판정
        for det in detections:
            track_id = det["track_id"]
            bbox = det["bbox"]

            # Footpoint 산출
            footpoint = self.box_geom.get_footpoint(bbox)
            
            # 이동 벡터 및 속도 계산 (Temporal)
            velocity, speed = self.box_geom.calculate_velocity(track_id, footpoint)

            # 위험구역 진입 여부 판정
            in_zone = self.zone_mgr.is_inside_zone(footpoint, "ZONE_01")

            # 위험도 산출
            risk_info = self.risk_eng.evaluate(in_risk_zone=in_zone, speed=speed)

            # 시각화 렌더링
            color = risk_info["color"]
            x1, y1, x2, y2 = bbox

            # BBox 및 Footpoint 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, footpoint, 5, color, -1)

            # 라벨 텍스트 표출 (ID, 상태, 속도)
            label = f"ID:{track_id} | {risk_info['label']} | V:{speed:.1f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame