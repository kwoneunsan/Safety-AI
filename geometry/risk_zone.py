# geometry/risk_zone.py
import cv2
import numpy as np

class RiskZoneManager:
    def __init__(self, zones=None):
        # 기본 테스트용 위험구역 다각형 (나중에 YAML 설정값으로 교체 가능)
        if zones is None:
            self.zones = {
                "ZONE_01": np.array([[150, 300], [500, 300], [550, 470], [100, 470]], dtype=np.int32)
            }
        else:
            self.zones = zones

    def is_inside_zone(self, footpoint, zone_name="ZONE_01"):
        """
        cv2.pointPolygonTest:
        양수(>0): 내부, 0: 경계선, 음수(<0): 외부
        """
        if zone_name not in self.zones:
            return False
            
        polygon = self.zones[zone_name]
        result = cv2.pointPolygonTest(polygon, footpoint, measureDist=False)
        return result >= 0

    def draw_zones(self, frame):
        """화면에 위험구역 다각형을 반투명 혹은 테두리로 시각화"""
        for zone_name, polygon in self.zones.items():
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 140, 255), thickness=2)
            # 구역 라벨 표시
            cv2.putText(frame, zone_name, (polygon[0][0], polygon[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        return frame