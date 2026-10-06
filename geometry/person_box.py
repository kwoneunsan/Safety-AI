# geometry/person_box.py
import numpy as np

class PersonBox:
    def __init__(self):
        # 작업자 ID별 이전 발 좌표 저장소 {track_id: (x, y)}
        self.prev_footpoints = {}

    def get_footpoint(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        발 접지점(Footpoint): BBox의 바닥 중앙 좌표 반환
        """
        x1, y1, x2, y2 = bbox
        foot_x = int((x1 + x2) / 2.0)
        foot_y = int(y2)
        return (foot_x, foot_y)

    def calculate_velocity(self, track_id, current_footpoint):
        """
        이전 위치와 현재 위치의 변화량(Δx, Δy)을 통해 이동 벡터 및 속도 계산
        """
        vx, vy = 0.0, 0.0
        if track_id in self.prev_footpoints:
            prev_x, prev_y = self.prev_footpoints[track_id]
            vx = current_footpoint[0] - prev_x
            vy = current_footpoint[1] - prev_y

        self.prev_footpoints[track_id] = current_footpoint
        speed = np.sqrt(vx**2 + vy**2)
        return (vx, vy), speed