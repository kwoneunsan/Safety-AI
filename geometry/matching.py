# geometry/matching.py
from typing import List, Tuple, Dict
import numpy as np


class WindowPersonMatcher:
    def __init__(self, margin_ratio: float = 0.05, max_margin_px: int = 20):
        """
        :param margin_ratio: 창호 둘레 기본 확장 비율 (기존 0.15 -> 0.05로 축소)
        :param max_margin_px: 원거리/근거리 편차 방지를 위한 최대 확장 픽셀 상한선
        """
        self.margin_ratio = margin_ratio
        self.max_margin_px = max_margin_px
        self.prev_centers: Dict[int, np.ndarray] = {}

    def expand_window_bbox(
        self, 
        window_bbox: List[int], 
        frame_shape: Tuple[int, int],
        directional: bool = True
    ) -> List[int]:
        """
        밀집 창호 간섭을 방지하는 상한선 기반 마진 확장
        - directional=True 적용 시: 상단(천장) 확장은 절반으로 줄이고, 
          작업자 접근 구역인 좌·우·하단(바닥 턱) 위주로 확장
        """
        x1, y1, x2, y2 = window_bbox
        w, h = x2 - x1, y2 - y1

        # 1. 비율 기반 마진 계산 후 최대 픽셀 상한선(Clamp) 적용
        side_margin = min(self.max_margin_px, int(w * self.margin_ratio))
        bottom_margin = min(self.max_margin_px, int(h * self.margin_ratio))
        
        # 2. 상단 마진은 최소화 (추락 위험 진입로가 아니므로 절반만 적용)
        top_margin = int(bottom_margin * 0.5) if directional else bottom_margin

        H, W = frame_shape[:2]
        return [
            max(0, x1 - side_margin),
            max(0, y1 - top_margin),
            min(W, x2 + side_margin),
            min(H, y2 + bottom_margin)
        ]

    def check_dangerous_overlap(
        self, 
        person_bbox: List[int], 
        buffer_bbox: List[int],
        min_overlap_ratio: float = 0.05
    ) -> Tuple[bool, float]:
        """
        단순 1픽셀 접촉으로 인한 오탐을 막기 위해 
        최소 겹침 비율(min_overlap_ratio, 기본 5%) 이상일 때만 침범으로 판정
        """
        px1, py1, px2, py2 = person_bbox
        bx1, by1, bx2, by2 = buffer_bbox

        inter_x1 = max(px1, bx1)
        inter_y1 = max(py1, by1)
        inter_x2 = min(px2, bx2)
        inter_y2 = min(py2, by2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            person_area = max(1, (px2 - px1) * (py2 - py1))
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            overlap_ratio = inter_area / float(person_area)

            # 옷깃이나 그림자가 살짝 닿은 것은 무시하고 실질 침범만 반환
            if overlap_ratio >= min_overlap_ratio:
                return True, round(overlap_ratio, 3)

        return False, 0.0

    def calculate_trajectory(self, track_id: int, curr_center: Tuple[int, int], window_bbox: List[int]) -> Tuple[float, bool]:
        curr_pt = np.array(curr_center, dtype=np.float32)
        win_center = np.array([
            (window_bbox[0] + window_bbox[2]) / 2.0,
            (window_bbox[1] + window_bbox[3]) / 2.0
        ], dtype=np.float32)

        speed = 0.0
        is_approaching = False

        if track_id in self.prev_centers:
            prev_pt = self.prev_centers[track_id]
            speed = float(np.linalg.norm(curr_pt - prev_pt))

            dist_prev = np.linalg.norm(prev_pt - win_center)
            dist_curr = np.linalg.norm(curr_pt - win_center)

            if (dist_prev - dist_curr) > 1.2 and speed > 1.5:
                is_approaching = True

        self.prev_centers[track_id] = curr_pt
        return round(speed, 2), is_approaching

    def clear_lost_tracks(self, active_track_ids: List[int]):
        active_set = set(active_track_ids)
        for tid in list(self.prev_centers.keys()):
            if tid not in active_set:
                del self.prev_centers[tid]