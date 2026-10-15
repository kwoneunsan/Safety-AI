# geometry/person_box.py
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional


@dataclass
class PersonBox:
    """
    작업자 Bounding Box 데이터 클래스
    - 신체 중심점(Center of Mass) 및 체형 비율(Aspect Ratio) 기반 동작 추론
    - 안전모 착용 상태(has_helmet) 포함
    """
    track_id: int
    bbox: List[int]                # [x1, y1, x2, y2]
    conf: float = 0.0
    has_helmet: Optional[bool] = None  # True: 착용, False: 미착용, None: 판정 보류
    helmet_conf: float = 0.0

    @property
    def x1(self) -> int:
        return int(self.bbox[0])

    @property
    def y1(self) -> int:
        return int(self.bbox[1])

    @property
    def x2(self) -> int:
        return int(self.bbox[2])

    @property
    def y2(self) -> int:
        return int(self.bbox[3])

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """
        신체 중심점 (Center of Mass)
        하체가 자재나 파라펫에 가려져도 인체 영역 중앙을 안정적으로 가리킴
        """
        cx = int((self.x1 + self.x2) / 2.0)
        cy = int((self.y1 + self.y2) / 2.0)
        return (cx, cy)

    @property
    def aspect_ratio(self) -> float:
        """
        박스 가로/세로 비율 (Width / Height)
        - 직립 상태: 0.3 ~ 0.5
        - 상체를 숙이거나 창밖으로 몸을 내민 상태: 0.65 이상
        """
        if self.height == 0:
            return 0.0
        return round(self.width / float(self.height), 3)

    @property
    def head_zone(self) -> List[int]:
        """
        작업자 머리 추정 영역 (박스 상단 25% 구역)
        - 검출된 안전모/머리 BBox와의 기하학적 매칭에 사용
        """
        head_y2 = self.y1 + int(self.height * 0.25)
        return [self.x1, self.y1, self.x2, head_y2]

    def is_leaning_or_bending(self, threshold: float = 0.65) -> bool:
        """상체를 앞으로 숙이거나 난간 너머로 기울인 위험 자세 판별"""
        return self.aspect_ratio >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "center": self.center,
            "aspect_ratio": self.aspect_ratio,
            "is_leaning": self.is_leaning_or_bending(),
            "has_helmet": self.has_helmet,
            "conf": float(self.conf),
        }