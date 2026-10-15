# vision/person_detector.py
from typing import List, Dict, Any
import torch
from ultralytics import YOLO
from geometry.person_box import PersonBox


class PPEPersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.35):
        """
        model_path: 
          - 기본 실행: 'yolov8n.pt' (파일이 없으면 ultralytics가 자동 다운로드)
          - 추후 학습 완료 시: 'best.pt' 또는 'best.engine'으로 교체
        """
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.conf = conf_thresh
        print(f"[PPEPersonDetector] 모델 로드 중: {model_path} (Device: {self.device})")
        self.model = YOLO(model_path)

        # 모델 클래스 인덱스 자동 매핑
        self.class_map = self._resolve_class_indices()
        self.has_ppe_classes = len(self.class_map["helmet"]) > 0 or len(self.class_map["no_helmet"]) > 0

    def _resolve_class_indices(self) -> Dict[str, List[int]]:
        names = self.model.names
        mapping = {"person": [], "helmet": [], "no_helmet": []}

        for idx, name in names.items():
            name_lower = name.lower()
            if "person" in name_lower or "worker" in name_lower:
                mapping["person"].append(idx)
            elif any(k in name_lower for k in ["hard-hat", "helmet", "hardhat"]):
                mapping["helmet"].append(idx)
            elif any(k in name_lower for k in ["no-helmet", "no-hardhat", "head"]):
                mapping["no_helmet"].append(idx)

        # 기본 yolov8n.pt (COCO 데이터셋: class 0번이 person)
        if not mapping["person"]:
            mapping["person"] = [0]

        return mapping

    def detect_and_track(self, frame) -> List[PersonBox]:
        results = self.model.track(
            source=frame,
            conf=self.conf,
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        if not results or results[0].boxes is None:
            return []

        boxes_data = results[0].boxes
        xyxy = boxes_data.xyxy.cpu().numpy()
        confs = boxes_data.conf.cpu().numpy()
        clss = boxes_data.cls.cpu().numpy().astype(int)
        track_ids = boxes_data.id.int().cpu().numpy() if boxes_data.id is not None else None

        persons: List[PersonBox] = []
        helmets: List[Dict[str, Any]] = []

        for i, (box, score, cls_id) in enumerate(zip(xyxy, confs, clss)):
            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

            if cls_id in self.class_map["person"]:
                tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else -1
                persons.append(PersonBox(track_id=tid, bbox=bbox, conf=float(score)))
            elif cls_id in self.class_map["helmet"]:
                helmets.append({"bbox": bbox, "type": "HELMET", "conf": float(score)})
            elif cls_id in self.class_map["no_helmet"]:
                helmets.append({"bbox": bbox, "type": "NO_HELMET", "conf": float(score)})

        # 안전모 클래스가 모델에 존재하는 경우에만 매칭 수행
        for person in persons:
            if not self.has_ppe_classes:
                person.has_helmet = None  # 판정 보류 (기본 모델 모드)
                continue

            hx1, hy1, hx2, hy2 = person.head_zone
            matched = False

            for h in helmets:
                h_bbox = h["bbox"]
                hcx = (h_bbox[0] + h_bbox[2]) / 2.0
                hcy = (h_bbox[1] + h_bbox[3]) / 2.0

                if hx1 <= hcx <= hx2 and hy1 <= hcy <= hy2:
                    person.has_helmet = (h["type"] == "HELMET")
                    person.helmet_conf = h["conf"]
                    matched = True
                    break

            if not matched:
                person.has_helmet = False

        return persons