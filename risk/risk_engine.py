# risk/risk_engine.py

class RiskEngine:
    def __init__(self):
        pass

    def evaluate(self, in_risk_zone, speed, net_status="NORMAL"):
        """
        5단계 안전 등급:
        Level 1: 매우 안전 | Level 2: 안전 | Level 3: 주의 | Level 4: 경고 | Level 5: 심각
        """
        # 기본 상태
        score = 0

        # 방지망 상태 점수 (추후 멘토 모델과 연동)
        if net_status in ["DAMAGED", "OPEN", "UNINSTALLED"]:
            score += 50

        # 위험구역 침범 점수
        if in_risk_zone:
            score += 30
            # 위험구역 내에서 빠른 이동 속도를 보일 때 가산
            if speed > 15.0:
                score += 20

        # 레벨 매핑
        if score == 0:
            return {"level": 1, "label": "매우 안전", "color": (0, 255, 0)}
        elif score <= 30:
            return {"level": 2, "label": "주의 (진입)", "color": (0, 255, 255)}
        elif score <= 60:
            return {"level": 3, "label": "경고", "color": (0, 165, 255)}
        else:
            return {"level": 5, "label": "심각 (추락위험)", "color": (0, 0, 255)}