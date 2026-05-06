"""
risk_score.py - 사진의 deepfake 위험도 점수 계산

입력 (다른 모듈에서 받음):
  - yaw (degrees) : 좌우 각도, head_pose.py가 리턴
  - pitch (degrees) : 상하 각도, head_pose.py가 리턴
  - face_ratio (0~1) : 얼굴 영역 / 전체 이미지, detect_face.py가 리턴

출력:
  - score (0~100) : 높을수록 deepfake에 취약 (= 보호 더 강하게 필요)
  - level ("LOW" / "MEDIUM" / "HIGH") : 등급

설계 원리:
  1) yaw 0° (정면) → 위험 ↑   |   ±90° (측면) → 위험 ↓
  2) pitch 0° (수평) → 위험 ↑  |   극단 각도 → 위험 ↓
  3) face_ratio 큼 → 위험 ↑   (얼굴이 커야 deepfake 품질 좋아짐)
"""

# ───────────── 튜닝 가능한 상수 (실험으로 조정) ─────────────
W_YAW   = 0.4    # 정면성 가중치 (가장 큼)
W_PITCH = 0.2    # 위아래 가중치
W_AREA  = 0.4    # 얼굴 크기 가중치

YAW_MAX_DEG   = 90.0   # 이 이상은 완전 측면
PITCH_MAX_DEG = 60.0   # 이 이상은 극단적 위/아래

AREA_SCALE = 4.0       # face_ratio 0.25 이상이면 최대 위험으로 간주
                        # (보통 셀카 0.2~0.4, 인물사진 0.05~0.2)

LEVEL_LOW    = 30      # score < 30 → LOW
LEVEL_MEDIUM = 60      # score < 60 → MEDIUM, 그 이상 → HIGH


# ───────────── 헬퍼 ─────────────
def _clip01(x):
    """[0, 1] 구간으로 클리핑"""
    return max(0.0, min(1.0, float(x)))


# ───────────── 메인 함수 ─────────────
def compute_risk_score(yaw, pitch, face_ratio, verbose=False):
    """
    Args:
        yaw (float): head_pose.py가 리턴한 좌우 각도 (degrees)
        pitch (float): head_pose.py가 리턴한 상하 각도 (degrees)
        face_ratio (float): detect_face.py가 리턴한 얼굴 비율 (0~1)
        verbose (bool): 중간 계산값 출력 여부

    Returns:
        float: 위험도 점수 (0~100)
    """
    # 정면 = 1.0, 측면 = 0.0
    yaw_score = _clip01(1.0 - abs(yaw) / YAW_MAX_DEG)

    # 수평 = 1.0, 극단 = 0.0
    pitch_score = _clip01(1.0 - abs(pitch) / PITCH_MAX_DEG)

    # 얼굴 클수록 ↑ (AREA_SCALE 이상이면 1.0 saturation)
    area_score = _clip01(face_ratio * AREA_SCALE)

    score = (W_YAW * yaw_score
             + W_PITCH * pitch_score
             + W_AREA * area_score) * 100.0

    if verbose:
        print(f"  yaw_score   = {yaw_score:.3f} (yaw={yaw:.2f}°)")
        print(f"  pitch_score = {pitch_score:.3f} (pitch={pitch:.2f}°)")
        print(f"  area_score  = {area_score:.3f} (face_ratio={face_ratio:.4f})")
        print(f"  weights     = ({W_YAW}, {W_PITCH}, {W_AREA})")
        print(f"  → final score = {score:.2f}")

    return score


def risk_level(score):
    """점수 → 등급 문자열"""
    if score < LEVEL_LOW:
        return "LOW"
    elif score < LEVEL_MEDIUM:
        return "MEDIUM"
    else:
        return "HIGH"


# ───────────── 단독 실행 테스트 ─────────────
if __name__ == "__main__":
    from detect_face import detect_face
    from head_pose import get_head_pose

    image_path = '/workspace/capstone_pipeline/test_images/sidepic.jpg'
    print(f"[입력] {image_path}\n")

    # 1. 얼굴 감지
    face_result = detect_face(image_path)
    if face_result is None:
        exit("얼굴 감지 실패")
    face_crop, face_ratio = face_result

    # 2. 머리 각도 추정
    pose_result = get_head_pose(face_crop)
    if pose_result is None:
        exit("head pose 추정 실패")
    yaw, pitch = pose_result

    # 3. 위험도 계산
    print("\n[중간 계산값]")
    score = compute_risk_score(yaw, pitch, face_ratio, verbose=True)
    level = risk_level(score)

    print(f"\n=== 위험도 분석 결과 ===")
    print(f"yaw         : {yaw:.2f}°")
    print(f"pitch       : {pitch:.2f}°")
    print(f"face_ratio  : {face_ratio:.4f}")
    print(f"score       : {score:.2f} / 100")
    print(f"level       : {level}")