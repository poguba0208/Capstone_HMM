"""
pipeline.py - 위험도 분석 통합 진입점

백엔드에서 이거 하나만 import해서 쓰면 됨.

사용 예:
    from pipeline import analyze_risk

    # 1. 파일 경로
    result = analyze_risk("/path/to/image.jpg")

    # 2. 이미지 bytes (백엔드 업로드 처리)
    result = analyze_risk(image_bytes)

    # 3. numpy 배열 (BGR, OpenCV 형식)
    result = analyze_risk(numpy_array)

    # 4. PIL 이미지
    result = analyze_risk(pil_image)

    print(result)
    # {
    #   "success": True,
    #   "score": 67.5,
    #   "level": "HIGH",
    #   "yaw": 5.2,
    #   "pitch": -3.1,
    #   "face_ratio": 0.183,
    #   "error": None
    # }
"""

import io
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from sixdrepnet import SixDRepNet
import os

from risk_score import compute_risk_score, risk_level


# ────── 모델은 한 번만 로드 (모듈 레벨) ──────
_face_model = YOLO(os.environ.get("YOLO_WEIGHTS_PATH", "weights/yolov8n-face.pt"))
_pose_model = SixDRepNet()


# ────── 이미지 입력 정규화 ──────
def _to_bgr_numpy(image_input):
    """
    파일경로/bytes/numpy/PIL 어떤 형태든 OpenCV BGR numpy로 변환.

    Returns:
        np.ndarray (H, W, 3) BGR  또는  None (실패 시)
    """
    if image_input is None:
        return None

    # 1. numpy 배열
    if isinstance(image_input, np.ndarray):
        return image_input

    # 2. PIL 이미지
    if isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 3. 파일 경로 (str)
    if isinstance(image_input, str):
        return cv2.imread(image_input)

    # 4. bytes (백엔드 업로드)
    if isinstance(image_input, (bytes, bytearray)):
        arr = np.frombuffer(image_input, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 5. file-like object (UploadFile 등)
    if hasattr(image_input, "read"):
        data = image_input.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return None


# ────── 메인 진입점 (백엔드가 이거 하나만 호출) ──────
def analyze_risk(image_input):
    """
    이미지의 deepfake 위험도 분석.

    Args:
        image_input: 파일경로(str) / bytes / numpy.ndarray / PIL.Image / file-like

    Returns:
        dict: {
            "success" (bool),
            "score" (float, 0~100),
            "level" (str, "LOW"/"MEDIUM"/"HIGH"),
            "yaw" (float, degrees),
            "pitch" (float, degrees),
            "face_ratio" (float, 0~1),
            "error" (str or None)
        }
    """
    # 결과 기본값
    result = {
        "success": False,
        "score": 0.0,
        "level": None,
        "yaw": None,
        "pitch": None,
        "face_ratio": None,
        "error": None,
    }

    # 1. 입력 정규화
    img = _to_bgr_numpy(image_input)
    if img is None:
        result["error"] = "이미지를 읽을 수 없습니다."
        return result

    h, w = img.shape[:2]

    # 2. YOLO 얼굴 감지
    detections = _face_model(img, verbose=False)[0]
    if len(detections.boxes) == 0:
        result["error"] = "얼굴을 감지하지 못했습니다."
        return result

    # 가장 큰 얼굴 선택
    boxes = detections.boxes.xyxy.cpu().numpy()
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    x1, y1, x2, y2 = map(int, boxes[areas.index(max(areas))])
    face_crop = img[y1:y2, x1:x2]
    face_ratio = float((x2 - x1) * (y2 - y1) / (w * h))

    # 3. 머리 각도 추정
    pitch_arr, yaw_arr, roll_arr = _pose_model.predict(face_crop)
    yaw = float(yaw_arr[0])
    pitch = float(pitch_arr[0])

    # 4. 점수 계산
    score = compute_risk_score(yaw, pitch, face_ratio)
    level = risk_level(score)

    result.update({
        "success": True,
        "score": round(score, 2),
        "level": level,
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "face_ratio": round(face_ratio, 4),
    })
    return result

"""
pipeline.py - 위험도 분석 통합 진입점

백엔드에서 이거 하나만 import해서 쓰면 됨.

사용 예:
    from pipeline import analyze_risk

    # 1. 파일 경로
    result = analyze_risk("/path/to/image.jpg")

    # 2. 이미지 bytes (백엔드 업로드 처리)
    result = analyze_risk(image_bytes)

    # 3. numpy 배열 (BGR, OpenCV 형식)
    result = analyze_risk(numpy_array)

    # 4. PIL 이미지
    result = analyze_risk(pil_image)

    print(result)
    # {
    #   "success": True,
    #   "score": 67.5,
    #   "level": "HIGH",
    #   "yaw": 5.2,
    #   "pitch": -3.1,
    #   "face_ratio": 0.183,
    #   "error": None
    # }
"""

import io
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from sixdrepnet import SixDRepNet
import os

from risk_score import compute_risk_score, risk_level


# ────── 모델은 한 번만 로드 (모듈 레벨) ──────
_face_model = YOLO(os.environ.get("YOLO_WEIGHTS_PATH", "weights/yolov8n-face.pt"))
_pose_model = SixDRepNet()


# ────── 이미지 입력 정규화 ──────
def _to_bgr_numpy(image_input):
    """
    파일경로/bytes/numpy/PIL 어떤 형태든 OpenCV BGR numpy로 변환.

    Returns:
        np.ndarray (H, W, 3) BGR  또는  None (실패 시)
    """
    if image_input is None:
        return None

    # 1. numpy 배열
    if isinstance(image_input, np.ndarray):
        return image_input

    # 2. PIL 이미지
    if isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 3. 파일 경로 (str)
    if isinstance(image_input, str):
        return cv2.imread(image_input)

    # 4. bytes (백엔드 업로드)
    if isinstance(image_input, (bytes, bytearray)):
        arr = np.frombuffer(image_input, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 5. file-like object (UploadFile 등)
    if hasattr(image_input, "read"):
        data = image_input.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return None


# ────── 메인 진입점 (백엔드가 이거 하나만 호출) ──────
def analyze_risk(image_input):
    """
    이미지의 deepfake 위험도 분석.

    Args:
        image_input: 파일경로(str) / bytes / numpy.ndarray / PIL.Image / file-like

    Returns:
        dict: {
            "success" (bool),
            "score" (float, 0~100),
            "level" (str, "LOW"/"MEDIUM"/"HIGH"),
            "yaw" (float, degrees),
            "pitch" (float, degrees),
            "face_ratio" (float, 0~1),
            "error" (str or None)
        }
    """
    # 결과 기본값
    result = {
        "success": False,
        "score": 0.0,
        "level": None,
        "yaw": None,
        "pitch": None,
        "face_ratio": None,
        "error": None,
    }

    # 1. 입력 정규화
    img = _to_bgr_numpy(image_input)
    if img is None:
        result["error"] = "이미지를 읽을 수 없습니다."
        return result

    h, w = img.shape[:2]

    # 2. YOLO 얼굴 감지
    detections = _face_model(img, verbose=False)[0]
    if len(detections.boxes) == 0:
        result["error"] = "얼굴을 감지하지 못했습니다."
        return result

    # 가장 큰 얼굴 선택
    boxes = detections.boxes.xyxy.cpu().numpy()
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    x1, y1, x2, y2 = map(int, boxes[areas.index(max(areas))])
    face_crop = img[y1:y2, x1:x2]
    face_ratio = float((x2 - x1) * (y2 - y1) / (w * h))

    # 3. 머리 각도 추정
    pitch_arr, yaw_arr, roll_arr = _pose_model.predict(face_crop)
    yaw = float(yaw_arr[0])
    pitch = float(pitch_arr[0])

    # 4. 점수 계산
    score = compute_risk_score(yaw, pitch, face_ratio)
    level = risk_level(score)

    result.update({
        "success": True,
        "score": round(score, 2),
        "level": level,
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "face_ratio": round(face_ratio, 4),
    })
    return result



if __name__ == "__main__":
    test_path = "/workspace/capstone_pipeline/test_images/test_sulyoon.jpg"
    print(f"[테스트 입력] {test_path}\n")

    result = analyze_risk(test_path)

    print("=== 분석 결과 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
