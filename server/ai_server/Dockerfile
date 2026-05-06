# 프로젝트 루트에서 빌드: docker build -f server/ai_server/Dockerfile -t ai-server .
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY server/ai_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# sixdrepnet 로컬 패키지 설치
COPY model/6DRepNet /app/model/6DRepNet
RUN pip install -e /app/model/6DRepNet

# 모델 코드 및 가중치
COPY model/risk_analyzer /app/model/risk_analyzer
COPY weights /app/model/weights

# 서버 코드
COPY server/ai_server/app /app/app

RUN mkdir -p uploads protected

ENV YOLO_WEIGHTS_PATH=/app/model/weights/yolov8n-face.pt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
