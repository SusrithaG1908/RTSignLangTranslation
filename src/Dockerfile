FROM python:3.11-slim

# System dependencies for OpenCV headless, MediaPipe, aiortc
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/mobilenet_mp_25%_v2_best.h5 ./models/mobilenet_mp_25%_v2_best.h5
COPY models/class_labels_mobilenet_mp_25%_v2.json ./models/class_labels_mobilenet_mp_25%_v2.json

USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]