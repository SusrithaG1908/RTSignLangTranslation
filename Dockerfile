FROM python:3.11-slim

# System dependencies — nginx added on top of your existing deps
RUN apt-get update && apt-get install -y \
    nginx \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
# This layer is cached as long as requirements.txt doesn't change — intentional.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Cache-bust: forces everything below to rebuild on every push ──────────────
# Pass a unique value each deployment, e.g. the git commit SHA or timestamp.
# HF Spaces: set CACHEBUST as a Space secret or repo variable, or just update
# the value below manually each time you want a guaranteed fresh build.
# You can also trigger via: git commit --allow-empty -m "rebuild"
ARG CACHEBUST=1
RUN echo "Cache busted at build arg: ${CACHEBUST}"

# Copy project files — always fresh after the ARG above
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/

# Copy nginx config and startup script
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh .
# Strip Windows CR line endings and ensure executable — safe even if already Unix
RUN sed -i 's/\r//' start.sh && chmod +x start.sh

# Nginx needs writable dirs — all redirected to /tmp so appuser can write
RUN mkdir -p /tmp/nginx_logs /tmp/nginx_body /tmp/nginx_proxy \
    && chown -R appuser:appuser /tmp/nginx_logs /tmp/nginx_body /tmp/nginx_proxy \
    && chown -R appuser:appuser /var/lib/nginx \
    && chown appuser:appuser /etc/nginx/nginx.conf

# Switch to non-root user (HF requirement)
USER appuser

# Expose 7860 — the single port HF Spaces Docker SDK makes public
# Nginx listens on 7860 and proxies internally to Streamlit :8501 and predict :8000.
EXPOSE 7860

# Healthcheck via nginx public port (not Streamlit directly)
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Entrypoint: nginx + streamlit launched together via start.sh
CMD ["./start.sh"]