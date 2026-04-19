#!/bin/bash
set -e
echo "===== Starting nginx on :7860 ====="
nginx -g "daemon off;" &
echo "===== Starting Streamlit on :8501 ====="
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false