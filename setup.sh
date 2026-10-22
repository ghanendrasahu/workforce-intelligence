#!/usr/bin/env bash
# Streamlit Cloud entry point for the dashboard.
#
# The dashboard is a thin HTTP client: only streamlit/httpx/pandas are needed
# (the full runtime.lock is for the API side, served by the Docker Space).
# API_BASE_URL must be set in Streamlit Cloud secrets to the deployed API.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -r apps/dashboard/requirements-streamlit.txt

exec streamlit run apps/dashboard/app.py
