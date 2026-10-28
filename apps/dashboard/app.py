"""Workforce Intelligence executive dashboard (Streamlit)."""

import sys
from pathlib import Path

# `streamlit run apps/dashboard/app.py` only puts apps/dashboard on sys.path.
# Add the repo root so `apps.dashboard.client` and the page modules resolve.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import ui  # noqa: E402

import apps.dashboard.client as api  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"

ui.setup("Workforce Intelligence", icon=ui.app_icon(ASSETS))

overview = st.Page("app_pages/overview.py", title="Overview", default=True)
employees = st.Page("app_pages/employees.py", title="Employee risk")
simulator = st.Page("app_pages/simulator.py", title="What-if simulator")
analyst = st.Page("app_pages/analyst.py", title="AI analyst")

try:
    health = api.get("/health")
except api.ApiError:
    ui.page_header(
        "Workforce intelligence",
        "Attrition risk, retention drivers and team health across the enterprise.",
        eyebrow="PEOPLE ANALYTICS",
        badge="Service unavailable",
        badge_tone="danger",
        icon_name="briefcase",
    )
    ui.error_state(
        "The analytics service is not responding right now. It may be waking up.",
        title="Dashboard unavailable",
        hint="Refresh the page in a moment.",
    )
    st.stop()

if health.get("status") != "ok":
    ui.page_header(
        "Workforce intelligence",
        "Attrition risk, retention drivers and team health across the enterprise.",
        eyebrow="PEOPLE ANALYTICS",
        badge="Models not ready",
        badge_tone="warning",
        icon_name="briefcase",
    )
    ui.error_state(
        "The analytics service is running but its trained models are not ready yet.",
        title="Models not ready",
        hint="Refresh once the deployment has finished starting.",
    )
    st.stop()

with st.sidebar:
    st.markdown(
        ui.badge("Model ready", "success"),
        unsafe_allow_html=True,
    )
    model_name = health.get("model_name") or "attrition model"
    st.caption(f"Scoring model: {model_name}")
    fingerprint = str(health.get("data_fingerprint") or "")[:8]
    if fingerprint:
        st.caption(f"Data snapshot: {fingerprint}")

pg = st.navigation([overview, employees, simulator, analyst], position="sidebar")
pg.run()
