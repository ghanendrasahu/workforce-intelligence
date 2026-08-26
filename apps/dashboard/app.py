"""Workforce Intelligence executive dashboard (Streamlit)."""
import streamlit as st

import apps.dashboard.client as api

st.set_page_config(
    page_title="Workforce Intelligence",
    page_icon=":material/insights:",
    layout="wide",
)

overview = st.Page("app_pages/overview.py", title="Overview",
                   icon=":material/dashboard:", default=True)
employees = st.Page("app_pages/employees.py", title="Employee risk",
                    icon=":material/person_search:")
simulator = st.Page("app_pages/simulator.py", title="What-if simulator",
                    icon=":material/tune:")
analyst = st.Page("app_pages/analyst.py", title="AI analyst",
                  icon=":material/smart_toy:")

try:
    health = api.get("/health")
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

if health.get("status") != "ok":
    st.error("API is running but artifacts are missing - generate data and train "
             "models first (see README quickstart).")
    st.stop()

with st.sidebar:
    model_name = health.get("model_name") or "?"
    fp = str(health.get("data_fingerprint") or "")[:8]
    st.markdown(f"**:material/model_training:** model `{model_name}`")
    st.caption(f"data fingerprint {fp}")

pg = st.navigation([overview, employees, simulator, analyst],
                   position="sidebar")
pg.run()
