"""Employee risk explorer: search, risk card, SHAP drivers, reviews."""
import pandas as pd
import streamlit as st

import apps.dashboard.client as api

st.title("Employee risk")


@st.cache_data(ttl=60)
def search(q: str) -> list[dict]:
    return api.get("/api/v1/employees/search", {"q": q, "limit": 15})  # type: ignore[no-any-return]


@st.cache_data(ttl=60)
def reviews(eid: str) -> list[dict]:
    return api.get(f"/api/v1/employees/{eid}/reviews")  # type: ignore[no-any-return]


q = st.text_input("Search by id, department or location", placeholder="e.g. EMP00"
                  "12 or Engineering")
rows = search(q)

if not rows:
    st.info("No matching employees.")
    st.stop()

frame = pd.DataFrame(rows)
frame["label"] = (frame["employee_id"] + " - " + frame["department"]
                  + " L" + frame["level"].astype(str))
choice = st.selectbox("Employee", frame["label"], label_visibility="collapsed")
selected = frame[frame["label"] == choice].iloc[0]
eid = str(selected["employee_id"])

risk = api.get(f"/api/v1/employees/{eid}/risk")

col1, col2, col3 = st.columns([2, 3, 3])

with col1:
    with st.container(border=True):
        band_color = {"high": "red", "medium": "orange", "low": "green"}[risk["band"]]
        st.metric("Attrition risk (6m)", f"{risk['risk'] * 100:.0f}%",
                  f"band: {risk['band']}", border=True)
        st.progress(risk["risk"], text=f":{band_color}[calibrated probability]")
        st.caption(f"raw model score {risk['risk_raw']:.3f} - confidence "
                   f"{risk['confidence']:.0%}")
        st.markdown(f"**{eid}** - {risk['department']} - level {risk['level']} "
                    f"- {risk['location']}")
        st.caption(f"salary Rs {risk['salary']:,.0f}")

with col2:
    with st.container(border=True):
        st.subheader("Why this score")
        drv = pd.DataFrame(risk["top_drivers"])
        if len(drv):
            drv = drv.set_index("feature")
            colors = ["#d64550" if v > 0 else "#3f9b6c" for v in drv["risk_points"]]
            st.bar_chart(drv["risk_points"], height=300)
            for _, row in drv.reset_index().iterrows():
                arrow = ":red[:material/north:]" if row["direction"] == "increases" \
                    else ":green[:material/south:]"
                st.markdown(
                    f"- {arrow} **{row['feature']}** "
                    f"`{row['value']}` -> {row['risk_points']:+.1f} pp")

with col3:
    with st.container(border=True):
        st.subheader("Recent review signals")
        revs = reviews(eid)
        if not revs:
            st.caption("No reviews on record")
        for r in revs[::-1]:
            tone = {"negative": "red", "positive": "green",
                    "neutral": "gray"}[r["sentiment_label"]]
            st.markdown(
                f":{tone}[:material/chat_bubble:] **{r['topic_primary'].replace('_', ' ')}** "
                f"({r['sentiment_score']:+.2f})")
            st.caption(str(r["text"]))

st.page_link("app_pages/simulator.py", label="Open what-if simulator for interventions",
             icon=":material/tune:")
