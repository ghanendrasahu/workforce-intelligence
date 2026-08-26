"""Executive overview: KPIs, attrition trend, high-risk teams, top drivers."""
import pandas as pd
import streamlit as st

import apps.dashboard.client as api

st.title("Workforce intelligence")


@st.cache_data(ttl=60)
def trend() -> list[dict]:
    return api.get("/api/v1/analytics/trend")  # type: ignore[no-any-return]


@st.cache_data(ttl=60)
def departments() -> list[dict]:
    return api.get("/api/v1/analytics/departments")  # type: ignore[no-any-return]


@st.cache_data(ttl=60)
def drivers() -> list[dict]:
    return api.get("/api/v1/insights/global-drivers")  # type: ignore[no-any-return]


@st.cache_data(ttl=60)
def high_risk() -> list[dict]:
    return api.get("/api/v1/analytics/high-risk", {"top_k": 10})  # type: ignore[no-any-return]


trend_rows = trend()
trend_df = pd.DataFrame(trend_rows).tail(18)
dept_df = pd.DataFrame(departments())

latest_rate = float(trend_df["annualized_rate"].iloc[-1])
prev_rate = float(trend_df["annualized_rate"].iloc[-2]) if len(trend_df) > 1 else latest_rate
high_risk_n = int(dept_df["high_risk_employees"].sum()) \
    if "high_risk_employees" in dept_df else 0
total_headcount = int(dept_df["headcount"].sum())
avg_comp = float(dept_df["avg_comp_ratio"].mean())

with st.container(horizontal=True):
    st.metric("Attrition (annualized)", f"{latest_rate:.1f}%",
              f"{latest_rate - prev_rate:+.1f} pp", border=True,
              chart_data=trend_df["annualized_rate"].tolist(), chart_type="line")
    st.metric("High-risk employees", f"~{max(high_risk_n, 0):,}", border=True)
    st.metric("Headcount", f"{total_headcount:,}", border=True,
              chart_data=pd.DataFrame(trend_rows)["n_active"].tail(18).tolist(),
              chart_type="area")
    st.metric("Avg comp ratio", f"{avg_comp:.2f}", border=True)

left, right = st.columns([3, 2])

with left:
    with st.container(border=True):
        st.subheader("Attrition trend")
        st.line_chart(trend_df, x="date", y="annualized_rate", height=280)

    with st.container(border=True):
        st.subheader("Department scorecard")
        show = dept_df.rename(columns={
            "department": "dept", "annualized_attrition_pct": "attrition %",
            "attrition_events_12m": "leavers 12m", "avg_comp_ratio":
            "comp ratio", "avg_engagement": "engagement",
            "avg_overtime": "overtime h", "avg_risk": "mean risk",
            "high_risk_employees": "high risk"})
        st.dataframe(show, hide_index=True)

with right:
    with st.container(border=True):
        st.subheader("Top attrition drivers (SHAP)")
        drv = pd.DataFrame(drivers()).head(8).set_index("feature")
        st.bar_chart(drv["risk_points"], height=280)

    with st.container(border=True):
        st.subheader("Highest-risk employees")
        hr = pd.DataFrame(high_risk())
        if len(hr):
            st.dataframe(
                hr[["employee_id", "department", "level", "risk"]],
                hide_index=True, height=260,
            )
        else:
            st.caption("No data")

st.caption("Data: synthetic enterprise workforce simulator (documented in docs/ml.md). "
           "Model scores are isotonic-calibrated probabilities of exit within 6 months.")
