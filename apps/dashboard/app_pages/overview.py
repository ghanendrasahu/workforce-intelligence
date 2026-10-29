"""Executive overview: KPIs, attrition trend, high-risk teams, top drivers."""

import altair as alt
import pandas as pd
import streamlit as st
import ui

import apps.dashboard.client as api

ui.page_header(
    "Workforce intelligence",
    "Attrition risk, retention drivers and team health across the enterprise.",
    eyebrow="EXECUTIVE OVERVIEW",
    badge="Live model",
    badge_tone="success",
    icon_name="briefcase",
)


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


try:
    trend_rows = trend()
    dept_rows = departments()
    driver_rows = drivers()
    high_risk_rows = high_risk()
except api.ApiError:
    ui.error_state(
        "The analytics service could not load the overview right now.",
        title="Overview unavailable",
        hint="Refresh the page in a moment.",
    )
    st.stop()

if not trend_rows:
    ui.empty_state(
        "No trend data yet",
        "The workforce snapshot has not been generated for this deployment.",
        hint="Check back once the analytics service finishes starting.",
        icon_name="chart",
    )
    st.stop()

trend_df = pd.DataFrame(trend_rows).tail(18)
trend_df["date"] = pd.to_datetime(trend_df["date"])
dept_df = pd.DataFrame(dept_rows)

latest_rate = float(trend_df["annualized_rate"].iloc[-1])
prev_rate = float(trend_df["annualized_rate"].iloc[-2]) if len(trend_df) > 1 else latest_rate
high_risk_n = int(dept_df["high_risk_employees"].sum()) if "high_risk_employees" in dept_df else 0
total_headcount = (
    int(dept_df["headcount"].sum())
    if "headcount" in dept_df
    else int(trend_df["n_active"].iloc[-1])
)
avg_comp = float(dept_df["avg_comp_ratio"].mean()) if "avg_comp_ratio" in dept_df else 0.0

ui.metric_grid(
    [
        {
            "label": "Attrition (annualized)",
            "value": f"{latest_rate:.1f}%",
            "delta": latest_rate - prev_rate,
            "suffix": " pp",
            "good_direction": "down",
            "caption": "vs. previous month",
        },
        {
            "label": "High-risk employees",
            "value": f"~{max(high_risk_n, 0):,}",
            "caption": "across all departments",
        },
        {
            "label": "Headcount",
            "value": f"{total_headcount:,}",
            "caption": "active employees",
        },
        {
            "label": "Avg comp ratio",
            "value": f"{avg_comp:.2f}",
            "caption": "market benchmark = 1.00",
        },
    ]
)

left, right = st.columns([3, 2])

with left:
    with st.container(border=True):
        st.subheader("Attrition trend")
        st.caption("Annualized attrition over the last 18 months.")
        st.line_chart(
            trend_df,
            x="date",
            y="annualized_rate",
            x_label="Month",
            y_label="Annualized attrition (%)",
            color=ui.PRIMARY,
            height=280,
        )

    with st.container(border=True):
        st.subheader("Department scorecard")
        st.caption("Headcount, attrition and compensation health by team.")
        scoreboard = pd.DataFrame(
            {
                "Department": dept_df["department"],
                "Headcount": dept_df["headcount"],
                "Attrition": dept_df["annualized_attrition_pct"],
                "Leavers 12m": dept_df["attrition_events_12m"],
                "Comp ratio": dept_df["avg_comp_ratio"],
                "High risk": dept_df["high_risk_employees"],
            }
        )
        st.dataframe(
            scoreboard,
            hide_index=True,
            width="stretch",
            column_config={
                "Headcount": st.column_config.NumberColumn(format="%d"),
                "Attrition": st.column_config.NumberColumn(format="%.1f%%"),
                "Leavers 12m": st.column_config.NumberColumn(format="%d"),
                "Comp ratio": st.column_config.NumberColumn(format="%.2f"),
                "High risk": st.column_config.NumberColumn(format="%d"),
            },
        )

with right:
    with st.container(border=True):
        st.subheader("Top attrition drivers")
        st.caption("Average impact on predicted risk (SHAP).")
        driver_df = pd.DataFrame(driver_rows).head(8)
        driver_df["Driver"] = driver_df["feature"].str.replace("_", " ").str.title()
        driver_df["Impact"] = driver_df["risk_points"]
        chart = (
            alt.Chart(driver_df)
            .mark_bar(color=ui.PRIMARY, cornerRadiusEnd=4)
            .encode(
                x=alt.X("Impact:Q", title="Impact (pp)"),
                y=alt.Y("Driver:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Driver:N"),
                    alt.Tooltip("Impact:Q", format="+.1f", title="Impact (pp)"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, width="stretch")

    with st.container(border=True):
        st.subheader("Highest-risk employees")
        st.caption("Employees with the highest calibrated exit probability.")
        hr = pd.DataFrame(high_risk_rows)
        if len(hr):
            hr = hr[["employee_id", "department", "level", "risk"]].copy()
            hr["risk"] = (hr["risk"] * 100).round(0)
            hr.columns = ["Employee", "Department", "Level", "Risk"]
            st.dataframe(
                hr,
                hide_index=True,
                width="stretch",
                height=260,
                column_config={
                    "Risk": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=100,
                        format="%.0f%%",
                    )
                },
            )
        else:
            ui.empty_state(
                "No high-risk employees",
                "No one is currently above the high-risk threshold.",
                icon_name="check",
            )

st.caption(
    "Scores are calibrated probabilities of voluntary exit within six months, "
    "trained on the synthetic enterprise workforce dataset."
)
