"""What-if simulator: interventions on one employee or a cohort, with ROI."""

import streamlit as st
import ui

import apps.dashboard.client as api

BANDS = {
    "high": ("High risk", "danger"),
    "medium": ("Medium risk", "warning"),
    "low": ("Low risk", "success"),
}

ui.page_header(
    "What-if simulator",
    "Re-score employees after retention interventions and estimate the cost, "
    "risk reduction and return.",
    eyebrow="SCENARIO PLANNING",
    icon_name="sliders",
)


@st.cache_data(ttl=120)
def departments() -> list[str]:
    rows = api.get("/api/v1/analytics/departments")
    return [row["department"] for row in rows]  # type: ignore[misc]


mode = st.segmented_control("Scope", ["Single employee", "Cohort"], default="Cohort")

with st.sidebar:
    st.subheader("Interventions")
    st.caption("Adjust the levers applied in this scenario.")
    raise_pct = st.slider("Salary adjustment", 0, 30, 10, 1, format="%d%%")
    workload_cut = st.slider("Workload reduction", 0, 60, 20, 5, format="%d%%")
    promote = st.toggle("Accelerate promotion")
    engagement_boost = st.slider("Training investment (engagement pts)", 0, 15, 5, 1)
    coaching = st.slider("Manager coaching (sentiment pts)", 0.0, 0.5, 0.2, 0.1)

params = {
    "salary_raise_pct": float(raise_pct),
    "workload_reduction_pct": float(workload_cut),
    "promotion": promote,
    "engagement_boost_points": float(engagement_boost),
    "manager_sentiment_boost": float(coaching),
}

if mode == "Single employee":
    ui.section(
        "Employee scenario",
        "Re-score one employee after the interventions in the sidebar.",
    )
    eid = st.text_input("Employee ID", placeholder="EMP00123")
    if st.button("Simulate employee", type="primary"):
        if not eid.strip():
            ui.error_state(
                "Enter an employee ID before running the simulation.",
                title="Employee ID required",
                hint="IDs look like EMP00123.",
            )
        else:
            try:
                res = api.post(
                    "/api/v1/whatif/simulate",
                    {"employee_id": eid.strip(), "interventions": params},
                )
            except api.ApiError:
                ui.error_state(
                    "The employee could not be re-scored right now.",
                    title="Simulation unavailable",
                    hint="Check the employee ID or try again in a moment.",
                )
            else:
                before_label, before_tone = BANDS.get(res["band_before"], ("Before", "neutral"))
                after_label, after_tone = BANDS.get(res["band_after"], ("After", "neutral"))
                ui.metric_grid(
                    [
                        {
                            "label": "Risk before",
                            "value": f"{res['risk_before'] * 100:.1f}%",
                            "caption": before_label,
                        },
                        {
                            "label": "Risk after",
                            "value": f"{res['risk_after'] * 100:.1f}%",
                            "delta": -float(res["reduction_pp"]),
                            "suffix": " pp",
                            "good_direction": "down",
                            "caption": after_label,
                        },
                        {
                            "label": "Risk reduction",
                            "value": f"{res['reduction_pp']:.1f} pp",
                            "caption": f"confidence {res['confidence']:.0%}",
                        },
                        {
                            "label": "Annual intervention cost",
                            "value": f"Rs {res['annual_cost'] / 100000:.1f}L",
                            "caption": "salary, training and coaching",
                        },
                        {
                            "label": "Expected annual cost avoided",
                            "value": f"Rs {res['annual_cost_avoided'] / 100000:.1f}L",
                            "caption": "replacement and lost productivity",
                        },
                    ]
                )
                st.markdown(
                    f"Risk band: {ui.badge(before_label, before_tone)} "
                    f"&rarr; {ui.badge(after_label, after_tone)}",
                    unsafe_allow_html=True,
                )
else:
    ui.section(
        "Cohort scenario",
        "Apply the interventions to a filtered group and compare outcomes.",
    )
    c1, c2, c3 = st.columns(3)
    dept = c1.selectbox("Department", ["All"] + departments())
    min_risk = c2.slider("Only employees with risk above", 0.0, 0.9, 0.3, 0.05)
    limit = c3.select_slider("Cohort size cap", [200, 500, 1000, 2000, 5000], value=2000)

    if st.button("Run scenario", type="primary"):
        payload = {
            "department": None if dept == "All" else dept,
            "min_risk": min_risk if min_risk > 0 else None,
            "limit": limit,
            "interventions": params,
        }
        with st.spinner("Re-scoring cohort..."):
            try:
                sc = api.post("/api/v1/whatif/scenario", payload)
            except api.ApiError:
                ui.error_state(
                    "The cohort could not be re-scored right now.",
                    title="Scenario unavailable",
                    hint="Try again in a moment.",
                )
            else:
                roi = sc.get("roi_multiple")
                ui.metric_grid(
                    [
                        {
                            "label": "Employees in scenario",
                            "value": f"{sc['n_employees']:,}",
                            "caption": "matched the cohort filters",
                        },
                        {
                            "label": "Avg risk",
                            "value": f"{sc['avg_risk_after'] * 100:.1f}%",
                            "delta": (sc["avg_risk_after"] - sc["avg_risk_before"]) * 100,
                            "suffix": " pp",
                            "good_direction": "down",
                            "caption": (f"from {sc['avg_risk_before'] * 100:.1f}%"),
                        },
                        {
                            "label": "Cost avoided / yr",
                            "value": f"Rs {sc['annual_cost_avoided'] / 100000:.1f}L",
                            "caption": "expected replacement savings",
                        },
                        {
                            "label": "ROI",
                            "value": f"{roi:.2f}x" if roi else "—",
                            "caption": "cost avoided per rupee spent",
                        },
                        {
                            "label": "Intervention cost / yr",
                            "value": f"Rs {sc['annual_cost_of_intervention'] / 100000:.1f}L",
                            "caption": "salary, training and coaching",
                        },
                        {
                            "label": "Moved out of high-risk band",
                            "value": f"{sc['employees_moved_out_of_high_risk']:,}",
                            "caption": "employees below the threshold",
                        },
                    ]
                )
                st.caption(
                    f"Expected attrition falls from "
                    f"{sc['expected_attrition_before']} to "
                    f"{sc['expected_attrition_after']}."
                )
