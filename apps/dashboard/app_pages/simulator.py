"""What-if simulator: interventions on one employee or a cohort, with ROI."""
import streamlit as st

import apps.dashboard.client as api

st.title("What-if simulator")

st.caption("Counterfactual re-scoring with the frozen model: interventions mutate "
           "employee state, the model re-predicts, and costs use configurable "
           "business constants.")


@st.cache_data(ttl=120)
def departments() -> list[str]:
    rows = api.get("/api/v1/analytics/departments")
    return [r["department"] for r in rows]  # type: ignore[misc]


mode = st.segmented_control("Scope", ["Single employee", "Cohort"], default="Cohort")

with st.sidebar:
    st.subheader("Interventions")
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
    eid = st.text_input("Employee id", placeholder="EMP00123")
    if st.button("Simulate", type="primary") and eid:
        try:
            res = api.post("/api/v1/whatif/simulate",
                           {"employee_id": eid.strip(), "interventions": params})
        except api.ApiError as exc:
            st.error(str(exc))
            st.stop()
        c1, c2, c3 = st.columns(3, border=True)
        c1.metric("Risk before", f"{res['risk_before'] * 100:.1f}%")
        c2.metric("Risk after", f"{res['risk_after'] * 100:.1f}%",
                  f"{-res['reduction_pp']:+.1f} pp", border=True)
        c3.metric("Confidence", f"{res['confidence']:.0%}", border=True)
        c4, c5 = st.columns(2)
        c4.metric("Annual intervention cost",
                  f"Rs {res['annual_cost'] / 100000:.1f}L", border=True)
        c5.metric("Expected annual cost avoided",
                  f"Rs {res['annual_cost_avoided'] / 100000:.1f}L", border=True)
        st.caption(f"band {res['band_before']} -> {res['band_after']}")
else:
    c1, c2, c3 = st.columns(3)
    dept = c1.selectbox("Department", ["All"] + departments())
    min_risk = c2.slider("Only employees with risk above", 0.0, 0.9, 0.3, 0.05)
    limit = c3.select_slider("Cohort size cap", [200, 500, 1000, 2000, 5000],
                             value=2000)

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
            except api.ApiError as exc:
                st.error(str(exc))
                st.stop()

        m1, m2, m3, m4 = st.columns(4, border=True)
        m1.metric("Employees", f"{sc['n_employees']:,}")
        m2.metric("Avg risk", f"{sc['avg_risk_before'] * 100:.1f}%",
                  f"{(sc['avg_risk_after'] - sc['avg_risk_before']) * 100:+.1f} pp",
                  border=True)
        m3.metric("Cost avoided / yr",
                  f"Rs {sc['annual_cost_avoided'] / 100000:.1f}L", border=True)
        roi = sc.get("roi_multiple")
        m4.metric("ROI", f"{roi:.2f}x" if roi else "-", border=True)

        k1, k2 = st.columns(2)
        k1.metric("Intervention cost / yr",
                  f"Rs {sc['annual_cost_of_intervention'] / 100000:.1f}L")
        k2.metric("Moved out of high-risk band",
                  f"{sc['employees_moved_out_of_high_risk']:,}")

        st.caption(f"expected attrition {sc['expected_attrition_before']} -> "
                   f"{sc['expected_attrition_after']}")
