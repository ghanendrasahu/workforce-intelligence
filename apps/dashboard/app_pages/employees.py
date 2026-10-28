"""Employee risk explorer: search, risk card, SHAP drivers, reviews."""

import altair as alt
import pandas as pd
import streamlit as st
import ui

import apps.dashboard.client as api

BANDS = {
    "high": ("High risk", "danger"),
    "medium": ("Medium risk", "warning"),
    "low": ("Low risk", "success"),
}
SENTIMENT_TONES = {"negative": "danger", "positive": "success", "neutral": "neutral"}

ui.page_header(
    "Employee risk",
    "Understand who is at risk, why the model scored them that way, and what "
    "their recent review signals say.",
    eyebrow="INDIVIDUAL VIEW",
    icon_name="target",
)


@st.cache_data(ttl=60)
def search(q: str) -> list[dict]:
    return api.get("/api/v1/employees/search", {"q": q, "limit": 15})  # type: ignore[no-any-return]


@st.cache_data(ttl=60)
def reviews(eid: str) -> list[dict]:
    return api.get(f"/api/v1/employees/{eid}/reviews")  # type: ignore[no-any-return]


ui.section(
    "Find an employee",
    "Search by employee ID, department or location to open their risk profile.",
)

q = st.text_input(
    "Search employees",
    placeholder="Employee ID, department or location",
)

try:
    rows = search(q)
except api.ApiError:
    ui.error_state(
        "The employee directory could not be loaded right now.",
        title="Search unavailable",
        hint="Refresh the page in a moment.",
    )
    st.stop()

if not rows:
    ui.empty_state(
        "No matching employees",
        "Try a different employee ID, department or location.",
        hint="Search is not case sensitive.",
        icon_name="search",
    )
    st.stop()

frame = pd.DataFrame(rows)
frame["label"] = (
    frame["employee_id"] + " - " + frame["department"] + " L" + frame["level"].astype(str)
)
if len(frame) == 1:
    selected = frame.iloc[0]
else:
    choice = st.selectbox("Employee", frame["label"])
    selected = frame[frame["label"] == choice].iloc[0]
eid = str(selected["employee_id"])

try:
    risk = api.get(f"/api/v1/employees/{eid}/risk")
except api.ApiError:
    ui.error_state(
        "The risk profile for this employee could not be loaded right now.",
        title="Profile unavailable",
        hint="Choose another employee or refresh the page.",
    )
    st.stop()

band_label, band_tone = BANDS.get(risk["band"], ("Risk band", "neutral"))

ui.section(
    "Risk profile",
    f"{eid} · {risk['department']} · Level {risk['level']} · {risk['location']}",
)

ui.metric_grid(
    [
        {
            "label": "Attrition risk (6 months)",
            "value": f"{risk['risk'] * 100:.0f}%",
            "caption": f"raw score {risk['risk_raw']:.3f}",
        },
        {
            "label": "Risk band",
            "value": band_label,
            "caption": f"confidence {risk['confidence']:.0%}",
        },
        {
            "label": "Role",
            "value": f"Level {risk['level']}",
            "caption": str(risk["department"]),
        },
        {
            "label": "Salary",
            "value": f"Rs {risk['salary'] / 100000:.1f}L",
            "caption": str(risk["location"]),
        },
    ]
)

ui.meter(
    float(risk["risk"]),
    tone=band_tone,
    caption="Calibrated probability of voluntary exit within six months",
)

left, right = st.columns([3, 2])

with left:
    with st.container(border=True):
        st.subheader("Why this score")
        st.caption("Each factor's contribution to the predicted risk, in percentage points.")
        driver_df = pd.DataFrame(risk["top_drivers"])
        if len(driver_df):
            driver_df["Feature"] = driver_df["feature"].str.replace("_", " ").str.title()
            driver_df["Value"] = driver_df["value"].map(
                lambda value: f"{value:.2f}" if isinstance(value, float) else str(value)
            )
            driver_df = driver_df.drop(columns=["value"])
            chart = (
                alt.Chart(driver_df)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("risk_points:Q", title="Impact on risk (pp)"),
                    y=alt.Y("Feature:N", sort="-x", title=None),
                    color=alt.Color(
                        "direction:N",
                        scale=alt.Scale(
                            domain=["increases", "decreases"],
                            range=[ui.DANGER, ui.SUCCESS],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Feature:N"),
                        alt.Tooltip("Value:N"),
                        alt.Tooltip("risk_points:Q", format="+.1f", title="Impact (pp)"),
                    ],
                )
                .properties(height=max(220, 34 * len(driver_df)))
            )
            st.altair_chart(chart, width="stretch")
            for _, row in driver_df.iterrows():
                colour = ui.DANGER if row["direction"] == "increases" else ui.SUCCESS
                st.markdown(
                    f'<div class="ui-kv">'
                    f"<span>{row['Feature']} "
                    f'<span style="color:var(--ui-muted)">· {row["Value"]}</span></span>'
                    f'<span style="color:{colour};font-weight:600">'
                    f"{row['risk_points']:+.1f} pp</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            ui.empty_state(
                "No driver detail",
                "The model did not return factor contributions for this employee.",
                icon_name="info",
            )

with right:
    with st.container(border=True):
        st.subheader("Recent review signals")
        st.caption("Themes from the employee's latest written reviews.")
        try:
            revs = reviews(eid)
        except api.ApiError:
            ui.error_state(
                "Review signals could not be loaded right now.",
                title="Reviews unavailable",
            )
        else:
            if not revs:
                ui.empty_state(
                    "No reviews on record",
                    "This employee has not submitted a written review yet.",
                    icon_name="message",
                )
            for review in revs[::-1]:
                tone = SENTIMENT_TONES.get(review["sentiment_label"], "neutral")
                topic = str(review["topic_primary"]).replace("_", " ").title()
                st.markdown(
                    f"{ui.badge(topic, tone)} "
                    f'<span class="ui-brand-sub">'
                    f"score {review['sentiment_score']:+.2f}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(str(review["text"]))

st.page_link(
    "app_pages/simulator.py",
    label="Open the what-if simulator for interventions",
)
