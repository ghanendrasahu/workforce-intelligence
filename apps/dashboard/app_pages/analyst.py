"""AI decision analyst: grounded chat over the analytics tool layer."""

import streamlit as st
import ui

import apps.dashboard.client as api

AVATARS = {
    "user": ":material/person:",
    "assistant": ":material/query_stats:",
}

SUGGESTIONS = {
    "Why is attrition increasing?": ("Why is attrition increasing and which departments drive it?"),
    "Which teams are at highest risk?": (
        "Which cohorts have the highest attrition risk right now?"
    ),
    "What if we raise salaries 10%?": (
        "What happens if we give a 10% salary increase to high-risk employees - estimate the ROI."
    ),
}

ui.page_header(
    "AI decision analyst",
    "Ask questions about attrition, teams, drivers and interventions. Answers "
    "are grounded in the platform's analytics engine.",
    eyebrow="CONVERSATIONAL ANALYTICS",
    icon_name="message",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    ui.empty_state(
        "Ask about your workforce",
        "The analyst answers from live attrition, driver and what-if analysis "
        "rather than general knowledge.",
        hint="Choose a starting question below or type your own.",
        icon_name="message",
    )

    def _queue_suggestion() -> None:
        selected = st.session_state.get("suggestion")
        if selected in SUGGESTIONS:
            st.session_state.pending = SUGGESTIONS[selected]

    st.pills(
        "Try asking:",
        list(SUGGESTIONS.keys()),
        key="suggestion",
        on_change=_queue_suggestion,
        label_visibility="collapsed",
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg.get("tools"):
            st.caption("Answer grounded in live analytics.")

prompt = st.chat_input("Ask anything about your workforce...", submit_mode="disable")

pending = st.session_state.pop("pending", "")
if pending and prompt is None:
    prompt = pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            resp = api.post("/api/v1/agent/ask", {"question": prompt})
            answer = str(resp["answer"])
            tools = list(resp.get("tools_used", []))
        except api.ApiError as exc:
            if exc.status_code == 503:
                answer = (
                    "The AI analyst is not enabled in this environment yet. "
                    "The overview, employee risk and what-if simulator remain "
                    "available."
                )
            else:
                answer = "The analyst is unavailable right now. Try again in a moment."
            tools = []
        st.markdown(answer)
        if tools:
            st.caption("Answer grounded in live analytics.")
    st.session_state.messages.append({"role": "assistant", "content": answer, "tools": tools})
    st.rerun()
