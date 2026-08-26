"""AI decision analyst: grounded chat over the analytics tool layer."""
import streamlit as st

import apps.dashboard.client as api

st.title("AI decision analyst")
st.caption("Answers are generated only from tool calls against the platform's "
           "analytics engine - attrition queries, cohort analysis, driver "
           "importance and what-if simulation.")

SUGGESTIONS = {
    ":blue[:material/trending_up:] Why is attrition increasing?":
        "Why is attrition increasing and which departments drive it?",
    ":orange[:material/groups:] Which teams are at highest risk?":
        "Which cohorts have the highest attrition risk right now?",
    ":green[:material/payments:] What if we raise salaries 10%?":
        "What happens if we give a 10% salary increase to high-risk employees - "
        "estimate the ROI.",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    selected = st.pills("Try asking:", list(SUGGESTIONS.keys()),
                        label_visibility="collapsed")
    if selected:
        st.session_state.pending = SUGGESTIONS[selected]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=":material/smart_toy:"
                         if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("tools"):
            st.caption("tools used: " + ", ".join(msg["tools"]))

prompt = st.chat_input("Ask anything about your workforce...",
                       submit_mode="disable")

if st.session_state.pop("pending", "") and prompt is None:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            resp = api.post("/api/v1/agent/ask", {"question": prompt})
            answer = str(resp["answer"])
            tools = list(resp.get("tools_used", []))
        except api.ApiError as exc:
            answer, tools = f"Analyst unavailable: {exc}", []
        st.markdown(answer)
        if tools:
            st.caption("tools used: " + ", ".join(tools))
    st.session_state.messages.append({"role": "assistant", "content": answer,
                                      "tools": tools})
    st.rerun()
