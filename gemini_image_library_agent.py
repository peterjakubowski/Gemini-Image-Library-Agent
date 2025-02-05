# Google Gemini Image Library Agent
#
# Author: Peter Jakubowski
# Date: 2/1/2025
# Description: Streamlit chat app using the Google Gemini API
#

import streamlit as st
from google.genai.errors import ClientError
from agent_tools import GeminiChat, process_response


# ==========================
# ===== Streamlit Begin ====
# ==========================

st.header('Gemini Image Library Agent')

# ==================
# === BEGIN CHAT ===
# ==================

avatars = {"assistant": "👾", "user": "😺"}

if "chat" not in st.session_state:
    # # Start a new chat
    st.session_state.chat = GeminiChat().chat

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar=avatars[msg["role"]]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=avatars['user']).write(prompt)
    response = st.session_state.chat.send_message(prompt)

    try:
        msg = process_response(response, st.session_state.chat)
    except ClientError as ce:
        msg = ce

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant", avatar=avatars['assistant']).write(msg)
