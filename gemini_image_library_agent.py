# Google Gemini Image Library Agent
#
# Author: Peter Jakubowski
# Date: 2/1/2025
# Description: Streamlit chat app using the Google Gemini API
#

import streamlit as st
from google.genai.errors import ClientError, ServerError
from agent_tools import GeminiChat, process_response


# ==========================
# ===== Streamlit Begin ====
# ==========================

st.header('🖼 Gemini Image Library Agent 🤖')

# ==================
# === BEGIN CHAT ===
# ==================

st.session_state.avatars = {"assistant": "🤖", "user": "😺"}

if "chat" not in st.session_state:
    # # Start a new chat
    st.session_state.chat = GeminiChat().chat

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar=st.session_state.avatars[msg["role"]]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=st.session_state.avatars['user']).write(prompt)

    with st.spinner("Processing request...", show_time=True):
        response = st.session_state.chat.send_message(prompt)
        try:
            message = process_response(response)
        except ClientError as ce:
            st.error(ce.__str__())
        except ServerError as se:
            st.error(se.__str__())
        else:
            for msg in message:
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant", avatar=st.session_state.avatars['assistant']).write(msg)
