# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:03:49 2026

@author: KIIT
"""

import streamlit as st
import tempfile
import base64

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Resume Coach",
    layout="wide"
)

st.title("AI Resume Coach")

# -----------------------------
# Gemini Model
# -----------------------------
google_api_key = st.text_input(
    "Google API Key",
    type="password"
)

if not google_api_key:
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

# -----------------------------
# Upload Resume
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Your Resume",
    type=["pdf"]
)

if uploaded_file:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # -----------------------------
    # Extract Resume Text
    # -----------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    resume_text = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # -----------------------------
    # Create System Prompt
    # -----------------------------
    system_message = SystemMessage(
        content=f"""
You are a professional career coach and resume mentor.

You help with:
- Career Guidance
- Resume Improvements
- Interview Preparation
- Job Search Strategy
- Skill Gap Analysis

The following is the candidate's resume:

{resume_text}

Always answer based on the candidate's resume whenever relevant.
"""
    )

    # -----------------------------
    # Chat History
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -----------------------------
    # Two Column Layout
    # -----------------------------
    left_col, right_col = st.columns([1, 1])

    # =====================================================
    # LEFT SIDE : PDF VIEWER
    # =====================================================
    with left_col:

        st.subheader("Resume")

        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        base64_pdf = base64.b64encode(
            pdf_bytes
        ).decode("utf-8")

        pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="900px"
            type="application/pdf">
        </iframe>
        """

        st.markdown(
            pdf_display,
            unsafe_allow_html=True
        )

    # =====================================================
    # RIGHT SIDE : CHATBOT
    # =====================================================
    with right_col:

        st.subheader("Career Coach")

        # Display previous messages
        for message in st.session_state.chat_history:

            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)

            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)

        # Chat Input
        user_input = st.chat_input(
            "Ask anything about your resume..."
        )

        if user_input:

            # Store User Message
            st.session_state.chat_history.append(
                HumanMessage(content=user_input)
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            messages = (
                [system_message]
                + st.session_state.chat_history
            )

            with st.chat_message("assistant"):

                response_placeholder = st.empty()
                response_text = ""

                for chunk in llm.stream(messages):

                    if chunk.content:
                        response_text += chunk.content
                        response_placeholder.markdown(
                            response_text
                        )

            # Store AI Response
            st.session_state.chat_history.append(
                AIMessage(content=response_text)
            )

else:
    st.info("Upload a PDF resume to start chatting.")