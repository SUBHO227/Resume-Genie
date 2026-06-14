# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 10:35:38 2026

@author: KIIT
"""

import streamlit as st

st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🚀",
    layout="wide"
)

import tempfile
import os
import base64
from datetime import datetime

from PIL import Image

logo = Image.open("Logo.png")
st.sidebar.image(logo, width=120)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# =====================================
# GEMINI API KEY
# =====================================

google_api_key = st.text_input(
    "Google API Key",
    type="password"
)

# =====================================
# PAGE CONFIG
# =====================================



# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🚀 AI Career Assistant")

page = st.sidebar.radio(
    "Select Service",
    [
        "Home",
        "Resume Reviewer",
        "ATS Matcher",
        "Cover Letter Generator",
        "Career Coach"
    ]
)

# =====================================
# MODEL SELECTOR
# =====================================
if not google_api_key:
    st.warning("Please enter your Google API Key")
    st.stop()
model_name = st.sidebar.selectbox(
    "Model",
    [
        "gemini-2.5-flash",
        "gemini-1.5-pro"
    ],
    index=0
)

# =====================================
# SHARED GEMINI MODEL
# =====================================

chat = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=google_api_key,
    temperature=0.2
)

# =====================================
# HOME PAGE
# =====================================

if page == "Home":

    st.title("🚀 AI Career Assistant")

    st.markdown(
        """
### Welcome

This platform provides AI-powered career tools:

### 📄 Resume Reviewer
Analyze your resume and receive ATS-friendly feedback.

### 🎯 ATS Matcher
Compare your resume against a job description and get an ATS score.

### ✉️ Cover Letter Generator
Generate personalized ATS-friendly cover letters.

### 🤖 AI Career Coach
Chat with an AI career mentor based on your resume.

---

Select a service from the sidebar to get started.
"""
    )
# =====================================
# RESUME REVIEWER
# =====================================

elif page == "Resume Reviewer":

    st.title("📄 Resume ATS & AI Reviewer")

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
        key="reviewer_pdf"
    )

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.getbuffer()
            )

            tmp_path = tmp_file.name

        try:

            loader = PyPDFLoader(tmp_path)

            documents = loader.load()

            context = "\n\n".join(
                [doc.page_content for doc in documents]
            )

            prompt_template = PromptTemplate(
                input_variables=["context"],
                template="""
You are an expert ATS (Applicant Tracking System) and Resume Checker.

Analyze the following resume and provide a detailed evaluation.

Resume:
{context}

Instructions:

1. Give an Overall Resume Score out of 100.
2. Score the following categories individually:
   * Resume Structure and Formatting (20 marks)
   * Technical Skills (20 marks)
   * Projects (20 marks)
   * Education (10 marks)
   * Experience / Internships (15 marks)
   * Achievements and Certifications (15 marks)

3. Provide:
   * Top 5 Strengths
   * Top 5 Weaknesses

4. Extract all skills mentioned in the resume.

5. Suggest additional skills that would improve the resume.

6. Suggest project improvements if applicable.

7. Identify missing sections (if any).

8. Give ATS optimization tips.

9. Give 5 actionable recommendations to improve the resume.

10. Give the next career path for this candidate.
"""
            )

            if st.button(
                "🔍 Evaluate Resume",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing Resume..."
                ):

                    formatted_prompt = (
                        prompt_template.format(
                            context=context
                        )
                    )

                    result = chat.invoke(
                        formatted_prompt
                    )

                    st.success(
                        "✅ Analysis Complete!"
                    )

                    st.markdown(
                        result.content
                    )

                    st.download_button(
                        label="📥 Download Report",
                        data=result.content,
                        file_name="resume_evaluation.md",
                        mime="text/markdown"
                    )

        finally:

            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:

        st.info(
            "👆 Upload a PDF resume to begin."
        )
        # =====================================
# ATS MATCHER
# =====================================

elif page == "ATS Matcher":

    st.title("🎯 ATS Resume Matcher & Scorer")

    col1, col2 = st.columns([1, 1])

    with col1:

        uploaded_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"],
            key="ats_pdf"
        )

    with col2:

        job_description = st.text_area(
            "Paste Job Description",
            height=350,
            key="ats_jd"
        )

    if uploaded_file is not None and job_description.strip():

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.getbuffer()
            )

            tmp_path = tmp_file.name

        try:

            loader = PyPDFLoader(tmp_path)

            documents = loader.load()

            context = "\n\n".join(
                doc.page_content
                for doc in documents
            )

            prompt_template = PromptTemplate(
                input_variables=[
                    "context",
                    "job_description"
                ],
                template="""
You are an expert ATS Resume Scoring Specialist.

Resume:
{context}

Job Description:
{job_description}

Instructions:

1. Carefully compare the resume with the job description.
2. Give an honest ATS Match Score out of 100.
3. Identify Matching Skills.
4. Identify Missing / Weak Skills.
5. Explain why you gave this score.
6. Provide actionable improvement suggestions.

Return the response strictly in this format:

ATS Match Score: XX/100

Matching Skills:
- Skill 1
- Skill 2
- Skill 3

Missing Skills:
- Skill 1
- Skill 2
- Skill 3

Score Explanation:
[Professional Explanation]

Improvement Suggestions:
- Suggestion 1
- Suggestion 2
- Suggestion 3
- Suggestion 4
"""
            )

            if st.button(
                "🔥 Analyze ATS Match",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing Resume against Job Description..."
                ):

                    formatted_prompt = prompt_template.format(
                        context=context,
                        job_description=job_description
                    )

                    result = chat.invoke(
                        formatted_prompt
                    )

                    st.success(
                        "✅ Analysis Complete!"
                    )

                    st.markdown(
                        result.content
                    )

                    st.download_button(
                        label="📥 Download ATS Report",
                        data=result.content,
                        file_name="ATS_Report.txt",
                        mime="text/plain"
                    )

        finally:

            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:

        st.info(
            "👆 Upload a resume and paste a job description."
        )
        # =====================================
# COVER LETTER GENERATOR
# =====================================

elif page == "Cover Letter Generator":

    st.title("✉️ AI ATS-Friendly Cover Letter Generator")

    col1, col2 = st.columns(2)

    with col1:

        uploaded_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"],
            key="cover_letter_pdf"
        )

    with col2:

        job_description = st.text_area(
            "Paste Job Description",
            height=300,
            key="cover_letter_jd"
        )

    if uploaded_file is not None and job_description.strip():

        if st.button(
            "🚀 Generate Cover Letter",
            use_container_width=True
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(
                    uploaded_file.getbuffer()
                )

                tmp_path = tmp_file.name

            try:

                with st.spinner(
                    "Generating Cover Letter..."
                ):

                    loader = PyPDFLoader(
                        tmp_path
                    )

                    documents = loader.load()

                    context = "\n\n".join(
                        doc.page_content
                        for doc in documents
                    )

                    prompt_template = PromptTemplate(
                        input_variables=[
                            "context",
                            "job_description",
                            "date"
                        ],
                        template="""
You are an expert career consultant, ATS specialist, and professional resume writer.

Generate a personalized ATS-friendly cover letter using the candidate's resume and job description.

Resume:
{context}

Job Description:
{job_description}

Date:
{date}

Instructions:

- Extract the candidate's name from the resume.
- Create a professional business cover letter.
- Match the candidate's skills, projects, education, and achievements with the job requirements.
- Highlight the most relevant qualifications.
- Keep the tone professional, confident, and enthusiastic.
- Do not invent skills, projects, or experiences.
- Avoid placeholders such as [Your Name], [Company Name], [Date].
- Keep the cover letter between 300-400 words.
- End with a professional closing.

Return only the final cover letter.
"""
                    )

                    formatted_prompt = (
                        prompt_template.format(
                            context=context,
                            job_description=job_description,
                            date=datetime.now().strftime(
                                "%B %d, %Y"
                            )
                        )
                    )

                    result = chat.invoke(
                        formatted_prompt
                    )

                    st.success(
                        "✅ Cover Letter Generated Successfully!"
                    )

                    st.markdown(
                        result.content
                    )

                    st.download_button(
                        label="📥 Download Cover Letter",
                        data=result.content,
                        file_name="cover_letter.txt",
                        mime="text/plain"
                    )

            finally:

                if os.path.exists(
                    tmp_path
                ):
                    os.remove(
                        tmp_path
                    )

    else:

        st.info(
            "👆 Upload a resume and paste a job description."
        )
        # =====================================
# AI CAREER COACH
# =====================================

elif page == "Career Coach":

    st.title("🤖 AI Resume Career Coach")

    uploaded_file = st.file_uploader(
        "Upload Your Resume",
        type=["pdf"],
        key="coach_pdf"
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(
                uploaded_file.read()
            )

            pdf_path = tmp_file.name

        loader = PyPDFLoader(
            pdf_path
        )

        documents = loader.load()

        resume_text = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        system_message = SystemMessage(
            content=f"""
You are a professional career coach and resume mentor.

You help with:

- Career Guidance
- Resume Improvements
- Interview Preparation
- Job Search Strategy
- Skill Gap Analysis

Candidate Resume:

{resume_text}

Always answer according to the candidate's resume.
"""
        )

        if "career_chat_history" not in st.session_state:
            st.session_state.career_chat_history = []

        left_col, right_col = st.columns([1, 1])

        # ==========================
        # PDF VIEWER
        # ==========================

        with left_col:

            st.subheader("📄 Resume")

            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            base64_pdf = base64.b64encode(
                pdf_bytes
            ).decode("utf-8")

            pdf_display = f"""
            <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="900"
            type="application/pdf">
            </iframe>
            """

            st.markdown(
                pdf_display,
                unsafe_allow_html=True
            )

        # ==========================
        # CHATBOT
        # ==========================

        with right_col:

            st.subheader("🤖 Career Coach")

            for message in st.session_state.career_chat_history:

                if isinstance(
                    message,
                    HumanMessage
                ):

                    with st.chat_message(
                        "user"
                    ):
                        st.markdown(
                            message.content
                        )

                elif isinstance(
                    message,
                    AIMessage
                ):

                    with st.chat_message(
                        "assistant"
                    ):
                        st.markdown(
                            message.content
                        )

            user_input = st.chat_input(
                "Ask anything about your resume..."
            )

            if user_input:

                st.session_state.career_chat_history.append(
                    HumanMessage(
                        content=user_input
                    )
                )

                with st.chat_message(
                    "user"
                ):
                    st.markdown(
                        user_input
                    )

                messages = (
                    [system_message]
                    + st.session_state.career_chat_history
                )

                with st.chat_message(
                    "assistant"
                ):

                    response_placeholder = st.empty()

                    response_text = ""

                    for chunk in chat.stream(
                        messages
                    ):

                        if chunk.content:

                            response_text += (
                                chunk.content
                            )

                            response_placeholder.markdown(
                                response_text
                            )

                st.session_state.career_chat_history.append(
                    AIMessage(
                        content=response_text
                    )
                )

    else:

        st.info(
            "👆 Upload a resume to start chatting."
        )