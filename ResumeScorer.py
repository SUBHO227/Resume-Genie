# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:20:27 2026

@author: KIIT
"""

import streamlit as st
import tempfile
import os


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

# ==========================
# Load Environment Variables
# ==========================

st.set_page_config(
    page_title="ATS Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

google_api_key = st.text_input(
    "Google API Key",
    type="password"
)

if not google_api_key:
    st.stop()

# ==========================
# Streamlit Config
# ==========================


st.title("🎯 ATS Resume Matcher & Scorer")

st.markdown(
    """
Upload your resume and paste the Job Description to get an
**ATS Match Score** and detailed analysis.
"""
)

# ==========================
# Sidebar
# ==========================

with st.sidebar:

    st.header("Configuration")

    model_name = st.selectbox(
        "Select Model",
        [
            "gemini-2.5-flash",
            "gemini-1.5-pro"
        ],
        index=0
    )

# ==========================
# Check API Key
# ==========================



# ==========================
# Main Layout
# ==========================

col1, col2 = st.columns([1, 1])

with col1:

    uploaded_file = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=["pdf"]
    )

with col2:

    job_description = st.text_area(
        "Paste Job Description",
        height=350,
        placeholder="Paste the full job description here...",
        help="The more detailed the JD, the better the analysis."
    )

# ==========================
# Processing
# ==========================

if uploaded_file is not None and job_description.strip():

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:

        # ==========================
        # Read Resume
        # ==========================

        loader = PyPDFLoader(tmp_path)

        documents = loader.load()

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        # ==========================
        # Prompt
        # ==========================

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

        # ==========================
        # Analyze Button
        # ==========================

        if st.button(
            "🔥 Analyze ATS Match",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing resume against Job Description..."
            ):

                try:

                    chat = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=google_api_key,
                        temperature=0.2
                    )

                    formatted_prompt = prompt_template.format(
                        context=context,
                        job_description=job_description
                    )

                    result = chat.invoke(
                        formatted_prompt
                    )

                    # ==========================
                    # Output
                    # ==========================

                    st.success(
                        "✅ Analysis Complete!"
                    )

                    st.markdown(
                        "## ATS Match Result"
                    )

                    st.markdown("---")

                    st.markdown(
                        result.content
                    )

                    # ==========================
                    # Download Report
                    # ==========================

                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=result.content,
                        file_name="ATS_Resume_Match_Report.txt",
                        mime="text/plain"
                    )

                except Exception as e:

                    st.error(
                        f"Error during analysis: {str(e)}"
                    )

    finally:

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:

    st.info(
        "👆 Please upload your resume and paste the job description to get started."
    )

st.markdown("---")

st.caption(
    "Built with Streamlit + LangChain + Gemini • ATS Optimized Analysis"
)