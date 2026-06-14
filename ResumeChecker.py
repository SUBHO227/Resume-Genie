# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:56:12 2026

@author: KIIT
"""
import streamlit as st
import tempfile
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

# =====================================
# GEMINI API KEY
# =====================================



# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Resume ATS Checker",
    page_icon="📄",
    layout="wide"
)
google_api_key = st.text_input(
    "Google API Key",
    type="password"
)

if not google_api_key:
    st.stop()

st.title("📄 Resume ATS & AI Reviewer")

st.markdown(
    "Upload your resume (PDF) and get a detailed ATS-friendly evaluation powered by Gemini."
)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.header("Configuration")

    model_name = st.selectbox(
        "Model",
        [
            "gemini-2.5-flash",
            "gemini-1.5-pro"
        ],
        index=0
    )

    st.info(
        "Make sure your API key has access to the selected model."
    )

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload your resume (PDF only)",
    type=["pdf"]
)

# =====================================
# PROCESS RESUME
# =====================================

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

        # Load PDF
        loader = PyPDFLoader(tmp_path)

        documents = loader.load()

        # Combine text
        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )

        # Prompt
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

Return the response in the following format:

# Resume Score

Overall Score: XX/100

# Category Scores

* Structure & Formatting: XX/20
* Technical Skills: XX/20
* Projects: XX/20
* Education: XX/10
* Experience: XX/15
* Achievements & Certifications: XX/15

# Strengths

1.
2.
3.
4.
5.

# Weaknesses

1.
2.
3.
4.
5.

# Skills Found

* Skill 1
* Skill 2

# Recommended Skills

* Skill 1
* Skill 2

# Missing Sections

# ATS Improvement Tips

# Final Recommendations

1.
2.
3.
4.
5.
"""
        )

        if st.button(
            "🔍 Evaluate Resume",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing resume with Gemini..."
            ):

                try:

                    chat = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=google_api_key,
                        temperature=0.2
                    )

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

                    st.markdown("---")

                    st.markdown(
                        result.content
                    )

                    st.download_button(
                        label="📥 Download Report as Markdown",
                        data=result.content,
                        file_name="resume_evaluation.md",
                        mime="text/markdown"
                    )

                except Exception as e:

                    st.error(
                        f"Error during analysis: {str(e)}"
                    )

                    st.info(
                        "Make sure your API key is valid and has sufficient quota."
                    )

    finally:

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:

    st.info(
        "👆 Upload a PDF resume to begin."
    )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "Built with Streamlit + LangChain + Gemini • Keep your API key secure!"
)
