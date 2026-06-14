# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:24:19 2026

@author: KIIT
"""

import streamlit as st
import tempfile
import os
from datetime import datetime

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
    page_title="AI Cover Letter Generator",
    page_icon="✉️",
    layout="wide"
)
google_api_key = st.text_input(
    "Google API Key",
    type="password"
)

if not google_api_key:
    st.stop()


st.title("✉️ AI ATS-Friendly Cover Letter Generator")

st.markdown(
    "Upload your resume and paste the job description to generate a personalized ATS-friendly cover letter."
)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.header("Configuration")

    model_name = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-1.5-pro"],
        index=0
    )

    st.info(
        "Recommended: Use Gemini 2.5 Flash for faster results."
    )

# =====================================
# MAIN UI
# =====================================

col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)",
        type=["pdf"]
    )

with col2:

    job_description = st.text_area(
        "Paste Job Description",
        height=300,
        placeholder="Paste the complete job description here..."
    )

# =====================================
# GENERATE BUTTON
# =====================================

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

                # ======================
                # LOAD RESUME
                # ======================

                loader = PyPDFLoader(
                    tmp_path
                )

                documents = loader.load()

                context = "\n\n".join(
                    doc.page_content
                    for doc in documents
                )

                # ======================
                # PROMPT
                # ======================

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

                # ======================
                # GEMINI MODEL
                # ======================

                chat = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3
                )

                # ======================
                # GENERATE COVER LETTER
                # ======================

                result = chat.invoke(
                    formatted_prompt
                )

                st.success(
                    "✅ Cover Letter Generated Successfully!"
                )

                st.markdown(
                    "## Generated Cover Letter"
                )

                st.markdown(
                    result.content
                )

                # ======================
                # DOWNLOAD BUTTON
                # ======================

                st.download_button(
                    label="📥 Download Cover Letter",
                    data=result.content,
                    file_name="ATS_Cover_Letter.txt",
                    mime="text/plain"
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
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
        "👆 Upload a resume and paste a job description to generate a cover letter."
    )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "Built with Streamlit + LangChain + Gemini • ATS Optimized"
)