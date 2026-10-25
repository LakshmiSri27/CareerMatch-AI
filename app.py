
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CareerMatch AI",
    page_icon="💼",
    layout="wide"
)

st.title("💼 CareerMatch AI")
st.write(
    "Upload your resume, analyze your skills, "
    "and discover job roles that match your profile."
)

# Load the job catalogue
@st.cache_data
def load_jobs():
    return pd.read_csv("jobs.csv").fillna("")

try:
    jobs = load_jobs()
except Exception as error:
    st.error(f"Could not load jobs.csv: {error}")
    st.stop()

# Create a skill dictionary from the job dataset
ALL_SKILLS = sorted({
    skill.strip()
    for skills in jobs["Skills"]
    for skill in str(skills).split(";")
    if skill.strip()
}, key=str.lower)

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()
    except Exception as error:
        st.error(f"Unable to read the PDF: {error}")
        return ""

def detect_skills(text):
    found = []
    text_lower = text.lower()

    for skill in ALL_SKILLS:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-z0-9])"
        )
        if re.search(pattern, text_lower):
            found.append(skill)

    return found

def recommend_jobs(candidate_text, candidate_skills):
    job_texts = (
        jobs["Role"].astype(str) + " "
        + jobs["Category"].astype(str) + " "
        + jobs["Skills"].astype(str).str.replace(";", " ", regex=False)
        + " " + jobs["Description"].astype(str)
    ).tolist()

    query = candidate_text + " " + " ".join(candidate_skills)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(job_texts + [query])

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    ).flatten()

    candidate_set = {s.lower() for s in candidate_skills}
    coverage = []
    matching_skills = []
    missing_skills = []

    for skills in jobs["Skills"]:
        required = [
            skill.strip()
            for skill in str(skills).split(";")
            if skill.strip()
        ]
        required_set = {s.lower() for s in required}

        matched = [
            skill for skill in required
            if skill.lower() in candidate_set
        ]
        missing = [
            skill for skill in required
            if skill.lower() not in candidate_set
        ]

        coverage.append(
            len(matched) / len(required) if required else 0
        )
        matching_skills.append(", ".join(matched) or "None detected")
        missing_skills.append(", ".join(missing) or "None listed")

    results = jobs.copy()
    results["Text Similarity (%)"] = (similarity * 100).round(1)
    results["Skill Coverage (%)"] = (
        pd.Series(coverage).to_numpy() * 100
    ).round(1)

    # Hybrid score: combine text similarity and skill coverage
    results["Match Score (%)"] = (
        0.6 * results["Text Similarity (%)"]
        + 0.4 * results["Skill Coverage (%)"]
    ).round(1)

    results["Matching Skills"] = matching_skills
    results["Skills to Develop"] = missing_skills

    return results.sort_values(
        "Match Score (%)",
        ascending=False
    )

# Sidebar controls
st.sidebar.header("Your profile")

uploaded_resume = st.sidebar.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

manual_skills = st.sidebar.text_area(
    "Or enter your skills",
    placeholder="Python, SQL, Pandas, Machine Learning"
)

categories = ["All Categories"] + sorted(
    jobs["Category"].unique().tolist()
)

selected_category = st.sidebar.selectbox(
    "Job category",
    categories
)

number_of_jobs = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=len(jobs),
    value=min(10, len(jobs))
)

# Analyze resume
resume_text = ""

if uploaded_resume:
    resume_text = extract_pdf_text(uploaded_resume)

    if not resume_text:
        st.warning(
            "No readable text was found. Scanned PDFs may require OCR."
        )

candidate_text = " ".join(
    part for part in [resume_text, manual_skills] if part.strip()
)

candidate_skills = detect_skills(candidate_text)

# Main dashboard
overview, resume_tab, recommendations_tab, catalogue_tab = st.tabs([
    "🏠 Overview",
    "📄 Resume Analysis",
    "🎯 Job Recommendations",
    "📚 Job Catalogue"
])

with overview:
    st.subheader("Your career dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Job profiles", len(jobs))
    c2.metric("Job categories", jobs["Category"].nunique())
    c3.metric("Skills detected", len(candidate_skills))

    st.markdown("### How it works")
    st.write(
        "1. Upload a PDF resume or enter skills manually.\n"
        "2. Extract text and identify skills from the catalogue.\n"
        "3. Compare your profile with job requirements using TF-IDF "
        "and cosine similarity.\n"
        "4. Review matching skills and potential skill gaps."
    )

with resume_tab:
    st.subheader("Resume analysis")

    if uploaded_resume and resume_text:
        st.success("Resume text extracted successfully.")

        c1, c2 = st.columns(2)
        c1.metric("Words extracted", len(resume_text.split()))
        c2.metric("Skills identified", len(candidate_skills))

        st.markdown("#### Detected skills")
        if candidate_skills:
            st.write(", ".join(candidate_skills))
        else:
            st.info(
                "No catalogue skills detected. Try entering skills manually."
            )

        with st.expander("View extracted resume text"):
            st.text_area(
                "Resume text",
                resume_text[:10000],
                height=250
            )

        st.download_button(
            "Download extracted text",
            data=resume_text,
            file_name="resume_text.txt",
            mime="text/plain"
        )

    elif manual_skills.strip():
        st.write("Skills detected from your manual input:")
        st.write(", ".join(candidate_skills) or "No catalogue skills found.")
    else:
        st.info("Upload a PDF resume or enter your skills in the sidebar.")

with recommendations_tab:
    st.subheader("Recommended job roles")

    if not candidate_text.strip():
        st.info(
            "Upload a resume or enter your skills in the sidebar "
            "to generate recommendations."
        )
    else:
        results = recommend_jobs(candidate_text, candidate_skills)

        if selected_category != "All Categories":
            results = results[
                results["Category"] == selected_category
            ]

        results = results.head(number_of_jobs)

        if results.empty:
            st.warning("No jobs match the selected category.")
        else:
            best = results.iloc[0]

            c1, c2 = st.columns(2)
            c1.metric("Recommended roles", len(results))
            c2.metric(
                "Top matching role",
                best["Role"],
                f'{best["Match Score (%)"]:.1f}% text/skill score'
            )

            st.caption(
                "Scores are heuristic similarity indicators, "
                "not probabilities of getting hired."
            )

            chart = results.set_index("Role")[["Match Score (%)"]]
            st.bar_chart(chart)

            for _, job in results.iterrows():
                with st.container(border=True):
                    st.subheader(job["Role"])
                    st.caption(
                        f'{job["Category"]} · {job["Experience Level"]}'
                    )
                    st.write(job["Description"])
                    st.write(
                        f'**Match score:** {job["Match Score (%)"]:.1f}%'
                    )
                    st.progress(
                        min(max(int(job["Match Score (%)"]), 0), 100)
                    )
                    st.write(
                        "**Matching skills:**",
                        job["Matching Skills"]
                    )
                    st.write(
                        "**Skills to develop:**",
                        job["Skills to Develop"]
                    )

            st.download_button(
                "Download recommendations as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="job_recommendations.csv",
                mime="text/csv"
            )

with catalogue_tab:
    st.subheader("Explore job profiles")
    st.dataframe(jobs, use_container_width=True, hide_index=True)

    st.download_button(
        "Download job catalogue",
        data=jobs.to_csv(index=False).encode("utf-8"),
        file_name="jobs_catalogue.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "CareerMatch AI | Portfolio prototype | "
    "Uses a curated sample catalogue, not live job vacancies."
)
