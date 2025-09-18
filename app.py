# --------------------- Full Optimized app.py ---------------------
import streamlit as st
import pandas as pd
import numpy as np
import difflib
import io
import re
import fitz
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

from chatbot import get_response
from recommender import recommend
from predict_career import predict_career

# ---------------- Streamlit Page Settings ----------------
st.set_page_config(page_title="Career Guidance AI", layout="centered")
st.title("Career Guidance AI")
st.caption("Describe your interests/skills and get career suggestions, courses, and chatbot help.")

# ---------------- Load dataset ----------------
df = pd.read_csv("generated_dataset.csv")
career_names = df['Career'].unique()

# ---------------- Career Info & Courses ----------------
career_info = {
    "Software Engineer": {"description": "Designs, develops, tests and maintains software applications.",
                          "next_steps": ["Learn Python/Java/C++", "Build personal projects", "Study algorithms and data structures"]},
    "Data Scientist": {"description": "Analyzes large datasets to extract insights and build predictive models.",
                       "next_steps": ["Learn Python/R", "Study statistics & ML", "Work on Kaggle projects"]},
    "AI/ML Engineer": {"description": "Designs, trains and deploys machine learning and deep learning models.",
                       "next_steps": ["Learn TensorFlow/PyTorch", "Practice with real datasets", "Study model deployment"]},
    "UX/UI Designer": {"description": "Designs intuitive interfaces and user experiences for digital products.",
                       "next_steps": ["Master Figma/Sketch", "Build prototypes", "Study human-centered design"]},
    "Digital Marketer": {"description": "Promotes products and brands online using social, search, and ads.",
                         "next_steps": ["Learn SEO/SEM", "Run small ad campaigns", "Analyze marketing metrics"]},
    "Full Stack Developer": {"description": "Works across front-end and back-end to deliver end-to-end solutions.","next_steps": ["Combine front-end & back-end skills", "Build full-stack projects", "Deploy apps to cloud"]},
    "Data Scientist": {"description": "Analyzes large datasets to extract insights and build predictive models.","next_steps": ["Learn Python/R", "Study statistics & ML", "Work on Kaggle projects"]},
    "Data Analyst": {"description": "Interprets data, builds dashboards, and supports decision-making with analytics.","next_steps": ["Learn SQL, Excel, Python", "Learn visualization (Tableau/PowerBI)", "Build analytic dashboards"]},
    "AI/ML Engineer": {"description": "Designs, trains and deploys machine learning and deep learning models.","next_steps": ["Learn TensorFlow/PyTorch", "Practice with real datasets", "Study model deployment"]},
    "Research Scientist (AI/ML)": {"description": "Conducts original research to advance algorithms and models.","next_steps": ["Pursue advanced degrees or research projects", "Read/implement papers", "Publish and collaborate"]},
    "Computer Vision Engineer": {"description": "Builds systems that interpret images and video.","next_steps": ["Study convolutional networks", "Work with OpenCV and deep learning", "Build object detection projects"]},
    "NLP Engineer": {"description": "Builds applications that understand and generate human language.","next_steps": ["Learn transformers and NLP libraries", "Work on text projects", "Study language modeling"]},
    "DevOps Engineer": {"description": "Automates and manages CI/CD pipelines and infrastructure.","next_steps": ["Learn Docker, Kubernetes", "Understand CI/CD (Jenkins/GitHub Actions)", "Practice infra-as-code"]},
    "Cloud Engineer": {"description": "Designs and maintains cloud infrastructure and services.","next_steps": ["Learn AWS/Azure/GCP", "Get cloud certifications", "Deploy scalable services"]},
    "Site Reliability Engineer": {"description": "Improves reliability, scalability, and performance of systems.","next_steps": ["Learn monitoring tools", "Understand capacity planning", "Practice incident response"]},
    "Security Engineer": {"description": "Secures systems and applications against threats and vulnerabilities.","next_steps": ["Learn network security", "Practice penetration testing", "Obtain security certifications"]},
    "Cybersecurity Analyst": {"description": "Monitors, detects and responds to cyber threats for organizations.","next_steps": ["Study SIEM tools", "Learn incident response", "Practice threat hunting"]},
    "Blockchain Developer": {"description": "Builds decentralized applications and smart contracts.","next_steps": ["Learn Solidity or Rust", "Understand blockchain protocols", "Build and deploy smart contracts"]},
    "Mobile Developer (iOS/Android)": {"description": "Develops native or cross-platform mobile applications.","next_steps": ["Learn Swift/Kotlin or Flutter/React Native", "Build mobile apps", "Understand app lifecycle & deployment"]},
    "Embedded Systems Engineer": {"description": "Works on software for hardware devices and IoT systems.","next_steps": ["Learn C/C++", "Work with microcontrollers", "Practice real-time programming"]},
    "Hardware Engineer": {"description": "Designs electronic circuits and hardware components.","next_steps": ["Study electronics", "Practice PCB design", "Work on hardware prototypes"]},
    "Product Manager": {"description": "Defines product vision, coordinates teams, and ensures delivery of valuable products.","next_steps": ["Learn product strategy and roadmapping", "Talk to users", "Practice prioritization and metrics tracking"]},
    "Project Manager": {"description": "Coordinates projects, timelines, resources, and stakeholders to meet goals.","next_steps": ["Learn project management methodologies", "Gain experience leading teams", "Get PMP or similar certification"]},
    "UX Researcher": {"description": "Studies users to inform product and design decisions.","next_steps": ["Learn user research methods", "Run usability tests", "Translate insights into design requirements"]},
    "UX/UI Designer": {"description": "Designs intuitive interfaces and user experiences for digital products.","next_steps": ["Master Figma/Sketch", "Build prototypes", "Study human-centered design"]},
    "Graphic Designer": {"description": "Creates visual assets for branding, marketing, and communication.","next_steps": ["Learn Adobe Suite", "Build a portfolio", "Practice typography and composition"]},
    "Visual Designer": {"description": "Focuses on the aesthetics and visuals of product interfaces and marketing material.","next_steps": ["Study color theory", "Practice UI layouts", "Build brand-style guides"]},
    "Motion Designer": {"description": "Creates animated content for video, web, and apps.","next_steps": ["Learn After Effects/animation tools", "Build short animated pieces", "Study timing and storytelling"]},
    "3D Artist / Animator": {"description": "Models and animates 3D content for film, games, and visualization.","next_steps": ["Learn Blender/Maya", "Build 3D portfolios", "Study lighting and rigging"]},
    "Game Developer": {"description": "Designs and codes video games and interactive experiences.","next_steps": ["Learn Unity/Unreal", "Build small game projects", "Learn game design principles"]},
    "Game Designer": {"description": "Designs game mechanics, systems and player experiences.","next_steps": ["Study ludology and level design", "Prototype games", "Playtest and iterate"]},
    "Sound Designer": {"description": "Creates audio assets and soundscapes for media and games.","next_steps": ["Learn audio tools (DAW)", "Practice creating sound FX and Foley", "Collaborate on small projects"]},
    "Animator": {"description": "Produces 2D/3D animation for film, television, or web.","next_steps": ["Practice animation principles", "Build reels", "Study character motion"]},
    "Content Writer": {"description": "Creates written content for blogs, marketing, documentation and more.","next_steps": ["Improve writing skills", "Learn SEO basics", "Build writing samples and a portfolio"]},
    "Technical Writer": {"description": "Produces technical documentation like manuals, APIs, and guides.","next_steps": ["Learn documentation tools", "Practice clear technical communication", "Work with engineering teams"]},
    "Journalist": {"description": "Reports on news, writes stories, and investigates topics for media outlets.","next_steps": ["Practice reporting and interviewing", "Write sample articles", "Intern at media outlets"]},
    "Public Relations Specialist": {"description": "Manages public image and communications for organizations or individuals.","next_steps": ["Learn media relations", "Develop press materials", "Practice crisis communication"]},
    "Digital Marketer": {"description": "Promotes products and brands online using social, search, and ads.","next_steps": ["Learn SEO/SEM", "Run small ad campaigns", "Analyze marketing metrics"]},
    "Social Media Manager": {"description": "Creates and manages social content and community engagement.","next_steps": ["Learn content strategy", "Practice community management", "Analyze social metrics"]},
    "Marketing Analyst": {"description": "Analyzes data to measure marketing effectiveness and customer behavior.","next_steps": ["Learn analytics tools", "Study A/B testing", "Create marketing dashboards"]},
    "Sales Specialist": {"description": "Sells products or services and builds customer relationships.","next_steps": ["Practice communication and negotiation", "Understand product value", "Develop CRM skills"]},
    "Customer Success Manager": {"description": "Ensures customers achieve value and success with products or services.","next_steps": ["Learn onboarding techniques", "Gather customer feedback", "Build relationship management skills"]},
    "Business Analyst": {"description": "Analyzes business processes and recommends improvements and solutions.","next_steps": ["Learn requirements elicitation", "Practice process mapping", "Work with stakeholders"]},
    "Management Consultant": {"description": "Solves business problems and advises organizations on strategy and operations.","next_steps": ["Develop problem-solving frameworks", "Practice case interviews", "Gain domain knowledge"]},
    "Entrepreneur / Founder": {"description": "Starts and scales new businesses, often leading product and strategy.","next_steps": ["Validate ideas with customers", "Create business plans", "Learn fundraising basics"]},
    "Product Designer": {"description": "Designs end-to-end product experiences balancing UX and business needs.","next_steps": ["Practice prototyping", "Learn user research", "Work cross-functionally with engineers"]},
    "HR Specialist": {"description": "Manages recruitment, benefits, and employee relations.","next_steps": ["Learn HR processes", "Understand labor law basics", "Practice interviewing"]},
    "Recruiter": {"description": "Finds and hires talent for companies, building candidate pipelines.","next_steps": ["Practice sourcing and interviewing", "Learn hiring tools", "Build candidate networks"]},
    "Training & Development Specialist": {"description": "Designs and delivers employee training programs.","next_steps": ["Learn instructional design", "Create training materials", "Measure learning outcomes"]},
    "Accountant": {"description": "Prepares financial records, taxes and ensures compliance.","next_steps": ["Learn accounting basics", "Gain experience with accounting software", "Consider CPA/CA certification"]},
    "Financial Analyst": {"description": "Evaluates financial performance and supports investment decisions.","next_steps": ["Learn financial modeling", "Study valuation", "Practice Excel and SQL"]},
    "Controller / Finance Manager": {"description": "Oversees accounting, financial reporting and internal controls.","next_steps": ["Gain experience in accounting/finance", "Study financial regulations", "Lead reporting processes"]},
    "Investment Banker": {"description": "Advises clients on investments, mergers, and capital raising.","next_steps": ["Study finance and investment banking", "Build financial modeling skills", "Intern with investment banks"]},
    "Auditor": {"description": "Examines financial records to ensure accuracy and compliance.","next_steps": ["Learn auditing standards", "Gain practical auditing experience", "Pursue CPA/CA certification"]},
    "Lawyer": {"description": "Provides legal advice, represents clients, and prepares legal documents.","next_steps": ["Attend law school", "Pass bar exam", "Specialize in a legal domain"]},
    "Paralegal": {"description": "Supports lawyers by preparing legal documents and research.","next_steps": ["Learn legal documentation", "Gain practical experience", "Consider paralegal certification"]},
    "Civil Engineer": {"description": "Designs, constructs and maintains infrastructure projects like roads and bridges.","next_steps": ["Study civil engineering fundamentals", "Work on construction projects", "Obtain PE license"]},
    "Mechanical Engineer": {"description": "Designs mechanical systems and products, analyzing forces and energy.","next_steps": ["Learn CAD tools", "Study thermodynamics and mechanics", "Build prototypes"]},
    "Electrical Engineer": {"description": "Designs electrical circuits, power systems, and electronics.","next_steps": ["Learn circuit design", "Practice embedded systems", "Work on electronics projects"]},
    "Chemical Engineer": {"description": "Designs processes for chemical manufacturing and materials.","next_steps": ["Learn chemical process design", "Gain lab experience", "Study safety and regulations"]},
    "Civil/Structural Designer": {"description": "Focuses on structural integrity and architectural planning.","next_steps": ["Learn structural analysis", "Practice CAD software", "Work on real projects"]},
    "Architecture / Architect": {"description": "Designs buildings and urban environments balancing aesthetics and function.","next_steps": ["Study architectural design", "Master CAD and 3D modeling", "Gain internships"]},
    "Interior Designer": {"description": "Designs interior spaces for aesthetics, functionality, and comfort.","next_steps": ["Learn interior design principles", "Practice with real spaces", "Build portfolio"]},
    "Fashion Designer": {"description": "Creates clothing and accessories, balancing creativity and market needs.","next_steps": ["Study fashion design", "Practice sketching and pattern making", "Build a collection/portfolio"]},
    "Chef / Culinary Artist": {"description": "Prepares meals, plans menus, and creates culinary experiences.","next_steps": ["Learn culinary techniques", "Gain kitchen experience", "Experiment with recipes"]},
    "Nutritionist / Dietitian": {"description": "Advises on diet, nutrition, and healthy lifestyle choices.","next_steps": ["Study nutrition science", "Get certification", "Create personalized meal plans"]},
    "Physician / Doctor": {"description": "Diagnoses and treats illnesses, promoting health.","next_steps": ["Complete medical degree", "Pass licensing exams", "Specialize if desired"]},
    "Nurse": {"description": "Provides patient care and support in healthcare settings.","next_steps": ["Complete nursing program", "Gain clinical experience", "Obtain licensure"]},
    "Pharmacist": {"description": "Dispenses medications and advises on their proper use.","next_steps": ["Study pharmacy", "Gain internship experience", "Get licensed"]},
    "Psychologist / Therapist": {"description": "Provides mental health support and counseling.","next_steps": ["Complete psychology degree", "Gain clinical experience", "Obtain license"]},
    "Teacher / Educator": {"description": "Educates students and develops curriculum.","next_steps": ["Earn teaching certification", "Plan lessons and assessments", "Gain classroom experience"]},
    "Professor / Academic Researcher": {"description": "Teaches and conducts research at higher education institutions.","next_steps": ["Obtain advanced degrees", "Publish research papers", "Engage in academic networking"]},
    "Fitness Trainer / Coach": {"description": "Designs exercise programs and motivates clients for health and performance.","next_steps": ["Get certified", "Practice personal training", "Design nutrition and fitness plans"]},
    "Pilot / Aviation Professional": {"description": "Operates aircraft, ensuring safety and efficiency in travel.","next_steps": ["Obtain pilot license", "Complete flight hours", "Pass aviation exams"]},
    "Scientist / Researcher": {"description": "Conducts scientific research in a specialized field.","next_steps": ["Study your field deeply", "Perform experiments", "Publish findings"]},

}

career_courses = {
    "Software Engineer": [("CS50 (Harvard)", "https://cs50.harvard.edu"),
                          ("The Odin Project", "https://www.theodinproject.com"),
                          ("Udemy Python Bootcamp", "https://www.udemy.com/course/complete-python-bootcamp/")],
    "Data Scientist": [("IBM Data Science", "https://www.coursera.org/professional-certificates/ibm-data-science"),
                       ("Kaggle Micro-courses", "https://www.kaggle.com/learn"),
                       ("Applied ML Specialization", "https://www.coursera.org/specializations/applied-machine-learning")],
    "AI/ML Engineer": [("DeepLearning.ai", "https://www.deeplearning.ai"),
                       ("fast.ai", "https://course.fast.ai"),
                       ("TensorFlow Developer Certificate", "https://www.tensorflow.org/certificate")],
    "UX/UI Designer": [("Interaction Design Foundation", "https://www.interaction-design.org"),
                       ("Coursera UX Design", "https://www.coursera.org/specializations/ux-design"),
                       ("Figma Tutorials", "https://help.figma.com")],
    "Digital Marketer": [("Google Digital Marketing", "https://learndigital.withgoogle.com"),
                         ("Hubspot Academy", "https://academy.hubspot.com")],
                        
    "Web Development": ["freeCodeCamp", "The Odin Project", "React/Node Bootcamp", "Full Stack Open"],
    "Mobile Development": ["Flutter Bootcamp", "iOS/Swift Bootcamp", "React Native course", "Android Kotlin Bootcamp"],
    "UI/UX Design": ["Interaction Design Foundation", "Coursera UX Design", "Figma tutorials", "DesignLab courses"],
    "Digital Marketing": ["Google Digital Marketing Course", "Hubspot Academy", "Facebook Ads Blueprint", "SEO Training"],
    "Finance": ["Coursera Finance Specializations", "CFA Prep", "Wall Street Prep", "Financial Modeling Courses"],
    "Cybersecurity": ["CompTIA Security+", "Certified Ethical Hacker", "TryHackMe Labs", "Cybrary Courses"],
    "Project Management": ["PMP Certification", "Agile & Scrum Courses", "Coursera PM Specialization", "LinkedIn Learning PM"],
}


career_courses = {
    "Software Engineer": [("CS50 (Harvard)", "https://cs50.harvard.edu"),
                          ("The Odin Project", "https://www.theodinproject.com"),
                          ("Udemy Python Bootcamp", "https://www.udemy.com/course/complete-python-bootcamp/")],
    "Data Scientist": [("IBM Data Science", "https://www.coursera.org/professional-certificates/ibm-data-science"),
                       ("Kaggle Micro-courses", "https://www.kaggle.com/learn"),
                       ("Applied ML Specialization", "https://www.coursera.org/specializations/applied-machine-learning")],
    "AI/ML Engineer": [("DeepLearning.ai", "https://www.deeplearning.ai"),
                       ("fast.ai", "https://course.fast.ai"),
                       ("TensorFlow Developer Certificate", "https://www.tensorflow.org/certificate")],
    "UX/UI Designer": [("Interaction Design Foundation", "https://www.interaction-design.org"),
                       ("Coursera UX Design", "https://www.coursera.org/specializations/ux-design"),
                       ("Figma Tutorials", "https://help.figma.com")],
    "Digital Marketer": [("Google Digital Marketing", "https://learndigital.withgoogle.com"),
                         ("Hubspot Academy", "https://academy.hubspot.com")],
     "Web Development": ["freeCodeCamp", "The Odin Project", "React/Node Bootcamp", "Full Stack Open"],
    "Mobile Development": ["Flutter Bootcamp", "iOS/Swift Bootcamp", "React Native course", "Android Kotlin Bootcamp"],
    "UI/UX Design": ["Interaction Design Foundation", "Coursera UX Design", "Figma tutorials", "DesignLab courses"],
    "Digital Marketing": ["Google Digital Marketing Course", "Hubspot Academy", "Facebook Ads Blueprint", "SEO Training"],
    "Finance": ["Coursera Finance Specializations", "CFA Prep", "Wall Street Prep", "Financial Modeling Courses"],
    "Cybersecurity": ["CompTIA Security+", "Certified Ethical Hacker", "TryHackMe Labs", "Cybrary Courses"],
    "Project Management": ["PMP Certification", "Agile & Scrum Courses", "Coursera PM Specialization", "LinkedIn Learning PM"],
}

key_map = {k.strip().lower(): k for k in career_info.keys()}

# ---------------- Cached Career Prediction ----------------
@st.cache_data(show_spinner=False)
def cached_predict(text):
    return predict_career(text)

# ---------------- Cached TF-IDF & Career Vectors ----------------
@st.cache_resource
def get_tfidf_and_vectors(career_info_dict):
    career_names = list(career_info_dict.keys())
    career_texts = [
        (career_info_dict[name].get("description", "") + " " + " ".join(career_info_dict[name].get("next_steps", []))).strip()
        for name in career_names
    ]
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    career_matrix = tfidf.fit_transform(career_texts)
    return tfidf, career_matrix, career_names

tfidf, career_matrix, career_names = get_tfidf_and_vectors(career_info)

# ---------------- Multi-Interest Career Suggestions ----------------
sample_examples = [
    "I enjoy coding and building web apps",
    "I like analyzing datasets and finding patterns",
    "I am passionate about machine learning",
    "I enjoy designing interfaces and interactions",
    "I love creating graphics and visual art",
    "I enjoy teaching and mentoring",
    "I like helping people with health",
    "I want to run my own business",
    "I enjoy writing articles and stories",
    "I love cooking and creating recipes",
    "I like working with electronics and hardware",
    "I love photographing landscapes and people",
    "I like solving logical puzzles and math problems",
    "I enjoy planning events and coordinating teams",
    "I like studying environmental issues and sustainability",
    "I want to design buildings and urban plans",
    "I enjoy analyzing financial markets and investments",
    "I like testing software and finding bugs",
    "I enjoy working on embedded and IoT devices",
    "I like producing music and audio content",
    "I enjoy game design and development",
    "I want to help people with counseling and therapy",
    "I like researching scientific problems",
    "I enjoy managing product lifecycles and roadmaps",
    "I like building mobile apps and UX experiments",
    "I enjoy marketing and social media campaigns",
    "I like logistics and supply chain management",
    "I enjoy working with animals and veterinary care",
    "I like fashion design and sewing clothes",
    "I enjoy learning about aviation and piloting",
]

# ---------------- MULTI CAREER SUGGESTIONS (REVISED) ----------------
query = st.text_input("Enter your interests or skills:")
k_single = st.slider("How many suggestions?", 1, 5, 3)

if query:
    # Use cosine similarity on TF-IDF to rank all careers and pick top k_single
    query_vec = tfidf.transform([query])
    sims = cosine_similarity(query_vec, career_matrix)[0]
    top_idx = sims.argsort()[::-1][:k_single]
    st.subheader("Top Career Suggestions")
    for idx in top_idx:
        career = career_names[idx]
        info = career_info.get(career, {"description": "No description", "next_steps": []})
        st.markdown(f"### {career}")
        st.write(f"Description: {info.get('description')}")
        if info.get("next_steps"):
            st.write("**Next Steps:**")
            for step in info["next_steps"]:
                st.write(f"- {step}")
        courses = career_courses.get(career)
        if courses:
            st.write("**Recommended Courses:**")
            for c in courses:
                if isinstance(c, tuple):
                    st.markdown(f"- [{c[0]}]({c[1]})")
                else:
                    st.write(f"- {c}")
        st.markdown("---")

# ----- Multi-Interest Section -----
st.subheader("Select multiple interests (up to 5)")
selected = st.multiselect("Choose:", options=sample_examples, max_selections=5)
k_multi = st.slider("How many multi-interest suggestions?", 1, 5, 3)

if st.button("Suggest careers"):
    if not selected:
        st.warning("Select at least one interest.")
    else:
        combined = ". ".join(selected)
        query_vec = tfidf.transform([combined])
        sims = cosine_similarity(query_vec, career_matrix)[0]
        top_idx = sims.argsort()[::-1][:k_multi]
        st.subheader("Top Career Suggestions")
        for idx in top_idx:
            career = career_names[idx]
            info = career_info.get(career, {"description": "No description", "next_steps": []})
            st.markdown(f"### {career}")
            st.write(f"Description: {info.get('description')}")
            if info.get("next_steps"):
                st.write("**Next Steps:**")
                for step in info["next_steps"]:
                    st.write(f"- {step}")
            courses = career_courses.get(career)
            if courses:
                st.write("**Recommended Courses:**")
                for c in courses:
                    if isinstance(c, tuple):
                        st.markdown(f"- [{c[0]}]({c[1]})")
                    else:
                        st.write(f"- {c}")
            st.markdown("---")


# ---------------- Bulk CSV Predictions ----------------
st.subheader("Bulk CSV Predictions")
uploaded_file = st.file_uploader("Upload CSV with 'description' column", type=["csv"])
if uploaded_file:
    try:
        df_csv = pd.read_csv(uploaded_file)
        if "description" not in df_csv.columns:
            st.error("CSV must contain 'description' column.")
        else:
            df_csv['LogReg_Predicted'] = df_csv['description'].apply(lambda x: cached_predict(x)['LogisticRegression']['career'])
            df_csv['RF_Predicted'] = df_csv['description'].apply(lambda x: cached_predict(x)['RandomForest']['career'])
            st.dataframe(df_csv)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ---------------- Resume Analyzer ----------------
st.subheader("📄 Resume Analyzer")
st.info("Upload a PDF, DOCX, or TXT resume. Analyzer checks keyword coverage, sections, contact info, and career alignment.")
uploaded_resume = st.file_uploader("Upload your resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
career_options = ["Auto-detect (best match)"] + list(career_info.keys())
target_career = st.selectbox("Target career for tailoring (optional)", career_options)

def extract_resume_text(uploaded_file):
    try:
        resume_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".pdf"):
            pdf_doc = fitz.open(stream=resume_bytes, filetype="pdf")
            return "\n".join([p.get_text() for p in pdf_doc])
        elif uploaded_file.name.lower().endswith(".docx"):
            doc = Document(io.BytesIO(resume_bytes))
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return resume_bytes.decode(errors="ignore")
    except Exception as e:
        st.error(f"Could not extract resume text: {e}")
        return ""

if uploaded_resume:
    resume_text = extract_resume_text(uploaded_resume)
    
    if not resume_text.strip():
        st.warning("Resume text could not be extracted or is empty.")
    else:
        st.write("### Extracted resume preview (first 1000 characters)")
        st.text_area("Resume preview", resume_text[:1000], height=180)

        resume_vec = tfidf.transform([resume_text])
        sims = cosine_similarity(resume_vec, career_matrix)[0]
        top_n = 5
        top_idx = sims.argsort()[::-1][:top_n]
        top_matches = [(career_names[i], float(sims[i])) for i in top_idx]

        st.write("### Top career matches:")
        for name, score in top_matches:
            st.write(f"- **{name}** — similarity {score:.2f}")

        chosen_career = top_matches[0][0] if target_career == "Auto-detect (best match)" else target_career
        st.write(f"### Tailored analysis for: **{chosen_career}**")

        # TF-IDF keywords for chosen career
        feature_names = tfidf.get_feature_names_out()
        chosen_idx = career_names.index(chosen_career) if chosen_career in career_names else None
        career_top_terms = []
        if chosen_idx is not None:
            career_vec = career_matrix[chosen_idx].toarray().ravel()
            top_terms_idx = career_vec.argsort()[::-1][:30]
            career_top_terms = [feature_names[idx] for idx in top_terms_idx if career_vec[idx] > 0][:20]

        resume_lower = resume_text.lower()
        matched_keywords = [t for t in career_top_terms if t.lower() in resume_lower]
        missing_keywords = [t for t in career_top_terms if t.lower() not in resume_lower]

        st.write("**Top keywords for this career:**")
        st.write(", ".join(career_top_terms) if career_top_terms else "No keywords.")
        st.write("**Matched keywords:**")
        st.success(", ".join(matched_keywords) if matched_keywords else "None")
        st.write("**Missing keywords:**")
        st.info(", ".join(missing_keywords) if missing_keywords else "None")

        # Detect contact info
        email_re = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        phone_re = r"(\+?\d[\d\-\s]{7,}\d)"
        emails = re.findall(email_re, resume_text)
        phones = re.findall(phone_re, resume_text)

        # Check sections
        sections = {sec: bool(re.search(rf"\b{sec}\b", resume_lower)) for sec in ["experience", "education", "skills", "projects"]}
        numbers = re.findall(r"\b\d{1,4}\b", resume_text)
        quant_present = any(int(n) >= 1 for n in numbers) if numbers else False
        word_count = len(re.findall(r"\w+", resume_text))

        st.write("### Quick checks")
        st.write(f"- Word count: **{word_count}**")
        st.write(f"- Email found: **{', '.join(emails) if emails else 'No'}**")
        st.write(f"- Phone found: **{', '.join(phones) if phones else 'No'}**")
        for sec, present in sections.items():
            st.write(f"- {sec.title()}: {'Yes' if present else 'No'}")
        st.write(f"- Quantifiable numbers: {'Yes' if quant_present else 'No'}")

        # Overall alignment score
        sim_score = top_matches[0][1] if top_matches else 0.0
        keyword_score = len(matched_keywords) / len(career_top_terms) if career_top_terms else 0.0
        sections_score = sum(sections.values()) / len(sections)
        contact_score = 1.0 if (emails or phones) else 0.0
        overall = (0.5 * sim_score) + (0.25 * keyword_score) + (0.15 * sections_score) + (0.10 * contact_score)
        overall_pct = round(overall * 100, 1)
        st.markdown(f"## Overall alignment score: **{overall_pct}%**")

        # Downloadable report
        report_lines = [
            "Resume Analyzer Report",
            f"Target career: {chosen_career}",
            f"Overall score: {overall_pct}%",
            "Matched keywords:", ", ".join(matched_keywords) or "None",
            "Missing keywords:", ", ".join(missing_keywords) or "None",
        ]
        report_text = "\n".join(report_lines)
        st.download_button("Download resume report", report_text, file_name="resume_report.txt")

# ---------------- Chatbot ----------------
st.header("🤖 Chatbot Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me about careers, skills, or courses:")
if user_input:
    bot_resp = get_response(user_input)
    if bot_resp.count(".") < 3:
        bot_resp += " Here’s more detail. This information provides further guidance for your career planning."
    st.session_state.chat_history.append({"user": user_input, "bot": bot_resp})

for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
