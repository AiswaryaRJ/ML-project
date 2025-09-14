#app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import difflib
from chatbot import get_response
from recommender import recommend
# add these near the top of app.py with your other imports
import io
import re
import fitz                # PyMuPDF for PDF text extraction
from docx import Document # python-docx for .docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Career Guidance AI", layout="centered")
st.title("Career Guidance AI")
st.caption("Describe your interests/skills and get career suggestions, courses, and chatbot help.")

# ---------------- Career Info ----------------
career_info = {
     "Software Engineer": {"description": "Designs, develops, tests and maintains software applications for desktop, web, or mobile.","next_steps": ["Learn Python/Java/C++", "Build personal projects", "Study algorithms and data structures"]},
    "Backend Engineer": {"description": "Builds server-side logic, databases, and APIs powering applications.","next_steps": ["Learn Node.js/Django/Flask", "Learn relational and NoSQL databases", "Design RESTful APIs"]},
    "Frontend Engineer": {"description": "Implements user-facing UI using HTML/CSS/JavaScript and modern frameworks.","next_steps": ["Master JavaScript and React/Vue/Angular", "Practice responsive design", "Build portfolio sites"]},
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


# ✅ Career Courses mapping
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


# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("career_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=-1, keepdims=True)

def get_top_k(text, k=5):
    X = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        probs = softmax(model.decision_function(X))
    else:
        return [(model.predict(X)[0], None)]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in top_idx]

# ---------------- Examples ----------------
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
    "I love cooking and creating recipes"
    "I like working with electronics and hardware",
    "I enjoy writing articles and stories",
    "I love photographing landscapes and people",
    "I enjoy cooking and creating recipes",
    "I like solving logical puzzles and math problems",
    "I enjoy planning events and coordinating teams",
    "I like studying environmental issues and sustainability",
    "I want to design buildings and urban plans",
    "I enjoy writing articles and stories",
    "I love photographing landscapes and people",
    "I enjoy cooking and creating recipes",
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

# ---------------- Multi-Interest ----------------
# add this import at top of the file if not present:
# import difflib

# ---------------- Multi-Interest ----------------
st.subheader("Select multiple interests (up to 5)")
selected = st.multiselect("Choose:", options=sample_examples, max_selections=5)
k_multi = st.slider("How many suggestions?", 1, 5, 3)

if st.button("Suggest careers"):
    if not selected:
        st.warning("Select at least one interest.")
    else:
        combined = ". ".join(selected)
        results = get_top_k(combined, k=k_multi)

        st.subheader("Top career suggestions")

        # prepare a normalized-key -> actual-key map for lookup
        key_map = {k.strip().lower(): k for k in career_info.keys()}

        # iterate results (career_raw may not be a string)
        for career_raw, prob in results:
            # 1) make string and normalize
            career_str = str(career_raw)
            career_norm = career_str.strip().lower()

            # 2) direct normalized lookup (preferred)
            real_key = key_map.get(career_norm)

            # 3) exact-case-insensitive fallback
            if real_key is None:
                for k in career_info.keys():
                    if k.lower() == career_norm:
                        real_key = k
                        break

            # 4) fuzzy match fallback (if still None)
            if real_key is None:
                candidate = difflib.get_close_matches(career_str, list(career_info.keys()), n=1, cutoff=0.6)
                if candidate:
                    real_key = candidate[0]

            # 5) final fallback to the original string (no crash)
            if real_key is None:
                real_key = career_str  # will display whatever the model returned

            # get career info safely
            info = career_info.get(real_key, {"description": "No description available.", "next_steps": []})
            prob_text = f"{prob:.2f}" if prob is not None else "—"

            st.markdown(f"### {real_key} — Confidence: {prob_text}")
            st.write(f"**Description:** {info.get('description','No description available.')}")
            next_steps = info.get("next_steps", [])
            if next_steps:
                st.write("**Next steps:**")
                for step in next_steps:
                    st.write(f"- {step}")
            else:
                st.write("**Next steps:** N/A")

            # Show courses safely — career_courses keys might differ in naming/format.
            # Try several lookup strategies.
            courses = None
            if real_key in career_courses:
                courses = career_courses[real_key]
            elif career_norm in career_courses:
                courses = career_courses[career_norm]
            else:
                # try fuzzy match in career_courses keys
                cc_match = difflib.get_close_matches(real_key, list(career_courses.keys()), n=1, cutoff=0.6)
                if cc_match:
                    courses = career_courses.get(cc_match[0])

            if courses:
                st.write("**Recommended courses:**")
                # support both tuple(name,url) lists and plain-string lists
                for item in courses:
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        name = item[0]
                        url = item[1] if len(item) > 1 else None
                        if url:
                            st.markdown(f"- [{name}]({url})")
                        else:
                            st.write(f"- {name}")
                    else:
                        st.write(f"- {item}")
            else:
                st.write("**Recommended courses:** No data available.")

            st.markdown("---")


# ---------------- Bulk CSV ----------------
st.subheader("Bulk CSV Predictions")
uploaded_file = st.file_uploader("Upload CSV with 'description' column", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "description" not in df.columns:
            st.error("CSV must contain 'description' column.")
        else:
            df['predicted'] = df['description'].apply(lambda x: get_top_k(x, k=1)[0][0])
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ---------------- Chatbot ----------------
st.header("🤖 Chatbot Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me about careers, skills, or courses:")
if user_input:
    bot_resp = get_response(user_input)
    if len(bot_resp.split(".")) < 3:  # Ensure at least 3 sentences
        bot_resp += " Here’s more detail. This advice is meant to guide you further."
    st.session_state.chat_history.append({"user": user_input, "bot": bot_resp})

for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")


# ------------------ Resume Analyzer ------------------
st.subheader("📄 Resume Analyzer")

st.info("Upload a PDF, DOCX, or TXT resume. The analyzer checks keyword coverage, sections, contact info, and alignment with careers defined in the app. Keep private data in mind when uploading.")

uploaded_resume = st.file_uploader("Upload your resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])

# Option: allow user to choose a target career to tailor the analysis or let system auto-detect
career_options = ["Auto-detect (best match)"] + list(career_info.keys())
target_career = st.selectbox("Target career for tailoring (optional)", career_options)

if uploaded_resume:
    # read bytes once
    resume_bytes = uploaded_resume.read()

    # 1) extract text depending on filetype
    resume_text = ""
    try:
        if uploaded_resume.name.lower().endswith(".pdf"):
            # PyMuPDF extraction
            pdf_doc = fitz.open(stream=resume_bytes, filetype="pdf")
            for p in pdf_doc:
                resume_text += p.get_text()
        elif uploaded_resume.name.lower().endswith(".docx"):
            # python-docx extraction
            doc = Document(io.BytesIO(resume_bytes))
            resume_text = "\n".join([p.text for p in doc.paragraphs])
        else:
            # txt or fallback
            resume_text = resume_bytes.decode(errors="ignore")
    except Exception as e:
        st.error(f"Could not extract resume text: {e}")
        resume_text = ""

    if not resume_text.strip():
        st.warning("Resume text could not be extracted or is empty.")
    else:
        st.write("### Extracted resume preview (first 1000 characters)")
        st.text_area("Resume preview", resume_text[:1000], height=180)

        # ---------------- prepare career texts for comparison ----------------
        career_names = list(career_info.keys())
        career_texts = [
            (career_info[name].get("description", "") + " " + " ".join(career_info[name].get("next_steps", []))).strip()
            for name in career_names
        ]

        # build TF-IDF on career descriptions + resume
        all_texts = career_texts + [resume_text]
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            matrix = tfidf.fit_transform(all_texts)
        except Exception as e:
            st.error(f"Error computing TF-IDF: {e}")
            matrix = None

        if matrix is not None:
            career_matrix = matrix[:-1]
            resume_vec = matrix[-1]

            # cosine similarities
            sims = cosine_similarity(resume_vec, career_matrix)[0]
            # top matches
            top_n = 5
            top_idx = sims.argsort()[::-1][:top_n]
            top_matches = [(career_names[i], float(sims[i])) for i in top_idx]

            st.write("### Top career matches (by content similarity):")
            for name, score in top_matches:
                st.write(f"- **{name}** — similarity {score:.2f}")

            # Determine which career to tailor against
            if target_career == "Auto-detect (best match)":
                chosen_career = top_matches[0][0]
            else:
                chosen_career = target_career

            st.write(f"### Tailored analysis for: **{chosen_career}**")

            # Get TF-IDF top terms for the chosen career
            feature_names = tfidf.get_feature_names_out()
            chosen_idx = career_names.index(chosen_career) if chosen_career in career_names else None
            career_top_terms = []
            if chosen_idx is not None:
                career_vec = career_matrix[chosen_idx].toarray().ravel()
                # pick top terms (non-zero)
                top_terms_idx = career_vec.argsort()[::-1][:30]
                for idx in top_terms_idx:
                    if career_vec[idx] > 0:
                        term = feature_names[idx]
                        career_top_terms.append(term)
                # unique & limit
                career_top_terms = list(dict.fromkeys(career_top_terms))[:20]

            # analyze presence of keywords in resume
            resume_lower = resume_text.lower()
            matched_keywords = []
            missing_keywords = []
            for term in career_top_terms:
                if term.lower() in resume_lower:
                    matched_keywords.append(term)
                else:
                    missing_keywords.append(term)

            st.write("**Top keywords for this career (from description):**")
            if career_top_terms:
                st.write(", ".join(career_top_terms[:15]))
            else:
                st.write("No top keywords extracted for this career.")

            st.write("**Keywords found in resume:**")
            if matched_keywords:
                st.success(", ".join(matched_keywords[:20]))
            else:
                st.warning("No top career keywords were found in the resume.")

            st.write("**Keywords missing (consider adding if relevant):**")
            if missing_keywords:
                st.info(", ".join(missing_keywords[:20]))
            else:
                st.write("None — good coverage.")

            # ---------------- basic resume checks ----------------
            # contact info
            email_re = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
            phone_re = r"(\+?\d[\d\-\s]{7,}\d)"
            emails = re.findall(email_re, resume_text)
            phones = re.findall(phone_re, resume_text)

            # sections
            sections = {
                "experience": bool(re.search(r"\bexperience\b", resume_lower)),
                "education": bool(re.search(r"\beducation\b", resume_lower)),
                "skills": bool(re.search(r"\bskill\b", resume_lower)),
                "projects": bool(re.search(r"\bproject\b", resume_lower)),
            }

            # quantifiable achievements
            numbers = re.findall(r"\b\d{1,4}\b", resume_text)
            quant_present = any(int(n) >= 1 for n in numbers) if numbers else False

            # word count
            word_count = len(re.findall(r"\w+", resume_text))

            st.write("### Quick checks")
            st.write(f"- Word count: **{word_count}**")
            st.write(f"- Email found: **{', '.join(emails) if emails else 'No'}**")
            st.write(f"- Phone found (approx): **{', '.join(phones) if phones else 'No'}**")
            st.write("- Sections found:")
            for sec, present in sections.items():
                st.write(f"  - {sec.title()}: {'Yes' if present else 'No'}")
            st.write(f"- Quantifiable numbers found: {'Yes' if quant_present else 'No'}")

            # ---------------- scoring (simple weighted heuristic) ----------------
            sim_score = top_matches[0][1] if top_matches else 0.0
            keyword_score = (len(matched_keywords) / len(career_top_terms)) if career_top_terms else 0.0
            sections_score = (sum(sections.values()) / len(sections)) if sections else 0.0
            contact_score = 1.0 if (emails or phones) else 0.0

            overall = (0.5 * sim_score) + (0.25 * keyword_score) + (0.15 * sections_score) + (0.10 * contact_score)
            overall_pct = round(overall * 100, 1)
            st.markdown(f"## Overall alignment score: **{overall_pct}%**")

            # ---------------- suggestions ----------------
            st.write("### Suggestions to improve your resume")
            if not emails and not phones:
                st.warning("- Add contact information (email and at least one phone number).")
            if not sections["experience"]:
                st.info("- Add an 'Experience' section with bullet points showing responsibilities and achievements.")
            if not sections["skills"]:
                st.info("- Add a 'Skills' section listing technical and domain skills (e.g., Python, SQL, React).")
            if not quant_present:
                st.info("- Add quantifiable achievements (numbers/percentages) to show impact (e.g., 'reduced load time by 30%').")
            if missing_keywords:
                st.info(f"- Consider adding relevant keywords for **{chosen_career}**: {', '.join(missing_keywords[:8])}")
            if word_count < 200:
                st.info("- Your resume is short — add more detail about projects or impact (aim 300–800 words).")

            # ---------------- Downloadable report ----------------
            report_lines = [
                "Resume Analyzer Report",
                "=======================",
                f"Target career: {chosen_career}",
                f"Overall alignment score: {overall_pct}%",
                "",
                "Top career matches:",
            ]
            for name, score in top_matches:
                report_lines.append(f"- {name}: {score:.2f}")
            report_lines += ["", "Matched keywords:", ", ".join(matched_keywords) or "None"]
            report_lines += ["", "Missing keywords (suggested):", ", ".join(missing_keywords) or "None"]
            report_lines += ["", "Quick checks:"]
            report_lines += [f"- Word count: {word_count}", f"- Email found: {', '.join(emails) or 'No'}", f"- Phone found: {', '.join(phones) or 'No'}"]
            report_lines += ["Sections found:"]
            for sec, present in sections.items():
                report_lines.append(f"- {sec.title()}: {'Yes' if present else 'No'}")
            report_lines += ["", "Suggestions:"]
            if not emails and not phones:
                report_lines.append("- Add contact information (email & phone).")
            if not sections["experience"]:
                report_lines.append("- Add an Experience section with achievements.")
            if not quant_present:
                report_lines.append("- Add quantifiable metrics (numbers/percentages).")
            if missing_keywords:
                report_lines.append(f"- Add missing keywords: {', '.join(missing_keywords[:20])}")

            report_text = "\n".join(report_lines)
            st.download_button("Download resume report (.txt)", report_text, file_name="resume_report.txt")
