# --------------------- Full Optimized app.py ---------------------

# --------------------- Full Optimized app.py ---------------------
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
from fuzzywuzzy import process
import pickle
from sentence_transformers import SentenceTransformer

from chatbot import get_response
from recommender import recommend
import streamlit as st
import wikipedia
import requests
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- Streamlit Page Settings ----------------
st.set_page_config(page_title="Career Guidance AI", layout="centered")
st.title("Career Guidance AI")
st.caption("Describe your interests/skills and get career suggestions, courses, and chatbot help.")

# ---------------- NLP Setup ----------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# ---------------- Load Dataset ----------------
df = pd.read_csv("generated_dataset.csv")
career_names = df['Career'].unique()

def correct_typo(text, choices=career_names, threshold=80):
    match, score = process.extractOne(text, choices)
    return match if score >= threshold else text

# ---------------- Load ML Model & Embeddings ----------------
with open("career_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ---------------- Top-3 Career Prediction ----------------
def predict_top3(user_input, top_n=3, use_embeddings=False):
    cleaned_input = preprocess_text(user_input)
    cleaned_input = correct_typo(cleaned_input)
    if use_embeddings:
        X_input = embedding_model.encode([cleaned_input])
    else:
        X_input = vectorizer.transform([cleaned_input])
    probs = model.predict_proba(X_input)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]
    results = [(model.classes_[i], round(probs[i]*100, 2)) for i in top_indices]
    return results


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
    "Software Engineer": {
        "description": "Designs, develops, tests and maintains software applications.",
        "next_steps": ["Learn Python/Java/C++", "Build personal projects", "Study algorithms and data structures"]
    },
    "Data Scientist": {
        "description": "Analyzes large datasets to extract insights and build predictive models.",
        "next_steps": ["Learn Python/R", "Study statistics & ML", "Work on Kaggle projects"]
    },
    "Graphic Designer": {
        "description": "Creates visual content for branding, websites, and marketing materials.",
        "next_steps": ["Master Photoshop/Illustrator", "Build a design portfolio", "Learn typography and color theory"]
    },
    "Mechanical Engineer": {
        "description": "Designs, builds, and maintains mechanical systems and machines.",
        "next_steps": ["Study CAD and SolidWorks", "Understand thermodynamics and mechanics", "Work on engineering projects"]
    },
    "Teacher": {
        "description": "Educates and guides students, creating lesson plans and promoting learning.",
        "next_steps": ["Earn teaching certification", "Prepare engaging lesson plans", "Develop classroom management skills"]
    },
    "Air Hostess": {
        "description": "Ensures passenger safety and comfort during flights while providing excellent service.",
        "next_steps": ["Complete flight attendant training", "Learn safety protocols", "Improve communication and customer service skills"]
    },
    "Tour Guide": {
        "description": "Leads tourists and provides information on destinations and cultural sites.",
        "next_steps": ["Study local history and culture", "Develop storytelling skills", "Learn foreign languages if needed"]
    },
    "Hydrologist": {
        "description": "Studies water resources to manage and protect water systems.",
        "next_steps": ["Study environmental science and hydrology", "Learn GIS and data analysis", "Participate in fieldwork projects"]
    },
    "Ambulance Driver": {
        "description": "Provides emergency transport and basic medical assistance to patients.",
        "next_steps": ["Obtain EMT or first aid certification", "Learn safe driving practices", "Develop crisis management skills"]
    },
    "Forensic Scientist": {
        "description": "Analyzes physical evidence to help solve crimes.",
        "next_steps": ["Study chemistry/biology forensics", "Practice lab techniques", "Learn criminal investigation procedures"]
    },
    "Pediatrician": {
        "description": "Provides medical care to children and monitors their health and growth.",
        "next_steps": ["Complete medical school and pediatric residency", "Learn child health assessment", "Stay updated on vaccinations and treatments"]
    },
    "Entrepreneur": {
        "description": "Starts and manages businesses, developing products and strategies for growth.",
        "next_steps": ["Identify business opportunities", "Learn financial and marketing basics", "Build a minimum viable product (MVP)"]
    },
    "Robotics Engineer": {
        "description": "Designs and develops robots and automation systems.",
        "next_steps": ["Study robotics and electronics", "Practice programming microcontrollers", "Work on robotic projects"]
    },
    "Advertising Executive": {
        "description": "Creates advertising campaigns to promote brands and products.",
        "next_steps": ["Learn marketing principles", "Develop creative writing and design skills", "Analyze ad performance metrics"]
    },
    "Firefighter": {
        "description": "Responds to fires and emergencies to protect people and property.",
        "next_steps": ["Complete fire safety training", "Develop physical fitness", "Learn emergency response protocols"]
    },
    "Nurse": {
        "description": "Provides patient care, administers medications, and supports medical teams.",
        "next_steps": ["Complete nursing degree and license", "Develop patient care skills", "Learn hospital procedures and recordkeeping"]
    },
    "Biotechnologist": {
        "description": "Uses biological processes to develop products in medicine, agriculture, and industry.",
        "next_steps": ["Study molecular biology and genetics", "Practice lab techniques", "Work on biotech research projects"]
    },
    "Microbiologist": {
        "description": "Studies microorganisms and their impact on health, environment, and industry.",
        "next_steps": ["Learn microbiology lab techniques", "Analyze microbial data", "Participate in research projects"]
    },
    "Doctor": {
        "description": "Diagnoses, treats, and prevents illnesses, providing patient care.",
        "next_steps": ["Complete medical school and residency", "Develop clinical skills", "Stay updated on medical research"]
    },
    "Farmer": {
        "description": "Cultivates crops and raises livestock, managing farm operations sustainably.",
        "next_steps": ["Learn modern agricultural practices", "Understand soil and crop management", "Adopt sustainable farming techniques"]
    },
    "B.Pharm Graduate": {
        "description": "Prepares and dispenses medications while studying drug interactions and effects.",
        "next_steps": ["Study pharmacy principles", "Gain experience in a pharmacy or hospital", "Learn about pharmaceutical regulations"]
    },
    "Pharm.D Graduate": {
        "description": "Manages clinical pharmacy operations and provides patient-specific medication care.",
        "next_steps": ["Complete Pharm.D degree", "Learn clinical pharmacy practices", "Stay updated on new drugs and therapies"]
    },
    "Historian": {
        "description": "Researches and analyzes historical events and societies.",
        "next_steps": ["Study history and research methods", "Work with archives and museums", "Publish historical findings"]
    },
    "Librarian": {
        "description": "Manages information resources and helps people access knowledge.",
        "next_steps": ["Learn library science", "Develop cataloging and classification skills", "Assist students and researchers"]
    },
    "Medical Coder": {
        "description": "Translates medical diagnoses into standardized codes for billing and records.",
        "next_steps": ["Learn ICD and CPT coding", "Practice with medical records", "Stay updated on healthcare regulations"]
    },
    "Business": {
        "description": "Manages and grows companies, making strategic decisions for success.",
        "next_steps": ["Learn business management principles", "Develop leadership skills", "Study finance and marketing"]
    },
    "Food Critic": {
        "description": "Reviews and evaluates food, restaurants, and culinary experiences.",
        "next_steps": ["Develop food tasting skills", "Write reviews and articles", "Study culinary trends"]
    },
    "Animal Controller": {
        "description": "Manages stray or dangerous animals and ensures public safety.",
        "next_steps": ["Learn animal handling and care", "Understand animal behavior", "Work with animal welfare organizations"]
    },
    "Veterinary Doctor": {
        "description": "Provides medical care for animals, diagnosing and treating illnesses.",
        "next_steps": ["Complete veterinary degree", "Develop clinical skills for animals", "Stay updated on animal health practices"]
    },
    "Singer": {
        "description": "Performs music vocally for audiences or recordings.",
        "next_steps": ["Practice vocal techniques", "Perform regularly", "Study music theory and genres"]
    },
    "Dancer": {
        "description": "Expresses ideas and stories through movement and choreography.",
        "next_steps": ["Train in dance styles", "Join dance groups or performances", "Create a dance portfolio"]
    },
    "Actor/Actress": {
        "description": "Portrays characters in theatre, film, or television.",
        "next_steps": ["Take acting classes", "Participate in plays or short films", "Develop a showreel"]
    },
    "Athlete": {
        "description": "Competes in sports and maintains physical fitness for performance.",
        "next_steps": ["Train regularly in chosen sport", "Maintain proper nutrition", "Participate in competitions"]
    },
    "Coach": {
        "description": "Trains and mentors athletes to improve their performance.",
        "next_steps": ["Study coaching techniques", "Develop leadership and communication skills", "Gain experience with teams"]
    },
    "Writer": {
        "description": "Creates written content for books, articles, or blogs.",
        "next_steps": ["Practice writing daily", "Read widely for inspiration", "Build a portfolio or blog"]
    },
    "Content Writer": {
        "description": "Produces online content for websites, blogs, and social media.",
        "next_steps": ["Learn SEO and digital writing", "Write regularly", "Develop a portfolio of work"]
    },
    "Model": {
        "description": "Promotes products or fashion by appearing in photoshoots, videos, or events.",
        "next_steps": ["Build a modeling portfolio", "Attend casting calls", "Work with photographers and agencies"]
    },
    "Food Inspector": {
        "description": "Ensures food safety and quality in production and restaurants.",
        "next_steps": ["Learn food safety regulations", "Inspect kitchens and products", "Report and recommend improvements"]
    },
    "Chemical Engineer": {
        "description": "Designs processes to produce chemicals, fuels, and materials efficiently.",
        "next_steps": ["Study chemical engineering principles", "Work on lab and industrial projects", "Learn process optimization"]
    },
    "Lab Technician": {
        "description": "Performs experiments, tests, and analysis in a laboratory setting.",
        "next_steps": ["Learn lab procedures and safety", "Practice analyzing samples", "Maintain accurate lab records"]
    },
    "Florist": {
        "description": "Designs and arranges flowers for events, decorations, and sales.",
        "next_steps": ["Learn flower arranging techniques", "Understand plant care", "Create a portfolio of arrangements"]
    },
    "Makeup Artist": {
        "description": "Applies cosmetics to enhance or alter appearance for events or media.",
        "next_steps": ["Study makeup techniques", "Build a portfolio", "Work with photographers and clients"]
    },
    "Content Creator": {
        "description": "Produces digital media content for social platforms and online audiences.",
        "next_steps": ["Learn video and photo editing", "Create engaging content", "Grow audience on social media"]
    },
    "Marketing Manager": {
        "description": "Plans and executes strategies to promote products and brands.",
        "next_steps": ["Study marketing and advertising", "Analyze market trends", "Lead marketing campaigns"]
    },
    "Project Manager": {
        "description": "Oversees projects from initiation to completion ensuring goals are met.",
        "next_steps": ["Learn project management tools", "Develop leadership skills", "Manage project timelines and resources"]
    },
    "HR Professional": {
        "description": "Manages recruitment, employee relations, and organizational development.",
        "next_steps": ["Learn HR policies and laws", "Develop communication and negotiation skills", "Handle recruitment and employee engagement"]
    },
    "CEO": {
        "description": "Leads and manages the overall operations of a company or organization.",
        "next_steps": ["Develop leadership skills", "Study business strategy", "Make key organizational decisions"]
    },
    "Weather Forecaster": {
        "description": "Analyzes meteorological data to predict weather conditions.",
        "next_steps": ["Study meteorology and climate science", "Learn data modeling tools", "Practice interpreting weather patterns"]
    },
    "News Reader": {
        "description": "Presents news and information to the public via TV, radio, or online platforms.",
        "next_steps": ["Develop public speaking skills", "Stay updated on current events", "Practice news reading and reporting"]
    },
    "Architect": {
        "description": "Designs buildings and structures, balancing aesthetics and functionality.",
        "next_steps": ["Study architecture and design", "Learn CAD tools", "Develop a portfolio of projects"]
    },
    "Archaeologist": {
        "description": "Studies human history through excavation and analysis of artifacts.",
        "next_steps": ["Learn archaeology methods", "Participate in field digs", "Research historical findings"]
    },
    "Psychologist": {
        "description": "Studies human behavior and mental processes to help individuals.",
        "next_steps": ["Earn psychology degree", "Learn counseling techniques", "Conduct assessments and research"]
    },
    "Psychiatrist": {
        "description": "Diagnoses and treats mental illnesses using therapy and medications.",
        "next_steps": ["Complete medical school and psychiatry residency", "Learn psychotherapy", "Stay updated on mental health research"]
    },
    "Dentist": {
        "description": "Examines, diagnoses, and treats dental issues to maintain oral health.",
        "next_steps": ["Complete dental school", "Learn dental procedures", "Practice patient care and hygiene"]
    },
    "Electrician": {
        "description": "Installs and maintains electrical systems in buildings and equipment.",
        "next_steps": ["Learn electrical systems and safety", "Gain hands-on experience", "Obtain required certifications"]
    },
    "Biologist": {
        "description": "Studies living organisms to understand life processes and ecosystems.",
        "next_steps": ["Study biology and ecology", "Conduct experiments and field research", "Publish findings or work in labs"]
    },
    "Animator": {
        "description": "Creates animations for movies, games, or advertisements.",
        "next_steps": ["Learn animation software", "Develop storyboarding skills", "Build an animation portfolio"]
    },
    "Astronomer": {
        "description": "Studies celestial objects, space, and the universe.",
        "next_steps": ["Study astrophysics or astronomy", "Learn telescopes and observation techniques", "Analyze astronomical data"]
    },
    "Geologist": {
        "description": "Studies the Earth, rocks, minerals, and natural resources.",
        "next_steps": ["Learn geology and field mapping", "Conduct sample analysis", "Participate in field studies"]
    },
    "Director": {
    "description": "Oversees the creative vision of a film or video project and directs actors and crew.",
    "next_steps": ["Study film directing", "Watch and analyze movies", "Practice directing short films"]
},

    "Cameraman": {
        "description": "Operates cameras to capture footage according to the director's vision.",
        "next_steps": ["Learn camera operation and cinematography", "Practice filming scenes", "Study lighting and framing techniques"]
    },

    "Producer": {
        "description": "Manages the production of films, including financing, scheduling, and coordination of the crew.",
        "next_steps": ["Learn film production management", "Network with industry professionals", "Assist in film projects"]
    },

    "Production Controller": {
        "description": "Oversees the logistics and schedule of a film production, ensuring smooth workflow and resource allocation.",
        "next_steps": ["Study production management", "Practice project coordination", "Gain experience in film sets"]
    },

    "Cinematographer": {
        "description": "Responsible for capturing the visual aesthetics of a film through camera work and lighting.",
        "next_steps": ["Learn cinematography techniques", "Experiment with camera angles and lighting", "Study visual storytelling"]
    },

    "Visual Editor": {
        "description": "Edits raw footage to create a polished final product, adding effects, transitions, and sound synchronization.",
        "next_steps": ["Learn video editing software (Premiere, Final Cut)", "Practice editing short films", "Study post-production techniques"]
    },

    "Mentalist": {
        "description": "Performs mental feats and illusions using psychology and observation.",
        "next_steps": ["Study human behavior and psychology", "Practice observation and memory techniques", "Develop performance skills"]
    }
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
                         ("Hubspot Academy", "https://academy.hubspot.com"),
                         ("Coursera Social Media Marketing", "https://www.coursera.org/specializations/social-media-marketing")],

    "Graphic Designer": [("Adobe Illustrator CC", "https://helpx.adobe.com/illustrator/tutorials.html"),
                         ("Canva Design School", "https://www.canva.com/learn/design-school/"),
                         ("Udemy Graphic Design Masterclass", "https://www.udemy.com/course/graphic-design-masterclass/")],

    "Mechanical Engineer": [("MIT OpenCourseWare: Mechanical Engineering", "https://ocw.mit.edu/courses/mechanical-engineering/"),
                            ("Coursera Mechanical Design", "https://www.coursera.org/learn/mechanical-design"),
                            ("SolidWorks Tutorials", "https://www.solidworks.com/learning")],

    "Teacher": [("Coursera: Teaching and Learning", "https://www.coursera.org/specializations/teaching-learning"),
                ("EdX: Foundations of Teaching", "https://www.edx.org/course/foundations-of-teaching"),
                ("Khan Academy Teacher Resources", "https://www.khanacademy.org/teachers")],

    "Air Hostess": [("IATA Cabin Crew Training", "https://www.iata.org/en/training/courses/airline/"),
                    ("Flight Attendant Training Institute", "https://www.fattraining.com/"),
                    ("Coursera Customer Service Skills", "https://www.coursera.org/learn/customer-service")],

    "Tour Guide": [("Tourism and Travel Courses - Alison", "https://alison.com/course/diploma-in-tourism"),
                   ("Coursera Cultural Tourism", "https://www.coursera.org/learn/cultural-tourism"),
                   ("National Tour Association Training", "https://www.ntaonline.com/education")],

    "Hydrologist": [("Coursera Water Resources", "https://www.coursera.org/learn/water-resources-management"),
                    ("EdX Hydrology", "https://www.edx.org/course/fundamentals-of-hydrology"),
                    ("USGS Hydrologic Resources", "https://www.usgs.gov/mission-areas/water-resources")],

    "Ambulance Driver": [("EMT Basic Training", "https://www.ems1academy.com/emt-basic-course/"),
                         ("First Aid & CPR - Red Cross", "https://www.redcross.org/take-a-class/first-aid"),
                         ("Defensive Driving for Emergency Vehicles", "https://www.nfpa.org/")],

    "Forensic Scientist": [("Forensic Science - Coursera", "https://www.coursera.org/specializations/forensic-science"),
                           ("Udemy Forensic Investigation", "https://www.udemy.com/course/forensic-investigation/"),
                           ("OpenLearn Forensics", "https://www.open.edu/openlearn/science-maths-technology/forensic-science")],

    "Pediatrician": [("Medscape Pediatrics Courses", "https://www.medscape.com/pediatrics"),
                     ("Coursera Child Health", "https://www.coursera.org/learn/child-health"),
                     ("Stanford Pediatrics Lectures", "https://med.stanford.edu/pediatrics.html")],

    "Entrepreneur": [("Y Combinator Startup School", "https://www.startupschool.org/"),
                     ("Coursera Entrepreneurship Specialization", "https://www.coursera.org/specializations/wharton-entrepreneurship"),
                     ("MIT OpenCourseWare: Entrepreneurship", "https://ocw.mit.edu/courses/sloan-school-of-management/15-390-new-enterprises-fall-2013/")],

    "Robotics Engineer": [("Coursera Robotics Specialization", "https://www.coursera.org/specializations/robotics"),
                          ("Udemy Robotics for Beginners", "https://www.udemy.com/course/robotics/"),
                          ("MIT OpenCourseWare Robotics", "https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-141-robotic-science-and-systems-i-fall-2014/")],

    "Advertising Executive": [("Hubspot Marketing Courses", "https://academy.hubspot.com/"),
                              ("Coursera Advertising Strategy", "https://www.coursera.org/learn/advertising-strategy"),
                              ("Google Ads Certification", "https://skillshop.exceedlms.com/student/path/18109")],

    "Firefighter": [("NFPA Firefighter Training", "https://www.nfpa.org/"),
                    ("Coursera Fire Safety", "https://www.coursera.org/learn/fire-safety"),
                    ("Red Cross Fire Safety", "https://www.redcross.org/take-a-class/fire-safety")],

    "Nurse": [("Coursera Nursing Courses", "https://www.coursera.org/browse/health/nursing"),
              ("MedlinePlus Nursing Resources", "https://medlineplus.gov/nursing.html"),
              ("American Nurses Association", "https://www.nursingworld.org/")],

    "Biotechnologist": [("Coursera Biotechnology Specialization", "https://www.coursera.org/specializations/biotechnology"),
                         ("MIT OpenCourseWare Biotechnology", "https://ocw.mit.edu/courses/biological-engineering/"),
                         ("Udemy Biotechnology", "https://www.udemy.com/course/biotechnology/")],

    "Microbiologist": [("Coursera Microbiology", "https://www.coursera.org/learn/microbiology"),
                       ("EdX Microbiology", "https://www.edx.org/learn/microbiology"),
                       ("OpenLearn Microbiology", "https://www.open.edu/openlearn/science-maths-technology/microbiology")],

    "Doctor": [("Medscape Clinical Courses", "https://www.medscape.com/"),
               ("Coursera Clinical Medicine", "https://www.coursera.org/learn/clinical-medicine"),
               ("Stanford Medicine Online", "https://med.stanford.edu/")],

    "Farmer": [("Alison Agriculture Courses", "https://alison.com/course/diploma-in-agriculture"),
               ("Coursera Sustainable Agriculture", "https://www.coursera.org/learn/sustainable-agriculture"),
               ("FAO eLearning", "https://elearning.fao.org/")],

    "B.Pharm Graduate": [("Coursera Pharmacology", "https://www.coursera.org/learn/pharmacology"),
                         ("Alison Pharmacy Courses", "https://alison.com/courses/pharmacy"),
                         ("MedlinePlus Drug Information", "https://medlineplus.gov/druginformation.html")],

    "Pharm.D Graduate": [("Coursera Clinical Pharmacy", "https://www.coursera.org/learn/clinical-pharmacy"),
                         ("EdX Pharmacy Courses", "https://www.edx.org/learn/pharmacy"),
                         ("Udemy Pharmacy Technician Training", "https://www.udemy.com/course/pharmacy-technician-training/")],

    "Historian": [("Coursera History Courses", "https://www.coursera.org/browse/arts-and-humanities/history"),
                  ("EdX History Programs", "https://www.edx.org/learn/history"),
                  ("Khan Academy History", "https://www.khanacademy.org/humanities/world-history")],

    "Librarian": [("Coursera Library Science", "https://www.coursera.org/learn/library-science"),
                  ("EdX Library Management", "https://www.edx.org/learn/library-management"),
                  ("American Library Association", "https://www.ala.org/educationcareers")],

    "Medical Coder": [("AAPC Medical Coding", "https://www.aapc.com/training/medical-coding/"),
                      ("Coursera Health Informatics", "https://www.coursera.org/learn/health-informatics"),
                      ("Udemy ICD-10 Coding", "https://www.udemy.com/course/icd-10-coding/")],

    "Business": [("Coursera Business Foundations", "https://www.coursera.org/specializations/wharton-business-foundations"),
                 ("MIT OpenCourseWare Business", "https://ocw.mit.edu/courses/sloan-school-of-management/"),
                 ("Harvard Online Business Courses", "https://online.hbs.edu/courses/")],

    "Food Critic": [("Udemy Food Writing", "https://www.udemy.com/course/food-writing/"),
                    ("Coursera Culinary Arts", "https://www.coursera.org/learn/culinary-arts"),
                    ("EdX Gastronomy Courses", "https://www.edx.org/learn/gastronomy")],

    "Animal Controller": [("Animal Behavior Courses - Coursera", "https://www.coursera.org/learn/animal-behavior"),
                          ("Udemy Animal Handling", "https://www.udemy.com/course/animal-handling/"),
                          ("ASPCA Training Resources", "https://www.aspca.org/")],

    "Veterinary Doctor": [("Coursera Veterinary Science", "https://www.coursera.org/specializations/veterinary-science"),
                          ("VetMed Online", "https://www.vetmedonline.com/"),
                          ("EdX Veterinary Courses", "https://www.edx.org/learn/veterinary")],

    "Singer": [("Berklee Online Vocal Courses", "https://online.berklee.edu/courses/vocals"),
               ("Coursera Music Performance", "https://www.coursera.org/learn/music-performance"),
               ("Udemy Singing Lessons", "https://www.udemy.com/course/singing-lessons/")],

    "Dancer": [("Udemy Dance Courses", "https://www.udemy.com/courses/dance/"),
               ("Coursera Dance & Movement", "https://www.coursera.org/learn/dance-movement"),
               ("DancePlug Tutorials", "https://www.danceplug.com/")],

    "Actor/Actress": [("MasterClass Acting with Natalie Portman", "https://www.masterclass.com/classes/natalie-portman-teaches-acting"),
                      ("Udemy Acting Classes", "https://www.udemy.com/course/acting-classes/"),
                      ("Coursera Performing Arts", "https://www.coursera.org/learn/performing-arts")],

    "Athlete": [("Coursera Sports Science", "https://www.coursera.org/learn/sports-science"),
                ("Udemy Fitness Training", "https://www.udemy.com/course/fitness-training/"),
                ("NASM Certified Personal Trainer", "https://www.nasm.org/certification")],

    "Coach": [("Coursera Coaching Skills", "https://www.coursera.org/learn/coaching-skills"),
              ("Udemy Sports Coaching", "https://www.udemy.com/course/sports-coaching/"),
              ("International Coaching Federation", "https://coachingfederation.org/")],

    "Writer": [("Coursera Creative Writing", "https://www.coursera.org/specializations/creative-writing"),
               ("Udemy Writing Courses", "https://www.udemy.com/courses/writing/"),
               ("OpenLearn Writing Skills", "https://www.open.edu/openlearn/skills-for-work")],

    "Content Writer": [("Hubspot Content Marketing", "https://academy.hubspot.com/courses/content-marketing"),
                       ("Coursera Content Writing", "https://www.coursera.org/learn/content-marketing"),
                       ("Udemy SEO Content Writing", "https://www.udemy.com/course/seo-content-writing/")],

    "Model": [("MasterClass Modeling with Naomi Campbell", "https://www.masterclass.com/classes/naomi-campbell-teaches-modeling"),
              ("Udemy Modeling Courses", "https://www.udemy.com/course/modeling/"),
              ("Modeling Workshop Tutorials", "https://www.modelingworkshop.com/")],

    "Food Inspector": [("Coursera Food Safety", "https://www.coursera.org/learn/food-safety"),
                       ("Alison Food Hygiene", "https://alison.com/course/diploma-in-food-hygiene"),
                       ("FDA Training Resources", "https://www.fda.gov/training")],

    "Chemical Engineer": [("MIT OpenCourseWare Chemical Engineering", "https://ocw.mit.edu/courses/chemical-engineering/"),
                          ("Coursera Chemical Process", "https://www.coursera.org/learn/chemical-process"),
                          ("Udemy Chemical Engineering", "https://www.udemy.com/course/chemical-engineering/")],

    "Lab Technician": [("Coursera Lab Techniques", "https://www.coursera.org/learn/lab-techniques"),
                       ("Udemy Lab Safety & Skills", "https://www.udemy.com/course/lab-safety/"),
                       ("EdX Laboratory Courses", "https://www.edx.org/learn/laboratory")],

    "Florist": [("Udemy Flower Arranging", "https://www.udemy.com/course/flower-arranging/"),
                ("Skillshare Floristry", "https://www.skillshare.com/browse/floristry"),
                ("Alison Floral Design", "https://alison.com/course/floral-design")],

    "Makeup Artist": [("Udemy Makeup Artistry", "https://www.udemy.com/course/makeup-artistry/"),
                      ("Skillshare Makeup Tutorials", "https://www.skillshare.com/browse/makeup"),
                      ("MasterClass Makeup with Bobbi Brown", "https://www.masterclass.com/classes/bobbi-brown-teaches-makeup")],

    "Content Creator": [("YouTube Creator Academy", "https://creatoracademy.youtube.com/"),
                        ("Skillshare Content Creation", "https://www.skillshare.com/browse/content-creation"),
                        ("Udemy Video Content Creation", "https://www.udemy.com/course/video-content-creation/")],

    "Marketing Manager": [("Coursera Marketing Management", "https://www.coursera.org/learn/marketing-management"),
                          ("Hubspot Marketing Courses", "https://academy.hubspot.com/courses/marketing"),
                          ("Google Digital Marketing", "https://learndigital.withgoogle.com/digitalgarage")],

    "Project Manager": [("Coursera Project Management", "https://www.coursera.org/specializations/project-management"),
                        ("Udemy PMP Certification", "https://www.udemy.com/course/pmp-exam-prep/"),
                        ("PMI Resources", "https://www.pmi.org/")],

    "HR Professional": [("Coursera Human Resources", "https://www.coursera.org/learn/human-resources"),
                        ("Udemy HR Management", "https://www.udemy.com/course/hr-management/"),
                        ("SHRM HR Resources", "https://www.shrm.org/")],

    "CEO": [("Harvard Business School Online Leadership", "https://online.hbs.edu/courses/leadership-principles/"),
            ("Coursera Executive Leadership", "https://www.coursera.org/specializations/executive-leadership"),
            ("MIT Sloan Leadership Programs", "https://executive.mit.edu/leadership/")],

    "Weather Forecaster": [("Coursera Meteorology", "https://www.coursera.org/learn/meteorology"),
                           ("EdX Weather and Climate", "https://www.edx.org/learn/weather"),
                           ("NOAA Meteorology Training", "https://www.noaa.gov/")],

    "News Reader": [("Coursera Journalism Courses", "https://www.coursera.org/browse/arts-and-humanities/journalism"),
                    ("Udemy News Reporting", "https://www.udemy.com/course/news-reporting/"),
                    ("BBC Academy Journalism", "https://www.bbc.co.uk/academy")],

    "Architect": [("MIT OpenCourseWare Architecture", "https://ocw.mit.edu/courses/architecture/"),
                  ("Coursera Architecture Design", "https://www.coursera.org/specializations/architecture-design"),
                  ("EdX Architectural Design", "https://www.edx.org/learn/architecture")],

    "Archaeologist": [("Coursera Archaeology", "https://www.coursera.org/learn/archaeology"),
                      ("Udemy Archaeology Courses", "https://www.udemy.com/course/archaeology/"),
                      ("OpenLearn Archaeology", "https://www.open.edu/openlearn/history/archaeology")],

    "Psychologist": [("Coursera Psychology", "https://www.coursera.org/specializations/psychology"),
                     ("Udemy Psychology Courses", "https://www.udemy.com/course/psychology/"),
                     ("EdX Psychology", "https://www.edx.org/learn/psychology")],

    "Psychiatrist": [("Coursera Psychiatry", "https://www.coursera.org/learn/psychiatry"),
                     ("Udemy Mental Health Courses", "https://www.udemy.com/course/mental-health/"),
                     ("Medscape Psychiatry", "https://www.medscape.com/psychiatry")],

    "Dentist": [("Coursera Dentistry", "https://www.coursera.org/learn/dentistry"),
                ("Udemy Dental Courses", "https://www.udemy.com/course/dentistry/"),
                ("Colgate Oral Health Resources", "https://www.colgate.com/en-us/oral-health")],

    "Electrician": [("Udemy Electrical Courses", "https://www.udemy.com/course/electrical/"),
                    ("Coursera Electrical Engineering", "https://www.coursera.org/specializations/electrical-engineering"),
                    ("Alison Electrical Training", "https://alison.com/course/electrician-training")],

    "Biologist": [("Coursera Biology", "https://www.coursera.org/browse/life-sciences/biology"),
                  ("EdX Biology Courses", "https://www.edx.org/learn/biology"),
                  ("Khan Academy Biology", "https://www.khanacademy.org/science/biology")],

    "Animator": [("Udemy Animation Courses", "https://www.udemy.com/course/animation/"),
                 ("Coursera Character Animation", "https://www.coursera.org/learn/character-animation"),
                 ("AnimSchool Tutorials", "https://www.animschool.com/")],

    "Astronomer": [("Coursera Astronomy", "https://www.coursera.org/learn/astronomy"),
                   ("EdX Astrophysics", "https://www.edx.org/learn/astrophysics"),
                   ("NASA Online Learning", "https://www.nasa.gov/education")],

    "Geologist": [("Coursera Geology", "https://www.coursera.org/learn/geology"),
                  ("EdX Earth Sciences", "https://www.edx.org/learn/earth-sciences"),
                  ("USGS Geological Training", "https://www.usgs.gov/education")],

    "Mentalist": [("Udemy Mentalism", "https://www.udemy.com/course/mentalism/"),
                  ("Skillshare Mentalism Classes", "https://www.skillshare.com/browse/mentalism"),
                  ("MasterClass Mind Tricks", "https://www.masterclass.com/")],
    "Director": [("MasterClass Filmmaking by Martin Scorsese", "https://www.masterclass.com/classes/martin-scorsese-teaches-filmmaking"),
                ("Coursera Filmmaking Specialization", "https://www.coursera.org/specializations/filmmaking"),
                ("Udemy Film Directing", "https://www.udemy.com/course/film-directing/")],

    "Cameraman": [("Udemy Cinematography & Camera Skills", "https://www.udemy.com/course/cinematography/"),
                ("MasterClass Cinematography by Werner Herzog", "https://www.masterclass.com/classes/werner-herzog-teaches-filmmaking"),
                ("Skillshare Cinematography Classes", "https://www.skillshare.com/browse/cinematography")],

    "Producer": [("Coursera Film Production", "https://www.coursera.org/learn/film-production"),
                ("Udemy Film Producing", "https://www.udemy.com/course/film-producing/"),
                ("MasterClass Producing by Shonda Rhimes", "https://www.masterclass.com/classes/shonda-rhimes-teaches-writing-for-television")],

    "Production Controller": [("Udemy Film Production Management", "https://www.udemy.com/course/film-production-management/"),
                            ("Coursera Project Management for Film", "https://www.coursera.org/learn/project-management-film"),
                            ("Skillshare Production Coordination", "https://www.skillshare.com/browse/production")],

    "Cinematographer": [("MasterClass Cinematography by Roger Deakins", "https://www.masterclass.com/classes/roger-deakins-teaches-cinematography"),
                        ("Udemy Cinematography Techniques", "https://www.udemy.com/course/cinematography-techniques/"),
                        ("Coursera Advanced Cinematography", "https://www.coursera.org/learn/advanced-cinematography")],

    "Visual Editor": [("Udemy Video Editing with Premiere Pro", "https://www.udemy.com/course/adobe-premiere-pro-video-editing/"),
                    ("Coursera Video Editing & Post Production", "https://www.coursera.org/learn/video-editing"),
                    ("Skillshare Final Cut Pro Editing", "https://www.skillshare.com/browse/final-cut-pro")],

}

key_map = {k.strip().lower(): k for k in career_info.keys()}

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
    "I enjoy building software and solving coding challenges",
    "I enjoy building software and solving coding challenges",
    "I am passionate about analyzing data to find insights",
    "I love designing and training intelligent models",
    "I enjoy creating intuitive interfaces and user experiences",
    "I like promoting brands and running marketing campaigns",
    "I love creating graphics and visual art",
    "I enjoy designing machines and mechanical systems",
    "I enjoy teaching and mentoring students",
    "I like helping passengers feel safe and comfortable",
    "I enjoy showing people new places and sharing stories",
    "I am passionate about studying water and environmental systems",
    "I like helping people in emergency medical situations",
    "I enjoy solving crimes using scientific methods",
    "I love caring for children and monitoring their health",
    "I want to run my own business and create solutions",
    "I enjoy designing and building automated machines",
    "I like creating campaigns that attract attention",
    "I enjoy saving lives and protecting people from fires",
    "I like helping patients recover and stay healthy",
    "I enjoy developing innovative biotech solutions",
    "I love studying microorganisms and their effects",
    "I enjoy diagnosing and treating illnesses",
    "I like growing crops and caring for livestock",
    "I enjoy learning about drugs and their effects",
    "I like helping patients with complex medications",
    "I enjoy studying the past and analyzing historical events",
    "I like organizing knowledge and helping people find information",
    "I enjoy translating medical records into codes",
    "I like analyzing markets and managing organizations",
    "I enjoy tasting and reviewing culinary creations",
    "I like managing and caring for animals safely",
    "I enjoy treating animals and ensuring their health",
    "I love singing and performing music",
    "I enjoy expressing myself through dance",
    "I love performing and portraying characters",
    "I enjoy training and competing in sports",
    "I like guiding athletes to improve their skills",
    "I enjoy writing articles, stories, and essays",
    "I like creating content that engages readers",
    "I enjoy modeling and presenting fashion creatively",
    "I like ensuring food safety and hygiene",
    "I enjoy designing chemical processes and systems",
    "I like performing experiments and analyzing samples",
    "I enjoy arranging flowers and creating beautiful designs",
    "I love enhancing appearances through makeup",
    "I enjoy producing videos and creative content",
    "I like planning strategies to promote products",
    "I enjoy organizing projects and leading teams",
    "I like helping employees grow and resolving workplace issues",
    "I enjoy leading organizations and making strategic decisions",
    "I enjoy predicting weather and studying climates",
    "I like presenting news and informing the public",
    "I love directing movies and guiding creative vision",
    "I enjoy capturing scenes and framing shots with my camera",
    "I like producing films and managing production logistics",
    "I enjoy coordinating schedules and resources as a production controller",
    "I love planning and executing cinematography for visual storytelling",
    "I enjoy editing visuals and creating compelling video sequences"
    "I enjoy designing buildings and functional spaces",
    "I like exploring ancient civilizations and artifacts",
    "I enjoy understanding human behavior and emotions",
    "I like helping people with mental health challenges",
    "I enjoy caring for teeth and oral health",
    "I like working with electrical systems and circuits",
    "I enjoy studying living organisms and ecosystems",
    "I love bringing stories to life through animation",
    "I enjoy exploring the universe and celestial bodies",
    "I like studying rocks, minerals, and Earth's processes",
    "I enjoy understanding and predicting human behavior"
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


# ----- Multi-Interest Section -----

st.subheader("Select multiple interests (up to 5)")
selected = st.multiselect("Choose:", options=sample_examples, max_selections=5)
k_multi = st.slider("How many multi-interest suggestions?", 1, 5, 3)


# Career alias mapping to match career_info with career_courses
career_map = {
    "Teacher / Educator": "Teacher",
    "Software Engineer / Developer": "Software Engineer",
    # Add other aliases as needed
}

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
            # Map career to correct key in career_courses
            mapped_career = career_map.get(career, career)
            courses = career_courses.get(mapped_career, [])
            if courses:
                st.markdown(f"### Recommended courses for {career}:")
                for course_name, course_link in courses:
                    st.markdown(f"- [{course_name}]({course_link})")
            else:
                st.warning(f"No courses found for the career: '{career}'.")

# ---------------- Bulk CSV Predictions ----------------
st.subheader("Bulk CSV Predictions")
uploaded_file = st.file_uploader("Upload CSV with 'description' column", type=["csv"])
if uploaded_file:
    try:
        df_csv = pd.read_csv(uploaded_file)
        if "description" not in df_csv.columns:
            st.error("CSV must contain 'description' column.")
        else:
            # Apply cached_predict to get predictions
            df_csv['LogReg_Predicted'] = df_csv['description'].apply(lambda x: cached_predict(x)['LogisticRegression']['career'])
            df_csv['RF_Predicted'] = df_csv['description'].apply(lambda x: cached_predict(x)['RandomForest']['career'])
            
            # Map predicted careers to correct keys in career_courses
            career_map = {
                "Teacher / Educator": "Teacher",
                "Software Engineer / Developer": "Software Engineer",
                # Add other aliases as needed
            }

            df_csv['Mapped_LogReg'] = df_csv['LogReg_Predicted'].apply(lambda x: career_map.get(x, x))
            df_csv['Mapped_RF'] = df_csv['RF_Predicted'].apply(lambda x: career_map.get(x, x))

            # Optionally, show the mapped careers for clarity
            st.write("### Bulk Predictions with Mapped Careers")
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

        # TF-IDF keywords
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

        # Contact info
        email_re = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        phone_re = r"(\+?\d[\d\-\s]{7,}\d)"
        emails = re.findall(email_re, resume_text)
        phones = re.findall(phone_re, resume_text)

        # Sections
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
        report_lines

#-----------Chatbot-----------------
# ----------------- Session State -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Small curated skills fallback
skills_fallback = {
    # ---------- IT & Technical ----------
    "software engineer": ["Proficiency in programming languages like Python, Java, or C++", "Knowledge of data structures and algorithms", "Experience with software development lifecycle", "Problem-solving and debugging skills", "Collaboration and version control (Git)"],
    "data scientist": ["Strong statistics and probability knowledge", "Experience with Python/R and ML libraries", "Data visualization skills (Matplotlib, Seaborn, Tableau)", "Feature engineering and data cleaning", "Ability to communicate insights effectively"],
    "ai/ml engineer": ["Experience with ML/DL frameworks (TensorFlow, PyTorch)", "Knowledge of neural networks and model architectures", "Data preprocessing and feature selection", "Hyperparameter tuning and model evaluation", "Deploying models and scalability considerations"],
    "cybersecurity analyst": ["Knowledge of network security and protocols", "Threat detection and incident response skills", "Familiarity with security tools and firewalls", "Problem-solving and risk assessment", "Staying updated on emerging threats"],
    "devops engineer": ["Experience with CI/CD pipelines", "Knowledge of cloud services (AWS, Azure, GCP)", "Containerization and orchestration (Docker, Kubernetes)", "Automation and scripting skills", "Monitoring and troubleshooting systems"],
    "network engineer": ["Understanding of networking protocols (TCP/IP, DNS, etc.)", "Router/switch configuration and maintenance", "Network security and firewall knowledge", "Troubleshooting connectivity issues", "Documentation and network planning"],
    "web developer": ["Proficiency in HTML, CSS, JavaScript", "Backend framework knowledge (Node.js, Django, Flask)", "Database management (SQL/NoSQL)", "Responsive design and UX principles", "Problem-solving and debugging"],
    "mobile app developer": ["Proficiency in Android/iOS development", "Knowledge of Flutter or React Native", "Understanding of UI/UX design for apps", "Debugging and testing skills", "Integration with APIs and databases"],
    "cloud architect": ["Cloud platform expertise (AWS, Azure, GCP)", "Infrastructure design and deployment", "Security and compliance knowledge", "Problem-solving and scalability planning", "Collaboration with development teams"],
    "game developer": ["Proficiency in C++, C#, or Unity/Unreal Engine", "Knowledge of game physics and graphics", "Problem-solving and debugging", "Creativity in gameplay design", "Collaboration in multi-disciplinary teams"],

    # ---------- Healthcare ----------
    "doctor": ["Medical knowledge and diagnostic skills", "Patient care and empathy", "Problem-solving and critical thinking", "Time management under pressure", "Collaboration with healthcare teams"],
    "nurse": ["Patient care and monitoring", "Medication administration and record-keeping", "Communication with patients and doctors", "Emergency response skills", "Compassion and empathy"],
    "pharmacist": ["Knowledge of medications and interactions", "Prescription verification and dispensing", "Patient counseling skills", "Attention to detail", "Regulatory compliance knowledge"],
    "physiotherapist": ["Knowledge of physical therapy techniques", "Patient assessment and treatment planning", "Manual therapy and exercise prescription", "Communication and empathy", "Monitoring progress and adapting treatments"],
    "psychologist": ["Understanding of human behavior and mental health", "Counseling and active listening skills", "Analytical and research skills", "Empathy and ethical judgment", "Communication and interpersonal skills"],
    "nutritionist": ["Knowledge of diet planning and nutrition science", "Ability to assess patient dietary needs", "Communication and counseling skills", "Research and evidence-based advice", "Adaptability to different client requirements"],
    "lab technician": ["Sample collection and handling", "Knowledge of lab procedures and protocols", "Attention to detail and accuracy", "Analytical and observational skills", "Safety and hygiene compliance"],
    "radiologist": ["Expertise in imaging techniques (X-ray, MRI, CT)", "Ability to interpret medical images", "Attention to detail", "Communication with healthcare teams", "Analytical and problem-solving skills"],
    "dentist": ["Knowledge of dental procedures and oral health", "Patient care and communication", "Manual dexterity and precision", "Problem-solving for dental issues", "Knowledge of hygiene and safety protocols"],
    "veterinarian": ["Animal care and medical knowledge", "Diagnosis and treatment skills", "Communication with pet owners", "Problem-solving and decision-making", "Empathy and patience"],

    # ---------- Business & Finance ----------
    "financial analyst": ["Financial modeling and analysis", "Knowledge of accounting and economics", "Excel and data visualization skills", "Critical thinking and problem-solving", "Communication and presentation skills"],
    "accountant": ["Financial reporting and bookkeeping", "Knowledge of tax laws and compliance", "Attention to detail and accuracy", "Problem-solving and analytical skills", "Communication with clients and teams"],
    "entrepreneur": ["Business planning and strategy", "Marketing and sales skills", "Financial management and budgeting", "Leadership and decision-making", "Networking and resilience"],
    "marketing manager": ["Market research and strategy planning", "Brand management and promotion", "Team leadership and collaboration", "Communication and negotiation skills", "Analytics and campaign tracking"],
    "sales executive": ["Client relationship management", "Negotiation and persuasion skills", "Product knowledge", "Communication and networking", "Goal orientation and resilience"],
    "hr manager": ["Recruitment and talent acquisition", "Employee engagement and retention", "Conflict resolution and mediation", "Policy implementation and compliance", "Communication and organizational skills"],
    "business analyst": ["Requirement gathering and documentation", "Process modeling and improvement", "Analytical and problem-solving skills", "Communication with stakeholders", "Knowledge of tools like Excel, SQL, Tableau"],
    "project manager": ["Project planning and scheduling", "Risk management and mitigation", "Team leadership and delegation", "Budgeting and resource management", "Communication and reporting skills"],
    "management consultant": ["Business strategy and problem-solving", "Analytical and research skills", "Communication and presentation", "Project management", "Stakeholder engagement and negotiation"],
    "investment banker": ["Financial modeling and valuation", "Market research and analysis", "Client communication and negotiation", "Time management under pressure", "Analytical and decision-making skills"],

    # ---------- Creative & Arts ----------
    "actor": ["Acting and emotional expression", "Memorization and improvisation skills", "Stage or camera presence", "Collaboration with directors and cast", "Adaptability to roles and scripts"],
    "singer": ["Vocal technique and control", "Performance and stage presence", "Music theory and rhythm understanding", "Collaboration with musicians and producers", "Consistency in practice and performance"],
    "dancer": ["Strong sense of rhythm and musicality", "Physical strength, flexibility, and stamina", "Ability to memorize choreography quickly", "Expressiveness and stage presence", "Collaboration and discipline for rehearsals"],
    "musician": ["Mastery of a primary instrument or vocals", "Understanding of music theory and composition", "Collaboration and teamwork with other artists", "Stage presence and confidence", "Basic audio production and recording knowledge"],
    "photographer": ["Technical camera operation skills", "Composition and lighting knowledge", "Photo editing and post-processing", "Creativity and artistic vision", "Patience and attention to detail"],
    "graphic designer": ["Proficiency in Photoshop, Illustrator, or Canva", "Creativity and visual storytelling", "Typography and color theory knowledge", "Branding and layout design", "Collaboration and time management"],
    "fashion designer": ["Creativity and trend awareness", "Sketching and garment design", "Fabric and material knowledge", "Pattern making and sewing skills", "Collaboration and presentation"],
    "interior designer": ["Spatial planning and creativity", "Knowledge of materials and furniture", "CAD and 3D modeling skills", "Client communication and customization", "Attention to detail and aesthetics"],
    "chef": ["Food preparation and cooking techniques", "Creativity in recipe development", "Kitchen management and hygiene", "Time management under pressure", "Teamwork and leadership"],
    "writer": ["Strong writing and grammar skills", "Creativity and storytelling ability", "Research and analytical skills", "Editing and proofreading", "Discipline and time management"],

    # ---------- Education ----------
    "teacher": ["Lesson planning and curriculum development", "Effective communication and explanation", "Classroom management and discipline", "Patience and adaptability", "Assessment and feedback skills"],
    "professor": ["Expertise in subject matter", "Curriculum development and teaching", "Research and publication skills", "Communication and mentorship", "Analytical and critical thinking"],
    "tutor": ["Subject knowledge and teaching skills", "Communication and patience", "Creating learning materials", "Assessment and feedback", "Adaptability to student needs"],
    "librarian": ["Cataloging and organization skills", "Knowledge of library systems", "Research and reference skills", "Communication and assistance", "Attention to detail and information management"],
    "educational counselor": ["Guidance and advising skills", "Understanding of career paths and education", "Communication and empathy", "Problem-solving and planning", "Analytical and organizational skills"],

    # ---------- Law & Public Service ----------
    "lawyer": ["Strong research and analytical skills", "Knowledge of legal frameworks and laws", "Negotiation and advocacy", "Critical thinking and argumentation", "Client communication and ethics"],
    "judge": ["Legal knowledge and interpretation", "Decision-making and impartiality", "Analytical thinking and reasoning", "Communication and courtroom management", "Ethics and integrity"],
    "police officer": ["Law enforcement knowledge", "Physical fitness and self-defense", "Problem-solving and decision-making", "Communication and investigation skills", "Ethics and integrity"],
    "civil services officer": ["Leadership and public administration", "Policy analysis and decision-making", "Problem-solving and crisis management", "Effective communication", "Ethical judgment and integrity"],
    "firefighter": ["Fire safety and emergency response", "Physical fitness and teamwork", "Problem-solving under pressure", "Communication and quick decision-making", "Courage and resilience"],

    # ---------- Sports & Fitness ----------
    "sports coach": ["Knowledge of sports rules and techniques", "Team management and motivation", "Strategy planning and game analysis", "Physical training and conditioning", "Communication and leadership"],
    "fitness trainer": ["Knowledge of exercise techniques and programs", "Ability to motivate clients", "Nutrition and wellness guidance", "Communication and interpersonal skills", "Monitoring progress and adjustments"],
    "yoga instructor": ["Expertise in yoga postures and breathing techniques", "Guiding and motivating students", "Knowledge of anatomy and safety", "Communication and patience", "Adaptability to student needs"],
    "athlete": ["Physical fitness and endurance", "Discipline and training routine", "Strategic thinking for performance", "Teamwork and collaboration (if team sport)", "Focus and mental toughness"],

    # ---------- Service & Trades ----------
    "chef": ["Food preparation and cooking", "Creativity in recipes", "Time management", "Teamwork and coordination", "Kitchen hygiene and safety"],
    "barber": ["Hair cutting and styling skills", "Customer service and communication", "Attention to detail", "Creativity and trend awareness", "Time management"],
    "cosmetologist": ["Skincare, makeup, and hair treatment knowledge", "Customer service and communication", "Creativity and trend awareness", "Attention to hygiene and safety", "Time management"],
    "electrician": ["Knowledge of electrical systems", "Safety protocols", "Problem-solving and troubleshooting", "Technical skill with wiring and tools", "Attention to detail"],
    "plumber": ["Plumbing repair and installation skills", "Problem-solving and troubleshooting", "Knowledge of tools and materials", "Safety awareness", "Time management"],
    "mechanic": ["Vehicle repair and diagnostics", "Problem-solving skills", "Technical knowledge of engines", "Attention to detail", "Safety and efficiency"],
    "carpenter": ["Woodworking and furniture making", "Technical knowledge of materials and tools", "Attention to detail", "Creativity and design skills", "Time management"],
    "painter": ["Painting and finishing skills", "Knowledge of colors and techniques", "Attention to detail", "Creativity and precision", "Time management"],
    "driver": ["Safe driving skills", "Vehicle maintenance knowledge", "Time management and punctuality", "Navigation and route planning", "Attention and alertness"],

    # ---------- Aviation ----------
    "pilot": ["Aircraft operation knowledge", "Navigation and communication skills", "Decision-making under pressure", "Physical and mental fitness", "Attention to safety protocols"],
    "air traffic controller": ["Airspace monitoring", "Quick decision-making", "Communication with pilots", "Problem-solving under stress", "Attention to detail"],

    # ---------- Miscellaneous / Niche ----------
    "photographer": ["Camera operation", "Lighting knowledge", "Editing skills", "Creativity", "Attention to detail"],
    "journalist": ["Research and investigation", "Writing and reporting skills", "Communication", "Ethics and accuracy", "Time management"],
    "translator": ["Proficiency in languages", "Writing and interpretation skills", "Cultural understanding", "Attention to detail", "Communication"],
    "social worker": ["Empathy and listening", "Case management", "Problem-solving", "Communication", "Organizational skills"],
    "event planner": ["Organization and planning", "Negotiation and communication", "Budgeting and logistics", "Creativity", "Problem-solving"],
    "travel guide": ["Knowledge of history and culture", "Communication skills", "Navigation and planning", "Customer service", "Adaptability"],
    "journalist": ["Research and investigation", "Writing and reporting skills", "Communication", "Ethics and accuracy", "Time management"],
    
    # ---------- Healthcare ----------
    "occupational therapist": ["Patient rehabilitation", "Activity planning", "Empathy", "Communication"],
    "speech therapist": ["Speech assessment", "Therapy planning", "Patience", "Communication"],
    "anesthesiologist": ["Drug administration", "Patient monitoring", "Critical thinking", "Precision"],
    "surgeon": ["Precision", "Medical knowledge", "Decision-making", "Teamwork"],
    "paramedic": ["Emergency response", "First aid", "Quick decision-making", "Resilience"],
    "radiologist": ["Imaging interpretation", "Attention to detail", "Communication", "Medical knowledge"],
    "pharmacist": ["Drug knowledge", "Attention to detail", "Patient counseling", "Regulatory compliance"],
    "dietitian": ["Nutrition planning", "Patient counseling", "Research", "Communication"],
    "optometrist": ["Vision testing", "Patient care", "Technical knowledge", "Communication"],
    "prosthetist": ["Custom device design", "Anatomy knowledge", "Technical skill", "Patient interaction"],
    "physical therapist": ["Exercise planning", "Patient guidance", "Observation", "Motivation"],
    "chiropractor": ["Spinal adjustment", "Patient assessment", "Manual dexterity", "Communication"],
    "audiologist": ["Hearing assessment", "Patient care", "Technical knowledge", "Communication"],
    "pathologist": ["Lab analysis", "Attention to detail", "Medical knowledge", "Reporting"],
    "medical coder": ["Healthcare coding", "Attention to detail", "Regulatory knowledge", "Data entry"],
    "genetic counselor": ["Genetics knowledge", "Patient counseling", "Communication", "Ethics"],
    "clinical researcher": ["Research design", "Data analysis", "Communication", "Ethics"],
    "biomedical engineer": ["Medical device design", "Engineering knowledge", "Problem-solving", "Technical skills"],
    "epidemiologist": ["Data analysis", "Public health knowledge", "Research", "Critical thinking"],
    "nuclear medicine technologist": ["Imaging techniques", "Technical knowledge", "Safety procedures", "Patient care"],
    "orthotist": ["Custom brace design", "Anatomy knowledge", "Technical skill", "Patient guidance"],
    "cardiologist": ["Heart assessment", "Medical knowledge", "Decision-making", "Patient care"],
    "dermatologist": ["Skin assessment", "Medical knowledge", "Treatment planning", "Communication"],
    "psychiatrist": ["Mental health assessment", "Therapy planning", "Empathy", "Decision-making"],
    "neurologist": ["Brain assessment", "Medical knowledge", "Diagnosis", "Problem-solving"],
    "urologist": ["Urinary system knowledge", "Surgical skills", "Patient care", "Diagnosis"],
    "obstetrician": ["Pregnancy care", "Medical knowledge", "Surgical skills", "Communication"],
    "pediatrician": ["Child health assessment", "Medical knowledge", "Empathy", "Communication"],
    "geriatric specialist": ["Elderly care", "Medical knowledge", "Communication", "Patience"],
    "oncologist": ["Cancer treatment planning", "Medical knowledge", "Decision-making", "Patient care"],

    # ---------- Business & Finance ----------
    "auditor": ["Financial review", "Risk assessment", "Attention to detail", "Analytical skills"],
    "compliance officer": ["Regulatory knowledge", "Policy enforcement", "Problem-solving", "Communication"],
    "actuary": ["Statistics", "Risk modeling", "Analytical skills", "Software proficiency"],
    "stock broker": ["Market analysis", "Client communication", "Decision-making", "Stress management"],
    "logistics manager": ["Supply chain planning", "Coordination", "Time management", "Problem-solving"],
    "financial analyst": ["Financial modeling", "Data analysis", "Attention to detail", "Communication"],
    "investment banker": ["Market knowledge", "Negotiation", "Decision-making", "Analytical skills"],
    "marketing manager": ["Market research", "Strategy planning", "Communication", "Leadership"],
    "hr manager": ["Recruitment", "Employee relations", "Communication", "Problem-solving"],
    "business analyst": ["Requirement analysis", "Data interpretation", "Communication", "Problem-solving"],
    "project manager": ["Planning", "Resource management", "Leadership", "Risk management"],
    "supply chain analyst": ["Data analysis", "Process optimization", "Problem-solving", "Attention to detail"],
    "insurance underwriter": ["Risk assessment", "Analytical skills", "Attention to detail", "Decision-making"],
    "tax consultant": ["Tax law knowledge", "Financial analysis", "Communication", "Problem-solving"],
    "venture capitalist": ["Investment analysis", "Networking", "Decision-making", "Negotiation"],
    "real estate manager": ["Property management", "Negotiation", "Client communication", "Planning"],
    "bank teller": ["Customer service", "Cash handling", "Attention to detail", "Communication"],
    "credit analyst": ["Financial statement analysis", "Decision-making", "Attention to detail", "Risk assessment"],
    "economist": ["Data analysis", "Research", "Forecasting", "Critical thinking"],
    "insurance agent": ["Client communication", "Policy knowledge", "Sales skills", "Problem-solving"],
    "portfolio manager": ["Investment analysis", "Risk management", "Decision-making", "Communication"],
    "business consultant": ["Strategy development", "Problem-solving", "Research", "Communication"],
    "management analyst": ["Process improvement", "Data analysis", "Problem-solving", "Communication"],
    "operations manager": ["Coordination", "Leadership", "Problem-solving", "Time management"],
    "credit controller": ["Financial monitoring", "Risk assessment", "Attention to detail", "Communication"],
    "treasurer": ["Financial planning", "Budgeting", "Decision-making", "Analytical skills"],
    "fund manager": ["Investment strategy", "Market analysis", "Decision-making", "Communication"],
    "equity analyst": ["Stock analysis", "Research", "Decision-making", "Report writing"],
    "mergers & acquisitions specialist": ["Negotiation", "Financial analysis", "Strategy", "Communication"],
    "procurement manager": ["Supplier negotiation", "Cost analysis", "Planning", "Coordination"],

    # ---------- Creative & Arts ----------
    "animator": ["Animation software skills", "Creativity", "Storytelling", "Attention to detail"],
    "film director": ["Leadership", "Creative vision", "Communication", "Problem-solving"],
    "set designer": ["Creativity", "Spatial planning", "Collaboration", "Project management"],
    "composer": ["Music composition", "Instrument mastery", "Creativity", "Collaboration"],
    "illustrator": ["Digital/hand illustration", "Creativity", "Typography", "Communication"],
    "photographer": ["Camera operation", "Lighting knowledge", "Editing skills", "Creativity"],
    "graphic designer": ["Design software", "Creativity", "Typography", "Branding"],
    "copywriter": ["Writing skills", "Creativity", "SEO knowledge", "Marketing understanding"],
    "fashion designer": ["Design skills", "Trend awareness", "Fabric knowledge", "Creativity"],
    "interior designer": ["Spatial planning", "Creativity", "Client communication", "Project management"],
    "video editor": ["Editing software", "Storytelling", "Attention to detail", "Creativity"],
    "sound engineer": ["Audio editing", "Mixing skills", "Attention to detail", "Creativity"],
    "makeup artist": ["Makeup techniques", "Creativity", "Client communication", "Trend awareness"],
    "voice actor": ["Vocal skills", "Acting", "Creativity", "Pronunciation"],
    "game designer": ["Game mechanics knowledge", "Creativity", "Storytelling", "Programming basics"],
    "stage actor": ["Acting", "Memorization", "Expressiveness", "Teamwork"],
    "stunt performer": ["Physical skills", "Safety awareness", "Precision", "Adaptability"],
    "dancer": ["Rhythm", "Flexibility", "Stamina", "Stage presence"],
    "illustration artist": ["Drawing", "Creativity", "Attention to detail", "Communication"],
    "content creator": ["Creativity", "Video editing", "Social media", "Communication"],
    "podcaster": ["Communication", "Audio editing", "Storytelling", "Creativity"],
    "web designer": ["HTML/CSS", "Creativity", "UX design", "Responsive design"],
    "fashion stylist": ["Fashion sense", "Creativity", "Client communication", "Trend awareness"],
    "cartoonist": ["Drawing", "Creativity", "Storytelling", "Attention to detail"],
    "screenwriter": ["Storytelling", "Creativity", "Dialogue writing", "Collaboration"],
    "producer": ["Project management", "Budgeting", "Communication", "Leadership"],
    "mural artist": ["Creativity", "Painting skills", "Spatial awareness", "Teamwork"],
    "concept artist": ["Visual design", "Creativity", "Attention to detail", "Collaboration"],
    "illustration designer": ["Illustration", "Digital tools", "Creativity", "Typography"],
    "craft artist": ["Manual skills", "Creativity", "Design", "Attention to detail"],

    # ---------- Education ----------
    "instructional designer": ["Curriculum planning", "E-learning tools", "Creativity", "Communication"],
    "special education teacher": ["Patience", "Tailored teaching", "Empathy", "Collaboration"],
    "education consultant": ["Research", "Policy knowledge", "Communication", "Planning"],
    "school principal": ["Leadership", "Administration", "Communication", "Decision-making"],
    "linguistics researcher": ["Language analysis", "Research skills", "Analytical thinking", "Writing"],
    "curriculum coordinator": ["Lesson planning", "Collaboration", "Communication", "Organization"],
    "academic advisor": ["Student counseling", "Planning", "Communication", "Problem-solving"],
    "library scientist": ["Cataloging", "Research", "Information management", "Attention to detail"],
    "tutor": ["Teaching skills", "Patience", "Adaptability", "Communication"],
    "e-learning specialist": ["Online platforms", "Instructional design", "Creativity", "Technology"],
    "teacher trainer": ["Mentoring", "Communication", "Pedagogy", "Patience"],
    "career counselor": ["Guidance skills", "Communication", "Empathy", "Problem-solving"],
    "principal secretary": ["Administration", "Coordination", "Communication", "Organization"],
    "educational researcher": ["Data analysis", "Research skills", "Report writing", "Critical thinking"],
    "language teacher": ["Linguistic knowledge", "Communication", "Patience", "Adaptability"],
    "math teacher": ["Math knowledge", "Problem-solving", "Communication", "Patience"],
    "science teacher": ["Science knowledge", "Experiment design", "Communication", "Observation"],
    "history teacher": ["Research", "Storytelling", "Communication", "Critical thinking"],
    "arts teacher": ["Creativity", "Art techniques", "Communication", "Mentoring"],
    "physical education teacher": ["Fitness knowledge", "Motivation", "Planning", "Safety"],
    "educational content writer": ["Writing", "Research", "Creativity", "Attention to detail"],
    "education administrator": ["Leadership", "Planning", "Organization", "Communication"],
    "exam coordinator": ["Scheduling", "Organization", "Communication", "Attention to detail"],
    "student support officer": ["Empathy", "Problem-solving", "Communication", "Organization"],
    "adult educator": ["Teaching skills", "Communication", "Adaptability", "Patience"],
    "online course instructor": ["Tech skills", "Communication", "Content creation", "Adaptability"],
    "reading specialist": ["Literacy skills", "Patience", "Assessment", "Communication"],
    "math tutor": ["Problem-solving", "Patience", "Communication", "Analytical skills"],
    "language tutor": ["Linguistics knowledge", "Communication", "Patience", "Adaptability"],

    # ---------- Law & Public Service ----------
    "immigration officer": ["Legal knowledge", "Communication", "Problem-solving", "Ethics"],
    "diplomat": ["Negotiation", "Cultural understanding", "Communication", "Problem-solving"],
    "forensic analyst": ["Evidence analysis", "Attention to detail", "Scientific knowledge", "Reporting"],
    "mediator": ["Conflict resolution", "Communication", "Negotiation", "Patience"],
    "policy advisor": ["Research", "Policy analysis", "Communication", "Problem-solving"],
    "law clerk": ["Legal research", "Writing", "Attention to detail", "Organization"],
    "paralegal": ["Legal knowledge", "Document preparation", "Research", "Communication"],
    "public relations officer": ["Media knowledge", "Communication", "Writing", "Problem-solving"],
    "legislative assistant": ["Research", "Policy analysis", "Writing", "Organization"],
    "government auditor": ["Financial knowledge", "Attention to detail", "Reporting", "Problem-solving"],
    "sheriff deputy": ["Law enforcement", "Observation", "Decision-making", "Physical fitness"],
    "probation officer": ["Monitoring", "Communication", "Empathy", "Documentation"],
    "public defender": ["Legal knowledge", "Communication", "Advocacy", "Research"],
    "judge's clerk": ["Legal research", "Attention to detail", "Organization", "Writing"],
    "corrections officer": ["Security", "Observation", "Communication", "Decision-making"],
    "customs officer": ["Inspection", "Attention to detail", "Communication", "Law knowledge"],
    "tax investigator": ["Financial analysis", "Investigation", "Attention to detail", "Communication"],
    "firefighter": ["Emergency response", "Physical fitness", "Teamwork", "Problem-solving"],
    "police officer": ["Law enforcement", "Observation", "Decision-making", "Communication"],
    "personal trainer": ["Fitness assessment", "Exercise planning", "Motivation", "Communication"],
    "yoga instructor": ["Yoga techniques", "Patience", "Flexibility", "Teaching skills"],
    "gym manager": ["Leadership", "Organization", "Customer service", "Fitness knowledge"],
    "nutrition coach": ["Diet planning", "Communication", "Motivation", "Research"],
    "swimming coach": ["Technique coaching", "Safety knowledge", "Patience", "Motivation"],
    "fitness model": ["Physical fitness", "Discipline", "Posing skills", "Endurance"],
    "sports psychologist": ["Mental health knowledge", "Counseling", "Communication", "Motivation"],
    "athletic trainer": ["Injury prevention", "Exercise planning", "First aid", "Observation"],
    "strength & conditioning coach": ["Training programs", "Motivation", "Anatomy knowledge", "Monitoring"],
    "marathon coach": ["Endurance training", "Planning", "Motivation", "Observation"],
    "cycling coach": ["Technique coaching", "Motivation", "Planning", "Safety knowledge"],
    "soccer coach": ["Team management", "Strategy planning", "Motivation", "Observation"],
    "basketball coach": ["Team management", "Strategy planning", "Motivation", "Observation"],
    "tennis coach": ["Technique coaching", "Patience", "Strategy", "Motivation"],
    "boxing coach": ["Technique coaching", "Fitness knowledge", "Motivation", "Observation"],
    "martial arts instructor": ["Discipline", "Technique coaching", "Safety", "Motivation"],
    "ski instructor": ["Technical skills", "Safety awareness", "Patience", "Motivation"],
    "climbing instructor": ["Safety knowledge", "Technique coaching", "Motivation", "Patience"],
    "dance fitness instructor": ["Rhythm", "Motivation", "Creativity", "Communication"],
    "sports commentator": ["Communication", "Knowledge of sport", "Quick thinking", "Writing"],
    "referee": ["Rule knowledge", "Decision-making", "Observation", "Communication"],
    "gymnastics coach": ["Technique coaching", "Motivation", "Observation", "Patience"],
    "track coach": ["Technique coaching", "Planning", "Motivation", "Observation"],
    "swim instructor": ["Swimming skills", "Safety knowledge", "Patience", "Motivation"],
    "sports analyst": ["Data analysis", "Knowledge of sport", "Reporting", "Observation"],
    "weightlifting coach": ["Strength training", "Technique coaching", "Motivation", "Safety"],
    "youth sports coordinator": ["Organization", "Motivation", "Planning", "Communication"],
    "fitness blogger": ["Content creation", "Fitness knowledge", "Communication", "Creativity"],
    "outdoor adventure guide": ["Safety knowledge", "Physical fitness", "Navigation", "Motivation"],
    "athlete scout": ["Observation", "Talent assessment", "Communication", "Knowledge of sport"],
    "electrician": ["Circuit knowledge", "Safety compliance", "Problem-solving", "Technical skills"],
    "plumber": ["Pipework knowledge", "Problem-solving", "Tool usage", "Safety"],
    "carpenter": ["Woodworking", "Measurement skills", "Creativity", "Tool handling"],
    "mechanic": ["Mechanical knowledge", "Problem-solving", "Tool usage", "Attention to detail"],
    "chef": ["Cooking skills", "Time management", "Creativity", "Organization"],
    "baker": ["Baking skills", "Precision", "Creativity", "Time management"],
    "barber": ["Hair cutting", "Customer service", "Attention to detail", "Creativity"],
    "beautician": ["Skincare knowledge", "Technique", "Customer service", "Creativity"],
    "painter": ["Painting skills", "Attention to detail", "Creativity", "Time management"],
    "welder": ["Welding techniques", "Safety compliance", "Precision", "Tool handling"],
    "plasterer": ["Plastering techniques", "Precision", "Tool usage", "Safety"],
    "bricklayer": ["Masonry skills", "Measurement", "Strength", "Precision"],
    "landscaper": ["Gardening knowledge", "Creativity", "Physical fitness", "Planning"],
    "cleaning supervisor": ["Team management", "Attention to detail", "Planning", "Organization"],
    "security guard": ["Observation", "Safety knowledge", "Communication", "Decision-making"],
    "janitor": ["Cleaning skills", "Time management", "Organization", "Attention to detail"],
    "delivery driver": ["Driving skills", "Time management", "Navigation", "Customer service"],
    "chauffeur": ["Driving skills", "Customer service", "Time management", "Navigation"],
    "tailor": ["Sewing skills", "Measurement", "Attention to detail", "Creativity"],
    "shoemaker": ["Leatherwork skills", "Tool usage", "Precision", "Creativity"],
    "receptionist": ["Communication", "Organization", "Customer service", "Multitasking"],
    "event planner": ["Organization", "Communication", "Coordination", "Problem-solving"],
    "catering manager": ["Food knowledge", "Organization", "Leadership", "Time management"],
    "housekeeping manager": ["Team management", "Organization", "Attention to detail", "Planning"],
    "laundry supervisor": ["Organization", "Efficiency", "Attention to detail", "Teamwork"],
    "photography assistant": ["Camera handling", "Organization", "Observation", "Creativity"],
    "pest control technician": ["Safety knowledge", "Problem-solving", "Chemical handling", "Observation"],
    "HVAC technician": ["Mechanical knowledge", "Problem-solving", "Tool usage", "Safety"],
    "plumbing contractor": ["Project management", "Technical knowledge", "Safety", "Problem-solving"],
    "furniture restorer": ["Woodworking", "Attention to detail", "Creativity", "Patience"],
    "commercial pilot": ["Flying skills", "Navigation", "Decision-making", "Safety procedures"],
    "flight attendant": ["Customer service", "Safety knowledge", "Communication", "Problem-solving"],
    "air traffic controller": ["Attention to detail", "Decision-making", "Communication", "Stress management"],
    "aviation mechanic": ["Mechanical skills", "Problem-solving", "Attention to detail", "Safety"],
    "drone operator": ["Drone piloting", "Navigation", "Safety knowledge", "Observation"],
    "flight instructor": ["Teaching", "Flying skills", "Communication", "Safety"],
    "cargo handler": ["Organization", "Safety knowledge", "Physical fitness", "Teamwork"],
    "airport manager": ["Leadership", "Organization", "Problem-solving", "Communication"],
    "aerospace engineer": ["Engineering knowledge", "Problem-solving", "Design skills", "Attention to detail"],
    "aircraft dispatcher": ["Coordination", "Communication", "Organization", "Decision-making"],
    "ground crew": ["Safety procedures", "Teamwork", "Attention to detail", "Physical fitness"],
    "aviation safety inspector": ["Regulations knowledge", "Observation", "Reporting", "Decision-making"],
    "aviation consultant": ["Industry knowledge", "Communication", "Problem-solving", "Analysis"],
    "flight scheduler": ["Planning", "Organization", "Attention to detail", "Coordination"],
    "aerial photographer": ["Drone/camera skills", "Observation", "Creativity", "Planning"],
    "aviation maintenance planner": ["Scheduling", "Technical knowledge", "Coordination", "Safety"],
    "airport security officer": ["Observation", "Security procedures", "Communication", "Decision-making"],
    "charter pilot": ["Flying skills", "Navigation", "Customer service", "Decision-making"],
    "flight operations analyst": ["Data analysis", "Attention to detail", "Problem-solving", "Reporting"],
    "aviation software developer": ["Programming", "Problem-solving", "System design", "Attention to detail"],
    "airline route planner": ["Data analysis", "Planning", "Decision-making", "Coordination"],
    "helicopter pilot": ["Flying skills", "Navigation", "Safety procedures", "Decision-making"],
    "aviation meteorologist": ["Weather analysis", "Communication", "Attention to detail", "Reporting"],
    "airport operations manager": ["Leadership", "Organization", "Coordination", "Decision-making"],
    "flight simulator technician": ["Technical skills", "Problem-solving", "Attention to detail", "Maintenance"],
    "aviation trainer": ["Teaching", "Communication", "Aviation knowledge", "Observation"],
    "cargo pilot": ["Flying skills", "Navigation", "Safety procedures", "Decision-making"],
    "airline customer service agent": ["Customer service", "Communication", "Problem-solving", "Organization"],
    "aviation safety trainer": ["Teaching", "Regulation knowledge", "Communication", "Observation"],
    "aircraft leasing specialist": ["Negotiation", "Market knowledge", "Communication", "Decision-making"],
    "translator": ["Language proficiency", "Cultural knowledge", "Attention to detail", "Communication"],
    "interpreter": ["Listening skills", "Language proficiency", "Quick thinking", "Communication"],
    "tour guide": ["Local knowledge", "Communication", "Storytelling", "Customer service"],
    "blogger": ["Writing", "Creativity", "Digital marketing", "Consistency"],
    "vlogger": ["Video creation", "Editing", "Creativity", "Communication"],
    "podcast host": ["Communication", "Storytelling", "Interviewing", "Content planning"],
    "event host": ["Public speaking", "Organization", "Communication", "Engagement"],
    "motivational speaker": ["Public speaking", "Empathy", "Storytelling", "Communication"],
    "life coach": ["Guidance", "Empathy", "Communication", "Problem-solving"],
    "parliament researcher": ["Research", "Writing", "Policy analysis", "Organization"],
    "translator": ["Language proficiency", "Cultural knowledge", "Attention to detail", "Communication"],
    "interpreter": ["Listening skills", "Language proficiency", "Quick thinking", "Communication"],
    "tour guide": ["Local knowledge", "Communication", "Storytelling", "Customer service"],
    "blogger": ["Writing", "Creativity", "Digital marketing", "Consistency"],
    "vlogger": ["Video creation", "Editing", "Creativity", "Communication"],
    "podcast host": ["Communication", "Storytelling", "Interviewing", "Content planning"],
    "event host": ["Public speaking", "Organization", "Communication", "Engagement"],
    "motivational speaker": ["Public speaking", "Empathy", "Storytelling", "Communication"],
    "life coach": ["Guidance", "Empathy", "Communication", "Problem-solving"],
    "career coach": ["Career guidance", "Empathy", "Communication", "Motivation"],
    "voiceover artist": ["Vocal control", "Pronunciation", "Recording skills", "Creativity"],
    "magician": ["Performance", "Creativity", "Manual dexterity", "Audience engagement"],
    "professional gamer": ["Gaming skills", "Strategy", "Focus", "Teamwork"],
    "e-sports coach": ["Game knowledge", "Strategy planning", "Motivation", "Communication"],
    "street artist": ["Creativity", "Performance", "Public interaction", "Adaptability"],
    "public speaker": ["Presentation skills", "Confidence", "Communication", "Persuasion"],
    "book author": ["Writing skills", "Creativity", "Research", "Storytelling"],
    "screenplay writer": ["Storytelling", "Dialogue writing", "Creativity", "Structure planning"],
    "film critic": ["Film analysis", "Writing skills", "Attention to detail", "Communication"],
    "cultural researcher": ["Research", "Writing", "Analysis", "Observation"],
    "museum curator": ["Artifact knowledge", "Organization", "Research", "Communication"],
    "archivist": ["Documentation", "Attention to detail", "Research", "Organization"],
    "social media manager": ["Content creation", "Analytics", "Creativity", "Communication"],
    "photography blogger": ["Photography skills", "Editing", "Writing", "Creativity"],
    "craft designer": ["Manual skills", "Creativity", "Attention to detail", "Planning"],
    "eco-tourism guide": ["Environmental knowledge", "Communication", "Planning", "Customer service"],
    "travel writer": ["Writing", "Research", "Creativity", "Storytelling"],
    "documentary filmmaker": ["Filming", "Storytelling", "Editing", "Research"],
    "street performer": ["Performance", "Creativity", "Audience engagement", "Adaptability"],
    "digital nomad entrepreneur": ["Business skills", "Digital marketing", "Adaptability", "Problem-solving"],
    "fashion designer": ["Creativity", "Trend analysis", "Sketching", "Sewing skills"],
    "interior decorator": ["Creativity", "Spatial planning", "Color sense", "Client communication"],
    "animator": ["Animation software", "Creativity", "Storytelling", "Attention to detail"],
    "comic artist": ["Drawing skills", "Storytelling", "Creativity", "Consistency"],
    "calligrapher": ["Handwriting", "Creativity", "Patience", "Design skills"],
    "ceramic artist": ["Clay modeling", "Creativity", "Patience", "Attention to detail"],
    "tattoo artist": ["Drawing skills", "Creativity", "Precision", "Hygiene awareness"],
    "graphic illustrator": ["Digital illustration", "Creativity", "Software skills", "Communication"],
    "sculptor": ["Sculpting", "Creativity", "Material knowledge", "Patience"],
    "stage designer": ["Creativity", "Spatial design", "Collaboration", "Problem-solving"],
    "costume designer": ["Creativity", "Sewing", "Research", "Collaboration"],
    "set designer": ["Creativity", "Architecture sense", "Teamwork", "Budgeting"],
    "puppeteer": ["Performance", "Creativity", "Dexterity", "Storytelling"],
    "lighting designer": ["Technical skills", "Creativity", "Collaboration", "Problem-solving"],
    "sound designer": ["Audio editing", "Creativity", "Attention to detail", "Collaboration"],
    "makeup artist": ["Makeup skills", "Creativity", "Client communication", "Precision"],
    "storyboard artist": ["Sketching", "Storytelling", "Creativity", "Time management"],
    "concept artist": ["Digital art", "Creativity", "Visualization", "Software skills"],
    "fashion stylist": ["Trend analysis", "Creativity", "Coordination", "Client communication"],
    "visual merchandiser": ["Creativity", "Marketing sense", "Spatial planning", "Presentation"],
    "photographer assistant": ["Photography skills", "Attention to detail", "Equipment knowledge", "Teamwork"],
    "art restorer": ["Art history", "Restoration techniques", "Precision", "Patience"],
    "video editor": ["Editing software", "Creativity", "Storytelling", "Attention to detail"],
    "cinematographer": ["Camera skills", "Creativity", "Lighting knowledge", "Storytelling"],
    "voice coach": ["Vocal training", "Patience", "Listening", "Communication"],
    "film set coordinator": ["Organization", "Problem-solving", "Teamwork", "Communication"],
    "choreographer": ["Creativity", "Physical fitness", "Memory", "Leadership"],
    "musical director": ["Music theory", "Leadership", "Creativity", "Collaboration"],
    "stage manager": ["Organization", "Communication", "Problem-solving", "Leadership"],
    "dance instructor": ["Physical fitness", "Teaching", "Creativity", "Patience"],
    "curriculum developer": ["Research", "Instructional design", "Writing", "Creativity"],
    "special education teacher": ["Patience", "Teaching skills", "Empathy", "Adaptability"],
    "online tutor": ["Subject knowledge", "Communication", "Tech skills", "Patience"],
    "educational consultant": ["Analysis", "Communication", "Research", "Problem-solving"],
    "school administrator": ["Organization", "Leadership", "Communication", "Planning"],
    "college counselor": ["Guidance", "Communication", "Empathy", "Planning"],
    "education researcher": ["Research", "Analysis", "Writing", "Attention to detail"],
    "academic advisor": ["Communication", "Planning", "Empathy", "Problem-solving"],
    "language instructor": ["Language skills", "Teaching", "Patience", "Communication"],
    "STEM educator": ["Subject expertise", "Teaching skills", "Creativity", "Problem-solving"],
    "art educator": ["Creativity", "Teaching skills", "Patience", "Communication"],
    "music educator": ["Music skills", "Teaching", "Patience", "Communication"],
    "science communicator": ["Research", "Writing", "Communication", "Creativity"],
    "educational content writer": ["Writing", "Research", "Creativity", "Subject knowledge"],
    "teacher trainer": ["Communication", "Training skills", "Patience", "Organization"],
    "literacy coach": ["Teaching", "Communication", "Observation", "Problem-solving"],
    "instructional coordinator": ["Planning", "Curriculum design", "Organization", "Communication"],
    "school librarian": ["Organization", "Research", "Communication", "Tech skills"],
    "education policy analyst": ["Research", "Writing", "Analysis", "Presentation"],
    "academic publisher": ["Editing", "Research", "Communication", "Organization"],
    "online course developer": ["Tech skills", "Content creation", "Instructional design", "Creativity"],
    "tutoring program manager": ["Organization", "Communication", "Planning", "Leadership"],
    "educational technologist": ["Tech skills", "Problem-solving", "Instructional design", "Creativity"],
    "e-learning specialist": ["Tech skills", "Content creation", "Instructional design", "Adaptability"],
    "school psychologist": ["Psychology knowledge", "Empathy", "Communication", "Observation"],
    "career counselor": ["Guidance", "Communication", "Empathy", "Research"],
    "language pathologist": ["Communication", "Patience", "Therapy skills", "Observation"],
    "adult educator": ["Teaching", "Communication", "Patience", "Organization"],
    "vocational trainer": ["Technical knowledge", "Teaching", "Communication", "Problem-solving"],
    "educational program evaluator": ["Research", "Analysis", "Communication", "Organization"],
    "parliament researcher": ["Research", "Writing", "Policy analysis", "Organization"],
    "policy advisor": ["Research", "Analysis", "Communication", "Strategic thinking"],
    "human rights officer": ["Research", "Empathy", "Advocacy", "Communication"],
    "urban planner": ["Planning", "Research", "Problem-solving", "Collaboration"],
    "public health officer": ["Research", "Communication", "Planning", "Analysis"],
    "nonprofit manager": ["Leadership", "Organization", "Fundraising", "Communication"],
    "diplomat": ["Negotiation", "Communication", "Cultural knowledge", "Research"],
    "legal researcher": ["Research", "Analysis", "Writing", "Attention to detail"],
    "policy analyst": ["Research", "Writing", "Data analysis", "Communication"],
    "government auditor": ["Analysis", "Attention to detail", "Reporting", "Organization"],
    "lobbyist": ["Communication", "Negotiation", "Research", "Persuasion"],
    "civil rights advocate": ["Research", "Communication", "Advocacy", "Empathy"],
    "social worker": ["Empathy", "Communication", "Problem-solving", "Organization"],
    "immigration officer": ["Communication", "Attention to detail", "Policy knowledge", "Problem-solving"],
    "municipal officer": ["Organization", "Communication", "Planning", "Problem-solving"],
    "fire inspector": ["Safety knowledge", "Attention to detail", "Inspection", "Communication"],
    "public information officer": ["Communication", "Writing", "Media knowledge", "Organization"],
    "court clerk": ["Organization", "Attention to detail", "Communication", "Problem-solving"],
    "policy coordinator": ["Organization", "Research", "Communication", "Planning"],
    "regulatory affairs officer": ["Research", "Compliance knowledge", "Attention to detail", "Communication"],
    "intelligence analyst": ["Analysis", "Research", "Problem-solving", "Attention to detail"],
    "government relations manager": ["Communication", "Networking", "Strategic thinking", "Planning"],
    "paralegal": ["Legal knowledge", "Research", "Writing", "Organization"],
    "public defender assistant": ["Research", "Legal knowledge", "Communication", "Attention to detail"],
    "tax officer": ["Accounting knowledge", "Attention to detail", "Analysis", "Communication"],
    "election officer": ["Organization", "Communication", "Attention to detail", "Problem-solving"],
    "compliance officer": ["Regulatory knowledge", "Attention to detail", "Analysis", "Communication"],
    "public policy researcher": ["Research", "Writing", "Data analysis", "Communication"],
    "humanitarian officer": ["Empathy", "Planning", "Organization", "Communication"],
    "community development officer": ["Planning", "Communication", "Organization", "Problem-solving"],
    "sports physiologist": ["Knowledge of physiology", "Analysis", "Fitness assessment", "Communication"],
    "athletic trainer": ["First aid", "Exercise planning", "Motivation", "Observation"],
    "yoga instructor": ["Flexibility", "Teaching", "Patience", "Communication"],
    "personal trainer": ["Fitness planning", "Motivation", "Exercise knowledge", "Communication"],
    "strength and conditioning coach": ["Exercise science", "Motivation", "Planning", "Analysis"],
    "sports nutritionist": ["Nutrition knowledge", "Planning", "Communication", "Analysis"],
    "martial arts instructor": ["Technique", "Discipline", "Teaching", "Motivation"],
    "swimming coach": ["Swimming skills", "Teaching", "Patience", "Motivation"],
    "fitness blogger": ["Writing", "Creativity", "Social media", "Communication"],
    "recreational therapist": ["Empathy", "Exercise knowledge", "Planning", "Motivation"],
    "climbing instructor": ["Safety knowledge", "Physical fitness", "Teaching", "Patience"],
    "ski instructor": ["Technique", "Patience", "Teaching", "Safety awareness"],
    "gym manager": ["Organization", "Leadership", "Communication", "Problem-solving"],
    "dance therapist": ["Dance skills", "Empathy", "Teaching", "Motivation"],
    "sports psychologist": ["Psychology knowledge", "Communication", "Analysis", "Empathy"],
    "referee": ["Rule knowledge", "Decision making", "Observation", "Communication"],
    "sports commentator": ["Communication", "Knowledge of sport", "Quick thinking", "Writing"],
    "surfing instructor": ["Surfing skills", "Patience", "Teaching", "Safety awareness"],
    "rowing coach": ["Technique", "Leadership", "Motivation", "Observation"],
    "triathlon coach": ["Endurance knowledge", "Planning", "Motivation", "Analysis"],
    "basketball scout": ["Observation", "Analysis", "Communication", "Reporting"],
    "soccer analyst": ["Analysis", "Communication", "Tactical knowledge", "Observation"],
    "cycling coach": ["Technique", "Endurance knowledge", "Motivation", "Observation"],
    "sports statistician": ["Data analysis", "Observation", "Reporting", "Attention to detail"],
    "ice skating coach": ["Technique", "Patience", "Teaching", "Motivation"],
    "gymnastics coach": ["Technique", "Physical fitness", "Motivation", "Observation"],
    "athlete manager": ["Organization", "Communication", "Negotiation", "Planning"],
    "parkour trainer": ["Physical fitness", "Teaching", "Motivation", "Safety awareness"],
    "sports therapist": ["Physiotherapy knowledge", "Empathy", "Observation", "Communication"],
    "outdoor adventure guide": ["Safety knowledge", "Leadership", "Planning", "Physical fitness"],
    "electrician": ["Technical knowledge", "Problem-solving", "Safety awareness", "Precision"],
    "plumber": ["Technical skills", "Problem-solving", "Attention to detail", "Manual dexterity"],
    "carpenter": ["Woodworking skills", "Precision", "Planning", "Creativity"],
    "mechanic": ["Technical knowledge", "Problem-solving", "Attention to detail", "Manual skills"],
    "barber": ["Haircutting skills", "Creativity", "Customer service", "Precision"],
    "chef": ["Cooking skills", "Creativity", "Time management", "Teamwork"],
    "baker": ["Baking skills", "Precision", "Time management", "Creativity"],
    "janitor": ["Cleaning skills", "Organization", "Time management", "Attention to detail"],
    "tailor": ["Sewing skills", "Precision", "Creativity", "Attention to detail"],
    "painter": ["Painting skills", "Creativity", "Precision", "Time management"],
    "locksmith": ["Technical skills", "Problem-solving", "Manual dexterity", "Precision"],
    "welder": ["Technical skills", "Safety awareness", "Manual skills", "Precision"],
    "glazier": ["Glass cutting", "Precision", "Safety awareness", "Problem-solving"],
    "roofing contractor": ["Manual skills", "Planning", "Safety awareness", "Problem-solving"],
    "landscaper": ["Creativity", "Physical fitness", "Planning", "Plant knowledge"],
    "gardener": ["Plant knowledge", "Physical fitness", "Planning", "Attention to detail"],
    "HVAC technician": ["Technical skills", "Problem-solving", "Safety awareness", "Analysis"],
    "pest control specialist": ["Technical knowledge", "Safety awareness", "Problem-solving", "Observation"],
    "seamstress": ["Sewing skills", "Precision", "Creativity", "Patience"],
    "house painter": ["Painting skills", "Precision", "Time management", "Attention to detail"],
    "flooring installer": ["Manual skills", "Precision", "Planning", "Problem-solving"],
    "tile setter": ["Manual skills", "Precision", "Planning", "Problem-solving"],
    "car detailer": ["Attention to detail", "Cleaning skills", "Time management", "Patience"],
    "handyman": ["Problem-solving", "Manual skills", "Versatility", "Time management"],
    "furniture maker": ["Woodworking", "Creativity", "Precision", "Planning"],
    "sign maker": ["Creativity", "Precision", "Design skills", "Planning"],
    "mason": ["Technical skills", "Precision", "Problem-solving", "Manual skills"],
    "drywall installer": ["Manual skills", "Precision", "Planning", "Problem-solving"],
    "window installer": ["Manual skills", "Precision", "Planning", "Problem-solving"],
    "equipment operator": ["Technical skills", "Safety awareness", "Attention to detail", "Problem-solving"],
    "commercial pilot": ["Flying skills", "Navigation", "Decision making", "Communication"],
    "flight attendant": ["Customer service", "Communication", "Safety knowledge", "Emergency response"],
    "air traffic controller": ["Attention to detail", "Decision making", "Communication", "Stress management"],
    "aircraft maintenance engineer": ["Technical knowledge", "Problem-solving", "Safety awareness", "Attention to detail"],
    "flight operations officer": ["Planning", "Coordination", "Communication", "Attention to detail"],
    "cargo handler": ["Physical fitness", "Organization", "Safety awareness", "Teamwork"],
    "aerospace engineer": ["Engineering skills", "Problem-solving", "Design", "Analysis"],
    "aviation inspector": ["Attention to detail", "Safety knowledge", "Regulatory knowledge", "Analysis"],
    "drone operator": ["Technical skills", "Navigation", "Safety awareness", "Problem-solving"],
    "airline scheduler": ["Planning", "Coordination", "Organization", "Communication"],
    "ground crew": ["Physical fitness", "Teamwork", "Safety awareness", "Communication"],
    "helicopter pilot": ["Flying skills", "Navigation", "Decision making", "Safety awareness"],
    "airport manager": ["Leadership", "Planning", "Communication", "Problem-solving"],
    "aviation safety officer": ["Safety knowledge", "Analysis", "Problem-solving", "Communication"],
    "flight dispatcher": ["Planning", "Communication", "Coordination", "Attention to detail"],
    "aviation analyst": ["Analysis", "Research", "Communication", "Problem-solving"],
    "maintenance planner": ["Planning", "Technical knowledge", "Coordination", "Problem-solving"],
    "avionics technician": ["Technical skills", "Problem-solving", "Attention to detail", "Manual skills"],
    "airline customer service agent": ["Customer service", "Communication", "Problem-solving", "Patience"],
    "aircraft fueling operator": ["Safety awareness", "Attention to detail", "Physical fitness", "Problem-solving"],
    "meteorologist (aviation)": ["Weather knowledge", "Analysis", "Communication", "Problem-solving"],
    "airport security officer": ["Observation", "Problem-solving", "Communication", "Attention to detail"],
    "pilot trainer": ["Flying skills", "Teaching", "Communication", "Patience"],
    "airline operations manager": ["Leadership", "Planning", "Coordination", "Problem-solving"],
    "cabin services manager": ["Leadership", "Customer service", "Communication", "Problem-solving"],
    "flight simulator instructor": ["Technical knowledge", "Teaching", "Patience", "Communication"],
    "aviation logistician": ["Planning", "Coordination", "Problem-solving", "Communication"],
    "aircraft parts specialist": ["Technical knowledge", "Attention to detail", "Organization", "Problem-solving"],
    "airline quality assurance officer": ["Analysis", "Attention to detail", "Problem-solving", "Communication"],
    "air traffic systems engineer": ["Technical knowledge", "Analysis", "Problem-solving", "Communication"],
    "tour guide": ["Communication", "Knowledge of culture", "Organization", "Interpersonal skills"],
    "event planner": ["Organization", "Communication", "Creativity", "Time management"],
    "photography assistant": ["Photography skills", "Attention to detail", "Teamwork", "Creativity"],
    "translator": ["Language skills", "Writing", "Cultural knowledge", "Attention to detail"],
    "interpreter": ["Language skills", "Listening", "Communication", "Quick thinking"],
    "market researcher": ["Research", "Analysis", "Communication", "Attention to detail"],
    "real estate agent": ["Negotiation", "Communication", "Organization", "Marketing knowledge"],
    "public speaker": ["Communication", "Confidence", "Storytelling", "Engagement"],
    "author": ["Writing", "Creativity", "Research", "Discipline"],
    "vlogger": ["Video skills", "Creativity", "Communication", "Editing"],
    "podcaster": ["Communication", "Storytelling", "Editing", "Consistency"],
    "content creator": ["Creativity", "Writing", "Social media skills", "Time management"],
    "life coach": ["Empathy", "Communication", "Problem-solving", "Motivation"],
    "motivational speaker": ["Communication", "Confidence", "Storytelling", "Empathy"],
    "travel blogger": ["Writing", "Photography", "Creativity", "Planning"],
    "crafts maker": ["Creativity", "Manual skills", "Patience", "Design"],
    "animal trainer": ["Patience", "Observation", "Training skills", "Empathy"],
    "pet groomer": ["Animal handling", "Patience", "Attention to detail", "Safety"],
    "florist": ["Creativity", "Design", "Manual skills", "Customer service"],
    "museum guide": ["Communication", "History knowledge", "Interpersonal skills", "Presentation"],
    "archivist": ["Organization", "Research", "Attention to detail", "Preservation skills"],
    "librarian assistant": ["Organization", "Communication", "Attention to detail", "Research"],
    "app tester": ["Attention to detail", "Problem-solving", "Technical skills", "Communication"],
    "UX researcher": ["Research", "Analysis", "Communication", "Observation"],
    "sound technician": ["Technical knowledge", "Attention to detail", "Problem-solving", "Collaboration"],
    "video producer": ["Planning", "Creativity", "Editing", "Communication"],
    "drone photographer": ["Drone handling", "Photography", "Creativity", "Safety awareness"],
    "fitness influencer": ["Social media", "Creativity", "Motivation", "Communication"],
    "social media manager": ["Content creation", "Planning", "Communication", "Analytics"],
    "startup mentor": ["Guidance", "Communication", "Experience", "Problem-solving"],
    "dance": [  # add a simpler key to catch "dance" or "dancing"
        "Rhythm",
        "Motivation",
        "Creativity",
        "Communication",
        "Energy and stamina"
    ],
    "dancing": [  # optional
        "Rhythm",
        "Motivation",
        "Creativity",
        "Communication",
        "Energy and stamina"],
    "singing": [
        "Mastery of a primary instrument or vocals",
        "Understanding of music theory and composition",
        "Collaboration and teamwork with other artists",
        "Stage presence and confidence",
        "Basic audio production and recording knowledge"
    ],
    "singer": [
        "Vocal control and breathing techniques",
        "Music theory and song interpretation",
        "Performance skills and stage presence",
        "Collaboration with other musicians",
        "Practice and consistency"
    ],
    "music": [  # general catch-all
        "Mastery of a primary instrument or vocals",
        "Understanding of music theory and composition",
        "Collaboration and teamwork with other artists",
        "Stage presence and confidence"
    ],
    # IT & Technical
    "software engineer": ["Proficiency in Python, Java, or C++", "Knowledge of data structures & algorithms", "Software development lifecycle experience", "Problem-solving & debugging", "Collaboration with Git"],
    "data scientist": ["Statistics & probability knowledge", "Python/R and ML libraries", "Data visualization (Matplotlib, Tableau)", "Feature engineering & data cleaning", "Communicate insights effectively"],
    "ai/ml engineer": ["ML/DL frameworks (TensorFlow, PyTorch)", "Neural networks knowledge", "Data preprocessing & feature selection", "Hyperparameter tuning & evaluation", "Deploying scalable models"],
    # Healthcare
    "doctor": ["Medical knowledge & diagnostics", "Patient care & empathy", "Critical thinking & problem-solving", "Time management under pressure", "Collaboration with healthcare teams"],
    "nurse": ["Patient monitoring & care", "Medication administration", "Communication with patients & doctors", "Emergency response", "Compassion & empathy"],
    # Creative & Arts
    "dance fitness instructor": ["Rhythm", "Motivation", "Creativity", "Communication"],
    "musician": ["Instrument/vocal mastery", "Music theory & composition", "Collaboration with artists", "Stage presence", "Basic audio production"],
    # Trending skills
    "trending skills 2026": ["AI & ML proficiency", "Data analytics", "Cloud computing", "Cybersecurity", "Emotional intelligence", "Creativity & innovation", "Critical thinking", "Communication", "Collaboration", "Adaptability"]

}

# Ensure all keys are lowercase for consistency
all_careers_skills = {**{k.lower(): v for k,v in skills_fallback.items()}}
# Add career_info if you have additional structured data
# all_careers_skills.update({k.lower(): v.get("next_steps", []) for k,v in career_info.items()})

# ------------------ Helper Functions ------------------

def normalize_text(text):
    return text.lower().strip()

def fetch_from_duckduckgo(query):
    """Fallback search using DuckDuckGo Instant Answer API."""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1"
        r = requests.get(url, timeout=5).json()
        if r.get("AbstractText"):
            return r["AbstractText"]
        if r.get("RelatedTopics"):
            for topic in r["RelatedTopics"]:
                if isinstance(topic, dict) and topic.get("Text"):
                    return topic["Text"]
    except Exception:
        return None
    return None

def get_wiki_summary(query):
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return None
        best_page, best_score = None, 0
        for title in search_results[:7]:
            try:
                page = wikipedia.page(title)
                score = SequenceMatcher(None, query.lower(), page.summary.lower()).ratio()
                if score > best_score:
                    best_score, best_page = score, page
            except:
                continue
        if best_page and best_score > 0.35:
            text = best_page.summary.strip()
            sentences = text.split(". ")
            return ". ".join(sentences[:5]) + "."
    except:
        return None
    return None

def fuzzy_match_multiple_skills(query, threshold=70):
    query_norm = normalize_text(query)
    matched_careers = []
    for career in all_careers_skills.keys():
        score = SequenceMatcher(None, career, query_norm).ratio() * 100
        if score >= threshold:
            matched_careers.append((career, all_careers_skills[career]))
    return matched_careers

# ------------------ GPT / AI Hybrid Placeholder ------------------
def generate_answer(prompt):
    """
    Placeholder for GPT/FLAN-T5 / other ML-generated answer.
    If not using GPT, this can be left empty or call a local model.
    """
    # Example static response for demonstration
    return ""  # Leave blank to fallback on Wikipedia/DuckDuckGo

# ------------------ ML Career Prediction Placeholder ------------------
def predict_top3_careers(query):
    query_lower = query.lower()
    # Simple keyword mapping to prioritize careers
    keyword_map = {
        "it": ["software engineer", "data scientist", "ai/ml engineer", "cybersecurity analyst", "devops engineer"],
        "software": ["software engineer", "web developer", "mobile app developer", "cloud architect"],
        "ai": ["ai/ml engineer", "data scientist"],
        "tech": ["software engineer", "network engineer", "cloud architect"],
        "health": ["doctor", "nurse", "pharmacist"],
        "music": ["musician", "dance fitness instructor"],
        "dance": ["dance fitness instructor"],
        "finance": ["financial analyst", "investment banker", "accountant"],
        "education": ["teacher", "professor", "curriculum designer"],
        "creative": ["musician", "graphic designer", "photographer"]
    }

    # Find matching keywords
    matched_careers = []
    for k, careers in keyword_map.items():
        if k in query_lower:
            matched_careers.extend(careers)
    
    # If none matched, fallback to all careers
    if not matched_careers:
        matched_careers = list(all_careers_skills.keys())
    
    # Pick top 3 (or less if less available)
    top3 = [(career.title(), np.random.rand()) for career in matched_careers[:3]]
    return top3
    
# ------------------ Main Answer Function ------------------
def get_hybrid_answer_multi(query):
    # 1️⃣ Check multiple fuzzy matches first
    matched_careers = fuzzy_match_multiple_skills(query)
    if matched_careers:
        result = ""
        for career, skills in matched_careers:
            result += f"💡 **Key skills / next steps for {career.title()}:**\n- " + "\n- ".join(skills) + "\n\n"
        return result.strip()
    
    # 2️⃣ Career prediction if query implies career advice
    career_keywords = ["career", "job", "suit me", "suggest", "what should i do", "profession"]
    if any(k in query.lower() for k in career_keywords):
        top3 = predict_top3_careers(query)
        result = "💼 **Top 3 career suggestions based on your input:**\n"
        for career, prob in top3:
            skills = all_careers_skills.get(career.lower(), [])
            result += f"- {career} (confidence: {prob*100:.1f}%)\n"
            if skills:
                result += "  **Skills / Next Steps:**\n"
                for s in skills:
                    result += f"    - {s}\n"
        return result

    # 3️⃣ GPT / AI hybrid answer
    prompt = f"You are a career guidance assistant. Provide detailed info for '{query}'."
    answer = generate_answer(prompt)

    # 4️⃣ Wikipedia fallback
    if len(answer) < 20:
        wiki = get_wiki_summary(query)
        if wiki:
            return wiki

    # 5️⃣ DuckDuckGo fallback
    if len(answer) < 20:
        duck = fetch_from_duckduckgo(query)
        if duck:
            return duck

    # 6️⃣ Default
    return "❌ I couldn't find a detailed answer. Try rephrasing or adding more context."

# ------------------ Streamlit UI ------------------

st.header("🤖 Chatbot Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores tuples: (question, answer)

user_query = st.text_input("Ask me about careers, skills, or any topic:")

if user_query:
    answer = get_hybrid_answer_multi(user_query)
    st.session_state.chat_history.insert(0, (user_query, answer))
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[:10]

# Show latest answer
if st.session_state.chat_history:
    st.subheader("🧠 Latest Answer")
    latest_q, latest_a = st.session_state.chat_history[0]
    st.markdown(f"**You:** {latest_q}")
    st.markdown(f"**Bot:** {latest_a}")

# Previous chats
st.markdown("---")
st.subheader("💬 Previous Chats")
if len(st.session_state.chat_history) > 1:
    for q, a in st.session_state.chat_history[1:]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
else:
    st.write("_No previous chats yet._")

# Clear chat button
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history.clear()
    st.experimental_rerun()

st.caption("💡 Tip: Ask about careers, skills, science, history, or any topic for detailed answers.")
