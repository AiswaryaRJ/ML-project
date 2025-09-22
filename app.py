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

from chatbot import get_response
from recommender import recommend
from predict_career import predict_career
import streamlit as st
import wikipedia
import requests
from difflib import SequenceMatcher

# ---------------- Text Preprocessing & Typo Handling ----------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import process
import re
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)
    
# List of all careers from your dataset
career_names = df['Career'].unique()

def correct_typo(text, choices=career_names, threshold=80):
    match, score = process.extractOne(text, choices)
    return match if score >= threshold else text


# ---------------- Streamlit Page Settings ----------------
st.set_page_config(page_title="Career Guidance AI", layout="centered")
st.title("Career Guidance AI")
st.caption("Describe your interests/skills and get career suggestions, courses, and chatbot help.")

# ---------------- Load dataset ----------------
df = pd.read_csv("generated_dataset.csv")
career_names = df['Career'].unique()

# Load trained ML model and vectorizer
with open("career_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Optional: for sentence embeddings instead of TF-IDF
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


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
            career_list = list(career_courses.keys())
            career_list.append("Other")

           # Use the predicted career directly
            # career = st.selectbox("Select a career:", career_list)  # <-- commented out

            courses = career_courses.get(career, [])

            if courses:
                st.markdown(f"### Recommended courses for {career}:")
                for course_name, course_link in courses:
                    st.markdown(f"- [{course_name}]({course_link})")
            else:
                st.warning(
                    f"No courses found for the career: '{career}'. "
                    "Please select another career or check back later."
                )


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

#-----------Chatbot-----------------


# ---------- Improved Answer Retrieval ----------
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
            # Take first 4-5 sentences for context
            sentences = text.split(". ")
            return ". ".join(sentences[:5]) + "."
    except:
        return None
    return None

# Small knowledge base for common skills
skills_fallback = {
    "dancer": [
        "Strong sense of rhythm and musicality",
        "Physical strength, flexibility, and stamina",
        "Ability to memorize choreography quickly",
        "Expressiveness and stage presence",
        "Collaboration and discipline for rehearsals"
    ],
    "musician": [
        "Mastery of a primary instrument or vocals",
        "Understanding of music theory and composition",
        "Collaboration and teamwork with other artists",
        "Stage presence and confidence",
        "Basic audio production and recording knowledge"
    ]
}

def get_answer(query):
    # Check curated skills first
    for key, skills in skills_fallback.items():
        if key in query.lower():
            return f"💡 **Key skills for {key.title()}:**\n- " + "\n- ".join(skills)
    # Try Wikipedia
    wiki = get_wiki_summary(query)
    if wiki:
        return wiki
    # Fallback: DuckDuckGo
    duck = fetch_from_duckduckgo(query)
    if duck:
        return duck
    return "❌ I couldn't find a detailed answer. Try rephrasing or adding more context."

# ---------- UI ----------
st.header("🤖 Chatbot Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # stores tuples: (question, answer)
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None

user_query = st.text_input("Ask me about careers, skills, or any topic:")

# Handle new query
if user_query:
    # Move previous Q&A to history
    if st.session_state.last_question and st.session_state.last_answer:
        st.session_state.chat_history.insert(
            0, (st.session_state.last_question, st.session_state.last_answer)
        )
    # Get new answer
    answer = get_answer(user_query)
    st.session_state.last_question = user_query
    st.session_state.last_answer = answer

# Show current answer under chatbot
if st.session_state.last_answer:
    st.markdown("### 🧠 Answer")
    st.write(f"**You:** {st.session_state.last_question}")
    st.write(f"**Bot:** {st.session_state.last_answer}")

# Divider + Chat History
st.markdown("---")
st.subheader("💬 Previous Chats")
if st.session_state.chat_history:
    for q, a in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")
else:
    st.write("_No previous chats yet._")

# Clear button
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history.clear()
    st.session_state.last_answer = None
    st.session_state.last_question = None
    st.rerun()  # ✅ Use st.rerun() instead of st.experimental_rerun()


st.caption("💡 Tip: Ask about careers, skills, science, history, or any topic for detailed answers.")
