import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Training dataset: 25 career paths × 15 examples = 375
data = {
    "user_input": [
        # ---------------- Software Engineer ----------------
        "I love coding and building software",
        "I enjoy solving programming problems",
        "I like writing Python and Java code",
        "I am interested in creating mobile apps",
        "I enjoy debugging and fixing errors",
        "I like developing web applications",
        "I am passionate about learning new programming languages",
        "I want to work on artificial intelligence projects",
        "I like working on databases and backend systems",
        "I enjoy solving algorithm challenges",
        "I want to become a full stack developer",
        "I like contributing to open source projects",
        "I am interested in cloud computing",
        "I want to specialize in cybersecurity",
        "I like designing scalable systems",

        # ---------------- Data Analyst ----------------
        "I like working with data and numbers",
        "I enjoy finding insights from data",
        "I am good at using Excel and SQL",
        "I love analyzing trends and patterns",
        "I am interested in statistics and reports",
        "I want to become a data scientist",
        "I like building dashboards",
        "I am skilled in data visualization",
        "I enjoy cleaning and preparing data",
        "I am passionate about predictive analysis",
        "I like solving business problems with data",
        "I want to learn Python for data analysis",
        "I am interested in machine learning models",
        "I like analyzing customer behavior",
        "I want to work with big data technologies",

        # ---------------- UI/UX Designer ----------------
        "I enjoy drawing and designing apps",
        "I love creating user-friendly interfaces",
        "I am passionate about graphic design",
        "I like using Figma and Adobe XD",
        "I enjoy working on app layouts and color schemes",
        "I want to make products more accessible",
        "I like designing user journeys",
        "I am interested in mobile UI design",
        "I love working on creative projects",
        "I want to learn motion graphics",
        "I like prototyping app designs",
        "I enjoy collaborating with developers",
        "I am passionate about typography",
        "I want to design logos and branding",
        "I like experimenting with colors and themes",

        # ---------------- Event Manager ----------------
        "I'm great at organizing events",
        "I love planning weddings and parties",
        "I enjoy coordinating teams for events",
        "I am good at scheduling and managing programs",
        "I like handling event logistics and decorations",
        "I want to manage concerts and shows",
        "I like working with caterers and decorators",
        "I enjoy negotiating with vendors",
        "I am good at budgeting for events",
        "I like planning conferences",
        "I enjoy handling large crowds",
        "I want to build a career in event planning",
        "I like creating memorable experiences",
        "I enjoy hosting programs",
        "I am passionate about social events",

        # ---------------- Fashion Designer ----------------
        "I love designing clothes and accessories",
        "I enjoy sketching fashion outfits",
        "I like working with fabrics and textiles",
        "I follow the latest fashion trends",
        "I am passionate about creating stylish looks",
        "I want to launch my own fashion brand",
        "I like experimenting with clothing designs",
        "I enjoy creating jewelry designs",
        "I am inspired by runway fashion",
        "I want to work with models",
        "I like exploring cultural fashion",
        "I enjoy sewing and tailoring",
        "I want to design eco-friendly clothing",
        "I am interested in luxury fashion",
        "I love fashion photography",

        # ---------------- Model ----------------
        "I enjoy posing for photoshoots",
        "I like walking on the runway",
        "I am confident in front of the camera",
        "I enjoy working with fashion brands",
        "I like showcasing clothes and styles",
        "I want to become a fashion model",
        "I enjoy lifestyle modeling",
        "I like doing product promotions",
        "I am interested in acting and modeling",
        "I like participating in pageants",
        "I enjoy fitness modeling",
        "I like working with photographers",
        "I want to build a modeling portfolio",
        "I am passionate about grooming and styling",
        "I like doing brand endorsements",

        # ---------------- Business Entrepreneur ----------------
        "I want to start my own company",
        "I like managing finances and profits",
        "I am good at business strategies",
        "I enjoy entrepreneurship",
        "I want to become a successful entrepreneur",
        "I like developing startups",
        "I am passionate about investments",
        "I want to scale my business globally",
        "I like pitching to investors",
        "I am interested in stock markets",
        "I want to build a tech company",
        "I enjoy solving customer problems",
        "I am good at building networks",
        "I like managing risks",
        "I want to innovate new products",

        # ---------------- Police Officer ----------------
        "I want to serve and protect people",
        "I like maintaining law and order",
        "I am interested in criminal justice",
        "I want to catch criminals",
        "I like working for public safety",
        "I want to stop crime in my city",
        "I am passionate about investigation",
        "I want to become a traffic officer",
        "I enjoy maintaining discipline",
        "I like helping victims",
        "I want to be in the police academy",
        "I enjoy physical training",
        "I want to join the cybercrime unit",
        "I like protecting my community",
        "I want to enforce the law",

        # ---------------- Government Officer ----------------
        "I want to work in public administration",
        "I am preparing for civil service exams",
        "I enjoy making policies for the nation",
        "I want to serve the government",
        "I like solving social issues as an officer",
        "I am interested in governance",
        "I want to become an IAS officer",
        "I enjoy working with public schemes",
        "I want to create development programs",
        "I am passionate about justice",
        "I want to join the revenue department",
        "I like working for the nation",
        "I want to contribute to society",
        "I am good at managing public funds",
        "I like working in administration",

        # ---------------- Manager ----------------
        "I like leading teams and projects",
        "I am good at managing people",
        "I enjoy decision-making and leadership",
        "I want to become a project manager",
        "I am skilled at handling responsibilities",
        "I like managing office operations",
        "I enjoy coordinating projects",
        "I am good at time management",
        "I like working in corporate offices",
        "I want to become a team leader",
        "I enjoy monitoring performance",
        "I am good at motivating people",
        "I like strategic planning",
        "I want to become a general manager",
        "I enjoy improving productivity",

        # ---------------- Doctor ----------------
        "I want to treat sick people",
        "I am interested in medicine and surgery",
        "I like working in hospitals",
        "I want to become a surgeon",
        "I love helping people recover from illness",
        "I enjoy learning human anatomy",
        "I want to specialize in cardiology",
        "I like studying medical science",
        "I want to become a pediatrician",
        "I enjoy working in emergency care",
        "I want to cure patients",
        "I am passionate about healthcare",
        "I want to save lives",
        "I am interested in medical research",
        "I want to become a family doctor",

        # ---------------- Nurse ----------------
        "I like taking care of patients",
        "I want to support doctors in hospitals",
        "I enjoy helping people recover",
        "I am compassionate and caring",
        "I want to work in healthcare",
        "I like assisting in surgeries",
        "I want to work in ICUs",
        "I enjoy patient care",
        "I like giving medicines",
        "I want to become a head nurse",
        "I enjoy comforting patients",
        "I want to become a midwife",
        "I am passionate about nursing",
        "I enjoy helping elderly patients",
        "I like hospital duties",

        # ---------------- Astronaut ----------------
        "I want to travel to space",
        "I am fascinated by planets and stars",
        "I like studying space science",
        "I want to work at NASA or ISRO",
        "I am passionate about exploring the universe",
        "I enjoy space technology",
        "I want to become a space scientist",
        "I am interested in astronomy",
        "I like studying physics",
        "I want to go on space missions",
        "I am fascinated by rockets",
        "I like learning about satellites",
        "I want to explore Mars",
        "I am passionate about astrophysics",
        "I want to walk on the moon",

        # ---------------- Teacher ----------------
        "I like teaching students",
        "I enjoy explaining concepts clearly",
        "I am passionate about education",
        "I love working in schools",
        "I want to inspire students to learn",
        "I enjoy preparing lessons",
        "I want to become a school principal",
        "I am good at explaining mathematics",
        "I enjoy teaching science",
        "I want to teach languages",
        "I like encouraging students",
        "I want to become a college professor",
        "I enjoy classroom teaching",
        "I am passionate about learning",
        "I like mentoring students",

        # ---------------- Researcher ----------------
        "I enjoy doing experiments",
        "I like finding new discoveries",
        "I am passionate about scientific research",
        "I want to work in laboratories",
        "I like publishing research papers",
        "I want to become a research scientist",
        "I enjoy working with microscopes",
        "I like studying biology",
        "I want to specialize in physics research",
        "I like testing new theories",
        "I enjoy chemical experiments",
        "I want to research renewable energy",
        "I enjoy field research",
        "I want to become a PhD scholar",
        "I like contributing to new inventions",

        # ---------------- Pilot ----------------
        "I want to fly airplanes",
        "I enjoy traveling across the world",
        "I am interested in aviation",
        "I want to become a commercial pilot",
        "I like operating aircraft controls",
        "I am fascinated by airports",
        "I want to join the air force",
        "I enjoy navigation and flight routes",
        "I like working with flight simulators",
        "I want to transport passengers safely",
        "I enjoy studying aerodynamics",
        "I want to become a captain",
        "I am passionate about flying",
        "I enjoy long flights",
        "I want to explore the aviation industry",

        # ---------------- Lawyer ----------------
        "I enjoy reading about laws",
        "I want to argue cases in court",
        "I am interested in justice",
        "I like helping people with legal issues",
        "I want to become a corporate lawyer",
        "I enjoy studying constitutional law",
        "I want to defend clients in trials",
        "I am passionate about human rights",
        "I like preparing legal documents",
        "I enjoy public speaking in court",
        "I want to become a judge someday",
        "I like debating and reasoning",
        "I want to work in legal research",
        "I am interested in criminal law",
        "I want to specialize in family law",

        # ---------------- Engineer (Civil) ----------------
        "I like building bridges and roads",
        "I want to design large buildings",
        "I am interested in construction projects",
        "I enjoy working on infrastructure",
        "I want to become a civil engineer",
        "I like drawing blueprints",
        "I enjoy surveying land",
        "I am good at structural design",
        "I want to construct safe houses",
        "I like studying materials for construction",
        "I am interested in urban planning",
        "I enjoy managing construction sites",
        "I want to build dams and highways",
        "I am fascinated by architecture",
        "I want to create sustainable structures",

        # ---------------- Chef ----------------
        "I love cooking different cuisines",
        "I want to become a professional chef",
        "I enjoy experimenting with recipes",
        "I am passionate about baking",
        "I want to open my own restaurant",
        "I like learning about food presentation",
        "I enjoy working in kitchens",
        "I want to create unique dishes",
        "I like watching cooking shows",
        "I am interested in food science",
        "I want to become a pastry chef",
        "I like cooking for people",
        "I am passionate about hospitality",
        "I want to master Italian cuisine",
        "I enjoy garnishing food beautifully",

        # ---------------- Actor ----------------
        "I want to act in movies",
        "I enjoy performing on stage",
        "I am passionate about drama",
        "I like participating in plays",
        "I want to become a film actor",
        "I enjoy rehearsing scripts",
        "I like expressing emotions on stage",
        "I am passionate about theatre",
        "I want to become a TV actor",
        "I enjoy auditions and casting",
        "I want to work in Hollywood",
        "I enjoy method acting",
        "I am interested in cinema",
        "I want to win acting awards",
        "I like playing different characters",

        # ---------------- Scientist ----------------
        "I enjoy discovering new theories",
        "I like working in laboratories",
        "I am passionate about scientific innovation",
        "I want to contribute to technology",
        "I like experimenting with new ideas",
        "I want to become a physicist",
        "I enjoy chemistry experiments",
        "I want to discover new medicines",
        "I am passionate about biology",
        "I like researching the environment",
        "I want to contribute to renewable energy",
        "I enjoy solving global challenges",
        "I like publishing research papers",
        "I want to develop new inventions",
        "I am passionate about nanotechnology",

        # ---------------- Architect ----------------
        "I want to design beautiful buildings",
        "I enjoy sketching modern houses",
        "I like studying architecture styles",
        "I want to become a licensed architect",
        "I enjoy interior design",
        "I like working with AutoCAD",
        "I want to design skyscrapers",
        "I am passionate about sustainable architecture",
        "I like studying structural design",
        "I want to create eco-friendly buildings",
        "I enjoy drawing house layouts",
        "I like designing city landscapes",
        "I want to contribute to urban planning",
        "I enjoy working on real estate projects",
        "I want to become a famous architect",

        # ---------------- Journalist ----------------
        "I enjoy writing news articles",
        "I want to work in media",
        "I like reporting current events",
        "I want to become a news anchor",
        "I enjoy investigating stories",
        "I am passionate about journalism",
        "I want to work for a newspaper",
        "I enjoy interviewing people",
        "I want to become a TV reporter",
        "I like writing blogs",
        "I want to expose corruption",
        "I am interested in mass communication",
        "I want to write feature stories",
        "I like covering international news",
        "I enjoy creating documentaries",

        # ---------------- Psychologist ----------------
        "I enjoy listening to people's problems",
        "I want to help people with mental health",
        "I am interested in human behavior",
        "I want to become a clinical psychologist",
        "I like studying psychology",
        "I want to provide counseling",
        "I am passionate about therapy",
        "I want to research human minds",
        "I like conducting psychological tests",
        "I enjoy helping patients with stress",
        "I want to work in hospitals",
        "I am interested in educational psychology",
        "I like studying emotions",
        "I want to write psychology books",
        "I am passionate about guiding people"
    ],
    "career_path": (
        ["Software Engineer"] * 15 +
        ["Data Analyst"] * 15 +
        ["UI/UX Designer"] * 15 +
        ["Event Manager"] * 15 +
        ["Fashion Designer"] * 15 +
        ["Model"] * 15 +
        ["Business Entrepreneur"] * 15 +
        ["Police Officer"] * 15 +
        ["Government Officer"] * 15 +
        ["Manager"] * 15 +
        ["Doctor"] * 15 +
        ["Nurse"] * 15 +
        ["Astronaut"] * 15 +
        ["Teacher"] * 15 +
        ["Researcher"] * 15 +
        ["Pilot"] * 15 +
        ["Lawyer"] * 15 +
        ["Civil Engineer"] * 15 +
        ["Chef"] * 15 +
        ["Actor"] * 15 +
        ["Scientist"] * 15 +
        ["Architect"] * 15 +
        ["Journalist"] * 15 +
        ["Psychologist"] * 15
    )
}

df = pd.DataFrame(data)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["user_input"])
y = df["career_path"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save
joblib.dump(model, "career_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
