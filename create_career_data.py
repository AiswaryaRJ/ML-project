import pandas as pd

data = {
    "description": [
        # Software Engineer
        "I enjoy building mobile apps and solving algorithmic problems",
        "Coding websites and debugging programs excites me",
        "I love creating efficient backend systems for apps",
        # Data Scientist
        "I like analyzing large datasets and finding patterns",
        "I enjoy building predictive models and data visualizations",
        "Solving business problems using data is my passion",
        # AI/ML Engineer
        "I love training machine learning models for real-world tasks",
        "Deep learning and neural networks fascinate me",
        "Building AI-powered applications excites me",
        # Web Developer
        "Designing and developing modern web pages is fun",
        "I enjoy coding front-end interfaces and styling them beautifully",
        "Creating responsive websites makes me happy",
        # UX/UI Designer
        "I love creating user-friendly app interfaces",
        "Designing intuitive user flows is my strength",
        "I enjoy using Figma to build beautiful designs",
        # Graphic Designer
        "Making creative posters and graphics excites me",
        "I enjoy working with Photoshop and Illustrator",
        "Creating brand logos and visuals is my passion",
        # Artist
        "Drawing, painting, and sketching are my hobbies",
        "I love creating digital art and illustrations",
        "Expressing creativity through art excites me",
        # Teacher
        "Helping students understand concepts brings me joy",
        "I love explaining complex topics in simple ways",
        "Mentoring and teaching others makes me fulfilled",
        # Doctor
        "I want to help patients recover and stay healthy",
        "Studying medicine and biology excites me",
        "Providing medical care to people in need fulfills me",
        # Nurse
        "Caring for patients and supporting doctors is rewarding",
        "I like providing bedside care and comfort",
        "Helping people heal makes me feel accomplished",
        # Lawyer
        "I love debating and defending legal rights",
        "Understanding and interpreting laws excites me",
        "Helping clients navigate legal issues interests me",
        # Entrepreneur
        "Building startups and solving business problems excites me",
        "I love creating new products and pitching ideas",
        "Starting my own business is my dream",
        # Accountant
        "Managing financial records and audits interests me",
        "I enjoy working with numbers and budgets",
        "Analyzing balance sheets excites me",
        # Civil Engineer
        "Designing and constructing buildings excites me",
        "I love working on infrastructure projects like bridges",
        "Solving structural engineering problems is fun",
        # Mechanical Engineer
        "I like designing and testing machines",
        "Building mechanical systems excites me",
        "Working on robotics and engines is fascinating",
        # Electrical Engineer
        "I enjoy working on circuits and power systems",
        "Designing electrical components interests me",
        "Creating innovative hardware solutions excites me",
        # Pilot
        "Flying airplanes excites me",
        "I love navigation and aerodynamics",
        "Becoming a commercial pilot is my dream",
        # Chef
        "Cooking and creating new recipes excites me",
        "I love working in a kitchen and experimenting with flavors",
        "Preparing delicious meals brings me joy",
        # Photographer
        "Capturing special moments through a lens excites me",
        "I love experimenting with camera settings and angles",
        "Taking beautiful landscape photos is my passion",
        # Journalist
        "Reporting news and writing articles interests me",
        "I love investigating stories and interviewing people",
        "Sharing important information with the public excites me",
        # Psychologist
        "Understanding human behavior fascinates me",
        "I love counseling people and helping them heal mentally",
        "Studying the mind and emotions excites me",
        # Social Worker
        "Helping underprivileged communities fulfills me",
        "I like working with NGOs to support families",
        "Advocating for social change excites me",
        # Environmental Scientist
        "Protecting nature and ecosystems interests me",
        "I enjoy studying climate change and sustainability",
        "Solving environmental issues excites me",
        # Pharmacist
        "I like understanding medicines and their effects",
        "Helping patients with prescriptions interests me",
        "Working in a pharmacy excites me",
        # Veterinarian
        "Caring for animals and treating their illnesses excites me",
        "I love working with pets and wildlife",
        "Helping animals heal makes me happy",
        # Fashion Designer
        "Designing stylish clothes excites me",
        "I love creating fashion collections and sketches",
        "Working with fabrics and trends interests me",
        # Architect
        "Designing buildings and urban spaces excites me",
        "I love creating floor plans and 3D models",
        "Bringing architectural concepts to life interests me",
        # Marketing Specialist
        "Creating advertising campaigns excites me",
        "I love analyzing market trends and consumer behavior",
        "Promoting brands and products interests me",
        # Software Tester
        "I like finding bugs and improving software quality",
        "Testing apps and reporting issues excites me",
        "Ensuring products are error-free makes me happy"
    ],
    "career": [
        "Software Engineer","Software Engineer","Software Engineer",
        "Data Scientist","Data Scientist","Data Scientist",
        "AI/ML Engineer","AI/ML Engineer","AI/ML Engineer",
        "Web Developer","Web Developer","Web Developer",
        "UX/UI Designer","UX/UI Designer","UX/UI Designer",
        "Graphic Designer","Graphic Designer","Graphic Designer",
        "Artist","Artist","Artist",
        "Teacher","Teacher","Teacher",
        "Doctor","Doctor","Doctor",
        "Nurse","Nurse","Nurse",
        "Lawyer","Lawyer","Lawyer",
        "Entrepreneur","Entrepreneur","Entrepreneur",
        "Accountant","Accountant","Accountant",
        "Civil Engineer","Civil Engineer","Civil Engineer",
        "Mechanical Engineer","Mechanical Engineer","Mechanical Engineer",
        "Electrical Engineer","Electrical Engineer","Electrical Engineer",
        "Pilot","Pilot","Pilot",
        "Chef","Chef","Chef",
        "Photographer","Photographer","Photographer",
        "Journalist","Journalist","Journalist",
        "Psychologist","Psychologist","Psychologist",
        "Social Worker","Social Worker","Social Worker",
        "Environmental Scientist","Environmental Scientist","Environmental Scientist",
        "Pharmacist","Pharmacist","Pharmacist",
        "Veterinarian","Veterinarian","Veterinarian",
        "Fashion Designer","Fashion Designer","Fashion Designer",
        "Architect","Architect","Architect",
        "Marketing Specialist","Marketing Specialist","Marketing Specialist",
        "Software Tester","Software Tester","Software Tester"
    ]
}

df = pd.DataFrame(data)
df.to_csv("career_data.csv", index=False)
print("career_data.csv created with", df['career'].nunique(), "unique careers and", len(df), "rows.")
