import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(r, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────
# REFERENCE CORPUS FOR LDA
# ─────────────────────────────────────────────

CORPUS_DOCS = [
    # Technology
    "Artificial intelligence and machine learning are transforming software development and data analysis",
    "Deep learning neural networks process large datasets to recognize patterns in images and text",
    "Cloud computing platforms provide scalable infrastructure for deploying machine learning models",
    "Python and TensorFlow are popular tools for building natural language processing systems",
    "Algorithms and data structures form the foundation of computer science and programming",
    # Health
    "Regular exercise and a balanced diet are essential for maintaining good physical health",
    "Mental health awareness has increased with more people seeking therapy and counseling",
    "Doctors recommend annual checkups and blood tests to monitor overall health conditions",
    "Vaccines and antibiotics have significantly reduced the spread of infectious diseases",
    "Sleep deprivation can seriously impact cognitive function and emotional wellbeing",
    # Business
    "Companies are investing in digital marketing strategies to reach more customers online",
    "Supply chain management and logistics are critical to the success of global businesses",
    "Startups are disrupting traditional industries with innovative products and services",
    "Financial planning and investment strategies help businesses manage risk and grow revenue",
    "Leadership and effective communication are key skills for successful business management",
    # Environment
    "Climate change and global warming are causing extreme weather events around the world",
    "Renewable energy sources like solar and wind power can reduce carbon emissions significantly",
    "Deforestation and habitat destruction threaten biodiversity and ecosystem balance",
    "Governments are signing international agreements to address environmental pollution",
    "Sustainable agriculture practices help preserve soil quality and reduce water usage",
    # Education
    "Online learning platforms have made quality education accessible to students worldwide",
    "Critical thinking and problem solving skills are essential for academic and career success",
    "Teachers play a vital role in shaping the intellectual and social development of children",
    "University research contributes to scientific discoveries and technological innovations",
    "Reading and writing proficiency are fundamental literacy skills taught in primary schools",
]

TOPIC_LABELS = {
    0: {"name": "Technology & AI",  "emoji": "💻", "color": "#7c6aff"},
    1: {"name": "Health & Wellness","emoji": "🏥", "color": "#34d399"},
    2: {"name": "Business",         "emoji": "💼", "color": "#f472b6"},
    3: {"name": "Environment",      "emoji": "🌱", "color": "#4ade80"},
    4: {"name": "Education",        "emoji": "📚", "color": "#60a5fa"},
}

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and t not in string.punctuation and t.isalpha() and len(t) > 2
    ]

# ─────────────────────────────────────────────
# KEYWORD-BASED TOPIC MATCHING (LDA-inspired)
# ─────────────────────────────────────────────

TOPIC_KEYWORDS = {
    0: {"ai", "machine", "learning", "deep", "neural", "algorithm", "data", "software",
        "python", "model", "compute", "cloud", "technology", "network", "system",
        "code", "program", "computer", "digital", "intelligence", "processing"},
    1: {"health", "exercise", "diet", "mental", "doctor", "vaccine", "sleep", "disease",
        "medicine", "therapy", "hospital", "patient", "treatment", "wellness", "body",
        "medical", "drug", "symptom", "fitness", "nutrition"},
    2: {"business", "company", "market", "investment", "startup", "revenue", "customer",
        "product", "service", "management", "strategy", "finance", "supply", "brand",
        "profit", "sale", "economy", "enterprise", "leadership", "industry"},
    3: {"climate", "environment", "energy", "solar", "carbon", "pollution", "forest",
        "ecosystem", "renewable", "sustainability", "biodiversity", "warming", "emission",
        "nature", "green", "ocean", "wildlife", "conservation", "weather", "planet"},
    4: {"education", "school", "student", "teacher", "learning", "university", "academic",
        "research", "reading", "writing", "literacy", "skill", "knowledge", "course",
        "curriculum", "classroom", "study", "degree", "exam", "scholarship"},
}

def analyze_topics(text):
    tokens = set(preprocess(text))

    if not tokens:
        return {
            "topics": [],
            "top_topic": None,
            "keywords_found": [],
        }

    # Score each topic
    scores = {}
    matched_keywords = {}
    for topic_id, keywords in TOPIC_KEYWORDS.items():
        matches = tokens & keywords
        scores[topic_id] = len(matches)
        matched_keywords[topic_id] = list(matches)

    total = sum(scores.values())

    if total == 0:
        return {
            "topics": [],
            "top_topic": {"name": "General", "emoji": "📄", "color": "#94a3b8", "score": 0},
            "keywords_found": list(tokens)[:6],
        }

    # Build topic results sorted by score
    topics = []
    for topic_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score > 0:
            meta = TOPIC_LABELS[topic_id]
            topics.append({
                "id":       topic_id,
                "name":     meta["name"],
                "emoji":    meta["emoji"],
                "color":    meta["color"],
                "score":    score,
                "percent":  round((score / total) * 100),
                "keywords": matched_keywords[topic_id],
            })

    top = topics[0] if topics else None

    return {
        "topics": topics[:3],
        "top_topic": top,
        "keywords_found": list(tokens)[:8],
    }