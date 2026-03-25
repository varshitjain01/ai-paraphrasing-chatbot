import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import math
from collections import Counter

for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(r, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────
# REFERENCE CORPUS (for IDF calculation)
# ─────────────────────────────────────────────
CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "Natural language processing is a field of artificial intelligence",
    "Machine learning models can understand and generate human text",
    "Deep learning has revolutionized speech recognition and translation",
    "Text classification assigns predefined categories to text documents",
    "Sentiment analysis determines the emotional tone of a piece of text",
    "Named entity recognition identifies proper nouns in text like names and places",
    "Topic modeling discovers abstract topics within a collection of documents",
    "Word embeddings represent words as dense vectors in continuous space",
    "The bag of words model represents text as an unordered set of words",
    "TF-IDF stands for term frequency inverse document frequency",
    "Information retrieval systems use TF-IDF to rank document relevance",
    "Tokenization splits text into individual words or sentences",
    "Lemmatization reduces words to their base dictionary form",
    "Stop words are common words that carry little meaningful information",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and t not in string.punctuation and t.isalpha()
    ]
    return tokens

def compute_tf(tokens):
    count = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: round(freq / total, 4) for word, freq in count.items()}

def compute_idf(word, corpus_tokens):
    N = len(corpus_tokens)
    df = sum(1 for doc in corpus_tokens if word in doc)
    return round(math.log((N + 1) / (df + 1)) + 1, 4)  # smoothed IDF

def compute_tfidf(text):
    tokens = preprocess(text)
    corpus_tokens = [set(preprocess(doc)) for doc in CORPUS]

    tf  = compute_tf(tokens)
    tfidf = {}
    for word, tf_val in tf.items():
        idf_val = compute_idf(word, corpus_tokens)
        tfidf[word] = round(tf_val * idf_val, 4)

    # Sort by score descending
    sorted_tfidf = dict(sorted(tfidf.items(), key=lambda x: x[1], reverse=True))
    return sorted_tfidf

# ─────────────────────────────────────────────
# BAG OF WORDS
# ─────────────────────────────────────────────

def compute_bow(text):
    tokens = preprocess(text)
    bow = Counter(tokens)
    return dict(sorted(bow.items(), key=lambda x: x[1], reverse=True))

# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze_features(text):
    tokens = preprocess(text)
    bow    = compute_bow(text)
    tfidf  = compute_tfidf(text)

    # Top keywords by TF-IDF
    top_keywords = list(tfidf.keys())[:8]

    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "unique_tokens": len(set(tokens)),
        "bow": bow,
        "tfidf": tfidf,
        "top_keywords": top_keywords,
    }