import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import string

for r in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']:
    nltk.download(r, quiet=True)

stop_words = set(stopwords.words('english'))

# ─────────────────────────────────────────────
# BUILT-IN TRAINING DATASET
# ─────────────────────────────────────────────
TRAINING_DATA = [
    # Positive
    ("I love this product, it works amazingly well", "positive"),
    ("This is absolutely fantastic and wonderful", "positive"),
    ("Great experience, highly recommend to everyone", "positive"),
    ("The service was excellent and very helpful", "positive"),
    ("I am so happy with the results, truly impressive", "positive"),
    ("Outstanding quality and fast delivery", "positive"),
    ("Best purchase I have ever made, very satisfied", "positive"),
    ("The team did a brilliant job on this project", "positive"),
    ("Incredible performance and beautiful design", "positive"),
    ("This makes me so joyful and excited about the future", "positive"),
    ("Super easy to use and really effective", "positive"),
    ("Perfectly done, exceeded all my expectations", "positive"),
    ("Very good work, I appreciate the effort", "positive"),
    ("Wonderful results, I am thoroughly impressed", "positive"),
    ("This is a great improvement over the previous version", "positive"),
    ("Loved every bit of it, totally worth it", "positive"),
    ("Amazing features and smooth performance", "positive"),
    ("Really happy with this, would buy again", "positive"),
    # Negative
    ("I hate this, it is completely useless and broken", "negative"),
    ("Terrible experience, would not recommend at all", "negative"),
    ("Very disappointing, the quality is really bad", "negative"),
    ("This is the worst product I have ever used", "negative"),
    ("Awful service and very rude staff", "negative"),
    ("Nothing works properly, total waste of money", "negative"),
    ("I am so frustrated with this poor performance", "negative"),
    ("Horrible design and very difficult to use", "negative"),
    ("This is a disaster, nothing meets expectations", "negative"),
    ("Very unhappy with the result, deeply disappointed", "negative"),
    ("Broken on arrival and customer support was useless", "negative"),
    ("Extremely slow and keeps crashing all the time", "negative"),
    ("Do not buy this, it is a complete scam", "negative"),
    ("Poor build quality and overpriced for what you get", "negative"),
    ("Failed completely, did not work as advertised", "negative"),
    ("Worst experience ever, total disappointment", "negative"),
    ("I regret this purchase, not worth the price", "negative"),
    ("Defective product with zero customer support", "negative"),
    # Neutral
    ("The product arrived on time as expected", "neutral"),
    ("It does what it says on the box, nothing more", "neutral"),
    ("The package was delivered yesterday afternoon", "neutral"),
    ("I used the service once and it was okay", "neutral"),
    ("The item is average, neither good nor bad", "neutral"),
    ("It is a standard product with basic features", "neutral"),
    ("The meeting was held at the usual time", "neutral"),
    ("The report contains data from last quarter", "neutral"),
    ("I read the manual and followed the instructions", "neutral"),
    ("The update was released on Monday morning", "neutral"),
    ("The file was saved successfully to the folder", "neutral"),
    ("The device requires two AA batteries to operate", "neutral"),
    ("Results were recorded and sent to the team", "neutral"),
    ("The software version is 3.2 and runs on Windows", "neutral"),
]

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return {word: True for word in tokens}

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

def train_model():
    dataset = [(extract_features(text), label) for text, label in TRAINING_DATA]
    # 80/20 split
    split = int(len(dataset) * 0.8)
    train_set = dataset[:split]
    test_set  = dataset[split:]
    classifier = NaiveBayesClassifier.train(train_set)
    acc = round(accuracy(classifier, test_set) * 100, 1)
    return classifier, acc

_classifier, _accuracy = train_model()

# ─────────────────────────────────────────────
# SENTIMENT SCORE
# ─────────────────────────────────────────────

EMOJI_MAP = {
    "positive": "😊",
    "negative": "😞",
    "neutral":  "😐"
}

COLOR_MAP = {
    "positive": "#4ade80",
    "negative": "#f87171",
    "neutral":  "#94a3b8"
}

def analyze_sentiment(text):
    features = extract_features(text)
    label = _classifier.classify(features)
    prob_dist = _classifier.prob_classify(features)
    confidence = round(prob_dist.prob(label) * 100, 1)

    return {
        "label": label,
        "confidence": confidence,
        "emoji": EMOJI_MAP[label],
        "color": COLOR_MAP[label],
        "model_accuracy": _accuracy
    }