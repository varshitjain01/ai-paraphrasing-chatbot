import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import random

# Download required resources
resources = [
    'punkt',
    'punkt_tab',
    'wordnet',
    'stopwords',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'omw-1.4'
]

for r in resources:
    nltk.download(r, quiet=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Manual replacements (BEST QUALITY)
simple_map = {
    "transforming": "changing",
    "improving": "enhancing",
    "improved": "enhanced",
    "important": "crucial",
    "big": "large",
    "small": "tiny",
    "fast": "quick",
    "slow": "gradual",
    "long": "lengthy",
    "good": "effective"
}

# Protected words (DO NOT CHANGE)
protected_words = {
    "guilty", "crime", "judge", "punishment",
    "sentence", "law", "court", "capital", "full"
}

# Protected phrases
protected_phrases = [
    "full stop",
    "capital letter",
    "human life"
]

# ✅ SAFE adjectives only
safe_adjectives = {
    "important", "big", "small", "fast", "slow",
    "long", "good"
}

# POS mapping
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    return None

# Synonym filtering
def get_synonyms(word, pos):
    synonyms = set()

    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")

            if (
                synonym.lower() != word.lower() and
                synonym.isalpha() and
                len(synonym.split()) == 1 and
                3 <= len(synonym) <= 10 and
                synonym.islower()
            ):
                synonyms.add(synonym)

    simple_words = [w for w in synonyms if len(w) <= len(word)]

    return simple_words if simple_words else list(synonyms)

# Main paraphrasing function
def paraphrase(text):

    # Phrase protection
    for phrase in protected_phrases:
        if phrase in text.lower():
            return text

    words = word_tokenize(text)
    tagged = pos_tag(words)

    new_sentence = []
    replace_count = 0
    max_replacements = 2   # ✅ FIXED

    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)

        # Lemmatization
        base_word = lemmatizer.lemmatize(word, wn_tag) if wn_tag else word

        # Manual mapping (highest priority)
        if word.lower() in simple_map and replace_count < max_replacements:
            new_sentence.append(simple_map[word.lower()])
            replace_count += 1

        # Controlled synonym replacement (SAFE)
        elif (
            wn_tag and
            word.lower() not in stop_words and
            word.lower() not in protected_words and
            word.lower() in safe_adjectives and   # ✅ SAFE FILTER
            word.isalpha() and
            replace_count < max_replacements and
            not word[0].isupper() and
            tag.startswith('J')   # adjectives only
        ):
            synonyms = get_synonyms(base_word, wn_tag)

            if synonyms:
                new_sentence.append(random.choice(synonyms))
                replace_count += 1
            else:
                new_sentence.append(word)

        else:
            new_sentence.append(word)

    return " ".join(new_sentence)