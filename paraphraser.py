import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import random

# Download required resources
resources = [
    'punkt', 'punkt_tab', 'wordnet', 'stopwords',
    'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'omw-1.4'
]
for r in resources:
    nltk.download(r, quiet=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────
# TONE MAPS
# ─────────────────────────────────────────────

formal_map = {
    "use": "utilize",
    "help": "assist",
    "get": "obtain",
    "show": "demonstrate",
    "find": "identify",
    "make": "create",
    "start": "commence",
    "end": "conclude",
    "tell": "inform",
    "ask": "inquire",
    "need": "require",
    "buy": "purchase",
    "give": "provide",
    "look": "examine",
    "try": "attempt",
    "keep": "maintain",
    "want": "desire",
    "think": "consider",
    "check": "verify",
    "big": "substantial",
    "small": "minimal",
    "good": "commendable",
    "bad": "unfavorable",
    "fast": "expeditious",
    "slow": "gradual",
    "hard": "challenging",
    "easy": "straightforward",
    "important": "significant",
    "enough": "sufficient",
    "about": "regarding",
    "but": "however",
    "also": "furthermore",
    "so": "therefore",
    "now": "currently",
}

casual_map = {
    "utilize": "use",
    "obtain": "get",
    "demonstrate": "show",
    "identify": "find",
    "commence": "start",
    "conclude": "end",
    "inquire": "ask",
    "require": "need",
    "purchase": "buy",
    "provide": "give",
    "attempt": "try",
    "maintain": "keep",
    "desire": "want",
    "consider": "think about",
    "verify": "check",
    "substantial": "big",
    "significant": "important",
    "sufficient": "enough",
    "regarding": "about",
    "however": "but",
    "furthermore": "also",
    "therefore": "so",
    "currently": "now",
    "important": "key",
    "difficult": "tough",
    "happy": "glad",
    "fast": "quick",
    "improve": "boost",
    "improving": "boosting",
    "big": "huge",
    "good": "great",
}

# ─────────────────────────────────────────────
# PROTECTED WORDS / PHRASES
# ─────────────────────────────────────────────

protected_words = {
    "guilty", "crime", "judge", "punishment", "sentence",
    "law", "court", "capital", "full"
}

protected_phrases = [
    "full stop", "capital letter", "human life"
]

# ─────────────────────────────────────────────
# POS MAPPING
# ─────────────────────────────────────────────

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None

# ─────────────────────────────────────────────
# SYNONYM FETCHER
# ─────────────────────────────────────────────

def get_synonyms(word, pos, tone):
    synonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if (
                synonym.lower() != word.lower() and
                synonym.isalpha() and
                len(synonym.split()) == 1 and
                3 <= len(synonym) <= 12 and
                synonym.islower()
            ):
                # For formal tone: prefer longer/complex synonyms
                # For casual tone: prefer shorter/simpler synonyms
                if tone == "formal" and len(synonym) >= len(word):
                    synonyms.add(synonym)
                elif tone == "casual" and len(synonym) <= len(word):
                    synonyms.add(synonym)
                else:
                    synonyms.add(synonym)

    return list(synonyms)

# ─────────────────────────────────────────────
# MAIN PARAPHRASER
# ─────────────────────────────────────────────

def paraphrase(text, tone="neutral"):
    """
    Paraphrase the input text with the given tone.
    tone: "formal" | "casual" | "neutral"
    """

    # Protected phrase check
    for phrase in protected_phrases:
        if phrase in text.lower():
            return text

    # Choose tone map
    if tone == "formal":
        tone_map = formal_map
        max_replacements = 4
    elif tone == "casual":
        tone_map = casual_map
        max_replacements = 4
    else:
        # Neutral: mix of formal_map and synonym replacement
        tone_map = {**formal_map, **casual_map}
        max_replacements = 3

    words = word_tokenize(text)
    tagged = pos_tag(words)

    new_sentence = []
    replace_count = 0

    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        base_word = lemmatizer.lemmatize(word.lower(), wn_tag) if wn_tag else word.lower()

        replaced = False

        # 1. Tone map replacement (highest priority)
        if word.lower() in tone_map and replace_count < max_replacements:
            replacement = tone_map[word.lower()]
            # Preserve original capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            new_sentence.append(replacement)
            replace_count += 1
            replaced = True

        # 2. WordNet synonym replacement (only for content words)
        elif (
            wn_tag and
            word.lower() not in stop_words and
            word.lower() not in protected_words and
            word.isalpha() and
            replace_count < max_replacements and
            not word[0].isupper() and
            len(word) > 3
        ):
            synonyms = get_synonyms(base_word, wn_tag, tone)
            if synonyms:
                chosen = random.choice(synonyms)
                new_sentence.append(chosen)
                replace_count += 1
                replaced = True

        if not replaced:
            new_sentence.append(word)

    result = " ".join(new_sentence)

    # Clean up tokenization spacing artifacts
    result = result.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" '", "'")

    return result