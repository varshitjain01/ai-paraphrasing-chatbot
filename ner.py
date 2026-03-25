import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

for r in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
          'averaged_perceptron_tagger_eng', 'maxent_ne_chunker',
          'maxent_ne_chunker_tab', 'words']:
    nltk.download(r, quiet=True)

# ─────────────────────────────────────────────
# ENTITY TYPE METADATA
# ─────────────────────────────────────────────

ENTITY_META = {
    "PERSON":       {"label": "Person",        "color": "#7c6aff", "emoji": "👤"},
    "ORGANIZATION": {"label": "Organization",  "color": "#f472b6", "emoji": "🏢"},
    "GPE":          {"label": "Location",      "color": "#34d399", "emoji": "📍"},
    "LOCATION":     {"label": "Location",      "color": "#34d399", "emoji": "📍"},
    "FACILITY":     {"label": "Facility",      "color": "#fb923c", "emoji": "🏛️"},
    "GSP":          {"label": "Geo-Political", "color": "#60a5fa", "emoji": "🌍"},
}

DEFAULT_META = {"label": "Entity", "color": "#94a3b8", "emoji": "🔷"}

# ─────────────────────────────────────────────
# NER FUNCTION
# ─────────────────────────────────────────────

def extract_entities(text):
    tokens  = word_tokenize(text)
    tagged  = pos_tag(tokens)
    chunked = ne_chunk(tagged)

    entities = []
    seen = set()

    for subtree in chunked:
        if isinstance(subtree, Tree):
            entity_name = " ".join(word for word, tag in subtree.leaves())
            entity_type = subtree.label()

            key = (entity_name.lower(), entity_type)
            if key not in seen:
                seen.add(key)
                meta = ENTITY_META.get(entity_type, DEFAULT_META)
                entities.append({
                    "name":  entity_name,
                    "type":  entity_type,
                    "label": meta["label"],
                    "color": meta["color"],
                    "emoji": meta["emoji"],
                })

    return {
        "entities": entities,
        "count": len(entities),
        "types_found": list(set(e["type"] for e in entities)),
    }