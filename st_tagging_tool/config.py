DATA_PATH = "data/test-medshake-score.json"
TAGS_PATH = "out/tags.json"
TAGS_EMPTY = "n/a"
TAGS_OPTS_NEGATION = (TAGS_EMPTY, "no", "yes")
TAGS_OPTS_COMPOSITION = (TAGS_EMPTY, "no", "yes")
TAGS_OPTS_HIGHLIGHT = (TAGS_EMPTY, "no", "yes")
TAGS_OPTS_MODE = (TAGS_EMPTY, "affirmation", "question", "instruction")
TAGS_OPTS_POSITIVE = (TAGS_EMPTY, "positive", "negative")
TAGS_OPTS_SINGLE = (TAGS_EMPTY, "undefined", "single", "multiple")
TAGS_OPTS_TYPOLOGY = (TAGS_EMPTY, "knowledge", "comprehension", "application",
                      "analysis")
TAGS_CONFIG = {
    "tag_negation": (
        "Has negation?",
        TAGS_OPTS_NEGATION,
    ),
    "tag_composition": (
        "Requires composition?",
        TAGS_OPTS_COMPOSITION,
    ),
    "tag_mode": (
        "Sentence mode",
        TAGS_OPTS_MODE,
    ),
    "tag_positive": (
        "Choose positive option?",
        TAGS_OPTS_POSITIVE,
    ),
    "tag_single": (
        "Number of choices",
        TAGS_OPTS_SINGLE,
    ),
    # "tag_typology": (
    #     "Typology",
    #     TAGS_OPTS_TYPOLOGY,
    # ),
    "tag_highlight": (
        "Highlight?",
        TAGS_OPTS_HIGHLIGHT,
    ),
}
