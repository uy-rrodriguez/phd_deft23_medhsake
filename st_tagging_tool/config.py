DATASETS_DEFAULT = "train"
DATASETS = {
    "train": (
        "data/train-with-medshake.json",
        "out/tags-train-with-medshake.json",
    ),
    "dev": (
        "data/dev-with-medshake.json",
        "out/tags-dev-with-medshake.json",
    ),
    "test": (
        "data/test-medshake-score.json",
        "out/tags-test-medshake-score.json",
    ),
}

TAGS_EMPTY = "n/a"
TAGS_OPTS_NEGATION = (TAGS_EMPTY, "no", "yes")
TAGS_OPTS_COMPOSITION = (TAGS_EMPTY, "no", "yes")
TAGS_OPTS_HIGHLIGHT = (TAGS_EMPTY, "no", "yes")
TAGS_OPTS_MODE = (TAGS_EMPTY, "affirmation", "question", "instruction")
TAGS_OPTS_INTRUDER = (TAGS_EMPTY, "no", "yes")
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
    "tag_intruder": (
        "Choose intruder option?",
        TAGS_OPTS_INTRUDER,
    ),
    "tag_answer": (
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
