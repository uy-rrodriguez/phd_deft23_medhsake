# Suppress FutureWarning messages
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from itertools import chain
import json

import streamlit as st
import pandas as pd

from config import *


# @st.cache_data
def load_data(path):
    data = pd.read_json(path)
    return data


@st.cache_data
def filter_data(df: pd.DataFrame, filters: pd.DataFrame) -> pd.DataFrame:
    return df[filters]


def sync_query_params(key, default):
    def handler():
        # Update query params from session value
        if st.session_state[key] != default:
            st.query_params[key] = st.session_state[key]
        elif key in st.query_params:
            del st.query_params[key]
        # Also reset index when filters change
        if "item_idx" in st.session_state:
            del st.session_state["item_idx"]
    return handler


def save_tags(df_tags, tags, key):
    def handler():
        # Get new tag value
        tags[key] = st.session_state[f"{tags['id']}_{key}"]
        # Concat frames, keep last value when tags already exist
        df_tags_concat = pd.concat(
            [df_tags, pd.DataFrame([tags], index=[tags["id"]])],
            axis=0, join="outer",
        ).groupby(level=0).agg("last")
        # st.write(df_tags_concat)
        # df_tags_concat.to_json(TAGS_PATH, orient="index", indent=4)
        # Save to file removing empty values
        df_tags_dict = {
            id: {
                k: v
                for k, v in data.items()
                if v is not None and v != TAGS_EMPTY
            }
            for id, data in df_tags_concat.to_dict(orient="index").items()
        }
        # Also remove items where only the id is not empty
        df_tags_dict = {
            id: data
            for id, data in df_tags_dict.items()
            if len(data.keys()) > 1
        }
        with open(TAGS_PATH, "w") as fp:
            json.dump(df_tags_dict, fp, indent=4)
    return handler


def btn_nav_click(next: bool = True):
    def handler():
        st.session_state.item_idx = item_idx + (1 if next else -1 ) * 1
    return handler


st.set_page_config(
    page_title="QCM annotator", page_icon=None,
    layout="wide", initial_sidebar_state="auto",
    menu_items=None)


df = load_data(DATA_PATH)
df_tags = pd.read_json(TAGS_PATH, orient="index")


# Custom styles
st.markdown("""
<style>
    div[class*="st-key-row"] {
        flex-direction: row;
    }
    div[class*="st-key-row"] > div {
        display: inline;
        width: auto;
    }
    div[class*="st-key-row"] > div > div {
        display: inline;
        width: auto;
    }

    div.st-key-row_edit {
        display: grid;
        grid-auto-flow: column;
    }
    .st-key-toggle_edit {
        justify-self: flex-end;
    }

    .st-key-tags input[type="radio"][checked][disabled] + div p {
        color: rgb(49, 51, 63);
        /*color: rgb(255, 75, 75);*/
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


################################################################################
# Sidebar with filters                                                         #
################################################################################

with st.sidebar:
    st.subheader("Filters")

    # Base filters (filters based on main datasource)
    # item_idx = st.text_input("Item")
    years = sorted([0] + df["year"].unique().tolist())
    topics = [""] + sorted(
        set(
            chain.from_iterable(
                df["topics"].tolist())))
    filters = {
        "id": st.text_input(
            "ID",
            key="id",
            value=st.query_params.get("id", None),
            on_change=sync_query_params("id", "")),
        "question": st.text_input(
            "Question text",
            key="question",
            value=st.query_params.get("question", None),
            on_change=sync_query_params("question", "")),
        "year": st.selectbox(
            "Year",
            key="year",
            options=years,
            format_func=lambda x: x if x else "",
            index=years.index(int(st.query_params.get("year", 0))),
            on_change=sync_query_params("year", 0)),
        "topic": st.selectbox(
            "Topic",
            key="topic",
            options=topics,
            index=topics.index(st.query_params.get("topic", "")),
            on_change=sync_query_params("topic", "")),
        "num_answers": st.number_input(
            "Correct answers",
            key="num_answers",
            step=1, min_value=0, max_value=5,
            value=int(st.query_params.get("num_answers", 0)),
            on_change=sync_query_params("num_answers", 0)),
        "difficulty": st.slider(
            "Difficulty", 0, 100, # min: 0, max: 100
            key="difficulty",
            value=[
                int(x)
                for x in st.query_params.get_all("difficulty") or (0, 100)
            ],
            on_change=sync_query_params("difficulty", (0, 100))),
    }

    # Tag filters (extra filters that look into tags assigned to questions)
    st.subheader("Tags")
    with st.expander("Search by tags"):
        filters_tags = {}
        for key, config in TAGS_CONFIG.items():
            filter_key = f"f{key}"
            filters_tags[key] = st.selectbox(
                config[0],
                key=filter_key,
                options=("",) + config[1],
                index=(("",) + config[1]).index(
                    st.query_params.get(filter_key, "")),
                on_change=sync_query_params(filter_key, ""))

    # Process filters
    filters_df = df["id"] != ""
    if filters["id"]:
        filters_df = df["id"] == filters["id"]
    if filters["question"]:
        filters_df = filters_df \
            & (df["question"].str.contains(f"(?i){filters['question']}"))
    if filters["year"]:
        filters_df = filters_df \
            & (df["year"] == filters["year"])
    if filters["topic"]:
        filters_df = filters_df \
            & (df["topics"].map(lambda v: filters["topic"] in v))
    if filters["num_answers"]:
        filters_df = filters_df \
            & (df["nbr_correct_answers"] == filters["num_answers"])
    if filters["difficulty"]:
        filters_df = filters_df \
            & (df["medshake_difficulty"] >= filters["difficulty"][0] / 100) \
            & (df["medshake_difficulty"] <= filters["difficulty"][1] / 100)

    # Filters from tags
    # These filters are done via a negation, so _df_tags will contain all items
    # that do not match the selected tags. Then, the main _df is filtered to
    # keep all items that do not appear in _df_tags.
    filters_df_tags = None
    for key, value in filters_tags.items():
        if value and value != "":
            if value == TAGS_EMPTY:
                _filter = ~df_tags[key].isna()
            else:
                _filter = df_tags[key] != value
            if filters_df_tags is None:
                filters_df_tags = _filter
            else:
                filters_df_tags = filters_df_tags & _filter
    if filters_df_tags is not None:
        _df_tags = df_tags[filters_df_tags]
        filters_df = filters_df & ~df["id"].isin(_df_tags.index)

    # Apply filters
    _df = filter_data(df, filters_df)
    st.markdown(f"_Found {len(_df.index)} in {len(df.index)}_")

    st.subheader("Sorting")
    sort_difficulty = st.checkbox(
        "Sort by difficulty",
        key="sort_by_difficulty",
        value=st.query_params.get("sort_by_difficulty", None) == "True",
        on_change=sync_query_params("sort_by_difficulty", False))
    sort_desc = st.checkbox(
        "Sort descending",
        key="sort_desc",
        value=st.query_params.get("sort_desc", None) == "True",
        on_change=sync_query_params("sort_desc", False))

    if sort_difficulty:
        _df = _df.sort_values(by="medshake_difficulty", ascending=not sort_desc)


################################################################################
# Main page                                                                    #
################################################################################

if len(_df.index) == 0:
    st.markdown("**No item found**")
    st.stop()

item_idx = st.query_params.get(
    "item_idx",
    st.session_state.get("item_idx", 0))
item = _df.iloc[item_idx]
if item.id in df_tags.index:
    tags = df_tags.loc[item.id].dropna().to_dict()
else:
    tags = {"id": item.id}

# Toggle to enable tag modification
with st.container(key="row_edit"):
    st.markdown(f"_Item {item_idx + 1} of {len(_df.index)}_")
    edit_mode = st.toggle("Edit mode", key="toggle_edit")

# Question details
st.markdown(f"**ID**: {item.id}")
st.markdown(f"**Question**: {item.question}")
answers = [
    f"\n  - **({k}) {v}** <-"
    if k in item.correct_answers
    else f"\n  - ({k}) {v}"
    for k, v in item.answers.items()
]
st.markdown(f"**Answers**:\n{''.join(answers)}")
st.markdown(f"**Difficulty**: {item.medshake_difficulty}")
st.markdown(f"**Year**: {item.year}")
st.markdown(f"**Topics**: {', '.join(item.topics)}")


# Tagging tools
st.subheader("Tags")
with st.container(key="tags"):
    cols_tags = st.columns(len(TAGS_CONFIG))
    for col, key in zip(cols_tags, TAGS_CONFIG):
        config = TAGS_CONFIG[key]
        with col:
            st.radio(
                config[0],
                key=f"{item.id}_{key}",
                options=config[1],
                index=config[1].index(tags.get(key, TAGS_EMPTY)),
                horizontal=False,
                disabled=not edit_mode,
                on_change=save_tags(df_tags, tags, key))


# Navigation between questions
with st.container(key="row_nav"):
    st.button(
        "< Prev", key="btn_prev", on_click=btn_nav_click(False),
        disabled=item_idx <= 0)
    st.button(
        "Next >", key="btn_next", on_click=btn_nav_click(),
        disabled=item_idx >= len(_df.index) - 1)

# Raw data table
# st.subheader("Raw data")
# st.text(f"Count: {len(_df.index)}")

# Highlight one row
# _df = _df.style.apply(
#     lambda x:
#         ['background-color: rgb(255, 75, 75)'] * len(x)
#         if x.id == item.id
#         else [''] * len(x),
#     axis=1,
# )

# st.dataframe(_df)
