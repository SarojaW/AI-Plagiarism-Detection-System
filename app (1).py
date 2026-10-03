import streamlit as st
import pandas as pd
import pickle
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Plagiarism Detection System",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open("plagiarismmodel.pkl", "rb") as f:
        model = pickle.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return model, embedder


model, embedder = load_model()


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    return text


# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------

def extract_features(source_text, submitted_text):

    embeddings = embedder.encode(
        [source_text, submitted_text]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    source_words = set(source_text.split())
    submitted_words = set(submitted_text.split())

    union = source_words | submitted_words

    if union:
        word_overlap = (
            len(source_words & submitted_words)
            / len(union)
        )
    else:
        word_overlap = 0.0

    len_diff = abs(
        len(source_words) -
        len(submitted_words)
    )

    return [similarity, word_overlap, len_diff]


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🔍 AI-Based Plagiarism Detection System")

st.write(
    "Compare a source text with a submitted text "
    "using semantic similarity and machine learning."
)

st.divider()


col1, col2 = st.columns(2)


with col1:

    st.subheader("📄 Source Text")

    source_text = st.text_area(
        "Enter the original/source text:",
        height=250,
        placeholder="Paste the original text here..."
    )


with col2:

    st.subheader("📝 Submitted Text")

    submitted_text = st.text_area(
        "Enter the submitted text:",
        height=250,
        placeholder="Paste the submitted text here..."
    )


st.divider()


if st.button("🔍 Check Plagiarism", use_container_width=True):

    if not source_text.strip() or not submitted_text.strip():

        st.warning(
            "Please enter both source text and submitted text."
        )

    else:

        # Clean text
        source_clean = clean_text(source_text)
        submitted_clean = clean_text(submitted_text)

        # Extract features
        features = extract_features(
            source_clean,
            submitted_clean
        )

        similarity = features[0]
        word_overlap = features[1]
        len_diff = features[2]

        # Convert features to DataFrame
        feature_df = pd.DataFrame(
            [features],
            columns=[
                "similarity",
                "word_overlap",
                "len_diff"
            ]
        )

        # ML prediction
        prediction = model.predict(feature_df)[0]

        # Final decision
        if prediction == 1 or similarity >= 0.60:

            st.error("🚨 Plagiarism Detected")

        else:

            st.success("✅ No Plagiarism Detected")


        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Semantic Similarity",
                f"{similarity * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Word Overlap",
                f"{word_overlap * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Unique Word Length Difference",
                f"{len_diff}"
            )


        st.progress(
            min(max(float(similarity), 0.0), 1.0)
        )


        st.info(
            "The system uses Sentence Transformer embeddings, "
            "cosine similarity, word overlap and a trained "
            "machine-learning classifier."
        )