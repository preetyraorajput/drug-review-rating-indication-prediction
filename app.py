"""
Drug Review Predictor — deployment app for the Capstone 2 LSTM models.

Run with:  streamlit run app.py
Needs, in the same folder: rating_model.keras, indication_model.keras,
tokenizer.pkl, label_encoder.pkl, drug_encoder.pkl
(all produced by the export cell at the end of the training notebook)
"""

import re
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

st.set_page_config(page_title="Drug Review Predictor", page_icon=":pill:", layout="centered")

# ---------------------------------------------------------------------------
# Styling — clean look, safe to screenshot/share on GitHub
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

.stApp {
    background-color: #f4f1fb;
}

h1 {
    color: #2c2150;
    font-weight: 700;
    letter-spacing: -0.02em;
}

[data-testid="stCaptionContainer"] {
    color: #5b5470;
}

[data-testid="stForm"] {
    background-color: #ffffff;
    border: 1px solid #e0dcf0;
    border-radius: 14px;
    padding: 28px 28px 12px 28px;
    box-shadow: 0 1px 3px rgba(44, 33, 80, 0.06);
}

label {
    font-weight: 600 !important;
    color: #2c2150 !important;
}

.stButton>button, [data-testid="stFormSubmitButton"]>button {
    background-color: #6d4fc4;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.55em 1.6em;
    font-weight: 600;
    transition: background-color 0.15s ease;
}
.stButton>button:hover, [data-testid="stFormSubmitButton"]>button:hover {
    background-color: #5a3fa8;
    color: #ffffff;
}

hr {
    border-color: #e0dcf0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    rating_model = load_model("rating_model.keras")
    indication_model = load_model("indication_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("drug_encoder.pkl", "rb") as f:
        drug_encoder = pickle.load(f)
    return rating_model, indication_model, tokenizer, label_encoder, drug_encoder


rating_model, indication_model, tokenizer, label_encoder, drug_encoder = load_everything()

MAXLEN = 100  # same cutoff used during training

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Same cleaning pipeline used to prepare the training data."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " <NUM> ", text)
    text = re.sub(r"[^A-Za-z\s<>]", " ", text)
    text = text.lower()
    text = " ".join(text.split())
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def main():
    st.title("Drug Review Predictor")
    st.caption(
        "Type a real-style drug review below. An LSTM neural network reads the text and "
        "predicts the rating a patient would likely give (1-10) and the medical condition "
        "the review is about. Educational/illustrative project, not medical advice."
    )

    with st.form("review_form"):
        review_text = st.text_area(
            "Review text",
            height=160,
            placeholder=(
                "e.g. This medication has really helped with my anxiety, though I did "
                "notice some nausea during the first week."
            ),
        )
        drug = st.selectbox("Which drug is this review about?", options=sorted(drug_encoder.classes_))
        submitted = st.form_submit_button("Predict")

    if submitted:
        if not review_text.strip():
            st.warning("Please enter a review first.")
            return

        cleaned = clean_text(review_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAXLEN)

        pred_rating = float(rating_model.predict(pad, verbose=0)[0][0])
        pred_rating = float(np.clip(pred_rating, 1, 10))

        drug_id = drug_encoder.transform([drug])[0]
        drug_id_arr = np.array([[drug_id]])
        pred_probs = indication_model.predict([pad, drug_id_arr], verbose=0)[0]
        pred_class_idx = int(pred_probs.argmax())
        pred_indication = label_encoder.classes_[pred_class_idx]
        pred_confidence = float(pred_probs[pred_class_idx])

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Rating", f"{pred_rating:.1f} / 10")
        with col2:
            st.metric("Predicted Condition", pred_indication)
        st.caption(f"Model confidence in this condition: {pred_confidence:.0%}")


if __name__ == "__main__":
    main()
