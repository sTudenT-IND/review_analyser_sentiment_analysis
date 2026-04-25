
import streamlit as st
import pickle

# Load model
classifier = pickle.load(open('naivebayes.pickle', 'rb'))
word_features = pickle.load(open('word_features.pickle', 'rb'))

# Feature extraction
def find_features(document):
    words = document.lower().split()
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Prediction function
def predict_sentiment(text):
    features = find_features(text)
    result = classifier.classify(features)
    confidence = classifier.prob_classify(features).max()
    return result, confidence

# UI
st.set_page_config(page_title="Movie Review Analyzer", page_icon="🎬")

st.title("🎬 Should I Watch This Movie?")
st.write("Enter a movie review and get recommendation")

user_input = st.text_area("Enter Movie Review:")

if st.button("Analyze"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)

        if sentiment == "pos":
            st.success(f"✅ Recommended to Watch!\nConfidence: {round(confidence*100,2)}%")
        else:
            st.error(f"❌ Not Recommended!\nConfidence: {round(confidence*100,2)}%")
    else:
        st.warning("Please enter a review!")
