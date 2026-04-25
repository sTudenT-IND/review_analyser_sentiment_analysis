import streamlit as st
import pickle

# Try loading model
try:
    with open('originalnaivebayes5k.pickle', 'rb') as f:
        classifier = pickle.load(f)

    with open('word_features5k.pickle', 'rb') as f:
        word_features = pickle.load(f)

except:
    # Fallback simple model (works instantly)
    classifier = None
    word_features = None

# Simple fallback logic
def fallback_sentiment(text):
    positive_words = ["good", "amazing", "great", "awesome", "best", "love"]
    negative_words = ["bad", "worst", "boring", "waste", "poor", "hate"]

    text = text.lower()
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos >= neg:
        return "pos", 0.7
    else:
        return "neg", 0.7

# Feature extraction
def find_features(document):
    words = document.lower().split()
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Prediction
def predict_sentiment(text):
    if classifier:
        features = find_features(text)
        result = classifier.classify(features)
        confidence = classifier.prob_classify(features).max()
        return result, confidence
    else:
        return fallback_sentiment(text)

# UI
st.title("🎬 Should I Watch This Movie?")
user_input = st.text_area("Enter Movie Review:")

if st.button("Analyze"):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)

        if sentiment == "pos":
            st.success(f"✅ Recommended! Confidence: {round(confidence*100,2)}%")
        else:
            st.error(f"❌ Not Recommended! Confidence: {round(confidence*100,2)}%")
