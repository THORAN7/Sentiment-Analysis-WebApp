import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER (first time only)
nltk.download('vader_lexicon')

# Create analyzer
sia = SentimentIntensityAnalyzer()

# Web App Title
st.title("Sentiment Analysis Web App")

# User input
user_text = st.text_area("Enter your text here:")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        score = sia.polarity_scores(user_text)

        if score['compound'] > 0:
            sentiment = "Positive 😊"
            st.success(f"Sentiment: {sentiment}")
        elif score['compound'] < 0:
            sentiment = "Negative 😡"
            st.error(f"Sentiment: {sentiment}")
        else:
            sentiment = "Neutral 😐"
            st.info(f"Sentiment: {sentiment}")

        st.write("### Sentiment Score:")
        st.write(score)