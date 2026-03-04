import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

st.title("😊 Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment.")

text = st.text_area("Your Text")

if st.button("Analyze"):
    if text:
        score = sia.polarity_scores(text)
        compound = score['compound']

        if compound > 0:
            st.success("Positive Sentiment 😊")
        elif compound < 0:
            st.error("Negative Sentiment 😡")
        else:
            st.info("Neutral Sentiment 😐")

        st.metric("Sentiment Score", compound)
    else:
        st.warning("Please enter text first!")


# Dont run it as normal program u Noob, Type this in the terminal "streamlit run web_app.py"