import streamlit as st
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import heapq

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Advanced Sentiment Analyzer", page_icon="📊")

st.title("📊 Advanced Sentiment & Review Analyzer")

# -----------------------------------
# SUMMARIZATION FUNCTION (Lightweight)
# -----------------------------------

def summarize_text(text, num_sentences=3):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    freq = defaultdict(int)
    for word in words:
        if word not in stop_words:
            freq[word] += 1

    sentences = sent_tokenize(text)
    sentence_scores = {}

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq[word]
                else:
                    sentence_scores[sentence] += freq[word]

    summary_sentences = heapq.nlargest(
        num_sentences,
        sentence_scores,
        key=sentence_scores.get
    )

    return " ".join(summary_sentences)


# -----------------------------------
# TEXT ANALYSIS SECTION
# -----------------------------------

st.header("1️⃣ Analyze Your Text")

text = st.text_area("Enter paragraph here")

def highlight_text(text):
    words = text.split()
    highlighted = ""
    for word in words:
        score = sia.polarity_scores(word)['compound']
        if score > 0:
            highlighted += f"<span style='color:green'>{word}</span> "
        elif score < 0:
            highlighted += f"<span style='color:red'>{word}</span> "
        else:
            highlighted += word + " "
    return highlighted

if st.button("Analyze Text"):
    if text:
        score = sia.polarity_scores(text)

        st.write("### Overall Sentiment Score")
        st.write(score)

        # Highlight words
        st.write("### Highlighted Text")
        st.markdown(highlight_text(text), unsafe_allow_html=True)

        # Summarization
        if len(text) > 50:
            summary = summarize_text(text)
            st.write("### Summary")
            st.write(summary)


# -----------------------------------
# WEB SCRAPING SECTION
# -----------------------------------

st.header("2️⃣ Analyze Reviews from URL")

url = st.text_input("Paste Product / Website URL")

if st.button("Analyze Reviews"):
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            reviews = soup.find_all("p")

            positive = 0
            negative = 0

            for review in reviews[:20]:  # limit to 20
                review_text = review.get_text()
                score = sia.polarity_scores(review_text)

                if score['compound'] > 0:
                    positive += 1
                elif score['compound'] < 0:
                    negative += 1

            st.write("### Review Analysis Result")
            st.success(f"Positive Reviews: {positive}")
            st.error(f"Negative Reviews: {negative}")

        except:
            st.error("Could not fetch reviews. Website may block scraping.")