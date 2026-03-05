import streamlit as st
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import heapq
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ---------------- NLTK SETUP ----------------

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

sia = SentimentIntensityAnalyzer()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Review Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- UI STYLE ----------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#667eea,#764ba2);
color:white;
}

section[data-testid="stSidebar"]{
background:linear-gradient(180deg,#1f2937,#111827);
}

h1{
text-align:center;
font-size:40px;
font-weight:800;
}

.stButton>button{
background:linear-gradient(135deg,#00c6ff,#0072ff);
color:white;
border-radius:12px;
border:none;
padding:10px 25px;
font-weight:600;
}

.stButton>button:hover{
background:linear-gradient(135deg,#ff7e5f,#feb47b);
transform:scale(1.05);
}

textarea{
background:rgba(255,255,255,0.1)!important;
color:white!important;
border-radius:10px!important;
}

.stTextInput input{
background:rgba(255,255,255,0.1);
color:white;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("📊 AI Product Review Intelligence Dashboard")
st.divider()

# ---------------- SENTIMENT GAUGE FUNCTION ----------------

def sentiment_gauge(score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [-1,1]},
            'bar': {'color': "#00c6ff"},
            'steps': [
                {'range': [-1,-0.3], 'color': "#ff4b4b"},
                {'range': [-0.3,0.3], 'color': "#f1c40f"},
                {'range': [0.3,1], 'color': "#2ecc71"}
            ]
        }
    ))

    fig.update_layout(height=350)

    return fig


# ---------------- SUMMARIZATION FUNCTION ----------------

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
                sentence_scores[sentence] = sentence_scores.get(sentence,0) + freq[word]

    summary_sentences = heapq.nlargest(
        num_sentences,
        sentence_scores,
        key=sentence_scores.get
    )

    return " ".join(summary_sentences)


# ---------------- KEY POINT EXTRACTION ----------------

def extract_key_points(text):

    sentences = sent_tokenize(text)

    positive_points = []
    negative_points = []
    suggestions = []

    for sentence in sentences:

        score = sia.polarity_scores(sentence)['compound']

        if score > 0.4:
            positive_points.append(sentence)

        elif score < -0.4:
            negative_points.append(sentence)

        if any(word in sentence.lower() for word in
               ["should","improve","could","needs","better","recommend"]):
            suggestions.append(sentence)

    return positive_points, negative_points, suggestions


# ---------------- TEXT ANALYZER ----------------

st.header("🧠 Text Sentiment Analyzer")

text = st.text_area("Enter paragraph here")

def highlight_text(text):

    words = text.split()
    highlighted = ""

    for word in words:

        score = sia.polarity_scores(word)['compound']

        if score > 0:
            highlighted += f"<span style='color:#00ff9d;font-weight:600'>{word}</span> "

        elif score < 0:
            highlighted += f"<span style='color:#ff4b4b;font-weight:600'>{word}</span> "

        else:
            highlighted += word + " "

    return highlighted


if st.button("Analyze Text") and text:

    score = sia.polarity_scores(text)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Sentiment Meter")

        gauge = sentiment_gauge(score['compound'])
        st.plotly_chart(gauge, use_container_width=True)

        if score['compound'] > 0:
            st.success("😊 Positive Sentiment")

        elif score['compound'] < 0:
            st.error("😡 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")

    with col2:

        if len(text) > 50:

            summary = summarize_text(text)

            st.subheader("📝 Summary")
            st.write(summary)

    st.subheader("✨ Highlighted Sentiment Words")

    st.markdown(highlight_text(text), unsafe_allow_html=True)


# ---------------- WEBSITE REVIEW ANALYZER ----------------

st.header("🌐 Website Review Analyzer")

url = st.text_input("Paste Product / Website URL")

if st.button("Analyze Reviews"):

    if url:

        with st.spinner("🤖 AI analyzing reviews..."):

            try:

                response = requests.get(url)

                soup = BeautifulSoup(response.text, "html.parser")

                reviews = soup.find_all("p")

                positive_count = 0
                negative_count = 0

                all_positive_points = []
                all_negative_points = []
                all_suggestions = []

                progress = st.progress(0)

                for i, review in enumerate(reviews[:20]):

                    progress.progress((i+1)/20)

                    review_text = review.get_text().strip()

                    if len(review_text) < 30:
                        continue

                    score = sia.polarity_scores(review_text)

                    if score['compound'] > 0:
                        positive_count += 1

                    elif score['compound'] < 0:
                        negative_count += 1

                    pos, neg, sug = extract_key_points(review_text)

                    all_positive_points.extend(pos)
                    all_negative_points.extend(neg)
                    all_suggestions.extend(sug)

                st.subheader("📊 Sentiment Dashboard")

                m1, m2 = st.columns(2)

                with m1:
                    st.metric("🟢 Positive Reviews", positive_count)

                with m2:
                    st.metric("🔴 Negative Reviews", negative_count)

                data = pd.DataFrame({
                    "Sentiment":["Positive","Negative"],
                    "Count":[positive_count,negative_count]
                })

                pie = px.pie(
                    data,
                    names="Sentiment",
                    values="Count",
                    hole=0.45,
                    title="Review Sentiment Distribution",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive":"#2ecc71",
                        "Negative":"#ff4b4b"
                    }
                )

                st.plotly_chart(pie, use_container_width=True)

                bar = px.bar(
                    data,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    title="Sentiment Comparison",
                    color_discrete_map={
                        "Positive":"#2ecc71",
                        "Negative":"#ff4b4b"
                    }
                )

                st.plotly_chart(bar, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("🟢 Positive Insights")

                    for point in all_positive_points[:5]:
                        st.success(point)

                with col2:

                    st.subheader("🔴 Negative Insights")

                    for point in all_negative_points[:5]:
                        st.error(point)

                st.subheader("💡 Suggestions")

                for suggestion in all_suggestions[:5]:
                    st.info(suggestion)

            except:

                st.error("Website blocked scraping or invalid URL.")