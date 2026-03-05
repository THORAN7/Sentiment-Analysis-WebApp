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

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Review Analyzer", "👨‍💻 About"]
)

theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

# ---------------- DARK MODE ----------------

if theme:

    st.markdown("""
    <style>

    .stApp{
    background: linear-gradient(135deg,#667eea,#764ba2);
    color:white;
    }

    section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1f2937,#111827);
    color:white;
    }

    textarea, .stTextInput input{
    background:rgba(255,255,255,0.1)!important;
    color:white!important;
    }

    .card{
    padding:25px;
    border-radius:20px;
    background:rgba(255,255,255,0.1);
    backdrop-filter:blur(12px);
    box-shadow:0px 8px 25px rgba(0,0,0,0.3);
    margin-bottom:20px;
    }

    </style>
    """, unsafe_allow_html=True)

# ---------------- LIGHT MODE ----------------

else:

    st.markdown("""
    <style>

    .stApp{
    background:#f5f7fb;
    color:#111111;
    }

    section[data-testid="stSidebar"]{
    background:#ffffff;
    color:#111111;
    }

    h1,h2,h3,h4,h5,h6,p,span,label,div{
    color:#111111 !important;
    }

    textarea,.stTextInput input{
    background:#ffffff!important;
    color:#111111!important;
    border:1px solid #ddd;
    }

    .card{
    padding:25px;
    border-radius:20px;
    background:white;
    box-shadow:0px 8px 25px rgba(0,0,0,0.1);
    margin-bottom:20px;
    }

    </style>
    """, unsafe_allow_html=True)

# ---------------- COMMON BUTTON STYLE ----------------

st.markdown("""
<style>

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
transition:0.3s;
}

.stButton>button:hover{
background:linear-gradient(135deg,#ff7e5f,#feb47b);
transform:scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# PAGE 1 : MAIN REVIEW ANALYZER
# =====================================================

if page == "📊 Review Analyzer":

    st.title("🤖 AI Product Review Intelligence Dashboard")
    st.caption("Advanced NLP-powered review analytics platform")
    st.divider()

# ---------------- TEXT SUMMARIZATION ----------------

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

            st.subheader("📊 Sentiment Score")
            st.json(score)

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

                    for review in reviews[:20]:

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
                        title="Review Sentiment Distribution"
                    )

                    st.plotly_chart(pie, use_container_width=True)

                    bar = px.bar(
                        data,
                        x="Sentiment",
                        y="Count",
                        color="Sentiment",
                        title="Sentiment Comparison"
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

# =====================================================
# PAGE 2 : ABOUT
# =====================================================

elif page == "👨‍💻 About":

    st.title("👨‍💻 About the Developer")

    col1, col2 = st.columns([1,2])

    with col1:
        st.image("assets/profile.jpeg", width=220)

    with col2:

        st.markdown("""
        ## Noob Nimisha

        **AI / ML Developer**

        Creator of the **AI Product Review Intelligence Dashboard**

        ### Skills
        • Playing GAMES like NOOB                                                              
        • Can study until 2 AM                          
        • Python  
        • NLP  
        • Data Visualization  
        • Machine Learning                
                                                       

        ### Links
        GitHub: https://github.com/Nimisha-Anand                                 
        Linkedin: idk she didnt GAVE
        """)