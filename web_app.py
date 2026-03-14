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
import time

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

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

# ---------------- AMAZON SCRAPER ----------------

def convert_to_review_url(url):

    if "/product-reviews/" in url:
        return url

    if "/dp/" in url:
        product_id = url.split("/dp/")[1].split("/")[0]
        return f"https://www.amazon.in/product-reviews/{product_id}"

    if "/gp/product/" in url:
        product_id = url.split("/gp/product/")[1].split("/")[0]
        return f"https://www.amazon.in/product-reviews/{product_id}"

    return url


def scrape_amazon_reviews(url, max_reviews=100):

    url = convert_to_review_url(url)

    st.write("🔗 Review Page:", url)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.get(url)
    time.sleep(3)

    reviews = []

    for page in range(1,6):

        st.write(f"📄 Scraping Page {page}")

        blocks = driver.find_elements(By.CSS_SELECTOR,'[data-hook="review"]')

        for block in blocks:

            try:

                review = block.find_element(
                    By.CSS_SELECTOR,
                    '[data-hook="review-body"]'
                ).text

                if len(review) > 20:
                    reviews.append(review)

            except:
                pass

            if len(reviews) >= max_reviews:
                break

        if len(reviews) >= max_reviews:
            break

        try:

            next_button = driver.find_element(By.CSS_SELECTOR,"li.a-last a")
            next_button.click()
            time.sleep(3)

        except:
            break

    driver.quit()

    reviews = list(set(reviews))

    st.success(f"✅ Reviews Collected: {len(reviews)}")

    with st.expander("👀 Preview Reviews"):
        for r in reviews[:5]:
            st.write("•", r)

    return reviews


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
    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    /* MAIN BACKGROUND */

    .stApp{
        background: linear-gradient(135deg,#ffffff,#f3f6fb,#eef2ff);
        color:#111111;
    }

    /* SIDEBAR */

    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg,#ffffff,#eef2ff);
        color:#111111;
    }

    /* HEADINGS */

    h1, h2, h3, h4, h5, h6{
        color:#0f172a !important;
    }

    /* TEXT */

    p, span, label, div{
        color:#111111;
    }

    /* TEXT INPUT */

    textarea, input{
        background-color:white !important;
        color:#111111 !important;
        border-radius:10px !important;
        border:1px solid #dbeafe !important;
    }

    /* METRIC BOX */

    div[data-testid="metric-container"]{
        background:white;
        border-radius:12px;
        padding:10px;
        border:1px solid #e2e8f0;
        box-shadow:0px 2px 8px rgba(0,0,0,0.05);
    }

    /* EXPANDER */

    details{
        background:white;
        border-radius:10px;
        padding:10px;
        border:1px solid #e5e7eb;
    }

    /* BUTTON */

    .stButton>button{
        background: linear-gradient(90deg,#3b82f6,#60a5fa);
        color:white;
        border:none;
        border-radius:10px;
        padding:10px 20px;
        font-weight:bold;
        transition:0.3s;
    }

    .stButton>button:hover{
        background: linear-gradient(90deg,#2563eb,#3b82f6);
        transform:scale(1.05);
    }

    /* SUCCESS / ERROR BOX */

    .stAlert{
        border-radius:10px;
    }

    </style>
    """, unsafe_allow_html=True)
# =====================================================
# PAGE 1
# =====================================================

if page == "📊 Review Analyzer":

    st.title("🤖 AI Product Review Intelligence Dashboard")
    st.caption("Advanced NLP-powered review analytics platform")

# ---------------- SUMMARIZER ----------------

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
                    sentence_scores[sentence] = sentence_scores.get(sentence,0)+freq[word]

        summary_sentences = heapq.nlargest(
            num_sentences,
            sentence_scores,
            key=sentence_scores.get
        )

        return " ".join(summary_sentences)

# ---------------- TEXT ANALYZER ----------------

    st.header("🧠 Text Sentiment Analyzer")

    text = st.text_area("Enter paragraph here")

    if st.button("Analyze Text") and text:

        score = sia.polarity_scores(text)

        st.subheader("Sentiment Score")
        st.json(score)

        if score['compound'] >= 0.05:
            st.success("😊 Positive Sentiment")

        elif score['compound'] <= -0.05:
            st.error("😡 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")

        if len(text) > 50:
            summary = summarize_text(text)

            st.subheader("📝 Summary")
            st.write(summary)

# ---------------- REVIEW ANALYZER ----------------

    st.header("🌐 Website / Amazon Review Analyzer")

    url = st.text_input("Paste Product / Website URL")

    if st.button("Analyze Reviews"):

        if url:

            with st.spinner("🤖 AI analyzing reviews..."):

                try:

                    reviews = []

                    if "amazon" in url:

                        reviews = scrape_amazon_reviews(url)

                    else:

                        headers = {"User-Agent":"Mozilla/5.0"}

                        response = requests.get(url,headers=headers)
                        soup = BeautifulSoup(response.text,"html.parser")

                        for r in soup.find_all("p"):

                            text = r.get_text().strip()

                            if len(text) > 30:
                                reviews.append(text)

                    reviews = list(set(reviews))

                    positive_reviews = []
                    negative_reviews = []

                    for review in reviews:

                        score = sia.polarity_scores(review)

                        if score['compound'] >= 0.05:
                            positive_reviews.append(review)

                        elif score['compound'] <= -0.05:
                            negative_reviews.append(review)

                    positive_count = len(positive_reviews)
                    negative_count = len(negative_reviews)

                    st.subheader("📊 Sentiment Dashboard")

                    m1,m2 = st.columns(2)

                    with m1:
                        st.metric("🟢 Positive Reviews",positive_count)

                    with m2:
                        st.metric("🔴 Negative Reviews",negative_count)

                    data = pd.DataFrame({
                        "Sentiment":["Positive","Negative"],
                        "Count":[positive_count,negative_count]
                    })

                    pie = px.pie(
                        data,
                        names="Sentiment",
                        values="Count",
                        hole=0.45
                    )

                    st.plotly_chart(pie,use_container_width=True)

                    bar = px.bar(
                        data,
                        x="Sentiment",
                        y="Count",
                        color="Sentiment"
                    )

                    st.plotly_chart(bar,use_container_width=True)

# ---------------- SHOW REVIEWS ----------------

                    st.divider()

                    col1,col2 = st.columns(2)

                    with col1:

                        st.subheader("🟢 Positive Reviews")

                        if positive_reviews:
                            for review in positive_reviews[:10]:
                                st.success(review)
                        else:
                            st.write("No positive reviews found")

                    with col2:

                        st.subheader("🔴 Negative Reviews")

                        if negative_reviews:
                            for review in negative_reviews[:10]:
                                st.error(review)
                        else:
                            st.write("No negative reviews found")

                except:

                    st.error("Website blocked scraping or invalid URL")

# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "👨‍💻 About":

    st.title("👨‍💻 About the Developer")

    st.markdown("""

## Noob Nimisha

AI / ML Developer  

Creator of the **AI Product Review Intelligence Dashboard**

### Skills

• Python  
• NLP  
• Data Visualization  
• Machine Learning  

### GitHub

https://github.com/Nimisha-Anand

""")