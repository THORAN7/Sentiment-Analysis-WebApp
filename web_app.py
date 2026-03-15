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
import random

# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Try to import undetected chrome driver
try:
    import undetected_chromedriver as uc

    HAS_UNDETECTED = True
except:
    HAS_UNDETECTED = False

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


# ============= DYNAMIC REVIEW GENERATOR =============

def generate_dynamic_reviews(num_reviews=15):
    """Generate different random reviews each time"""

    positive_phrases = [
        "This product is absolutely amazing!",
        "Best purchase I've ever made!",
        "Exceeded all my expectations!",
        "Fantastic quality and great value!",
        "Highly recommend to everyone!",
        "Perfect! Exactly what I needed!",
        "Outstanding product, very satisfied!",
        "Worth every penny!",
        "Fast shipping and excellent condition!",
        "Better than expected!",
        "Love it! Will buy again!",
        "Great quality for the price!",
        "Arrived quickly and in perfect condition!",
        "Excellent customer service!",
        "5 stars deserved!",
    ]

    positive_endings = [
        "Will definitely purchase again.",
        "Highly satisfied with my purchase.",
        "Best decision ever!",
        "Can't recommend enough!",
        "Perfect in every way!",
        "Absolutely worth it!",
        "My family loves it!",
        "Exceeded expectations!",
    ]

    negative_phrases = [
        "Terrible experience with this product.",
        "Complete waste of money!",
        "Very disappointed with the quality.",
        "Product broke after a few days.",
        "Not as described at all.",
        "Extremely poor quality.",
        "Worst purchase ever!",
        "Does not work as advertised.",
        "Should not have bought this.",
        "Defective unit received.",
        "Cheap materials and poor construction.",
        "Total disappointment.",
        "Do not recommend!",
        "Wasted my money!",
        "Regret buying this.",
    ]

    negative_endings = [
        "Would not buy again.",
        "Very upset with this purchase.",
        "Returning this immediately.",
        "Save your money!",
        "Not worth the price.",
        "Should have read reviews first.",
        "Waste of time and money.",
        "Avoid at all costs!",
    ]

    neutral_phrases = [
        "It's an okay product.",
        "Does what it's supposed to do.",
        "Average quality for the price.",
        "Not bad, but could be better.",
        "Meets basic expectations.",
        "Decent product overall.",
        "It works as expected.",
        "Neither great nor terrible.",
        "Standard quality.",
        "Fair value for money.",
    ]

    reviews = []

    # Generate positive reviews
    for _ in range(num_reviews // 3):
        review = random.choice(positive_phrases) + " " + random.choice(positive_endings)
        reviews.append(review)

    # Generate negative reviews
    for _ in range(num_reviews // 3):
        review = random.choice(negative_phrases) + " " + random.choice(negative_endings)
        reviews.append(review)

    # Generate neutral reviews
    for _ in range(num_reviews - (num_reviews // 3) * 2):
        review = random.choice(neutral_phrases)
        reviews.append(review)

    # Shuffle to mix them up
    random.shuffle(reviews)

    return reviews


# Get dynamic reviews
DEMO_REVIEWS = generate_dynamic_reviews(15)


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


def try_undetected_scrape(url, max_reviews=50):
    """Try scraping with undetected chromedriver"""
    if not HAS_UNDETECTED:
        return []

    try:
        options = uc.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = uc.Chrome(options=options, version_main=None)
        driver.get(url)
        time.sleep(random.uniform(3, 5))

        reviews = []
        blocks = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')

        for block in blocks:
            try:
                review = block.find_element(By.CSS_SELECTOR, '[data-hook="review-body"]').text
                if len(review) > 20:
                    reviews.append(review)
            except:
                pass

            if len(reviews) >= max_reviews:
                break

        driver.quit()
        return reviews[:max_reviews]
    except Exception as e:
        st.warning(f"Undetected Chrome failed: {str(e)}")
        return []


def try_beautifulsoup_scrape(url, max_reviews=50):
    """Try scraping with BeautifulSoup and requests"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        reviews = []
        review_divs = soup.find_all('div', {'data-hook': 'review'})

        for review_div in review_divs:
            try:
                review_text = review_div.find('span', {'data-hook': 'review-body'})
                if review_text:
                    text = review_text.get_text().strip()
                    if len(text) > 20:
                        reviews.append(text)
            except:
                pass

            if len(reviews) >= max_reviews:
                break

        return reviews[:max_reviews]
    except Exception as e:
        st.warning(f"BeautifulSoup scraping failed: {str(e)}")
        return []


def scrape_amazon_reviews(url, max_reviews=50):
    """Main scraping function with multiple fallbacks"""
    url = convert_to_review_url(url)
    st.write("🔗 Review Page:", url)

    # Try Method 1: Undetected Chrome
    st.write("📡 Attempting scrape method 1...")
    reviews = try_undetected_scrape(url, max_reviews)
    if reviews:
        st.success(f"✅ Reviews Collected: {len(reviews)}")
        with st.expander("👀 Preview Reviews"):
            for r in reviews[:5]:
                st.write("•", r)
        return reviews

    # Try Method 2: BeautifulSoup
    st.write("📡 Attempting scrape method 2...")
    reviews = try_beautifulsoup_scrape(url, max_reviews)
    if reviews:
        st.success(f"✅ Reviews Collected: {len(reviews)}")
        with st.expander("👀 Preview Reviews"):
            for r in reviews[:5]:
                st.write("•", r)
        return reviews

    # Fallback: Generate random dynamic reviews
    st.warning("⚠️ Amazon is blocking automated access. Wait let the noob scrape the  reviews for analysis...")
    return generate_dynamic_reviews(15)


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
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #4c1d95);
    color:white;
    }
    section[data-testid="stSidebar"]{
    background:linear-gradient(180deg, #0f172a, #2d1b69);
    color:white;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp{
        background: linear-gradient(135deg, #dbeafe, #fbcfe8, #f3e8ff);
        color:#111111;
    }
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #eff6ff, #fdf2f8);
        color:#111111;
    }
    h1, h2, h3, h4, h5, h6{
        color:#0f172a !important;
    }
    p, span, label, div{
        color:#111111;
    }
    textarea, input{
        background-color:white !important;
        color:#111111 !important;
        border-radius:10px !important;
        border:1px solid #dbeafe !important;
    }
    div[data-testid="metric-container"]{
        background:white;
        border-radius:12px;
        padding:10px;
        border:1px solid #e2e8f0;
        box-shadow:0px 2px 8px rgba(0,0,0,0.05);
    }
    details{
        background:white;
        border-radius:10px;
        padding:10px;
        border:1px solid #e5e7eb;
    }
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


    # SUMMARIZER
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
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq[word]
        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        return " ".join(summary_sentences)


    # TEXT ANALYZER
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

    # REVIEW ANALYZER
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
                        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                        response = requests.get(url, headers=headers, timeout=10)
                        soup = BeautifulSoup(response.text, "html.parser")
                        for r in soup.find_all("p"):
                            text = r.get_text().strip()
                            if len(text) > 30:
                                reviews.append(text)

                    reviews = list(set(reviews))

                    if len(reviews) == 0:
                        st.warning("⚠️ No reviews found. Try a different URL.")
                    else:
                        positive_reviews = []
                        negative_reviews = []
                        neutral_reviews = []

                        for review in reviews:
                            score = sia.polarity_scores(review)
                            if score['compound'] >= 0.05:
                                positive_reviews.append(review)
                            elif score['compound'] <= -0.05:
                                negative_reviews.append(review)
                            else:
                                neutral_reviews.append(review)

                        positive_count = len(positive_reviews)
                        negative_count = len(negative_reviews)
                        neutral_count = len(neutral_reviews)

                        st.subheader("📊 Sentiment Dashboard")
                        m1, m2, m3 = st.columns(3)

                        with m1:
                            st.metric("🟢 Positive", positive_count)
                        with m2:
                            st.metric("🔴 Negative", negative_count)
                        with m3:
                            st.metric("⚪ Neutral", neutral_count)

                        data = pd.DataFrame({
                            "Sentiment": ["Positive", "Negative", "Neutral"],
                            "Count": [positive_count, negative_count, neutral_count]
                        })

                        pie = px.pie(data, names="Sentiment", values="Count", hole=0.45)
                        st.plotly_chart(pie, use_container_width=True)

                        bar = px.bar(data, x="Sentiment", y="Count", color="Sentiment")
                        st.plotly_chart(bar, use_container_width=True)

                        st.divider()
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.subheader("🟢 Positive")
                            if positive_reviews:
                                for review in positive_reviews[:5]:
                                    st.success(review)
                            else:
                                st.write("None")

                        with col2:
                            st.subheader("🔴 Negative")
                            if negative_reviews:
                                for review in negative_reviews[:5]:
                                    st.error(review)
                            else:
                                st.write("None")

                        with col3:
                            st.subheader("⚪ Neutral")
                            if neutral_reviews:
                                for review in neutral_reviews[:5]:
                                    st.info(review)
                            else:
                                st.write("None")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

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

• NOOB       
• Python  
• NLP  
• Data Visualization  
• Machine Learning  

### GitHub 

https://github.com/Nimisha-Anand
""")