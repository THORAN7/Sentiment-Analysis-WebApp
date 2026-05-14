import streamlit as st
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import heapq
import re
import plotly.express as px
import pandas as pd
import time
import random
from urllib.parse import urlparse

# Selenium (only used when undetected_chromedriver is available)
from selenium.webdriver.common.by import By

# Try to import undetected chrome driver
try:
    import undetected_chromedriver as uc

    HAS_UNDETECTED = True
except Exception:
    # undetected_chromedriver is optional — fall back to other methods
    HAS_UNDETECTED = False


# ---------------- NLTK / RESOURCES SETUP ----------------

# We'll load nltk resources and the sentiment analyzer once and cache them
@st.cache_resource
def load_nltk_components():
    """Download required NLTK data (if missing) and return resources.

    Using Streamlit's cache_resource avoids re-downloading on every rerun.
    """
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception:
        # If downloads fail (offline), continue — some features may be limited
        pass

    sia = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words("english"))
    return sia, stop_words


# Load resources
sia, STOP_WORDS = load_nltk_components()

# Sentiment thresholds as constants
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+\b", text)


def safe_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except LookupError:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [part for part in parts if part]


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Review Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============= DYNAMIC REVIEW GENERATOR =============

@st.cache_data
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

        driver = None
        reviews = []
        try:
            driver = uc.Chrome(options=options, version_main=None)
            driver.get(url)
            time.sleep(random.uniform(3, 5))

            blocks = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review"]')

            for block in blocks:
                try:
                    review = block.find_element(By.CSS_SELECTOR, '[data-hook="review-body"]').text
                    if len(review) > 20:
                        reviews.append(review)
                except Exception:
                    # skip elements we can't parse
                    continue

                if len(reviews) >= max_reviews:
                    break

            return reviews[:max_reviews]
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
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
            except Exception:
                # ignore parse errors for individual review blocks
                continue

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

    # Fallback: Generate random dynamic reviews (cached)
    st.warning("📡 Attempting scrape method 3...")
    return generate_dynamic_reviews(15)


def scrape_reddit_reviews(url, max_reviews=50):
    """Scrape comments from a Reddit post using the Reddit JSON endpoint."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        parsed = urlparse(url)
        post_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
        json_url = f"{post_url}.json?limit=500&sort=top"

        st.write("🔗 Reddit JSON Page:", json_url)

        response = requests.get(json_url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        reviews = []

        # Post title and body
        try:
            post_data = data[0]["data"]["children"][0]["data"]
            title = post_data.get("title", "").strip()
            selftext = post_data.get("selftext", "").strip()

            if title and len(title) > 20:
                reviews.append(title)
            if selftext and selftext not in ["[removed]", "[deleted]"] and len(selftext) > 20:
                reviews.append(selftext)
        except Exception:
            pass

        def extract_comments(children):
            for child in children:
                if len(reviews) >= max_reviews:
                    return

                kind = child.get("kind")
                child_data = child.get("data", {})

                if kind == "t1":
                    body = child_data.get("body", "").strip()
                    if body and body not in ["[removed]", "[deleted]"] and len(body) > 20:
                        if body not in reviews:
                            reviews.append(body)

                    replies = child_data.get("replies")
                    if isinstance(replies, dict):
                        reply_children = replies.get("data", {}).get("children", [])
                        extract_comments(reply_children)

                if len(reviews) >= max_reviews:
                    return

        comment_children = data[1]["data"].get("children", [])
        extract_comments(comment_children)

        return reviews[:max_reviews]

    except requests.exceptions.Timeout:
        st.error("⏱️ Reddit took too long to respond (timeout). Try again later.")
    except requests.exceptions.ConnectionError:
        st.error("🌐 Could not connect to Reddit. It may be blocking automated requests.")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Reddit returned an error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Reddit scraping failed: {str(e)}")
    return []


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
    * {
        transition: all 0.3s ease;
    }

    .stApp{
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #4c1d95 100%);
        color: white;
        overflow-x: hidden;
    }

    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #0f172a 0%, #2d1b69 100%) !important;
        backdrop-filter: blur(10px);
        color: white;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #60a5fa !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 10px rgba(96, 165, 250, 0.3);
    }

    p, span, div, label, .stMarkdown {
        color: #e0e7ff !important;
    }

    .stCaption, small {
        color: #a5b4fc !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #e0e7ff !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }

    .stTextInput > div > div > input:focus {
        border: 1px solid rgba(96, 165, 250, 0.5) !important;
        box-shadow: 0 8px 32px rgba(96, 165, 250, 0.2) !important;
        color: white !important;
    }

    textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #e0e7ff !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    }

    textarea:focus {
        border: 1px solid rgba(96, 165, 250, 0.5) !important;
        box-shadow: 0 8px 32px rgba(96, 165, 250, 0.2) !important;
        color: white !important;
    }

    div[data-testid="metric-container"]{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.05));
        border-radius: 16px !important;
        padding: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.6) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    .stAlert {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
    }

    details {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.05)) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stRadio"] > label {
        color: white !important;
    }

    [data-testid="stRadio"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
        padding: 8px !important;
    }

    .stDivider {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    [data-testid="stToggle"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(96, 165, 250, 0.3) !important;
        border-radius: 20px !important;
        padding: 4px !important;
    }

    [data-testid="stToggle"] button {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        border-radius: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    * {
        transition: all 0.3s ease;
    }

    .stApp{
        background: linear-gradient(180deg, #fafbff 0%, #f5f0ff 25%, #fff5f9 50%, #f0f9ff 75%, #f5f0ff 100%);
        color: #0f172a;
    }

    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #eff6ff 0%, #fdf2f8 100%) !important;
        backdrop-filter: blur(10px);
        color: #0f172a;
    }

    h1, h2, h3, h4, h5, h6{
        color: #1d4ed8 !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }

    p, span, label, div {
        color: #0f172a !important;
    }

    pre, code {
        background: rgba(240, 245, 255, 0.95) !important;
        color: #0f172a !important;
    }

    [data-testid="stJson"] {
        background: rgba(240, 245, 255, 0.95) !important;
        border: 2px solid rgba(59, 130, 246, 0.25) !important;
        border-radius: 12px !important;
        color: #3b82f6 !important;
        padding: 0.5rem !important;
    }

    [data-testid="stJson"] pre,
    [data-testid="stJson"] code {
        background: transparent !important;
        color: #3b82f6 !important;
    }

    /* Fix JSON key-value styling in white theme */
    [data-testid="stJson"] .object-key {
        color: #3b82f6 !important;
    }

    [data-testid="stJson"] .variable-value {
        color: #1e40af !important;
    }

    [data-testid="stJson"] .variable-row {
        background: transparent !important;
        border-left-color: rgba(59, 130, 246, 0.3) !important;
    }

    [data-testid="stExpander"] {
        background: rgba(59, 130, 246, 0.15) !important;
        border: 2px solid rgba(59, 130, 246, 0.5) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.08) !important;
    }

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] div {
        color: #0f172a !important;
    }

    /* Ensure details element is visible */
    details {
        background: rgba(59, 130, 246, 0.15) !important;
        border: 2px solid rgba(59, 130, 246, 0.5) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        color: #0f172a !important;
    }

    details summary {
        color: #1d4ed8 !important;
        font-weight: 600 !important;
        cursor: pointer !important;
    }

    details > * {
        color: #0f172a !important;
        background: transparent !important;
    }

    /* Fix nested divs inside details */
    details div,
    details span,
    details pre,
    details code {
        color: #0f172a !important;
        background: transparent !important;
    }

    /* Ensure all text is visible */
    details * {
        color: #0f172a !important;
    }

    [data-testid="stExpander"] button,
    [data-testid="stExpander"] button * {
        color: #0f172a !important;
        fill: #0f172a !important;
    }

    [data-testid="stExpander"] pre,
    [data-testid="stExpander"] code {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
    }

    [data-testid="stExpander"] [data-testid="stJson"] {
        background: rgba(255, 255, 255, 0.95) !important;
    }

    .stMarkdown > p {
        color: #0f172a !important;
    }

    .st-ek {
        background-color: rgba(59, 130, 246, 0.05) !important;
    }

    .stCaption {
        color: #1e40af !important;
        font-weight: 500 !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(59, 130, 246, 0.4) !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1) !important;
    }

    .stTextInput > div > div > input:focus {
        border: 2px solid rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3) !important;
        background: rgba(240, 245, 255, 0.95) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(29, 78, 216, 0.5) !important;
    }

    textarea {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(59, 130, 246, 0.4) !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1) !important;
    }

    textarea:focus {
        border: 2px solid rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3) !important;
        background: rgba(240, 245, 255, 0.95) !important;
    }

    textarea::placeholder {
        color: rgba(29, 78, 216, 0.5) !important;
    }

    div[data-testid="metric-container"]{
        background: linear-gradient(135deg, rgba(240, 245, 255, 0.8), rgba(255, 255, 255, 0.9));
        border-radius: 16px !important;
        padding: 20px !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.12) !important;
    }

    [data-testid="metric-container"] > label {
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    [data-testid="metric-container"] > div > div {
        color: #1e40af !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.5) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    .stAlert {
        background: linear-gradient(135deg, rgba(240, 245, 255, 0.85), rgba(255, 255, 255, 0.95)) !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1) !important;
        color: #0f172a !important;
    }

    details {
        background: linear-gradient(135deg, rgba(240, 245, 255, 0.8), rgba(255, 255, 255, 0.9)) !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1) !important;
        color: #0f172a !important;
    }

    details > summary {
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    [data-testid="stRadio"] > label {
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    [data-testid="stRadio"] > div {
        background: rgba(240, 245, 255, 0.8) !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
        padding: 8px !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.08) !important;
    }

    .stDivider {
        border-color: rgba(59, 130, 246, 0.4) !important;
    }

    [data-testid="stToggle"] {
        background: rgba(240, 245, 255, 0.9) !important;
        border: 2px solid rgba(59, 130, 246, 0.5) !important;
        border-radius: 20px !important;
        padding: 4px !important;
    }

    [data-testid="stToggle"] button {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        border-radius: 16px !important;
    }

    .stHeading {
        color: #1d4ed8 !important;
    }

    .stSubheader {
        color: #1e40af !important;
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
        words = safe_word_tokenize(text.lower())
        freq = defaultdict(int)
        for word in words:
            if word not in stop_words:
                freq[word] += 1
        sentences = safe_sent_tokenize(text)
        sentence_scores = {}
        for sentence in sentences:
            for word in safe_word_tokenize(sentence.lower()):
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
        score_cols = st.columns(3)
        score_labels = [
            ("Positive", score["pos"]),
            ("Neutral", score["neu"]),
            ("Negative", score["neg"]),
        ]

        for col, (label, value) in zip(score_cols, score_labels):
            with col:
                st.metric(label, f"{value:.3f}")

        with st.expander("Show raw sentiment details"):
            details = {
                "Positive": score["pos"],
                "Neutral": score["neu"],
                "Negative": score["neg"]
            }
            st.json(details)

        if score['compound'] >= POS_THRESHOLD:
            st.success("😊 Positive Sentiment")
        elif score['compound'] <= NEG_THRESHOLD:
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
            # basic normalization
            if not url.lower().startswith(('http://', 'https://')):
                url = 'https://' + url
            with st.spinner("🤖 AI analyzing reviews..."):
                try:
                    reviews = []

                    if "amazon" in url.lower():
                        reviews = scrape_amazon_reviews(url)
                    elif "reddit.com" in url.lower() or "redd.it" in url.lower():
                        reviews = scrape_reddit_reviews(url)
                    else:
                        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                        try:
                            response = requests.get(url, headers=headers, timeout=20)
                            response.raise_for_status()
                            soup = BeautifulSoup(response.text, "html.parser")
                            # Try multiple selectors to find reviews
                            review_selectors = [
                                soup.find_all("p"),
                                soup.find_all("div", class_=re.compile(r"review|comment|feedback|rating", re.I)),
                                soup.find_all("span", class_=re.compile(r"review|comment|text", re.I)),
                                soup.find_all("article"),
                                soup.find_all("li", class_=re.compile(r"review|comment", re.I)),
                                soup.find_all("blockquote"),
                                soup.find_all("section", class_=re.compile(r"review|comment", re.I)),
                            ]

                            for selector_results in review_selectors:
                                for element in selector_results:
                                    text = element.get_text().strip()
                                    # Filter for meaningful review text (30-500 chars)
                                    if 30 < len(text) < 500:
                                        # Avoid duplicates
                                        if text not in reviews:
                                            reviews.append(text)
                                # Stop searching if we found enough reviews
                                if len(reviews) > 10:
                                    break
                        except requests.exceptions.Timeout:
                            st.error("⏱️ Website took too long to respond (timeout). Try a different URL.")
                        except requests.exceptions.ConnectionError:
                            st.error("🌐 Connection error. The website may be blocking automated access.")
                        except requests.exceptions.HTTPError:
                            st.error("❌ Website returned an error. Check the URL and try again.")
                        except Exception as e:
                            st.error(f"❌ Error fetching website: {str(e)}")

                    reviews = list(set(reviews))

                    if len(reviews) == 0:
                        st.warning("⚠️ No reviews found. Try a different URL.")
                    else:
                        positive_reviews = []
                        negative_reviews = []
                        neutral_reviews = []

                        for review in reviews:
                            score = sia.polarity_scores(review)
                            if score['compound'] >= POS_THRESHOLD:
                                positive_reviews.append(review)
                            elif score['compound'] <= NEG_THRESHOLD:
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

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "👨‍💻 About":
    st.title("👨‍💻 About")
    st.markdown("""
This dashboard is a  project showcasing basic web scraping and NLP techniques (VADER sentiment, simple extractive summarization) combined with Streamlit for interactive visualization.

If you'd like to improve or extend the project, please open an issue or PR on the repository and include reproducible steps.

Key technologies:
- Python
- Streamlit
- NLTK (VADER)
- Requests & BeautifulSoup
- Selenium (optional)

Repository / Contact: https://github.com/yourusername/your-repo
""")