import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (only first time)
nltk.download('vader_lexicon')

# Create analyzer
sia = SentimentIntensityAnalyzer()

print("Sentiment Analysis Program Started!")
print("Type 'exit' to quit.\n")

while True:
    text = input("Enter a sentence: ")

    if text.lower() == "exit":
        print("Program closed.")
        break

    score = sia.polarity_scores(text)

    print("\nDetailed Scores:", score)

    if score['compound'] > 0:
        print("Overall Sentiment: Positive 😊\n")
    elif score['compound'] < 0:
        print("Overall Sentiment: Negative 😡\n")
    else:
        print("Overall Sentiment: Neutral 😐\n")