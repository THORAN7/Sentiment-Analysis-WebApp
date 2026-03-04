import tkinter as tk
from tkinter import messagebox
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER (only first time)
nltk.download('vader_lexicon')

# Create sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Function to analyze sentiment
def analyze_sentiment():
    text = text_entry.get()

    if text == "":
        messagebox.showwarning("Warning", "Please enter some text!")
        return

    score = sia.polarity_scores(text)

    if score['compound'] > 0:
        result = "Positive 😊"
    elif score['compound'] < 0:
        result = "Negative 😡"
    else:
        result = "Neutral 😐"

    result_label.config(
        text=f"Sentiment: {result}\n\nScore: {score['compound']}"
    )


# Create main window
window = tk.Tk()
window.title("Sentiment Analysis App")
window.geometry("400x300")

# Title label
title_label = tk.Label(window, text="Sentiment Analyzer", font=("Arial", 16))
title_label.pack(pady=10)

# Text entry box
text_entry = tk.Entry(window, width=40)
text_entry.pack(pady=10)

# Analyze button
analyze_button = tk.Button(window, text="Analyze", command=analyze_sentiment)
analyze_button.pack(pady=10)

# Result label
result_label = tk.Label(window, text="", font=("Arial", 12))
result_label.pack(pady=20)

# Run app
window.mainloop()