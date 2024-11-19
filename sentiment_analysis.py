from transformers import pipeline

def analyze_sentiment(stock_symbol):
    # Use HuggingFace's sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
    sample_news = [
        f"{stock_symbol} sees strong growth prospects",
        f"{stock_symbol} faces challenges in the market",
        f"Mixed opinions about {stock_symbol}'s future"
    ]

    scores = [sentiment_analyzer(text)[0]["score"] for text in sample_news]
    average_score = sum(scores) / len(scores)

    return average_score
