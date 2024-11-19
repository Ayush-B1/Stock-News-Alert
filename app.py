import streamlit as st
from data_utils import fetch_stock_data, calculate_technical_indicators
from ml_models import train_price_prediction_model
from sentiment_analysis import analyze_sentiment

st.title("Stock Trend Analyzer")

# Input for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL")

if st.button("Analyze"):
    # Step 1: Fetch stock data
    st.subheader("Fetching Stock Data...")
    try:
        stock_data = fetch_stock_data(stock_symbol)
        if stock_data.empty:
            st.error("Failed to fetch stock data. Check the stock symbol.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        st.stop()

    st.success("Data fetched successfully!")
    st.write("Stock Data Sample", stock_data.head())

    # Step 2: Display stock data
    st.subheader("Stock Price Chart")
    st.line_chart(stock_data["Close"])

    # Step 3: Add and display technical indicators
    st.subheader("Technical Indicators")
    try:
        stock_data = calculate_technical_indicators(stock_data)
        st.write("Technical Indicators Sample", stock_data.head())

        if "Moving_Avg" in stock_data.columns and "RSI" in stock_data.columns:
            st.line_chart(stock_data[["Close", "Moving_Avg", "RSI"]])
        else:
            st.error("Failed to calculate technical indicators.")
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")

    # Step 4: Train and visualize predictions
    st.subheader("Stock Price Prediction")
    try:
        model, predictions = train_price_prediction_model(stock_data)
        st.line_chart(predictions)
    except Exception as e:
        st.error(f"Error in prediction model: {e}")

    # Step 5: Perform sentiment analysis
    st.subheader("Sentiment Analysis")
    try:
        sentiment_score = analyze_sentiment(stock_symbol)
        st.write(f"Sentiment Score: {sentiment_score}")
    except ImportError as e:
        st.error("Error in sentiment analysis:")
        st.warning(
            "It seems there's an issue with your libraries. Please ensure you have TensorFlow 2.x or PyTorch installed."
        )
        st.code(
            "pip install tensorflow==2.11.0", language="bash"
        )
    except Exception as e:
        st.error(f"Unexpected error in sentiment analysis: {e}")
