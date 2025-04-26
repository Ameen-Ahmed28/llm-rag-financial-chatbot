import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob
from groq import Groq
import requests
from googlesearch import search
import finnhub

# Load API Keys
try:
    with open("API_KEY", "r") as file:
        GROQ_API_KEY = file.read().strip()
    with open("FINNHUB_KEY", "r") as file:
        FINNHUB_API_KEY = file.read().strip()
except FileNotFoundError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    FINNHUB_API_KEY = os.getenv("FINNHUB_KEY")
    if not GROQ_API_KEY or not FINNHUB_API_KEY:
        st.error("API Keys missing in file or environment.")
        st.stop()

# Initialize Clients
client = Groq(api_key=GROQ_API_KEY)
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def fetch_google_news(ticker):
    query = f"{ticker} stock news site:finance.yahoo.com OR site:bloomberg.com OR site:cnbc.com OR site:reuters.com"
    try:
        news_links = list(search(query, num_results=20))
        return [{"title": link.split("/")[-1], "link": link} for link in news_links] if news_links else []
    except Exception:
        return []

def analyze_sentiment(news_list):
    sentiment_scores = [TextBlob(article["title"]).sentiment.polarity for article in news_list]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    sentiment_label = "Neutral" if -0.1 <= avg_sentiment <= 0.1 else ("Positive (Bullish)" if avg_sentiment > 0.1 else "Negative (Bearish)")
    return avg_sentiment, sentiment_label

def generate_investment_advice(news_list, user_prompt, ticker, sentiment_score, sentiment_label):
    if not news_list:
        return f"No recent news found for {ticker}. Please analyze technical indicators instead."
    
    news_content = "\n\n".join(f"Title: {article['title']}\nLink: {article['link']}" for article in news_list)
    full_prompt = f"""
You are an advanced financial AI assistant specializing in stock analysis.
Your task is to analyze the following news articles related to {ticker} and provide an investment recommendation.

News Articles:
{news_content}

Sentiment Score: {sentiment_score} ({sentiment_label})

User Question: {user_prompt}

Please consider:
- Recent stock news and sentiment.
- Financial indicators like earnings, market trends, and risks.
- Whether the stock is a **BUY, SELL, or HOLD** based on the news impact.

Provide a **direct and clear** investment recommendation.
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial assistant providing stock analysis."},
                {"role": "user", "content": full_prompt},
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

@st.cache_data(show_spinner=False)
def fetch_latest_quote(ticker):
    try:
        st.write("📡 Fetching stock data from Finnhub...")
        quote = finnhub_client.quote(ticker)
        if not quote or quote['c'] == 0:
            return f"No data found for {ticker}."
        return pd.DataFrame({
            "Metric": ["Current Price", "Open", "High", "Low", "Previous Close"],
            "Value": [quote["c"], quote["o"], quote["h"], quote["l"], quote["pc"]]
        })
    except Exception as e:
        return f"Finnhub quote error: {e}"

def plot_latest_price(quote_df, ticker):
    try:
        price = quote_df.loc[quote_df['Metric'] == 'Current Price', 'Value'].values[0]
        fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = price,
            delta = {"reference": quote_df.loc[quote_df['Metric'] == 'Previous Close', 'Value'].values[0]},
            title = {"text": f"{ticker.upper()} Current Price"}
        ))
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting stock price: {e}")

# Streamlit App
st.title("📈 LLM Financial Advisor")

ticker_input = st.text_input("🧾 Enter Stock Ticker (e.g. AAPL):")
user_prompt = st.text_area("🧠 Your Question (e.g. Should I invest now?):")

if st.button("🔍 Analyze and Respond"):
    if not ticker_input:
        st.warning("Please enter a stock ticker symbol.")
    else:
        st.write("📰 Fetching news...")
        news_data = fetch_google_news(ticker_input)
        sentiment_score, sentiment_label = analyze_sentiment(news_data)
        response = generate_investment_advice(news_data, user_prompt, ticker_input, sentiment_score, sentiment_label)
        st.subheader("💡 Investment Advice:")
        st.write(response)

if st.button("📊 Get Latest Quote"):
    if not ticker_input:
        st.warning("Please enter a stock ticker symbol.")
    else:
        quote_data = fetch_latest_quote(ticker_input)
        if isinstance(quote_data, pd.DataFrame):
            st.dataframe(quote_data)
        else:
            st.error(quote_data)

if st.button("📈 Plot Current Price"):
    if not ticker_input:
        st.warning("Please enter a stock ticker symbol.")
    else:
        quote_data = fetch_latest_quote(ticker_input)
        if isinstance(quote_data, pd.DataFrame):
            plot_latest_price(quote_data, ticker_input)
        else:
            st.error(quote_data)
