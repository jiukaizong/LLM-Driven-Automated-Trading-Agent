# LLM-Based Trading Agent

## Overview

This project implements an automated trading agent that combines financial news sentiment analysis with technical indicators to generate trading decisions.

The system integrates:

Financial news sentiment using FinBERT
Technical analysis using a 50-day moving average (MA50)
Risk management through stop-loss mechanisms

## How to set up API keys
To run this project, you need to configure your API keys using a .env file.

Step 1: Create a file named .env in the project root directory.

Step 2: Add the following content:

ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key

Optional parameters:

TICKER=AAPL
NEWS_LIMIT=200
LOOKBACK_HOURS=4320
STOP_LOSS_PCT=0.02
CONVICTION_THRESHOLD=7.0
INITIAL_CAPITAL=10000
EXECUTE_ORDERS=false

Step 3: Make sure your code loads the environment variables using:

load_dotenv()

## The logic behind the LLM sentiment module

This project uses FinBERT, which is a transformer model trained on financial text, to analyze news sentiment.

Instead of using LLMs based on prompt, this approach uses a classification model. It is more stable and easier to reproduce results.

The process works like this:

Input text
Each news headline is sent into the FinBERT model.
Model output
The model gives three probabilities:
Positive
Negative
Neutral
Pick sentiment
We choose the label with the highest probability as the final sentiment.
Conviction score
We treat the highest probability as confidence, and scale it to a score from 0 to 10:

conviction_score = max_probability × 10

This tells us how strong the sentiment is.

Convert to numbers
We map sentiment into numbers:
Positive = +1
Neutral = 0
Negative = -1
Daily aggregation
For each day, we combine all news into one value using a weighted average:

daily_sentiment = sum(sentiment × conviction) / sum(conviction)

Generate signal
We convert the daily sentiment into a signal:
Positive if value > 0
Negative if value < 0
Neutral otherwise
Use in trading
A trade only happens when:
Sentiment is positive
Conviction is high enough
Price is above MA50

## Results Table（conservative mode）
Metric: Sharpe Ratio
Strategy: -1.00
Benchmark: 0.63

Metric: Max Drawdown (%)
Strategy: -0.47
Benchmark: -22.99

Metric: Total Return (%)
Strategy: -1.11
Benchmark: 16.28

## Results Table（aggressive mode）
Metric: Sharpe Ratio
Strategy: 1.26
Benchmark: 0.57

Metric: Max Drawdown (%)
Strategy: -8.61
Benchmark: -22.99

Metric: Total Return (%)
Strategy: 20.98
Benchmark: 14.17

# Online google colab
conservative mode: https://colab.research.google.com/drive/17rSAIRQ4pt4cHzozyjoWV_u04ZGOXE9G?usp=sharing 
aggressive mode: https://colab.research.google.com/drive/1PMwfP3HtOeNG91M_aMEA9T_x3hPmEwx5?usp=sharing 