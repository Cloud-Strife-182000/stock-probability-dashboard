# 📈 Stock Probability Dashboard

A powerful, interactive Streamlit application that fetches live Indian stock market data and applies a Machine Learning prediction layer to forecast immediate asset momentum.

## Core Features

- **Live Market Data:** Instantly pulls 6-month historical tracking data for any NSE or BSE ticker symbol via `yfinance`.
- **Advanced Technical Indicators:** Automatically evaluates standard technical analytics overlaid with a beautiful UI array:
  - **SMA 20 & SMA 50:** Short and medium-term Simple Moving Averages.
  - **RSI (14):** Relative Strength Index (overbought/oversold momentum).
  - **MACD Histogram:** Moving Average Convergence Divergence.
- **Medium-Term Swing Signal:** Generates an algorithmic, aggregate `BUY / SELL / HOLD` consensus signal by calculating the active mathematical convergence across all four technical indicators.
- **Machine Learning Price Forecast:** Utilizes a strict `scikit-learn` Random Forest Classifier to mathematically calculate the probability of a >0.5% upward momentum swing for the subsequent trading day.
  - Implements strict anti-overfitting constraints (`max_depth=5`, `min_samples_leaf=10`).
  - Features real **Out-of-Sample Evaluation** ensuring the displayed backend accuracy percentage reflects a true forward-test against the most recent 20% timeline of untouched market data.
  - Fully transparent raw historical dataframe export included in the UI metrics expander.

## Installation & Setup

1. Install the required data science engines explicitly bound to their safe, locked versions:

   ```bash
   pip install -r requirements.txt
   ```

2. Boot the local dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

## Built With

- **[Streamlit](https://streamlit.io/):** UI framework engine.
- **[yfinance](https://pypi.org/project/yfinance/):** Market API connection layer.
- **[scikit-learn](https://scikit-learn.org/):** The core Machine Learning classification matrix.
- **[pandas](https://pandas.pydata.org/):** Structural dataset transformation.
