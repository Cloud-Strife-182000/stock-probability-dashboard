# 📈 Stock Probability Dashboard

A powerful, interactive Streamlit application configured to dynamically fetch live Indian stock market data and apply a predictive Random Forest machine learning model to track intraday asset momentum. This repository also includes an exhaustive brute-force feature selection script to optimize prediction accuracy.

## Core Architecture & Features

- **Sustained 730-Day Hourly Engine:** The system extracts up to 2 years of 1-hour interval data directly from `yfinance`, explicitly targeting the 9:15 AM to 10:15 AM morning session to capture structural opening hour momentum bounds.
- **Dynamic Feature Selection UI:** Users can interactively toggle 13 distinct technical and macro features (such as Order Flow Imbalance, VWAP Distance, Fractional Differencing, NIFTY 50 macro trends, and S&P 500 overnight sentiment) to instantly re-train the Random Forest model directly within the Streamlit interface.
- **Macroscopic Indicator Merging:** Integrates broad market indices (`^NSEI` for local sentiment and `^GSPC` for overnight US sentiment) to inform the predictive matrix with external risk-on/risk-off regimes.
- **Random Forest Ensemble:** Utilizes a robust `RandomForestClassifier` (`n_estimators=100`, class balancing, and leaf regulations) to predict binary outcomes for targeted intraday thresholds without overfitting simple sequences.
- **5-Day Walk Forward Validation Engine:** A strictly governed historical validation block that automatically runs 5 out-of-sample chronological prediction checks against previous session geometries, confirming the model's actual predictive synchronization locally.
- **Batch Watchlist & Excel Export:** Allows users to upload a `.txt` batch of tickers to automatically process predictions sequentially. The generated watchlist state can then be exported cleanly as an `.xlsx` file incorporating historical performance baselines and actual accuracy edges safely.

---

## 🚀 The Brute-Force Feature Selector

To discover the absolute strongest combination of features for any specific asset, the repository includes a standalone Python script: `brute_force_selection.py`. 

Because different stocks respond differently to distinct indicators, this script systematically trains and tests all **8191 possible feature combinations** ($2^{13} - 1$) using exact replica Random Forest settings. It ensures identical dataset shapes by dropping `NaN`s uniformly up front across all potential indicators, yielding a perfectly fair, chronological `train_test_split` cross-validation comparison.

**Running the Feature Selector:**

```bash
# Defaults to mapping the NSE exchange
python brute_force_selection.py --ticker PNB

# Specifying an alternate exchange
python brute_force_selection.py --ticker RELIANCE --exchange BSE
```

**Outputs:**
The script utilizes CPU-bound multiprocessing (`ProcessPoolExecutor`) to sprint rapidly through the matrix. Upon completion, it automatically creates a directory named `optimal_features/` and outputs a dedicated JSON configuration (e.g., `optimal_features/PNB_optimal_features.json`) detailing the optimal list of indicators generating the highest true test-set accuracy globally.

---

## Installation & Setup

1. Install the required data science dependencies from the root directory:

   ```bash
   pip install -r requirements.txt
   ```

2. Boot the local Streamlit dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

## Built With

- **[Streamlit](https://streamlit.io/):** Interactive execution UI and visual component mapping.
- **[yfinance](https://pypi.org/project/yfinance/):** Fundamental ticker data generation parsing bounds organically.
- **[scikit-learn](https://scikit-learn.org/):** Core Random Forest execution layer handling validation matrix testing properly.
- **[pandas](https://pandas.pydata.org/):** Advanced positional arrays tracking historical index targets natively.
- **[pandas-ta-classic](https://github.com/twopirllc/pandas-ta):** Technical analysis computations accurately calculating structural geometries.
