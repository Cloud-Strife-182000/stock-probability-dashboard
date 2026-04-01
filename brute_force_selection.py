import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import urllib.request
import xml.etree.ElementTree as ET
import datetime
import time
import itertools
import json
import sys
import os
import concurrent.futures

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone

FEATURE_MAP = {
    "Closing Momentum": 'Closing_Momentum',
    "Volume Surge": 'Closing_Volume_Surge',
    "Dist to SMA5": 'Distance_to_Fast_SMA',
    "ATR %": 'ATR_Percent',
    "Daily RSI": 'Daily_RSI_14',
    "VWAP Dist": 'VWAP_Distance',
    "OFI (Order Flow)": 'OFI',
    "Frac Diff (Memory)": 'Frac_Diff_Close',
    "Nifty Momentum": 'Nifty_Momentum',
    "Nifty RSI": 'Nifty_RSI_14',
    "Nifty Trend": 'Nifty_Trend_Dist',
    "Morning Autocorr": 'Morning_Autocorr',
    "US Overnight Ret": 'US_Overnight_Return'
}
ALL_FEATURES = list(FEATURE_MAP.values())

def fetch_stock_data(ticker, exch):
    suffix = ".NS" if exch == "NSE" else ".BS"
    sym = f"{ticker}{suffix}"
    
    stock = yf.Ticker(sym)
    df = stock.history(period="730d", interval="1h")
    
    if df.empty and exch == "BSE":
        sym_bo = f"{ticker}.BO"
        stock_bo = yf.Ticker(sym_bo)
        df = stock_bo.history(period="730d", interval="1h")
        if not df.empty:
            sym = sym_bo
            
    df_1d = stock.history(period="2y", interval="1d")
    return df_1d, df, sym

def fetch_nifty_data():
    try:
        nifty = yf.Ticker("^NSEI")
        df_nifty = nifty.history(period="2y", interval="1d")
        if df_nifty.empty:
            return pd.DataFrame()
        df_nifty = df_nifty.reset_index()
        df_nifty['DateStr'] = pd.to_datetime(df_nifty['Date']).dt.strftime('%Y-%m-%d')
        df_nifty['Nifty_Momentum'] = (df_nifty['Close'] - df_nifty['Open']) / df_nifty['Open']
        df_nifty['Nifty_RSI_14'] = df_nifty.ta.rsi(length=14)
        ema20 = df_nifty['Close'].ewm(span=20, adjust=False).mean()
        df_nifty['Nifty_Trend_Dist'] = (df_nifty['Close'] - ema20) / ema20
        return df_nifty[['DateStr', 'Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist']]
    except Exception:
        return pd.DataFrame()

def fetch_global_sentiment_data():
    try:
        sp500 = yf.Ticker("^GSPC")
        df_sp = sp500.history(period="2y", interval="1d")
        if df_sp.empty:
            return pd.DataFrame()
        df_sp = df_sp.reset_index()
        df_sp['DateStr'] = pd.to_datetime(df_sp['Date']).dt.strftime('%Y-%m-%d')
        df_sp['US_Overnight_Return'] = df_sp['Close'].pct_change().shift(1)
        return df_sp[['DateStr', 'US_Overnight_Return']]
    except Exception:
        return pd.DataFrame()

def frac_diff_ffd(series, d=0.4, thresh=1e-5):
    w = [1.0]
    k = 1
    while abs(w[-1]) > thresh:
        w.append(-w[-1] * (d - k + 1) / k)
        k += 1
    w = np.array(w[::-1])
    width = len(w)
    output = []
    vals = series.values
    for i in range(width - 1, len(vals)):
        output.append(np.dot(w, vals[i - width + 1:i + 1]))
    result = pd.Series([np.nan] * (width - 1) + output, index=series.index)
    return result

def prepare_data(ticker, exchange):
    print(f"Fetching data for {ticker} ({exchange})...")
    data_1d, data_1h, symbol = fetch_stock_data(ticker, exchange)
    
    if data_1d.empty or data_1h.empty:
        raise ValueError(f"No data found for {ticker}")
        
    df_1d = data_1d.copy()
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = [col[0] if isinstance(col, tuple) else col for col in df_1d.columns]
    df_1d = df_1d.reset_index()
    
    if 'Date' in df_1d.columns:
        df_1d['DateStr'] = pd.to_datetime(df_1d['Date']).dt.strftime('%Y-%m-%d')
    elif 'Datetime' in df_1d.columns:
        df_1d['DateStr'] = pd.to_datetime(df_1d['Datetime']).dt.strftime('%Y-%m-%d')
        
    if 'Close' in df_1d.columns and len(df_1d) >= 14:
        df_1d['Daily_SMA_5'] = df_1d['Close'].rolling(window=5).mean()
        df_1d['Daily_ATR_14'] = df_1d.ta.atr(length=14)
        df_1d['Daily_RSI_14'] = df_1d.ta.rsi(length=14)
        
    nifty_df = fetch_nifty_data()
    if not nifty_df.empty:
        df_1d = pd.merge(df_1d, nifty_df, on='DateStr', how='left')
        for col in ['Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist']:
            if col in df_1d.columns:
                df_1d[col] = df_1d[col].ffill()
                
    sp_df = fetch_global_sentiment_data()
    if not sp_df.empty:
        df_1d = pd.merge(df_1d, sp_df, on='DateStr', how='left')
        if 'US_Overnight_Return' in df_1d.columns:
            df_1d['US_Overnight_Return'] = df_1d['US_Overnight_Return'].ffill()
        
    df = data_1h.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    
    dt_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df['DatetimeObj'] = pd.to_datetime(df[dt_col])
    
    if df['DatetimeObj'].dt.tz is not None:
        df['DatetimeObj'] = df['DatetimeObj'].dt.tz_convert('Asia/Kolkata')
    else:
        df['DatetimeObj'] = df['DatetimeObj'].dt.tz_localize('Asia/Kolkata')
        
    df['DateStr'] = df['DatetimeObj'].dt.strftime('%Y-%m-%d')
    df['TimeStr'] = df['DatetimeObj'].dt.strftime('%H:%M')
    
    df['Closing_Momentum'] = (df['Close'] - df['Open']) / df['Open']
    df['Closing_Volume_Surge'] = df['Volume'] / df['Volume'].rolling(window=35, min_periods=5).mean()
    
    daily_cols = ['DateStr', 'Daily_SMA_5', 'Daily_ATR_14', 'Daily_RSI_14', 'Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist', 'US_Overnight_Return']
    merge_cols = [c for c in daily_cols if c in df_1d.columns]
    daily_subset = df_1d[merge_cols].dropna()
    df = pd.merge(df, daily_subset, on='DateStr', how='left')
    
    for col in merge_cols:
        if col != 'DateStr':
            df[col] = df[col].ffill()
    
    df['Distance_to_Fast_SMA'] = (df['Close'] - df['Daily_SMA_5']) / df['Daily_SMA_5']
    df['ATR_Percent'] = df['Daily_ATR_14'] / df['Close']
    
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cum_Vol'] = df.groupby('DateStr')['Volume'].cumsum()
    df['Cum_TP_Vol'] = df.groupby('DateStr')['TP_Volume'].cumsum()
    df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
    df['VWAP_Distance'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    df['Frac_Diff_Close'] = frac_diff_ffd(df['Close'], d=0.4)
    
    df = df.sort_values(['DateStr', 'DatetimeObj'])
    day_opens = df.groupby('DateStr')['Open'].transform('first')
    p1015 = df[df['TimeStr'] == '10:15'].set_index('DateStr')['Close']
    df['Morning_Autocorr'] = (df['DateStr'].map(p1015) - day_opens) / day_opens
    
    hl_range = df['High'] - df['Low']
    hl_range = hl_range.replace(0, np.nan)
    df['OFI'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range).rolling(window=5, min_periods=1).mean()
    
    daily_targets = {}
    for date_str, group in df.groupby('DateStr'):
        group = group.sort_values(by='DatetimeObj')
        candle_915 = group[group['TimeStr'] == '09:15']
        candle_1015 = group[group['TimeStr'] == '10:15']
        
        if candle_915.empty or candle_1015.empty:
            continue
            
        open_price = candle_915.iloc[0]['Open']
        close_price = candle_1015.iloc[0]['Close']
        
        if close_price > open_price:
            daily_targets[date_str] = 1.0
        else:
            daily_targets[date_str] = -1.0
            
    ml_df = df.groupby('DateStr').tail(1).copy()
    
    date_to_next_date = {}
    all_trading_dates = sorted(list(df['DateStr'].unique()))
    for i in range(len(all_trading_dates) - 1):
        date_to_next_date[all_trading_dates[i]] = all_trading_dates[i+1]
        
    def map_target(row):
        next_date = date_to_next_date.get(row['DateStr'])
        if next_date and next_date in daily_targets:
            return daily_targets[next_date]
        return float('nan')
        
    ml_df['Target'] = ml_df.apply(map_target, axis=1)
    
    # 1. Clear out active/incomplete sessions where the target doesn't exist yet
    ml_df = ml_df.dropna(subset=['Target'])
    
    # Features remain dynamically evaluated inside evaluate_combination.
    return ml_df

def evaluate_combination(args):
    feature_combo, ml_df = args
    combo_list = list(feature_combo)
    
    # Drop NAs dynamically just for this specific feature combination
    ml_df_subset = ml_df.dropna(subset=combo_list + ['Target'])
    
    if len(ml_df_subset) < 10:
        return feature_combo, 0.0, 0, 0
        
    X = ml_df_subset[combo_list].astype(float)
    y_series = ml_df_subset['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y_series, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=15, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    
    return feature_combo, acc, len(ml_df_subset), len(y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brute Force Feature Selection for Stock Prediction")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g. PNB, BANKINDIA)")
    parser.add_argument("--exchange", default="NSE", choices=["NSE", "BSE"], help="Exchange (NSE or BSE)")
    
    args = parser.parse_args()
    
    try:
        ml_df = prepare_data(args.ticker, args.exchange)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    if len(ml_df) < 10:
        print("Not enough raw data to train the model.")
        sys.exit(1)
        
    last_valid_date = ml_df.iloc[-1]['DateStr']
    print(f"Dataset prepared. Total raw valid rows available: {len(ml_df)} | Last Valid Training Date: {last_valid_date}")
    
    # Generate all combinations
    all_combinations = []
    for r in range(1, len(ALL_FEATURES) + 1):
        all_combinations.extend(list(itertools.combinations(ALL_FEATURES, r)))
        
    total_combinations = len(all_combinations)
    print(f"Total combinations to test: {total_combinations}")
    
    all_results = []
    best_acc = 0.0
    
    completed = 0
    start_time = time.time()
    
    # Use multiprocessing to speed up Random Forest training grid
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_combination, (combo, ml_df)): combo for combo in all_combinations}
        
        for future in concurrent.futures.as_completed(futures):
            combo, acc, c_rows, c_test_size = future.result()
            
            # Store all combinations to cleanly extract a Top-3 leaderboard later
            all_results.append((acc, len(combo), tuple(sorted(list(combo))), combo, c_rows, c_test_size))
            best_acc = max(best_acc, acc)
                
            completed += 1
            if completed % 500 == 0 or completed == total_combinations:
                elapsed = time.time() - start_time
                print(f"Processed {completed}/{total_combinations} | Best Acc: {best_acc:.4f} | Time: {elapsed:.1f}s")

    # Tie-breaker deterministic sorting:
    # 1. Highest Accuracy (Desc: -acc)
    # 2. Fewest Features (Asc: length)
    # 3. Alphabetical determinism (Asc: string sorting)
    all_results.sort(key=lambda x: (-x[0], x[1], x[2]))
    top_3 = all_results[:3]

    print("-" * 40)
    print("Top 3 Feature Combinations Analysis:")
    for i, res in enumerate(top_3, 1):
        print(f"Rank {i} | Accuracy: {res[0]:.4f}")
        print(f"Features: {list(res[3])}\n")
    
    output_data = {
        "ticker": args.ticker,
        "exchange": args.exchange,
        "top_combinations": [
            {
                "rank": i,
                "accuracy": round(res[0], 4),
                "features": list(res[3]),
                "dataset_rows": res[4],
                "out_of_sample_size": res[5]
            } for i, res in enumerate(top_3, 1)
        ]
    }
    
    os.makedirs("optimal_features", exist_ok=True)
    output_filename = os.path.join("optimal_features", f"{args.ticker}_optimal_features.json")
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Results saved to {output_filename}")
