import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta_classic as ta
import urllib.request
import xml.etree.ElementTree as ET
import io
import datetime
import time
import json
from contextlib import nullcontext

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone

st.set_page_config(page_title="Stock Probability Dashboard", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, exch):
    suffix = ".NS" if exch == "NSE" else ".BS"
    sym = f"{ticker}{suffix}"
    
    # Fetch intraday data using yfinance
    stock = yf.Ticker(sym)
    df = stock.history(period="730d", interval="1h") # Changed to 730d, 1h
    
    # If 1h data is empty for BSE, try .BO suffix
    if df.empty and exch == "BSE":
        sym_bo = f"{ticker}.BO"
        stock_bo = yf.Ticker(sym_bo)
        df = stock_bo.history(period="730d", interval="1h")
        if not df.empty:
            sym = sym_bo
            
    # Fetch daily data explicitly to calculate moving averages smoothly
    df_1d = stock.history(period="2y", interval="1d") # Changed to 2y
    
    return df_1d, df, sym

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty_data():
    """Fetch NIFTY 50 (^NSEI) daily data for macro context."""
    try:
        nifty = yf.Ticker("^NSEI")
        df_nifty = nifty.history(period="2y", interval="1d")
        if df_nifty.empty:
            return pd.DataFrame()
        df_nifty = df_nifty.reset_index()
        df_nifty['DateStr'] = pd.to_datetime(df_nifty['Date']).dt.strftime('%Y-%m-%d')
        # Macro Indicators
        df_nifty['Nifty_Momentum'] = (df_nifty['Close'] - df_nifty['Open']) / df_nifty['Open']
        df_nifty['Nifty_RSI_14'] = df_nifty.ta.rsi(length=14)
        ema20 = df_nifty['Close'].ewm(span=20, adjust=False).mean()
        df_nifty['Nifty_Trend_Dist'] = (df_nifty['Close'] - ema20) / ema20
        return df_nifty[['DateStr', 'Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist']]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_global_sentiment_data():
    """Fetch S&P 500 (^GSPC) daily data for global overnight sentiment."""
    try:
        sp500 = yf.Ticker("^GSPC")
        df_sp = sp500.history(period="2y", interval="1d")
        if df_sp.empty:
            return pd.DataFrame()
            
        df_sp = df_sp.reset_index()
        df_sp['DateStr'] = pd.to_datetime(df_sp['Date']).dt.strftime('%Y-%m-%d')
        
        # Calculate US daily return and shift strictly forward by 1 day
        # so Monday's US performance aligns with Tuesday's India row.
        df_sp['US_Overnight_Return'] = df_sp['Close'].pct_change().shift(1)
        
        return df_sp[['DateStr', 'US_Overnight_Return']]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_top_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}+share+news&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            news_items = root.findall('.//item')[:3]
            results = []
            for item in news_items:
                title = item.find('title').text
                link = item.find('link').text
                pubDate = item.find('pubDate').text
                results.append({"title": title, "link": link, "date": pubDate})
            return results
    except Exception as e:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_munafasutra_prediction(ticker):
    """Fetch predictive text from MunafaSutra."""
    try:
        import requests
        from bs4 import BeautifulSoup
        base_ticker = ticker.replace('.NS', '').replace('.BO', '').replace('.BS', '')
        url = f"https://munafasutra.com/nse/tomorrow/{base_ticker}"
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        resp = requests.get(url, headers=headers, timeout=5)
        
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            paragraphs = soup.find_all('p')
            prediction_text = ""
            
            for p in paragraphs:
                txt = p.get_text(strip=True)
                # Filter out garbage tags or short UI strings, looking for actual analysis paragraphs
                if len(txt) > 80 and "AdBlock" not in txt and "MunafaSutra" not in txt:
                    prediction_text += txt + "\n\n"
                    if len(prediction_text) > 1000:
                        prediction_text = prediction_text[:1000] + "..."
                        break
            
            if prediction_text:
                return prediction_text.strip()
            return "No readable prediction paragraphs available on MunafaSutra."
        else:
            return f"Failed to fetch data (Status Code: {resp.status_code})"
    except Exception as e:
        return f"Error fetching MunafaSutra prediction: {e}"

def frac_diff_ffd(series, d=0.4, thresh=1e-5):
    """Fixed-Width Window Fractional Differencing (Lopez de Prado)."""
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



st.title("Stock Probability Dashboard")

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter Indian Ticker (e.g., PNB, BANKINDIA):", "").strip().upper()
with col2:
    exchange = st.selectbox("Select Exchange:", ["NSE", "BSE"])

# --- NEW: DYNAMIC FEATURE SELECTION UI ---
FEATURE_MAP = {
    "Closing Momentum": 'Closing_Momentum',
    "OBV Slope": 'OBV_Slope',
    "Anchored CR5": 'Anchored_Close_Return_5',
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

# Initial setup for persistence of engine features
if 'confirmed_features' not in st.session_state:
    st.session_state['confirmed_features'] = list(FEATURE_MAP.values())

for col_name in FEATURE_MAP.values():
    key = f"feat_{col_name}"
    if key not in st.session_state:
        st.session_state[key] = True

# Detect ticker change to sync engine to current UI selection
if ticker_input:
    if 'last_ticker' not in st.session_state or st.session_state['last_ticker'] != ticker_input:
        st.session_state['last_ticker'] = ticker_input
        # Committing CURRENT UI state to ENGINE state for the new search
        st.session_state['confirmed_features'] = [c for c in FEATURE_MAP.values() if st.session_state.get(f"feat_{c}", True)]
        st.session_state['skip_render'] = False

with st.expander("🛠️ Advanced Model Settings", expanded=False):
    c1, c2 = st.columns(2)
    # Select/Deselect Utilities
    if c1.button("✅ Select All Features", use_container_width=True):
        for col_name in FEATURE_MAP.values():
            st.session_state[f"feat_{col_name}"] = True
        st.session_state['skip_render'] = True
        st.rerun()
    
    if c2.button("❌ Deselect All Features", use_container_width=True):
        for col_name in FEATURE_MAP.values():
            st.session_state[f"feat_{col_name}"] = False
        st.session_state['skip_render'] = True
        st.rerun()

    with st.form("feature_selection_form"):
        st.markdown("Select features to include in the Random Forest model training:")
        cols = st.columns(3)
        for i, (label, col_name) in enumerate(FEATURE_MAP.items()):
            cols[i % 3].checkbox(label, key=f"feat_{col_name}")

        if st.form_submit_button("🚀 Re-Train Model", use_container_width=True):
            st.session_state['confirmed_features'] = [c for c in FEATURE_MAP.values() if st.session_state.get(f"feat_{c}", True)]
            st.session_state['skip_render'] = False
            st.toast("Re-training model with new feature set...", icon="🔄")

# The model ALWAYS uses the 'confirmed' set for calculation
selected_features = st.session_state['confirmed_features']

# Watchlist Initialization
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = {}


def evaluate_custom_features(ticker_input, exchange, selected_features):
    try:
        data_1d, data_1h, symbol = fetch_stock_data(ticker_input, exchange)
        if data_1d.empty or data_1h.empty:
            return None
        
        df_1d = data_1d.copy()
        if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = [col[0] if isinstance(col, tuple) else col for col in df_1d.columns]
        df_1d = df_1d.reset_index()
        
        if 'Date' in df_1d.columns:
            df_1d['DateStr'] = pd.to_datetime(df_1d['Date']).dt.strftime('%Y-%m-%d')
        elif 'Datetime' in df_1d.columns:
            df_1d['DateStr'] = pd.to_datetime(df_1d['Datetime']).dt.strftime('%Y-%m-%d')
            
        if 'Close' in df_1d.columns and len(df_1d) >= 14:
            df_1d['Anchored_Close_Return_5'] = (df_1d['Close'] - df_1d['Close'].shift(5)) / df_1d['Close'].shift(5)
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
        
        obv_sign = np.where(df['Close'] > df['Close'].shift(1), 1,
                   np.where(df['Close'] < df['Close'].shift(1), -1, 0))
        df['OBV'] = (obv_sign * df['Volume']).cumsum()
        df['OBV_Slope'] = df['OBV'].diff(14) / df['Volume'].rolling(window=14, min_periods=5).mean()
        
        daily_cols = ['DateStr', 'Anchored_Close_Return_5', 'Daily_ATR_14', 'Daily_RSI_14', 'Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist', 'US_Overnight_Return']
        merge_cols = [c for c in daily_cols if c in df_1d.columns]
        
        # Shift daily features by 1 day to prevent intraday data leakage
        cols_to_shift = ['Anchored_Close_Return_5', 'Daily_ATR_14', 'Daily_RSI_14', 'Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist']
        for c in cols_to_shift:
            if c in df_1d.columns:
                df_1d[c] = df_1d[c].shift(1)
                
        daily_subset = df_1d[merge_cols].dropna()
        df = pd.merge(df, daily_subset, on='DateStr', how='left')
        
        for col in merge_cols:
            if col != 'DateStr':
                df[col] = df[col].ffill()
        
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
        ml_df = ml_df.dropna(subset=selected_features + ['Target'])
        
        prob_pct = 0.0
        prob_long = 0.0
        prob_short = 0.0
        test_accuracy = 0.0
        baseline_accuracy = 0.0
        true_edge = 0.0
        latest_result_html = ""
        hist_long_pct = 0.0
        hist_short_pct = 0.0
        ml_pred_label = ""
        ml_color = "#AAAAAA"
        ml_bg_color = "rgba(128,128,128,0.05)"
        baseline_label = "N/A"
        y_test_len = 0
        
        if len(ml_df) > 10:
            X = ml_df[selected_features].astype(float)
            y = ml_df['Target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            y_test_len = len(y_test)
            
            target_counts = y.value_counts(normalize=True)
            hist_long_pct = target_counts.get(1.0, 0.0) * 100
            hist_short_pct = target_counts.get(-1.0, 0.0) * 100
            
            base_ensemble = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=15, class_weight='balanced', random_state=42)
            
            eval_model = clone(base_ensemble)
            eval_model.fit(X_train, y_train)
            test_accuracy = eval_model.score(X_test, y_test)
            
            y_test_series = pd.Series(y_test)
            baseline_accuracy = y_test_series.value_counts(normalize=True).max()
            baseline_class_raw = y_test_series.value_counts(normalize=True).idxmax()
            baseline_label = "LONG" if baseline_class_raw == 1.0 else "SHORT"
            true_edge = test_accuracy - baseline_accuracy
            
            today_ist_wf = pd.Timestamp.today(tz='Asia/Kolkata')
            last_df_date_str_wf = df['DateStr'].iloc[-1]
            is_current_day_live = (today_ist_wf.hour < 16 and last_df_date_str_wf == today_ist_wf.strftime('%Y-%m-%d'))
            
            def get_amo_val(val):
                return "LONG" if val == 1.0 else "SHORT"
                
            lookback_days = min(5, len(X) - 2)
            correct_count = 0
            eval_results = []
            
            if lookback_days > 0:
                start_idx = 1
                last_x_target_date_wf = date_to_next_date.get(ml_df.iloc[-1]['DateStr'], "")
                if is_current_day_live and last_x_target_date_wf == today_ist_wf.strftime('%Y-%m-%d'):
                    start_idx = 2
                    
                end_idx = lookback_days + start_idx
                max_available = len(X) - 1
                
                for i in range(start_idx, min(end_idx, max_available + 1)):
                    test_idx = -i
                    eval_wf_model = clone(base_ensemble)
                    eval_wf_model.fit(X.iloc[:test_idx], y.iloc[:test_idx])
                    
                    test_X = X.iloc[[test_idx]]
                    actual_y = y.iloc[test_idx]
                    pred_y = eval_wf_model.predict(test_X)[0]
                    
                    is_correct = (pred_y == actual_y)
                    if is_correct:
                        correct_count += 1
                        
                    actual_lbl = get_amo_val(actual_y)
                    pred_lbl = get_amo_val(pred_y)
                    
                    feature_date = ml_df.iloc[test_idx]['DateStr']
                    date_label = date_to_next_date.get(feature_date, feature_date)
                    
                    if is_correct:
                        eval_results.append(f"<li style='margin-bottom: 4px;'><span style='color: #00C073;'>✅ {date_label}: Validated (Predicted {pred_lbl})</span></li>")
                    else:
                        eval_results.append(f"<li style='margin-bottom: 4px;'><span style='color: #FF2B2B;'>❌ {date_label}: Failed (Pred {pred_lbl} != Act {actual_lbl})</span></li>")
                        
                eval_results.reverse()
                val_count = len(eval_results)
                latest_result_html = f"<div style='margin-bottom: 8px;'><b style='color: black;'>Recent Regime Sync: {correct_count}/{val_count} Correct</b></div>"
                latest_result_html += f"<ul style='list-style-type: none; padding-left: 0; margin: 0; font-size: 0.95rem;'>" + "".join(eval_results) + "</ul>"
            else:
                latest_result_html = "<span>Not enough data for 5-Day Validation.</span>"
            
            model = clone(base_ensemble)
            
            today_ist = pd.Timestamp.today(tz='Asia/Kolkata')
            today_ist_str = today_ist.strftime('%Y-%m-%d')
            last_df_date_str = df['DateStr'].iloc[-1]
            
            if today_ist.hour < 16 and last_df_date_str == today_ist_str:
                last_x_date = ml_df.iloc[-1]['DateStr']
                last_x_target_date = date_to_next_date.get(last_x_date, "")
                if last_x_target_date == today_ist_str:
                    model.fit(X.iloc[:-1], y.iloc[:-1])
                else:
                    model.fit(X, y)
                
                available_dates = list(df['DateStr'].unique())
                feature_day_str = available_dates[-2] if len(available_dates) > 1 else available_dates[-1]
                today_features = df[df['DateStr'] == feature_day_str].tail(1)[selected_features].astype(float)
            else:
                model.fit(X, y)
                today_features = df.groupby('DateStr').tail(1).iloc[-1][selected_features].to_frame().T.astype(float)

            if not today_features.isna().any().any():
                prob_array = model.predict_proba(today_features)[0]
                pred_class = model.predict(today_features)[0]
                
                class_labels = list(model.classes_)
                try:
                    prob_long = prob_array[class_labels.index(1.0)] * 100 if 1.0 in class_labels else 0.0
                    prob_short = prob_array[class_labels.index(-1.0)] * 100 if -1.0 in class_labels else 0.0
                except ValueError:
                    pass
                
                if pred_class == 1.0:
                    ml_pred_label = "LONG AMO"
                    ml_color = "#00C073"
                    ml_bg_color = "rgba(0, 192, 115, 0.05)"
                else:
                    ml_pred_label = "SHORT AMO"
                    ml_color = "#FF2B2B"
                    ml_bg_color = "rgba(255, 43, 43, 0.05)"
                    
                prob_pct = max(prob_array) * 100

        return {
            "symbol": symbol,
            "prob_pct": prob_pct,
            "prob_long": prob_long,
            "prob_short": prob_short,
            "hist_long": hist_long_pct,
            "hist_short": hist_short_pct,
            "test_accuracy": test_accuracy * 100,
            "test_samples": y_test_len,
            "baseline_accuracy": baseline_accuracy * 100,
            "baseline_label": baseline_label,
            "true_edge": true_edge * 100,
            "ml_pred_label": ml_pred_label,
            "ml_color": ml_color,
            "ml_bg_color": ml_bg_color,
            "latest_result_html": latest_result_html,
            "features_used": ", ".join([{v: k for k, v in FEATURE_MAP.items()}.get(f, f) for f in selected_features])
        }
    except Exception as e:
        print(f"Error in evaluate_custom_features: {e}")
        return None

def render_main_dashboard(ticker_input, exchange, selected_features, render_ui=True):
    ctx = st.spinner(f"Fetching data and calculating indicators for {ticker_input}...") if render_ui else nullcontext()
    with ctx:
        try:
            data_1d, data_1h, symbol = fetch_stock_data(ticker_input, exchange) # Changed data_15m to data_1h
            
            if data_1d.empty or data_1h.empty:
                if render_ui:
                    st.warning(f"No data found for {ticker_input}. Please check the ticker symbol.")
                    st.stop()
                return
            
            
            # 1. PROCESS DAILY DATA
            df_1d = data_1d.copy()
            if isinstance(df_1d.columns, pd.MultiIndex):
                df_1d.columns = [col[0] if isinstance(col, tuple) else col for col in df_1d.columns]
            df_1d = df_1d.reset_index()
            
            if 'Date' in df_1d.columns:
                df_1d['DateStr'] = pd.to_datetime(df_1d['Date']).dt.strftime('%Y-%m-%d')
            elif 'Datetime' in df_1d.columns:
                df_1d['DateStr'] = pd.to_datetime(df_1d['Datetime']).dt.strftime('%Y-%m-%d')
                
            if 'Close' in df_1d.columns and len(df_1d) >= 14:
                df_1d['Anchored_Close_Return_5'] = (df_1d['Close'] - df_1d['Close'].shift(5)) / df_1d['Close'].shift(5)
                df_1d['Daily_ATR_14'] = df_1d.ta.atr(length=14)
                df_1d['Daily_RSI_14'] = df_1d.ta.rsi(length=14)
                
            # 1.5 MERGE NIFTY MACO DATA
            nifty_df = fetch_nifty_data()
            if not nifty_df.empty:
                df_1d = pd.merge(df_1d, nifty_df, on='DateStr', how='left')
                # Forward fill nifty data in case of slight timestamp mismatches
                for col in ['Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist']:
                    if col in df_1d.columns:
                        df_1d[col] = df_1d[col].ffill()
                        
            # 1.6 MERGE US OVERNIGHT DATA
            sp_df = fetch_global_sentiment_data()
            if not sp_df.empty:
                df_1d = pd.merge(df_1d, sp_df, on='DateStr', how='left')
                if 'US_Overnight_Return' in df_1d.columns:
                    df_1d['US_Overnight_Return'] = df_1d['US_Overnight_Return'].ffill()
                
            # 2. PROCESS 1h DATA
            df = data_1h.copy() # Changed data_15m to data_1h
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
            
            # Calculate rolling indicators directly on 1h chart
            df['Closing_Momentum'] = (df['Close'] - df['Open']) / df['Open']
            
            obv_sign = np.where(df['Close'] > df['Close'].shift(1), 1,
                       np.where(df['Close'] < df['Close'].shift(1), -1, 0))
            df['OBV'] = (obv_sign * df['Volume']).cumsum()
            df['OBV_Slope'] = df['OBV'].diff(14) / df['Volume'].rolling(window=14, min_periods=5).mean()
            
            # 3. MERGE DAILY DATA (Stock + NIFTY Macro)
            daily_cols = ['DateStr', 'Anchored_Close_Return_5', 'Daily_ATR_14', 'Daily_RSI_14', 'Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist', 'US_Overnight_Return']
            merge_cols = [c for c in daily_cols if c in df_1d.columns]
            
            # Shift daily features by 1 day to prevent intraday data leakage
            cols_to_shift = ['Anchored_Close_Return_5', 'Daily_ATR_14', 'Daily_RSI_14', 'Nifty_Momentum', 'Nifty_RSI_14', 'Nifty_Trend_Dist']
            for c in cols_to_shift:
                if c in df_1d.columns:
                    df_1d[c] = df_1d[c].shift(1)
                    
            daily_subset = df_1d[merge_cols].dropna()
            df = pd.merge(df, daily_subset, on='DateStr', how='left')
            
            # Forward fill all merged daily indicators to ensure 1h rows have macro context
            for col in merge_cols:
                if col != 'DateStr':
                    df[col] = df[col].ffill()
            
            df['ATR_Percent'] = df['Daily_ATR_14'] / df['Close']
            
            # 3.5 ADVANCED INTRADAY FEATURES
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['TP_Volume'] = df['Typical_Price'] * df['Volume']
            df['Cum_Vol'] = df.groupby('DateStr')['Volume'].cumsum()
            df['Cum_TP_Vol'] = df.groupby('DateStr')['TP_Volume'].cumsum()
            df['VWAP'] = df['Cum_TP_Vol'] / df['Cum_Vol']
            df['VWAP_Distance'] = (df['Close'] - df['VWAP']) / df['VWAP']
            
            # 3.55 DYNAMIC FEATURES
            df['Frac_Diff_Close'] = frac_diff_ffd(df['Close'], d=0.4)
            
            # 3.55 MORNING AUTOCORRELATION (10:15 AM Price vs Open)
            df = df.sort_values(['DateStr', 'DatetimeObj'])
            day_opens = df.groupby('DateStr')['Open'].transform('first')
            p1015 = df[df['TimeStr'] == '10:15'].set_index('DateStr')['Close']
            df['Morning_Autocorr'] = (df['DateStr'].map(p1015) - day_opens) / day_opens
            
            # 3.55 ORDER FLOW IMBALANCE
            hl_range = df['High'] - df['Low']
            hl_range = hl_range.replace(0, np.nan)
            df['OFI'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range).rolling(window=5, min_periods=1).mean()
            # 4. ENGINEER AMO TARGET (Positionally Anchored Sustained Trend)
            daily_targets = {}
            today_ist_str = pd.Timestamp.today(tz='Asia/Kolkata').strftime('%Y-%m-%d')
            market_open = pd.Timestamp.today(tz='Asia/Kolkata').hour < 16
            
            for date_str, group in df.groupby('DateStr'):
                group = group.sort_values(by='DatetimeObj')
                
                # Explicitly look up the 9:15 AM candle (market open) and 10:15 AM candle
                # to avoid misalignment caused by missing candles (circuit limits, API drops).
                candle_915 = group[group['TimeStr'] == '09:15']
                candle_1015 = group[group['TimeStr'] == '10:15']
                
                if candle_915.empty or candle_1015.empty:
                    continue  # Skip days where either key candle is missing
                    
                open_price = candle_915.iloc[0]['Open']
                close_price = candle_1015.iloc[0]['Close']
                
                if close_price > open_price:
                    daily_targets[date_str] = 1.0
                else:
                    daily_targets[date_str] = -1.0
            
            # 5. ML DATASET FILTRATION (Strictly Final Hourly Candles)
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
            
            # ML DropNA strictly applies to ALL selected features plus the resulting prediction target
            ml_df = ml_df.dropna(subset=selected_features + ['Target'])
            
            bullish_prob = None
            ml_details = None
            ml_pred_label = ""
            ml_color = "#AAAAAA"
            ml_bg_color = "rgba(128,128,128,0.05)"
            prob_pct = 0.0
            prob_long = 0.0
            prob_short = 0.0
            test_accuracy = 0.0
            baseline_accuracy = 0.0
            true_edge = 0.0
            latest_result_html = ""
            hist_long_pct = 0.0
            hist_short_pct = 0.0
            
            if len(ml_df) > 10:
                X = ml_df[selected_features].astype(float)
                y = ml_df['Target']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                # Historical class distribution from training data
                target_counts = y.value_counts(normalize=True)
                hist_long_pct = target_counts.get(1.0, 0.0) * 100
                hist_short_pct = target_counts.get(-1.0, 0.0) * 100
                
                base_ensemble = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=15, class_weight='balanced', random_state=42)
                
                eval_model = clone(base_ensemble)
                eval_model.fit(X_train, y_train)
                test_accuracy = eval_model.score(X_test, y_test)
                
                y_test_series = pd.Series(y_test)
                baseline_accuracy = y_test_series.value_counts(normalize=True).max()
                
                baseline_class_raw = y_test_series.value_counts(normalize=True).idxmax()
                
                if baseline_class_raw == 1.0:
                    baseline_label = "LONG"
                else:
                    baseline_label = "SHORT"
                    
                true_edge = test_accuracy - baseline_accuracy
                
                today_ist_wf = pd.Timestamp.today(tz='Asia/Kolkata')
                last_df_date_str_wf = df['DateStr'].iloc[-1]
                is_current_day_live = (today_ist_wf.hour < 16 and last_df_date_str_wf == today_ist_wf.strftime('%Y-%m-%d'))
                
                # 5-Day Walk-Forward Validation
                def get_amo_val(val):
                    return "LONG" if val == 1.0 else "SHORT"
                    
                lookback_days = min(5, len(X) - 2)
                correct_count = 0
                eval_results = []
                
                if lookback_days > 0:
                    start_idx = 1
                    
                    last_x_target_date_wf = date_to_next_date.get(ml_df.iloc[-1]['DateStr'], "")
                    if is_current_day_live and last_x_target_date_wf == today_ist_wf.strftime('%Y-%m-%d'):
                        # Skip validating the current active day to hide leakage on the UI
                        start_idx = 2
                        
                    end_idx = lookback_days + start_idx
                    max_available = len(X) - 1
                    
                    for i in range(start_idx, min(end_idx, max_available + 1)):
                        test_idx = -i
                        eval_wf_model = clone(base_ensemble)
                        
                        eval_wf_model.fit(X.iloc[:test_idx], y.iloc[:test_idx])
                            
                        # Process single validation day
                        test_X = X.iloc[[test_idx]]
                        actual_y = y.iloc[test_idx]
                        
                        pred_y = eval_wf_model.predict(test_X)[0]
                        
                        is_correct = (pred_y == actual_y)
                        if is_correct:
                            correct_count += 1
                            
                        actual_lbl = get_amo_val(actual_y)
                        pred_lbl = get_amo_val(pred_y)
                        
                        feature_date = ml_df.iloc[test_idx]['DateStr']
                        date_label = date_to_next_date.get(feature_date, feature_date)
                        
                        if is_correct:
                            eval_results.append(f"<li style='margin-bottom: 4px;'><span style='color: #00C073;'>✅ {date_label}: Validated (Predicted {pred_lbl})</span></li>")
                        else:
                            eval_results.append(f"<li style='margin-bottom: 4px;'><span style='color: #FF2B2B;'>❌ {date_label}: Failed (Pred {pred_lbl} != Act {actual_lbl})</span></li>")
                            
                    eval_results.reverse() # Show oldest to newest
                    
                    val_count = len(eval_results)
                    latest_result_html = f"<div style='margin-bottom: 8px;'><b style='color: black;'>Recent Regime Sync: {correct_count}/{val_count} Correct</b></div>"
                    latest_result_html += f"<ul style='list-style-type: none; padding-left: 0; margin: 0; font-size: 0.95rem;'>" + "".join(eval_results) + "</ul>"
                else:
                    latest_result_html = "<span>Not enough data for 5-Day Validation.</span>"
                
                # Primary model must train strictly preserving out-of-sample prediction integrity
                model = clone(base_ensemble)
                
                today_ist = pd.Timestamp.today(tz='Asia/Kolkata')
                today_ist_str = today_ist.strftime('%Y-%m-%d')
                last_df_date_str = df['DateStr'].iloc[-1]
                
                if today_ist.hour < 16 and last_df_date_str == today_ist_str:
                    # Market is still open. Predict for TODAY using YESTERDAY's data.
                    last_x_date = ml_df.iloc[-1]['DateStr']
                    last_x_target_date = date_to_next_date.get(last_x_date, "")
                    
                    # If today's 10:15 close exists in the dataset and was assigned as a target,
                    # the final row in X holds today's true target. We must exclude it
                    # from the training set to avoid data leakage!
                    if last_x_target_date == today_ist_str:
                        model.fit(X.iloc[:-1], y.iloc[:-1])
                        training_end_date = ml_df.iloc[-2]['DateStr'] if len(ml_df) > 1 else ml_df.iloc[-1]['DateStr']
                    else:
                        model.fit(X, y)
                        training_end_date = ml_df.iloc[-1]['DateStr']
                    
                    available_dates = list(df['DateStr'].unique())
                    feature_day_str = available_dates[-2] if len(available_dates) > 1 else available_dates[-1]
                    today_features = df[df['DateStr'] == feature_day_str].tail(1)[selected_features].astype(float)
                    st.session_state['forecast_type'] = "Current Day"
                    st.session_state['feature_day_str'] = feature_day_str
                    st.session_state['training_end_day_str'] = training_end_date
                else:
                    # Market is closed (>= 4PM). Predict for TOMORROW using TODAY's data.
                    model.fit(X, y)
                    training_end_date = ml_df.iloc[-1]['DateStr']
                    
                    feature_day_str = df.groupby('DateStr').tail(1).iloc[-1]['DateStr']
                    today_features = df.groupby('DateStr').tail(1).iloc[-1][selected_features].to_frame().T.astype(float)
                    st.session_state['forecast_type'] = "Next Day"
                    st.session_state['feature_day_str'] = feature_day_str
                    st.session_state['training_end_day_str'] = training_end_date

                if not today_features.isna().any().any():
                    prob_array = model.predict_proba(today_features)[0]
                    pred_class = model.predict(today_features)[0]
                    
                    class_labels = list(model.classes_)
                    try:
                        prob_long = prob_array[class_labels.index(1.0)] * 100 if 1.0 in class_labels else 0.0
                        prob_short = prob_array[class_labels.index(-1.0)] * 100 if -1.0 in class_labels else 0.0
                    except ValueError:
                        pass
                    
                    if pred_class == 1.0:
                        ml_pred_label = "LONG AMO"
                        ml_color = "#00C073" # Green
                        ml_bg_color = "rgba(0, 192, 115, 0.05)"
                    else:
                        ml_pred_label = "SHORT AMO"
                        ml_color = "#FF2B2B" # Red
                        ml_bg_color = "rgba(255, 43, 43, 0.05)"
                        
                    prob_pct = max(prob_array) * 100
                    
                    # Dynamic Importance Mapping
                    rev_map = {v: k for k, v in FEATURE_MAP.items()}
                    importances_dict = {rev_map[f]: model.feature_importances_[i] for i, f in enumerate(selected_features)}
                    
                    ml_details = {
                        "accuracy": test_accuracy,
                        "baseline": baseline_accuracy,
                        "true_edge": true_edge,
                        "baseline_label": baseline_label,
                        "samples": len(ml_df),
                        "test_samples": len(y_test),
                        "importances": importances_dict
                    }
            
            # 6. UI CONSTRUCTION LAYER
            if render_ui:
                st.markdown("---")
                st.markdown(f"<h2 style='text-align: left; color: black;'>Stock: <b style='color: #1D4ED8;'>{symbol}</b></h2>", unsafe_allow_html=True)

            def render_indicator(col, title, value, title_color="gray"):
                html = f"""
                <div style="background-color: rgba(255,255,255,0.05); border: 2px solid #EEE; border-radius: 10px; padding: 15px 5px; text-align: center; height: 110px; margin-bottom: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <p style="margin:0; font-size:12px; font-weight:bold; color:{title_color};">{title}</p>
                    <h2 style="margin:10px 0; font-size:22px; color:black; padding: 0px 5px;">{value}</h2>
                </div>
                """
                col.markdown(html, unsafe_allow_html=True)
                
            st.write("")
            latest_day = df.iloc[-1]
            
            cols = st.columns(6)
            close_val = f"Rs.{latest_day['Close']:.2f}" if 'Close' in df.columns and pd.notna(latest_day['Close']) else "N/A"
            render_indicator(cols[0], "Current Close", close_val, "#1D4ED8")
            
            mom_v = latest_day['Closing_Momentum'] if 'Closing_Momentum' in latest_day else float('nan')
            mom_str = f"{mom_v*100:+.2f}%" if pd.notna(mom_v) else "N/A"
            render_indicator(cols[1], "Momentum (1H)", mom_str)
            
            obv_v = latest_day['OBV_Slope'] if 'OBV_Slope' in latest_day else float('nan')
            obv_str = f"{obv_v:+.2f}" if pd.notna(obv_v) else "N/A"
            render_indicator(cols[2], "OBV Slope", obv_str)
            
            anchor_cr5 = latest_day['Anchored_Close_Return_5'] if 'Anchored_Close_Return_5' in latest_day else float('nan')
            anchor_cr5_str = f"{anchor_cr5*100:+.2f}%" if pd.notna(anchor_cr5) else "N/A"
            render_indicator(cols[3], "Anchored CR5", anchor_cr5_str)
            
            atr_v = latest_day['ATR_Percent'] if 'ATR_Percent' in latest_day else float('nan')
            atr_str = f"{atr_v*100:.2f}%" if pd.notna(atr_v) else "N/A"
            render_indicator(cols[4], "Volatility (%)", atr_str)
            
            rsi_v = latest_day['Daily_RSI_14'] if 'Daily_RSI_14' in latest_day else float('nan')
            rsi_str = f"{rsi_v:.1f}" if pd.notna(rsi_v) else "N/A"
            render_indicator(cols[5], "Daily RSI", rsi_str)
            
            # 7. RENDER ML PREDICTION DOM
            if ml_pred_label:
                last_df_date = pd.to_datetime(df['DateStr'].iloc[-1])
                forecast_type = st.session_state.get('forecast_type', "Next Day")
                
                if forecast_type == "Current Day":
                    prediction_ts = last_df_date
                    forecast_title = f"AMO Current Day Forecast ({prediction_ts.strftime('%A, %b %d, %Y')})"
                else:
                    # Use the later of today or last data date as the base,
                    # then find the next weekday (Mon-Fri) after it.
                    today_date = pd.Timestamp.today(tz='Asia/Kolkata').normalize().tz_localize(None)
                    base_date = max(last_df_date, today_date)
                    prediction_ts = base_date + pd.Timedelta(days=1)
                    while prediction_ts.weekday() >= 5:  # 5=Sat, 6=Sun
                        prediction_ts += pd.Timedelta(days=1)
                    forecast_title = f"AMO Next Day Forecast ({prediction_ts.strftime('%A, %b %d, %Y')})"
                    
                st.session_state['last_data_date'] = prediction_ts
                
                st.session_state['watchlist'][symbol] = {
                    'prob': prob_pct, 
                    'prob_long': prob_long,
                    'prob_short': prob_short,
                    'hist_long': hist_long_pct,
                    'hist_short': hist_short_pct,
                    'acc': test_accuracy * 100,
                    'test_samples': len(y_test),
                    'baseline': baseline_accuracy * 100,
                    'baseline_label': baseline_label,
                    'edge': true_edge * 100,
                    'label': ml_pred_label,
                    'color': ml_color,
                    'bg_color': ml_bg_color,
                    'latest_result': latest_result_html,
                    'selected_features': ", ".join([{v: k for k, v in FEATURE_MAP.items()}.get(f, f) for f in selected_features])
                }
                
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 2.5rem; border-radius: 12px; background-color: {ml_bg_color}; border: 2px solid {ml_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 2rem; margin-bottom: 1rem;">
                        <h4 style="margin-bottom: 0px; margin-top: 0px; color: black; font-weight: 600;">{forecast_title}</h4>
                        <p style="margin: 4px 0 0 0; font-size: 1rem; color: black;">Predicting the price movement for the first two candles of the day so that you can close your AMO position by 11:15 AM.</p>
                        <h1 style="color: {ml_color}; font-size: 3.5rem; margin: 10px 0px;">{prob_pct:.1f}% <span style="font-size: 1.8rem; font-weight: 400;">({ml_pred_label})</span></h1>
                        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                            <div style="background-color: rgba(0, 192, 115, 0.1); border: 1px solid #00C073; padding: 10px 20px; border-radius: 8px;">
                                <p style="margin: 0; font-size: 0.9rem; color: #555; text-transform: uppercase; font-weight: 600;">Long AMO</p>
                                <h3 style="margin: 5px 0 0 0; color: #00C073;">{prob_long:.1f}%</h3>
                            </div>
                            <div style="background-color: rgba(255, 43, 43, 0.1); border: 1px solid #FF2B2B; padding: 10px 20px; border-radius: 8px;">
                                <p style="margin: 0; font-size: 0.9rem; color: #555; text-transform: uppercase; font-weight: 600;">Short AMO</p>
                                <h3 style="margin: 5px 0 0 0; color: #FF2B2B;">{prob_short:.1f}%</h3>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 15px;">
                            <div style="background-color: rgba(0,0,0,0.03); padding: 8px 16px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.08);">
                                <p style="margin: 0; font-size: 0.75rem; color: #888; text-transform: uppercase; font-weight: 600;">Historical Long</p>
                                <h4 style="margin: 3px 0 0 0; color: #00C073; font-size: 1rem;">{hist_long_pct:.1f}%</h4>
                            </div>
                            <div style="background-color: rgba(0,0,0,0.03); padding: 8px 16px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.08);">
                                <p style="margin: 0; font-size: 0.75rem; color: #888; text-transform: uppercase; font-weight: 600;">Historical Short</p>
                                <h4 style="margin: 3px 0 0 0; color: #FF2B2B; font-size: 1rem;">{hist_short_pct:.1f}%</h4>
                            </div>
                        </div>
                        <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid rgba(0,0,0,0.05);">
                            <p style="margin: 0; font-size: 1rem; color: #555; text-transform: uppercase; font-weight: 600; margin-bottom: 10px;">5-Day Walk-Forward Validation</p>
                            <div style="text-align: left; padding: 8px; border-radius: 6px; border: 1px solid #DDD; display: inline-block;">{latest_result_html}</div>
                        </div>
                        <p style="color: {ml_color}; font-size: 1.1rem; margin-top: 20px;"><em>Multi-class Random Forest Matrix targeting sustained (10:15 AM) close thresholds</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if ml_details:
                    st.markdown("""
                        <style>
                        div[data-testid="stExpander"] details summary p {
                            font-size: 20px !important;
                            color: black !important;
                            font-weight: 600 !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("View AMO Machine Learning Model Details", expanded=False):
                        raw_sessions = df['DateStr'].nunique()
                        trained_sessions = ml_details['samples']
                        st.markdown(
                            f"**Valid Hourly Training Intervals:** {trained_sessions} market sessions "
                            f"<span style='color:#888; font-size:0.9em;'>({raw_sessions} total calendar sessions fetched -- "
                            f"difference accounts for weekends, NSE holidays, circuit-breaker days with missing candles, and rolling-indicator warmup)</span>",
                            unsafe_allow_html=True
                        )
                        
                        acc = ml_details['accuracy'] * 100
                        base = ml_details['baseline'] * 100
                        edge = ml_details['true_edge'] * 100
                        
                        edge_color = "#00C073" if edge > 0 else "#FF2B2B"
                        
                        st.markdown(f"""
                        <ul style="list-style-type: none; padding-left: 0; margin-top: 10px; margin-bottom: 20px;">
                            <li><b>Predictive Accuracy:</b> {acc:.1f}% <span style="color: #666; font-size: 0.9em;">(Tested on {ml_details['test_samples']} Out-of-Sample sessions)</span></li>
                            <li><b>Baseline Accuracy:</b> {base:.1f}% <span style="color: #666; font-size: 0.9em;">(Always guessing '{ml_details['baseline_label']}')</span></li>
                            <li><b>True Edge:</b> <span style="color: {edge_color}; font-weight: bold;">{edge:+.1f}%</span></li>
                        </ul>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**AMO Feature Importances:**")
                        fi_df = pd.DataFrame(
                            list(ml_details['importances'].values()),
                            index=list(ml_details['importances'].keys()),
                            columns=["Relative Importance"]
                        )
                        st.bar_chart(fi_df, height=200)
                        
                        st.markdown("**AMO Feature Correlation Matrix:**")
                        ml_features = ml_df[selected_features + ['Target']]
                        styled_corr = ml_features.corr().style.background_gradient(cmap="Oranges").format("{:.2f}")
                        st.dataframe(styled_corr, use_container_width=True)
                        
                    with st.expander("View Raw Hourly Machine Learning Training Data", expanded=False):
                        st.markdown("This targeted intraday matrix maps exclusively the final hourly closing datasets evaluated natively across 730 days exactly against the sustained 10:15 AM close target thresholds:")
                        display_df = ml_df[selected_features + ['Target', 'DateStr']].copy()
                        
                        # Apply the same exact training boundary cut-off to reflect the true model state without leaks
                        train_limit = st.session_state.get('training_end_day_str', display_df['DateStr'].iloc[-1])
                        display_df = display_df[display_df['DateStr'] <= train_limit]
                        
                        display_df = display_df.set_index('DateStr')
                        st.dataframe(display_df, use_container_width=True)
                    
                    with st.expander("🔍 View Feature Snapshot (Prediction Input)", expanded=False):
                        feat_day_str = st.session_state.get('feature_day_str', df['DateStr'].iloc[-1])
                        train_end_str = st.session_state.get('training_end_day_str', df['DateStr'].iloc[-1])
                        
                        feat_day = pd.to_datetime(feat_day_str).strftime('%A, %d %b %Y')
                        train_end_day = pd.to_datetime(train_end_str).strftime('%A, %d %b %Y')
                        
                        st.markdown(f"**Training Data End:** The model was trained aggressively on all valid historical data strictly up to **{train_end_day}**.")
                        st.markdown(f"**Feature Snapshot Date:** This is the **single feature row** sourced directly from the session on **{feat_day}** to generate the forecast above.")
                        
                        if today_features is not None and not today_features.empty:
                            snapshot_df = today_features.copy()
                            snapshot_df.index = [st.session_state.get('last_ticker', symbol)]
                            snapshot_df.index.name = "Ticker"
                            st.dataframe(snapshot_df.T.rename(columns={snapshot_df.index[0]: "Value"}).style.format("{:.4f}"), use_container_width=True)
                        else:
                            st.warning("Feature snapshot is unavailable -- today's data may still be loading.")
                        
            with st.expander(f"View Latest Market News on {symbol}", expanded=False):
                news_articles = get_top_news(ticker_input)
                if news_articles:
                    for article in news_articles:
                        st.markdown(f"""
                        <div style="padding: 1rem; border-radius: 8px; background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); margin-bottom: 0.8rem;">
                            <a href="{article['link']}" target="_blank" style="text-decoration: none; color: #1D4ED8; font-size: 1.1rem; font-weight: bold;">{article['title']}</a>
                            <p style="margin-top: 0.5rem; margin-bottom: 0px; color: #888; font-size: 0.9rem;">{article['date']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #888;'>No recent news articles found for this ticker.</p>", unsafe_allow_html=True)
            
            with st.expander(f"🔮 View Munafa Sutra Prediction for {symbol}", expanded=False):
                with st.spinner("Fetching prediction from Munafa Sutra..."):
                    munafa_pred = fetch_munafasutra_prediction(ticker_input).replace('\n', '<br>')
                    st.markdown(f"<div style='padding:1rem; border-radius:8px; background-color:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05);'><p style='color: black; font-size: 1rem; line-height: 1.6;'>{munafa_pred}</p></div>", unsafe_allow_html=True)
            
        except Exception as e:
            if render_ui:
                import traceback
                st.error(f"An error occurred while fetching data: {e}\n\nTraceback: {traceback.format_exc()}")

with st.expander("📂 Batch Watchlist Import", expanded=False):
    st.markdown("Upload a `.txt` file with one stock ticker per line. The model will run on each ticker using the currently selected features and add results to the watchlist.")
    batch_uploaded_file = st.file_uploader("Select ticker file", type=["txt"], key="batch_upload")
    if batch_uploaded_file is not None:
        batch_tickers_raw = batch_uploaded_file.getvalue().decode("utf-8").splitlines()
        batch_tickers = [line.strip().upper() for line in batch_tickers_raw if line.strip()]
        if batch_tickers:
            st.caption(f"{len(batch_tickers)} tickers detected: {', '.join(batch_tickers[:10])}{'...' if len(batch_tickers) > 10 else ''}")
            if st.button(f"▶ Run Model on {len(batch_tickers)} Tickers & Add to Watchlist", type="primary"):
                batch_progress = st.progress(0)
                batch_status = st.empty()
                for i, t in enumerate(batch_tickers):
                    batch_status.text(f"Processing {t} ({i+1}/{len(batch_tickers)})...")
                    render_main_dashboard(t, exchange, selected_features, render_ui=False)
                    batch_progress.progress((i + 1) / len(batch_tickers))
                    time.sleep(1.5)  # Respect yfinance rate limit
                batch_status.success(f"Done! {len(batch_tickers)} tickers processed and added to the Watchlist.")
                time.sleep(1)
                st.rerun()
        else:
            st.warning("The uploaded file appears to be empty or has no valid ticker names.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Dashboard", "⭐ Watchlist", "🔬 Backtest JSON", "📖 Feature Glossary"])

with tab3:
    st.markdown("### 🔬 Evaluate Custom Feature Sets")
    st.markdown("Upload a feature combination JSON file generated by the brute force search to see their respective evaluations side-by-side.")
    
    uploaded_file = st.file_uploader("Upload optimal_features.json", type=['json'])
    
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            j_ticker = data.get("ticker", "")
            j_exch = data.get("exchange", "NSE")
            j_combs = data.get("top_combinations", [])
            
            if not j_ticker:
                st.error("Invalid JSON: Missing ticker")
            else:
                st.success(f"Successfully loaded feature analysis for **{j_ticker} ({j_exch})**.")
                
                for i, combo in enumerate(j_combs[:3]):
                    rnk = combo.get("rank", i+1)
                    feats = combo.get("features", [])
                    
                    st.markdown(f"<div style='margin-top: 20px; padding: 15px; border-left: 4px solid #1D4ED8; background-color: rgba(0,0,0,0.03); border-radius: 4px;'>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='margin-bottom: 5px; color: black;'>Rank {rnk} Combination</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #555; text-transform: uppercase; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px;'>{', '.join(feats)}</p>", unsafe_allow_html=True)
                    
                    with st.spinner(f"Running ML Evaluation for Rank {rnk}..."):
                        res = evaluate_custom_features(j_ticker, j_exch, feats)
                        
                    if res:
                        c1, c2, c3 = st.columns(3)
                        
                        col1_html = f"""
                        <div style="background-color: {res['ml_bg_color']}; border: 1px solid {res['ml_color']}; border-radius: 8px; padding: 10px; text-align: center; height: 100%;">
                            <p style="margin:0; font-size:12px; font-weight:bold; color:#777; text-transform:uppercase;">AMO Prediction</p>
                            <h2 style="margin:10px 0; font-size:24px; color:{res['ml_color']};">{res['prob_pct']:.1f}% ({res['ml_pred_label']})</h2>
                        </div>
                        """
                        c1.markdown(col1_html, unsafe_allow_html=True)
                        
                        edge_color = "#00C073" if res['true_edge'] > 0 else "#FF2B2B"
                        col2_html = f"""
                        <div style="background-color: rgba(255,255,255,0.05); border: 1px solid #EEE; border-radius: 8px; padding: 10px; text-align: center; height: 100%;">
                            <p style="margin:0; font-size:12px; font-weight:bold; color:#777; text-transform:uppercase;">Test Accuracy</p>
                            <h2 style="margin:5px 0; font-size:24px; color:black;">{res['test_accuracy']:.1f}%</h2>
                            <p style="margin:0; font-size:12px; font-weight:bold; color:{edge_color};">Edge: {res['true_edge']:+.1f}%</p>
                        </div>
                        """
                        c2.markdown(col2_html, unsafe_allow_html=True)
                        
                        c3.markdown(f"<div style='background-color: rgba(0,0,0,0.02); border-radius: 8px; padding: 10px; height: 100%; border: 1px solid #EEE;'><p style='margin:0; margin-bottom:5px; font-size:12px; font-weight:bold; color:#777; text-transform:uppercase;'>5-Day Regime Sync</p><div style='font-size: 0.9rem;'>{res['latest_result_html']}</div></div>", unsafe_allow_html=True)
                        
                    else:
                        st.warning("Failed to evaluate this feature set.")
                        
                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
            
    st.markdown("<hr style='margin-top: 15px; margin-bottom: 15px;'>", unsafe_allow_html=True)
    
    with st.expander("📂 Batch JSON Evaluator & Exporter", expanded=False):
        st.markdown("Upload multiple `.json` files to automatically compile their top optimal evaluations into a single downloadable Excel report.")
        
        batch_jsons = st.file_uploader("Select multiple optimal JSON files", type=["json"], accept_multiple_files=True, key="batch_json_upload")
        
        if batch_jsons:
            if st.button("▶ Run Batch Evaluation on All Files", type="primary"):
                compiled_results = []
                batch_progress = st.progress(0)
                batch_status = st.empty()
                
                total_files = len(batch_jsons)
                
                for idx, file_obj in enumerate(batch_jsons):
                    try:
                        b_data = json.load(file_obj)
                        b_ticker = b_data.get("ticker", "")
                        b_exch = b_data.get("exchange", "NSE")
                        b_combs = b_data.get("top_combinations", [])
                        
                        batch_status.markdown(f"**Processing:** `{b_ticker}` ({idx+1}/{total_files}). Please wait...")
                        
                        if b_ticker and b_combs:
                            for idx_c, combo in enumerate(b_combs[:3]):
                                b_rnk = combo.get("rank", idx_c+1)
                                b_feats = combo.get("features", [])
                                
                                res = evaluate_custom_features(b_ticker, b_exch, b_feats)
                                if res:
                                    compiled_results.append({
                                        "Ticker": b_ticker,
                                        "Rank": b_rnk,
                                        "Exchange": b_exch,
                                        "Highest Prob (%)": round(res.get("prob_pct", 0.0), 1),
                                        "Prediction": res.get("ml_pred_label", "N/A"),
                                        "Test Accuracy (%)": round(res.get("test_accuracy", 0.0), 1),
                                        "True Edge (%)": round(res.get("true_edge", 0.0), 1),
                                        "Features Used": ", ".join(b_feats),
                                        "Long AMO (%)": round(res.get("prob_long", 0.0), 1),
                                        "Short AMO (%)": round(res.get("prob_short", 0.0), 1),
                                        "Hist Long (%)": round(res.get("hist_long", 0.0), 1),
                                        "Hist Short (%)": round(res.get("hist_short", 0.0), 1)
                                    })
                    except Exception as e:
                        st.error(f"Failed processing {file_obj.name}: {e}")
                    
                    batch_progress.progress((idx + 1) / total_files)
                
                batch_status.empty()
                
                if compiled_results:
                    st.success(f"Successfully processed {len(compiled_results)} total feature combinations across {total_files} files!")
                    
                    df_bulk = pd.DataFrame(compiled_results)
                    st.dataframe(df_bulk, use_container_width=True)
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_bulk.to_excel(writer, index=False, sheet_name='Batch Evaluation')
                        
                        # Apply auto-width for better UX natively via openpyxl
                        worksheet = writer.sheets['Batch Evaluation']
                        for col in worksheet.columns:
                            max_length = 0
                            column = col[0].column_letter
                            for cell in col:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(cell.value)
                                except:
                                    pass
                            adjusted_width = (max_length + 2)
                            worksheet.column_dimensions[column].width = adjusted_width

                    st.download_button(
                        label="📥 Download Excel Report",
                        data=buffer.getvalue(),
                        file_name=f"Batch_Evaluations_{pd.Timestamp.today().strftime('%Y_%m_%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No valid results were generated from the uploaded files.")

if ticker_input:
    with tab1:
        if not selected_features:
            st.warning("⚠️ Please select at least one feature in 'Advanced Model Settings' to train the model.")
        elif st.session_state.get('skip_render', False):
            # Show a helpful reminder that calculation hasn't run yet if skip_render is true
            st.info("💡 Selection updated. Dashboard will refresh automatically on your next re-train or search.")
        else:
            render_main_dashboard(ticker_input, exchange, selected_features)

# Reset skip_render flag for future interactions
st.session_state['skip_render'] = False

with tab2:
    st.markdown("### ⭐ Saved Watchlist")
    
    today_ist = pd.Timestamp.today(tz='Asia/Kolkata')
    
    def export_watchlist_to_excel(watchlist_dict):
        export_data = []
        color_map = {}
        for ticker, info in watchlist_dict.items():
            export_data.append({
                "Ticker": ticker,
                "Highest Prob (%)": round(info['prob'], 1),
                "Prediction": info.get('label', 'N/A'),
                "Long AMO (%)": round(info.get('prob_long', 0.0), 1),
                "Short AMO (%)": round(info.get('prob_short', 0.0), 1),
                "Hist Long (%)": round(info.get('hist_long', 0.0), 1),
                "Hist Short (%)": round(info.get('hist_short', 0.0), 1),
                "Model Accuracy (%)": round(info['acc'], 1),
                "Out-of-Sample Count": info.get('test_samples', 'N/A'),
                "Baseline Accuracy (%)": round(info.get('baseline', 0.0), 1),
                "Baseline Guess": info.get('baseline_label', 'N/A'),
                "True Edge (%)": round(info.get('edge', 0.0), 1),
                "Features Used": info.get('selected_features', 'N/A'),
                "Latest Result": info.get('latest_result', '').split('Recent Regime Sync: ')[1].split(' Correct')[0] if 'Recent Regime Sync:' in info.get('latest_result', '') else 'N/A'
            })
            color_map[ticker] = info.get('color', '#D99300')
            
        df_export = pd.DataFrame(export_data)
        
        def apply_excel_color(row):
            ticker = row['Ticker']
            color = color_map.get(ticker, '#D99300')
            return [f"background-color: {color}; color: white; font-weight: bold;" if col in ['Highest Prob (%)', 'Prediction'] else "" for col in row.index]

        styled_df = df_export.style.apply(apply_excel_color, axis=1)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            styled_df.to_excel(writer, index=False, sheet_name='Watchlist')
        return buffer.getvalue()
        
    prediction_ts = st.session_state.get('last_data_date', today_ist)
    forecast_type = st.session_state.get('forecast_type', "Next Day")
    
    if 'last_data_date' not in st.session_state:
        # Fallback if app just loaded without main dashboard run
        prediction_ts = prediction_ts + pd.Timedelta(days=1)
        while prediction_ts.weekday() >= 5:  # 5=Sat, 6=Sun
            prediction_ts += pd.Timedelta(days=1)

    global_next_day_str = prediction_ts.strftime('%A, %b %d, %Y')
    forecast_text = "current active AMO session" if forecast_type == "Current Day" else "next active AMO session"
    st.markdown(f"<p style='color: #666; font-style: italic; margin-top: -10px;'>All model predictions are forecasting for the {forecast_text}: <b>{global_next_day_str}</b></p>", unsafe_allow_html=True)
    
    if st.session_state['watchlist']:
        excel_data = export_watchlist_to_excel(st.session_state['watchlist'])
        st.download_button(
            label="📥",
            data=excel_data,
            file_name=f"Stock_Watchlist_{today_ist.strftime('%Y_%m_%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Export Watchlist to Excel"
        )
        
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    
    if not st.session_state['watchlist']:
        st.info("Your watchlist is currently empty. Search for a stock ticker in the Main Dashboard to add it here automatically!")
    else:
        for w_ticker, w_data in st.session_state['watchlist'].items():
            label = w_data.get('label', "")
            color = w_data.get('color', "#888")
            bg_color = w_data.get('bg_color', "rgba(255,255,255,0.05)")
            label_html = f" <span style='font-size: 1.2rem; font-weight: 400; color: {color};'>({label})</span>" if label else ""
            
            p_long = w_data.get('prob_long', 0.0)
            p_short = w_data.get('prob_short', 0.0)
            
            st.markdown(f"""
            <div style="padding: 1.2rem; border-radius: 10px; background-color: {bg_color}; border: 1px solid {color}; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
                    <div style="flex: 1;">
                        <h3 style="margin: 0; color: black; font-weight: 700; font-size: 1.6rem;">{w_ticker}</h3>
                    </div>
                    <div style="flex: 2; text-align: center;">
                        <p style="margin: 0; font-size: 0.9rem; color: #555; font-weight: 600; text-transform: uppercase;">Highest Probability</p>
                        <h2 style="margin: 5px 0 0 0; color: {color}; font-size: 2.2rem;">{w_data['prob']:.1f}%{label_html}</h2>
                    </div>
                    <div style="flex: 1; text-align: right;">
                        <p style="margin: 0; font-size: 0.85rem; color: #555; font-weight: 600; text-transform: uppercase;">AMO Model Accuracy</p>
                        <h3 style="margin: 5px 0 0 0; color: black; font-size: 1.4rem;">{w_data['acc']:.1f}% <span style="font-size: 0.9rem; color: #888; font-weight: 400;">(n={w_data.get('test_samples', '?')})</span></h3>
                        <div style="margin-top: 5px;">
                            <span style="font-size: 0.8rem; color: #777;">Base: {w_data.get('baseline', 0.0):.1f}% ({w_data.get('baseline_label', 'N/A')})</span> | 
                            <span style="font-size: 0.8rem; color: {'#00C073' if w_data.get('edge', 0.0) > 0 else '#FF2B2B'}; font-weight: bold;">Edge: {w_data.get('edge', 0.0):+.1f}%</span>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 5px; margin-bottom: 20px; text-align: center;">
                    <p style="margin: 0; font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600; margin-bottom: 10px;">5-Day Walk-Forward Validation</p>
                    <div style="text-align: left; padding: 8px; border-radius: 6px; border: 1px solid #DDD; display: inline-block;">{w_data.get('latest_result', 'N/A')}</div>
                </div>
                <div style="margin-bottom: 15px; padding: 8px; background: rgba(0,0,0,0.02); border-radius: 4px; border-left: 3px solid {color};">
                    <p style="margin: 0; font-size: 0.75rem; color: #666; line-height: 1.4;"><b>Features:</b> {w_data.get('selected_features', 'N/A')}</p>
                </div>
                <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 15px;">
                    <div style="flex: 1; text-align: center; border-right: 1px solid rgba(0,0,0,0.05);">
                        <p style="margin: 0; font-size: 0.75rem; color: #777; text-transform: uppercase;">Long AMO</p>
                        <h4 style="margin: 5px 0 0 0; color: #00C073; font-size: 1.1rem;">{p_long:.1f}%</h4>
                    </div>
                    <div style="flex: 1; text-align: center; border-right: 1px solid rgba(0,0,0,0.05);">
                        <p style="margin: 0; font-size: 0.75rem; color: #777; text-transform: uppercase;">Short AMO</p>
                        <h4 style="margin: 5px 0 0 0; color: #FF2B2B; font-size: 1.1rem;">{p_short:.1f}%</h4>
                    </div>
                    <div style="flex: 1; text-align: center; border-right: 1px solid rgba(0,0,0,0.05);">
                        <p style="margin: 0; font-size: 0.75rem; color: #777; text-transform: uppercase;">Hist Long</p>
                        <h4 style="margin: 5px 0 0 0; color: #00C073; font-size: 1.1rem;">{w_data.get('hist_long', 0.0):.1f}%</h4>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <p style="margin: 0; font-size: 0.75rem; color: #777; text-transform: uppercase;">Hist Short</p>
                        <h4 style="margin: 5px 0 0 0; color: #FF2B2B; font-size: 1.1rem;">{w_data.get('hist_short', 0.0):.1f}%</h4>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown("### 📖 Feature Glossary")
    st.markdown("A plain-language guide to every feature the model can use. Understanding these helps you interpret the prediction and tune feature selection.")
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)

    glossary_entries = [
        {
            "name": "Closing Momentum",
            "col": "Closing_Momentum",
            "icon": "📈",
            "what": "Measures how much the price moved within a single 1-hour candle, expressed as a percentage of the open price.",
            "formula": "(Close - Open) / Open",
            "positive": "The candle closed higher than it opened -- buyers dominated that hour.",
            "negative": "The candle closed lower than it opened -- sellers dominated that hour.",
            "high": "A large absolute value (e.g. +/-2%+) means a strong directional move happened in that hour -- high conviction from one side.",
            "low": "Near zero means the candle was indecisive -- open and close were almost the same (doji-like).",
        },
        {
            "name": "OBV Slope",
            "col": "OBV_Slope",
            "icon": "⚖️",
            "what": "Tracks the direction of On Balance Volume (OBV) over 14 periods. OBV adds volume on up-candles and subtracts it on down-candles. The slope tells you if smart money is accumulating or distributing, normalized by average volume so it's comparable across stocks.",
            "formula": "OBV.diff(14) / Volume.rolling(14).mean()",
            "positive": "OBV is rising -- net volume flow is into the stock (accumulation). Institutions are likely buying.",
            "negative": "OBV is falling -- net volume flow is out of the stock (distribution). Institutions are likely selling.",
            "high": "A large positive value (e.g. +3) means OBV surged upward by 3x the average volume over 14 periods -- very strong accumulation.",
            "low": "A large negative value (e.g. -3) means heavy distribution. Values near zero mean volume flow is balanced with no clear directional intent.",
        },
        {
            "name": "Anchored CR5",
            "col": "Anchored_Close_Return_5",
            "icon": "⚓",
            "what": "The exact percentage return of the current price relative to the closing price from 5 trading days ago. Unlike a moving average, this anchors to a single rigid historical level -- giving the model an un-smoothed structural reference point.",
            "formula": "(Close_t - Close_t-5) / Close_t-5",
            "positive": "Price has risen over the past 5 days -- the stock is in a short-term uptrend relative to its exact anchor.",
            "negative": "Price has fallen over the past 5 days -- the stock is in a short-term downtrend relative to its exact anchor.",
            "high": "A value like +5% means a strong 5-day rally. Algorithmic traders may view this as overextension from structure.",
            "low": "A value like -5% means a sharp 5-day decline. May signal oversold conditions relative to the structural anchor.",
        },
        {
            "name": "ATR %",
            "col": "ATR_Percent",
            "icon": "🌊",
            "what": "The 14-day Average True Range expressed as a percentage of the current price. Measures how volatile the stock is -- higher ATR% means bigger daily swings.",
            "formula": "ATR(14) / Close",
            "positive": "Always positive. This is a magnitude-only indicator.",
            "negative": "N/A -- this value is always positive.",
            "high": "High values (e.g. 4%+) mean the stock swings a lot day-to-day. More volatile = riskier but potentially more profitable for directional bets.",
            "low": "Low values (e.g. <1%) mean the stock barely moves. Calm, low-volatility regime -- smaller profit potential per trade.",
        },
        {
            "name": "Daily RSI",
            "col": "Daily_RSI_14",
            "icon": "🔥",
            "what": "The 14-day Relative Strength Index. Classic momentum oscillator that ranges from 0 to 100. Measures if a stock has been overbought or oversold recently.",
            "formula": "RSI(14) on daily closes",
            "positive": "N/A -- RSI is always between 0 and 100.",
            "negative": "N/A -- RSI is always between 0 and 100.",
            "high": "Above 70 -> Overbought. The stock has risen a lot recently and may be due for a pullback.",
            "low": "Below 30 -> Oversold. The stock has fallen a lot recently and may be due for a bounce. Between 40-60 is neutral territory.",
        },
        {
            "name": "VWAP Dist",
            "col": "VWAP_Distance",
            "icon": "⚡",
            "what": "How far the current price is from the intraday Volume Weighted Average Price (VWAP). VWAP is the \"fair price\" institutional traders benchmark against.",
            "formula": "(Close - VWAP) / VWAP",
            "positive": "Price is above VWAP -- buyers are in control and willing to pay more than the average traded price today.",
            "negative": "Price is below VWAP -- sellers are in control. Institutions may view the stock as expensive relative to today's volume.",
            "high": "A large positive distance means strong bullish intraday momentum. The stock is trading well above its volume-weighted average.",
            "low": "A large negative distance means strong bearish intraday pressure. Near zero means price is trading at fair value.",
        },
        {
            "name": "OFI (Order Flow)",
            "col": "OFI",
            "icon": "🔄",
            "what": "Order Flow Imbalance -- estimates whether buyers or sellers dominated each candle based on where the close falls within the high-low range, smoothed over 5 periods.",
            "formula": "((Close-Low) - (High-Close)) / (High-Low), rolled over 5 candles",
            "positive": "Close is consistently near the high of each candle -- buyers are absorbing all selling pressure (bullish flow).",
            "negative": "Close is consistently near the low -- sellers are overwhelming buyers (bearish flow).",
            "high": "Near +1.0 means almost every recent candle closed at its high -- extreme buying pressure.",
            "low": "Near -1.0 means every recent candle closed at its low -- extreme selling pressure. Near 0 means balanced, tug-of-war action.",
        },
        {
            "name": "Frac Diff (Memory)",
            "col": "Frac_Diff_Close",
            "icon": "🧠",
            "what": "Fractionally Differenced Close price (d=0.4). A technique from Lopez de Prado that makes the price series stationary for ML while preserving long-term memory. Unlike simple returns that forget history, this retains trend information.",
            "formula": "FFD(Close, d=0.4)",
            "positive": "The memory-preserving transformation is trending upward -- the stock has persistent bullish momentum with historical context.",
            "negative": "The transformation is trending downward -- persistent bearish momentum.",
            "high": "Large absolute values indicate the stock is in a strong, sustained trend (the memory effect amplifies persistent moves).",
            "low": "Small values near zero indicate the stock is range-bound or the trend has stalled.",
        },
        {
            "name": "Nifty Momentum",
            "col": "Nifty_Momentum",
            "icon": "🇮🇳",
            "what": "The daily open-to-close return of the NIFTY 50 index. Captures the broad market sentiment in India for that day.",
            "formula": "(NIFTY Close - NIFTY Open) / NIFTY Open",
            "positive": "The broad Indian market closed up that day -- bullish macro sentiment. Individual stocks tend to follow.",
            "negative": "The broad market closed down -- bearish macro sentiment. Headwind for individual long positions.",
            "high": "A value like +1.5% means the market had a strong rally day. Most stocks benefit from this tailwind.",
            "low": "A value like -1.5% means a broad sell-off. Even fundamentally strong stocks can get dragged down.",
        },
        {
            "name": "Nifty RSI",
            "col": "Nifty_RSI_14",
            "icon": "📊",
            "what": "The 14-day RSI of the NIFTY 50 index. Tells you if the overall Indian market is overbought or oversold.",
            "formula": "RSI(14) on NIFTY 50 daily closes",
            "positive": "N/A -- always between 0 and 100.",
            "negative": "N/A -- always between 0 and 100.",
            "high": "Above 70 -> The entire market has been rallying hard and may be due for a correction. Risk of mean-reversion is elevated.",
            "low": "Below 30 -> The entire market is deeply oversold -- historically a contrarian buy signal for the broader market.",
        },
        {
            "name": "Nifty Trend",
            "col": "Nifty_Trend_Dist",
            "icon": "📐",
            "what": "How far the NIFTY 50 is from its 20-day Exponential Moving Average. Measures whether the broad market is extended or at equilibrium.",
            "formula": "(NIFTY Close - EMA20) / EMA20",
            "positive": "NIFTY is trading above its 20-day trend -- the macro environment is bullish.",
            "negative": "NIFTY is below its 20-day trend -- the macro environment is bearish.",
            "high": "A value like +3% means the market is significantly stretched above trend -- potential for a pullback.",
            "low": "A value like -3% means the market is significantly below trend -- potential for a relief rally.",
        },
        {
            "name": "Morning Autocorr",
            "col": "Morning_Autocorr",
            "icon": "🌅",
            "what": "The return from market open (9:15 AM) to 10:15 AM, expressed as a percentage. Captures the early-morning directional bias. Autocorrelation in morning moves can predict the rest of the day.",
            "formula": "(Close at 10:15 - Open at 9:15) / Open at 9:15",
            "positive": "The stock rallied in the first hour -- morning buyers were aggressive. This momentum often carries forward.",
            "negative": "The stock sold off in the first hour -- morning sellers dominated. Negative autocorrelation may signal continued weakness.",
            "high": "A large positive value (e.g. +2%) means a strong opening rally -- high conviction from early participants.",
            "low": "A large negative value means a sharp morning sell-off. Near zero means a flat, indecisive open.",
        },
        {
            "name": "US Overnight Ret",
            "col": "US_Overnight_Return",
            "icon": "🇺🇸",
            "what": "The previous day's S&P 500 daily return. Since US markets close after Indian markets, this captures the overnight global sentiment that Indian markets will react to the next morning.",
            "formula": "S&P 500 daily return, shifted forward by 1 day",
            "positive": "The US market closed up overnight -- typically a bullish tailwind for Indian markets at the open.",
            "negative": "The US market closed down overnight -- typically a bearish headwind. Indian markets often gap down following US sell-offs.",
            "high": "A value like +2% means a strong US rally -- expect a positive gap-up in Indian markets.",
            "low": "A value like -2% means a sharp US sell-off -- expect a negative gap-down. Near zero means neutral overnight sentiment.",
        },
    ]

    for entry in glossary_entries:
        st.markdown(f"""
        <div style="padding: 1.2rem; border-radius: 10px; background-color: rgba(0,0,0,0.02); border: 1px solid #E0E0E0; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span style="font-size: 1.6rem;">{entry['icon']}</span>
                <div>
                    <h4 style="margin: 0; color: #1D4ED8; font-size: 1.15rem;">{entry['name']}</h4>
                    <code style="font-size: 0.8rem; color: #888; background: rgba(0,0,0,0.04); padding: 2px 6px; border-radius: 3px;">{entry['col']}</code>
                </div>
            </div>
            <p style="margin: 0 0 12px 0; color: #333; font-size: 0.95rem; line-height: 1.5;">{entry['what']}</p>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                <div style="flex: 1; min-width: 220px; padding: 8px 12px; border-radius: 6px; background-color: rgba(0, 192, 115, 0.08); border-left: 3px solid #00C073;">
                    <p style="margin: 0 0 2px 0; font-size: 0.75rem; font-weight: 700; color: #00C073; text-transform: uppercase;">Positive (+)</p>
                    <p style="margin: 0; font-size: 0.85rem; color: #333;">{entry['positive']}</p>
                </div>
                <div style="flex: 1; min-width: 220px; padding: 8px 12px; border-radius: 6px; background-color: rgba(255, 43, 43, 0.08); border-left: 3px solid #FF2B2B;">
                    <p style="margin: 0 0 2px 0; font-size: 0.75rem; font-weight: 700; color: #FF2B2B; text-transform: uppercase;">Negative (-)</p>
                    <p style="margin: 0; font-size: 0.85rem; color: #333;">{entry['negative']}</p>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;">
                <div style="flex: 1; min-width: 220px; padding: 8px 12px; border-radius: 6px; background-color: rgba(29, 78, 216, 0.06); border-left: 3px solid #1D4ED8;">
                    <p style="margin: 0 0 2px 0; font-size: 0.75rem; font-weight: 700; color: #1D4ED8; text-transform: uppercase;">High Value</p>
                    <p style="margin: 0; font-size: 0.85rem; color: #333;">{entry['high']}</p>
                </div>
                <div style="flex: 1; min-width: 220px; padding: 8px 12px; border-radius: 6px; background-color: rgba(128, 128, 128, 0.08); border-left: 3px solid #888;">
                    <p style="margin: 0 0 2px 0; font-size: 0.75rem; font-weight: 700; color: #666; text-transform: uppercase;">Low Value</p>
                    <p style="margin: 0; font-size: 0.85rem; color: #333;">{entry['low']}</p>
                </div>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 0.8rem; color: #999;"><b>Formula:</b> <code style="background: rgba(0,0,0,0.04); padding: 1px 5px; border-radius: 3px;">{entry['formula']}</code></p>
        </div>
        """, unsafe_allow_html=True)