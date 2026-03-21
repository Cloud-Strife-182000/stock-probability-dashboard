import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import urllib.request
import xml.etree.ElementTree as ET
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Stock Probability Dashboard", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, exch):
    suffix = ".NS" if exch == "NSE" else ".BS"
    sym = f"{ticker}{suffix}"
    data = yf.download(sym, period="6mo", interval="1d", progress=False)
    if data.empty and exch == "BSE":
        sym_bo = f"{ticker}.BO"
        data = yf.download(sym_bo, period="6mo", interval="1d", progress=False)
        if not data.empty:
            sym = sym_bo
    return data, sym

@st.cache_data(ttl=1800, show_spinner=False)
def get_top_news(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}+share+news&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        # Construct raw request to mask simple Python user-agent bindings
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            # Map up to 3 individual news items
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

st.title("Stock Probability Dashboard")

# Search Bar and Exchange Dropdown
col1, col2 = st.columns([3, 1])

with col1:
    ticker_input = st.text_input("Enter Indian Ticker (e.g., PNB, BANKINDIA):", "").strip().upper()

with col2:
    exchange = st.selectbox("Select Exchange:", ["NSE", "BSE"])

if ticker_input:
    with st.spinner(f"Fetching data and calculating indicators for {ticker_input}..."):
        try:
            data, symbol = fetch_stock_data(ticker_input, exchange)
            
            if data.empty:
                st.warning(f"No data found for {ticker_input}. Please check the ticker symbol.")
                st.stop()
            
            # Handle MultiIndex columns if present (newer yfinance versions)
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns (e.g., ('Close', 'TCS.NS') -> 'Close')
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
            df = df.reset_index()
            
            # Ensure Date column exists and formatting
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
            elif 'Datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['Datetime']).dt.date
            
            # Calculate daily change for formatting
            if 'Close' in df.columns:
                df['Change'] = df['Close'].diff()
                df['Change %'] = (df['Change'] / df['Close'].shift(1)) * 100
                
                # Calculate Indicators
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                
                # RSI 14
                delta = df['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -1 * delta.clip(upper=0)
                ema_gain = gain.ewm(com=13, adjust=False).mean()
                ema_loss = loss.ewm(com=13, adjust=False).mean()
                rs = ema_gain / ema_loss
                df['RSI_14'] = 100 - (100 / (1 + rs))
                
                # ATR 14
                df['ATR_14'] = df.ta.atr(length=14)
                
                # Volume Surge & Gap Percent
                df['Volume_Surge'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
                df['Gap_Percent'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100
                
                # --- Machine Learning Target Preparation ---
                # A bullish signal is only generated if the next day's close is at least 0.2% higher (hurdle for slippage/fees)
                df['Target'] = (df['Close'].shift(-1) >= (df['Close'] * 1.002)).astype(float)
                if len(df) > 0:
                    df.iloc[-1, df.columns.get_loc('Target')] = float('nan')
            
            # We skip the table as requested and go straight to visually attractive indicators
            st.markdown("---")
            
            st.markdown(f"<h2 style='text-align: left; color: black;'>Stock: <b style='color: #1D4ED8;'>{symbol}</b></h2>", unsafe_allow_html=True)

            # Global CSS to perfectly align all toggle switches in the center
            st.markdown("""
                <style>
                div[data-testid="stWidgetLabel"] {
                    display: flex;
                    justify-content: center;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Dynamic indicator rendering function for big colored square blocks
            def render_indicator(col, title, value, diff, is_good):
                bg_color = "rgba(0, 192, 115, 0.15)" if is_good is True else ("rgba(255, 43, 43, 0.15)" if is_good is False else "rgba(128, 128, 128, 0.1)")
                border_color = "#00C073" if is_good is True else ("#FF2B2B" if is_good is False else "gray")
                text_color = border_color if is_good is not None else "black"
                
                html = f"""
                <div style="background-color: {bg_color}; border: 3px solid {border_color}; border-radius: 15px; padding: 15px 5px; text-align: center; height: 130px; margin-bottom: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <p style="margin:0; font-size:14px; font-weight:bold; color:#555;">{title}</p>
                    <h2 style="margin:10px 0; font-size:26px; color:{text_color}; padding: 0px 5px;">{value}</h2>
                    <p style="margin:0; font-size:13px; color:{text_color}; font-weight:bold;">{diff}</p>
                </div>
                """
                col.markdown(html, unsafe_allow_html=True)

            st.write("")
            
            # Get latest day metrics
            if len(df) > 0:
                latest_day = df.iloc[-1]
                prev_day = df.iloc[-2] if len(df) > 1 else latest_day
                
                cols = st.columns(7)
                
                # Current Price
                close_val = f"₹{latest_day['Close']:.2f}" if 'Close' in df.columns and pd.notna(latest_day['Close']) else "N/A"
                change_price = latest_day['Change'] if 'Change' in df.columns else 0
                change_pct = f"{latest_day['Change %']:.2f}%" if 'Change %' in df.columns and pd.notna(latest_day['Change %']) else ""
                price_is_good = True if change_price > 0 else (False if change_price < 0 else None)
                render_indicator(cols[0], "Current", close_val, change_pct, price_is_good)
                
                # SMA 20
                sma20_v = latest_day['SMA_20'] if 'SMA_20' in df.columns else None
                sma50_v = latest_day['SMA_50'] if 'SMA_50' in df.columns else None
                sma20_str = f"₹{sma20_v:.2f}" if pd.notna(sma20_v) else "N/A"
                sma20_diff_v = (sma20_v - prev_day['SMA_20']) if pd.notna(sma20_v) and pd.notna(prev_day['SMA_20']) else 0
                sma20_is_good = True if pd.notna(sma20_v) and pd.notna(sma50_v) and sma20_v > sma50_v else (False if pd.notna(sma20_v) and pd.notna(sma50_v) and sma20_v < sma50_v else None)
                render_indicator(cols[1], "SMA (20)", sma20_str, f"{sma20_diff_v:+.2f}", sma20_is_good)
                
                # SMA 50
                sma50_str = f"₹{sma50_v:.2f}" if pd.notna(sma50_v) else "N/A"
                sma50_diff_v = (sma50_v - prev_day['SMA_50']) if pd.notna(sma50_v) and pd.notna(prev_day['SMA_50']) else 0
                sma50_is_good = True if 'Close' in df.columns and pd.notna(sma50_v) and latest_day['Close'] > sma50_v else (False if 'Close' in df.columns and pd.notna(sma50_v) and latest_day['Close'] < sma50_v else None)
                render_indicator(cols[2], "SMA (50)", sma50_str, f"{sma50_diff_v:+.2f}", sma50_is_good)
                
                # RSI 14
                rsi_v = latest_day['RSI_14'] if 'RSI_14' in df.columns else None
                rsi_str = f"{rsi_v:.2f}" if pd.notna(rsi_v) else "N/A"
                rsi_diff_v = (rsi_v - prev_day['RSI_14']) if pd.notna(rsi_v) and pd.notna(prev_day['RSI_14']) else 0
                rsi_diff_str = f"{rsi_diff_v:+.2f}"
                rsi_is_good = True if pd.notna(rsi_v) and rsi_v < 40 else (False if pd.notna(rsi_v) and rsi_v > 60 else None)
                render_indicator(cols[3], "RSI (14)", rsi_str, rsi_diff_str, rsi_is_good)
                
                # ATR 14
                atr_v = latest_day['ATR_14'] if 'ATR_14' in df.columns else None
                atr_str = f"₹{atr_v:.2f}" if pd.notna(atr_v) else "N/A"
                atr_diff_v = (atr_v - prev_day['ATR_14']) if pd.notna(atr_v) and pd.notna(prev_day['ATR_14']) else 0
                atr_is_good = None # Volatility is neutral colored by default
                render_indicator(cols[4], "ATR (14)", atr_str, f"{atr_diff_v:+.2f}", atr_is_good)
                
                # Volume Surge
                vol_v = latest_day['Volume_Surge'] if 'Volume_Surge' in df.columns else None
                vol_str = f"{vol_v:.2f}x" if pd.notna(vol_v) else "N/A"
                vol_diff_v = (vol_v - prev_day['Volume_Surge']) if pd.notna(vol_v) and pd.notna(prev_day['Volume_Surge']) else 0
                vol_is_good = True if pd.notna(vol_v) and vol_v > 1.2 else (False if pd.notna(vol_v) and vol_v < 0.8 else None)
                render_indicator(cols[5], "Vol Surge", vol_str, f"{vol_diff_v:+.2f}x", vol_is_good)
                
                # Gap Percent
                gap_v = latest_day['Gap_Percent'] if 'Gap_Percent' in df.columns else None
                gap_str = f"{gap_v:.2f}%" if pd.notna(gap_v) else "N/A"
                gap_diff_v = (gap_v - prev_day['Gap_Percent']) if pd.notna(gap_v) and pd.notna(prev_day['Gap_Percent']) else 0
                gap_is_good = True if pd.notna(gap_v) and gap_v > 0 else (False if pd.notna(gap_v) and gap_v < 0 else None)
                render_indicator(cols[6], "Gap %", gap_str, f"{gap_diff_v:+.2f}%", gap_is_good)
                
                bullish_prob = None
                ml_details = None
                ml_pred_label = ""
                ml_df = df[['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'Volume_Surge', 'Gap_Percent', 'Target']].dropna()
                if len(ml_df) > 20:
                    X = ml_df[['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'Volume_Surge', 'Gap_Percent']]
                    y = ml_df['Target']
                    
                    try:
                        # Calculate out-of-sample accuracy using a time-series strict train/test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                        eval_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
                        eval_model.fit(X_train, y_train)
                        test_accuracy = eval_model.score(X_test, y_test)
                        
                        # Train production model on ALL data for tomorrow's prediction
                        model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
                        model.fit(X, y)
                        
                        today_features = df.iloc[-1][['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'Volume_Surge', 'Gap_Percent']].to_frame().T
                        if not today_features.isna().any().any():
                            prob = model.predict_proba(today_features)[0]
                            pred_class = model.predict(today_features)[0]
                            ml_pred_label = "BULLISH" if pred_class == 1.0 else "BEARISH"
                            
                            if len(model.classes_) == 2:
                                bullish_prob = prob[1]
                            else:
                                bullish_prob = 1.0 if model.classes_[0] == 1 else 0.0
                                
                            ml_details = {
                                "accuracy": test_accuracy,
                                "samples": len(ml_df),
                                "importances": {
                                    "SMA 20": model.feature_importances_[0],
                                    "SMA 50": model.feature_importances_[1],
                                    "RSI (14)": model.feature_importances_[2],
                                    "ATR (14)": model.feature_importances_[3],
                                    "Vol Surge": model.feature_importances_[4],
                                    "Gap %": model.feature_importances_[5]
                                }
                            }
                    except Exception as e:
                        print("ML Training error:", e)
                
                signals = []
                reasons = []
                included_names = []
                
                if pd.notna(sma20_v) and pd.notna(sma50_v):
                    included_names.append("SMA 20")
                    if sma20_v > sma50_v:
                        signals.append(1)
                        reasons.append("SMA20 > SMA50")
                    else:
                        signals.append(-1)
                        reasons.append("SMA20 < SMA50")
                        
                if pd.notna(sma50_v) and 'Close' in df.columns:
                    included_names.append("SMA 50")
                    if latest_day['Close'] > sma50_v:
                        signals.append(1)
                        reasons.append("Price > SMA50")
                    else:
                        signals.append(-1)
                        reasons.append("Price < SMA50")
                        
                if pd.notna(rsi_v):
                    included_names.append("RSI")
                    if rsi_v < 40:
                        signals.append(1)
                        reasons.append("RSI Bullish")
                    elif rsi_v > 60:
                        signals.append(-1)
                        reasons.append("RSI Bearish")
                    else:
                        signals.append(0)
                        
                if pd.notna(vol_v) and vol_v > 1.2:
                    included_names.append("Vol Surge")
                    signals.append(1)
                    reasons.append("High Conviction Vol")
                    
                if pd.notna(gap_v):
                    included_names.append("Gap")
                    if gap_v > 0:
                        signals.append(1)
                        reasons.append("Gap Up")
                    elif gap_v < 0:
                        signals.append(-1)
                        reasons.append("Gap Down")
                        
                if len(signals) > 0:
                    total_score = sum(signals)
                    if total_score > 0:
                        signal = "BUY"
                        signal_color = "#00C073" # Green
                    elif total_score < 0:
                        signal = "SELL"
                        signal_color = "#FF2B2B" # Red
                    else:
                        signal = "HOLD"
                        signal_color = "#D99300" # Darker Yellow
                    final_reason = f"<strong>Indicators Used:</strong> {', '.join(included_names)}<br><strong>Reasoning:</strong> {' | '.join(reasons)}" if reasons else "Mixed Signals"
                else:
                    signal = "HOLD"
                    signal_color = "#AAAAAA"
                    final_reason = "Insufficient data to compute signal."
                            
                # Display beautifully centered Signal Card
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 2rem; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); margin-top: 1rem;">
                        <h4 style="margin-bottom: 0px; margin-top: 0px; color: black; font-weight: 600;">Medium-Term Swing Signal</h4>
                        <h1 style="color: {signal_color}; font-size: 3.5rem; margin: 10px 0px;">{signal}</h1>
                        <p style="color: gray; font-size: 1.1rem; margin-top: 0px;">{final_reason}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                            
                # Display ML Prediction Card
                if bullish_prob is not None:
                    prob_pct = bullish_prob * 100
                    if prob_pct >= 55:
                        ml_color = "#00C073" # Green
                        ml_bg_color = "rgba(0, 192, 115, 0.05)"
                    elif prob_pct <= 45:
                        ml_color = "#FF2B2B" # Red
                        ml_bg_color = "rgba(255, 43, 43, 0.05)"
                    else:
                        ml_color = "#D99300" # Darker Yellow
                        ml_bg_color = "rgba(217, 147, 0, 0.05)"
                    
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 1.5rem; border-radius: 10px; background-color: {ml_bg_color}; border: 1px solid {ml_color}; margin-top: 2rem; margin-bottom: 0.5rem;">
                            <h4 style="margin-bottom: 0px; margin-top: 0px; color: black; font-weight: 600;">Next-Day Bullish Probability</h4>
                            <h1 style="color: {ml_color}; font-size: 3rem; margin: 5px 0px;">{prob_pct:.1f}% <span style="font-size: 1.5rem; font-weight: 400;">({ml_pred_label})</span></h1>
                            <p style="color: {ml_color}; font-size: 1rem; margin-top: 0px;"><em>Powered by scikit-learn RandomForestClassifier</em></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if ml_details:
                        # Inject CSS to make the specific Expander title big and black
                        st.markdown("""
                            <style>
                            div[data-testid="stExpander"] details summary p {
                                font-size: 22px !important;
                                color: black !important;
                                font-weight: 600 !important;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("View Machine Learning Model Details", expanded=False):
                            st.markdown(f"**Training Set Size:** {ml_details['samples']} market days")
                            st.markdown(f"**Predictive Accuracy:** {ml_details['accuracy']*100:.1f}% (Recent 20% Out-of-Sample Test)")
                            
                            st.markdown("**Feature Importances:**")
                            # Create a clean dictionary to dataframe mapping natively supported by Streamlit bar_chart
                            fi_df = pd.DataFrame(
                                list(ml_details['importances'].values()),
                                index=list(ml_details['importances'].keys()),
                                columns=["Relative Importance"]
                            )
                            st.bar_chart(fi_df, height=200)
                            
                            st.markdown("**Feature Correlation Matrix:**")
                            # Explicitly output historical dataset correlation coefficients dynamically mapped with sequential heat gradients
                            styled_corr = ml_df.corr().style.background_gradient(cmap="Oranges").format("{:.2f}")
                            st.dataframe(styled_corr, use_container_width=True)
                            
                        with st.expander("View Raw Machine Learning Training Data", expanded=False):
                            st.markdown("This historical data matrix was aggressively fed into the `scikit-learn` algorithm to train its prediction trees:")
                            display_df = ml_df.copy()
                            if 'Date' in df.columns:
                                display_df['Date'] = df.loc[display_df.index, 'Date']
                                display_df = display_df.set_index('Date')
                            st.dataframe(display_df, use_container_width=True)
                            
                # --- Top News Section ---
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
                
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
