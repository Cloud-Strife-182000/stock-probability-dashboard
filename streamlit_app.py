import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Stock Market Data", layout="wide")

st.title("Indian Stock Market Price Viewer")

# Search Bar and Exchange Dropdown
col1, col2 = st.columns([3, 1])

with col1:
    ticker_input = st.text_input("Enter Indian Ticker (e.g., TCS, ZOMATO):", "").strip().upper()

with col2:
    exchange = st.selectbox("Select Exchange:", ["NSE", "BSE"])

if ticker_input:
    # Append suffix based on selected exchange
    suffix = ".NS" if exchange == "NSE" else ".BS"
    symbol = f"{ticker_input}{suffix}"
    
    with st.spinner(f"Fetching data and calculating indicators for {symbol}..."):
        try:
            # Fetch 6 months of data to calculate indicators accurately (like 50-day SMA)
            data = yf.download(symbol, period="6mo", interval="1d", progress=False)
            
            if data.empty:
                # Yahoo Finance uses .BO for BSE often, fallback if .BS returns empty
                if exchange == "BSE":
                    symbol_bo = f"{ticker_input}.BO"
                    data = yf.download(symbol_bo, period="6mo", interval="1d", progress=False)
                    if data.empty:
                        st.warning(f"No data found for {symbol} or {symbol_bo}. Please check the ticker symbol.")
                        st.stop()
                    else:
                        symbol = symbol_bo # Use .BO since .BS was empty
                else:
                    st.warning(f"No data found for {symbol}. Please check the ticker symbol.")
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
            
            # We skip the table as requested and go straight to visually attractive indicators
            st.markdown("---")
            st.markdown(f"<h3 style='text-align: center; color: #4F8BF9;'>Technical Indicators for {symbol}</h3>", unsafe_allow_html=True)
            st.write("")
            
            # Get latest day metrics
            if len(df) > 0:
                latest_day = df.iloc[-1]
                prev_day = df.iloc[-2] if len(df) > 1 else latest_day
                
                c1, c2, c3, c4 = st.columns(4)
                
                # Check for columns existing before rendering
                close_val = f"₹{latest_day['Close']:.2f}" if 'Close' in df.columns and pd.notna(latest_day['Close']) else "N/A"
                change_val = f"{latest_day['Change %']:.2f}%" if 'Change %' in df.columns and pd.notna(latest_day['Change %']) else None
                c1.metric("Current Price", close_val, change_val)
                
                rsi_val = f"{latest_day['RSI_14']:.2f}" if 'RSI_14' in df.columns and pd.notna(latest_day['RSI_14']) else "N/A"
                rsi_diff = f"{(latest_day['RSI_14'] - prev_day['RSI_14']):.2f}" if 'RSI_14' in df.columns and pd.notna(latest_day['RSI_14']) and pd.notna(prev_day['RSI_14']) else None
                c2.metric("RSI (14)", rsi_val, rsi_diff)
                
                sma20_val = f"₹{latest_day['SMA_20']:.2f}" if 'SMA_20' in df.columns and pd.notna(latest_day['SMA_20']) else "N/A"
                sma20_diff = f"{(latest_day['SMA_20'] - prev_day['SMA_20']):.2f}" if 'SMA_20' in df.columns and pd.notna(latest_day['SMA_20']) and pd.notna(prev_day['SMA_20']) else None
                c3.metric("SMA (20)", sma20_val, sma20_diff)
                
                sma50_val = f"₹{latest_day['SMA_50']:.2f}" if 'SMA_50' in df.columns and pd.notna(latest_day['SMA_50']) else "N/A"
                sma50_diff = f"{(latest_day['SMA_50'] - prev_day['SMA_50']):.2f}" if 'SMA_50' in df.columns and pd.notna(latest_day['SMA_50']) and pd.notna(prev_day['SMA_50']) else None
                c4.metric("SMA (50)", sma50_val, sma50_diff)

                # Signal Generation Logic
                st.markdown("<br><br>", unsafe_allow_html=True)
                signal = "HOLD"
                signal_color = "#AAAAAA" # Gray
                reason = "Not enough data or mixed signals."
                
                if 'SMA_20' in df.columns and 'SMA_50' in df.columns and 'RSI_14' in df.columns:
                    sma20 = latest_day['SMA_20']
                    sma50 = latest_day['SMA_50']
                    rsi = latest_day['RSI_14']
                    
                    if pd.notna(sma20) and pd.notna(sma50) and pd.notna(rsi):
                        # Trading strategy logic
                        if sma20 > sma50 and rsi < 70:
                            signal = "BUY"
                            signal_color = "#00C073" # Green
                            reason = "SMA20 is above SMA50 (Uptrend) and RSI is not overbought."
                        elif sma20 < sma50 or rsi >= 70:
                            signal = "SELL"
                            signal_color = "#FF2B2B" # Red
                            if rsi >= 70:
                                reason = "Asset is Overbought (RSI >= 70)."
                            else:
                                reason = "SMA20 is below SMA50 (Downtrend)."
                        else:
                            signal = "HOLD"
                            signal_color = "#FFB92B" # Yellow
                            reason = "Awaiting clearer trends."
                            
                # Display beautifully centered Signal Card
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 2rem; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); margin-top: 1rem;">
                        <h4 style="margin-bottom: 0px; margin-top: 0px; color: #888; font-weight: normal;">Automated Trading Signal</h4>
                        <h1 style="color: {signal_color}; font-size: 3.5rem; margin: 10px 0px;">{signal}</h1>
                        <p style="color: gray; font-size: 1.1rem; margin-top: 0px;"><em>{reason}</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
