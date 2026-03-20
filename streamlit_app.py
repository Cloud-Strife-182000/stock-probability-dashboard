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
                
                # MACD (12, 26, 9)
                ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema_12 - ema_26
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # We skip the table as requested and go straight to visually attractive indicators
            st.markdown("---")
            # Dynamic indicator rendering function for big colored square blocks
            def render_indicator(col, title, value, diff, is_good, state_key=None):
                bg_color = "rgba(0, 192, 115, 0.15)" if is_good is True else ("rgba(255, 43, 43, 0.15)" if is_good is False else "rgba(128, 128, 128, 0.1)")
                border_color = "#00C073" if is_good is True else ("#FF2B2B" if is_good is False else "gray")
                text_color = border_color if is_good is not None else "black"
                
                is_active = st.session_state.get(state_key, True) if state_key else True
                opacity = "1.0" if is_active else "0.3"
                
                html = f"""
                <div style="background-color: {bg_color}; border: 3px solid {border_color}; border-radius: 15px; padding: 15px 5px; text-align: center; height: 130px; margin-bottom: 5px; opacity: {opacity}; transition: opacity 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <p style="margin:0; font-size:14px; font-weight:bold; color:#555;">{title}</p>
                    <h2 style="margin:10px 0; font-size:26px; color:{text_color}; padding: 0px 5px;">{value}</h2>
                    <p style="margin:0; font-size:13px; color:{text_color}; font-weight:bold;">{diff}</p>
                </div>
                """
                col.markdown(html, unsafe_allow_html=True)
                
                if state_key:
                    if state_key not in st.session_state:
                        st.session_state[state_key] = True
                    # A sleek toggle switch natively underneath the colored card square
                    st.session_state[state_key] = col.toggle("Include", value=st.session_state[state_key], key=f"tgl_{state_key}")
                    return st.session_state[state_key]
                else:
                    col.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # visual alignment spacer
                    return True

            st.write("")
            
            # Get latest day metrics
            if len(df) > 0:
                latest_day = df.iloc[-1]
                prev_day = df.iloc[-2] if len(df) > 1 else latest_day
                
                cols = st.columns(5)
                
                # Current Price
                close_val = f"₹{latest_day['Close']:.2f}" if 'Close' in df.columns and pd.notna(latest_day['Close']) else "N/A"
                change_price = latest_day['Change'] if 'Change' in df.columns else 0
                change_pct = f"{latest_day['Change %']:.2f}%" if 'Change %' in df.columns and pd.notna(latest_day['Change %']) else ""
                price_is_good = True if change_price > 0 else (False if change_price < 0 else None)
                render_indicator(cols[0], "Current Price", close_val, change_pct, price_is_good, None)
                
                # RSI 14
                rsi_v = latest_day['RSI_14'] if 'RSI_14' in df.columns else None
                rsi_str = f"{rsi_v:.2f}" if pd.notna(rsi_v) else "N/A"
                rsi_diff_v = (rsi_v - prev_day['RSI_14']) if pd.notna(rsi_v) and pd.notna(prev_day['RSI_14']) else 0
                rsi_diff_str = f"{rsi_diff_v:+.2f}"
                rsi_is_good = True if pd.notna(rsi_v) and rsi_v < 40 else (False if pd.notna(rsi_v) and rsi_v > 60 else None)
                use_rsi = render_indicator(cols[1], "RSI (14)", rsi_str, rsi_diff_str, rsi_is_good, "use_rsi")
                
                # SMA 20
                sma20_v = latest_day['SMA_20'] if 'SMA_20' in df.columns else None
                sma50_v = latest_day['SMA_50'] if 'SMA_50' in df.columns else None
                sma20_str = f"₹{sma20_v:.2f}" if pd.notna(sma20_v) else "N/A"
                sma20_diff_v = (sma20_v - prev_day['SMA_20']) if pd.notna(sma20_v) and pd.notna(prev_day['SMA_20']) else 0
                sma20_is_good = True if pd.notna(sma20_v) and pd.notna(sma50_v) and sma20_v > sma50_v else (False if pd.notna(sma20_v) and pd.notna(sma50_v) and sma20_v < sma50_v else None)
                use_sma20 = render_indicator(cols[2], "SMA (20)", sma20_str, f"{sma20_diff_v:+.2f}", sma20_is_good, "use_sma20")
                
                # SMA 50
                sma50_str = f"₹{sma50_v:.2f}" if pd.notna(sma50_v) else "N/A"
                sma50_diff_v = (sma50_v - prev_day['SMA_50']) if pd.notna(sma50_v) and pd.notna(prev_day['SMA_50']) else 0
                sma50_is_good = True if 'Close' in df.columns and pd.notna(sma50_v) and latest_day['Close'] > sma50_v else (False if 'Close' in df.columns and pd.notna(sma50_v) and latest_day['Close'] < sma50_v else None)
                use_sma50 = render_indicator(cols[3], "SMA (50)", sma50_str, f"{sma50_diff_v:+.2f}", sma50_is_good, "use_sma50")
                
                # MACD
                macd_v = latest_day['MACD_Hist'] if 'MACD_Hist' in df.columns else None
                macd_str = f"{macd_v:.2f}" if pd.notna(macd_v) else "N/A"
                macd_diff_v = (macd_v - prev_day['MACD_Hist']) if pd.notna(macd_v) and pd.notna(prev_day['MACD_Hist']) else 0
                macd_is_good = True if pd.notna(macd_v) and macd_v > 0 else (False if pd.notna(macd_v) and macd_v < 0 else None)
                use_macd = render_indicator(cols[4], "MACD Hist", macd_str, f"{macd_diff_v:+.2f}", macd_is_good, "use_macd")

                # Dynamic Signal Generation entirely based on interactive tile toggles
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                signals = []
                reasons = []
                
                if not (use_rsi or use_sma20 or use_sma50 or use_macd):
                    signal = "NONE"
                    signal_color = "#AAAAAA"
                    final_reason = "Please selectively enable at least one indicator."
                else:
                    if use_sma20 and pd.notna(sma20_v) and pd.notna(sma50_v):
                        if sma20_v > sma50_v:
                            signals.append(1)
                            reasons.append("SMA20 > SMA50")
                        else:
                            signals.append(-1)
                            reasons.append("SMA20 < SMA50")
                            
                    if use_sma50 and pd.notna(sma50_v) and 'Close' in df.columns:
                        if latest_day['Close'] > sma50_v:
                            signals.append(1)
                            reasons.append("Price > SMA50")
                        else:
                            signals.append(-1)
                            reasons.append("Price < SMA50")
                            
                    if use_rsi and pd.notna(rsi_v):
                        if rsi_v < 40:
                            signals.append(1)
                            reasons.append("RSI Bullish")
                        elif rsi_v > 60:
                            signals.append(-1)
                            reasons.append("RSI Bearish")
                        else:
                            signals.append(0)
                            
                    if use_macd and pd.notna(macd_v):
                        if macd_v > 0:
                            signals.append(1)
                            reasons.append("MACD Bullish")
                        else:
                            signals.append(-1)
                            reasons.append("MACD Bearish")
                            
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
                            signal_color = "#FFB92B" # Yellow
                        final_reason = " | ".join(reasons) if reasons else "Mixed Signals"
                    else:
                        signal = "HOLD"
                        signal_color = "#AAAAAA"
                        final_reason = "Insufficient data to compute signal."
                            
                # Display beautifully centered Signal Card
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 2rem; border-radius: 10px; background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); margin-top: 1rem;">
                        <h4 style="margin-bottom: 0px; margin-top: 0px; color: #888; font-weight: normal;">Automated Trading Signal</h4>
                        <h1 style="color: {signal_color}; font-size: 3.5rem; margin: 10px 0px;">{signal}</h1>
                        <p style="color: gray; font-size: 1.1rem; margin-top: 0px;"><em>{final_reason}</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
