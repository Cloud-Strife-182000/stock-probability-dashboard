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
            
            # Extract last ~1 month (22 trading days) for the table display to keep it clean
            table_df = df.tail(22).copy()

            def color_financials(val):
                if pd.isna(val):
                    return ''
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: #00C073;' # Streamlit green
                    elif val < 0:
                        return 'color: #FF2B2B;' # Streamlit red
                return ''

            # Columns to display and format
            display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Change %', 'Volume']
            # Only keep columns that actually exist in the dataframe
            display_cols = [col for col in display_cols if col in table_df.columns]
            
            # Sort descending by date to show latest date first
            if 'Date' in table_df.columns:
                table_df = table_df.sort_values(by='Date', ascending=False)
            
            # Format the columns nicely
            styler_format = {
                'Open': '{:.2f}',
                'High': '{:.2f}',
                'Low': '{:.2f}',
                'Close': '{:.2f}',
                'Change': '{:+.2f}',
                'Change %': '{:+.2f}%',
                'Volume': '{:,.0f}'
            }
            # Only format existing cols
            styler_format = {k: v for k, v in styler_format.items() if k in table_df.columns}
            
            styler = table_df[display_cols].style
            subset_cols = [col for col in ['Change', 'Change %'] if col in table_df.columns]
            
            # Python versions/pandas mapping handling
            if hasattr(styler, 'map'):
                styled_df = styler.map(color_financials, subset=subset_cols).format(styler_format)
            else:
                styled_df = styler.applymap(color_financials, subset=subset_cols).format(styler_format)
            
            st.subheader(f"Price Data for {symbol} (Last 1 Month)")
            
            # Render in Streamlit
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Indicators below the table
            st.markdown("---")
            st.subheader("Technical Indicators (Last Traded Day)")
            
            # Get latest day metrics
            if len(df) > 0:
                latest_day = df.iloc[-1]
                prev_day = df.iloc[-2] if len(df) > 1 else latest_day
                
                c1, c2, c3, c4 = st.columns(4)
                
                # Check for columns existing before rendering
                close_val = f"₹{latest_day['Close']:.2f}" if 'Close' in df.columns and pd.notna(latest_day['Close']) else "N/A"
                change_val = f"{latest_day['Change %']:.2f}%" if 'Change %' in df.columns and pd.notna(latest_day['Change %']) else None
                c1.metric("Close Price", close_val, change_val)
                
                rsi_val = f"{latest_day['RSI_14']:.2f}" if 'RSI_14' in df.columns and pd.notna(latest_day['RSI_14']) else "N/A"
                rsi_diff = f"{(latest_day['RSI_14'] - prev_day['RSI_14']):.2f}" if 'RSI_14' in df.columns and pd.notna(latest_day['RSI_14']) and pd.notna(prev_day['RSI_14']) else None
                c2.metric("RSI (14)", rsi_val, rsi_diff)
                
                sma20_val = f"₹{latest_day['SMA_20']:.2f}" if 'SMA_20' in df.columns and pd.notna(latest_day['SMA_20']) else "N/A"
                sma20_diff = f"{(latest_day['SMA_20'] - prev_day['SMA_20']):.2f}" if 'SMA_20' in df.columns and pd.notna(latest_day['SMA_20']) and pd.notna(prev_day['SMA_20']) else None
                c3.metric("SMA (20)", sma20_val, sma20_diff)
                
                sma50_val = f"₹{latest_day['SMA_50']:.2f}" if 'SMA_50' in df.columns and pd.notna(latest_day['SMA_50']) else "N/A"
                sma50_diff = f"{(latest_day['SMA_50'] - prev_day['SMA_50']):.2f}" if 'SMA_50' in df.columns and pd.notna(latest_day['SMA_50']) and pd.notna(prev_day['SMA_50']) else None
                c4.metric("SMA (50)", sma50_val, sma50_diff)
                
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
