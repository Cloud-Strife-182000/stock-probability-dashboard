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
    
    with st.spinner(f"Fetching last 1 month of daily data for {symbol}..."):
        try:
            # Fetch data using yfinance
            data = yf.download(symbol, period="1mo", interval="1d", progress=False)
            
            if data.empty:
                # Yahoo Finance uses .BO for BSE often, fallback if .BS returns empty
                if exchange == "BSE":
                    symbol_bo = f"{ticker_input}.BO"
                    data = yf.download(symbol_bo, period="1mo", interval="1d", progress=False)
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
            display_cols = [col for col in display_cols if col in df.columns]
            
            # Sort descending by date to show latest date first
            if 'Date' in df.columns:
                df = df.sort_values(by='Date', ascending=False)
            
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
            styler_format = {k: v for k, v in styler_format.items() if k in df.columns}
            
            styler = df[display_cols].style
            subset_cols = [col for col in ['Change', 'Change %'] if col in df.columns]
            
            # Python versions/pandas mapping handling
            if hasattr(styler, 'map'):
                styled_df = styler.map(color_financials, subset=subset_cols).format(styler_format)
            else:
                styled_df = styler.applymap(color_financials, subset=subset_cols).format(styler_format)
            
            st.subheader(f"Price Data for {symbol}")
            
            # Render in Streamlit
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
