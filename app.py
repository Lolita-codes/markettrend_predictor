import streamlit as st
import pandas as pd
from predictor import fetch_market_data, prepare_features, train_and_predict

# 1. Page Configuration
st.set_page_config(page_title="Market Predictor & Extractor", layout="wide")
st.title("📈 Market Trend Predictor & Data Extractor")
st.markdown("Extract live financial data, run predictive modeling, and export to spreadsheets.")

# 2. Sidebar Inputs
st.sidebar.header("Configure Extraction")
ticker = st.sidebar.text_input("Asset Ticker (e.g., AAPL, TSLA, BTC-USD)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# 3. Connect Logic with Streamlit Caching
@st.cache_data
def run_pipeline(ticker_symbol, start, end):
    """Wraps the backend logic so Streamlit can cache the results."""
    raw_data = fetch_market_data(ticker_symbol, start, end)
    if raw_data is None:
        return None, None
    
    engineered_data = prepare_features(raw_data)
    final_data, model_accuracy = train_and_predict(engineered_data)
    
    return final_data, model_accuracy

# 4. App Execution & UI Display
with st.spinner("Extracting Market Data & Training Model..."):
    df, accuracy = run_pipeline(ticker, start_date, end_date)

if df is not None:
    # Dashboard Metrics
    col1, col2 = st.columns(2)
    col1.metric(label="Model Accuracy (Test Set)", value=f"{accuracy * 100:.2f}%")
    col2.metric(label="Total Rows Extracted", value=len(df))
    
    # Visualizations
    st.subheader(f"Historical Closing Price: {ticker}")
    st.line_chart(df['Close'])
    
    st.subheader("Extracted Data & Predictive Output")
    st.dataframe(df.tail(10))

    # Export Functionality
    st.markdown("---")
    st.subheader("📥 Export to Spreadsheet")
    
    csv = df.to_csv().encode('utf-8')
    
    st.download_button(
        label="Download Full Data as CSV",
        data=csv,
        file_name=f"{ticker}_market_predictions.csv",
        mime="text/csv",
        type="primary"
    )
else:
    st.error("Could not extract data. Please check the ticker symbol.")
