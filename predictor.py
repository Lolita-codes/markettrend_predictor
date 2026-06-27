import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def fetch_market_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Fetches raw market data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    return data

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers financial indicators and sets up the target variable."""
    # Quantitative Indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Target Variable: 1 if tomorrow's price goes up, 0 if it goes down
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    
    return df

def train_and_predict(df: pd.DataFrame):
    """Trains the Random Forest model and generates predictions."""
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50', 'Daily_Return']
    X = df[features]
    y = df['Target']
    
    # Train/Test Split (Chronological for time-series)
    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Append predictions to the dataframe 
    df['ML_Prediction'] = np.nan
    df.iloc[split:, df.columns.get_loc('ML_Prediction')] = predictions
    
    return df, accuracy
    