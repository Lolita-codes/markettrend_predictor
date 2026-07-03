# 📈 Market Trend Predictor & Data Extractor

This repository contains a full-stack data pipeline and interactive web dashboard built for financial data extraction and predictive modeling. The application allows users to query live market data for any asset, analyze quantitative indicators, and predict future price movements using machine learning.

## Key Features

**Live Data Extraction**: Integrates with the Yahoo Finance API (yfinance) to instantly pull years of historical market data.

**Feature Engineering**: Automatically calculates Simple Moving Averages (SMA 10 & 50) and daily percentage returns.

**Predictive Modeling**: Utilizes a scikit-learn Random Forest Classifier to predict binary market trends (Higher/Lower closing prices).

**Automated Export**: Features a 1-click data structuring pipeline that formats the raw data, engineered features, and machine learning predictions into a downloadable CSV spreadsheet.

## Tech Stack: 
* Python
* Streamlit
* Scikit-Learn
* Pandas
* NumPy
* YFinance
