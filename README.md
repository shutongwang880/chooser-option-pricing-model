# chooser-option-pricing-model
An end-to-end quantitative pipeline to price exotic Chooser Options using ML models and real-world financial data.

## Feature Engineering Documentation

The preprocessing pipeline generates 10+ features designed for the Chooser Option Pricing Model.

### 1. Traditional Quantitative Features
- **Log Returns**: Calculated for JPM price to ensure stationarity for time-series analysis.
- **Rolling Volatility (21d)**: Captures the short-term realized risk (approx. 1 business month).
- **Dividend Yield**: Derived from annualizing the 252-day rolling dividends relative to the current price.

### 2. Advanced Macro & Sentiment Features
- **VIX-JPM Correlation**: A 21-day rolling correlation between the S&P 500 Volatility Index and JPM price, capturing the "fear factor" impact on financial stocks.
- **Interest Rate Momentum**: The slope (1st derivative) of the 10-Year Treasury Yield, reflecting shifts in the discount rate environment.
- **Sentiment Proxy**: A normalized score (0-1) derived from price-action momentum and volatility regimes to simulate market sentiment.

### 3. Data Quality Assurance
- **Outlier Removal**: Applied IQR (Interquartile Range) filtering on macro variables to remove extreme noise while preserving dividend events.
- **Time Alignment**: All features are synchronized using a strict `dropna()` policy, ensuring the model only trains on "complete" data days.
