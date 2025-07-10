# Stock Market Analyzer

A comprehensive Flask web application for analyzing Indian stock market data with advanced features for traders and investors.

## Features

- **Real-time Stock Analysis**: Get current stock prices, historical data, and performance metrics
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Financial Metrics**: P/E ratio, EPS, ROE, Debt-to-Equity, and more
- **News Sentiment Analysis**: AI-powered sentiment analysis of recent news articles
- **Returns Analysis**: Performance across multiple time periods (1D, 1W, 1M, 3M, 6M, 1Y, YTD)
- **Investment Recommendations**: AI-generated buy/sell/hold recommendations
- **Pros & Cons Analysis**: Automated analysis of company strengths and weaknesses
- **Interactive Charts**: Visual representation of stock performance and trends

## Technology Stack

- **Backend**: Flask (Python)
- **Data Sources**: Yahoo Finance API, News API
- **Analytics**: NLTK for sentiment analysis, Pandas for data processing
- **Frontend**: Bootstrap 5, Chart.js for visualizations
- **Deployment**: Render

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file:
   ```
   NEWS_API_KEY=your_news_api_key
   DEFAULT_LANGUAGE=en
   DEFAULT_NUM_ARTICLES=5
   ```
4. Run the application: `python app.py`

## Environment Variables

- `NEWS_API_KEY`: API key from NewsAPI.org for fetching news articles
- `DEFAULT_LANGUAGE`: Language for news articles (default: 'en')
- `DEFAULT_NUM_ARTICLES`: Number of articles to analyze (default: 5)

## Usage

1. Enter a stock ticker symbol (e.g., TCS.NS, RELIANCE.NS)
2. Click "Analyze Stock" to get comprehensive analysis
3. View results including financial metrics, charts, news sentiment, and investment recommendations

## Supported Stock Exchanges

- NSE (National Stock Exchange of India) - use .NS suffix
- BSE (Bombay Stock Exchange) - use .BO suffix

## API Dependencies

- Yahoo Finance (yfinance) - for stock data
- NewsAPI.org - for news articles (requires API key)

## License

MIT License
