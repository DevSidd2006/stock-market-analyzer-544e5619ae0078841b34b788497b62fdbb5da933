# 📈 Stock Market Analyzer

A comprehensive Flask web application for analyzing Indian stock market data with advanced features for traders and investors.

![Stock Market Analyzer](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

### 📊 **Comprehensive Stock Analysis**
- **Real-time Stock Data**: Current prices, historical performance, and market metrics
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages (50-day, 200-day)
- **Financial Ratios**: P/E, P/B, P/S, PEG, ROE, Debt-to-Equity, and more
- **Returns Analysis**: Performance across multiple timeframes (1D, 1W, 1M, 3M, 6M, 1Y, YTD)

### 🎯 **Advanced Analytics**
- **AI-Powered Sentiment Analysis**: News sentiment analysis using NLTK
- **Investment Recommendations**: AI-generated Buy/Sell/Hold recommendations with confidence scores
- **Pros & Cons Analysis**: Automated analysis of company strengths and weaknesses
- **Peer Comparison**: Compare with industry competitors
- **Risk Metrics**: Volatility, maximum drawdown, and beta calculations

### 📱 **Modern User Interface**
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Interactive Charts**: Dynamic price charts and technical indicators
- **Tabbed Interface**: Organized view of different analysis aspects
- **Real-time Data**: Live market data integration

### 🔄 **API Support**
- **RESTful API**: JSON endpoints for frontend integration
- **CORS Enabled**: Ready for frontend-backend separation
- **Health Monitoring**: Built-in health check endpoints

## 🛠️ Technology Stack

- **Backend**: Flask (Python 3.11+)
- **Data Sources**: 
  - Yahoo Finance API (yfinance)
  - NewsAPI.org for news articles
  - Alpha Vantage (optional)
- **Analytics**: 
  - Pandas for data processing
  - NumPy for mathematical operations
  - NLTK for sentiment analysis
- **Frontend**: 
  - Bootstrap 5 for responsive UI
  - Chart.js for interactive visualizations
  - Vanilla JavaScript
- **Deployment**: 
  - Render (Backend)
  - Vercel (Frontend - optional)

## 📋 Prerequisites

- Python 3.11 or higher
- pip package manager
- NewsAPI.org API key (free)
- Alpha Vantage API key (optional)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/stock-market-analyzer.git
cd stock-market-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_RUN_PORT=5000

# API Keys
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here

# Configuration
DEFAULT_LANGUAGE=en
DEFAULT_NUM_ARTICLES=5
```

### 4. Run the Application
```bash
python app.py
```

Visit `http://localhost:5000` to access the application.

## 🔑 API Keys Setup

### NewsAPI.org (Required)
1. Visit [newsapi.org](https://newsapi.org)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add it to your `.env` file as `NEWS_API_KEY`

### Alpha Vantage (Optional)
1. Visit [alphavantage.co](https://www.alphavantage.co)
2. Get a free API key
3. Add it to your `.env` file as `ALPHA_VANTAGE_API_KEY`

## 📊 Supported Exchanges

- **NSE (National Stock Exchange)**: Use `.NS` suffix (e.g., `TCS.NS`, `RELIANCE.NS`)
- **BSE (Bombay Stock Exchange)**: Use `.BO` suffix (e.g., `TCS.BO`, `RELIANCE.BO`)

## 🌐 API Endpoints

### Web Interface
- `GET /` - Home page with stock search
- `POST /analyze` - Analyze stock (returns HTML)
- `GET /all_stocks` - List all available stocks

### JSON API
- `POST /api/analyze` - Stock analysis (returns JSON)
- `GET /api/stocks` - Available stocks list (JSON)
- `GET /api/health` - Health check

### Example API Usage
```javascript
// Analyze a stock
fetch('/api/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    ticker: 'TCS.NS'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🚀 Deployment

### Deploy to Render (Backend)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration

3. **Set Environment Variables** in Render dashboard:
   - `NEWS_API_KEY`: Your NewsAPI key
   - `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage key
   - `FLASK_ENV`: `production`

### Deploy Frontend to Vercel (Optional)

If you want to separate frontend and backend:

1. Create a separate frontend using React/Next.js
2. Use the `/api/*` endpoints for data
3. Deploy frontend to Vercel
4. Update CORS settings for your frontend domain

## 📁 Project Structure

```
stock-market-analyzer/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version for deployment
├── Procfile              # Deployment configuration
├── render.yaml           # Render deployment config
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
└── templates/
    ├── index.html       # Home page
    └── results.html     # Analysis results page
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: `development` or `production`
- `FLASK_DEBUG`: `True` or `False`
- `NEWS_API_KEY`: NewsAPI.org API key
- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage API key
- `DEFAULT_LANGUAGE`: Language for news (default: `en`)
- `DEFAULT_NUM_ARTICLES`: Number of news articles to analyze (default: `5`)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues and Support

If you encounter any issues or need support:

1. Check the [Issues](https://github.com/yourusername/stock-market-analyzer/issues) page
2. Create a new issue with detailed description
3. Include error logs and steps to reproduce

## 🚀 Future Enhancements

- [ ] Portfolio tracking and management
- [ ] Real-time price alerts
- [ ] Advanced charting with candlestick patterns
- [ ] Options and derivatives analysis
- [ ] Backtesting framework
- [ ] Machine learning price predictions
- [ ] Mobile app (React Native)
- [ ] Multi-language support

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

---

⭐ **If you find this project helpful, please consider giving it a star on GitHub!**
