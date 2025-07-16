from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import yfinance as yf
import requests
import csv
from io import StringIO

# Load environment variables
load_dotenv()

# Use environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
DEFAULT_NUM_ARTICLES = int(os.getenv("DEFAULT_NUM_ARTICLES", 5))

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')



app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Add this template filter before your routes
@app.template_filter()
def format_market_cap(value):
    """Format market cap in Indian currency format (Crores)"""
    try:
        if not value:
            return 'N/A'
        
        # Convert to crores (1 crore = 10 million)
        crores = value / 10000000
        
        if (crores >= 100000):
            return f"₹{crores/1000:.2f}L Cr"  # Convert to lakh crores
        elif (crores >= 1000):
            return f"₹{crores/1000:.2f}K Cr"  # Convert to thousand crores
        else:
            return f"₹{crores:.2f} Cr"
    except:
        return 'N/A'

class IndianStockAnalyzer:
    def __init__(self, ticker):
        """
        Initialize the IndianStockAnalyzer.
        
        Parameters:
            ticker (str): The stock ticker symbol (e.g., 'TCS.NS' or 'RELIANCE.BO').
        """
        self.ticker = ticker
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        self.analysis_results = {}
        self.recommendation = None
        self.confidence = None

    def get_historical_data(self, period='10y'):
        """Get historical price data for the specified period"""
        try:
            stock = yf.Ticker(self.ticker)
            
            # Calculate start date based on period
            end_date = datetime.now()
            if (period == '1w'):
                start_date = end_date - timedelta(days=7)
            elif (period == '1m'):
                start_date = end_date - timedelta(days=30)
            elif (period == '1y'):
                start_date = end_date - timedelta(days=365)
            elif (period == '5y'):
                start_date = end_date - timedelta(days=365 * 5)
            else:  # 10y
                start_date = end_date - timedelta(days=365 * 10)
            
            # Fetch data using exact date range
            hist_data = stock.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                print(f"No historical data available for {self.ticker}")
                return None
            
            # Calculate daily returns
            hist_data['Daily_Return'] = hist_data['Close'].pct_change()
            
            # Calculate cumulative returns
            hist_data['Cumulative_Return'] = (1 + hist_data['Daily_Return']).cumprod() - 1
            
            return hist_data
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return None

    def get_period_returns(self):
        """Calculate returns for different time periods"""
        try:
            returns_data = {}
            chart_data = {}
            current_price = None

            # Get data for each period separately
            for period in ['1w', '1m', '1y', '5y', '10y']:
                hist_data = self.get_historical_data(period=period)
                if hist_data is not None and not hist_data.empty:
                    if current_price is None:
                        current_price = hist_data['Close'].iloc[-1]
                    
                    # Calculate return for the period
                    start_price = hist_data['Close'].iloc[0]
                    end_price = hist_data['Close'].iloc[-1]
                    period_return = (end_price - start_price) / start_price
                    
                    returns_data[period] = {
                        "return": period_return,
                        "start_price": start_price,
                        "end_price": end_price,
                        "start_date": hist_data.index[0].strftime('%Y-%m-%d'),
                        "end_date": hist_data.index[-1].strftime('%Y-%m-%d')
                    }
                    
                    # Prepare chart data
                    chart_data[period] = {
                        "dates": hist_data.index.strftime('%Y-%m-%d').tolist(),
                        "prices": hist_data['Close'].tolist(),
                        "returns": hist_data['Cumulative_Return'].tolist()
                    }

            if not returns_data:
                return {"status": "error", "message": "No historical data available"}

            return {
                "status": "success",
                "current_price": current_price,
                "returns": returns_data,
                "chart_data": chart_data
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _calculate_period_return(self, hist_data, days=None):
        """Calculate return for a specific period"""
        if days:
            if (len(hist_data) < days):
                return None
            period_data = hist_data.tail(days)
        else:
            period_data = hist_data

        if (len(period_data) < 2):
            return None

        start_price = period_data['Close'].iloc[0]
        end_price = period_data['Close'].iloc[-1]
        period_return = (end_price - start_price) / start_price
        
        return {
            "return": period_return,
            "start_price": start_price,
            "end_price": end_price
        }

    def _prepare_chart_data(self, hist_data, days=None):
        """Prepare chart data for a specific period"""
        if days:
            if (len(hist_data) < days):
                return None
            period_data = hist_data.tail(days)
        else:
            period_data = hist_data

        return {
            "dates": period_data.index.strftime('%Y-%m-%d').tolist(),
            "prices": period_data['Close'].tolist(),
            "returns": period_data['Cumulative_Return'].tolist()
        }

    def analyze_price_trends(self, hist_data):
        """Analyze price trends and calculate key metrics"""
        if (hist_data is None or hist_data.empty):
            return {"status": "error", "message": "No historical data available"}
        
        # Calculate daily returns
        hist_data['Daily_Return'] = hist_data['Close'].pct_change()
        
        # Calculate moving averages
        hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
        hist_data['MA200'] = hist_data['Close'].rolling(window=200).mean()
        
        # Calculate volatility (standard deviation of returns)
        volatility = hist_data['Daily_Return'].std() * np.sqrt(252)  # Annualized
        
        # Calculate total return
        total_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0]) - 1
        
        # Calculate MACD
        exp1 = hist_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist_data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist_data['MACD'] = macd
        hist_data['Signal_Line'] = signal

        # Calculate Bollinger Bands
        hist_data['20_MA'] = hist_data['Close'].rolling(window=20).mean()
        hist_data['Upper_Band'] = hist_data['20_MA'] + (hist_data['Close'].rolling(window=20).std() * 2)
        hist_data['Lower_Band'] = hist_data['20_MA'] - (hist_data['Close'].rolling(window=20).std() * 2)

        # Check if stock is in a bullish trend (50-day MA > 200-day MA)
        latest_data = hist_data.iloc[-1]
        bullish_trend = latest_data['MA50'] > latest_data['MA200']
        
        # Calculate RSI (Relative Strength Index)
        delta = hist_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        hist_data['RSI'] = 100 - (100 / (1 + rs))
        current_rsi = hist_data['RSI'].iloc[-1]
        
        results = {
            "status": "success",
            "total_return": total_return,
            "volatility": volatility,
            "bullish_trend": bullish_trend,
            "current_price": latest_data['Close'],
            "rsi": current_rsi,
            "ma50": latest_data['MA50'],
            "ma200": latest_data['MA200'],
            "macd": latest_data['MACD'],
            "signal_line": latest_data['Signal_Line'],
            "upper_band": latest_data['Upper_Band'],
            "lower_band": latest_data['Lower_Band']
        }
        
        return results

    def analyze_financial_metrics(self):
        """Analyze financial metrics like P/E ratio, EPS growth, etc."""
        try:
            # Fetch financial data using Yahoo Finance
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            if not info:
                return {"status": "error", "message": "No financial metrics available"}
            
            # Extract key financial metrics
            metrics = {
                "pe_ratio": info.get("trailingPE", 0),
                "eps": info.get("trailingEps", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "profit_margin": info.get("profitMargins", 0),
                "revenue_growth": info.get("revenueGrowth", 0)
            }
            
            return {"status": "success", "metrics": metrics}
            
        except Exception as e:
            return {"status": "error", "message": f"Error fetching financial metrics: {str(e)}"}

    def analyze_news_sentiment(self, num_articles=5):
        """Analyze news sentiment for the stock"""
        try:
            # Fetch news articles using Yahoo Finance
            stock = yf.Ticker(self.ticker)
            news = stock.news[:num_articles]
            
            if not news:
                return {"status": "error", "message": "No news articles available"}
            
            # Analyze sentiment for each article
            sentiment_scores = []
            for article in news:
                title = article.get("title", "")
                summary = article.get("summary", "")
                content = title + " " + summary
                score = self.sentiment_analyzer.polarity_scores(content)
                sentiment_scores.append(score)
            
            # Calculate average sentiment
            avg_positive = sum(score['pos'] for score in sentiment_scores) / len(sentiment_scores)
            avg_negative = sum(score['neg'] for score in sentiment_scores) / len(sentiment_scores)
            avg_neutral = sum(score['neu'] for score in sentiment_scores) / len(sentiment_scores)
            avg_compound = sum(score['compound'] for score in sentiment_scores) / len(sentiment_scores)
            
            return {
                "status": "success",
                "avg_sentiment": {
                    "positive": avg_positive,
                    "negative": avg_negative,
                    "neutral": avg_neutral,
                    "compound": avg_compound
                },
                "sentiment_scores": sentiment_scores
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error analyzing news sentiment: {str(e)}"}

    def get_company_details(self):
        """Fetch detailed company information"""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            company_details = {
                "status": "success",
                "details": {
                    "company_name": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "description": info.get("longBusinessSummary", "N/A"),
                    "website": info.get("website", "N/A"),
                    "market_cap": info.get("marketCap", "N/A"),
                    "employees": info.get("fullTimeEmployees", "N/A"),
                    "country": info.get("country", "N/A"),
                    "state": info.get("state", "N/A"),
                    "city": info.get("city", "N/A"),
                    "exchange": info.get("exchange", "N/A"),
                    "currency": info.get("currency", "N/A"),
                    "beta": info.get("beta", "N/A"),
                    "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                    "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                    "avg_volume": info.get("averageVolume", "N/A"),
                    "market_rank": info.get("marketRank", "N/A")
                }
            }
            return company_details
        except Exception as e:
            return {"status": "error", "message": f"Error fetching company details: {str(e)}"}

    def get_detailed_financials(self):
        """
        Fetch comprehensive financial details for the stock including ratios, growth metrics, and valuations.
        Returns a dictionary containing all financial metrics.
        """
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info

            # Basic company information
            company_info = {
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
                "website": info.get("website", "N/A"),
                "employees": info.get("fullTimeEmployees", "N/A"),
                "country": info.get("country", "N/A"),
                "state": info.get("state", "N/A"),
                "city": info.get("city", "N/A")
            }

            # Market data
            market_data = {
                "exchange": info.get("exchange", "N/A"),
                "currency": info.get("currency", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "market_rank": info.get("marketRank", "N/A")
            }

            # Financial metrics and ratios
            financial_metrics = {
                "pe_ratio": info.get("trailingPE", 0),
                "eps": info.get("trailingEps", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "profit_margin": info.get("profitMargins", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "return_on_equity": info.get("returnOnEquity", 0),
                "book_value": info.get("bookValue", 0),
                "price_to_book": info.get("priceToBook", 0),
                "current_ratio": info.get("currentRatio", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "quick_ratio": info.get("quickRatio", 0)
            }

            # Additional financial data
            try:
                # Get income statement data
                income_stmt = stock.income_stmt
                if not income_stmt.empty:
                    latest_income = income_stmt.iloc[:, 0]  # Get most recent quarter/year
                    financial_metrics.update({
                        "total_revenue": latest_income.get("Total Revenue", 0),
                        "operating_income": latest_income.get("Operating Income", 0),
                        "net_income": latest_income.get("Net Income", 0)
                    })

                # Get balance sheet data
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty:
                    latest_balance = balance_sheet.iloc[:, 0]  # Get most recent quarter/year
                    financial_metrics.update({
                        "total_assets": latest_balance.get("Total Assets", 0),
                        "total_liabilities": latest_balance.get("Total Liabilities Net Minority Interest", 0),
                        "total_equity": latest_balance.get("Total Equity Gross Minority Interest", 0)
                    })
            except Exception as e:
                print(f"Error fetching detailed financial statements: {str(e)}")

            # Combine all data
            detailed_info = {
                "status": "success",
                "details": {
                    **company_info,
                    **market_data,
                    **financial_metrics
                }
            }

            return detailed_info

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching financial details: {str(e)}"
            }

    def analyze_pros_and_cons(self):
        """Analyze company strengths and weaknesses based on various metrics"""
        try:
            pros = []
            cons = []
            
            # Get company details and financial metrics
            company_details = self.get_detailed_financials()
            if (company_details.get('status') != 'success'):
                return {"status": "error", "message": "Unable to analyze pros and cons"}
            
            details = company_details['details']
            
            # Analyze Market Position
            if (details.get('market_cap', 0) > 100000000000):  # 10,000 Cr
                pros.append("Strong market position with large market capitalization")
            elif (details.get('market_cap', 0) < 10000000000):  # 1,000 Cr
                cons.append("Smaller market cap might indicate limited market presence")

            # Analyze Financial Health
            if (details.get('debt_to_equity', 0) < 1):
                pros.append("Healthy debt-to-equity ratio indicating strong financial position")
            elif (details.get('debt_to_equity', 0) > 2):
                cons.append("High debt-to-equity ratio might pose financial risks")

            # Analyze Profitability
            profit_margin = details.get('profit_margin', 0)
            if (profit_margin > 0.15):  # 15%
                pros.append("Strong profit margins indicating efficient operations")
            elif (profit_margin < 0.05):  # 5%
                cons.append("Low profit margins might indicate operational inefficiencies")

            # Analyze Growth
            revenue_growth = details.get('revenue_growth', 0)
            if (revenue_growth > 0.10):  # 10%
                pros.append("Impressive revenue growth showing business expansion")
            elif (revenue_growth < 0):
                cons.append("Declining revenue might indicate business challenges")

            # Analyze Returns
            roe = details.get('return_on_equity', 0)
            if (roe > 0.15):  # 15%
                pros.append("Strong return on equity indicating efficient use of shareholder funds")
            elif (roe < 0.08):  # 8%
                cons.append("Low return on equity might indicate inefficient capital utilization")

            # Analyze Valuation
            pe_ratio = details.get('pe_ratio', 0)
            industry = details.get('industry', '')
            if (15 <= pe_ratio <= 25):
                pros.append("Reasonable valuation compared to market standards")
            elif (pe_ratio > 50):
                cons.append("High P/E ratio might indicate overvaluation")
            elif (pe_ratio < 10 and pe_ratio > 0):
                pros.append("Potentially undervalued with low P/E ratio")

            # Analyze Dividend
            dividend_yield = details.get('dividend_yield', 0)
            if (dividend_yield > 0.03):  # 3%
                pros.append("Attractive dividend yield providing regular income")
            elif (dividend_yield == 0):
                cons.append("No dividend payments might deter income-focused investors")

            # Analyze Market Sentiment
            beta = details.get('beta', 0)
            if (0.8 <= beta <= 1.2):
                pros.append("Moderate market sensitivity indicating balanced risk")
            elif (beta > 1.5):
                cons.append("High market sensitivity might indicate increased volatility")

            # Analyze Business Model
            if (details.get('employees', 0) > 10000):
                pros.append("Large workforce indicating established business operations")
            
            # Get historical performance
            hist_returns = self.get_period_returns()
            if (hist_returns.get('status') == 'success'):
                yearly_return = hist_returns.get('returns', {}).get('1y', {}).get('return', 0)
                if (yearly_return > 0.15):  # 15%
                    pros.append("Strong stock performance over the past year")
                elif (yearly_return < -0.15):  # -15%
                    cons.append("Poor stock performance over the past year")

            return {
                "status": "success",
                "analysis": {
                    "pros": pros,
                    "cons": cons
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing pros and cons: {str(e)}"
            }

    def analyze_peer_comparison(self):
        """Analyze how the stock compares to its industry peers."""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            industry = info.get('industry')
            if not industry:
                return {"status": "error", "message": "Industry information not available."}

            # This is a simplified peer finding logic.
            # A more robust implementation would use a pre-compiled list of companies by industry.
            all_stocks = fetch_all_stocks()
            peers = [s['symbol'] for s in all_stocks if yf.Ticker(s['symbol'] + '.NS').info.get('industry') == industry and s['symbol'] != self.ticker.replace('.NS', '')][:5]

            peer_data = []
            for peer_ticker in peers:
                try:
                    peer_stock = yf.Ticker(peer_ticker + '.NS')
                    peer_info = peer_stock.info
                    peer_data.append({
                        'ticker': peer_ticker,
                        'pe_ratio': peer_info.get('trailingPE'),
                        'roe': peer_info.get('returnOnEquity'),
                        'debt_to_equity': peer_info.get('debtToEquity')
                    })
                except Exception as e:
                    print(f"Could not fetch data for peer {peer_ticker}: {e}")

            return {"status": "success", "peers": peer_data}
        except Exception as e:
            return {"status": "error", "message": f"Error in peer comparison: {str(e)}"}

    def get_factor_scores(self):
        """Calculate scores for different factors like value, growth, quality, and momentum."""
        try:
            financial_metrics = self.analysis_results['financial_metrics']['metrics']
            price_trends = self.analysis_results['price_trends']

            # Value Score
            pe_ratio = financial_metrics.get('pe_ratio', 0)
            pb_ratio = financial_metrics.get('pb_ratio', 0)
            value_score = 0
            if pe_ratio > 0 and pe_ratio < 20: value_score += 1
            if pb_ratio > 0 and pb_ratio < 3: value_score += 1

            # Growth Score
            revenue_growth = financial_metrics.get('revenue_growth', 0)
            eps_growth = financial_metrics.get('eps', 0) # Simplified, should be growth rate
            growth_score = 0
            if revenue_growth > 0.1: growth_score += 1
            if eps_growth > 0.1: growth_score += 1

            # Quality Score
            roe = financial_metrics.get('roe', 0)
            debt_to_equity = financial_metrics.get('debt_to_equity', 0)
            quality_score = 0
            if roe > 0.15: quality_score += 1
            if debt_to_equity < 1: quality_score += 1

            # Momentum Score
            rsi = price_trends.get('rsi', 0)
            macd = price_trends.get('macd', 0)
            momentum_score = 0
            if rsi > 50: momentum_score += 1
            if macd > 0: momentum_score += 1

            return {
                "status": "success",
                "scores": {
                    "value": value_score,
                    "growth": growth_score,
                    "quality": quality_score,
                    "momentum": momentum_score
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"Error in factor scoring: {str(e)}"}

    def run_analysis(self):
        """Run all analyses and generate a recommendation"""
        print(f"Analyzing {self.ticker}...")
        
        # Get company details and financial metrics
        company_details = self.get_detailed_financials()
        self.analysis_results['company_details'] = company_details
        
        # Get historical data and returns for different periods
        historical_returns = self.get_period_returns()
        self.analysis_results['historical_returns'] = historical_returns
        
        # Analyze peer comparison
        peer_comparison_results = self.analyze_peer_comparison()
        self.analysis_results['peer_comparison'] = peer_comparison_results

        # Analyze pros and cons
        pros_and_cons = self.analyze_pros_and_cons()
        self.analysis_results['pros_and_cons'] = pros_and_cons
        
        # Get historical data
        hist_data = self.get_historical_data()
        
        if (hist_data is None):
            return {"status": "error", "message": "No historical data available for this stock."}
        
        # Analyze price trends
        price_results = self.analyze_price_trends(hist_data)
        self.analysis_results['price_trends'] = price_results
        
        # Analyze financial metrics
        financial_results = self.analyze_financial_metrics()
        self.analysis_results['financial_metrics'] = financial_results
        
        # Analyze news sentiment
        sentiment_results = self.analyze_news_sentiment()
        self.analysis_results['news_sentiment'] = sentiment_results
        
        # Add P/E ratio to analysis factors
        info = yf.Ticker(self.ticker).info
        pe_ratio = info.get('trailingPE')
        
        if pe_ratio:
            if (pe_ratio < 15):
                pe_analysis = "The stock appears undervalued based on P/E ratio"
                pe_score = 1
            elif (pe_ratio < 25):
                pe_analysis = "The stock appears fairly valued based on P/E ratio"
                pe_score = 0.5
            else:
                pe_analysis = "The stock appears overvalued based on P/E ratio"
                pe_score = 0
            
            # Add P/E analysis to recommendation factors
            if ('recommendation' in self.analysis_results):
                self.analysis_results['recommendation']['explanation'].append(
                    f"P/E Ratio: {pe_ratio:.2f} - {pe_analysis}"
                )
                
                # Adjust confidence based on P/E score
                current_confidence = self.analysis_results['recommendation']['confidence']
                pe_weight = 0.2  # 20% weight to P/E ratio
                new_confidence = (current_confidence * (1 - pe_weight)) + (pe_score * 100 * pe_weight)
                self.analysis_results['recommendation']['confidence'] = new_confidence
        
        # Get factor scores
        factor_scores = self.get_factor_scores()
        self.analysis_results['factor_scores'] = factor_scores

        # Generate recommendation
        self.generate_recommendation()
        
        return self.analysis_results

    def generate_recommendation(self):
        """Generate investment recommendation based on all analyses"""
        # Initialize scoring
        score = 0
        max_score = 0
        explanation = []
        
        # Price trend analysis
        price_trends = self.analysis_results['price_trends']
        if (price_trends['status'] == 'success'):
            max_score += 40
            
            # Total return score (up to 15 points)
            total_return = price_trends['total_return']
            if (total_return > 0.5):  # 50% or more return
                score += 15
                explanation.append(f"Strong historical return of {total_return:.1%}")
            elif (total_return > 0.2):  # 20% or more return
                score += 10
                explanation.append(f"Good historical return of {total_return:.1%}")
            elif (total_return > 0):
                score += 5
                explanation.append(f"Positive historical return of {total_return:.1%}")
            else:
                explanation.append(f"Negative historical return of {total_return:.1%}")
            
            # Bullish trend score (up to 10 points)
            if (price_trends['bullish_trend']):
                score += 10
                explanation.append("Bullish trend (50-day MA > 200-day MA)")
            else:
                explanation.append("Bearish trend (50-day MA <= 200-day MA)")
            
            # RSI score (up to 10 points)
            rsi = price_trends['rsi']
            if (rsi < 30):
                score += 10
                explanation.append(f"RSI indicates oversold conditions ({rsi:.1f})")
            elif (rsi < 50):
                score += 5
                explanation.append(f"RSI indicates neutral conditions ({rsi:.1f})")
            else:
                explanation.append(f"RSI indicates overbought conditions ({rsi:.1f})")
        
        # Financial metrics analysis
        financial_metrics = self.analysis_results['financial_metrics']
        if (financial_metrics['status'] == 'success'):
            max_score += 30
            
            # P/E ratio score (up to 10 points)
            pe_ratio = financial_metrics['metrics'].get("pe_ratio", 0)
            if (pe_ratio < 15):
                score += 10
                explanation.append(f"Low P/E ratio ({pe_ratio:.1f})")
            elif (pe_ratio < 25):
                score += 7
                explanation.append(f"Moderate P/E ratio ({pe_ratio:.1f})")
            else:
                explanation.append(f"High P/E ratio ({pe_ratio:.1f})")
            
            # Revenue growth score (up to 10 points)
            revenue_growth = financial_metrics['metrics'].get("revenue_growth", 0)
            if (revenue_growth > 0.2):
                score += 10
                explanation.append(f"Strong revenue growth ({revenue_growth:.1%})")
            elif (revenue_growth > 0.1):
                score += 7
                explanation.append(f"Moderate revenue growth ({revenue_growth:.1%})")
            else:
                explanation.append(f"Low revenue growth ({revenue_growth:.1%})")
            
            # Profit margin score (up to 10 points)
            profit_margin = financial_metrics['metrics'].get("profit_margin", 0)
            if (profit_margin > 0.2):
                score += 10
                explanation.append(f"High profit margin ({profit_margin:.1%})")
            elif (profit_margin > 0.1):
                score += 7
                explanation.append(f"Moderate profit margin ({profit_margin:.1%})")
            else:
                explanation.append(f"Low profit margin ({profit_margin:.1%})")
        
        # News sentiment analysis
        news_sentiment = self.analysis_results['news_sentiment']
        if (news_sentiment['status'] == 'success'):
            max_score += 30
            
            # Compound sentiment score (up to 30 points)
            compound_sentiment = news_sentiment['avg_sentiment']['compound']
            if (compound_sentiment > 0.2):
                score += 30
                explanation.append(f"Positive news sentiment ({compound_sentiment:.2f})")
            elif (compound_sentiment > -0.2):
                score += 15
                explanation.append(f"Neutral news sentiment ({compound_sentiment:.2f})")
            else:
                explanation.append(f"Negative news sentiment ({compound_sentiment:.2f})")
        
        # Calculate confidence level
        if (max_score > 0):
            confidence = (score / max_score) * 100
            self.confidence = confidence
        else:
            self.confidence = 0
        
        # Generate recommendation based on score
        if (score >= 80):
            self.recommendation = "Strong Buy"
        elif (score >= 60):
            self.recommendation = "Buy"
        elif (score >= 40):
            self.recommendation = "Hold"
        elif (score >= 20):
            self.recommendation = "Sell"
        else:
            self.recommendation = "Strong Sell"
        
        # Add explanation to analysis results
        self.analysis_results['recommendation'] = {
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "explanation": explanation
        }

def analyze_company_sentiment(text, company_info):
    """
    Analyze sentiment specifically for company-related content with improved accuracy
    """
    try:
        sia = SentimentIntensityAnalyzer()
        
        # Get base sentiment
        sentiment_scores = sia.polarity_scores(text)
        
        # Custom words and phrases that indicate positive/negative sentiment in financial context
        financial_pos_words = {
            'outperform', 'beat', 'exceeded', 'growth', 'profit', 'upgrade', 'innovative',
            'strong', 'positive', 'success', 'advantage', 'leading', 'opportunity', 'efficient',
            'breakthrough', 'expansion', 'partnership', 'collaboration', 'launch', 'dividend',
            'surpass', 'momentum', 'robust', 'promising', 'optimistic'
        }
        
        financial_neg_words = {
            'underperform', 'miss', 'decline', 'loss', 'downgrade', 'weak', 'negative',
            'lawsuit', 'investigation', 'recall', 'delay', 'suspend', 'crisis', 'risk',
            'concern', 'volatile', 'uncertainty', 'bearish', 'debt', 'bankruptcy',
            'restructuring', 'layoff', 'dispute', 'penalty', 'fine'
        }
        
        # Context-specific sentiment adjustment
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Count financial sentiment words
        pos_count = sum(1 for word in financial_pos_words if word in text_lower)
        neg_count = sum(1 for word in financial_neg_words if word in text_lower)
        
        # Adjust sentiment based on financial context
        sentiment_adjustment = (pos_count - neg_count) * 0.1
        
        # Check for specific phrases that might indicate company performance
        performance_indicators = {
            'positive': ['revenue growth', 'profit increase', 'market share gain', 'cost reduction',
                        'new contract', 'strategic partnership', 'successful launch'],
            'negative': ['revenue decline', 'profit drop', 'market share loss', 'cost increase',
                        'contract termination', 'partnership end', 'failed launch']
        }
        
        # Adjust sentiment based on performance indicators
        for phrase in performance_indicators['positive']:
            if (phrase in text_lower):
                sentiment_adjustment += 0.15
        
        for phrase in performance_indicators['negative']:
            if (phrase in text_lower):
                sentiment_adjustment -= 0.15
        
        # Final sentiment score
        final_compound = max(min(sentiment_scores['compound'] + sentiment_adjustment, 1.0), -1.0)
        
        return {
            'score': final_compound,
            'magnitude': abs(final_compound),
            'classification': 'positive' if final_compound > 0.2 else 'negative' if final_compound < -0.2 else 'neutral'
        }
    
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return {'score': 0, 'magnitude': 0, 'classification': 'neutral'}

def fetch_company_news(ticker):
    """Fetch and process news articles that are directly or indirectly related to the company"""
    try:
        # Remove .NS or .BO from ticker for news search
        company_name = ticker.split('.')[0]
        
        # Get company info to help with news filtering
        stock = yf.Ticker(ticker)
        info = stock.info
        company_info = {
            'name': info.get('longName', company_name),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'competitors': info.get('companyOfficers', []),
            'products': info.get('longBusinessSummary', '')
        }
        
        # Create search queries for different news sources
        search_terms = [
            company_info['name'],  # Company name
            f"\"{company_info['name']}\"",  # Exact company name match
            company_name,  # Ticker name
            f"({company_info['sector']} AND {company_info['name']})",  # Sector related news
            f"({company_info['industry']} AND {company_info['name']})"  # Industry related news
        ]
        
        # Additional search terms from business summary
        key_terms = [word for word in company_info['products'].split() 
                    if (len(word) > 4 and word.lower() not in {'the', 'and', 'with', 'from'})]
        if key_terms:
            search_terms.append(f"({company_info['name']} AND ({' OR '.join(key_terms[:5])}))")
        
        query = ' OR '.join(search_terms)
        
        # List of news sources to query
        news_sources = [
            {'api_key': NEWS_API_KEY, 'name': "News API"},
            # Add more news source APIs here if available
        ]
        
        all_articles = []
        days_ago = 14  # Extend to 2 weeks for more articles
        from_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Fetch from News API
        url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language={DEFAULT_LANGUAGE}&pageSize=100"
        response = requests.get(url)
        if (response.status_code == 200):
            news_data = response.json()
            all_articles.extend(news_data.get('articles', []))
        
        # Process and filter articles
        processed_articles = []
        seen_titles = set()  # To avoid duplicate news
        
        for article in all_articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"
            
            # Skip if we've seen this title or if title/description is empty
            if (title in seen_titles or not title or not description):
                continue
            seen_titles.add(title)
            
            # Check relevance
            relevance_score = 0
            
            # Direct company name mention
            if (company_info['name'].lower() in content or company_name.lower() in content):
                relevance_score += 3
            
            # Industry and sector relevance
            if (company_info['sector'].lower() in content):
                relevance_score += 1
            if (company_info['industry'].lower() in content):
                relevance_score += 1
            
            # Product or service mention
            for product in company_info['products'].lower().split():
                if (len(product) > 4 and product in content):
                    relevance_score += 1
                    break
            
            # Only include articles with sufficient relevance
            if (relevance_score >= 3):
                # Calculate enhanced sentiment
                full_text = f"{title}. {description}"
                sentiment_analysis = analyze_company_sentiment(full_text, company_info)
                
                # Format date
                pub_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                formatted_date = pub_date.strftime("%B %d, %Y")
                
                processed_article = {
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'url': article.get('url'),
                    'urlToImage': article.get('urlToImage'),
                    'publishedAt': formatted_date,
                    'source': article.get('source', {}).get('name', 'Unknown Source'),
                    'sentiment': sentiment_analysis['score'],
                    'sentiment_magnitude': sentiment_analysis['magnitude'],
                    'sentiment_classification': sentiment_analysis['classification'],
                    'relevance_score': relevance_score
                }
                processed_articles.append(processed_article)
            
            # Limit to top 12 most relevant articles
            if (len(processed_articles) >= 12):
                break
        
        # Sort by relevance score and recency
        processed_articles.sort(key=lambda x: (x['relevance_score'], x['sentiment_magnitude'], x['publishedAt']), reverse=True)
        
        return processed_articles
        
    except Exception as e:
        print(f"Error processing news: {str(e)}")
        return []

@app.route("/")
def home():
    return render_template("index.html")

def fetch_alpha_vantage_data(ticker):
    """
    Fetch company overview data from the Alpha Vantage API.
    Returns a dictionary with the data or an error message.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # If the API returns a note or an error message, handle it as an error.
        if not data or "Note" in data or "Error Message" in data:
            return {"status": "error", "message": "Alpha Vantage API returned no data or an error."}
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route("/analyze", methods=['POST'])
def analyze():
    try:
        ticker = request.form['ticker']
        
        # Get returns analysis
        returns_analysis = calculate_returns_analysis(ticker)
        
        # Get other analyses
        analyzer = IndianStockAnalyzer(ticker)
        results = analyzer.run_analysis()
        
        # Combine all analyses
        results['returns_analysis'] = returns_analysis
        results['ticker'] = ticker
        
        # Fetch news articles
        news_articles = fetch_company_news(ticker)
        results['news_articles'] = news_articles
        
        # Add fundamental metrics from Yahoo Finance
        fundamental_metrics = get_fundamental_metrics(ticker)
        results['fundamental_metrics'] = fundamental_metrics

        # Fetch additional data from Alpha Vantage and add to results 
        alpha_vantage_data = fetch_alpha_vantage_data(ticker)
        results['alpha_vantage_data'] = alpha_vantage_data
        
        # Ensure all required sections exist with proper structure
        if ('price_trends' not in results):
            results['price_trends'] = {'status': 'error'}
        if ('news_sentiment' not in results):
            results['news_sentiment'] = {'status': 'error'}
        if ('recommendation' not in results):
            results['recommendation'] = {
                'recommendation': 'Unknown',
                'confidence': 0,
                'explanation': ['Insufficient data for analysis']
            }
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('results.html', 
                             results={
                                 'status': 'error',
                                 'message': str(e),
                                 'ticker': request.form.get('ticker', 'Unknown')
                             })

def calculate_returns_analysis(ticker):
    """Calculate returns analysis for different time periods"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data for different periods
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            return {'status': 'error', 'message': 'No historical data available'}

        # Initialize default values
        returns_analysis = {
            'status': 'success',
            'one_day_return': 0,          # Add one-day return
            'one_day_change': 0,          # Absolute change
            'weekly_return': 0,
            'monthly_return': 0,
            'three_month_return': 0,
            'six_month_return': 0,
            'yearly_return': 0,
            'ytd_return': 0,
            'weekly_volatility': 0,
            'max_drawdown': 0,
            'weekly_dates': [],
            'weekly_prices': [],
            'monthly_dates': [],
            'monthly_prices': [],
            'yearly_dates': [],
            'yearly_prices': []
        }

        try:
            # Calculate returns for different periods
            current_price = hist_data['Close'][-1]
            
            # One-day return (most recent day)
            if (len(hist_data) >= 2):
                prev_day_price = hist_data['Close'][-2]
                returns_analysis['one_day_return'] = (current_price - prev_day_price) / prev_day_price
                returns_analysis['one_day_change'] = current_price - prev_day_price
            
            # Weekly returns (5 trading days)
            if (len(hist_data) >= 5):
                week_ago_price = hist_data['Close'][-5]
                returns_analysis['weekly_return'] = (current_price - week_ago_price) / week_ago_price
                returns_analysis['weekly_dates'] = hist_data.index[-5:].strftime('%Y-%m-%d').tolist()
                returns_analysis['weekly_prices'] = hist_data['Close'][-5:].tolist()
            
            # Monthly returns (21 trading days)
            if (len(hist_data) >= 21):
                month_ago_price = hist_data['Close'][-21]
                returns_analysis['monthly_return'] = (current_price - month_ago_price) / month_ago_price
                returns_analysis['monthly_dates'] = hist_data.index[-21:].strftime('%Y-%m-%d').tolist()
                returns_analysis['monthly_prices'] = hist_data['Close'][-21:].tolist()
            
            # 3-month returns (63 trading days)
            if (len(hist_data) >= 63):
                three_month_price = hist_data['Close'][-63]
                returns_analysis['three_month_return'] = (current_price - three_month_price) / three_month_price
            
            # 6-month returns (126 trading days)
            if (len(hist_data) >= 126):
                six_month_price = hist_data['Close'][-126]
                returns_analysis['six_month_return'] = (current_price - six_month_price) / six_month_price
            
            # Yearly return
            first_price = hist_data['Close'][0]
            returns_analysis['yearly_return'] = (current_price - first_price) / first_price
            returns_analysis['yearly_dates'] = hist_data.index.strftime('%Y-%m-%d').tolist()
            returns_analysis['yearly_prices'] = hist_data['Close'].tolist()
            
            # YTD return
            year_start = datetime(end_date.year, 1, 1)
            ytd_data = hist_data[hist_data.index >= year_start]
            if not ytd_data.empty:
                ytd_start_price = ytd_data['Close'][0]
                returns_analysis['ytd_return'] = (current_price - ytd_start_price) / ytd_start_price
            
            # Calculate volatility
            returns = hist_data['Close'].pct_change().dropna()
            if not returns.empty:
                returns_analysis['weekly_volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate maximum drawdown
            rolling_max = hist_data['Close'].cummax()
            drawdowns = (hist_data['Close'] - rolling_max) / rolling_max
            returns_analysis['max_drawdown'] = drawdowns.min()

            # Convert numpy float64 to regular Python float
            for key in returns_analysis:
                if isinstance(returns_analysis[key], np.float64):
                    returns_analysis[key] = float(returns_analysis[key])

        except Exception as e:
            print(f"Error in calculations: {str(e)}")
            
        return returns_analysis
        
    except Exception as e:
        print(f"Error calculating returns: {str(e)}")
        return {'status': 'error', 'message': str(e)}

def get_fundamental_metrics(ticker):
    """Fetch fundamental metrics for the stock using Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        metrics = {
            # Valuation Ratios
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'peg_ratio': info.get('pegRatio'),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else None,
            
            # Profitability Metrics
            'eps': info.get('trailingEps'),
            'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else None,
            'debt_to_equity': info.get('debtToEquity'),
            
            # Margin Metrics
            'gross_margin': info.get('grossMargins', 0) * 100 if info.get('grossMargins') else None,
            'operating_margin': info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else None,
            'net_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else None,
            
            # Growth Metrics
            'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else None,
            'quarterly_revenue_growth': info.get('quarterlyRevenueGrowth', 0) * 100 if info.get('quarterlyRevenueGrowth') else None,
            
            # Cash Flow Metrics
            'free_cash_flow': info.get('freeCashflow'),
            'operating_cash_flow': info.get('operatingCashflow'),
            
            # Size Metrics
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue')
        }
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error fetching fundamental metrics: {str(e)}"
        }

def fetch_all_stocks():
    """
    Fetch all active stock listings from the NSE website.
    Returns a list of dictionaries containing stock symbols and names.
    """
    try:
        url = 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        csv_file = StringIO(response.content.decode('utf-8'))
        df_stocks = pd.read_csv(csv_file)
        # Rename columns to be consistent with the old format
        df_stocks = df_stocks.rename(columns={'SYMBOL': 'symbol', 'NAME OF COMPANY': 'name'})
        # Filter for equity series
        df_stocks = df_stocks[df_stocks['SERIES'] == 'EQ']
        return df_stocks[['symbol', 'name']].to_dict('records')
    except Exception as e:
        print(f"Error fetching stock listings from NSE: {str(e)}")
        return []

@app.route("/all_stocks")
def all_stocks():
    """Display all stocks fetched via Alpha Vantage API."""
    stocks = fetch_all_stocks()
    return render_template("all_stocks.html", stocks=stocks)

@app.route("/api/analyze", methods=['POST'])
def api_analyze():
    """API endpoint for stock analysis that returns JSON"""
    try:
        data = request.get_json()
        ticker = data.get('ticker') if data else request.form.get('ticker')
        
        if not ticker:
            return jsonify({"error": "Ticker symbol is required"}), 400
        
        # Get returns analysis
        returns_analysis = calculate_returns_analysis(ticker)
        
        # Get other analyses
        analyzer = IndianStockAnalyzer(ticker)
        results = analyzer.run_analysis()
        
        # Combine all analyses
        results['returns_analysis'] = returns_analysis
        results['ticker'] = ticker
        
        # Fetch news articles
        news_articles = fetch_company_news(ticker)
        results['news_articles'] = news_articles
        
        # Add fundamental metrics from Yahoo Finance
        fundamental_metrics = get_fundamental_metrics(ticker)
        results['fundamental_metrics'] = fundamental_metrics

        # Fetch additional data from Alpha Vantage and add to results 
        alpha_vantage_data = fetch_alpha_vantage_data(ticker)
        results['alpha_vantage_data'] = alpha_vantage_data
        
        # Ensure all required sections exist with proper structure
        if ('price_trends' not in results):
            results['price_trends'] = {'status': 'error'}
        if ('news_sentiment' not in results):
            results['news_sentiment'] = {'status': 'error'}
        if ('recommendation' not in results):
            results['recommendation'] = {
                'recommendation': 'Unknown',
                'confidence': 0,
                'explanation': ['Insufficient data for analysis']
            }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy", "message": "Stock Market Analyzer is running"})

if __name__ == "__main__":
    # For local development
    app.run(debug=True, port=5000)
else:
    # For production deployment
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))