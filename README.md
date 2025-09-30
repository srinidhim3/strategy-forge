# Strategy Forge 🎯

A powerful, user-friendly web application that allows users to design, backtest, and analyze quantitative stock trading strategies based on custom, rule-based screeners using accurate, point-in-time historical data.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in_development-yellow.svg)

## 🚀 Vision

To create a comprehensive platform that bridges the gap between complex quantitative finance theory and practical implementation, enabling traders and analysts to build, test, and deploy sophisticated trading strategies with confidence.

## ✨ Core Features

- **🔍 Dynamic Screener Builder**: Web interface to define filtering rules based on fundamental and technical indicators
- **📈 Portfolio Backtesting Engine**: Simulates strategies over historical data with accurate point-in-time calculations
- **📊 Performance Analytics Dashboard**: Comprehensive visualizations with key metrics and interactive charts
- **⚡ Job-Based Architecture**: Background workers for long-running backtests with real-time progress tracking
- **🎯 Point-in-Time Accuracy**: Eliminates lookahead bias with proper reporting lag simulation
- **🌐 Multi-Asset Support**: Handle individual stocks or entire portfolios with dynamic rebalancing

## 🛠️ Technology Stack

### Backend
- **API Framework**: FastAPI (high-performance REST API)
- **Background Jobs**: Celery with Redis (async task processing)
- **Database**: PostgreSQL (persistent data storage)
- **Data Processing**: pandas, numpy (efficient data manipulation)

### Frontend
- **Web Interface**: Streamlit (interactive dashboard)
- **Visualization**: matplotlib, seaborn, plotly (comprehensive charts)

### Data Sources
- **Market Data**: Yahoo Finance via yfinance library
- **Coverage**: Daily OHLCV data + Quarterly financial statements

### DevOps
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions
- **Environment**: Python virtual environments

## 📋 Prerequisites

- **Python**: 3.12+ 
- **Git**: For version control
- **Redis**: For background job processing (optional for development)
- **PostgreSQL**: For production deployment (optional for development)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/srinidhim3/strategy-forge.git
cd strategy-forge
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Development Server
```bash
# Start the CLI version (Phase 1)
python main.py

# Start the web interface (Phase 3)
streamlit run src/ui/app.py
```

## 📁 Project Structure

```
strategy-forge/
├── src/                          # Source code
│   ├── data/                     # Data fetching and processing
│   │   ├── price_fetcher.py      # Daily OHLCV data retrieval
│   │   ├── financial_fetcher.py  # Quarterly financial statements
│   │   ├── data_aligner.py       # Point-in-time data alignment
│   │   └── pipeline.py           # Data processing orchestration
│   ├── models/                   # Core business logic
│   │   ├── financial_calculator.py  # Financial metrics computation
│   │   ├── ratio_calculator.py      # Point-in-time ratio calculations
│   │   ├── strategies.py            # Trading strategy implementations
│   │   ├── backtester.py            # Backtesting engine
│   │   ├── screener.py              # Stock screening logic
│   │   └── portfolio.py             # Portfolio management
│   ├── utils/                    # Utility functions
│   │   └── performance.py        # Performance metrics calculation
│   ├── api/                      # FastAPI web API
│   │   ├── main.py               # API entry point
│   │   ├── models.py             # Pydantic data models
│   │   ├── tasks.py              # Celery background tasks
│   │   └── celery_app.py         # Celery configuration
│   └── ui/                       # Streamlit web interface
│       └── app.py                # Web UI entry point
├── tests/                        # Test suite
├── data/                         # Downloaded data cache
├── reports/                      # Generated reports
├── docs/                         # Additional documentation
├── requirements.txt              # Python dependencies
├── DESIGN_DOCUMENT.md           # Detailed technical design
├── docker-compose.yml           # Docker services configuration
├── Dockerfile                   # Container definition
└── README.md                    # This file
```

## 🏗️ Development Roadmap

### Phase 1: Core Data & Backtesting Engine (CLI) ✅
- [ ] Data fetcher modules for prices and financial statements
- [ ] Financial metrics calculator (EPS, BVPS, ROE, etc.)
- [ ] Point-in-time data alignment pipeline
- [ ] Single-asset backtesting engine
- [ ] Command-line interface

### Phase 2: Screener & Portfolio Logic (CLI) 🚧
- [ ] Multi-stock data processing
- [ ] Dynamic screener with flexible rules
- [ ] Portfolio-level backtesting
- [ ] Rebalancing strategies

### Phase 3: Web Interface 🔮
- [ ] FastAPI backend development
- [ ] Celery background job processing
- [ ] Streamlit dashboard creation
- [ ] Real-time progress tracking

### Phase 4: DevOps & Deployment 🔮
- [ ] Docker containerization
- [ ] Production deployment setup
- [ ] CI/CD pipeline implementation
- [ ] Performance optimization

## 💡 Key Innovation: Point-in-Time Data Accuracy

Strategy Forge's core strength lies in its sophisticated data preparation pipeline that eliminates lookahead bias:

1. **Fetch Raw Data**: Download daily prices and quarterly financial statements
2. **Calculate Metrics**: Compute financial ratios from raw statement data
3. **Simulate Reporting Lag**: Shift fundamental data by 45 days to simulate real-world delays
4. **Forward-Fill Data**: Propagate last known values for daily calculations
5. **Generate Signals**: Calculate point-in-time ratios for every trading day

This ensures that backtests reflect realistic trading conditions where fundamental data isn't immediately available.

## 🧪 Example Usage

### Basic Single Stock Backtest
```python
from src.data.pipeline import DataPipeline
from src.models.strategies import PERatioStrategy
from src.models.backtester import Backtester

# Initialize components
pipeline = DataPipeline()
strategy = PERatioStrategy(threshold=15)
backtester = Backtester()

# Run backtest
data = pipeline.process_stock("RELIANCE.NS", "2020-01-01", "2023-12-31")
results = backtester.run(data, strategy)

# Display results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

### Portfolio Screening Strategy
```python
from src.models.screener import Screener
from src.models.portfolio import Portfolio

# Define screening rules
screener = Screener()
screener.add_rule("pe_ratio", "<", 20)
screener.add_rule("debt_to_equity", "<", 0.5)
screener.add_rule("roe", ">", 0.15)

# Run portfolio backtest
portfolio = Portfolio(universe=["NIFTY50"])
results = portfolio.backtest(screener, rebalance_freq="quarterly")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📈 Performance Metrics

Strategy Forge calculates comprehensive performance metrics including:

- **Returns**: Total Return, CAGR, Annual/Monthly Returns
- **Risk Metrics**: Volatility, Sharpe Ratio, Sortino Ratio
- **Drawdown**: Maximum Drawdown, Drawdown Duration
- **Portfolio Analysis**: Sector Allocation, Stock Contribution
- **Benchmark Comparison**: Alpha, Beta, Information Ratio

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended as investment advice. Past performance does not guarantee future results. Always conduct your own research and consult with financial professionals before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free access to financial data
- **Pandas/NumPy**: For powerful data manipulation capabilities
- **FastAPI/Streamlit**: For excellent web framework and UI library
- **Quantitative Finance Community**: For inspiration and best practices

## 📞 Contact

**Project Maintainer**: [Your Name]
- GitHub: [@srinidhim3](https://github.com/srinidhim3)
- Email: [srinidhim.kattimani@gmail.com]

## 🔗 Links

- [Documentation](docs/)
- [Issue Tracker](https://github.com/srinidhim3/strategy-forge/issues)
- [Project Board](https://github.com/srinidhim3/strategy-forge/projects)
- [Wiki](https://github.com/srinidhim3/strategy-forge/wiki)

---

**⭐ Star this repository if you find it useful!**
