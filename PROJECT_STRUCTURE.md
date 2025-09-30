# Strategy Forge - Project Structure

## Directory Layout

```
strategy-forge/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── data/                    # Data fetching and processing
│   │   ├── __init__.py          # Data module initialization
│   │   ├── price_fetcher.py     # Daily OHLCV data retrieval (Phase 1)
│   │   ├── financial_fetcher.py # Quarterly financial statements (Phase 1)
│   │   ├── data_aligner.py      # Point-in-time data alignment (Phase 1)
│   │   ├── pipeline.py          # Data processing orchestration (Phase 1)
│   │   └── universe.py          # Stock universe management (Phase 2)
│   ├── models/                  # Core business logic
│   │   ├── __init__.py          # Models module initialization
│   │   ├── financial_calculator.py  # Financial metrics computation (Phase 1)
│   │   ├── ratio_calculator.py      # Point-in-time ratio calculations (Phase 1)
│   │   ├── strategies.py            # Trading strategy implementations (Phase 1)
│   │   ├── backtester.py            # Backtesting engine (Phase 1)
│   │   ├── screener.py              # Stock screening logic (Phase 2)
│   │   └── portfolio.py             # Portfolio management (Phase 2)
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py          # Utils module initialization
│   │   └── performance.py       # Performance metrics calculation (Phase 1)
│   ├── api/                     # FastAPI web API (Phase 3)
│   │   ├── main.py              # API entry point
│   │   ├── models.py            # Pydantic data models
│   │   ├── tasks.py             # Celery background tasks
│   │   └── celery_app.py        # Celery configuration
│   └── ui/                      # Streamlit web interface (Phase 3)
│       └── app.py               # Web UI entry point
├── tests/                       # Test suite
│   ├── __init__.py              # Test module initialization
│   ├── test_data/               # Data module tests
│   ├── test_models/             # Models module tests
│   ├── test_utils/              # Utils module tests
│   ├── test_integration/        # Integration tests
│   ├── fixtures/                # Test data and fixtures
│   └── conftest.py              # Pytest configuration
├── data/                        # Downloaded data cache (gitignored)
├── reports/                     # Generated reports (gitignored)
├── docs/                        # Additional documentation
├── .gitignore                   # Git exclusions
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
├── docker-compose.yml           # Docker services configuration
├── Dockerfile                   # Container definition
├── main.py                      # CLI entry point (Phase 1)
├── README.md                    # Project overview
├── DESIGN_DOCUMENT.md           # Technical design document
├── CONTRIBUTING.md              # Contributing guidelines
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT license
└── kanpilot.toml               # Project management configuration
```

## Module Responsibilities

### src/data/
- **price_fetcher.py**: Download daily OHLCV data from Yahoo Finance
- **financial_fetcher.py**: Download quarterly financial statements
- **data_aligner.py**: Merge and align data with proper reporting lag simulation
- **pipeline.py**: Orchestrate complete data processing workflow
- **universe.py**: Manage stock lists (NIFTY 50, custom universes)

### src/models/
- **financial_calculator.py**: Calculate EPS, BVPS, ROE, ROA, etc.
- **ratio_calculator.py**: Calculate point-in-time P/E, P/B, PEG ratios
- **strategies.py**: Trading strategy implementations (P/E threshold, moving averages)
- **backtester.py**: Single and multi-asset backtesting engine
- **screener.py**: Flexible rule-based stock screening
- **portfolio.py**: Portfolio management and rebalancing logic

### src/utils/
- **performance.py**: Calculate Sharpe ratio, max drawdown, CAGR, volatility

### tests/
- Comprehensive test coverage for all modules
- Integration tests for complete workflows
- Test fixtures and mock data

## Development Phases

### ✅ Phase 0: Project Setup (Complete)
- [x] Repository creation and documentation
- [x] Virtual environment and dependencies
- [x] Project structure and package initialization

### 🚧 Phase 1: Core Engine (Current)
- [ ] Data fetching modules
- [ ] Financial calculations  
- [ ] Single-asset backtesting
- [ ] CLI interface

### 🔮 Phase 2: Portfolio Logic
- [ ] Multi-asset processing
- [ ] Dynamic screening
- [ ] Portfolio backtesting

### 🔮 Phase 3: Web Interface
- [ ] FastAPI backend
- [ ] Streamlit frontend
- [ ] Background job processing

### 🔮 Phase 4: DevOps
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Production deployment

## Next Steps

The project structure is now ready for Phase 1 development. The next tasks are:
1. Implement data fetching modules (price_fetcher.py, financial_fetcher.py)
2. Create financial metrics calculator
3. Build data alignment pipeline
4. Develop backtesting engine
5. Create CLI interface