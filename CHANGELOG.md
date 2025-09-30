# Changelog

All notable changes to Strategy Forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 1: Core Data & Backtesting Engine (CLI)
#### Added
- Initial project setup with comprehensive documentation
- Python virtual environment configuration
- Core dependencies installation (pandas, yfinance, numpy, matplotlib, seaborn)
- Enhanced .gitignore with Python and project-specific exclusions
- Git repository initialization and GitHub integration
- Project structure documentation
- Contributing guidelines

#### Infrastructure
- Virtual environment setup in `venv/`
- Requirements.txt with comprehensive dependency list
- Git repository connected to GitHub origin
- Comprehensive project documentation

#### Development Environment
- Python 3.12+ support
- FastAPI, Streamlit, Celery, Redis, PostgreSQL dependencies
- Development-ready environment with all core packages

### Planned Features

#### Phase 1 Remaining Tasks
- [ ] Data fetcher modules (price_fetcher.py, financial_fetcher.py)
- [ ] Financial metrics calculator (EPS, BVPS, ROE, etc.)
- [ ] Point-in-time data alignment pipeline
- [ ] Single-asset backtesting engine
- [ ] Command-line interface (main.py)

#### Phase 2: Portfolio & Screening
- [ ] Multi-stock data processing capabilities
- [ ] Dynamic screener with flexible rule engine
- [ ] Portfolio-level backtesting with rebalancing
- [ ] Stock universe management (NIFTY 50, custom lists)

#### Phase 3: Web Interface
- [ ] FastAPI backend with async job processing
- [ ] Celery background workers
- [ ] Streamlit interactive dashboard
- [ ] Real-time progress tracking and results visualization

#### Phase 4: DevOps & Deployment
- [ ] Docker containerization
- [ ] Docker Compose multi-service setup
- [ ] GitHub Actions CI/CD pipeline
- [ ] Production deployment configuration

## [0.1.0] - 2025-10-01

### Added
- Initial project repository creation
- Basic project structure setup
- Core documentation framework
- Development environment configuration

### Infrastructure
- GitHub repository: strategy-forge
- Local development workspace setup
- Git version control initialization

---

## Version History

| Version | Date | Phase | Status |
|---------|------|-------|--------|
| 0.1.0 | 2025-10-01 | Setup | ✅ Complete |
| 0.2.0 | TBD | Phase 1 | 🚧 In Progress |
| 0.3.0 | TBD | Phase 2 | 🔮 Planned |
| 0.4.0 | TBD | Phase 3 | 🔮 Planned |
| 1.0.0 | TBD | Phase 4 | 🔮 Planned |

---

### Legend
- ✅ Complete
- 🚧 In Progress  
- 🔮 Planned
- ❌ Cancelled
- 🐛 Bug Fix
- 🔧 Maintenance