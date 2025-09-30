# Project Design Document: Strategy Forge (Version 2.0)

---

## 1. Vision

To create a powerful, user-friendly web application that allows users to design, backtest, and analyze quantitative stock trading strategies based on custom, rule-based screeners using accurate, point-in-time historical data.

---

## 2. Core Features (Unchanged)

* **Dynamic Screener Builder:** A web interface to define filtering rules.
* **Portfolio Backtesting Engine:** Simulates strategies over historical data.
* **Performance Analytics Dashboard:** Visualizes results with key metrics and charts.
* **Job-Based Architecture:** Uses background workers for long-running backtests.

---

## 3. Technology Stack

* **Frontend:** `Streamlit`
* **Backend API:** `FastAPI`
* **Background Jobs / Orchestration:** `Celery` with `Redis`
* **Database:** `PostgreSQL`
* **DevOps:** `Docker`, `Docker Compose`, and `GitHub Actions`
* **Primary Data Source:**
    * **Provider:** Yahoo Finance, accessed via the **`yfinance` Python library**.
    * **Technical Data:** Daily historical price and volume data (Open, High, Low, Close, Volume).
    * **Fundamental Data:** Historical quarterly and annual financial statements (Income Statement, Balance Sheet, Cash Flow). This raw statement data will be used to calculate financial ratios.

---

## 4. High-Level Architecture & Data Strategy

The core of this project's accuracy relies on a sophisticated data preparation pipeline that correctly handles point-in-time fundamental data.



**The Data Preparation Pipeline:**
This pipeline must be executed before any backtest. Its purpose is to create a single, unified dataset for each stock that correctly aligns high-frequency (daily) price data with low-frequency (quarterly) fundamental data.

1.  **Fetch Price Data:** Download the complete daily price/volume history for a target stock into a pandas DataFrame.
2.  **Fetch Financial Statements:** Download the quarterly financial statements. `yfinance` provides this data with period-end dates (e.g., '2025-09-30').
3.  **Calculate Key Metrics:** Process the raw statement data to calculate essential metrics for each reporting period. This includes:
    * Earnings Per Share (EPS) = Net Income / Shares Outstanding
    * Book Value Per Share (BVPS) = Total Stockholder Equity / Shares Outstanding
    * And any other required metrics.
4.  **Align Data by Date:** Merge the daily price DataFrame with the calculated quarterly metrics DataFrame. After the merge, the metric columns will only have values on the period-end dates and `NaN` everywhere else.
5.  **Simulate Reporting Lag & Forward-Fill:** To avoid lookahead bias, a crucial step is to simulate the delay between a quarter's end and the public reporting date.
    * A simple approach is to shift the fundamental data forward by a fixed period (e.g., 45 days).
    * After shifting, use pandas' **`ffill()` (forward-fill)** method on the metric columns. This propagates the last known value forward for each day until a new financial report becomes available.
6.  **Calculate Ratios:** With the forward-filled data, you can now calculate point-in-time ratios for **every single day** in your history (e.g., `P/E Ratio = Daily Close Price / Daily Forward-Filled EPS`). The final output is a rich DataFrame ready for the backtesting engine.

---

## 5. Development Roadmap (Revised & Detailed)

### Phase 1: The Core Data & Backtesting Engine (CLI)
* **Goal:** Build a robust command-line tool that can process data accurately and run a backtest on a single asset.
* **Tasks:**
    1.  Develop a `Data Fetcher` module in `data_fetcher.py` to get both daily prices and quarterly financial statements from `yfinance`.
    2.  Create a `Financial Calculator` module to process raw statement data and compute metrics like EPS and BVPS.
    3.  Implement the **Data Alignment Pipeline**. This is a critical script that merges, shifts (for reporting lag), and forward-fills the data to produce an analysis-ready DataFrame.
    4.  Build the `Backtesting Engine` in `backtester.py` to consume this rich DataFrame and simulate a strategy (e.g., buy when historical P/E < 15).
    5.  Create a `main.py` runner to execute the full pipeline: Fetch -> Process -> Backtest -> Print Results.

### Phase 2: Screener & Portfolio Logic (CLI)
* **Goal:** Expand the engine to handle a universe of stocks and a dynamic portfolio strategy.
* **Tasks:**
    1.  Modify the `Data Fetcher` and `Processing Pipeline` to handle a list of multiple stocks efficiently.
    2.  Build the `Screener` module. This will run on a specific date and filter the universe of stocks based on their calculated point-in-time fundamental and technical data for that day.
    3.  Refactor the `Backtesting Engine` to manage a portfolio, handle rebalancing based on screener output, and track portfolio-level metrics.

*(Phases 3 and 4 remain conceptually the same but will build upon this more robust foundation)*

### Phase 3: The Web Interface
* **Goal:** Wrap the engine in a user-friendly web application.

### Phase 4: DevOps & Deployment
* **Goal:** Containerize and deploy the application.