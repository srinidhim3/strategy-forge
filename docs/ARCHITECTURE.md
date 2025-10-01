# Strategy Forge System Architecture

## 📊 High-Level Data Flow

```mermaid
graph TD
    A[Yahoo Finance API] --> B[Price Fetcher]
    A --> C[Financial Fetcher]
    
    B --> D[Daily OHLCV Data]
    C --> E[Quarterly Financial Statements]
    
    E --> F[Financial Calculator]
    F --> G[Financial Metrics<br/>ROE, ROA, EPS, etc.]
    
    D --> H[Data Aligner]
    G --> H
    H --> I[Point-in-Time Aligned Dataset]
    
    I --> J[Ratio Calculator]
    J --> K[Daily P/E, P/B, PEG Ratios]
    
    K --> L[Strategy Engine]
    L --> M[Buy/Sell Signals]
    
    M --> N[Backtester]
    N --> O[Portfolio Performance]
    
    O --> P[Results Dashboard]
```

## 🔄 Data Alignment Challenge (The Core Problem)

```mermaid
timeline
    title Point-in-Time Data Alignment Challenge
    
    section Q1 2024 (Jan-Mar)
        Q1 Financial Data Published (Apr 15) : Q1 metrics available
        
    section Q2 2024 (Apr-Jun)  
        Q2 Financial Data Published (Aug 15) : Q2 metrics available
        
    section Q3 2024 (Jul-Sep)
        Q3 Financial Data Published (Nov 15) : Q3 metrics available
        
    section Daily Trading
        Jan 1 - Apr 14 : Use Previous Quarter Data
        Apr 15 - Aug 14 : Use Q1 Data (Forward Fill)
        Aug 15 - Nov 14 : Use Q2 Data (Forward Fill)
        Nov 15 onwards : Use Q3 Data (Forward Fill)
```

## 🏗️ Module Architecture

```mermaid
graph LR
    subgraph "src/data/"
        A[price_fetcher.py]
        B[financial_fetcher.py]
        C[data_aligner.py]
        D[pipeline.py]
    end
    
    subgraph "src/models/"
        E[financial_calculator.py]
        F[ratio_calculator.py]
        G[strategies.py]
        H[backtester.py]
        I[portfolio.py]
    end
    
    subgraph "src/utils/"
        J[performance.py]
        K[helpers.py]
    end
    
    A --> C
    B --> E
    E --> C
    C --> F
    F --> G
    G --> H
    H --> J
```

## 📈 Example: Data Transformation Pipeline

```mermaid
flowchart LR
    A["Raw Price Data<br/>📅 Daily OHLCV"] --> B["Raw Financial Data<br/>📊 Quarterly Statements"]
    
    B --> C["Financial Calculator<br/>🧮 ROE, ROA, EPS"]
    
    C --> D["Data Aligner<br/>⏰ Apply 45-day lag<br/>📈 Forward fill gaps"]
    A --> D
    
    D --> E["Aligned Dataset<br/>✅ Point-in-time accurate"]
    
    E --> F["Ratio Calculator<br/>📊 Daily P/E, P/B ratios"]
    
    F --> G["Strategy Signals<br/>🎯 Buy/Sell decisions"]
    
    G --> H["Backtester<br/>💰 Portfolio simulation"]
    
    H --> I["Performance Metrics<br/>📈 Returns, Sharpe, Drawdown"]
```