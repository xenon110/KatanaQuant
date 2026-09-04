# KatanaQuant - Automated Multi-Agent Stock Trading System

An institutional-grade automated trading system for **US Stocks & ETFs**, designed with a strict safety principle:
> **AI agents propose and analyze. Only deterministic, testable code authorizes an order to be sent to the broker.**

---

## Architecture Pipeline

```
1. Market Data Layer (Live/Historical Bar Cache & Staleness Check)
       │
2. Deterministic Rule Engine (EMA Crossover, RSI Filter, ATR Stops)
       │
3. Multi-Agent Analysis Layer (LLM with Strict Pydantic Schemas)
       │ - Signal / Research Agent
       │ - Market Context & Event Risk Agent
       │ - Adversarial Risk Reviewer Agent
       │ - Volatility-Adjusted Portfolio Sizing Agent
       │ - Anomaly & Operational Health Agent (Kill-Switch Authority)
       │
4. Orchestrator (Coordinates agents & merges into one ProposedTrade)
       │
5. Deterministic Risk Gate (NO AI - Hard Limits & Circuit Breakers)
       │ - Max position size ($ and % NAV)
       │ - Max daily loss circuit breaker ($ and %)
       │ - T+1 Settled vs. Unsettled cash check (GFV prevention)
       │ - Pattern Day Trader (PDT) 5-day rolling counter
       │ - Live broker buying power check
       │ - Market feed staleness timeout
       │
6. Execution Engine & Broker Client (Alpaca Live / Paper / Dry-Run)
       │
7. Reconciliation & Persistence Layer (SQLite Audit Trail & T+1 Ledger)
```

---

## Quickstart

### 1. Installation
```bash
pip install -e .
```

### 2. Configure Environment
Copy `.env.example` to `.env` and configure your credentials:
```bash
cp .env.example .env
```

### 3. CLI Commands

#### **Run Historical Backtest**
```bash
python -m src.main backtest --symbol SPY --days 60 --capital 30000
```

#### **Run Safe Offline DRY-RUN Simulation**
```bash
python -m src.main dry-run --symbols AAPL,MSFT,NVDA,SPY --bars-count 50
```

#### **Run Paper Trading via Alpaca API**
```bash
python -m src.main live --paper
```

#### **Run Live Real Money Trading (Requires Explicit Confirmation)**
```bash
python -m src.main live --live --confirm-real-money
```

#### **Trigger Emergency Kill Switch**
```bash
python -m src.main kill-switch
```

---

## Project Structure
- `src/core/`: Domain models, enums, Pydantic schemas.
- `src/config/`: System settings & configuration.
- `src/data/`: Technical indicators & market data clients.
- `src/strategy/`: Deterministic quantitative strategy implementations.
- `src/backtesting/`: Event-driven backtesting engine with performance analytics.
- `src/risk/`: Deterministic Risk Gate (100% testable, zero AI).
- `src/agents/`: Specialized LLM agents (Signal, Context, Risk, Sizing, Anomaly).
- `src/orchestrator/`: Agent synthesis pipeline.
- `src/reconciliation/`: T+1 settlement ledger & PDT tracker.
- `src/execution/`: Dry-run & Alpaca broker integrations.
- `src/storage/`: SQLite database persistence & audit logging.
- `tests/`: Comprehensive test suite for risk gate, PDT, settlement, and strategy.
