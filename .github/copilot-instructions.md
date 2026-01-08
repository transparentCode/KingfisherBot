# KingfisherBot Copilot Instructions

## Project Overview
KingfisherBot is a Python-based crypto trading and analysis bot. It uses a modular architecture to manage market data, calculate technical indicators, and execute trading strategies. The system is built on **Flask** (web/API), **TimescaleDB** (historical data), and **Redis** (real-time state).

## Architecture & Core Components

### 1. Service Layer (`app/services/`)
- **MarketService (`market_service.py`):** The central orchestrator. It runs in a background thread (started in `run.py`), managing data ingestion, pipeline execution, and persistence.
- **IndicatorRegistry (`indicator_registers.py`):** Singleton registry for all technical indicators. New indicators *must* be registered here to be usable via config.
- **ConfigurationManager (`config/asset_indicator_config.py`):** Singleton that manages asset-specific configurations and runtime overrides.

### 2. Domain Logic
- **Indicators (`app/indicators/`):**
  - Must inherit from `BaseIndicatorInterface`.
  - Must implement `calculate(data: pd.DataFrame) -> pd.DataFrame` (usually appends columns).
  - Must implement `plot(data: pd.DataFrame)`.
  - **Pattern:** Use `app.utils.price_utils.get_price_source_data` to handle flexible input columns (open/high/low/close).
- **Strategies (`app/strategy/`):**
  - Must inherit from `BaseStrategyInterface`.
  - Must implement `initialize`, `execute`, and `plot`.
  - `execute` returns a DataFrame with signals.

### 3. Data & Infrastructure
- **Database:** TimescaleDB (Postgres) for candle storage. Handled by `app/db/db_handler.py`.
- **Cache/State:** Redis for real-time data and inter-service communication. Handled by `app/db/redis_handler.py`.
- **Web:** Flask blueprints in `app/routes/`. SocketIO for real-time frontend updates.

## Developer Workflows

### Adding a New Indicator
1.  **Create Class:** Add a new file in `app/indicators/` (e.g., `my_indicator.py`).
2.  **Inherit:** Subclass `BaseIndicatorInterface`.
3.  **Implement:** Define `calculate` (using VectorBT or TA-Lib where possible) and `plot`.
4.  **Register:** Add the indicator to `IndicatorRegistry.register_default_indicators` in `run.py` or `app/services/indicator_registers.py`.

### Running the Project
- **Docker (Recommended):**
  ```bash
  docker-compose up --build
  ```
  *Note: The Dockerfile compiles TA-Lib from source, which can take time.*
- **Local Development:**
  - Requires `poetry` and system-level `ta-lib` installed.
  - Run: `python run.py`

### Configuration
- **Global Config:** `config/configs.py` (Flask settings).
- **Asset Config:** `assets_config.yaml` (defines which indicators run for which assets).
- **Environment:** `.env` file for secrets (API keys, DB URLs).

## Coding Conventions
- **DataFrames:** `pandas` is the primary data structure. Ensure index is DatetimeIndex for time-series operations.
- **Async/Sync:** The web layer (Flask) is synchronous. The core logic (`MarketService`) uses `asyncio`. Be careful when bridging these contexts.
- **Type Hinting:** Use Python type hints extensively (`typing.List`, `typing.Dict`, `pd.DataFrame`).
- **Logging:** Use `logging.getLogger("app")`. Do not use `print`.

## Key Files
- `run.py`: Entry point. Starts Flask and MarketService.
- `app/services/market_service.py`: Core logic loop.
- `app/indicators/BaseIndicatorInterface.py`: Contract for indicators.
- `docker-compose.yml`: Infrastructure definition.
