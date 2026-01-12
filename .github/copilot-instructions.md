# KingfisherBot Copilot Instructions

## Project Overview
KingfisherBot is a Python-based crypto trading system with a modular architecture for market analysis, signal generation, and automated execution. It features a Svelte-based frontend for real-time monitoring and is designed for scalability with TimescaleDB and Redis.

## 🏗 System Architecture

### 1. Core Services (`app/services/`)
- **MarketService (`market_service.py`)**: The central hub. Manages `WebSocketListenerService` (data ingestion), `IndicatorCalcService` (computation), and `ExecutionRouter`.
- **BotBrain (`app/brain/bot_brain.py`)**: The decision engine.
  - **FeatureExtractor**: Retrieves/computes market states (Regimes) and tactical indicators.
  - **MTFConfluenceEngine**: Analyzes cross-timeframe alignment.
  - **Logic**: Implements high-level "Playbooks" (e.g., `TREND_PULLBACK`, `MEAN_REVERSION`).
- **Orchestration (`app/orchestration/`)**:
  - `IndicatorPipeline`: Runs a sequence of **Orchestrators** (e.g., `TAOrchestrator`, `RegimeOrchestrator`).
  - Decouples calculation scheduling from business logic.

### 2. Execution & Risk (`app/execution/`, `app/risk/`)
- **ExecutionRouter**: Routes orders to `BinanceExecutionService` (live) or `PaperExecutionService` (sim).
- **CoinClustering**: Manages asset correlation to prevent concentrated risk.
- **RiskManager**: Handles position sizing and safety checks.

### 3. Data Flow
1.  **Ingestion**: `BinanceConnector` -> `Redis` (Hot path) & `TimescaleDB` (Cold storage).
2.  **Processing**: `MarketService` triggers `IndicatorPipeline` on new candles/schedules.
3.  **Broadcasting**: `LiveDataManager` (`app/routes/websocket_routes.py`) pushes updates to Frontend via SocketIO.

### 4. Frontend (`frontend/`)
- **Stack**: Svelte + Vite.
- **Communication**: WebSockets (SocketIO) for real-time candle/signal data.
- **Status**: Visualizes "Regimes", "Signals", and active "Trades".

## 🛠 Developer Workflows

### Environment Setup
- **Python**: Requires `poetry`. Note specific constraints: `pandas==1.5.3`, `numpy<1.24` (for `vectorbt`).
- **Run**: `python run.py` (Starts Flask + MarketService).
- **Docker**: `docker-compose up` (Includes Redis, TimescaleDB, App, Frontend).

### Adding a New Indicator
1.  **Implements**: Create class in `app/indicators/`, inheriting `BaseIndicatorInterface`.
2.  **Register**: Add to `IndicatorRegistry` in `app/services/indicator_registers.py`.
3.  **Config**: Enable in `assets_config.yaml` to be picked up by `TAOrchestrator`.

### Backend <-> Frontend Integration
- **Backend emits**: `socketio.emit('candle_update', ...)` in `app/routes/websocket_routes.py`.
- **Frontend subscribes**: Svelte stores/components listen to SocketIO events.

## ⚠️ Conventions & Patterns

- **Async/Sync**:
    - **Flask Routes**: Synchronous.
    - **MarketService/IO**: Asynchronous (`asyncio`).
    - *Caution*: Be careful when calling async code from Flask routes (use `run_coroutine_threadsafe` or similar bridges).
- **DataFrames**:
    - Use `pandas` for all time-series data.
    - **Index**: Must be `DatetimeIndex`.
    - **VectorBT**: Used for heavy vector calculations (`app/indicators/`).
- **Configuration**:
    - `global_config.yaml`: System settings.
    - `assets_config.yaml`: Per-asset indicator rules (active timeframes, params).
    - `ConfigurationManager`: Handles hot-reloading of configs.
- **Logging**: Use `logging.getLogger("app")`. Never use `print()`.

## 📂 Key Directories
- `app/brain/`: Decision logic (Signals, Confluence).
- `app/orchestration/`: Pipelines for data processing.
- `app/indicators/`: Mathematical indicator logic.
- `app/execution/`: Exchange integration & order routing.
- `frontend/src/`: Svelte UI source.
