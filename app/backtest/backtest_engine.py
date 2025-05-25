from typing import Optional, Dict, AnyStr
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from app.strategy.base_strategy_interface import BaseStrategyInterface
import pandas as pd
import vectorbt as vbt

class VectorBTBacktest:
    """
    This class is responsible for running backtests using vectorbt.
    """

    def __init__(self,
                 strategy: BaseStrategyInterface,
                 initial_capital: float = 10000,
                 commission: float = 0.07,
                 fee_rate: float = 0.007,
                 slippage: float = 0.05,
                 leverage: float = 20.0,
                 timeframe: str = "5m",
                 direction: str = "both"):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.timeframe = timeframe
        self.direction = direction
        self.portfolio = None
        self.results = None

    def run_backtest(self, data: Optional[pd.DataFrame] = None) -> vbt.Portfolio:
        """
        Run backtest using vectorbt.

        Args:
            data: DataFrame containing OHLCV data and strategy signals.
                  If None, will use the strategy's data.

        Returns:
            Portfolio object containing backtest results.
        """
        if data is None:
            if self.strategy.data is None:
                raise ValueError("Data must be provided for backtesting.")
            data = self.strategy.data

        # Configure trade size and fees
        fees = self.fee_rate
        slippage = self.slippage

        # Create portfolio object - fix the leverage parameter
        self.portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=data['signal'] == 1,
            exits=data['signal'] == -1,
            init_cash=self.initial_capital,
            fees=fees,
            slippage=slippage,
            freq=data.index.inferred_freq,
            # Remove 'leverage' from here and apply it below
        )

        # Apply leverage to the portfolio
        # For vectorbt, you typically manage leverage by adjusting position sizes
        # or by post-processing the returns

        # You can access leveraged metrics like this:
        self.leveraged_returns = self.portfolio.returns() * self.leverage

        return self.portfolio

    def get_performance_metrics(self) -> Dict[str, AnyStr]:
        """
        Get key performance metrics from the backtest.

        :return: Dictionary of performance metrics
        """
        if self.portfolio is None:
            raise ValueError("Backtest not run yet. Call run() method first.")

        # Create a safer metrics dictionary that checks for attribute existence
        metrics = {
            'total_return': float(self.portfolio.total_return()),
            'max_drawdown': float(self.portfolio.max_drawdown()),
        }

        # Add optional metrics that might not be available in all vectorbt versions
        if hasattr(self.portfolio, 'cagr'):
            metrics['cagr'] = float(self.portfolio.cagr())
        else:
            metrics['cagr'] = 0.0

        if hasattr(self.portfolio, 'sharpe_ratio'):
            metrics['sharpe_ratio'] = float(self.portfolio.sharpe_ratio())
        else:
            metrics['sharpe_ratio'] = 0.0

        if hasattr(self.portfolio, 'sortino_ratio'):
            metrics['sortino_ratio'] = float(self.portfolio.sortino_ratio())
        else:
            metrics['sortino_ratio'] = 0.0

        # Handle trades metrics
        if hasattr(self.portfolio, 'trades') and len(self.portfolio.trades) > 0:
            metrics['win_rate'] = float(self.portfolio.trades.win_rate())
            metrics['profit_factor'] = float(self.portfolio.trades.profit_factor()) if hasattr(self.portfolio.trades,
                                                                                               'profit_factor') else 0.0
            metrics['expectancy'] = float(self.portfolio.trades.expectancy()) if hasattr(self.portfolio.trades,
                                                                                         'expectancy') else 0.0
            metrics['avg_trade_duration'] = str(self.portfolio.trades.avg_duration()) if hasattr(self.portfolio.trades,
                                                                                                 'avg_duration') else "N/A"
            metrics['num_trades'] = len(self.portfolio.trades)
        else:
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
            metrics['expectancy'] = 0.0
            metrics['avg_trade_duration'] = "N/A"
            metrics['num_trades'] = 0

        return metrics

    def plot_results(self) -> go.Figure:
        """
        Generate an interactive plot of backtest results using Plotly.

        :return: Plotly Figure object
        """
        if self.portfolio is None:
            raise ValueError("Backtest not run yet. Call run_backtest() method first.")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=["Portfolio Value", "Drawdown", "Trade Signals"]
        )

        # Get the timestamp index from portfolio
        timestamps = self.portfolio.wrapper.index

        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.portfolio.value(),
                mode='lines',
                line=dict(color='blue', width=2),
                name='Portfolio Value'
            ),
            row=1, col=1
        )

        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.portfolio.drawdown() * 100,  # Convert to percentage
                mode='lines',
                line=dict(color='red', width=1),
                fill='tozeroy',
                name='Drawdown %'
            ),
            row=2, col=1
        )

        # Add trade markers if available
        if hasattr(self.portfolio, 'trades') and len(self.portfolio.trades) > 0:
            # Extract entry points
            entries = self.portfolio.trades.records['entry_idx']
            entry_prices = self.portfolio.trades.records['entry_price']

            # Extract exit points
            exits = self.portfolio.trades.records['exit_idx']
            exit_prices = self.portfolio.trades.records['exit_price']

            # Add entry markers
            fig.add_trace(
                go.Scatter(
                    x=[timestamps[i] for i in entries],
                    y=entry_prices,
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name='Entries'
                ),
                row=3, col=1
            )

            # Add exit markers
            fig.add_trace(
                go.Scatter(
                    x=[timestamps[i] for i in exits if i < len(timestamps)],  # Ensure index is valid
                    y=exit_prices,
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Exits'
                ),
                row=3, col=1
            )

        # Add performance metrics as an annotation
        metrics = self.get_performance_metrics()
        annotation_text = (
            f"Total Return: {metrics['total_return']:.2%}<br>"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}<br>"
            f"Win Rate: {metrics.get('win_rate', 0):.2%}<br>"
            f"Trades: {metrics.get('num_trades', 0)}"
        )

        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.01, y=0.99,
            text=annotation_text,
            showarrow=False,
            font=dict(size=12),
            align='left',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )

        # Update layout
        fig.update_layout(
            title=f"Backtest Results: {self.strategy.strategy_name if hasattr(self.strategy, 'strategy_name') else 'Strategy'}",
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Ensure y-axes have appropriate titles
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=3, col=1)

        return fig