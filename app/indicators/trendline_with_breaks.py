import logging

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from app.indicators.BaseIndicatorInterface import BaseIndicatorInterface

class TrendLineWithBreaks(BaseIndicatorInterface):
    """
    TrendLine with break indicator - Improved version closer to PineScript logic.
    """

    def __init__(self, name="TrendLineWithBreaks",
                 length: int = 14,
                 multiplier: float = 1.0,
                 calculate_method: str = 'atr',
                 source: str = 'close', # Note: source isn't explicitly used in PineScript slope calcs but kept for potential future use
                 backpaint: bool = True, # Added backpaint parameter
                 upper_color: str = 'teal', # Matched PineScript default
                 down_color: str = 'red',
                 show_extension: bool = True, # Matched PineScript default
                 **kwargs):
        """
        Initialize the TrendLineWithBreaks indicator.
        :param name: Name of the indicator
        :param length: Swing Detection Lookback
        :param multiplier: Slope multiplier
        :param calculate_method: Method to calculate the slope ('atr', 'stdev', 'linreg')
        :param source: Source data (e.g., 'close') - primarily for stdev/linreg if adapted
        :param backpaint: Backpaint offset for plotting
        :param upper_color: Color for the upper trend line
        :param down_color: Color for the lower trend line
        :param show_extension: Whether to show extended trend lines
        """
        super().__init__(name, input_columns=['open', 'high', 'low', 'close'], **kwargs)
        self.name = name
        self.length = length
        self.multiplier = multiplier
        self.calculate_method = calculate_method.lower() # Ensure lowercase
        self.source = source
        self.backpaint = backpaint
        self.upper_color = upper_color
        self.down_color = down_color
        self.show_extension = show_extension
        self.logger = logging.getLogger("app")

    def _detect_pivot_high(self, data, length):
        """Detect pivot highs using rolling window logic similar to PineScript"""
        highs = data['high']
        # Check if current high is the max in the lookback window [i-length, i+length]
        # Ensure sufficient data points before and after
        pivot_highs = highs.rolling(window=2*length+1, center=True).max() == highs
        # Filter out pivots too close to the start/end where the window is incomplete
        # PineScript's ta.pivothigh effectively requires 'length' bars on both sides fully formed
        valid_range = np.concatenate(([False]*length, [True]*(len(highs)-2*length), [False]*length))
        detected_pivots = np.where(pivot_highs & valid_range, highs, 0) # Use 0 for no pivot
        return detected_pivots

    def _detect_pivot_low(self, data, length):
        """Detect pivot lows using rolling window logic similar to PineScript"""
        lows = data['low']
        pivot_lows = lows.rolling(window=2*length+1, center=True).min() == lows
        valid_range = np.concatenate(([False]*length, [True]*(len(lows)-2*length), [False]*length))
        detected_pivots = np.where(pivot_lows & valid_range, lows, 0) # Use 0 for no pivot
        return detected_pivots

    def _calculate_slope(self, data, calculate_method, length, multiplier, source_col):
        """Calculate slope based on the specified method"""
        if calculate_method == 'atr':
            high = data['high']
            low = data['low']
            close = data['close']
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=length, min_periods=length).mean()
            slope = atr / length * multiplier
            # Handle potential initial NaNs if needed, e.g., backfill or use a default small slope
            slope = slope.fillna(method='bfill').fillna(0.0001) # Example fill
            return slope

        elif calculate_method == 'stdev':
            stdev = data[source_col].rolling(window=length, min_periods=length).std()
            slope = stdev / length * multiplier
            slope = slope.fillna(method='bfill').fillna(0.0001) # Example fill
            return slope

        elif calculate_method == 'linreg':
            # PineScript: math.abs(ta.sma(src * n, length) - ta.sma(src, length) * ta.sma(n, length)) / ta.variance(n, length) / 2 * mult
            # This is equivalent to abs(slope_of_linreg) / 2 * mult
            src = data[source_col]
            n_series = pd.Series(np.arange(len(src)), index=src.index)

            sma_src_n = (src * n_series).rolling(window=length, min_periods=length).mean()
            sma_src = src.rolling(window=length, min_periods=length).mean()
            sma_n = n_series.rolling(window=length, min_periods=length).mean()
            # Use unbiased variance (ddof=1) by default in pandas, PineScript might use population (ddof=0)
            # For simplicity, using pandas default. Adjust ddof if needed for exact match.
            var_n = n_series.rolling(window=length, min_periods=length).var()

            # Avoid division by zero or NaN variance
            var_n = var_n.replace(0, np.nan)
            slope_val = (sma_src_n - sma_src * sma_n) / var_n
            slope = abs(slope_val) / 2 * multiplier
            slope = slope.fillna(method='bfill').fillna(0.0001) # Example fill
            return slope
        else:
            raise ValueError(f"Invalid calculation method: {calculate_method}")

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate the trendlines and breakouts, closer to PineScript logic.

        :param data: DataFrame with price data
        :return: DataFrame with added indicator columns
        """

        self.logger.info(f"TrendLineWithBreaks received df_input. Shape: {data.shape}")
        self.logger.info(f"Input dtypes:\n{data.dtypes.to_string()}")
        self.logger.info(f"Input NaNs check:\n{data.isnull().sum().to_string()}")
        self.logger.info(f"Input head:\n{data.head().to_string()}")
        self.logger.info(f"Input tail:\n{data.tail().to_string()}")


        self.logger.info(f"enters the indicator with this df: {data}")
        df = data.copy()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Get parameters merging instance defaults with kwargs
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        length = kwargs.get('length', self.length)
        multiplier = kwargs.get('multiplier', self.multiplier)
        calculate_method = kwargs.get('calculate_method', self.calculate_method)
        source_col = kwargs.get('source', self.source)
        backpaint = kwargs.get('backpaint', self.backpaint)

        # Calculate offset for plotting
        offset = length if backpaint else 0

        # Detect pivot points (using improved rolling logic)
        ph = self._detect_pivot_high(df, length)
        pl = self._detect_pivot_low(df, length)
        # Convert pivot values back to boolean flags (True if pivot exists)
        ph_bool = ph > 0
        pl_bool = pl > 0

        # Calculate slope
        slope = self._calculate_slope(df, calculate_method, length, multiplier, source_col)

        # Initialize arrays
        upper = np.full(len(df), 0.0) # PineScript var starts at 0.
        lower = np.full(len(df), 0.0)
        slope_ph = np.full(len(df), np.nan)
        slope_pl = np.full(len(df), np.nan)
        upos = np.zeros(len(df)) # Equivalent to PineScript upos state
        dnos = np.zeros(len(df)) # Equivalent to PineScript dnos state

        # --- Iterative Calculation (mimicking PineScript's bar-by-bar state) ---
        # Vectorization is tricky due to the stateful nature (depending on previous values)
        for i in range(1, len(df)): # Start from 1 to use i-1 indices

            # Store previous values needed for breakout check (before potential updates)
            prev_upper = upper[i-1]
            prev_lower = lower[i-1]
            prev_slope_ph = slope_ph[i-1]
            prev_slope_pl = slope_pl[i-1]

            # Update slopes based on pivots (use current bar's slope if pivot detected)
            # Handle potential NaN slope at the beginning
            current_slope = slope[i] if not np.isnan(slope[i]) else (slope_ph[i-1] if not np.isnan(slope_ph[i-1]) else 0.0001)

            if ph_bool[i]:
                slope_ph[i] = current_slope
            else:
                # Carry forward the last known slope_ph
                slope_ph[i] = slope_ph[i-1] # Will be NaN initially, handle below

            if pl_bool[i]:
                slope_pl[i] = current_slope
            else:
                # Carry forward the last known slope_pl
                slope_pl[i] = slope_pl[i-1] # Will be NaN initially, handle below

            # Handle initial NaNs for slopes (use current_slope if previous is NaN)
            if np.isnan(slope_ph[i]): slope_ph[i] = current_slope
            if np.isnan(slope_pl[i]): slope_pl[i] = current_slope

            # Calculate upper trendline
            if ph_bool[i]:
                upper[i] = ph[i] # Reset to pivot high value
            else:
                # Continue trendline: prev_upper - prev_slope_ph
                upper[i] = upper[i-1] - slope_ph[i-1] # Uses the slope from the *previous* bar that was active

            # Calculate lower trendline
            if pl_bool[i]:
                lower[i] = pl[i] # Reset to pivot low value
            else:
                 # Continue trendline: prev_lower + prev_slope_pl
                lower[i] = lower[i-1] + slope_pl[i-1] # Uses the slope from the *previous* bar that was active

            # --- Breakout Calculation (Matching PineScript Logic) ---
            # Uses values *before* potential updates on the current bar
            # Check for Upward Break (close breaks above projected past lower trendline)
            if pl_bool[i]: # Reset on pivot low
                upos[i] = 0
            else:
                 # Check condition: close > prev_lower + prev_slope_pl * length
                 # Ensure prev_slope_pl is not NaN before multiplying
                 if not np.isnan(prev_slope_pl):
                      break_level = prev_lower + prev_slope_pl * length
                      if df['close'].iloc[i] > break_level:
                          upos[i] = 1
                      else:
                          upos[i] = upos[i-1] # Maintain previous state if no break
                 else:
                     upos[i] = upos[i-1] # Maintain state if slope is undefined


            # Check for Downward Break (close breaks below projected past upper trendline)
            if ph_bool[i]: # Reset on pivot high
                dnos[i] = 0
            else:
                 # Check condition: close < prev_upper - prev_slope_ph * length
                 # Ensure prev_slope_ph is not NaN before multiplying
                 if not np.isnan(prev_slope_ph):
                     break_level = prev_upper - prev_slope_ph * length
                     if df['close'].iloc[i] < break_level:
                          dnos[i] = 1
                     else:
                          dnos[i] = dnos[i-1] # Maintain previous state if no break
                 else:
                    dnos[i] = dnos[i-1] # Maintain state if slope is undefined

        # --- End Iterative Calculation ---

        # Add calculated values to DataFrame
        df['ph'] = ph # Keep original pivot values
        df['pl'] = pl
        df['ph_bool'] = ph_bool # Boolean flags for easier use
        df['pl_bool'] = pl_bool
        df['upper'] = upper
        df['lower'] = lower
        df['slope_ph'] = slope_ph
        df['slope_pl'] = slope_pl

        # Determine plot values based on backpaint setting
        # PineScript: plot(backpaint ? upper : upper - slope_ph * length, ...)
        # Need to calculate the projected value for backpaint=False
        # Requires iterating again or careful shifting, let's calculate directly
        projected_upper = df['upper'] - df['slope_ph'] * length
        projected_lower = df['lower'] + df['slope_pl'] * length

        df['plot_upper'] = np.where(backpaint, df['upper'], projected_upper)
        df['plot_lower'] = np.where(backpaint, df['lower'], projected_lower)

        # Add NaN where pivots occur for plotting breaks (for backpaint=True case)
        # If backpaint=False, the projected value is plotted, which doesn't necessarily touch the pivot
        df['plot_upper_final'] = np.where(df['ph_bool'] & backpaint, np.nan, df['plot_upper'])
        df['plot_lower_final'] = np.where(df['pl_bool'] & backpaint, np.nan, df['plot_lower'])

        # Calculate breakout signals (change in state from 0 to 1)
        df['upos'] = upos
        df['dnos'] = dnos
        df['upbreak_signal'] = ((df['upos'] == 1) & (df['upos'].shift(1) == 0)).astype(int)
        df['downbreak_signal'] = ((df['dnos'] == 1) & (df['dnos'].shift(1) == 0)).astype(int)

        df['offset'] = offset # Store offset for plotting use

        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def plot1(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """
        Plot the trendlines and breakouts with improved logic and backpainting.

        :param data: DataFrame with calculated indicator values
        :return: Plotly figure object
        """
        # Get parameters merging instance defaults with kwargs
        length = kwargs.get('length', self.length)
        upper_color = kwargs.get('upper_color', self.upper_color)
        down_color = kwargs.get('down_color', self.down_color)
        show_extension = kwargs.get('show_extension', self.show_extension)
        backpaint = kwargs.get('backpaint', self.backpaint) # Needed for title/info maybe

        # Check if indicator data exists, otherwise calculate
        required_cols = ['plot_upper_final', 'plot_lower_final', 'upbreak_signal', 'downbreak_signal', 'offset', 'ph', 'pl', 'slope_ph', 'slope_pl']
        if not all(col in data.columns for col in required_cols):
            print("Indicator columns missing, recalculating...") # Debug print
            data = self.calculate(data, **kwargs)

        # Create figure
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        # Extract x-axis data (use index if 'open_time' not present)
        x_axis = data['open_time'] if 'open_time' in data.columns else data.index
        x_axis_plot = x_axis # Use original for candlestick

        # Calculate offset from the data for plotting
        offset = data['offset'].iloc[-1] if 'offset' in data.columns and not data.empty else 0
        # Create shifted x-axis for plotting lines/markers with offset
        # Note: Shifting timestamps directly can be complex. A simpler way for visual offset
        # is to plot against the original index shifted, then format labels.
        # For simplicity here, we'll plot against the original x_axis but note that
        # true timestamp offsetting might require more complex handling depending on x_axis type.
        # If x_axis is just numbers (index), shifting is easier.
        # Let's assume x_axis is compatible with simple plotting for now.
        # We will plot data intended for index `i` at `x_axis[i - offset]`
        x_indices = np.arange(len(x_axis))
        x_axis_shifted_indices = x_indices - offset
        # Ensure indices are within bounds for mapping back to x_axis values
        valid_shifted_indices = (x_axis_shifted_indices >= 0) & (x_axis_shifted_indices < len(x_axis))

        # Add price candlestick (plotted against original x-axis)
        fig.add_trace(
            go.Candlestick(
                x=x_axis_plot,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
        )

        # Prepare data for plotting with offset
        plot_x = x_axis[x_axis_shifted_indices[valid_shifted_indices]]
        plot_upper_y = data['plot_upper_final'].iloc[valid_shifted_indices]
        plot_lower_y = data['plot_lower_final'].iloc[valid_shifted_indices]

        # Plot Upper Trendline (with NaNs for breaks and offset)
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_upper_y,
                line=dict(color=upper_color, width=1.5),
                name="Upper Trendline",
                connectgaps=False # Important: Do not connect across NaNs
            )
        )

        # Plot Lower Trendline (with NaNs for breaks and offset)
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_lower_y,
                line=dict(color=down_color, width=1.5),
                name="Lower Trendline",
                connectgaps=False # Important: Do not connect across NaNs
            )
        )

        # --- Plot Extended Lines ---
        if show_extension and not data.empty:
            last_ph_idx = data.index[data['ph'] > 0].max() if (data['ph'] > 0).any() else None
            last_pl_idx = data.index[data['pl'] > 0].max() if (data['pl'] > 0).any() else None

            # Find corresponding index in the original x_axis
            x_axis_list = list(x_axis)

            if last_ph_idx is not None and last_ph_idx in data.index:
                idx_loc = data.index.get_loc(last_ph_idx)
                if idx_loc >= 0:
                    ph_val = data.loc[last_ph_idx, 'ph']
                    ph_slope = data.loc[last_ph_idx, 'slope_ph']
                    start_idx_shifted = idx_loc - offset

                    if start_idx_shifted >= 0 and start_idx_shifted < len(x_axis_list):
                        x1 = x_axis_list[start_idx_shifted]
                        y1 = ph_val # Line starts at the pivot value
                        # Extend to the end of the chart
                        x2_idx_shifted = len(x_axis_list) - 1
                        if x2_idx_shifted > start_idx_shifted:
                             x2 = x_axis_list[x2_idx_shifted]
                             # Calculate y2 based on slope from pivot
                             # Need number of bars extended (relative to pivot's original position)
                             bars_extended = (len(x_axis_list) - 1) - idx_loc
                             y2 = y1 - ph_slope * bars_extended
                             fig.add_trace(go.Scatter(
                                 x=[x1, x2], y=[y1, y2], mode='lines',
                                 line=dict(color=upper_color, width=1, dash='dash'),
                                 name='Upper Extension', showlegend=False))

            if last_pl_idx is not None and last_pl_idx in data.index:
                 idx_loc = data.index.get_loc(last_pl_idx)
                 if idx_loc >= 0:
                     pl_val = data.loc[last_pl_idx, 'pl']
                     pl_slope = data.loc[last_pl_idx, 'slope_pl']
                     start_idx_shifted = idx_loc - offset

                     if start_idx_shifted >= 0 and start_idx_shifted < len(x_axis_list):
                         x1 = x_axis_list[start_idx_shifted]
                         y1 = pl_val # Line starts at the pivot value
                         # Extend to the end of the chart
                         x2_idx_shifted = len(x_axis_list) - 1
                         if x2_idx_shifted > start_idx_shifted:
                              x2 = x_axis_list[x2_idx_shifted]
                              bars_extended = (len(x_axis_list) - 1) - idx_loc
                              y2 = y1 + pl_slope * bars_extended
                              fig.add_trace(go.Scatter(
                                  x=[x1, x2], y=[y1, y2], mode='lines',
                                  line=dict(color=down_color, width=1, dash='dash'),
                                  name='Lower Extension', showlegend=False))

        # --- Plot Breakout Markers ---
        upbreak_indices = data.index[data['upbreak_signal'] == 1]
        downbreak_indices = data.index[data['downbreak_signal'] == 1]

        if not upbreak_indices.empty:
            # Apply offset to breakout marker positions
            upbreak_plot_indices = [idx - offset for idx in data.index.get_indexer(upbreak_indices)]
            valid_upbreak_plot_indices = [i for i in upbreak_plot_indices if 0 <= i < len(x_axis)]
            upbreak_x = x_axis[valid_upbreak_plot_indices]
            # Get corresponding 'low' values for plotting marker below the bar
            upbreak_y_ref_indices = data.index.get_indexer(upbreak_indices) # Get original indices
            valid_y_ref_indices = [i for i in upbreak_y_ref_indices if i != -1] # Ensure index found
            if valid_y_ref_indices:
                  upbreak_y = data['low'].iloc[valid_y_ref_indices] * 0.995 # Place below low
                  fig.add_trace(go.Scatter(
                      x=upbreak_x, y=upbreak_y, mode='markers+text',
                      marker=dict(color=upper_color, size=10, symbol='triangle-up', line=dict(color='white', width=1)),
                      text='B', textfont=dict(color='white', size=10), textposition='middle center',
                      name='Upward Break'))

        if not downbreak_indices.empty:
            downbreak_plot_indices = [idx - offset for idx in data.index.get_indexer(downbreak_indices)]
            valid_downbreak_plot_indices = [i for i in downbreak_plot_indices if 0 <= i < len(x_axis)]
            downbreak_x = x_axis[valid_downbreak_plot_indices]
            # Get corresponding 'high' values
            downbreak_y_ref_indices = data.index.get_indexer(downbreak_indices)
            valid_y_ref_indices = [i for i in downbreak_y_ref_indices if i != -1]
            if valid_y_ref_indices:
                 downbreak_y = data['high'].iloc[valid_y_ref_indices] * 1.005 # Place above high
                 fig.add_trace(go.Scatter(
                     x=downbreak_x, y=downbreak_y, mode='markers+text',
                     marker=dict(color=down_color, size=10, symbol='triangle-down', line=dict(color='white', width=1)),
                     text='B', textfont=dict(color='white', size=10), textposition='middle center',
                     name='Downward Break'))

        # Update layout
        fig.update_layout(
            title=f"{self.name} (L:{length}, M:{self.multiplier}, Method:{self.calculate_method.upper()}, BP:{self.backpaint})",
            xaxis_rangeslider_visible=False,
            height=800, # Keep height or adjust as needed
            template="plotly_dark", # Keep template or adjust
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_yaxes(title_text="Price")
        # Ensure x-axis reflects time correctly if using timestamps
        # fig.update_xaxes(type='category') # Or 'date' if x_axis is datetime

        return fig

    def plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """
        Plot the trendlines and breakouts with price as a line chart.

        :param data: DataFrame with calculated indicator values
        :return: Plotly figure object
        """
        # Get parameters merging instance defaults with kwargs
        length = kwargs.get('length', self.length)
        upper_color = kwargs.get('upper_color', self.upper_color)
        down_color = kwargs.get('down_color', self.down_color)
        show_extension = kwargs.get('show_extension', self.show_extension)
        backpaint = kwargs.get('backpaint', self.backpaint)
        price_color = kwargs.get('price_color', 'lightblue')  # Optional: color for price line

        # Check if indicator data exists, otherwise calculate
        required_cols = ['plot_upper_final', 'plot_lower_final', 'upbreak_signal', 'downbreak_signal', 'offset', 'ph',
                         'pl', 'slope_ph', 'slope_pl', 'close']
        if not all(col in data.columns for col in required_cols):
            print("Indicator columns missing, recalculating...")  # Debug print
            data = self.calculate(data, **kwargs)

        # Create figure
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        # Extract x-axis data (use index if 'open_time' not present)
        x_axis = data['open_time'] if 'open_time' in data.columns else data.index
        x_axis_plot = x_axis  # Use original for price line

        # Calculate offset from the data for plotting
        offset = data['offset'].iloc[-1] if 'offset' in data.columns and not data.empty else 0
        # Create shifted x-axis for plotting lines/markers with offset
        x_indices = np.arange(len(x_axis))
        x_axis_shifted_indices = x_indices - offset
        # Ensure indices are within bounds for mapping back to x_axis values
        valid_shifted_indices = (x_axis_shifted_indices >= 0) & (x_axis_shifted_indices < len(x_axis))

        # Add price line chart (plotted against original x-axis)
        fig.add_trace(
            go.Scatter(
                x=x_axis_plot,
                y=data['close'],
                mode='lines',
                line=dict(color=price_color, width=1.5),
                name="Price (Close)"
            )
        )

        # Prepare data for plotting trendlines with offset
        plot_x = x_axis[x_axis_shifted_indices[valid_shifted_indices]]
        plot_upper_y = data['plot_upper_final'].iloc[valid_shifted_indices]
        plot_lower_y = data['plot_lower_final'].iloc[valid_shifted_indices]

        # Plot Upper Trendline (with NaNs for breaks and offset)
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_upper_y,
                line=dict(color=upper_color, width=1.5),
                name="Upper Trendline",
                connectgaps=False  # Important: Do not connect across NaNs
            )
        )

        # Plot Lower Trendline (with NaNs for breaks and offset)
        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_lower_y,
                line=dict(color=down_color, width=1.5),
                name="Lower Trendline",
                connectgaps=False  # Important: Do not connect across NaNs
            )
        )

        # --- Plot Extended Lines ---
        if show_extension and not data.empty:
            last_ph_idx = data.index[data['ph'] > 0].max() if (data['ph'] > 0).any() else None
            last_pl_idx = data.index[data['pl'] > 0].max() if (data['pl'] > 0).any() else None

            # Find corresponding index in the original x_axis
            x_axis_list = list(x_axis)

            if last_ph_idx is not None and last_ph_idx in data.index:
                idx_loc = data.index.get_loc(last_ph_idx)
                if idx_loc >= 0:
                    ph_val = data.loc[last_ph_idx, 'ph']
                    ph_slope = data.loc[last_ph_idx, 'slope_ph']
                    start_idx_shifted = idx_loc - offset

                    if start_idx_shifted >= 0 and start_idx_shifted < len(x_axis_list):
                        x1 = x_axis_list[start_idx_shifted]
                        y1 = ph_val  # Line starts at the pivot value
                        # Extend to the end of the chart
                        x2_idx_shifted = len(x_axis_list) - 1
                        if x2_idx_shifted > start_idx_shifted:
                            x2 = x_axis_list[x2_idx_shifted]
                            # Calculate y2 based on slope from pivot
                            bars_extended = (len(x_axis_list) - 1) - idx_loc
                            y2 = y1 - ph_slope * bars_extended
                            fig.add_trace(go.Scatter(
                                x=[x1, x2], y=[y1, y2], mode='lines',
                                line=dict(color=upper_color, width=1, dash='dash'),
                                name='Upper Extension', showlegend=False))

            if last_pl_idx is not None and last_pl_idx in data.index:
                idx_loc = data.index.get_loc(last_pl_idx)
                if idx_loc >= 0:
                    pl_val = data.loc[last_pl_idx, 'pl']
                    pl_slope = data.loc[last_pl_idx, 'slope_pl']
                    start_idx_shifted = idx_loc - offset

                    if start_idx_shifted >= 0 and start_idx_shifted < len(x_axis_list):
                        x1 = x_axis_list[start_idx_shifted]
                        y1 = pl_val  # Line starts at the pivot value
                        # Extend to the end of the chart
                        x2_idx_shifted = len(x_axis_list) - 1
                        if x2_idx_shifted > start_idx_shifted:
                            x2 = x_axis_list[x2_idx_shifted]
                            bars_extended = (len(x_axis_list) - 1) - idx_loc
                            y2 = y1 + pl_slope * bars_extended
                            fig.add_trace(go.Scatter(
                                x=[x1, x2], y=[y1, y2], mode='lines',
                                line=dict(color=down_color, width=1, dash='dash'),
                                name='Lower Extension', showlegend=False))

        # --- Plot Breakout Markers ---
        upbreak_indices = data.index[data['upbreak_signal'] == 1]
        downbreak_indices = data.index[data['downbreak_signal'] == 1]

        if not upbreak_indices.empty:
            # Apply offset to breakout marker positions
            upbreak_plot_indices = [idx - offset for idx in data.index.get_indexer(upbreak_indices)]
            valid_upbreak_plot_indices = [i for i in upbreak_plot_indices if 0 <= i < len(x_axis)]
            upbreak_x = x_axis[valid_upbreak_plot_indices]
            # Get corresponding 'close' values for plotting marker near the price line
            upbreak_y_ref_indices = data.index.get_indexer(upbreak_indices)  # Get original indices
            valid_y_ref_indices = [i for i in upbreak_y_ref_indices if i != -1]  # Ensure index found
            if valid_y_ref_indices:
                # Place marker slightly below the close price at the break bar
                upbreak_y = data['close'].iloc[valid_y_ref_indices] * 0.995
                fig.add_trace(go.Scatter(
                    x=upbreak_x, y=upbreak_y, mode='markers+text',
                    marker=dict(color=upper_color, size=10, symbol='triangle-up', line=dict(color='white', width=1)),
                    text='B', textfont=dict(color='white', size=10), textposition='middle center',
                    name='Upward Break'))

        if not downbreak_indices.empty:
            downbreak_plot_indices = [idx - offset for idx in data.index.get_indexer(downbreak_indices)]
            valid_downbreak_plot_indices = [i for i in downbreak_plot_indices if 0 <= i < len(x_axis)]
            downbreak_x = x_axis[valid_downbreak_plot_indices]
            # Get corresponding 'close' values
            downbreak_y_ref_indices = data.index.get_indexer(downbreak_indices)
            valid_y_ref_indices = [i for i in downbreak_y_ref_indices if i != -1]
            if valid_y_ref_indices:
                # Place marker slightly above the close price at the break bar
                downbreak_y = data['close'].iloc[valid_y_ref_indices] * 1.005
                fig.add_trace(go.Scatter(
                    x=downbreak_x, y=downbreak_y, mode='markers+text',
                    marker=dict(color=down_color, size=10, symbol='triangle-down', line=dict(color='white', width=1)),
                    text='B', textfont=dict(color='white', size=10), textposition='middle center',
                    name='Downward Break'))

        # Update layout
        fig.update_layout(
            title=f"{self.name} (Line Chart - L:{length}, M:{self.multiplier}, Method:{self.calculate_method.upper()}, BP:{self.backpaint})",
            xaxis_rangeslider_visible=False,
            height=800,  # Keep height or adjust as needed
            template="plotly_dark",  # Keep template or adjust
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_yaxes(title_text="Price")
        # Ensure x-axis reflects time correctly if using timestamps
        # fig.update_xaxes(type='category') # Or 'date' if x_axis is datetime

        return fig