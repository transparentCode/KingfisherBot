import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import numpy as np
import os

from dotenv import load_dotenv

from app.indicators.trendline_with_breaks import TrendLineWithBreaks

dotenv = load_dotenv()

@dataclass
class IndicatorCalcServiceConfig:
    def __init__(self):
        self.logger_name = "app"
        self.timeframes_to_monitor = ['15m', '30m']  # Timeframes to track breakouts
        self.telegram_enabled = os.getenv('TELEGRAM_NOTIFICATIONS_ENABLED')  # Enable telegram notifications
        self.save_charts = os.getenv('SAVE_CHARTS')  # Save chart images for notifications
        self.charts_dir = os.path.join(os.path.dirname(__file__), "../charts")


class IndicatorCalcService:

    def __init__(self, calculator_id: int, calc_queue: asyncio.Queue, db_pool, last_calculation: Dict[str, float],
                 config: Optional[IndicatorCalcServiceConfig] = None):
        self.calculator_id = calculator_id
        self.calc_queue = calc_queue
        self.db_pool = db_pool
        self.last_calculation = last_calculation
        self.should_run = False
        self.config = IndicatorCalcServiceConfig() if config is None else config
        self.logger = logging.getLogger(self.config.logger_name)
        self.lookback_hours = 96  # 4 days
        self.trendline_length = 14

        self.timeframe_minutes = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '12h': 720,
            '1d': 1440
        }

        # Initialize trendline indicators for different timeframes
        self.trendline_indicators = {}
        for tf in self.config.timeframes_to_monitor:
            self.trendline_indicators[tf] = TrendLineWithBreaks(
                name=f"TrendLine_{tf}"
            )

        # Create charts directory if it doesn't exist
        if self.config.save_charts and not os.path.exists(self.config.charts_dir):
            os.makedirs(self.config.charts_dir)

        # Initialize telegram client if enabled
        self.telegram_client = None
        if self.config.telegram_enabled:
            from app.telegram import TelegramClient
            self.telegram_client = TelegramClient()

    async def start(self):
        self.logger.info(f"Starting indicator calculator {self.calculator_id}")
        self.should_run = True

        while self.should_run:
            try:
                asset = await asyncio.wait_for(self.calc_queue.get(), timeout=1.0)
                try:
                    await self._calculate_indicators(asset)
                except Exception as e:
                    self.logger.error(f"Error calculating indicators for {asset}: {e}")
                finally:
                    self.calc_queue.task_done()

            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self.logger.error(f"Error in indicator calculator {self.calculator_id}: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing

    async def _calculate_indicators(self, asset: str):
        self.logger.debug(f"Calculating indicators for {asset}")

        try:
            # Fetch 1-min candles from database
            async with self.db_pool.read_pool.acquire() as conn:
                rows = await conn.fetch("""
                SELECT
                    timestamp as bucket,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM candles
                    WHERE symbol = $1
                    AND timestamp > NOW() - INTERVAL '{} hours'
                    ORDER BY timestamp ASC
                """.format(self.lookback_hours), asset)

                df = self._rows_to_dataframe(rows)

                if len(df) > 0:
                    # Process data for each timeframe
                    for timeframe in self.config.timeframes_to_monitor:
                        await self._process_timeframe(asset, timeframe, df)

                    # Update last calculation time
                    self.last_calculation[asset] = time.time()
                else:
                    self.logger.warning(f"No data available for {asset}")
        except Exception as e:
            self.logger.error(f"Database error while calculating indicators: {e}")

    async def _process_timeframe(self, asset: str, timeframe: str, df_1m: pd.DataFrame):
        """Process data for a specific timeframe, identify breakouts and send alerts"""
        try:
            self.logger.info(f"Starting processing {timeframe} for {asset}")
            # Resample 1m data to the target timeframe
            tf_minutes = self.timeframe_minutes.get(timeframe, 0)
            if tf_minutes == 0:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return

            # Set the timestamp as index if it's not already
            if 'bucket' in df_1m.columns:
                self.logger.debug(f"Setting bucket as index for {asset} {timeframe}")
                df_1m = df_1m.set_index('bucket')

            # Optional: Check for gaps in 1m data
            # expected_range = pd.date_range(start=df_1m.index.min(), end=df_1m.index.max(), freq='1min')
            # missing_timestamps = expected_range.difference(df_1m.index)
            # if not missing_timestamps.empty:
            #     self.logger.info(f"Found {len(missing_timestamps)} missing 1-minute timestamps for {asset} {timeframe}.")
            #     if len(missing_timestamps) < 10: # Log if few, otherwise it's too much
            #         self.logger.info(f"Missing 1-minute timestamps: {missing_timestamps}")

            # for col in ['open', 'high', 'low', 'close', 'volume']:
            #     if col in df_1m.columns:
            #         df_1m[col] = pd.to_numeric(df_1m[col], errors='coerce')
            # Resample to target timeframe
            df_tf = self._resample_ohlcv(df_1m, tf_minutes)

            self.logger.info(f"Resampled data shape: {df_tf.shape} for {asset} {timeframe}")

            # Verify that data is valid
            if df_tf.isnull().values.any():
                self.logger.warning(f"Resampled data contains NULL values for {asset} {timeframe}")

            if (df_tf == 0).all().any():
                self.logger.warning(f"Resampled data contains columns with all zeros for {asset} {timeframe}")

            if len(df_tf) < self.trendline_length * 3:
                self.logger.warning(
                    f"Not enough data for {asset} at {timeframe}. Need at least {self.trendline_length * 3} bars.")
                return

            # Convert any potential nan/inf values to avoid decimal errors
            df_tf = df_tf.replace([np.inf, -np.inf], np.nan).dropna()

            # Check if we still have enough data after cleaning
            if len(df_tf) < self.trendline_length * 3:
                self.logger.warning(f"Not enough clean data for {asset} at {timeframe} after removing NaN/Inf values")
                return

            # Calculate trendline indicator
            trendline_indicator = self.trendline_indicators[timeframe]
            df_with_trendline = trendline_indicator.calculate(df_tf)

            self.logger.info(f"Trendline calculation completed for {asset} at {timeframe}")

            # Make sure the result doesn't contain NaN or Inf values
            if df_with_trendline is not None and not df_with_trendline.empty:
                df_with_trendline = df_with_trendline.replace([np.inf, -np.inf], np.nan)

                # Check for breakouts only if we have valid signals
                if 'upbreak_signal' in df_with_trendline.columns and 'downbreak_signal' in df_with_trendline.columns:
                    if not df_with_trendline['upbreak_signal'].isna().all() or not df_with_trendline[
                        'downbreak_signal'].isna().all():
                        await self._check_for_breakouts(asset, timeframe, df_with_trendline)
                    else:
                        self.logger.debug(f"No valid breakout signals for {asset} at {timeframe}")
                else:
                    self.logger.warning(f"Missing breakout signal columns for {asset} at {timeframe}")
            else:
                self.logger.warning(f"No trendline data returned for {asset} at {timeframe}")

        except Exception as e:
            # Log more detailed error information
            import traceback
            self.logger.error(f"Error processing {timeframe} for {asset}: {e.__class__.__name__}: {e}")
            self.logger.debug(traceback.format_exc())

    def _resample_ohlcv(self, df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
        """Resample OHLCV data to a target timeframe"""
        # Make sure DataFrame has a datetime index
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not DatetimeIndex, cannot resample properly")
            return df

        # Resample rule (e.g., '15T' for 15 minutes)
        rule = f"{target_minutes}T"

        # Resample OHLCV data
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        return resampled.dropna()

    async def _check_for_breakouts(self, asset: str, timeframe: str, df: pd.DataFrame):
        """Check for new breakouts and process them"""
        try:
            # Get the latest upward and downward breakouts
            upbreaks = df[df['upbreak_signal'] == 1].index
            downbreaks = df[df['downbreak_signal'] == 1].index

            if len(upbreaks) == 0 and len(downbreaks) == 0:
                self.logger.debug(f"No breakouts found for {asset} at {timeframe}")
                return

            # Check for the latest breakout
            latest_up_breakout = upbreaks[-1] if len(upbreaks) > 0 else None
            latest_down_breakout = downbreaks[-1] if len(downbreaks) > 0 else None

            # Determine the latest breakout
            latest_breakout = None
            breakout_type = None

            if latest_up_breakout is not None and latest_down_breakout is not None:
                if latest_up_breakout > latest_down_breakout:
                    latest_breakout = latest_up_breakout
                    breakout_type = "UPWARD"
                else:
                    latest_breakout = latest_down_breakout
                    breakout_type = "DOWNWARD"
            elif latest_up_breakout is not None:
                latest_breakout = latest_up_breakout
                breakout_type = "UPWARD"
            elif latest_down_breakout is not None:
                latest_breakout = latest_down_breakout
                breakout_type = "DOWNWARD"

            if latest_breakout is None:
                return

            # Convert to datetime if it's not already
            if not isinstance(latest_breakout, datetime):
                latest_breakout = pd.to_datetime(latest_breakout)

            # Check if this breakout is already stored in database
            is_new_breakout = await self._check_if_new_breakout(asset, timeframe, latest_breakout, breakout_type)

            if is_new_breakout:
                # Get bar data at breakout
                breakout_bar = df.loc[latest_breakout]

                # Save breakout to database
                await self._save_breakout_to_db(
                    asset=asset,
                    timeframe=timeframe,
                    timestamp=latest_breakout,
                    breakout_type=breakout_type,
                    price=breakout_bar['close'],
                    bar_data={
                        'open': breakout_bar['open'],
                        'high': breakout_bar['high'],
                        'low': breakout_bar['low'],
                        'close': breakout_bar['close'],
                        'volume': breakout_bar['volume']
                    }
                )

                # Save chart for notification if enabled
                chart_path = None
                if self.config.save_charts:
                    chart_path = await self._save_chart(asset, timeframe, df, breakout_type, latest_breakout)

                # Send notification
                if self.config.telegram_enabled and self.telegram_client:
                    await self._send_breakout_notification(
                        asset=asset,
                        timeframe=timeframe,
                        breakout_type=breakout_type,
                        timestamp=latest_breakout,
                        price=breakout_bar['close'],
                        bar_data={
                            'open': breakout_bar['open'],
                            'high': breakout_bar['high'],
                            'low': breakout_bar['low'],
                            'close': breakout_bar['close'],
                            'volume': breakout_bar['volume']
                        },
                        chart_path=chart_path
                    )
        except Exception as e:
            self.logger.error(f"Error checking breakouts for {asset} at {timeframe}: {e}")

    async def _check_if_new_breakout(self, asset: str, timeframe: str, timestamp, breakout_type: str) -> bool:
        """Check if this breakout is already in the database"""
        try:
            async with self.db_pool.read_pool.acquire() as conn:
                # Query the database for the latest breakout for this asset and timeframe
                last_breakout = await conn.fetchrow("""
                    SELECT timestamp 
                    FROM trendline_breakouts 
                    WHERE symbol = $1 AND timeframe = $2 AND breakout_type = $3
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, asset, timeframe, breakout_type)

                if last_breakout is None:
                    # No previous breakout found
                    return True

                last_timestamp = last_breakout['timestamp']
                if timestamp > last_timestamp:
                    # This is a newer breakout
                    return True

                # Not a new breakout
                return False

        except Exception as e:
            self.logger.error(f"Error checking if breakout is new: {e}")
            # Return True to be safe (try to save it anyway)
            return True

    async def _save_breakout_to_db(self, asset: str, timeframe: str, timestamp, breakout_type: str,
                                   price: float, bar_data: Dict[str, float]):
        """Save breakout information to database"""
        try:
            async with self.db_pool.write_pool.acquire() as conn:
                # Check if we need to create the table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trendline_breakouts (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        breakout_type TEXT NOT NULL,
                        price NUMERIC NOT NULL,
                        open NUMERIC NOT NULL,
                        high NUMERIC NOT NULL,
                        low NUMERIC NOT NULL,
                        close NUMERIC NOT NULL,
                        volume NUMERIC,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Ensure all numeric values are valid for PostgreSQL NUMERIC type
                # Convert any potential NaN, inf, or extremely large values
                def safe_numeric(value):
                    if pd.isna(value) or np.isinf(value):
                        return 0.0
                    try:
                        # Ensure it's within PostgreSQL NUMERIC limits
                        return float(value)
                    except (ValueError, OverflowError, TypeError):
                        return 0.0

                # Apply safe conversion to all numeric values
                safe_price = safe_numeric(price)
                safe_open = safe_numeric(bar_data.get('open', 0))
                safe_high = safe_numeric(bar_data.get('high', 0))
                safe_low = safe_numeric(bar_data.get('low', 0))
                safe_close = safe_numeric(bar_data.get('close', 0))
                safe_volume = safe_numeric(bar_data.get('volume', 0))

                # Insert the breakout
                await conn.execute("""
                    INSERT INTO trendline_breakouts
                    (symbol, timeframe, timestamp, breakout_type, price, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                                   asset, timeframe, timestamp, breakout_type,
                                   safe_price, safe_open, safe_high, safe_low, safe_close, safe_volume)

                self.logger.info(f"Saved {breakout_type} breakout for {asset} at {timeframe} timestamp {timestamp}")

        except Exception as e:
            self.logger.error(f"Error saving breakout to database: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    async def _save_chart(self, asset: str, timeframe: str, df: pd.DataFrame,
                          breakout_type: str, breakout_timestamp) -> Optional[str]:
        """Save chart of the breakout for notification"""
        try:
            # Generate chart using the indicator's plotting function
            trendline_indicator = self.trendline_indicators[timeframe]
            fig = trendline_indicator.plot(df)

            # Create a filename with timestamp
            timestamp_str = breakout_timestamp.strftime("%Y%m%d_%H%M%S")
            chart_filename = f"{asset.replace('/', '_')}_{timeframe}_{breakout_type.lower()}_{timestamp_str}.png"
            chart_path = os.path.join(self.config.charts_dir, chart_filename)

            # Save the chart
            fig.write_image(chart_path)
            self.logger.info(f"Saved chart to {chart_path}")

            return chart_path
        except Exception as e:
            self.logger.error(f"Error saving chart: {e}")
            return None

    async def _send_breakout_notification(self, asset: str, timeframe: str, breakout_type: str, timestamp,
                                          price: float, bar_data: dict = None, chart_path: str = None):
        """Send breakout notification to Telegram with chart image"""
        try:
            # Format timestamp for message
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

            # Create message text
            message = f"📊 {breakout_type.upper()} BREAKOUT DETECTED 📊\n\n" \
                      f"💰 Asset: {asset}\n" \
                      f"⏱️ Timeframe: {timeframe}\n" \
                      f"🕒 Time: {formatted_time}\n" \
                      f"💵 Price: {price:.2f}"

            # Add OHLC data if provided
            if bar_data:
                message += f"\n\n📈 OHLC Data:\n" \
                           f"Open: {bar_data.get('open', 0):.2f}\n" \
                           f"High: {bar_data.get('high', 0):.2f}\n" \
                           f"Low: {bar_data.get('low', 0):.2f}\n" \
                           f"Close: {bar_data.get('close', 0):.2f}\n" \
                           f"Volume: {bar_data.get('volume', 0):.2f}"

            # Use provided chart_path or generate one if not provided
            if chart_path is None:
                date_str = timestamp.strftime('%Y%m%d_%H%M%S')
                chart_filename = f"{asset.replace('/', '_')}_{timeframe}_{breakout_type.lower()}_{date_str}.png"
                chart_path = os.path.join(self.config.charts_dir, chart_filename)

            # Send message with photo - using the string path directly
            if os.path.exists(chart_path):
                # Pass the file path as a string, not as a file object
                success = self.telegram_client.send_photo(
                    photo_path=chart_path,  # Pass the path as a string
                    caption=message
                )
                if success:
                    self.logger.info(f"Sent breakout notification with chart to Telegram for {asset} {timeframe}")
                else:
                    self.logger.warning(f"Failed to send chart for {asset} {timeframe}")
            else:
                # Fallback to text-only message
                success = self.telegram_client.send_message(
                    message=message
                )
                if success:
                    self.logger.warning(f"Sent text-only notification - chart not found at {chart_path}")
                else:
                    self.logger.warning(f"Failed to send text notification for {asset} {timeframe}")

        except Exception as e:
            self.logger.error(f"Failed to send breakout notification for {asset} at {timeframe}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _rows_to_dataframe(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame([dict(row) for row in rows])

    def _calculate_ta_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators from OHLCV data."""
        results = {}
        # Original indicator calculation code goes here
        return results

    async def stop(self):
        self.logger.info(f"Stopping indicator calculator {self.calculator_id}")
        self.should_run = False