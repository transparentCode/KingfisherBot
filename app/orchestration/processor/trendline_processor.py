from typing import Dict, Any
import pandas as pd
from datetime import datetime
import numpy as np
import logging


class TrendlineBreakoutProcessor:
    def __init__(self, db_pool, config):
        self.db_pool = db_pool
        self.config = config
        self.logger = logging.getLogger("app")

    async def process_results(self, asset: str, results: Dict[str, Any]):
        """Process indicator results to detect and save trendline breakouts"""
        try:
            # Look for trend analysis results with breakout signals
            for result_key, result_data in results.items():
                if not self._is_trendline_result(result_key, result_data):
                    continue

                metadata = result_data.get('metadata', {})
                timeframe = metadata.get('timeframe')

                # Only process monitored timeframes
                if timeframe not in self.config.timeframes_to_monitor:
                    continue

                # Check for breakout signals
                latest_signal = metadata.get('latest_signal', {})

                if self._has_breakout_signal(latest_signal):
                    await self._handle_breakout(asset, timeframe, result_data, latest_signal)

        except Exception as e:
            self.logger.error(f"Error processing trendline results for {asset}: {e}")

    def _is_trendline_result(self, result_key: str, result_data: Dict) -> bool:
        """Check if this is a trendline analysis result"""
        metadata = result_data.get('metadata', {})
        return (metadata.get('category') == 'Trend Analysis' and
                'trendline' in result_key.lower())

    def _has_breakout_signal(self, signal: Dict[str, Any]) -> bool:
        """Check if signal contains breakout information"""
        return (signal.get('upbreak_signal') == 1 or
                signal.get('downbreak_signal') == 1)

    async def _handle_breakout(self, asset: str, timeframe: str, result_data: Dict, signal: Dict):
        """Handle detected trendline breakout"""
        try:
            df = result_data['data']

            # Use your existing breakout detection logic
            await self._check_for_breakouts(asset, timeframe, df)

        except Exception as e:
            self.logger.error(f"Error handling trendline breakout for {asset}: {e}")

    async def _check_for_breakouts(self, asset: str, timeframe: str, df: pd.DataFrame):
        """Check for new breakouts and process them - from your original code"""
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

                self.logger.info(f"Processed {breakout_type} trendline breakout for {asset} {timeframe}")

        except Exception as e:
            self.logger.error(f"Error checking trendline breakouts for {asset} at {timeframe}: {e}")

    async def _check_if_new_breakout(self, asset: str, timeframe: str, timestamp, breakout_type: str) -> bool:
        """Check if this breakout is already in the database - from your original code"""
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
        """Save breakout information to database - from your original code"""
        try:
            async with self.db_pool.write_pool.acquire() as conn:
                # Check if we need to create the table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trendline_breakouts (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        breakout_type VARCHAR(10) NOT NULL,
                        price NUMERIC(20, 8) NOT NULL,
                        open_price NUMERIC(20, 8),
                        high_price NUMERIC(20, 8),
                        low_price NUMERIC(20, 8),
                        close_price NUMERIC(20, 8),
                        volume NUMERIC(20, 8),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Ensure all numeric values are valid for PostgreSQL NUMERIC type
                def safe_numeric(value):
                    if pd.isna(value) or np.isinf(value):
                        return 0.0
                    try:
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
                    (symbol, timeframe, timestamp, breakout_type, price,
                     open_price, high_price, low_price, close_price, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                                   asset, timeframe, timestamp, breakout_type,
                                   safe_price, safe_open, safe_high, safe_low, safe_close, safe_volume)

                self.logger.info(
                    f"Saved {breakout_type} trendline breakout for {asset} at {timeframe} timestamp {timestamp}")

        except Exception as e:
            self.logger.error(f"Error saving trendline breakout to database: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
