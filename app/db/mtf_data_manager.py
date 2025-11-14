import logging
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd


class MTFDataManager:
    def __init__(self, db_handler, lookback_hours: int = 96):
        self.db_handler = db_handler
        self.lookback_hours = lookback_hours
        self.logger = logging.getLogger("app")

        # Map timeframe formats to DB intervals
        self.timeframe_to_db_interval = {
            '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '2h': '120', '4h': '240', '6h': '360',
            '12h': '720', '1d': '1440'
        }

        # Minutes for resampling
        self.timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '12h': 720, '1d': 1440
        }

    async def get_mtf_data(self, asset: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes efficiently"""
        try:
            # Check if we can fetch base timeframe data directly or need to resample
            base_timeframe = self._get_optimal_base_timeframe(timeframes)

            if base_timeframe in timeframes:
                # Fetch base data directly from DB
                base_data = await self._fetch_base_data(asset, base_timeframe)
            else:
                # Fetch 1m data for resampling
                base_data = await self._fetch_base_data(asset, '1m')
                base_timeframe = '1m'

            if base_data.empty:
                self.logger.warning(f"No base data found for {asset}")
                return {tf: pd.DataFrame() for tf in timeframes}

            mtf_data = {}
            for tf in timeframes:
                if tf == base_timeframe:
                    mtf_data[tf] = base_data.copy()
                else:
                    mtf_data[tf] = self._resample_data(base_data, base_timeframe, tf)

            return mtf_data

        except Exception as e:
            self.logger.error(f"Error getting MTF data for {asset}: {e}")
            return {tf: pd.DataFrame() for tf in timeframes}

    def _get_optimal_base_timeframe(self, timeframes: List[str]) -> str:
        """Determine the best base timeframe to minimize resampling"""
        # Sort timeframes by minutes to find the smallest
        sorted_tfs = sorted(timeframes, key=lambda x: self.timeframe_minutes.get(x, float('inf')))
        return sorted_tfs[0] if sorted_tfs else '1m'

    async def _fetch_base_data(self, asset: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from database and convert to DataFrame"""
        try:
            # Calculate lookback time
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.lookback_hours)

            # Get DB interval format
            db_interval = self.timeframe_to_db_interval.get(timeframe, timeframe)

            self.logger.debug(f"Fetching {timeframe} data for {asset} from {start_time} to {end_time}")

            # Fetch from database
            rows = await self.db_handler.read_candles(
                symbol=asset,
                interval=db_interval,
                start_time=start_time,
                end_time=end_time,
                limit=5000  # Adjust based on your needs
            )

            if not rows:
                self.logger.warning(f"No data found for {asset} {timeframe}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])

            # Ensure proper column names and types
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Sort by timestamp (oldest first for proper resampling)
            df.sort_index(inplace=True)

            self.logger.debug(f"Fetched {len(df)} candles for {asset} {timeframe}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching base data for {asset} {timeframe}: {e}")
            return pd.DataFrame()

    def _resample_data(self, df: pd.DataFrame, from_timeframe: str, to_timeframe: str) -> pd.DataFrame:
        """Resample data from one timeframe to another"""
        try:
            if df.empty:
                return pd.DataFrame()

            from_minutes = self.timeframe_minutes.get(from_timeframe, 1)
            to_minutes = self.timeframe_minutes.get(to_timeframe, 1)

            # Can only upsample (e.g., 1m to 5m), not downsample
            if to_minutes < from_minutes:
                self.logger.warning(f"Cannot downsample from {from_timeframe} to {to_timeframe}")
                return pd.DataFrame()

            # Create resampling rule
            if to_timeframe.endswith('m'):
                rule = f"{to_minutes}T"  # T = minutes
            elif to_timeframe.endswith('h'):
                rule = f"{to_minutes // 60}H"  # H = hours
            elif to_timeframe.endswith('d'):
                rule = f"{to_minutes // 1440}D"  # D = days
            else:
                rule = f"{to_minutes}T"

            # Resample OHLCV data
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            self.logger.debug(f"Resampled {len(df)} -> {len(resampled)} candles ({from_timeframe} -> {to_timeframe})")
            return resampled

        except Exception as e:
            self.logger.error(f"Error resampling data from {from_timeframe} to {to_timeframe}: {e}")
            return pd.DataFrame()

    async def validate_data_availability(self, asset: str, timeframes: List[str]) -> Dict[str, bool]:
        """Check data availability for each timeframe"""
        availability = {}

        for tf in timeframes:
            try:
                data = await self._fetch_base_data(asset, tf)
                availability[tf] = not data.empty and len(data) > 10  # Minimum viable data
            except Exception:
                availability[tf] = False

        return availability



## usage example:
# mtf_manager = MTFDataManager(db_handler, lookback_hours=96)
#
# # Get multiple timeframes
# timeframes = ['1m', '5m', '15m', '1h']
# mtf_data = await mtf_manager.get_mtf_data('NIFTY50', timeframes)
#
# # Check what you got
# for tf, data in mtf_data.items():
#     print(f"{tf}: {len(data)} candles")
#     if not data.empty:
#         print(f"  Latest: {data.index[-1]} - Close: {data['close'].iloc[-1]}")

