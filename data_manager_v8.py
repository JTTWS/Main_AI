#!/usr/bin/env python3
"""
================================================================================
DATA MANAGER V8 - Enhanced Data Loading for JTTWS Trading Bot
================================================================================

Handles:
- Multiple CSV files per symbol (3-year chunks)
- 15-minute OHLCV candlestick data
- Weekly range files (ATR, volatility metrics)
- Economic calendar integration
- Data aggregation and preprocessing

Format Support:
- OHLCV: Local time, Open, High, Low, Close, Volume (15M timeframe)
- Weekly Ranges: time, high, low, close, atr, ma, range, range_pips
- Economic Calendar: datetime, Name, Impact, Currency, Category

Author: E1 AI Agent + Grok Integration
Date: January 2025
Version: 8.0
================================================================================
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger('DataManagerV8')


class DataManagerV8:
    """
    Enhanced Data Manager for V8 Bot.
    
    Loads and preprocesses:
    - Multi-file OHLCV candlestick data (15M)
    - Weekly range statistics
    - Economic calendar events
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize DataManagerV8.
        
        Args:
            data_dir: Root directory containing data files (default: ./data relative to script)
        """
        if data_dir is None:
            # Use relative path: ./data from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
        self.data_dir = data_dir
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.weekly_ranges: Dict[str, pd.DataFrame] = {}
        self.economic_calendar: Optional[pd.DataFrame] = None
        
        logger.info(f"📂 DataManagerV8 initialized with data_dir: {data_dir}")
    
    def load_symbol_data(
        self,
        symbol: str,
        start_date: str = '2003-01-01',
        end_date: str = '2024-12-31',
        use_mock: bool = False
    ) -> pd.DataFrame:
        """
        Load OHLCV candlestick data for a symbol.
        
        Handles multiple CSV files (e.g., EURUSD2003-2024/*.csv)
        and concatenates them into a single DataFrame.
        
        Args:
            symbol: Symbol name (EURUSD, GBPUSD, USDJPY)
            start_date: Start date for filtering
            end_date: End date for filtering
            use_mock: If True, generate mock data as fallback
        
        Returns:
            pd.DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"📥 Loading {symbol} data ({start_date} to {end_date})...")
        
        # Try to load from directory structure
        symbol_dir = os.path.join(self.data_dir, f"{symbol}2003-2024")
        
        if os.path.exists(symbol_dir):
            try:
                df = self._load_from_directory(symbol, symbol_dir, start_date, end_date)
                logger.info(f"✅ Loaded {len(df)} rows for {symbol} from {symbol_dir}")
                self.symbol_data[symbol] = df
                return df
            except Exception as e:
                logger.error(f"❌ Error loading {symbol} from directory: {e}")
        
        # Try single CSV file
        single_csv = os.path.join(self.data_dir, f"{symbol}.csv")
        if os.path.exists(single_csv):
            try:
                df = self._load_from_single_csv(single_csv, start_date, end_date)
                logger.info(f"✅ Loaded {len(df)} rows for {symbol} from {single_csv}")
                self.symbol_data[symbol] = df
                return df
            except Exception as e:
                logger.error(f"❌ Error loading {symbol} from CSV: {e}")
        
        # Fallback to mock data
        if use_mock:
            logger.warning(f"⚠️  No real data found for {symbol}, generating mock data...")
            df = self._generate_mock_data(symbol, start_date, end_date)
            self.symbol_data[symbol] = df
            return df
        else:
            logger.error(f"❌ No data found for {symbol} and use_mock=False")
            return pd.DataFrame()
    
    def _load_from_directory(
        self,
        symbol: str,
        directory: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load data from multiple CSV files in a directory.
        
        Args:
            symbol: Symbol name
            directory: Directory containing CSV files
            start_date: Filter start date
            end_date: Filter end date
        
        Returns:
            Concatenated DataFrame
        """
        # Find all CSV files matching the symbol
        csv_files = sorted(glob.glob(os.path.join(directory, f"{symbol}*.csv")))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found for {symbol} in {directory}")
        
        logger.info(f"📂 Found {len(csv_files)} CSV files for {symbol}")
        
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Standardize column names
                df = self._standardize_columns(df)
                
                # Parse timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                dfs.append(df)
                logger.debug(f"   ✓ Loaded {len(df)} rows from {os.path.basename(csv_file)}")
            except Exception as e:
                logger.warning(f"   ⚠️ Skipped {os.path.basename(csv_file)}: {e}")
        
        if not dfs:
            raise ValueError(f"No valid data loaded for {symbol}")
        
        # Concatenate all dataframes
        df = pd.concat(dfs, axis=0)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"⚠️  Missing column '{col}', filling with default values")
                df[col] = 1.0 if col != 'volume' else 0.0
        
        return df[required_cols].reset_index()
    
    def _load_from_single_csv(
        self,
        csv_path: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Load data from a single CSV file."""
        df = pd.read_csv(csv_path)
        df = self._standardize_columns(df)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        return df[required_cols].reset_index()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to lowercase.
        
        Maps: 'Local time' -> 'timestamp', 'Open' -> 'open', etc.
        """
        # Column mapping
        col_mapping = {
            'Local time': 'timestamp',
            'Time': 'timestamp',
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Rename columns
        df = df.rename(columns=col_mapping)
        
        # Lowercase all column names
        df.columns = df.columns.str.lower()
        
        return df
    
    def _generate_mock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate realistic mock OHLCV data.
        
        Uses random walk with realistic forex characteristics:
        - EURUSD: ~1.08, volatility ~0.0001
        - GBPUSD: ~1.26, volatility ~0.00012
        - USDJPY: ~150, volatility ~0.05
        """
        logger.info(f"🎲 Generating mock data for {symbol}...")
        
        # Symbol-specific parameters
        params = {
            'EURUSD': {'base': 1.08, 'vol': 0.0001, 'spread': 0.00005},
            'GBPUSD': {'base': 1.26, 'vol': 0.00012, 'spread': 0.00006},
            'USDJPY': {'base': 150.0, 'vol': 0.05, 'spread': 0.03}
        }
        
        p = params.get(symbol, {'base': 1.0, 'vol': 0.0001, 'spread': 0.00001})
        
        # Generate timestamps (15-minute intervals)
        dates = pd.date_range(start=start_date, end=end_date, freq='15min')
        n = len(dates)
        
        # Random walk for close prices
        returns = np.random.normal(0, p['vol'], n)
        close = p['base'] + np.cumsum(returns)
        
        # Generate OHLC from close
        high = close + np.abs(np.random.normal(0, p['spread'], n))
        low = close - np.abs(np.random.normal(0, p['spread'], n))
        open_price = low + np.random.uniform(0, 1, n) * (high - low)
        
        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))
        
        # Generate volume
        volume = np.random.randint(1000, 10000, n)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        return df
    
    def load_weekly_ranges(self, symbol: str) -> pd.DataFrame:
        """
        Load weekly range statistics for a symbol.
        
        Args:
            symbol: Symbol name (EURUSD, GBPUSD, USDJPY)
        
        Returns:
            DataFrame with weekly range data
        """
        weekly_file = os.path.join(self.data_dir, f"{symbol}_weekly_ranges.csv")
        
        if not os.path.exists(weekly_file):
            logger.warning(f"⚠️  Weekly ranges not found for {symbol}: {weekly_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(weekly_file)
            df['time'] = pd.to_datetime(df['time'])
            self.weekly_ranges[symbol] = df
            logger.info(f"✅ Loaded {len(df)} weekly ranges for {symbol}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading weekly ranges for {symbol}: {e}")
            return pd.DataFrame()
    
    def load_economic_calendar(self) -> pd.DataFrame:
        """
        Load economic calendar events.
        
        Returns:
            DataFrame with economic events
        """
        calendar_file = os.path.join(self.data_dir, 'combined_economic_calendar.csv')
        
        if not os.path.exists(calendar_file):
            logger.warning(f"⚠️  Economic calendar not found: {calendar_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(calendar_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            self.economic_calendar = df
            logger.info(f"✅ Loaded {len(df)} economic events")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading economic calendar: {e}")
            return pd.DataFrame()
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of loaded data.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            'symbols_loaded': list(self.symbol_data.keys()),
            'date_ranges': {},
            'total_rows': {},
            'weekly_ranges_available': list(self.weekly_ranges.keys()),
            'economic_events': len(self.economic_calendar) if self.economic_calendar is not None else 0
        }
        
        for symbol, df in self.symbol_data.items():
            if not df.empty and 'timestamp' in df.columns:
                summary['date_ranges'][symbol] = {
                    'start': df['timestamp'].min().strftime('%Y-%m-%d'),
                    'end': df['timestamp'].max().strftime('%Y-%m-%d')
                }
                summary['total_rows'][symbol] = len(df)
        
        return summary


if __name__ == '__main__':
    # Test DataManagerV8
    logging.basicConfig(level=logging.INFO)
    
    dm = DataManagerV8()  # Will use ./data by default
    
    # Test loading EURUSD
    df = dm.load_symbol_data('EURUSD', use_mock=True)
    print(f"\n✅ EURUSD Data Shape: {df.shape}")
    print(df.head())
    
    # Test weekly ranges
    weekly = dm.load_weekly_ranges('EURUSD')
    print(f"\n✅ Weekly Ranges Shape: {weekly.shape}")
    
    # Test economic calendar
    calendar = dm.load_economic_calendar()
    print(f"\n✅ Economic Calendar Shape: {calendar.shape}")
    
    # Summary
    summary = dm.get_data_summary()
    print(f"\n📊 Data Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
