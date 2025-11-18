#!/usr/bin/env python3
"""
================================================================================
DATA AGGREGATOR V8 - 15M to Daily Conversion
================================================================================

Converts 15-minute OHLCV data to daily aggregates for walk-forward training.

Features:
- OHLCV aggregation (open: first, high: max, low: min, close: last, volume: sum)
- Performance metrics calculation (returns, sharpe, rolling metrics)
- Walk-forward compatible format (date, sharpe, reward, returns)

Author: E1 AI Agent + Grok Integration
Date: November 2025
Version: 8.0
================================================================================
"""

import logging
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger('DataAggregatorV8')


class DataAggregatorV8:
    """
    Aggregates 15-minute OHLCV data to daily format for walk-forward training.
    """
    
    def __init__(self):
        """Initialize DataAggregatorV8."""
        self.aggregated_data: Dict[str, pd.DataFrame] = {}
        logger.info("📊 DataAggregatorV8 initialized")
    
    def aggregate_to_daily(
        self,
        df: pd.DataFrame,
        symbol: str = 'EURUSD'
    ) -> pd.DataFrame:
        """
        Aggregate 15-minute data to daily OHLCV.
        
        Args:
            df: DataFrame with timestamp, open, high, low, close, volume
            symbol: Symbol name for logging
        
        Returns:
            Daily aggregated DataFrame
        """
        logger.info(f"🔄 Aggregating {symbol} from 15M to Daily...")
        logger.info(f"   Input: {len(df)} rows (15M bars)")
        
        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns:
            logger.error("❌ Missing 'timestamp' column")
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Aggregate to daily
        daily = pd.DataFrame()
        daily['open'] = df['open'].resample('D').first()
        daily['high'] = df['high'].resample('D').max()
        daily['low'] = df['low'].resample('D').min()
        daily['close'] = df['close'].resample('D').last()
        daily['volume'] = df['volume'].resample('D').sum()
        
        # Remove NaN rows (non-trading days)
        daily = daily.dropna()
        
        # Calculate returns
        daily['returns'] = daily['close'].pct_change()
        
        # Calculate rolling Sharpe (20-day window)
        rolling_window = 20
        daily['rolling_mean'] = daily['returns'].rolling(window=rolling_window).mean()
        daily['rolling_std'] = daily['returns'].rolling(window=rolling_window).std()
        daily['sharpe'] = (daily['rolling_mean'] / (daily['rolling_std'] + 1e-6)) * np.sqrt(252)
        
        # Calculate cumulative returns
        daily['cum_returns'] = (1 + daily['returns']).cumprod()
        
        # Calculate reward (normalized returns)
        daily['reward'] = daily['returns'] * 100  # Convert to basis points
        
        # Fill NaN values
        daily['sharpe'] = daily['sharpe'].fillna(0)
        daily['reward'] = daily['reward'].fillna(0)
        
        # Reset index to have date column
        daily = daily.reset_index()
        daily = daily.rename(columns={'timestamp': 'date'})
        
        logger.info(f"   Output: {len(daily)} days")
        logger.info(f"   Date range: {daily['date'].min()} → {daily['date'].max()}")
        logger.info(f"   Avg Sharpe: {daily['sharpe'].mean():.3f}")
        
        self.aggregated_data[symbol] = daily
        
        return daily
    
    def prepare_walk_forward_data(
        self,
        symbols_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Prepare combined data for walk-forward training.
        
        Args:
            symbols_data: Dict of {symbol: daily_df}
        
        Returns:
            Combined DataFrame for walk-forward (date, sharpe, reward, returns)
        """
        logger.info("🔀 Preparing walk-forward data from multiple symbols...")
        
        all_dfs = []
        
        for symbol, df in symbols_data.items():
            if df.empty:
                continue
            
            # Select relevant columns
            df_copy = df[['date', 'sharpe', 'reward', 'returns']].copy()
            df_copy['symbol'] = symbol
            all_dfs.append(df_copy)
        
        if not all_dfs:
            logger.error("❌ No data available for walk-forward")
            return pd.DataFrame()
        
        # Combine all symbols
        combined = pd.concat(all_dfs, axis=0)
        
        # Group by date and take average across symbols
        wf_data = combined.groupby('date').agg({
            'sharpe': 'mean',
            'reward': 'mean',
            'returns': 'mean'
        }).reset_index()
        
        # Sort by date
        wf_data = wf_data.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✅ Walk-forward data prepared: {len(wf_data)} days")
        logger.info(f"   Date range: {wf_data['date'].min()} → {wf_data['date'].max()}")
        logger.info(f"   Avg Sharpe: {wf_data['sharpe'].mean():.3f}")
        logger.info(f"   Avg Reward: {wf_data['reward'].mean():.6f}")
        
        return wf_data
    
    def get_summary(self) -> Dict:
        """
        Get summary of aggregated data.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'symbols': list(self.aggregated_data.keys()),
            'total_days': {},
            'date_ranges': {},
            'avg_sharpe': {},
            'avg_returns': {}
        }
        
        for symbol, df in self.aggregated_data.items():
            summary['total_days'][symbol] = len(df)
            summary['date_ranges'][symbol] = {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            }
            summary['avg_sharpe'][symbol] = df['sharpe'].mean()
            summary['avg_returns'][symbol] = df['returns'].mean()
        
        return summary


if __name__ == '__main__':
    # Test DataAggregatorV8
    logging.basicConfig(level=logging.INFO)
    
    from data_manager_v8 import DataManagerV8
    
    # Load data
    dm = DataManagerV8()  # Will use ./data by default
    aggregator = DataAggregatorV8()
    
    # Test with EURUSD
    eurusd_15m = dm.load_symbol_data('EURUSD', '2020-01-01', '2024-12-31', use_mock=False)
    print(f"\n✅ EURUSD 15M Data: {len(eurusd_15m)} rows")
    
    # Aggregate to daily
    eurusd_daily = aggregator.aggregate_to_daily(eurusd_15m, 'EURUSD')
    print(f"\n✅ EURUSD Daily Data: {len(eurusd_daily)} days")
    print(eurusd_daily.head())
    
    # Test with all symbols
    all_symbols = {}
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
        df_15m = dm.load_symbol_data(symbol, '2020-01-01', '2024-12-31', use_mock=False)
        df_daily = aggregator.aggregate_to_daily(df_15m, symbol)
        all_symbols[symbol] = df_daily
    
    # Prepare walk-forward data
    wf_data = aggregator.prepare_walk_forward_data(all_symbols)
    print(f"\n✅ Walk-Forward Data: {len(wf_data)} days")
    print(wf_data.head(10))
    
    # Summary
    summary = aggregator.get_summary()
    print(f"\n📊 Summary:")
    print(f"   Symbols: {summary['symbols']}")
    for symbol in summary['symbols']:
        print(f"   {symbol}:")
        print(f"      Days: {summary['total_days'][symbol]}")
        print(f"      Range: {summary['date_ranges'][symbol]}")
        print(f"      Avg Sharpe: {summary['avg_sharpe'][symbol]:.3f}")
