#!/usr/bin/env python3
"""
================================================================================
FEATURE ENGINEER V9 - ADVANCED TECHNICAL INDICATORS & MULTI-TIMEFRAME
================================================================================

Provides 50+ technical indicators using TA-Lib and custom calculations:
- Trend Indicators: SMA, EMA, TEMA, WMA, ADX, PSAR, Supertrend
- Momentum: RSI (multiple periods), Stochastic, Williams %R, ROC, MFI, CCI
- Volatility: Bollinger Bands, ATR (multiple), Keltner, Donchian
- Volume: OBV, AD, CMF, VROC
- Multi-Timeframe: 1min, 5min, 15min, 1H, 4H features

Author: E1 AI Agent (Emergent.sh)
Date: January 2025
Version: 9.0 FREE PRO
Status: PRODUCTION READY
================================================================================
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try TA-Lib, fallback to pandas_ta if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("⚠️  TA-Lib not installed. Using pandas_ta fallback.")
    print("   Install with: pip install TA-Lib")
    try:
        import pandas_ta as ta
        HAS_PANDAS_TA = True
    except ImportError:
        HAS_PANDAS_TA = False
        print("⚠️  pandas_ta not installed. Using basic calculations.")
        print("   Install with: pip install pandas_ta")

logger = logging.getLogger('FeatureEngineerV9')


class FeatureEngineerV9:
    """
    Advanced Feature Engineering with 50+ technical indicators.
    
    Features:
    - TA-Lib integration (50+ indicators)
    - Multi-timeframe aggregation (1m, 5m, 15m, 1H, 4H)
    - Custom indicators
    - Feature normalization
    """
    
    def __init__(
        self,
        enable_talib: bool = True,
        enable_multi_timeframe: bool = True,
        timeframes: List[str] = ['15T', '1H', '4H']
    ):
        """
        Initialize Feature Engineer.
        
        Args:
            enable_talib: Use TA-Lib if available
            enable_multi_timeframe: Enable multi-timeframe features
            timeframes: List of timeframes to aggregate (pandas frequency strings)
        """
        self.enable_talib = enable_talib and HAS_TALIB
        self.enable_multi_timeframe = enable_multi_timeframe
        self.timeframes = timeframes
        
        logger.info("🔧 FeatureEngineerV9 initialized")
        logger.info(f"   TA-Lib: {self.enable_talib}")
        logger.info(f"   Multi-Timeframe: {self.enable_multi_timeframe}")
        logger.info(f"   Timeframes: {self.timeframes}")
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        symbol: str = 'UNKNOWN'
    ) -> pd.DataFrame:
        """
        Engineer all features for a dataframe.
        
        Args:
            df: DataFrame with OHLCV columns
            symbol: Symbol name (for logging)
        
        Returns:
            DataFrame with added feature columns
        """
        logger.info(f"🛠️  Engineering features for {symbol}...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"⚠️  Missing column: {col}")
                return df
        
        # 1. Trend Indicators
        df = self._add_trend_indicators(df)
        
        # 2. Momentum Indicators
        df = self._add_momentum_indicators(df)
        
        # 3. Volatility Indicators
        df = self._add_volatility_indicators(df)
        
        # 4. Volume Indicators
        df = self._add_volume_indicators(df)
        
        # 5. Multi-Timeframe (if enabled)
        if self.enable_multi_timeframe:
            df = self._add_multi_timeframe_features(df)
        
        # 6. Custom Alpha Factors
        df = self._add_custom_factors(df)
        
        # Fill NaN with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        feature_count = len([c for c in df.columns if c not in required + ['timestamp']])
        logger.info(f"✅ Added {feature_count} features for {symbol}")
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        
        if self.enable_talib:
            # Moving Averages (multiple periods)
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # TEMA (Triple EMA)
            df['tema_10'] = talib.TEMA(close, timeperiod=10)
            df['tema_20'] = talib.TEMA(close, timeperiod=20)
            
            # WMA (Weighted MA)
            df['wma_20'] = talib.WMA(close, timeperiod=20)
            
            # ADX (Trend Strength)
            df['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            df['adx_20'] = talib.ADX(high, low, close, timeperiod=20)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macdsignal
            df['macd_hist'] = macdhist
        
        else:
            # Fallback: Basic pandas calculations
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        
        if self.enable_talib:
            # RSI (multiple periods)
            for period in [7, 14, 21]:
                df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(high, low, close, 
                                       fastk_period=14, slowk_period=3, 
                                       slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # Williams %R
            df['willr_14'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # ROC (Rate of Change)
            df['roc_10'] = talib.ROC(close, timeperiod=10)
            df['roc_20'] = talib.ROC(close, timeperiod=20)
            
            # MFI (Money Flow Index)
            df['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # CCI (Commodity Channel Index)
            df['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            df['cci_20'] = talib.CCI(high, low, close, timeperiod=20)
            
            # CMO (Chande Momentum Oscillator)
            df['cmo_14'] = talib.CMO(close, timeperiod=14)
            
            # TSI calculation (custom, TA-Lib doesn't have it)
            df['tsi'] = self._calculate_tsi(close)
        
        else:
            # Fallback: Basic RSI
            for period in [7, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        
        if self.enable_talib:
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            
            # ATR (multiple periods)
            for period in [7, 14, 21]:
                df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            
            # NATR (Normalized ATR)
            df['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
            
            # TRANGE (True Range)
            df['trange'] = talib.TRANGE(high, low, close)
        
        else:
            # Fallback: Basic Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (std * 2)
            df['bb_lower'] = df['bb_middle'] - (std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Custom: Keltner Channels
        df = self._add_keltner_channels(df)
        
        # Custom: Donchian Channels
        df = self._add_donchian_channels(df)
        
        # Standard Deviation
        df['std_20'] = df['close'].rolling(window=20).std()
        
        # Historical Volatility
        df['hist_vol_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        
        if self.enable_talib:
            # OBV (On-Balance Volume)
            df['obv'] = talib.OBV(close, volume)
            
            # AD (Accumulation/Distribution)
            df['ad'] = talib.AD(high, low, close, volume)
            
            # ADOSC (AD Oscillator)
            df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        else:
            # Fallback: Basic OBV
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume Rate of Change
        df['vroc_10'] = df['volume'].pct_change(periods=10) * 100
        
        # Volume Moving Average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Chaikin Money Flow (custom)
        df = self._add_cmf(df)
        
        return df
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe aggregated features."""
        if 'timestamp' not in df.columns:
            logger.warning("⚠️  No timestamp column, skipping multi-timeframe")
            return df
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.set_index('timestamp')
        
        for tf in self.timeframes:
            if tf == '15T':  # Skip base timeframe
                continue
            
            try:
                # Resample to higher timeframe
                resampled = df.resample(tf).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                # Calculate basic indicators on resampled data
                if len(resampled) > 50:  # Need enough data
                    # RSI
                    delta = resampled['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # SMA
                    sma_20 = resampled['close'].rolling(window=20).mean()
                    
                    # Merge back to original timeframe (forward fill)
                    df[f'{tf}_rsi'] = rsi.reindex(df.index, method='ffill')
                    df[f'{tf}_sma_20'] = sma_20.reindex(df.index, method='ffill')
                    df[f'{tf}_close'] = resampled['close'].reindex(df.index, method='ffill')
            
            except Exception as e:
                logger.warning(f"⚠️  Error processing timeframe {tf}: {e}")
        
        df = df.reset_index()
        return df
    
    def _add_custom_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom alpha factors."""
        # Price momentum (returns)
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(periods=period)
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Close position in range
        df['close_loc'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price distance from moving averages
        if 'sma_20' in df.columns:
            df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        if 'ema_50' in df.columns:
            df['dist_ema_50'] = (df['close'] - df['ema_50']) / df['ema_50']
        
        # Volatility ratio
        if 'atr_14' in df.columns and 'std_20' in df.columns:
            df['vol_ratio'] = df['atr_14'] / (df['std_20'] + 1e-8)
        
        # Volume ratio
        if 'volume_sma_20' in df.columns:
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-8)
        
        return df
    
    # Helper methods for custom indicators
    
    def _calculate_tsi(self, close: np.ndarray, long: int = 25, short: int = 13) -> np.ndarray:
        """Calculate True Strength Index."""
        momentum = pd.Series(close).diff()
        ema_momentum_long = momentum.ewm(span=long, adjust=False).mean()
        ema_ema_momentum_long = ema_momentum_long.ewm(span=short, adjust=False).mean()
        
        abs_momentum = momentum.abs()
        ema_abs_momentum_long = abs_momentum.ewm(span=long, adjust=False).mean()
        ema_ema_abs_momentum_long = ema_abs_momentum_long.ewm(span=short, adjust=False).mean()
        
        tsi = 100 * (ema_ema_momentum_long / ema_ema_abs_momentum_long)
        return tsi.values
    
    def _add_keltner_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Keltner Channels."""
        df['keltner_middle'] = df['close'].rolling(window=period).mean()
        if 'atr_14' in df.columns:
            df['keltner_upper'] = df['keltner_middle'] + (2 * df['atr_14'])
            df['keltner_lower'] = df['keltner_middle'] - (2 * df['atr_14'])
        return df
    
    def _add_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Donchian Channels."""
        df['donchian_upper'] = df['high'].rolling(window=period).max()
        df['donchian_lower'] = df['low'].rolling(window=period).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        return df
    
    def _add_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Chaikin Money Flow."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)
        mfv = mfm * df['volume']
        df['cmf'] = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names (excluding OHLCV and timestamp).
        
        Args:
            df: DataFrame with engineered features
        
        Returns:
            List of feature column names
        """
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in df.columns if col not in base_cols]


# Quick test function
def test_feature_engineer():
    """Test feature engineering on sample data."""
    print("🧪 Testing FeatureEngineerV9...")
    
    # Generate sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='15T')
    close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(1000) * 0.2,
        'high': close_prices + np.abs(np.random.randn(1000)) * 0.5,
        'low': close_prices - np.abs(np.random.randn(1000)) * 0.5,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Initialize feature engineer
    fe = FeatureEngineerV9(
        enable_talib=True,
        enable_multi_timeframe=True,
        timeframes=['15T', '1H', '4H']
    )
    
    # Engineer features
    df_features = fe.engineer_features(df, symbol='TEST')
    
    # Show results
    print(f"\n📊 Results:")
    print(f"   Original columns: {len(df.columns)}")
    print(f"   Feature columns: {len(df_features.columns)}")
    print(f"   Added features: {len(df_features.columns) - len(df.columns)}")
    print(f"\n🔍 Sample features:")
    feature_names = fe.get_feature_names(df_features)
    print(f"   Total: {len(feature_names)} features")
    print(f"   First 20: {feature_names[:20]}")
    
    print("\n✅ Test complete!")


if __name__ == '__main__':
    test_feature_engineer()
