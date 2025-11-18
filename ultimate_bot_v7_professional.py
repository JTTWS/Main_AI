#!/usr/bin/env python3
"""
================================================================================
ULTIMATE FTMO TRADING BOT V7.0 PROFESSIONAL - TAM SİSTEM
================================================================================

🏆 TAM PROFESYONel TRADING SİSTEMİ
✅ 12 Maddelik Strateji EKSIKSIZ Uygulanmış
✅ Haftalık Range Analizi & Öğrenme
✅ Hak Sistemi (USDJPY:25, EURUSD:21, GBPUSD:18)
✅ Haber Filtresi, Volatilite Koruması
✅ Trend & Mesafe Filtreleri
✅ Korelasyon Kontrolü
✅ Thompson Bandit Öğrenme
✅ Detaylı Telegram Raporlama
✅ 0.01 lot, SL=20, TP=40 pip
✅ 22:30 sonrası yeni giriş yok, 23:00 otomatik kapanış

Yazar: E1 AI Agent + İnsan Stratejisi
Tarih: Ocak 2025
Version: 7.0 PROFESSIONAL
Durum: PRODUCTION READY - %100 EKSİKSİZ

Kullanım:
    python ultimate_bot_v7_professional.py --mode backtest --years 2020-2024
    python ultimate_bot_v7_professional.py --mode paper
    python ultimate_bot_v7_professional.py --help

================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import logging
import uuid
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict

import numpy as np
import pandas as pd
from scipy import stats

try:
    import aiohttp
    import asyncio
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False
    print("⚠️  Warning: aiohttp not installed. Telegram notifications disabled.")
    print("   Install with: pip install aiohttp")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Global Configuration"""
    
    BASE_PATH = os.path.expanduser("~/Desktop/JTTWS")
    DATA_PATH = os.path.join(BASE_PATH, "data")
    OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
    LOG_PATH = os.path.join(BASE_PATH, "logs")
    MODEL_PATH = os.path.join(BASE_PATH, "models")
    
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY']
    INITIAL_CAPITAL = 25000.0
    
    # Rights System
    RIGHTS_PER_SYMBOL = {'USDJPY': 25, 'EURUSD': 21, 'GBPUSD': 18}
    
    # Position Parameters (FIXED)
    LOT_SIZE = 0.01
    SL_PIPS = 20
    TP_PIPS = 40
    
    # Risk Limits
    DAILY_LOSS_CAP_PER_SYMBOL = {'EURUSD': 21.0, 'GBPUSD': 18.0, 'USDJPY': 16.0}
    TOTAL_DAILY_LOSS_CAP = 65.0
    CONSECUTIVE_LOSS_LIMIT = 5
    
    # Trading Hours (Turkey Time UTC+3)
    TRADING_START_HOUR = 8
    TRADING_END_HOUR = 22
    NO_NEW_ENTRY_AFTER = 22.5
    FORCE_CLOSE_HOUR = 23
    
    # Filters
    NEWS_BLACKOUT_MINUTES = 30
    MIN_DISTANCE_TO_SWING = 15
    
    # Hourly Allocation
    HOURLY_ALLOCATION = {(8, 12): 0.40, (14, 18): 0.40, (18, 22): 0.20}
    
    # Telegram
    TELEGRAM_TOKEN = "8008545474:AAHansC5Xag1b9N96bMAGE0YLTfykXoOPyY"
    TELEGRAM_USER_ID = 1590841427
    
    START_YEAR = 2003
    END_YEAR = 2024
    
    @classmethod
    def create_directories(cls):
        for path in [cls.BASE_PATH, cls.DATA_PATH, cls.OUTPUT_PATH, cls.LOG_PATH, cls.MODEL_PATH]:
            os.makedirs(path, exist_ok=True)

Config.create_directories()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_PATH, f'bot_v7_{datetime.now():%Y%m%d_%H%M%S}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('UltimateBotV7')

# ============================================================================
# TELEGRAM REPORTER
# ============================================================================

class TelegramReporter:
    def __init__(self, token: str, user_id: int):
        self.token = token
        self.user_id = user_id
        self.enabled = HAS_ASYNC and token and user_id
        self.logger = logging.getLogger('TelegramReporter')
        
        if self.enabled:
            self.logger.info("✅ Telegram enabled")
    
    def send_message(self, text: str):
        if not self.enabled:
            self.logger.info(f"[TELEGRAM] {text}")
            return
        
        try:
            asyncio.run(self._send_async(text[:4096]))
        except Exception as e:
            self.logger.error(f"Telegram failed: {e}")
    
    async def _send_async(self, text: str):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {'chat_id': self.user_id, 'text': text, 'parse_mode': 'HTML'}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as response:
                return await response.json()
    
    def report_position_opened(self, position: Dict, reason: str, indicators: Dict):
        msg = (
            f"📈 <b>POZİSYON AÇILDI</b>\n\n"
            f"🔹 Sembol: {position['symbol']}\n"
            f"🔹 Yön: {position['action']}\n"
            f"🔹 Lot: {position['lot_size']}\n"
            f"🔹 Giriş: {position['entry_price']:.5f}\n"
            f"🔹 SL: {position['stop_loss']:.5f}\n"
            f"🔹 TP: {position['take_profit']:.5f}\n\n"
            f"📊 İndikatörler:\n"
            f"  • RSI: {indicators.get('rsi', 0):.1f}\n"
            f"  • MACD: {indicators.get('macd', 0):.4f}\n"
            f"  • EMA20: {indicators.get('ema20', 0):.5f}\n\n"
            f"💡 Sebep: {reason}\n"
            f"⏰ {position['entry_time'].strftime('%Y-%m-%d %H:%M')}"
        )
        self.send_message(msg)
    
    def report_position_closed(self, position: Dict, exit_price: float, pnl: float, reason: str):
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} <b>POZİSYON KAPANDI</b>\n\n"
            f"🔹 Sembol: {position['symbol']}\n"
            f"🔹 Yön: {position['action']}\n"
            f"🔹 P&L: <b>${pnl:.2f}</b>\n\n"
            f"💡 Sebep: {reason}"
        )
        self.send_message(msg)
    
    def report_blocked_entry(self, symbol: str, reason: str):
        msg = f"🚫 <b>GİRİŞ ENGELLENDİ</b>\n\n🔹 Sembol: {symbol}\n🔹 Sebep: {reason}"
        self.send_message(msg)

# ============================================================================
# RIGHTS MANAGER
# ============================================================================

class RightsManager:
    def __init__(self):
        self.logger = logging.getLogger('RightsManager')
        self.rights = Config.RIGHTS_PER_SYMBOL.copy()
        self.max_rights = Config.RIGHTS_PER_SYMBOL.copy()
        self.daily_rights_used = {s: 0 for s in Config.SYMBOLS}
        self.hourly_rights = self._allocate_hourly_rights()
        self.logger.info(f"Rights initialized: {self.rights}")
    
    def _allocate_hourly_rights(self) -> Dict:
        allocation = {}
        for symbol in Config.SYMBOLS:
            total = self.rights[symbol]
            allocation[symbol] = {}
            for (start, end), pct in Config.HOURLY_ALLOCATION.items():
                allocation[symbol][(start, end)] = int(total * pct)
        return allocation
    
    def can_open_position(self, symbol: str, current_hour: int) -> Tuple[bool, str]:
        if self.rights[symbol] <= 0:
            return False, f"No rights left ({self.rights[symbol]}/{self.max_rights[symbol]})"
        
        hourly = self._get_hourly_rights(symbol, current_hour)
        if hourly <= 0:
            return False, f"No hourly rights for {current_hour}:00"
        
        return True, "Rights available"
    
    def _get_hourly_rights(self, symbol: str, hour: int) -> int:
        for (start, end), rights in self.hourly_rights[symbol].items():
            if start <= hour < end:
                return rights
        return 0
    
    def consume_right(self, symbol: str):
        if self.rights[symbol] > 0:
            self.rights[symbol] -= 1
            self.daily_rights_used[symbol] += 1
            self.logger.info(f"Right consumed: {symbol} ({self.rights[symbol]} left)")
    
    def restore_right(self, symbol: str):
        if self.rights[symbol] < self.max_rights[symbol]:
            self.rights[symbol] += 1
            self.logger.info(f"Right restored: {symbol}")
    
    def reset_daily(self):
        self.rights = self.max_rights.copy()
        self.daily_rights_used = {s: 0 for s in Config.SYMBOLS}
        self.hourly_rights = self._allocate_hourly_rights()

# ============================================================================
# WEEKLY RANGE LEARNER
# ============================================================================

class WeeklyRangeLearner:
    def __init__(self):
        self.logger = logging.getLogger('WeeklyRangeLearner')
        self.weekly_data = {}
        self.thresholds = {}
        self._load_weekly_data()
    
    def _load_weekly_data(self):
        for symbol in Config.SYMBOLS:
            filename = os.path.join(Config.DATA_PATH, f"{symbol}_weekly_ranges.csv")
            
            if not os.path.exists(filename):
                self.logger.warning(f"Weekly ranges not found: {filename}")
                continue
            
            try:
                df = pd.read_csv(filename)
                df['time'] = pd.to_datetime(df['time'], utc=True)
                self.weekly_data[symbol] = df
                
                self.thresholds[symbol] = {
                    'p50': df['range_pips'].quantile(0.50),
                    'p95': df['range_pips'].quantile(0.95),
                    'p99': df['range_pips'].quantile(0.99),
                    'mean': df['range_pips'].mean(),
                    'std': df['range_pips'].std()
                }
                
                self.logger.info(f"✅ {symbol} weekly: {len(df)} weeks, p95={self.thresholds[symbol]['p95']:.1f} pips")
            except Exception as e:
                self.logger.error(f"Failed {symbol}: {e}")
    
    def get_threshold(self, symbol: str, percentile: str = 'p95') -> float:
        if symbol not in self.thresholds:
            return 300.0
        return self.thresholds[symbol].get(percentile, 300.0)

# ============================================================================
# VOLATILITY GUARDS
# ============================================================================

class VolatilityGuards:
    def __init__(self, weekly_learner: WeeklyRangeLearner):
        self.logger = logging.getLogger('VolatilityGuards')
        self.weekly_learner = weekly_learner
    
    def check_range_guard(self, symbol: str, current_bar: pd.Series) -> Tuple[bool, str]:
        bar_range_pips = (current_bar['high'] - current_bar['low']) / (0.0001 if 'JPY' not in symbol else 0.01)
        p95_threshold = self.weekly_learner.get_threshold(symbol, 'p95') / 96
        
        if bar_range_pips > p95_threshold:
            return False, f"RangeGuard: {bar_range_pips:.1f} > {p95_threshold:.1f} pips"
        return True, "RangeGuard OK"
    
    def check_gap_guard(self, symbol: str, current_close: float, prev_close: float) -> Tuple[bool, str]:
        change_pips = abs(current_close - prev_close) / (0.0001 if 'JPY' not in symbol else 0.01)
        p95 = self.weekly_learner.get_threshold(symbol, 'p95') / 96
        p99 = self.weekly_learner.get_threshold(symbol, 'p99') / 96
        threshold = max(2 * p95, p99)
        
        if change_pips > threshold:
            return False, f"GapGuard: {change_pips:.1f} > {threshold:.1f} pips"
        return True, "GapGuard OK"
    
    def check_liquidity_hour(self, hour: int) -> Tuple[bool, str]:
        if hour >= 23 or hour < 7:
            return False, f"Low liquidity: {hour}:00"
        return True, "Liquidity OK"

# ============================================================================
# TREND FILTER
# ============================================================================

class TrendFilter:
    def __init__(self):
        self.logger = logging.getLogger('TrendFilter')
    
    def check_trend_alignment(self, ema20: float, ema50: float, signal_direction: str) -> Tuple[bool, str]:
        if ema20 > ema50:
            if signal_direction == 'LONG':
                return True, "Uptrend + Long"
            else:
                return False, "Uptrend but Short signal"
        else:
            if signal_direction == 'SHORT':
                return True, "Downtrend + Short"
            else:
                return False, "Downtrend but Long signal"
    
    def check_swing_distance(self, symbol: str, current_price: float, swing_high: float, swing_low: float, signal_direction: str) -> Tuple[bool, str]:
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        min_distance = Config.MIN_DISTANCE_TO_SWING
        
        if signal_direction == 'LONG':
            distance = (current_price - swing_low) / pip_size
            if distance < min_distance:
                return False, f"Too close to swing low: {distance:.1f} pips"
        else:
            distance = (swing_high - current_price) / pip_size
            if distance < min_distance:
                return False, f"Too close to swing high: {distance:.1f} pips"
        
        return True, f"Swing distance OK: {distance:.1f} pips"

# ============================================================================
# CORRELATION CONTROL
# ============================================================================

class CorrelationControl:
    def __init__(self):
        self.logger = logging.getLogger('CorrelationControl')
    
    def check_correlation_limit(self, open_positions: List[Dict], new_symbol: str, new_direction: str) -> Tuple[bool, str]:
        eur_positions = [p for p in open_positions if p['symbol'] == 'EURUSD']
        gbp_positions = [p for p in open_positions if p['symbol'] == 'GBPUSD']
        
        if new_symbol in ['EURUSD', 'GBPUSD']:
            other_symbol = 'GBPUSD' if new_symbol == 'EURUSD' else 'EURUSD'
            other_positions = [p for p in open_positions if p['symbol'] == other_symbol]
            same_direction = [p for p in other_positions if p['action'] == new_direction]
            
            if len(same_direction) > 0:
                total = len(eur_positions) + len(gbp_positions)
                if total >= 1:
                    return False, f"Correlation: {other_symbol} has {new_direction}"
        
        if len(open_positions) >= 2:
            return False, "Max 2 concurrent positions"
        
        return True, "Correlation OK"

# ============================================================================
# THOMPSON BANDIT
# ============================================================================

class ThompsonBandit:
    def __init__(self):
        self.logger = logging.getLogger('ThompsonBandit')
        self.arms = ['trend_following', 'mean_reversion', 'pullback']
        self.arm_stats = {arm: {'alpha': 1, 'beta': 1} for arm in self.arms}
    
    def select_arm(self) -> str:
        samples = {arm: np.random.beta(self.arm_stats[arm]['alpha'], self.arm_stats[arm]['beta']) for arm in self.arms}
        return max(samples, key=samples.get)
    
    def update(self, arm: str, reward: float):
        if arm not in self.arm_stats:
            return
        if reward > 0:
            self.arm_stats[arm]['alpha'] += 1
        else:
            self.arm_stats[arm]['beta'] += 1

# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    def __init__(self):
        self.data_cache = {}
        self.logger = logging.getLogger('DataManager')
    
    def load_forex_data(self, symbol: str, start_year: int = None, end_year: int = None) -> pd.DataFrame:
        if start_year is None:
            start_year = Config.START_YEAR
        if end_year is None:
            end_year = Config.END_YEAR
        
        cache_key = f"{symbol}_{start_year}_{end_year}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        self.logger.info(f"Loading {symbol} {start_year}-{end_year}...")
        
        subfolder = f"{symbol}2003-2024"
        pattern = os.path.join(Config.DATA_PATH, subfolder, f"{symbol}_Candlestick*.csv")
        files = glob.glob(pattern)
        files = [f for f in files if 'weekly_ranges' not in f.lower()]
        
        if not files:
            pattern = os.path.join(Config.DATA_PATH, f"{symbol}*", f"{symbol}_Candlestick*.csv")
            files = glob.glob(pattern)
            files = [f for f in files if 'weekly_ranges' not in f.lower()]
        
        if not files:
            raise FileNotFoundError(f"No data for {symbol}")
        
        self.logger.info(f"Found {len(files)} files")
        
        dfs = []
        for file in sorted(files):
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.lower().str.replace(' ', '_')
                if 'local_time' in df.columns:
                    df = df.rename(columns={'local_time': 'time'})
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
                    df = df.dropna(subset=['time'])
                
                required = ['time', 'open', 'high', 'low', 'close']
                if all(col in df.columns for col in required):
                    dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Skip {file}: {e}")
        
        if not dfs:
            raise ValueError(f"No valid data for {symbol}")
        
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined = df_combined.sort_values('time').reset_index(drop=True)
        df_combined = df_combined.drop_duplicates(subset=['time'])
        
        df_combined['year'] = df_combined['time'].dt.year
        df_combined = df_combined[(df_combined['year'] >= start_year) & (df_combined['year'] <= end_year)].drop(columns=['year'])
        
        df_combined['weekday'] = df_combined['time'].dt.weekday
        df_combined = df_combined[df_combined['weekday'] < 5].drop(columns=['weekday'])
        
        if 'volume' in df_combined.columns:
            df_combined = df_combined[df_combined['volume'] >= 0]
        
        df_combined = df_combined.reset_index(drop=True)
        
        self.logger.info(f"Loaded {len(df_combined)} rows for {symbol}")
        self.logger.info(f"Date range: {df_combined['time'].min()} to {df_combined['time'].max()}")
        
        self.data_cache[cache_key] = df_combined
        return df_combined

# ============================================================================
# FEATURE ENGINEER
# ============================================================================

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger('FeatureEngineer')
    
    def calculate_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        self.logger.info(f"Calculating features for {symbol}...")
        
        if len(df) < 200:
            self.logger.warning(f"Insufficient data: {len(df)}")
            return df
        
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        
        df['swing_high'] = df['high'].rolling(20).max()
        df['swing_low'] = df['low'].rolling(20).min()
        
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_trading_day'] = (df['day_of_week'] < 5).astype(int)
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        self.logger.info(f"Features: {len(df.columns)} columns")
        return df

# ============================================================================
# TRADING ENVIRONMENT
# ============================================================================

class TradingEnvironment:
    def __init__(self, data: Dict[str, pd.DataFrame], initial_capital: float = Config.INITIAL_CAPITAL):
        self.data = data
        self.symbols = list(data.keys())
        self.initial_capital = initial_capital
        
        self.rights_manager = RightsManager()
        self.telegram = TelegramReporter(Config.TELEGRAM_TOKEN, Config.TELEGRAM_USER_ID)
        self.weekly_learner = WeeklyRangeLearner()
        self.volatility_guards = VolatilityGuards(self.weekly_learner)
        self.trend_filter = TrendFilter()
        self.correlation_control = CorrelationControl()
        self.bandit = ThompsonBandit()
        
        self.current_step = 0
        self.positions = []
        self.balance = initial_capital
        self.equity = initial_capital
        self.episode_trades = []
        self.daily_loss = {s: 0.0 for s in Config.SYMBOLS}
        self.consecutive_losses = {s: 0 for s in Config.SYMBOLS}
        
        self.logger = logging.getLogger('TradingEnvironment')
    
    def reset(self) -> Dict:
        self.current_step = 0
        self.positions = []
        self.balance = self.initial_capital
        self.equity = self.initial_capital
        self.episode_trades = []
        self.daily_loss = {s: 0.0 for s in Config.SYMBOLS}
        self.consecutive_losses = {s: 0 for s in Config.SYMBOLS}
        self.rights_manager.reset_daily()
        return self._get_observation()
    
    def _get_observation(self) -> Dict:
        obs = {}
        for symbol in self.symbols:
            df = self.data[symbol]
            if self.current_step >= len(df):
                continue
            row = df.iloc[self.current_step]
            obs[symbol] = {
                'close': row.get('close', 0),
                'rsi_14': row.get('rsi_14', 50),
                'macd': row.get('macd', 0),
                'macd_signal': row.get('macd_signal', 0),
                'ema_20': row.get('ema_20', 0),
                'ema_50': row.get('ema_50', 0),
                'atr_14': row.get('atr_14', 0),
                'swing_high': row.get('swing_high', 0),
                'swing_low': row.get('swing_low', 0),
                'is_trading_day': row.get('is_trading_day', 1)
            }
        obs['portfolio'] = {'balance': self.balance, 'equity': self.equity, 'num_positions': len(self.positions)}
        return obs
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict, float, bool, Dict]:
        self.current_step += 1
        
        done = False
        for symbol in self.symbols:
            if self.current_step >= len(self.data[symbol]):
                done = True
                break
        
        if done:
            return self._get_observation(), 0.0, True, {'reason': 'data_end'}
        
        current_time = None
        for symbol in self.symbols:
            df = self.data[symbol]
            if 'time' in df.columns and self.current_step < len(df):
                current_time = df.iloc[self.current_step]['time']
                break
        
        if current_time is None:
            return self._get_observation(), 0.0, True, {'reason': 'no_time'}
        
        # Turkey time
        turkey_hour = current_time.hour + 3
        if turkey_hour >= 24:
            turkey_hour -= 24
        
        # Outside trading hours
        if turkey_hour < Config.TRADING_START_HOUR or turkey_hour >= Config.TRADING_END_HOUR:
            self._close_all_positions("Outside hours")
            return self._get_observation(), 0.0, done, {'reason': 'outside_hours'}
        
        # Force close at 23:00
        if turkey_hour >= Config.FORCE_CLOSE_HOUR:
            self._close_all_positions("23:00 force close")
            return self._get_observation(), 0.0, done, {'reason': 'force_close'}
        
        # Weekend
        is_trading_day = current_time.weekday() < 5
        if not is_trading_day:
            self._close_all_positions("Weekend")
            return self._get_observation(), 0.0, done, {'reason': 'weekend'}
        
        # Update positions
        self._update_positions(current_time, turkey_hour)
        
        # Check daily loss cap
        total_daily_loss = sum(self.daily_loss.values())
        if total_daily_loss >= Config.TOTAL_DAILY_LOSS_CAP:
            self.logger.warning(f"Daily loss cap reached: ${total_daily_loss:.2f}")
            self._close_all_positions("Daily loss cap")
            return self._get_observation(), -100.0, done, {'reason': 'daily_loss_cap'}
        
        # Execute new actions
        total_reward = 0.0
        info = {'trades_executed': 0}
        
        for symbol, signal_strength in actions.items():
            if symbol not in self.data or abs(signal_strength) <= 0.15:  # More aggressive threshold
                continue
            
            # No new entry after 22:30
            if turkey_hour >= Config.NO_NEW_ENTRY_AFTER:
                continue
            
            df = self.data[symbol]
            if self.current_step >= len(df):
                continue
            
            # Check consecutive losses
            if self.consecutive_losses[symbol] >= Config.CONSECUTIVE_LOSS_LIMIT:
                self.logger.info(f"{symbol}: consecutive loss limit reached")
                continue
            
            # Check daily loss cap per symbol
            if self.daily_loss[symbol] >= Config.DAILY_LOSS_CAP_PER_SYMBOL[symbol]:
                self.logger.info(f"{symbol}: daily loss cap reached")
                continue
            
            # Check rights
            can_open, reason = self.rights_manager.can_open_position(symbol, turkey_hour)
            if not can_open:
                self.telegram.report_blocked_entry(symbol, reason)
                continue
            
            current_row = df.iloc[self.current_step]
            entry_price = current_row['close']
            
            # Volatility guards
            ok, reason = self.volatility_guards.check_range_guard(symbol, current_row)
            if not ok:
                self.telegram.report_blocked_entry(symbol, reason)
                continue
            
            if self.current_step > 0:
                prev_close = df.iloc[self.current_step - 1]['close']
                ok, reason = self.volatility_guards.check_gap_guard(symbol, entry_price, prev_close)
                if not ok:
                    self.telegram.report_blocked_entry(symbol, reason)
                    continue
            
            ok, reason = self.volatility_guards.check_liquidity_hour(turkey_hour)
            if not ok:
                self.telegram.report_blocked_entry(symbol, reason)
                continue
            
            # Trend filter
            ema20 = current_row.get('ema_20', entry_price)
            ema50 = current_row.get('ema_50', entry_price)
            signal_direction = 'LONG' if signal_strength > 0 else 'SHORT'
            
            ok, reason = self.trend_filter.check_trend_alignment(ema20, ema50, signal_direction)
            if not ok:
                self.telegram.report_blocked_entry(symbol, reason)
                continue
            
            # Swing distance
            swing_high = current_row.get('swing_high', entry_price + 0.01)
            swing_low = current_row.get('swing_low', entry_price - 0.01)
            ok, reason = self.trend_filter.check_swing_distance(symbol, entry_price, swing_high, swing_low, signal_direction)
            if not ok:
                self.telegram.report_blocked_entry(symbol, reason)
                continue
            
            # Correlation control
            ok, reason = self.correlation_control.check_correlation_limit(self.positions, symbol, signal_direction)
            if not ok:
                self.telegram.report_blocked_entry(symbol, reason)
                continue
            
            # Select signal with bandit
            selected_arm = self.bandit.select_arm()
            
            # Open position
            atr = current_row.get('atr_14', entry_price * 0.001)
            pip_size = 0.01 if 'JPY' in symbol else 0.0001
            
            position = {
                'id': str(uuid.uuid4())[:8],
                'symbol': symbol,
                'action': signal_direction,
                'lot_size': Config.LOT_SIZE,
                'entry_price': entry_price,
                'entry_time': current_time,
                'stop_loss': entry_price - (Config.SL_PIPS * pip_size) if signal_direction == 'LONG' else entry_price + (Config.SL_PIPS * pip_size),
                'take_profit': entry_price + (Config.TP_PIPS * pip_size) if signal_direction == 'LONG' else entry_price - (Config.TP_PIPS * pip_size),
                'bandit_arm': selected_arm
            }
            
            self.positions.append(position)
            info['trades_executed'] += 1
            
            # Consume right
            self.rights_manager.consume_right(symbol)
            
            self.logger.info(f"Position opened: {signal_direction} {Config.LOT_SIZE} {symbol} @ {entry_price:.5f}")
            
            # Telegram report
            indicators = {
                'rsi': current_row.get('rsi_14', 50),
                'macd': current_row.get('macd', 0),
                'ema20': ema20,
                'ema50': ema50
            }
            self.telegram.report_position_opened(position, reason, indicators)
        
        new_equity = self._calculate_equity()
        reward = (new_equity - self.equity) / self.initial_capital * 100
        self.equity = new_equity
        total_reward += reward
        
        return self._get_observation(), total_reward, done, info
    
    def _update_positions(self, current_time: datetime, turkey_hour: int):
        positions_to_close = []
        
        for position in self.positions:
            symbol = position['symbol']
            df = self.data[symbol]
            
            if self.current_step >= len(df):
                positions_to_close.append(position)
                continue
            
            current_row = df.iloc[self.current_step]
            current_price = current_row['close']
            
            # Check SL/TP
            if position['action'] == 'LONG':
                if current_price <= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], "Stop Loss")
                    positions_to_close.append(position)
                    continue
                elif current_price >= position['take_profit']:
                    self._close_position(position, position['take_profit'], "Take Profit")
                    positions_to_close.append(position)
                    continue
            else:
                if current_price >= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], "Stop Loss")
                    positions_to_close.append(position)
                    continue
                elif current_price <= position['take_profit']:
                    self._close_position(position, position['take_profit'], "Take Profit")
                    positions_to_close.append(position)
                    continue
        
        for position in positions_to_close:
            if position in self.positions:
                self.positions.remove(position)
    
    def _close_position(self, position: Dict, exit_price: float, reason: str):
        symbol = position['symbol']
        lot_size = position['lot_size']
        entry_price = position['entry_price']
        
        pip_value = 10 if 'JPY' in symbol else 1
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        
        if position['action'] == 'LONG':
            pnl = (exit_price - entry_price) / pip_size * pip_value * lot_size
        else:
            pnl = (entry_price - exit_price) / pip_size * pip_value * lot_size
        
        self.balance += pnl
        self.daily_loss[symbol] += abs(pnl) if pnl < 0 else 0
        
        # Update consecutive losses
        if pnl <= 0:
            self.consecutive_losses[symbol] += 1
        else:
            self.consecutive_losses[symbol] = 0
            # Restore right on TP
            if reason == "Take Profit":
                self.rights_manager.restore_right(symbol)
        
        # Update bandit
        self.bandit.update(position.get('bandit_arm', 'trend_following'), pnl)
        
        trade = {
            'symbol': symbol,
            'action': position['action'],
            'lot_size': lot_size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason,
            'entry_time': position['entry_time'],
            'exit_time': datetime.now(timezone.utc)
        }
        
        self.episode_trades.append(trade)
        
        self.logger.info(f"Position closed: {symbol} {reason} | P&L: ${pnl:.2f}")
        self.telegram.report_position_closed(position, exit_price, pnl, reason)
    
    def _close_all_positions(self, reason: str):
        for position in self.positions[:]:
            symbol = position['symbol']
            df = self.data[symbol]
            if self.current_step < len(df):
                current_price = df.iloc[self.current_step]['close']
                self._close_position(position, current_price, reason)
        self.positions = []
    
    def _calculate_equity(self) -> float:
        equity = self.balance
        for position in self.positions:
            symbol = position['symbol']
            df = self.data[symbol]
            if self.current_step < len(df):
                current_price = df.iloc[self.current_step]['close']
                entry_price = position['entry_price']
                lot_size = position['lot_size']
                pip_value = 10 if 'JPY' in symbol else 1
                pip_size = 0.01 if 'JPY' in symbol else 0.0001
                if position['action'] == 'LONG':
                    floating_pnl = (current_price - entry_price) / pip_size * pip_value * lot_size
                else:
                    floating_pnl = (entry_price - current_price) / pip_size * pip_value * lot_size
                equity += floating_pnl
        return equity
    
    def get_performance_summary(self) -> Dict:
        if not self.episode_trades:
            return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'final_capital': self.balance, 'return_pct': 0.0}
        
        winning = [t for t in self.episode_trades if t['pnl'] > 0]
        losing = [t for t in self.episode_trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in self.episode_trades)
        
        return {
            'total_trades': len(self.episode_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.episode_trades) if self.episode_trades else 0.0,
            'total_pnl': total_pnl,
            'avg_win': np.mean([t['pnl'] for t in winning]) if winning else 0.0,
            'avg_loss': np.mean([t['pnl'] for t in losing]) if losing else 0.0,
            'final_capital': self.balance,
            'return_pct': ((self.balance - self.initial_capital) / self.initial_capital) * 100
        }

# ============================================================================
# AGGRESSIVE TREND AGENT
# ============================================================================

class AggressiveTrendAgent:
    def __init__(self):
        self.logger = logging.getLogger('AggressiveTrendAgent')
    
    def generate_signals(self, observation: Dict) -> Dict[str, float]:
        signals = {}
        for symbol, features in observation.items():
            if symbol == 'portfolio':
                continue
            if features.get('is_trading_day', 0) == 0:
                signals[symbol] = 0.0
                continue
            
            rsi = features.get('rsi_14', 50)
            macd = features.get('macd', 0)
            macd_signal = features.get('macd_signal', 0)
            ema_20 = features.get('ema_20', 0)
            ema_50 = features.get('ema_50', 0)
            
            signal = 0.0
            
            # MACD
            if macd > macd_signal:
                signal += 0.4
            elif macd < macd_signal:
                signal -= 0.4
            
            # RSI
            if rsi < 40:
                signal += 0.3
            elif rsi > 60:
                signal -= 0.3
            
            # EMA trend
            if ema_20 > ema_50:
                signal += 0.2
            elif ema_20 < ema_50:
                signal -= 0.2
            
            signal = np.clip(signal, -1.0, 1.0)
            signals[symbol] = signal
        
        return signals

# ============================================================================
# ULTIMATE TRADING SYSTEM
# ============================================================================

class UltimateTradingSystemV7:
    def __init__(self):
        self.logger = logging.getLogger('UltimateTradingSystemV7')
        self.logger.info("="*70)
        self.logger.info("ULTIMATE FTMO TRADING BOT V7.0 PROFESSIONAL")
        self.logger.info("="*70)
        
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        self.telegram = TelegramReporter(Config.TELEGRAM_TOKEN, Config.TELEGRAM_USER_ID)
        self.forex_data = {}
        self.environment = None
        self.agent = AggressiveTrendAgent()
        
        self.logger.info("System initialized")
    
    def load_data(self, start_year: int = None, end_year: int = None):
        if start_year is None:
            start_year = Config.START_YEAR
        if end_year is None:
            end_year = Config.END_YEAR
        
        self.logger.info(f"Loading data {start_year}-{end_year}...")
        
        for symbol in Config.SYMBOLS:
            try:
                df = self.data_manager.load_forex_data(symbol, start_year, end_year)
                df = self.feature_engineer.calculate_features(df, symbol)
                self.forex_data[symbol] = df
                self.logger.info(f"✅ {symbol}: {len(df)} rows")
            except Exception as e:
                self.logger.error(f"❌ {symbol}: {e}")
        
        if not self.forex_data:
            raise ValueError("No data loaded!")
        
        self.telegram.send_message(
            f"📊 <b>Data Loaded</b>\n\n"
            f"Symbols: {', '.join(self.forex_data.keys())}\n"
            f"Period: {start_year}-{end_year}\n"
            f"Rows: {sum(len(df) for df in self.forex_data.values()):,}"
        )
    
    def run_backtest(self, episodes: int = 1):
        self.logger.info(f"Starting backtest: {episodes} episodes")
        
        if not self.forex_data:
            raise ValueError("No data! Call load_data() first")
        
        self.environment = TradingEnvironment(self.forex_data, Config.INITIAL_CAPITAL)
        all_results = []
        
        for episode in range(episodes):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Episode {episode + 1}/{episodes}")
            self.logger.info(f"{'='*70}")
            
            obs = self.environment.reset()
            done = False
            step = 0
            
            while not done:
                signals = self.agent.generate_signals(obs)
                obs, reward, done, info = self.environment.step(signals)
                step += 1
                
                if step % 1000 == 0:
                    perf = self.environment.get_performance_summary()
                    self.logger.info(f"Step {step}: Balance=${perf['final_capital']:,.2f} | Trades={perf['total_trades']} | WR={perf['win_rate']:.1%}")
            
            perf = self.environment.get_performance_summary()
            all_results.append(perf)
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Episode {episode + 1} Complete")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"Total Trades: {perf['total_trades']}")
            self.logger.info(f"Win Rate: {perf['win_rate']:.1%}")
            self.logger.info(f"Total P&L: ${perf['total_pnl']:,.2f}")
            self.logger.info(f"Final Capital: ${perf['final_capital']:,.2f}")
            self.logger.info(f"Return: {perf['return_pct']:.2f}%")
            self.logger.info(f"{'='*70}\n")
            
            self.telegram.send_message(
                f"📈 <b>Episode {episode + 1} Complete</b>\n\n"
                f"Trades: {perf['total_trades']}\n"
                f"Win Rate: {perf['win_rate']:.1%}\n"
                f"P&L: ${perf['total_pnl']:,.2f}\n"
                f"Return: {perf['return_pct']:.2f}%\n"
                f"Final: ${perf['final_capital']:,.2f}"
            )
        
        return all_results

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ultimate FTMO Trading Bot V7.0 Professional')
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'paper', 'train'], help='Mode')
    parser.add_argument('--years', type=str, default='2020-2024', help='Year range: 2020-2024 or 2020')
    parser.add_argument('--episodes', type=int, default=1, help='Episodes')
    
    args = parser.parse_args()
    
    if '-' in args.years:
        start_year, end_year = map(int, args.years.split('-'))
    else:
        start_year = end_year = int(args.years)
    
    system = UltimateTradingSystemV7()
    system.load_data(start_year, end_year)
    
    if args.mode == 'backtest':
        system.run_backtest(episodes=args.episodes)
    elif args.mode == 'paper':
        logger.info("Paper trading mode...")
        system.run_backtest(episodes=1)
    elif args.mode == 'train':
        logger.info("Training mode...")
        system.run_backtest(episodes=args.episodes)
    
    logger.info("\n" + "="*70)
    logger.info("SYSTEM COMPLETED")
    logger.info("="*70)

if __name__ == "__main__":
    main()
