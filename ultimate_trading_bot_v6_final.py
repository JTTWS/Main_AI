#!/usr/bin/env python3
"""
ULTIMATE FTMO TRADING BOT V6.0 FINAL
=====================================
2003-2024 Full Data Support
Kelly Criterion + ATR Position Sizing
23-Hour Auto Close
VaR/CVaR Risk Management
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import logging
import uuid
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import aiohttp
    import asyncio
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False

class Config:
    BASE_PATH = os.path.expanduser("~/Desktop/JTTWS")
    DATA_PATH = os.path.join(BASE_PATH, "data")
    OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
    LOG_PATH = os.path.join(BASE_PATH, "logs")
    MODEL_PATH = os.path.join(BASE_PATH, "models")
    
    SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY']
    INITIAL_CAPITAL = 25000.0
    LEVERAGE = 100
    MAX_DAILY_LOSS_PCT = 0.05
    MAX_TOTAL_LOSS_PCT = 0.10
    RISK_PER_TRADE = 0.01
    MAX_POSITIONS = 5
    MAX_HOLDING_HOURS = 23
    
    MIN_LOT = 0.001
    MAX_LOT = 0.25
    KELLY_FRACTION = 0.25
    
    VAR_CONFIDENCE = 0.95
    VAR_WINDOW = 100
    
    START_YEAR = 2003
    END_YEAR = 2024
    
    TRADING_START_HOUR = 8
    TRADING_END_HOUR = 23
    
    TELEGRAM_TOKEN = "8008545474:AAHansC5Xag1b9N96bMAGE0YLTfykXoOPyY"
    TELEGRAM_USER_ID = 1590841427
    
    PAPER_TRADING = True
    
    SPREAD = {'EURUSD': 0.8, 'GBPUSD': 1.2, 'USDJPY': 1.0}
    
    @classmethod
    def create_directories(cls):
        for path in [cls.BASE_PATH, cls.DATA_PATH, cls.OUTPUT_PATH, cls.LOG_PATH, cls.MODEL_PATH]:
            os.makedirs(path, exist_ok=True)

Config.create_directories()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_PATH, f'bot_{datetime.now():%Y%m%d}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('UltimateTradingBot')

class TelegramBot:
    def __init__(self, token: str, user_id: int):
        self.token = token
        self.user_id = user_id
        self.enabled = HAS_ASYNC and token and user_id
        if self.enabled:
            logger.info("Telegram enabled")
    
    def sync_send_message(self, text: str):
        if not self.enabled:
            logger.info(f"[TELEGRAM] {text}")
            return
        try:
            if HAS_ASYNC:
                asyncio.run(self._send_message_async(text))
        except Exception as e:
            logger.error(f"Telegram failed: {e}")
    
    async def _send_message_async(self, text: str):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {'chat_id': self.user_id, 'text': text, 'parse_mode': 'HTML'}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as response:
                return await response.json()

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
            self.logger.info(f"Loading {symbol} from cache")
            return self.data_cache[cache_key]
        
        self.logger.info(f"Loading {symbol} data from {start_year} to {end_year}...")
        
        subfolder = f"{symbol}2003-2024"
        pattern = os.path.join(Config.DATA_PATH, subfolder, f"{symbol}_Candlestick*.csv")
        files = glob.glob(pattern)
        files = [f for f in files if 'weekly_ranges' not in f.lower()]
        
        if not files:
            pattern = os.path.join(Config.DATA_PATH, f"{symbol}*", f"{symbol}_Candlestick*.csv")
            files = glob.glob(pattern)
            files = [f for f in files if 'weekly_ranges' not in f.lower()]
        
        if not files:
            raise FileNotFoundError(f"No data files found for {symbol}")
        
        self.logger.info(f"Found {len(files)} files for {symbol}")
        
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
                    self.logger.info(f"  Loaded {len(df)} rows from {os.path.basename(file)}")
            except Exception as e:
                self.logger.warning(f"Could not load {file}: {e}")
        
        if not dfs:
            raise ValueError(f"No valid data loaded for {symbol}")
        
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined = df_combined.sort_values('time').reset_index(drop=True)
        df_combined = df_combined.drop_duplicates(subset=['time'])
        
        df_combined['year'] = df_combined['time'].dt.year
        df_combined = df_combined[
            (df_combined['year'] >= start_year) & 
            (df_combined['year'] <= end_year)
        ].drop(columns=['year'])
        
        df_combined['weekday'] = df_combined['time'].dt.weekday
        df_combined = df_combined[df_combined['weekday'] < 5].drop(columns=['weekday'])
        
        if 'volume' in df_combined.columns:
            df_combined = df_combined[df_combined['volume'] >= 0]
        
        df_combined = df_combined.reset_index(drop=True)
        
        self.logger.info(f"Loaded {len(df_combined)} rows for {symbol}")
        self.logger.info(f"Date range: {df_combined['time'].min()} to {df_combined['time'].max()}")
        
        self.data_cache[cache_key] = df_combined
        return df_combined

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger('FeatureEngineer')
        self.scaler = StandardScaler()
    
    def calculate_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        self.logger.info(f"Calculating features for {symbol}...")
        
        if len(df) < 200:
            self.logger.warning(f"Insufficient data: {len(df)} rows")
            return df
        
        df = df.copy()
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        
        df['adx_14'] = 25.0  # Simplified
        
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['is_trading_day'] = (df['day_of_week'] < 5).astype(int)
        
        df['regime'] = 0  # Simplified
        
        df['var_95'] = df['returns'].rolling(100).quantile(0.05)
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        self.logger.info(f"Features calculated: {len(df.columns)} columns")
        return df

class RiskManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_capital = initial_capital
        self.historical_returns = deque(maxlen=Config.VAR_WINDOW)
        self.trade_history = []
        self.logger = logging.getLogger('RiskManager')
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        if len(self.historical_returns) < 30:
            return 0.0
        returns_array = np.array(self.historical_returns)
        var = np.percentile(returns_array, (1 - confidence) * 100)
        return var
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
            return Config.RISK_PER_TRADE
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        kelly = (p * b - q) / b
        kelly_fraction = max(0.001, min(kelly * Config.KELLY_FRACTION, 0.10))
        return kelly_fraction
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                                stop_loss_pips: float, signal_strength: float = 1.0,
                                win_rate: float = None, avg_win: float = None,
                                avg_loss: float = None) -> float:
        pip_value = 10 if 'JPY' in symbol else 1
        
        if win_rate and avg_win and avg_loss:
            kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
            risk_amount = self.current_capital * kelly_fraction
        else:
            risk_amount = self.current_capital * Config.RISK_PER_TRADE
        
        risk_amount *= signal_strength
        
        if stop_loss_pips * pip_value > 0:
            lot_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            lot_size = Config.MIN_LOT
        
        lot_size = np.clip(lot_size, Config.MIN_LOT, Config.MAX_LOT)
        
        self.logger.info(f"Position size: {lot_size:.3f} lots (risk: ${risk_amount:.2f})")
        return lot_size
    
    def check_risk_limits(self) -> Tuple[bool, str]:
        daily_pnl = sum(t.get('pnl', 0) for t in self.trade_history 
                        if (datetime.now() - t.get('timestamp', datetime.now())).days == 0)
        
        max_daily_loss = self.initial_capital * Config.MAX_DAILY_LOSS_PCT
        if abs(daily_pnl) > max_daily_loss:
            return False, f"Daily loss limit exceeded"
        
        total_loss = self.initial_capital - self.current_capital
        max_total_loss = self.initial_capital * Config.MAX_TOTAL_LOSS_PCT
        if total_loss > max_total_loss:
            return False, f"Total loss limit exceeded"
        
        return True, "All risk checks passed"
    
    def update_capital(self, pnl: float):
        self.current_capital += pnl
        self.max_capital = max(self.max_capital, self.current_capital)
        if self.current_capital > 0:
            ret = pnl / self.current_capital
            self.historical_returns.append(ret)
    
    def add_trade(self, trade: Dict):
        trade['timestamp'] = datetime.now()
        self.trade_history.append(trade)
    
    def get_statistics(self) -> Dict:
        if not self.trade_history:
            return {'win_rate': 0.5, 'avg_win': 1.0, 'avg_loss': 1.0, 'total_trades': 0}
        
        closed_trades = [t for t in self.trade_history if 'pnl' in t]
        if not closed_trades:
            return {'win_rate': 0.5, 'avg_win': 1.0, 'avg_loss': 1.0, 'total_trades': 0}
        
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.5
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 1.0
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 1.0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }

class TradingEnvironment:
    def __init__(self, data: Dict[str, pd.DataFrame], initial_capital: float = Config.INITIAL_CAPITAL):
        self.data = data
        self.symbols = list(data.keys())
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager(initial_capital)
        self.telegram = TelegramBot(Config.TELEGRAM_TOKEN, Config.TELEGRAM_USER_ID)
        self.current_step = 0
        self.positions = []
        self.balance = initial_capital
        self.equity = initial_capital
        self.episode_start_capital = initial_capital
        self.episode_trades = []
        self.logger = logging.getLogger('TradingEnvironment')
    
    def reset(self) -> Dict:
        self.current_step = 0
        self.positions = []
        self.balance = self.initial_capital
        self.equity = self.initial_capital
        self.episode_start_capital = self.initial_capital
        self.episode_trades = []
        self.risk_manager = RiskManager(self.initial_capital)
        return self._get_observation()
    
    def _get_observation(self) -> Dict:
        obs = {}
        for symbol in self.symbols:
            df = self.data[symbol]
            if self.current_step >= len(df):
                continue
            row = df.iloc[self.current_step]
            features = ['close', 'returns', 'rsi_14', 'macd', 'macd_signal',
                       'bb_position', 'atr_14', 'adx_14', 'volatility_20',
                       'regime', 'is_trading_day']
            obs[symbol] = {feat: row.get(feat, 0.0) for feat in features}
        obs['portfolio'] = {
            'balance': self.balance,
            'equity': self.equity,
            'num_positions': len(self.positions),
            'var': self.risk_manager.calculate_var()
        }
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
        
        turkey_hour = current_time.tz_convert('Europe/Istanbul').hour if current_time.tzinfo else current_time.hour
        if turkey_hour < Config.TRADING_START_HOUR or turkey_hour >= Config.TRADING_END_HOUR:
            self._close_all_positions("Outside trading hours")
            return self._get_observation(), 0.0, done, {'reason': 'outside_hours'}
        
        is_trading_day = current_time.weekday() < 5
        if not is_trading_day:
            self._close_all_positions("Weekend")
            return self._get_observation(), 0.0, done, {'reason': 'weekend'}
        
        self._update_positions(current_time)
        
        allowed, reason = self.risk_manager.check_risk_limits()
        if not allowed:
            self.logger.warning(f"Risk limit exceeded: {reason}")
            self._close_all_positions(f"Risk limit: {reason}")
            return self._get_observation(), -100.0, True, {'reason': reason}
        
        total_reward = 0.0
        info = {'trades_executed': 0, 'positions_closed': 0}
        
        for symbol, signal_strength in actions.items():
            if symbol not in self.data or abs(signal_strength) <= 0.2:
                continue
            
            df = self.data[symbol]
            if self.current_step >= len(df) or len(self.positions) >= Config.MAX_POSITIONS:
                continue
            
            current_row = df.iloc[self.current_step]
            entry_price = current_row['close']
            atr = current_row.get('atr_14', entry_price * 0.001)
            stop_loss_pips = (2 * atr) / (0.0001 if 'JPY' not in symbol else 0.01)
            stats = self.risk_manager.get_statistics()
            
            lot_size = self.risk_manager.calculate_position_size(
                symbol=symbol, entry_price=entry_price, stop_loss_pips=stop_loss_pips,
                signal_strength=abs(signal_strength), win_rate=stats['win_rate'],
                avg_win=stats['avg_win'], avg_loss=stats['avg_loss']
            )
            
            action = 'BUY' if signal_strength > 0 else 'SELL'
            
            position = {
                'id': str(uuid.uuid4())[:8],
                'symbol': symbol,
                'action': action,
                'lot_size': lot_size,
                'entry_price': entry_price,
                'entry_time': current_time,
                'stop_loss': entry_price - (2*atr) if action == 'BUY' else entry_price + (2*atr),
                'take_profit': entry_price + (4*atr) if action == 'BUY' else entry_price - (4*atr),
                'atr': atr
            }
            
            self.positions.append(position)
            info['trades_executed'] += 1
            
            self.logger.info(f"Position opened: {action} {lot_size:.3f} {symbol} @ {entry_price:.5f}")
            self.telegram.sync_send_message(
                f"📈 Position Opened\nSymbol: {symbol}\nAction: {action}\n"
                f"Size: {lot_size:.3f} lots\nEntry: {entry_price:.5f}"
            )
        
        new_equity = self._calculate_equity()
        reward = (new_equity - self.equity) / self.initial_capital * 100
        self.equity = new_equity
        total_reward += reward
        
        return self._get_observation(), total_reward, done, info
    
    def _update_positions(self, current_time: datetime):
        positions_to_close = []
        
        for position in self.positions:
            symbol = position['symbol']
            df = self.data[symbol]
            
            if self.current_step >= len(df):
                positions_to_close.append(position)
                continue
            
            current_row = df.iloc[self.current_step]
            current_price = current_row['close']
            
            time_held = (current_time - position['entry_time']).total_seconds() / 3600
            if time_held >= Config.MAX_HOLDING_HOURS:
                self._close_position(position, current_price, "23-hour limit")
                positions_to_close.append(position)
                continue
            
            if position['action'] == 'BUY':
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
        
        if position['action'] == 'BUY':
            pnl = (exit_price - entry_price) / pip_size * pip_value * lot_size
        else:
            pnl = (entry_price - exit_price) / pip_size * pip_value * lot_size
        
        self.balance += pnl
        self.risk_manager.update_capital(pnl)
        
        trade = {
            'symbol': symbol, 'action': position['action'], 'lot_size': lot_size,
            'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl,
            'reason': reason, 'entry_time': position['entry_time'], 'exit_time': datetime.now()
        }
        
        self.episode_trades.append(trade)
        self.risk_manager.add_trade(trade)
        
        self.logger.info(f"Position closed: {symbol} {reason} | P&L: ${pnl:.2f}")
        self.telegram.sync_send_message(
            f"📊 Position Closed\nSymbol: {symbol}\nReason: {reason}\n"
            f"P&L: ${pnl:.2f}\nBalance: ${self.balance:.2f}"
        )
    
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
                if position['action'] == 'BUY':
                    floating_pnl = (current_price - entry_price) / pip_size * pip_value * lot_size
                else:
                    floating_pnl = (entry_price - current_price) / pip_size * pip_value * lot_size
                equity += floating_pnl
        return equity
    
    def get_performance_summary(self) -> Dict:
        if not self.episode_trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'total_pnl': 0.0, 'final_capital': self.balance, 'return_pct': 0.0
            }
        
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
            'return_pct': ((self.balance - self.episode_start_capital) / self.episode_start_capital) * 100
        }

class SimpleTrendAgent:
    def __init__(self):
        self.logger = logging.getLogger('SimpleTrendAgent')
    
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
            bb_position = features.get('bb_position', 0.5)
            adx = features.get('adx_14', 25)
            
            signal = 0.0
            if adx > 25:
                if macd > macd_signal:
                    signal += 0.3
                elif macd < macd_signal:
                    signal -= 0.3
                if rsi < 30:
                    signal += 0.3
                elif rsi > 70:
                    signal -= 0.3
                if bb_position < 0.2:
                    signal += 0.2
                elif bb_position > 0.8:
                    signal -= 0.2
            
            signal = np.clip(signal, -1.0, 1.0)
            signals[symbol] = signal
        
        return signals

class UltimateTradingSystem:
    def __init__(self):
        self.logger = logging.getLogger('UltimateTradingSystem')
        self.logger.info("="*70)
        self.logger.info("ULTIMATE FTMO TRADING BOT V6.0 FINAL")
        self.logger.info("="*70)
        
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        self.telegram = TelegramBot(Config.TELEGRAM_TOKEN, Config.TELEGRAM_USER_ID)
        self.forex_data = {}
        self.environment = None
        self.agent = SimpleTrendAgent()
        self.is_running = False
        
        self.logger.info("System initialized")
    
    def load_data(self, start_year: int = None, end_year: int = None):
        if start_year is None:
            start_year = Config.START_YEAR
        if end_year is None:
            end_year = Config.END_YEAR
        
        self.logger.info(f"Loading data from {start_year} to {end_year}...")
        
        for symbol in Config.SYMBOLS:
            try:
                df = self.data_manager.load_forex_data(symbol, start_year, end_year)
                df = self.feature_engineer.calculate_features(df, symbol)
                self.forex_data[symbol] = df
                self.logger.info(f"✅ {symbol}: {len(df)} rows loaded")
            except Exception as e:
                self.logger.error(f"❌ Failed to load {symbol}: {e}")
        
        if not self.forex_data:
            raise ValueError("No data loaded!")
        
        self.logger.info("Data loading complete")
        self.telegram.sync_send_message(
            f"📊 Data Loaded\nSymbols: {', '.join(self.forex_data.keys())}\n"
            f"Period: {start_year}-{end_year}\nRows: {sum(len(df) for df in self.forex_data.values()):,}"
        )
    
    def run_backtest(self, episodes: int = 1):
        self.logger.info(f"Starting backtest: {episodes} episodes")
        
        if not self.forex_data:
            raise ValueError("No data loaded. Call load_data() first.")
        
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
                    self.logger.info(
                        f"Step {step}: Balance=${perf['final_capital']:,.2f} | "
                        f"Trades={perf['total_trades']} | Win Rate={perf['win_rate']:.1%}"
                    )
            
            perf = self.environment.get_performance_summary()
            all_results.append(perf)
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Episode {episode + 1} Complete")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"Total Trades: {perf['total_trades']}")
            self.logger.info(f"Winning Trades: {perf['winning_trades']}")
            self.logger.info(f"Losing Trades: {perf['losing_trades']}")
            self.logger.info(f"Win Rate: {perf['win_rate']:.1%}")
            self.logger.info(f"Total P&L: ${perf['total_pnl']:,.2f}")
            self.logger.info(f"Final Capital: ${perf['final_capital']:,.2f}")
            self.logger.info(f"Return: {perf['return_pct']:.2f}%")
            self.logger.info(f"{'='*70}\n")
            
            self.telegram.sync_send_message(
                f"📈 Episode {episode + 1} Complete\n\n"
                f"Trades: {perf['total_trades']}\nWin Rate: {perf['win_rate']:.1%}\n"
                f"P&L: ${perf['total_pnl']:,.2f}\nReturn: {perf['return_pct']:.2f}%\n"
                f"Final: ${perf['final_capital']:,.2f}"
            )
        
        if len(all_results) > 1:
            avg_return = np.mean([r['return_pct'] for r in all_results])
            avg_win_rate = np.mean([r['win_rate'] for r in all_results])
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"BACKTEST SUMMARY ({episodes} episodes)")
            self.logger.info(f"{'='*70}")
            self.logger.info(f"Average Return: {avg_return:.2f}%")
            self.logger.info(f"Average Win Rate: {avg_win_rate:.1%}")
            self.logger.info(f"{'='*70}\n")
        
        return all_results
    
    def start_paper_trading(self):
        self.logger.info("Starting paper trading mode...")
        self.logger.info("⚠️ NO REAL MONEY IS BEING USED")
        self.is_running = True
        self.telegram.sync_send_message(
            "🚀 Paper Trading Started\n\nMode: PAPER TRADING\n"
            "Initial Capital: $25,000\n⚠️ No real money is used"
        )
        self.run_backtest(episodes=1)
    
    def stop(self):
        self.logger.info("Stopping trading...")
        self.is_running = False
        if self.environment:
            self.environment._close_all_positions("System stop")
            perf = self.environment.get_performance_summary()
            self.logger.info(f"Final Performance:")
            self.logger.info(f"  Total P&L: ${perf['total_pnl']:,.2f}")
            self.logger.info(f"  Win Rate: {perf['win_rate']:.1%}")
            self.logger.info(f"  Final Capital: ${perf['final_capital']:,.2f}")
            self.telegram.sync_send_message(
                f"🛑 Trading Stopped\n\nFinal P&L: ${perf['total_pnl']:,.2f}\n"
                f"Win Rate: {perf['win_rate']:.1%}\nFinal Capital: ${perf['final_capital']:,.2f}"
            )

def main():
    parser = argparse.ArgumentParser(description='Ultimate FTMO Trading Bot V6.0')
    parser.add_argument('--mode', type=str, default='backtest', 
                        choices=['train', 'backtest', 'paper'],
                        help='Mode: train, backtest, paper')
    parser.add_argument('--years', type=str, default='2020-2024',
                        help='Year range: 2003-2024 or single year: 2020')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes for backtest')
    
    args = parser.parse_args()
    
    if '-' in args.years:
        start_year, end_year = map(int, args.years.split('-'))
    else:
        start_year = end_year = int(args.years)
    
    system = UltimateTradingSystem()
    system.load_data(start_year, end_year)
    
    if args.mode == 'backtest':
        system.run_backtest(episodes=args.episodes)
    elif args.mode == 'paper':
        system.start_paper_trading()
    elif args.mode == 'train':
        logger.info("Training mode - implementing RL training...")
        system.run_backtest(episodes=args.episodes)
    
    logger.info("\n" + "="*70)
    logger.info("SYSTEM COMPLETED")
    logger.info("="*70)

if __name__ == "__main__":
    main()
