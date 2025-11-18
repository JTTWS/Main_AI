#!/usr/bin/env python3
"""
================================================================================
PROFESSIONAL TRADING ENVIRONMENT V8
================================================================================
Profesyonel trading environment with:
- Real balance tracking and updates
- Position management (open/close/TP/SL)
- Commission, spread, margin handling
- Risk management (position sizing, max positions, drawdown)
- Performance metrics and detailed logging

Author: E1 AI Agent
Date: January 2025
Version: 8.0 Professional
================================================================================
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

logger = logging.getLogger('TradingEnvPro')


class Position:
    """Professional position class with full tracking."""
    
    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        entry_step: int,
        entry_time: pd.Timestamp,
        sl_pips: float = 50.0,
        tp_pips: float = 100.0
    ):
        """
        Initialize a trading position.
        
        Args:
            symbol: Trading symbol (EURUSD, GBPUSD, USDJPY)
            direction: LONG or SHORT
            entry_price: Entry price
            size: Position size in lots (0.01 = micro lot)
            entry_step: Environment step when opened
            entry_time: Timestamp when opened
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
        """
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.size = size
        self.entry_step = entry_step
        self.entry_time = entry_time
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        
        # Calculate SL/TP prices
        pip_value = 0.0001 if symbol != 'USDJPY' else 0.01
        
        if direction == 'LONG':
            self.sl_price = entry_price - (sl_pips * pip_value)
            self.tp_price = entry_price + (tp_pips * pip_value)
        else:  # SHORT
            self.sl_price = entry_price + (sl_pips * pip_value)
            self.tp_price = entry_price - (tp_pips * pip_value)
        
        # Position state
        self.is_open = True
        self.close_price = None
        self.close_step = None
        self.close_time = None
        self.close_reason = None
        self.pnl = 0.0
        self.pnl_pips = 0.0
    
    def calculate_floating_pnl(self, current_price: float) -> Tuple[float, float]:
        """
        Calculate current unrealized PnL.
        
        Args:
            current_price: Current market price
            
        Returns:
            (pnl_in_currency, pnl_in_pips)
        """
        pip_value = 0.0001 if self.symbol != 'USDJPY' else 0.01
        
        if self.direction == 'LONG':
            pips = (current_price - self.entry_price) / pip_value
        else:  # SHORT
            pips = (self.entry_price - current_price) / pip_value
        
        # Calculate monetary PnL
        # Standard lot value: 100,000 units
        # Pip value for standard lot: ~$10 for XXX/USD pairs
        pip_value_money = 10.0 if self.symbol != 'USDJPY' else 1000.0 / current_price
        pnl_money = pips * pip_value_money * self.size
        
        return pnl_money, pips
    
    def check_sl_tp(self, current_price: float) -> Optional[str]:
        """
        Check if SL or TP is hit.
        
        Args:
            current_price: Current market price
            
        Returns:
            'SL' if stop loss hit, 'TP' if take profit hit, None otherwise
        """
        if self.direction == 'LONG':
            if current_price <= self.sl_price:
                return 'SL'
            elif current_price >= self.tp_price:
                return 'TP'
        else:  # SHORT
            if current_price >= self.sl_price:
                return 'SL'
            elif current_price <= self.tp_price:
                return 'TP'
        
        return None
    
    def close(
        self,
        close_price: float,
        close_step: int,
        close_time: pd.Timestamp,
        reason: str
    ):
        """
        Close the position.
        
        Args:
            close_price: Closing price
            close_step: Environment step when closed
            close_time: Timestamp when closed
            reason: Reason for closing ('TP', 'SL', 'TIMEOUT', 'MANUAL')
        """
        self.is_open = False
        self.close_price = close_price
        self.close_step = close_step
        self.close_time = close_time
        self.close_reason = reason
        
        # Calculate final PnL
        self.pnl, self.pnl_pips = self.calculate_floating_pnl(close_price)
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for logging."""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'size': self.size,
            'entry_step': self.entry_step,
            'entry_time': self.entry_time,
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'is_open': self.is_open,
            'close_price': self.close_price,
            'close_step': self.close_step,
            'close_time': self.close_time,
            'close_reason': self.close_reason,
            'pnl': self.pnl,
            'pnl_pips': self.pnl_pips
        }


class ProfessionalTradingEnvironmentV8(gym.Env):
    """
    Professional Trading Environment with full balance and position management.
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 25000.0,
        max_positions: int = 3,
        position_size_pct: float = 0.02,  # 2% risk per trade
        commission_pips: float = 2.0,
        spread_pips: float = 1.0,
        max_drawdown_pct: float = 0.20,  # 20% max drawdown
        position_timeout_steps: int = 96  # Close after 24 hours (96 * 15min)
    ):
        """
        Initialize Professional Trading Environment.
        
        Args:
            data: Dictionary of DataFrames (symbol -> OHLCV data)
            initial_capital: Starting capital in USD
            max_positions: Maximum simultaneous positions
            position_size_pct: Position size as % of equity
            commission_pips: Commission per trade in pips
            spread_pips: Bid-ask spread in pips
            max_drawdown_pct: Maximum allowed drawdown as decimal
            position_timeout_steps: Auto-close positions after this many steps
        """
        super().__init__()
        
        self.data = data
        self.symbols = list(data.keys())
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.commission_pips = commission_pips
        self.spread_pips = spread_pips
        self.max_drawdown_pct = max_drawdown_pct
        self.position_timeout_steps = position_timeout_steps
        
        # State variables
        self.current_step = 0
        self.balance = initial_capital
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.balance_history = []
        
        # Define Gym spaces
        # Observation: [balance, equity, num_positions, drawdown, + features per symbol]
        # Features per symbol: close, rsi, macd, atr (4 features)
        obs_dim = 4 + len(self.symbols) * 4  # 4 account + 4*3 symbols = 16
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action: Discrete(7) - [Hold, Buy_S1, Sell_S1, Buy_S2, Sell_S2, Buy_S3, Sell_S3]
        self.action_space = spaces.Discrete(7)
        
        logger.info(f"🏦 Professional Trading Environment initialized")
        logger.info(f"   Capital: ${initial_capital:,.2f}")
        logger.info(f"   Max Positions: {max_positions}")
        logger.info(f"   Position Size: {position_size_pct*100:.1f}% of equity")
        logger.info(f"   Commission: {commission_pips} pips")
        logger.info(f"   Spread: {spread_pips} pips")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.balance = self.initial_capital
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.positions = []
        self.closed_positions = []
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.balance_history = [self.balance]
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as flat numpy array."""
        # Calculate current drawdown
        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        # Account features
        obs = [
            self.balance / self.initial_capital,      # Normalized balance
            self.equity / self.initial_capital,       # Normalized equity
            len(self.positions) / self.max_positions, # Position utilization
            drawdown                                   # Current drawdown
        ]
        
        # Symbol features
        for symbol in self.symbols:
            df = self.data[symbol]
            if self.current_step < len(df):
                row = df.iloc[self.current_step]
                obs.extend([
                    row.get('close', 1.0),
                    row.get('rsi_14', 50.0) / 100.0,  # Normalize RSI
                    row.get('macd', 0.0),
                    row.get('atr_14', 0.001)
                ])
            else:
                obs.extend([1.0, 0.5, 0.0, 0.001])  # Default values
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Discrete action [0-6]
                0: Hold
                1-2: Buy/Sell Symbol 1
                3-4: Buy/Sell Symbol 2
                5-6: Buy/Sell Symbol 3
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Check if episode is done
        max_steps = min(len(self.data[s]) for s in self.symbols)
        done = self.current_step >= max_steps
        
        # Check drawdown limit
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if current_drawdown >= self.max_drawdown_pct:
            logger.warning(f"⚠️  Max drawdown reached: {current_drawdown*100:.2f}%")
            done = True
        
        if done:
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info
        
        # 1. Update all open positions (check SL/TP, update floating PnL)
        self._update_positions()
        
        # 2. Execute new action if valid
        reward = 0.0
        if action > 0:
            success = self._execute_action(action)
            if success:
                reward += 0.1  # Small reward for taking action
        
        # 3. Calculate reward based on PnL change
        previous_equity = self.equity
        self._update_equity()
        equity_change = self.equity - previous_equity
        reward += equity_change / self.initial_capital * 100  # Normalize reward
        
        # 4. Update balance history
        self.balance_history.append(self.balance)
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, done, False, info
    
    def _update_positions(self):
        """Update all open positions - check SL/TP, timeout, update floating PnL."""
        positions_to_close = []
        
        for pos in self.positions:
            if not pos.is_open:
                continue
            
            # Get current price
            symbol_data = self.data[pos.symbol]
            if self.current_step >= len(symbol_data):
                continue
            
            current_price = symbol_data.iloc[self.current_step]['close']
            current_time = symbol_data.iloc[self.current_step].get('timestamp', pd.Timestamp.now())
            
            # Check SL/TP
            sl_tp_hit = pos.check_sl_tp(current_price)
            if sl_tp_hit:
                pos.close(current_price, self.current_step, current_time, sl_tp_hit)
                positions_to_close.append(pos)
                logger.info(f"   🎯 {sl_tp_hit} Hit: {pos.direction} {pos.symbol} @ {current_price:.5f}, PnL: ${pos.pnl:.2f} ({pos.pnl_pips:.1f} pips)")
                continue
            
            # Check timeout
            if self.current_step - pos.entry_step >= self.position_timeout_steps:
                pos.close(current_price, self.current_step, current_time, 'TIMEOUT')
                positions_to_close.append(pos)
                logger.info(f"   ⏱️  Timeout: {pos.direction} {pos.symbol} @ {current_price:.5f}, PnL: ${pos.pnl:.2f} ({pos.pnl_pips:.1f} pips)")
        
        # Close positions and update balance
        for pos in positions_to_close:
            self._close_position(pos)
    
    def _execute_action(self, action: int) -> bool:
        """
        Execute a trading action.
        
        Args:
            action: Action ID (1-6)
            
        Returns:
            True if action executed successfully, False otherwise
        """
        # Check if we can open more positions
        if len(self.positions) >= self.max_positions:
            return False
        
        # Decode action
        action_idx = action - 1  # 0-5
        symbol_idx = action_idx // 2  # 0, 1, 2
        direction = 'LONG' if action_idx % 2 == 0 else 'SHORT'
        
        if symbol_idx >= len(self.symbols):
            return False
        
        symbol = self.symbols[symbol_idx]
        
        # Get current price
        symbol_data = self.data[symbol]
        if self.current_step >= len(symbol_data):
            return False
        
        row = symbol_data.iloc[self.current_step]
        current_price = row['close']
        current_time = row.get('timestamp', pd.Timestamp.now())
        
        # Calculate position size
        position_size = self._calculate_position_size(symbol, current_price)
        if position_size <= 0:
            return False
        
        # Calculate entry costs (commission + spread)
        pip_value = 0.0001 if symbol != 'USDJPY' else 0.01
        pip_value_money = 10.0 if symbol != 'USDJPY' else 1000.0 / current_price
        
        commission_cost = self.commission_pips * pip_value_money * position_size
        spread_cost = self.spread_pips * pip_value_money * position_size
        total_cost = commission_cost + spread_cost
        
        # Check if we have enough balance
        if total_cost > self.balance:
            return False
        
        # Deduct costs from balance
        self.balance -= total_cost
        
        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            size=position_size,
            entry_step=self.current_step,
            entry_time=current_time,
            sl_pips=50.0,
            tp_pips=100.0
        )
        
        self.positions.append(position)
        
        logger.info(f"🔔 TRADE OPENED: {direction} {symbol} @ {current_price:.5f}")
        logger.info(f"   Size: {position_size:.2f} lots, SL: {position.sl_price:.5f}, TP: {position.tp_price:.5f}")
        logger.info(f"   Costs: ${total_cost:.2f} (Comm: ${commission_cost:.2f}, Spread: ${spread_cost:.2f})")
        logger.info(f"   Balance: ${self.balance:.2f}")
        
        return True
    
    def _close_position(self, position: Position):
        """
        Close a position and update balance.
        
        Args:
            position: Position to close
        """
        # Add PnL to balance
        self.balance += position.pnl
        self.total_pnl += position.pnl
        
        # Update statistics
        self.total_trades += 1
        if position.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to closed positions
        self.closed_positions.append(position)
        self.positions.remove(position)
        
        logger.info(f"🔔 TRADE CLOSED: {position.direction} {position.symbol}")
        logger.info(f"   Entry: {position.entry_price:.5f} → Exit: {position.close_price:.5f}")
        logger.info(f"   PnL: ${position.pnl:.2f} ({position.pnl_pips:.1f} pips)")
        logger.info(f"   Reason: {position.close_reason}")
        logger.info(f"   Balance: ${self.balance:.2f}")
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """
        Calculate position size based on risk percentage.
        
        Args:
            symbol: Trading symbol
            price: Current price
            
        Returns:
            Position size in lots
        """
        # Risk amount in dollars
        risk_amount = self.equity * self.position_size_pct
        
        # Standard lot value
        lot_value = 100000  # 100k units
        
        # Calculate position size
        # For simplicity, use fixed lot size of 0.01 (micro lot)
        # In production, this should be more sophisticated
        position_size = 0.01
        
        return position_size
    
    def _update_equity(self):
        """Update equity = balance + unrealized PnL from open positions."""
        unrealized_pnl = 0.0
        
        for pos in self.positions:
            if not pos.is_open:
                continue
            
            # Get current price
            symbol_data = self.data[pos.symbol]
            if self.current_step >= len(symbol_data):
                continue
            
            current_price = symbol_data.iloc[self.current_step]['close']
            pnl, _ = pos.calculate_floating_pnl(current_price)
            unrealized_pnl += pnl
        
        self.equity = self.balance + unrealized_pnl
        
        # Update peak equity for drawdown calculation
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
    
    def _get_info(self) -> Dict:
        """Get detailed info dictionary."""
        # Calculate performance metrics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        avg_win = sum(p.pnl for p in self.closed_positions if p.pnl > 0) / max(self.winning_trades, 1)
        avg_loss = sum(p.pnl for p in self.closed_positions if p.pnl < 0) / max(self.losing_trades, 1)
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'total_pnl': self.total_pnl,
            'open_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'peak_equity': self.peak_equity,
            'drawdown': (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        }
    
    def get_performance_summary(self) -> str:
        """Get formatted performance summary."""
        info = self._get_info()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              TRADING PERFORMANCE SUMMARY                      ║
╠══════════════════════════════════════════════════════════════╣
║  Balance:         ${info['balance']:>12,.2f}                 ║
║  Equity:          ${info['equity']:>12,.2f}                  ║
║  Total PnL:       ${info['total_pnl']:>12,.2f}               ║
║  Return:          {(info['equity']/self.initial_capital-1)*100:>11.2f}%  ║
╠══════════════════════════════════════════════════════════════╣
║  Total Trades:    {info['total_trades']:>5}                               ║
║  Winning Trades:  {info['winning_trades']:>5}  ({info['win_rate']*100:>5.1f}%)                  ║
║  Losing Trades:   {info['losing_trades']:>5}  ({(1-info['win_rate'])*100:>5.1f}%)                  ║
║  Avg Win:         ${info['avg_win']:>12,.2f}                 ║
║  Avg Loss:        ${info['avg_loss']:>12,.2f}                ║
║  Profit Factor:   {info['profit_factor']:>12.2f}                  ║
╠══════════════════════════════════════════════════════════════╣
║  Peak Equity:     ${info['peak_equity']:>12,.2f}             ║
║  Max Drawdown:    {info['drawdown']*100:>11.2f}%              ║
║  Open Positions:  {info['open_positions']:>5}                               ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary
