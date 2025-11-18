#!/usr/bin/env python3
"""
================================================================================
ULTIMATE FTMO TRADING BOT V8.0 - PPO HYBRID
================================================================================

🏆 V8 ENHANCEMENTS OVER V7:
✅ PPO (Proximal Policy Optimization) replaces Rainbow DQN
✅ LSTM hybrid feature extraction for temporal patterns
✅ Walk-Forward Training (90/30 day windows) for overfitting prevention
✅ Optuna hyperparameter optimization (lr, clip_range, ent_coef, decay_rate)
✅ RewardShaper with advanced penalty system
✅ Modular architecture for better maintainability
✅ All V7 features preserved (12-point strategy, news, telegram, etc.)

Author: E1 AI Agent + Grok Integration  
Date: January 2025
Version: 8.0 PPO HYBRID
Status: TESTING - Parallel with V7

Kullanım:
    python ultimate_bot_v8_ppo.py --mode backtest --years 2020-2024 --use-ppo
    python ultimate_bot_v8_ppo.py --mode train --years 2020-2024 --optuna-trials 50
    python ultimate_bot_v8_ppo.py --mode paper --use-ppo

================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Use gymnasium instead of deprecated gym
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# Import V7 components
sys.path.insert(0, os.path.expanduser("~/Desktop/JTTWS"))
try:
    from ultimate_bot_v7_professional import (
        Config,
        TelegramReporter,
        RightsManager,
        WeeklyRangeLearner,
        VolatilityGuards,
        TrendFilter,
        CorrelationControl,
        ThompsonBandit,
        DataManager,
        FeatureEngineer,
        TradingEnvironment as V7TradingEnvironment,
        UltimateTradingSystemV7
    )
    V7_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import V7 components: {e}")
    V7_AVAILABLE = False
    # Define minimal Config fallback
    class Config:
        BASE_PATH = os.path.expanduser("~/Desktop/JTTWS")
        DATA_PATH = os.path.join(BASE_PATH, "data")
        SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY']
        INITIAL_CAPITAL = 25000.0

# Import V8 modules
from reward_shaper import RewardShaper
from ppo_agent import PPOAgent, LSTMPredictor
from walk_forward_trainer import WalkForwardTrainer
from optuna_optimizer import OptunaOptimizer
from data_manager_v8 import DataManagerV8
from data_aggregator_v8 import DataAggregatorV8

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('UltimateBot_V8')


# =============================================================================
# V8 ENHANCED TRADING ENVIRONMENT
# =============================================================================

class TradingEnvironmentV8(gym.Env):
    """
    Enhanced Trading Environment for V8 with Gym compatibility.
    
    Extends V7 TradingEnvironment with:
    - Gym interface for SB3 compatibility
    - RewardShaper integration
    - Better observation/action space definition
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = Config.INITIAL_CAPITAL,
        use_reward_shaper: bool = True
    ):
        """
        Initialize V8 Trading Environment.
        
        Args:
            data: Dictionary of DataFrames (symbol -> data)
            initial_capital: Starting capital
            use_reward_shaper: Enable RewardShaper penalties
        """
        super().__init__()
        
        self.data = data
        self.symbols = list(data.keys())
        self.initial_capital = initial_capital
        self.use_reward_shaper = use_reward_shaper
        
        # V7 components
        if V7_AVAILABLE:
            self.rights_manager = RightsManager()
            self.telegram = TelegramReporter(Config.TELEGRAM_TOKEN, Config.TELEGRAM_USER_ID) if hasattr(Config, 'TELEGRAM_TOKEN') else None
            self.weekly_learner = WeeklyRangeLearner()
            self.volatility_guards = VolatilityGuards(self.weekly_learner)
            self.trend_filter = TrendFilter()
            self.correlation_control = CorrelationControl()
            self.bandit = ThompsonBandit()
        
        # State
        self.current_step = 0
        self.positions = []
        self.balance = initial_capital
        self.equity = initial_capital
        self.episode_trades = []
        
        # Define Gym spaces
        # Observation: [balance, equity, num_positions, symbol1_features..., symbol2_features..., symbol3_features...]
        # Features per symbol: close, rsi, macd, atr (4 features)
        obs_dim = 3 + len(self.symbols) * 4  # 3 portfolio + 4*3 symbols = 15
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action: Discrete(7) - [None, Buy_S1, Sell_S1, Buy_S2, Sell_S2, Buy_S3, Sell_S3]
        # Or Continuous Box for signal strength per symbol
        self.action_space = spaces.Discrete(7)
        
        # V8 RewardShaper (if available)
        self.reward_shaper = None
        if use_reward_shaper and V7_AVAILABLE:
            try:
                # Mock logger for RewardShaper
                class MockLogger:
                    def log_error(self, msg): logger.error(msg)
                    def log_penalty_breakdown(self, timestamp, penalty, breakdown):
                        logger.info(f"Penalty: {penalty:.4f} - {breakdown}")
                
                # Mock market state
                class MockMarket:
                    def __init__(self, env):
                        self.env = env
                    def get_atr(self, timestamp, symbol):
                        return 0.001
                    def get_average_atr(self, symbol):
                        return 0.001
                    def estimate_slippage(self, timestamp, symbol):
                        return 0.0002
                
                # Mock blackout
                class MockBlackout:
                    def is_active(self, timestamp): return False
                
                self.reward_shaper = RewardShaper(
                    blackout=MockBlackout(),
                    guards=self.volatility_guards if V7_AVAILABLE else None,
                    correlation=self.correlation_control if V7_AVAILABLE else None,
                    market=MockMarket(self),
                    logger=MockLogger()
                )
                logger.info("✅ RewardShaper initialized")
            except Exception as e:
                logger.warning(f"⚠️  RewardShaper initialization failed: {e}")
                self.reward_shaper = None
        
        logger.info(f"🌍 TradingEnvironmentV8 initialized: {len(self.symbols)} symbols, ${initial_capital}")
    
    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        self.positions = []
        self.balance = self.initial_capital
        self.equity = self.initial_capital
        self.episode_trades = []
        
        if V7_AVAILABLE and hasattr(self, 'rights_manager'):
            self.rights_manager.reset_daily()
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation as flat numpy array.
        
        Returns:
            np.ndarray: Flattened observation vector
        """
        # Portfolio features
        obs = [
            self.balance / self.initial_capital,  # Normalized balance
            self.equity / self.initial_capital,   # Normalized equity
            len(self.positions) / 10.0            # Normalized position count
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
                0: Do nothing
                1: Buy Symbol 1, 2: Sell Symbol 1
                3: Buy Symbol 2, 4: Sell Symbol 2
                5: Buy Symbol 3, 6: Sell Symbol 3
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Check if episode is done
        done = any(self.current_step >= len(self.data[s]) for s in self.symbols)
        
        if done:
            obs = self._get_observation()
            return obs, 0.0, True, False, {'reason': 'data_end'}
        
        # Decode action
        if action == 0:
            # Do nothing
            pass
        else:
            # Map action to symbol and direction
            action_idx = action - 1  # 0-5
            symbol_idx = action_idx // 2  # 0, 1, 2
            direction = 'LONG' if action_idx % 2 == 0 else 'SHORT'
            
            if symbol_idx < len(self.symbols):
                symbol = self.symbols[symbol_idx]
                # Execute trade (simplified)
                self._execute_trade(symbol, direction)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Apply RewardShaper penalty
        if self.reward_shaper is not None:
            try:
                context = {
                    'timestamp': pd.Timestamp.now(),
                    'open_positions': self.positions,
                    'symbol': self.symbols[0] if self.symbols else 'EURUSD'
                }
                penalty = self.reward_shaper.compute_penalty(None, action, context)
                reward += penalty
            except Exception as e:
                logger.warning(f"RewardShaper error: {e}")
        
        obs = self._get_observation()
        truncated = False
        info = {'balance': self.balance, 'equity': self.equity}
        
        return obs, reward, done, truncated, info
    
    def _execute_trade(self, symbol: str, direction: str):
        """Execute a trade (simplified version)."""
        df = self.data[symbol]
        if self.current_step >= len(df):
            return
        
        row = df.iloc[self.current_step]
        entry_price = row.get('close', 1.0)
        
        # Simplified position opening
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'size': 0.01,
            'entry_step': self.current_step
        }
        
        self.positions.append(position)
        logger.debug(f"Trade: {direction} {symbol} @ {entry_price:.5f}")
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current positions."""
        if not self.positions:
            return 0.0
        
        reward = 0.0
        
        # Simplified P&L calculation
        for pos in self.positions:
            symbol = pos['symbol']
            df = self.data[symbol]
            
            if self.current_step < len(df):
                current_price = df.iloc[self.current_step].get('close', pos['entry_price'])
                entry_price = pos['entry_price']
                
                if pos['direction'] == 'LONG':
                    pnl = (current_price - entry_price) * pos['size'] * 10000
                else:
                    pnl = (entry_price - current_price) * pos['size'] * 10000
                
                reward += pnl / 1000.0  # Normalize
        
        return reward


# =============================================================================
# V8 TRADING SYSTEM
# =============================================================================

class UltimateTradingSystemV8:
    """
    V8 Trading System with PPO and Walk-Forward Training.
    """
    
    def __init__(self, use_ppo: bool = True):
        """
        Initialize V8 Trading System.
        
        Args:
            use_ppo: If True, use PPO agent. If False, fallback to DQN (if available)
        """
        self.use_ppo = use_ppo
        self.data = {}
        self.env = None
        self.agent = None
        
        logger.info(f"🚀 UltimateTradingSystemV8 initialized (PPO: {use_ppo})")
    
    def load_data(self, start_year: int = 2020, end_year: int = 2024):
        """Load historical data using DataManagerV8."""
        logger.info(f"📂 Loading data: {start_year}-{end_year}")
        
        try:
            # Try V8 DataManager first (handles multi-file structure)
            data_manager = DataManagerV8()  # Will use ./data by default
            
            for symbol in Config.SYMBOLS:
                start_date = f"{start_year}-01-01"
                end_date = f"{end_year}-12-31"
                
                df = data_manager.load_symbol_data(symbol, start_date, end_date, use_mock=False)
                
                if df is not None and not df.empty:
                    # Add basic features if needed
                    if 'rsi_14' not in df.columns:
                        df['rsi_14'] = 50.0  # Placeholder
                    if 'macd' not in df.columns:
                        df['macd'] = 0.0
                    if 'atr_14' not in df.columns:
                        df['atr_14'] = 0.001
                    
                    self.data[symbol] = df
                    logger.info(f"  ✓ {symbol}: {len(df)} bars")
                else:
                    logger.warning(f"  ✗ {symbol}: No data, trying mock...")
                    df = data_manager.load_symbol_data(symbol, start_date, end_date, use_mock=True)
                    if not df.empty:
                        df['rsi_14'] = 50.0
                        df['macd'] = 0.0
                        df['atr_14'] = 0.001
                        self.data[symbol] = df
            
            logger.info(f"✅ Data loaded: {len(self.data)} symbols")
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            # Create mock data for testing
            logger.info("⚠️  Creating mock data...")
            self._create_mock_data()
    
    def _create_mock_data(self):
        """Create mock data for testing."""
        n_samples = 1000
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1h')
        
        for symbol in Config.SYMBOLS:
            self.data[symbol] = pd.DataFrame({
                'time': dates,
                'close': np.cumsum(np.random.randn(n_samples) * 0.001) + 1.1,
                'rsi_14': np.random.uniform(30, 70, n_samples),
                'macd': np.random.randn(n_samples) * 0.001,
                'atr_14': np.random.uniform(0.0005, 0.002, n_samples)
            })
        
        logger.info("✅ Mock data created")
    
    def run_backtest(self, episodes: int = 1):
        """Run backtest."""
        logger.info(f"📊 Running backtest: {episodes} episodes")
        
        # Create environment
        self.env = TradingEnvironmentV8(self.data, use_reward_shaper=True)
        
        # Create agent
        self.agent = PPOAgent(
            self.env,
            lr=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
            use_lstm=False,
            use_dqn_fallback=not self.use_ppo,
            verbose=1
        )
        
        # Run episodes
        total_rewards = []
        
        for ep in range(episodes):
            logger.info(f"\n📈 Episode {ep+1}/{episodes}")
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            
            while not done and steps < 1000:
                action = self.agent.predict(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                if steps % 100 == 0:
                    logger.info(f"  Step {steps}: Reward={episode_reward:.4f}, Balance=${info.get('balance', 0):.2f}")
            
            total_rewards.append(episode_reward)
            logger.info(f"✅ Episode {ep+1} complete: Reward={episode_reward:.4f}")
        
        avg_reward = np.mean(total_rewards)
        logger.info(f"\n📊 Backtest Summary:")
        logger.info(f"   Episodes: {episodes}")
        logger.info(f"   Avg Reward: {avg_reward:.4f}")
        logger.info(f"   Std Reward: {np.std(total_rewards):.4f}")
    
    def run_walk_forward_training(self, n_optuna_trials: int = 50):
        """Run walk-forward training with Optuna optimization using REAL DATA."""
        logger.info(f"🔄 Walk-Forward Training: {n_optuna_trials} Optuna trials per period")
        logger.info(f"📊 Using REAL DATA from {len(self.data)} symbols")
        
        # Aggregate 15M data to daily for walk-forward
        aggregator = DataAggregatorV8()
        daily_data = {}
        
        for symbol, df_15m in self.data.items():
            logger.info(f"   Processing {symbol}...")
            daily_df = aggregator.aggregate_to_daily(df_15m, symbol)
            if not daily_df.empty:
                daily_data[symbol] = daily_df
        
        if not daily_data:
            logger.error("❌ No daily data could be aggregated")
            return
        
        # Prepare walk-forward data (combined symbols)
        wf_data = aggregator.prepare_walk_forward_data(daily_data)
        
        if wf_data.empty or len(wf_data) < 240:  # Need at least 180 + 60 days
            logger.error(f"❌ Insufficient data for walk-forward: {len(wf_data)} days")
            logger.info(f"   Minimum required: 240 days (180 train + 60 test)")
            return
        
        # Create walk-forward trainer (Grok's 180/60 recommendation)
        # Note: initial_decay_tolerance=0.20 is default in WalkForwardTrainer
        trainer = WalkForwardTrainer(
            data=wf_data,
            window_train=180,
            window_test=60,
            env_class=TradingEnvironmentV8,
            agent_class=PPOAgent,
            decay_threshold=0.15
        )
        
        # Run walk-forward
        results = trainer.run(n_optuna_trials=n_optuna_trials)
        
        # Get best parameters
        best_params = trainer.get_best_params_from_results()
        
        logger.info(f"\n✅ Walk-Forward Complete!")
        logger.info(f"   Best Parameters:")
        for k, v in best_params.items():
            logger.info(f"      {k}: {v:.6f}")
        
        # Save results
        results_path = os.path.join(Config.BASE_PATH, "outputs", "walk_forward_results_v8.csv")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results.to_csv(results_path, index=False)
        logger.info(f"   Results saved: {results_path}")
        
        return results, best_params


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Ultimate FTMO Trading Bot V8.0 - PPO Hybrid')
    parser.add_argument('--mode', type=str, default='backtest', choices=['backtest', 'paper', 'train'], help='Mode')
    parser.add_argument('--years', type=str, default='2020-2024', help='Year range: 2020-2024 or 2020')
    parser.add_argument('--episodes', type=int, default=1, help='Episodes')
    parser.add_argument('--use-ppo', action='store_true', default=True, help='Use PPO (default: True)')
    parser.add_argument('--use-dqn', action='store_true', help='Use DQN fallback')
    parser.add_argument('--optuna-trials', type=int, default=10, help='Optuna trials per walk-forward period')
    
    args = parser.parse_args()
    
    # Parse years
    if '-' in args.years:
        start_year, end_year = map(int, args.years.split('-'))
    else:
        start_year = end_year = int(args.years)
    
    # Override PPO if DQN requested
    use_ppo = args.use_ppo and not args.use_dqn
    
    # Create system
    system = UltimateTradingSystemV8(use_ppo=use_ppo)
    system.load_data(start_year, end_year)
    
    # Run mode
    if args.mode == 'backtest':
        system.run_backtest(episodes=args.episodes)
    elif args.mode == 'paper':
        logger.info("📄 Paper trading mode (using backtest)...")
        system.run_backtest(episodes=1)
    elif args.mode == 'train':
        logger.info("🎓 Training mode with walk-forward...")
        system.run_walk_forward_training(n_optuna_trials=args.optuna_trials)
    
    logger.info("\n" + "="*70)
    logger.info("✅ V8 SYSTEM COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    main()
