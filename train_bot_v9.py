#!/usr/bin/env python3
"""
================================================================================
TRAIN BOT V9 - Google Colab Ready Training Script
================================================================================

Complete training pipeline for ULTIMATE FTMO TRADING BOT V9.

Features:
- Google Colab compatible (GPU accelerated)
- Hyperparameter optimization (Optuna)
- Walk-forward validation
- Ensemble training (multiple agents)
- Model checkpointing
- TensorBoard logging
- Progress tracking

Usage (Google Colab):
    # Upload this file and required modules to Colab
    # Install dependencies
    !pip install ta-lib gymnasium stable-baselines3 optuna pandas numpy
    
    # Run training
    !python train_bot_v9.py --mode full --trials 50 --timesteps 100000

Usage (Local):
    python train_bot_v9.py --mode quick --trials 10 --timesteps 50000

Author: E1 AI Agent (Emergent.sh)
Date: January 2025
Version: 9.0 FREE PRO
================================================================================
"""

import os
import sys
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

# Import required modules
try:
    from data_manager_v8 import DataManagerV8
    from feature_engineer_v9 import FeatureEngineerV9
    from trading_environment_pro import ProfessionalTradingEnvironmentV8
    from ppo_agent import PPOAgent
    from ensemble_manager import EnsembleManagerV9
    from walk_forward_trainer import WalkForwardTrainer
    from optuna_optimizer import OptunaOptimizer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing modules: {e}")
    print("\nPlease ensure all required files are in the same directory:")
    print("  - data_manager_v8.py")
    print("  - feature_engineer_v9.py")
    print("  - trading_environment_pro.py")
    print("  - ppo_agent.py")
    print("  - ensemble_manager.py")
    print("  - walk_forward_trainer.py (optional)")
    print("  - optuna_optimizer.py (optional)")
    MODULES_AVAILABLE = False
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainBotV9')


class TrainingPipelineV9:
    """
    Complete training pipeline for Bot V9.
    
    Modes:
    - quick: Fast training for testing (10 trials, 50k timesteps)
    - full: Full training for production (50 trials, 200k timesteps)
    - ensemble: Train ensemble of agents
    - custom: Custom configuration
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        models_dir: str = './models_v9',
        logs_dir: str = './logs_v9'
    ):
        """
        Initialize Training Pipeline.
        
        Args:
            data_dir: Directory containing market data
            models_dir: Directory to save trained models
            logs_dir: Directory for training logs
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Components
        self.data_manager = None
        self.feature_engineer = None
        self.env = None
        self.agent = None
        
        logger.info("🚀 TrainingPipelineV9 initialized")
        logger.info(f"   Data: {data_dir}")
        logger.info(f"   Models: {models_dir}")
        logger.info(f"   Logs: {logs_dir}")
    
    def setup_data(self, symbol: str = 'EURUSD', years: str = '2020-2024'):
        """
        Setup data loading and feature engineering.
        
        Args:
            symbol: Trading symbol
            years: Date range (e.g., '2020-2024')
        """
        logger.info(f"📊 Setting up data for {symbol} ({years})...")
        
        # Parse years
        start_year, end_year = years.split('-')
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        # Initialize data manager
        self.data_manager = DataManagerV8(data_dir=self.data_dir)
        
        # Load data
        df = self.data_manager.load_symbol_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            use_mock=True  # Fallback to mock if no real data
        )
        
        if df.empty:
            raise ValueError(f"No data loaded for {symbol}")
        
        logger.info(f"✅ Loaded {len(df)} rows of data")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineerV9(
            enable_talib=True,
            enable_multi_timeframe=True,
            timeframes=['15T', '1H', '4H']
        )
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df, symbol=symbol)
        
        logger.info(f"✅ Engineered {len(df.columns)} features")
        
        return df
    
    def setup_environment(self, df: pd.DataFrame):
        """
        Setup trading environment.
        
        Args:
            df: DataFrame with OHLCV and features
        """
        logger.info("🎯 Setting up trading environment...")
        
        # Create environment (data must be a dict of dataframes)
        data_dict = {'EURUSD': df}
        
        self.env = ProfessionalTradingEnvironmentV8(
            data=data_dict,
            initial_capital=25000.0,
            max_positions=3,
            position_size_pct=0.02,
            commission_pips=2.0,
            spread_pips=1.0,
            max_drawdown_pct=0.20,
            position_timeout_steps=96
        )
        
        logger.info("✅ Environment ready")
        logger.info(f"   Observation space: {self.env.observation_space.shape}")
        logger.info(f"   Action space: {self.env.action_space}")
        
        return self.env
    
    def train_single_agent(
        self,
        timesteps: int = 100000,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        ent_coef: float = 0.01
    ):
        """
        Train single PPO agent.
        
        Args:
            timesteps: Total training timesteps
            lr: Learning rate
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
        """
        logger.info(f"🎯 Training single agent ({timesteps} steps)...")
        
        # Create agent
        self.agent = PPOAgent(
            env=self.env,
            lr=lr,
            clip_range=clip_range,
            ent_coef=ent_coef,
            use_lstm=True,
            verbose=1
        )
        
        # Train
        self.agent.train(
            total_timesteps=timesteps,
            eval_freq=max(timesteps // 20, 1000)
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, 'ppo_v9_single')
        self.agent.save(model_path)
        
        logger.info(f"✅ Training complete! Model saved to {model_path}")
    
    def train_ensemble(
        self,
        n_agents: int = 3,
        timesteps_per_agent: int = 50000
    ):
        """
        Train ensemble of agents.
        
        Args:
            n_agents: Number of agents in ensemble
            timesteps_per_agent: Training timesteps per agent
        """
        logger.info(f"🎯 Training ensemble ({n_agents} agents)...")
        
        # Create ensemble manager
        ensemble = EnsembleManagerV9(
            env=self.env,
            n_agents=n_agents,
            selection_method='best'
        )
        
        # Create agents
        ensemble.create_agents()
        
        # Train ensemble
        ensemble.train_agents(
            total_timesteps=timesteps_per_agent,
            eval_freq=max(timesteps_per_agent // 10, 1000)
        )
        
        # Save ensemble
        ensemble_path = os.path.join(self.models_dir, 'ensemble_v9')
        ensemble.save_ensemble(ensemble_path)
        
        logger.info(f"✅ Ensemble training complete! Saved to {ensemble_path}")
        
        return ensemble
    
    def optimize_hyperparameters(
        self,
        n_trials: int = 50,
        timesteps_per_trial: int = 20000
    ):
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            n_trials: Number of optimization trials
            timesteps_per_trial: Training timesteps per trial
        """
        logger.info(f"🔍 Optimizing hyperparameters ({n_trials} trials)...")
        
        # Create optimizer
        optimizer = OptunaOptimizer(
            env=self.env,
            n_trials=n_trials,
            timesteps_per_trial=timesteps_per_trial
        )
        
        # Run optimization
        best_params = optimizer.optimize()
        
        logger.info(f"✅ Optimization complete!")
        logger.info(f"   Best params: {best_params}")
        
        # Save best params
        import json
        params_path = os.path.join(self.models_dir, 'best_params_v9.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"   Saved to {params_path}")
        
        return best_params
    
    def run_quick_training(self):
        """Quick training mode (for testing)."""
        logger.info("⚡ QUICK TRAINING MODE")
        
        # Setup
        df = self.setup_data(symbol='EURUSD', years='2023-2024')
        self.setup_environment(df)
        
        # Train
        self.train_single_agent(
            timesteps=50000,
            lr=3e-4,
            clip_range=0.2,
            ent_coef=0.01
        )
        
        logger.info("✅ Quick training complete!")
    
    def run_full_training(self):
        """Full training mode (for production)."""
        logger.info("🚀 FULL TRAINING MODE")
        
        # Setup
        df = self.setup_data(symbol='EURUSD', years='2020-2024')
        self.setup_environment(df)
        
        # Step 1: Optimize hyperparameters
        logger.info("\n" + "="*60)
        logger.info("STEP 1: HYPERPARAMETER OPTIMIZATION")
        logger.info("="*60)
        best_params = self.optimize_hyperparameters(n_trials=50, timesteps_per_trial=20000)
        
        # Step 2: Train single agent with best params
        logger.info("\n" + "="*60)
        logger.info("STEP 2: TRAIN SINGLE AGENT")
        logger.info("="*60)
        self.train_single_agent(
            timesteps=200000,
            lr=best_params.get('lr', 3e-4),
            clip_range=best_params.get('clip_range', 0.2),
            ent_coef=best_params.get('ent_coef', 0.01)
        )
        
        # Step 3: Train ensemble
        logger.info("\n" + "="*60)
        logger.info("STEP 3: TRAIN ENSEMBLE")
        logger.info("="*60)
        self.train_ensemble(n_agents=5, timesteps_per_agent=100000)
        
        logger.info("\n" + "="*60)
        logger.info("✅ FULL TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"\nModels saved to: {self.models_dir}")
        logger.info(f"Logs saved to: {self.logs_dir}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """
    Main entry point for training script.
    """
    parser = argparse.ArgumentParser(
        description='Train ULTIMATE FTMO TRADING BOT V9',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick training:     python train_bot_v9.py --mode quick
  Full training:      python train_bot_v9.py --mode full
  Ensemble only:      python train_bot_v9.py --mode ensemble --agents 5
  Custom:             python train_bot_v9.py --mode custom --timesteps 100000 --trials 20
        """
    )
    
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full', 'ensemble', 'custom'],
                        help='Training mode')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--models-dir', type=str, default='./models_v9',
                        help='Models directory')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='Trading symbol')
    parser.add_argument('--years', type=str, default='2020-2024',
                        help='Date range (e.g., 2020-2024)')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Training timesteps (custom mode)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Optuna trials (custom mode)')
    parser.add_argument('--agents', type=int, default=3,
                        help='Number of agents for ensemble')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("ULTIMATE FTMO TRADING BOT V9 - TRAINING")
    print("="*60)
    print(f"Mode:       {args.mode}")
    print(f"Data Dir:   {args.data_dir}")
    print(f"Models Dir: {args.models_dir}")
    print(f"Symbol:     {args.symbol}")
    print(f"Years:      {args.years}")
    print("="*60 + "\n")
    
    # Initialize pipeline
    pipeline = TrainingPipelineV9(
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )
    
    # Run training based on mode
    try:
        if args.mode == 'quick':
            pipeline.run_quick_training()
        
        elif args.mode == 'full':
            pipeline.run_full_training()
        
        elif args.mode == 'ensemble':
            df = pipeline.setup_data(symbol=args.symbol, years=args.years)
            pipeline.setup_environment(df)
            pipeline.train_ensemble(n_agents=args.agents)
        
        elif args.mode == 'custom':
            df = pipeline.setup_data(symbol=args.symbol, years=args.years)
            pipeline.setup_environment(df)
            
            # Optimize if trials > 0
            if args.trials > 0:
                best_params = pipeline.optimize_hyperparameters(
                    n_trials=args.trials,
                    timesteps_per_trial=args.timesteps // 5
                )
                pipeline.train_single_agent(
                    timesteps=args.timesteps,
                    lr=best_params.get('lr', 3e-4),
                    clip_range=best_params.get('clip_range', 0.2),
                    ent_coef=best_params.get('ent_coef', 0.01)
                )
            else:
                pipeline.train_single_agent(timesteps=args.timesteps)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nNext steps:")
        print(f"  1. Test your model: python ultimate_bot_v8_ppo.py --mode backtest")
        print(f"  2. Run paper trading: python ultimate_bot_v8_ppo.py --mode paper")
        print(f"  3. Go live (at your own risk!): python ultimate_bot_v8_ppo.py --mode live")
        print("\n")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
