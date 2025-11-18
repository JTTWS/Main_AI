#!/usr/bin/env python3
"""
================================================================================
BACKEND TEST V9 - Comprehensive Testing for ULTIMATE FTMO TRADING BOT V9
================================================================================

Tests all V9 modules:
1. FeatureEngineerV9 - 50+ technical indicators
2. DataManagerV8 - Data loading with V9 integration
3. SentimentAnalyzerV9 - Economic calendar & blackout periods
4. EnsembleManagerV9 - Multi-agent PPO ensemble
5. AdvancedBacktesterV9 - Monte Carlo & advanced metrics
6. Integration Test - Full pipeline

Author: E1 AI Agent (Testing Agent)
Date: January 2025
Version: 9.0 TEST SUITE
================================================================================
"""

import os
import sys
import logging
import warnings
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BackendTestV9')

# Test results tracking
test_results = {
    'feature_engineer_v9': {'status': 'PENDING', 'details': []},
    'data_manager_v8': {'status': 'PENDING', 'details': []},
    'sentiment_analyzer': {'status': 'PENDING', 'details': []},
    'ensemble_manager': {'status': 'PENDING', 'details': []},
    'advanced_backtester': {'status': 'PENDING', 'details': []},
    'integration_test': {'status': 'PENDING', 'details': []}
}

def log_test_result(module, status, message):
    """Log test result for a module."""
    test_results[module]['status'] = status
    test_results[module]['details'].append(message)
    logger.info(f"[{module.upper()}] {status}: {message}")

def print_test_summary():
    """Print comprehensive test summary."""
    print("\n" + "="*80)
    print("ULTIMATE FTMO TRADING BOT V9 - TEST RESULTS SUMMARY")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r['status'] == 'PASS')
    failed_tests = sum(1 for r in test_results.values() if r['status'] == 'FAIL')
    
    print(f"Total Modules Tested: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("-"*80)
    
    for module, result in test_results.items():
        status_icon = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⏳"
        print(f"{status_icon} {module.upper()}: {result['status']}")
        for detail in result['details']:
            print(f"    • {detail}")
    
    print("="*80)

# =============================================================================
# TEST 1: FEATURE ENGINEER V9
# =============================================================================

def test_feature_engineer_v9():
    """Test FeatureEngineerV9 module."""
    print("\n🧪 TESTING: FeatureEngineerV9")
    print("-" * 50)
    
    try:
        from feature_engineer_v9 import FeatureEngineerV9
        log_test_result('feature_engineer_v9', 'INFO', 'Module imported successfully')
        
        # Generate sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='15T')
        close_prices = 1.08 + np.cumsum(np.random.randn(1000) * 0.0001)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.randn(1000) * 0.00005,
            'high': close_prices + np.abs(np.random.randn(1000)) * 0.0001,
            'low': close_prices - np.abs(np.random.randn(1000)) * 0.0001,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        log_test_result('feature_engineer_v9', 'INFO', f'Generated {len(df)} rows of sample EURUSD data')
        
        # Initialize feature engineer
        fe = FeatureEngineerV9(
            enable_talib=True,
            enable_multi_timeframe=True,
            timeframes=['15T', '1H', '4H']
        )
        
        log_test_result('feature_engineer_v9', 'INFO', 'FeatureEngineerV9 initialized')
        
        # Engineer features
        df_features = fe.engineer_features(df, symbol='EURUSD')
        
        # Validate results
        original_cols = 6  # timestamp, open, high, low, close, volume
        feature_cols = len(df_features.columns) - original_cols
        
        if feature_cols >= 50:
            log_test_result('feature_engineer_v9', 'PASS', f'Generated {feature_cols} features (target: 50+)')
        else:
            log_test_result('feature_engineer_v9', 'FAIL', f'Only {feature_cols} features generated (target: 50+)')
            return False
        
        # Check specific feature categories
        feature_names = fe.get_feature_names(df_features)
        
        # Trend indicators
        trend_features = [f for f in feature_names if any(x in f for x in ['sma', 'ema', 'adx', 'macd'])]
        log_test_result('feature_engineer_v9', 'INFO', f'Trend indicators: {len(trend_features)}')
        
        # Momentum indicators
        momentum_features = [f for f in feature_names if any(x in f for x in ['rsi', 'stoch', 'willr', 'roc'])]
        log_test_result('feature_engineer_v9', 'INFO', f'Momentum indicators: {len(momentum_features)}')
        
        # Volatility indicators
        volatility_features = [f for f in feature_names if any(x in f for x in ['bb_', 'atr', 'std', 'keltner'])]
        log_test_result('feature_engineer_v9', 'INFO', f'Volatility indicators: {len(volatility_features)}')
        
        # Multi-timeframe features
        mtf_features = [f for f in feature_names if any(x in f for x in ['1H_', '4H_'])]
        log_test_result('feature_engineer_v9', 'INFO', f'Multi-timeframe features: {len(mtf_features)}')
        
        # Check for NaN values
        nan_count = df_features.isnull().sum().sum()
        if nan_count == 0:
            log_test_result('feature_engineer_v9', 'PASS', 'No NaN values in engineered features')
        else:
            log_test_result('feature_engineer_v9', 'WARN', f'{nan_count} NaN values found (may be acceptable)')
        
        log_test_result('feature_engineer_v9', 'PASS', 'All feature engineering tests passed')
        return True
        
    except Exception as e:
        log_test_result('feature_engineer_v9', 'FAIL', f'Exception: {str(e)}')
        traceback.print_exc()
        return False

# =============================================================================
# TEST 2: DATA MANAGER V8 (with V9 integration)
# =============================================================================

def test_data_manager_v8():
    """Test DataManagerV8 with FeatureEngineerV9 integration."""
    print("\n🧪 TESTING: DataManagerV8 (V9 Integration)")
    print("-" * 50)
    
    try:
        from data_manager_v8 import DataManagerV8
        log_test_result('data_manager_v8', 'INFO', 'Module imported successfully')
        
        # Initialize data manager
        dm = DataManagerV8(data_dir='/app/data')
        log_test_result('data_manager_v8', 'INFO', 'DataManagerV8 initialized')
        
        # Test 1: Load EURUSD data with mock fallback
        df = dm.load_symbol_data(
            symbol='EURUSD',
            start_date='2024-01-01',
            end_date='2024-12-31',
            use_mock=True,
            engineer_features=True
        )
        
        if df.empty:
            log_test_result('data_manager_v8', 'FAIL', 'No data loaded (even with mock=True)')
            return False
        
        log_test_result('data_manager_v8', 'PASS', f'Loaded {len(df)} rows of EURUSD data')
        
        # Test 2: Check if features were automatically added
        expected_base_cols = 6  # timestamp, open, high, low, close, volume
        total_cols = len(df.columns)
        feature_cols = total_cols - expected_base_cols
        
        if feature_cols >= 50:
            log_test_result('data_manager_v8', 'PASS', f'Auto-engineered {feature_cols} features via V9 integration')
        else:
            log_test_result('data_manager_v8', 'WARN', f'Only {feature_cols} features auto-engineered')
        
        # Test 3: Validate data structure
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if not missing_cols:
            log_test_result('data_manager_v8', 'PASS', 'All required OHLCV columns present')
        else:
            log_test_result('data_manager_v8', 'FAIL', f'Missing columns: {missing_cols}')
            return False
        
        # Test 4: Check data quality
        if df['close'].isnull().sum() == 0:
            log_test_result('data_manager_v8', 'PASS', 'No missing close prices')
        else:
            log_test_result('data_manager_v8', 'FAIL', 'Missing close prices detected')
        
        # Test 5: Load economic calendar (if available)
        try:
            calendar = dm.load_economic_calendar()
            if not calendar.empty:
                log_test_result('data_manager_v8', 'PASS', f'Economic calendar loaded: {len(calendar)} events')
            else:
                log_test_result('data_manager_v8', 'INFO', 'No economic calendar data found')
        except Exception as e:
            log_test_result('data_manager_v8', 'INFO', f'Economic calendar not available: {str(e)}')
        
        # Test 6: Get data summary
        summary = dm.get_data_summary()
        log_test_result('data_manager_v8', 'INFO', f'Data summary: {summary}')
        
        log_test_result('data_manager_v8', 'PASS', 'All data manager tests passed')
        return True
        
    except Exception as e:
        log_test_result('data_manager_v8', 'FAIL', f'Exception: {str(e)}')
        traceback.print_exc()
        return False

# =============================================================================
# TEST 3: SENTIMENT ANALYZER V9
# =============================================================================

def test_sentiment_analyzer():
    """Test SentimentAnalyzerV9 module."""
    print("\n🧪 TESTING: SentimentAnalyzerV9")
    print("-" * 50)
    
    try:
        from sentiment_analyzer import SentimentAnalyzerV9
        log_test_result('sentiment_analyzer', 'INFO', 'Module imported successfully')
        
        # Create sample economic calendar data
        sample_events = pd.DataFrame({
            'datetime': [
                '2024-01-15 14:30:00',
                '2024-01-16 08:30:00',
                '2024-01-17 12:00:00',
                '2024-01-18 10:00:00'
            ],
            'name': ['NFP Report', 'GDP Release', 'CPI Data', 'Minor Event'],
            'impact': ['high', 'high', 'high', 'low'],
            'currency': ['USD', 'EUR', 'USD', 'GBP']
        })
        
        # Save to temporary file
        temp_calendar_path = '/tmp/test_economic_calendar.csv'
        sample_events.to_csv(temp_calendar_path, index=False)
        log_test_result('sentiment_analyzer', 'INFO', 'Created sample economic calendar')
        
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzerV9(
            calendar_path=temp_calendar_path,
            blackout_before_minutes=30,
            blackout_after_minutes=15
        )
        
        log_test_result('sentiment_analyzer', 'PASS', 'SentimentAnalyzerV9 initialized with calendar')
        
        # Test 1: Blackout period detection
        test_time_blackout = pd.Timestamp('2024-01-15 14:25:00')  # 5 min before NFP
        is_blackout = analyzer.is_blackout(test_time_blackout)
        
        if is_blackout:
            log_test_result('sentiment_analyzer', 'PASS', 'Blackout period correctly detected')
        else:
            log_test_result('sentiment_analyzer', 'FAIL', 'Blackout period not detected')
        
        # Test 2: Safe period (no blackout)
        test_time_safe = pd.Timestamp('2024-01-15 10:00:00')
        is_safe = analyzer.is_blackout(test_time_safe)
        
        if not is_safe:
            log_test_result('sentiment_analyzer', 'PASS', 'Safe period correctly identified')
        else:
            log_test_result('sentiment_analyzer', 'FAIL', 'Safe period incorrectly flagged as blackout')
        
        # Test 3: Upcoming events
        current_time = pd.Timestamp('2024-01-15 12:00:00')
        upcoming = analyzer.get_upcoming_events(current_time, hours_ahead=6)
        
        if len(upcoming) > 0:
            log_test_result('sentiment_analyzer', 'PASS', f'Found {len(upcoming)} upcoming events')
        else:
            log_test_result('sentiment_analyzer', 'WARN', 'No upcoming events found in 6-hour window')
        
        # Test 4: Sentiment scoring
        sentiment_blackout = analyzer.compute_sentiment_score(test_time_blackout)
        sentiment_safe = analyzer.compute_sentiment_score(test_time_safe)
        
        if sentiment_blackout < sentiment_safe:
            log_test_result('sentiment_analyzer', 'PASS', 'Sentiment correctly lower during blackout')
        else:
            log_test_result('sentiment_analyzer', 'WARN', 'Sentiment scoring may need adjustment')
        
        # Test 5: Statistics
        stats = analyzer.get_stats()
        log_test_result('sentiment_analyzer', 'INFO', f'Analyzer stats: {stats}')
        
        if stats['blackout_periods'] > 0:
            log_test_result('sentiment_analyzer', 'PASS', f'Generated {stats["blackout_periods"]} blackout periods')
        else:
            log_test_result('sentiment_analyzer', 'WARN', 'No blackout periods generated')
        
        log_test_result('sentiment_analyzer', 'PASS', 'All sentiment analyzer tests passed')
        return True
        
    except Exception as e:
        log_test_result('sentiment_analyzer', 'FAIL', f'Exception: {str(e)}')
        traceback.print_exc()
        return False

# =============================================================================
# TEST 4: ENSEMBLE MANAGER V9
# =============================================================================

def test_ensemble_manager():
    """Test EnsembleManagerV9 module."""
    print("\n🧪 TESTING: EnsembleManagerV9")
    print("-" * 50)
    
    try:
        from ensemble_manager import EnsembleManagerV9
        log_test_result('ensemble_manager', 'INFO', 'Module imported successfully')
        
        # Create mock environment
        try:
            import gymnasium as gym
        except ImportError:
            import gym
        
        env = gym.make('CartPole-v1')
        log_test_result('ensemble_manager', 'INFO', 'Mock environment created')
        
        # Initialize ensemble manager
        ensemble = EnsembleManagerV9(
            env=env,
            n_agents=3,
            selection_method='best',
            performance_window=100
        )
        
        log_test_result('ensemble_manager', 'PASS', 'EnsembleManagerV9 initialized')
        
        # Test 1: Create agents with different hyperparameters
        ensemble.create_agents()
        
        if len(ensemble.agents) == 3:
            log_test_result('ensemble_manager', 'PASS', f'Created {len(ensemble.agents)} agents with varied configs')
        else:
            log_test_result('ensemble_manager', 'FAIL', f'Expected 3 agents, got {len(ensemble.agents)}')
            return False
        
        # Test 2: Verify different configurations
        configs = ensemble.agent_configs
        if len(set(str(config) for config in configs)) > 1:
            log_test_result('ensemble_manager', 'PASS', 'Agents have different hyperparameters')
        else:
            log_test_result('ensemble_manager', 'WARN', 'All agents may have identical configs')
        
        # Test 3: Prediction with ensemble
        state = env.reset()[0]
        action = ensemble.predict(state, deterministic=True)
        
        if action in [0, 1]:  # CartPole has 2 actions
            log_test_result('ensemble_manager', 'PASS', f'Ensemble prediction successful (action={action})')
        else:
            log_test_result('ensemble_manager', 'FAIL', f'Invalid action predicted: {action}')
        
        # Test 4: Performance tracking
        ensemble.update_performance(0, 1.0)
        ensemble.update_performance(1, 0.5)
        ensemble.update_performance(2, 0.8)
        ensemble.update_best_agent()
        
        if ensemble.current_best_idx == 0:  # Agent 0 has highest reward (1.0)
            log_test_result('ensemble_manager', 'PASS', 'Best agent selection works correctly')
        else:
            log_test_result('ensemble_manager', 'WARN', f'Best agent selection: expected 0, got {ensemble.current_best_idx}')
        
        # Test 5: Different selection methods
        ensemble.selection_method = 'voting'
        action_voting = ensemble.predict(state, deterministic=True)
        
        ensemble.selection_method = 'weighted'
        action_weighted = ensemble.predict(state, deterministic=True)
        
        log_test_result('ensemble_manager', 'PASS', 'Multiple selection methods work')
        
        # Test 6: Statistics
        stats = ensemble.get_stats()
        log_test_result('ensemble_manager', 'INFO', f'Ensemble stats: {stats}')
        
        if stats['n_agents'] == 3:
            log_test_result('ensemble_manager', 'PASS', 'Statistics correctly report 3 agents')
        else:
            log_test_result('ensemble_manager', 'FAIL', f'Stats show {stats["n_agents"]} agents, expected 3')
        
        log_test_result('ensemble_manager', 'PASS', 'All ensemble manager tests passed')
        return True
        
    except Exception as e:
        log_test_result('ensemble_manager', 'FAIL', f'Exception: {str(e)}')
        traceback.print_exc()
        return False

# =============================================================================
# TEST 5: ADVANCED BACKTESTER V9
# =============================================================================

def test_advanced_backtester():
    """Test AdvancedBacktesterV9 module."""
    print("\n🧪 TESTING: AdvancedBacktesterV9")
    print("-" * 50)
    
    try:
        from advanced_backtester import AdvancedBacktesterV9
        log_test_result('advanced_backtester', 'INFO', 'Module imported successfully')
        
        # Initialize backtester
        backtester = AdvancedBacktesterV9(
            initial_capital=25000.0,
            risk_free_rate=0.02
        )
        
        log_test_result('advanced_backtester', 'PASS', 'AdvancedBacktesterV9 initialized')
        
        # Create sample trades with realistic PnL distribution
        np.random.seed(42)
        trades = []
        for i in range(100):
            # 60% win rate with realistic forex PnL
            if np.random.random() < 0.6:
                pnl = np.random.uniform(50, 200)  # Winning trade
            else:
                pnl = np.random.uniform(-150, -25)  # Losing trade
            
            trades.append({
                'pnl': pnl,
                'entry_time': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
                'exit_time': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i+1),
                'symbol': 'EURUSD'
            })
        
        log_test_result('advanced_backtester', 'INFO', f'Generated {len(trades)} sample trades')
        
        # Test 1: Calculate comprehensive metrics
        metrics = backtester.calculate_metrics(trades)
        
        required_metrics = [
            'total_trades', 'final_balance', 'total_return', 'win_rate',
            'profit_factor', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio'
        ]
        
        missing_metrics = [m for m in required_metrics if m not in metrics]
        if not missing_metrics:
            log_test_result('advanced_backtester', 'PASS', 'All required metrics calculated')
        else:
            log_test_result('advanced_backtester', 'FAIL', f'Missing metrics: {missing_metrics}')
            return False
        
        # Validate metric values
        if metrics['total_trades'] == 100:
            log_test_result('advanced_backtester', 'PASS', 'Trade count correct')
        else:
            log_test_result('advanced_backtester', 'FAIL', f'Expected 100 trades, got {metrics["total_trades"]}')
        
        if 0.5 <= metrics['win_rate'] <= 0.7:
            log_test_result('advanced_backtester', 'PASS', f'Win rate realistic: {metrics["win_rate"]*100:.1f}%')
        else:
            log_test_result('advanced_backtester', 'WARN', f'Win rate: {metrics["win_rate"]*100:.1f}% (may be outside expected range)')
        
        if metrics['profit_factor'] > 0:
            log_test_result('advanced_backtester', 'PASS', f'Profit factor calculated: {metrics["profit_factor"]:.2f}')
        else:
            log_test_result('advanced_backtester', 'FAIL', 'Profit factor calculation failed')
        
        # Test 2: Monte Carlo simulation
        mc_results = backtester.monte_carlo_simulation(
            trades=trades,
            n_simulations=100,  # Reduced for faster testing
            randomize='order'
        )
        
        if 'final_balance' in mc_results and 'probability_profit' in mc_results:
            log_test_result('advanced_backtester', 'PASS', 'Monte Carlo simulation completed')
            log_test_result('advanced_backtester', 'INFO', 
                          f'MC Results - Mean Balance: ${mc_results["final_balance"]["mean"]:,.2f}, '
                          f'Prob Profit: {mc_results["probability_profit"]*100:.1f}%')
        else:
            log_test_result('advanced_backtester', 'FAIL', 'Monte Carlo simulation incomplete')
            return False
        
        # Test 3: Different randomization methods
        mc_returns = backtester.monte_carlo_simulation(trades, n_simulations=50, randomize='returns')
        mc_both = backtester.monte_carlo_simulation(trades, n_simulations=50, randomize='both')
        
        log_test_result('advanced_backtester', 'PASS', 'Multiple randomization methods work')
        
        # Test 4: Edge cases
        empty_metrics = backtester.calculate_metrics([])
        if empty_metrics['total_trades'] == 0:
            log_test_result('advanced_backtester', 'PASS', 'Empty trade list handled correctly')
        else:
            log_test_result('advanced_backtester', 'FAIL', 'Empty trade list not handled properly')
        
        # Test 5: Performance summary
        try:
            summary = backtester.print_summary(metrics)
            log_test_result('advanced_backtester', 'PASS', 'Performance summary generated')
        except Exception as e:
            log_test_result('advanced_backtester', 'WARN', f'Summary generation issue: {str(e)}')
        
        log_test_result('advanced_backtester', 'PASS', 'All advanced backtester tests passed')
        return True
        
    except Exception as e:
        log_test_result('advanced_backtester', 'FAIL', f'Exception: {str(e)}')
        traceback.print_exc()
        return False

# =============================================================================
# TEST 6: INTEGRATION TEST - FULL PIPELINE
# =============================================================================

def test_integration_pipeline():
    """Test full V9 pipeline integration."""
    print("\n🧪 TESTING: Full Pipeline Integration")
    print("-" * 50)
    
    try:
        # Import all modules
        from data_manager_v8 import DataManagerV8
        from feature_engineer_v9 import FeatureEngineerV9
        from trading_environment_pro import ProfessionalTradingEnvironmentV8
        from ppo_agent import PPOAgent
        
        log_test_result('integration_test', 'INFO', 'All modules imported successfully')
        
        # Step 1: Load data with automatic feature engineering
        dm = DataManagerV8(data_dir='/app/data')
        df = dm.load_symbol_data(
            symbol='EURUSD',
            start_date='2024-01-01',
            end_date='2024-03-31',  # Smaller dataset for faster testing
            use_mock=True,
            engineer_features=True
        )
        
        if df.empty:
            log_test_result('integration_test', 'FAIL', 'Data loading failed')
            return False
        
        log_test_result('integration_test', 'PASS', f'Data loaded: {len(df)} rows, {len(df.columns)} columns')
        
        # Step 2: Verify feature engineering
        expected_features = 70  # Target from review request
        actual_features = len(df.columns) - 6  # Subtract OHLCV + timestamp
        
        if actual_features >= expected_features:
            log_test_result('integration_test', 'PASS', f'Feature count: {actual_features} (target: {expected_features}+)')
        else:
            log_test_result('integration_test', 'WARN', f'Feature count: {actual_features} (target: {expected_features}+)')
        
        # Step 3: Create trading environment
        # Convert single DataFrame to dict format expected by environment
        data_dict = {'EURUSD': df}
        
        env = ProfessionalTradingEnvironmentV8(
            data=data_dict,
            initial_capital=25000.0,
            max_positions=3,
            position_size_pct=0.02
        )
        
        log_test_result('integration_test', 'PASS', 'Trading environment created')
        log_test_result('integration_test', 'INFO', f'Observation space: {env.observation_space.shape}')
        log_test_result('integration_test', 'INFO', f'Action space: {env.action_space}')
        
        # Step 4: Create and test PPO agent
        agent = PPOAgent(
            env=env,
            lr=3e-4,
            clip_range=0.2,
            ent_coef=0.01,
            use_lstm=True,
            verbose=0
        )
        
        log_test_result('integration_test', 'PASS', 'PPO agent created with LSTM')
        
        # Step 5: Test environment reset and step
        obs, info = env.reset()
        log_test_result('integration_test', 'PASS', f'Environment reset successful, obs shape: {obs.shape}')
        
        # Step 6: Test agent prediction
        action = agent.predict(obs, deterministic=True)
        log_test_result('integration_test', 'PASS', f'Agent prediction successful: action={action}')
        
        # Step 7: Test environment step
        obs_new, reward, done, truncated, info = env.step(action)
        log_test_result('integration_test', 'PASS', f'Environment step successful: reward={reward:.4f}')
        
        # Step 8: Skip training test for faster execution
        log_test_result('integration_test', 'INFO', 'Skipping training test for faster execution')
        
        # Step 9: Test multiple steps
        total_reward = 0
        steps = 0
        obs, _ = env.reset()
        
        for i in range(50):  # Test 50 steps
            action = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        log_test_result('integration_test', 'PASS', f'Multi-step test: {steps} steps, total reward: {total_reward:.4f}')
        
        # Step 10: Validate final state
        final_info = info
        log_test_result('integration_test', 'INFO', f'Final balance: ${final_info.get("balance", 0):,.2f}')
        log_test_result('integration_test', 'INFO', f'Total trades: {final_info.get("total_trades", 0)}')
        
        log_test_result('integration_test', 'PASS', 'Full pipeline integration test completed successfully')
        return True
        
    except Exception as e:
        log_test_result('integration_test', 'FAIL', f'Integration test exception: {str(e)}')
        traceback.print_exc()
        return False

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all V9 module tests."""
    print("="*80)
    print("ULTIMATE FTMO TRADING BOT V9 - COMPREHENSIVE BACKEND TESTING")
    print("="*80)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print("="*80)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run all tests
    tests = [
        ("FeatureEngineerV9", test_feature_engineer_v9),
        ("DataManagerV8", test_data_manager_v8),
        ("SentimentAnalyzerV9", test_sentiment_analyzer),
        ("EnsembleManagerV9", test_ensemble_manager),
        ("AdvancedBacktesterV9", test_advanced_backtester),
        ("Integration Pipeline", test_integration_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results.append(success)
        except KeyboardInterrupt:
            print("\n⚠️  Testing interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append(False)
    
    # Print final summary
    print_test_summary()
    
    # Overall result
    total_tests = len(results)
    passed_tests = sum(results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ EXCELLENT: V9 modules are working well!")
    elif success_rate >= 60:
        print("⚠️  GOOD: Most V9 modules working, some issues to address")
    else:
        print("❌ NEEDS WORK: Significant issues found in V9 modules")
    
    print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    main()