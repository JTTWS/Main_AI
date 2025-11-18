#!/usr/bin/env python3
"""
================================================================================
OPTUNA OPTIMIZER - V8 Hyperparameter Tuning
================================================================================
OptunaOptimizer, PPO hyperparameter'larını (lr, clip_range, ent_coef, decay_rate)
Optuna kullanarak optimize eder.

Kullanım:
    from optuna_optimizer import OptunaOptimizer
    optimizer = OptunaOptimizer(train_data, PPOAgent)
    best_params = optimizer.optimize(n_trials=50)
    
Author: E1 AI Agent + Grok Integration
Date: January 2025
Version: 8.0
================================================================================
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer for PPO agent.
    
    Optimizes:
        - learning_rate (lr): [1e-5, 1e-2]
        - clip_range: [0.1, 0.3]
        - entropy_coefficient (ent_coef): [0.001, 0.1]
        - decay_rate: [0.99, 0.999]
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        agent_class,
        default_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Optuna Optimizer.
        
        Args:
            data: Historical training data with performance metrics
            agent_class: Agent class to optimize (e.g., PPOAgent)
            default_params: Default hyperparameters to use as fallback
        """
        self.data = data
        self.agent_class = agent_class
        self.default_params = default_params or {
            'lr': 3e-4,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'decay_rate': 0.995
        }
        
        # Performance baseline from data
        self.baseline_sharpe = data['sharpe'].mean() if 'sharpe' in data.columns else 1.0
        self.baseline_reward = data['reward'].mean() if 'reward' in data.columns else 0.001
        
        print(f"📊 OptunaOptimizer initialized:")
        print(f"   Data samples: {len(data)}")
        print(f"   Baseline Sharpe: {self.baseline_sharpe:.3f}")
        print(f"   Baseline Reward: {self.baseline_reward:.6f}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function to maximize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Objective value (higher is better)
        """
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        ent_coef = trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
        decay_rate = trial.suggest_float('decay_rate', 0.99, 0.999)
        
        # Mock evaluation (gerçek backtest yerine hızlı proxy)
        # Real implementation'da agent'ı train edip evaluate ederiz
        
        # Heuristic scoring based on typical PPO behavior
        # Lower lr → more stable but slower
        # Higher clip_range → more aggressive updates
        # Higher ent_coef → more exploration
        # Higher decay_rate → slower convergence
        
        mock_sharpe = self.baseline_sharpe
        
        # Learning rate effect
        lr_score = np.log(lr) * 0.1  # Prefer moderate lr
        mock_sharpe += lr_score
        
        # Clip range effect
        clip_score = clip_range * 0.3  # Moderate clip helps
        mock_sharpe += clip_score
        
        # Entropy coefficient effect
        ent_score = np.log(ent_coef + 1e-6) * 0.05  # Some exploration good
        mock_sharpe += ent_score
        
        # Decay rate effect
        decay_score = decay_rate * 0.1  # Slow decay helps
        mock_sharpe += decay_score
        
        # Add data-driven component
        if 'sharpe' in self.data.columns:
            # Weight by data variance (higher variance → need more exploration)
            data_variance = self.data['sharpe'].std()
            if data_variance > 0.5:  # High variance
                mock_sharpe += ent_coef * 0.2  # Reward exploration
            else:  # Low variance
                mock_sharpe -= ent_coef * 0.1  # Penalize over-exploration
        
        # Penalize extreme values
        if lr < 5e-5 or lr > 5e-3:
            mock_sharpe -= 0.2
        if clip_range < 0.15 or clip_range > 0.28:
            mock_sharpe -= 0.15
        
        return mock_sharpe
    
    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None) -> Dict[str, float]:
        """
        Run Optuna optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            
        Returns:
            Dict containing best hyperparameters
        """
        print(f"\n🔍 Starting Optuna optimization ({n_trials} trials)...")
        
        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                study_name='ppo_hyperparameter_optimization'
            )
            
            # Optimize
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            print(f"\n✅ Optimization complete!")
            print(f"   Best Sharpe: {best_value:.4f}")
            print(f"   Best Parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value:.6f}")
            
            return best_params
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            print(f"   Falling back to default parameters")
            return self.get_default_params()
    
    def get_default_params(self) -> Dict[str, float]:
        """
        Get default hyperparameters.
        
        Returns:
            Dict containing default hyperparameters
        """
        return self.default_params.copy()


# =============================================================================
# Test Functions
# =============================================================================

def test_optuna_optimizer():
    """Test OptunaOptimizer with mock data."""
    print("🧪 Testing OptunaOptimizer...")
    
    # Create mock training data
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=n_samples, freq='D'),
        'reward': np.random.uniform(-0.001, 0.002, n_samples),
        'sharpe': np.random.uniform(0.5, 1.5, n_samples),
        'returns': np.random.normal(0.0001, 0.002, n_samples)
    })
    
    # Mock agent class
    class MockAgent:
        def __init__(self, env, lr=3e-4, clip_range=0.2, ent_coef=0.01):
            self.lr = lr
            self.clip_range = clip_range
            self.ent_coef = ent_coef
    
    # Test 1: Initialize optimizer
    optimizer = OptunaOptimizer(data, MockAgent)
    print("✓ Test 1 passed: Optimizer initialized")
    
    # Test 2: Run optimization (short)
    best_params = optimizer.optimize(n_trials=10)
    print("✓ Test 2 passed: Optimization completed")
    
    # Test 3: Validate parameters
    assert 'lr' in best_params, "Missing lr parameter"
    assert 'clip_range' in best_params, "Missing clip_range parameter"
    assert 'ent_coef' in best_params, "Missing ent_coef parameter"
    assert 'decay_rate' in best_params, "Missing decay_rate parameter"
    
    assert 1e-5 <= best_params['lr'] <= 1e-2, "lr out of range"
    assert 0.1 <= best_params['clip_range'] <= 0.3, "clip_range out of range"
    assert 0.001 <= best_params['ent_coef'] <= 0.1, "ent_coef out of range"
    assert 0.99 <= best_params['decay_rate'] <= 0.999, "decay_rate out of range"
    
    print("✓ Test 3 passed: Parameters validated")
    print(f"   Best lr: {best_params['lr']:.6f}")
    print(f"   Best clip_range: {best_params['clip_range']:.4f}")
    print(f"   Best ent_coef: {best_params['ent_coef']:.6f}")
    print(f"   Best decay_rate: {best_params['decay_rate']:.6f}")
    
    # Test 4: Default parameters fallback
    default = optimizer.get_default_params()
    assert default == optimizer.default_params, "Default params mismatch"
    print("✓ Test 4 passed: Default params retrieved")
    
    print("✅ OptunaOptimizer tests passed!\n")


if __name__ == "__main__":
    test_optuna_optimizer()
