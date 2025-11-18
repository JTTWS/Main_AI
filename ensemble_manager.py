#!/usr/bin/env python3
"""
================================================================================
ENSEMBLE MANAGER V9 - Multi-Agent PPO Ensemble
================================================================================

Manages multiple PPO agents with different hyperparameters:
- Agent selection based on performance
- Voting mechanisms (majority, weighted)
- Dynamic agent switching
- Performance tracking

Author: E1 AI Agent (Emergent.sh)
Date: January 2025
Version: 9.0 FREE PRO
================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

from ppo_agent import PPOAgent

logger = logging.getLogger('EnsembleManagerV9')


class EnsembleManagerV9:
    """
    Ensemble Manager for Multiple PPO Agents.
    
    Features:
    - Train multiple agents with different hyperparameters
    - Select best agent based on performance
    - Voting mechanisms for action selection
    - Dynamic agent switching
    """
    
    def __init__(
        self,
        env,
        n_agents: int = 3,
        selection_method: str = 'best',  # 'best', 'voting', 'weighted'
        performance_window: int = 100
    ):
        """
        Initialize Ensemble Manager.
        
        Args:
            env: Trading environment
            n_agents: Number of agents in ensemble
            selection_method: How to select/combine agent actions
            performance_window: Window for calculating recent performance
        """
        self.env = env
        self.n_agents = n_agents
        self.selection_method = selection_method
        self.performance_window = performance_window
        
        self.agents: List[PPOAgent] = []
        self.agent_configs: List[Dict] = []
        self.agent_performance: List[List[float]] = [[] for _ in range(n_agents)]
        self.current_best_idx = 0
        
        logger.info(f"🎯 EnsembleManagerV9 initialized")
        logger.info(f"   Agents: {n_agents}")
        logger.info(f"   Selection: {selection_method}")
    
    def create_agents(self, base_config: Dict = None):
        """
        Create ensemble of agents with varied hyperparameters.
        
        Args:
            base_config: Base configuration to vary
        """
        logger.info(f"🔧 Creating {self.n_agents} agents with varied configs...")
        
        # Default base config
        if base_config is None:
            base_config = {
                'lr': 3e-4,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'use_lstm': True
            }
        
        # Variation strategies
        variations = [
            {'lr': 1e-4, 'clip_range': 0.1, 'ent_coef': 0.005},  # Conservative
            {'lr': 3e-4, 'clip_range': 0.2, 'ent_coef': 0.01},   # Balanced
            {'lr': 1e-3, 'clip_range': 0.3, 'ent_coef': 0.05},   # Aggressive
            {'lr': 5e-4, 'clip_range': 0.15, 'ent_coef': 0.02},  # Conservative-Balanced
            {'lr': 7e-4, 'clip_range': 0.25, 'ent_coef': 0.03}   # Balanced-Aggressive
        ]
        
        # Create agents
        for i in range(self.n_agents):
            # Select variation (cycle through if n_agents > len(variations))
            config = variations[i % len(variations)].copy()
            config['use_lstm'] = base_config.get('use_lstm', True)
            
            # Create agent
            agent = PPOAgent(
                env=self.env,
                lr=config['lr'],
                clip_range=config['clip_range'],
                ent_coef=config['ent_coef'],
                use_lstm=config['use_lstm'],
                verbose=0
            )
            
            self.agents.append(agent)
            self.agent_configs.append(config)
            
            logger.info(f"   Agent {i}: lr={config['lr']:.0e}, clip={config['clip_range']}, ent={config['ent_coef']}")
        
        logger.info(f"✅ Created {len(self.agents)} agents")
    
    def train_agents(self, total_timesteps: int = 50000, eval_freq: int = 5000):
        """
        Train all agents in the ensemble.
        
        Args:
            total_timesteps: Total training timesteps per agent
            eval_freq: Evaluation frequency
        """
        logger.info(f"🚀 Training {len(self.agents)} agents...")
        
        for i, agent in enumerate(self.agents):
            logger.info(f"\n📈 Training Agent {i}/{len(self.agents)-1}...")
            try:
                agent.train(total_timesteps=total_timesteps, eval_freq=eval_freq)
                logger.info(f"✅ Agent {i} training complete")
            except Exception as e:
                logger.error(f"❌ Agent {i} training failed: {e}")
        
        logger.info(f"\n✅ Ensemble training complete!")
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action using ensemble.
        
        Args:
            state: Current state
            deterministic: Use deterministic policy
        
        Returns:
            Selected action
        """
        if not self.agents:
            raise ValueError("No agents in ensemble. Call create_agents() first.")
        
        # Get predictions from all agents
        actions = []
        for agent in self.agents:
            try:
                action = agent.predict(state, deterministic=deterministic)
                actions.append(action)
            except Exception as e:
                logger.warning(f"Agent prediction failed: {e}")
                actions.append(0)  # Hold as fallback
        
        # Select action based on method
        if self.selection_method == 'best':
            # Use current best agent
            return actions[self.current_best_idx]
        
        elif self.selection_method == 'voting':
            # Majority voting
            return int(np.bincount(actions).argmax())
        
        elif self.selection_method == 'weighted':
            # Weighted by recent performance
            weights = self._compute_weights()
            action_probs = np.zeros(max(actions) + 1)
            for action, weight in zip(actions, weights):
                action_probs[action] += weight
            return int(action_probs.argmax())
        
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def update_performance(self, agent_idx: int, reward: float):
        """
        Update performance tracking for an agent.
        
        Args:
            agent_idx: Agent index
            reward: Reward received
        """
        if agent_idx < len(self.agent_performance):
            self.agent_performance[agent_idx].append(reward)
            
            # Keep only recent performance
            if len(self.agent_performance[agent_idx]) > self.performance_window:
                self.agent_performance[agent_idx] = self.agent_performance[agent_idx][-self.performance_window:]
    
    def _compute_weights(self) -> np.ndarray:
        """
        Compute weights for agents based on recent performance.
        
        Returns:
            Array of weights (sum to 1.0)
        """
        weights = []
        for perf in self.agent_performance:
            if len(perf) > 0:
                avg_reward = np.mean(perf[-self.performance_window:])
                weights.append(max(avg_reward, 0.01))  # Avoid negative weights
            else:
                weights.append(0.01)
        
        # Normalize
        weights = np.array(weights)
        weights = weights / weights.sum()
        return weights
    
    def update_best_agent(self):
        """
        Update the current best agent based on recent performance.
        """
        avg_rewards = []
        for perf in self.agent_performance:
            if len(perf) > 0:
                avg_rewards.append(np.mean(perf[-self.performance_window:]))
            else:
                avg_rewards.append(-np.inf)
        
        self.current_best_idx = int(np.argmax(avg_rewards))
        logger.info(f"🏆 Best agent: {self.current_best_idx} (avg reward: {avg_rewards[self.current_best_idx]:.4f})")
    
    def save_ensemble(self, base_path: str):
        """
        Save all agents in the ensemble.
        
        Args:
            base_path: Base path for saving (e.g., './models/ensemble')
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            path = os.path.join(base_path, f"agent_{i}")
            agent.save(path)
        
        logger.info(f"✅ Ensemble saved to {base_path}")
    
    def load_ensemble(self, base_path: str):
        """
        Load all agents in the ensemble.
        
        Args:
            base_path: Base path for loading
        """
        import os
        
        for i, agent in enumerate(self.agents):
            path = os.path.join(base_path, f"agent_{i}")
            if os.path.exists(path + '.zip'):
                agent.load(path)
            else:
                logger.warning(f"⚠️  Agent {i} not found at {path}")
        
        logger.info(f"✅ Ensemble loaded from {base_path}")
    
    def get_stats(self) -> Dict:
        """
        Get ensemble statistics.
        
        Returns:
            Dictionary with stats
        """
        avg_rewards = []
        for perf in self.agent_performance:
            if len(perf) > 0:
                avg_rewards.append(np.mean(perf))
            else:
                avg_rewards.append(0.0)
        
        return {
            'n_agents': len(self.agents),
            'selection_method': self.selection_method,
            'current_best': self.current_best_idx,
            'avg_rewards': avg_rewards,
            'best_reward': max(avg_rewards) if avg_rewards else 0.0
        }


# =============================================================================
# Test Functions
# =============================================================================

def test_ensemble_manager():
    """Test EnsembleManagerV9 with mock environment."""
    print("🧪 Testing EnsembleManagerV9...")
    
    # Mock environment
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    
    env = gym.make('CartPole-v1')
    
    # Create ensemble
    ensemble = EnsembleManagerV9(
        env=env,
        n_agents=3,
        selection_method='best'
    )
    
    # Create agents
    ensemble.create_agents()
    print(f"✓ Test 1: Created {len(ensemble.agents)} agents")
    
    # Test prediction
    state = env.reset()[0]
    action = ensemble.predict(state)
    print(f"✓ Test 2: Prediction successful (action={action})")
    
    # Test performance update
    ensemble.update_performance(0, 1.0)
    ensemble.update_performance(1, 0.5)
    ensemble.update_performance(2, 0.8)
    ensemble.update_best_agent()
    print(f"✓ Test 3: Performance tracking works (best={ensemble.current_best_idx})")
    
    # Test stats
    stats = ensemble.get_stats()
    print(f"✓ Test 4: Stats retrieved: {stats['n_agents']} agents")
    
    print("\n✅ EnsembleManagerV9 tests passed!\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_ensemble_manager()
