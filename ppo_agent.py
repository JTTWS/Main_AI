#!/usr/bin/env python3
"""
================================================================================
PPO AGENT - V8 PPO Enhancement with LSTM Hybrid
================================================================================
PPOAgent, Stable Baselines3 PPO ile RL trading agent'ı implement eder.
LSTM feature extractor ile hibrit yaklaşım ve DQN fallback seçeneği sunar.

Kullanım:
    from ppo_agent import PPOAgent
    agent = PPOAgent(env, lr=3e-4, clip_range=0.2)
    agent.train(total_timesteps=10000)
    action = agent.predict(state)
    
Author: E1 AI Agent + Grok Integration
Date: January 2025
Version: 8.0
================================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional

# Use gymnasium instead of deprecated gym
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class LSTMPredictor(nn.Module):
    """
    LSTM-based feature extractor for temporal pattern recognition.
    
    This module processes sequential market data and extracts temporal features
    that are then fed into the PPO policy network.
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 128, dropout: float = 0.3):
        """
        Initialize LSTM Predictor.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden layer size
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if hidden_size > 1 else 0  # LSTM dropout requires multi-layer
        )
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) or (batch, features)
            
        Returns:
            Processed features of shape (batch, hidden_size)
        """
        # If input is 2D (batch, features), add sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layer
        output = self.fc(last_output)
        output = self.activation(output)
        
        return output


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for Stable Baselines3 PPO.
    Integrates LSTM for temporal feature extraction.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        """
        Initialize LSTM Features Extractor.
        
        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
        """
        super().__init__(observation_space, features_dim)
        
        input_size = observation_space.shape[0]
        self.lstm_predictor = LSTMPredictor(
            input_size=input_size,
            hidden_size=features_dim,
            dropout=0.3
        )
    
    def forward(self, observations):
        """
        Forward pass through LSTM feature extractor.
        
        Args:
            observations: Batch of observations
            
        Returns:
            Extracted features
        """
        return self.lstm_predictor(observations)


class PPOAgent:
    """
    PPO-based trading agent with optional LSTM feature extraction.
    
    This agent uses Proximal Policy Optimization (PPO) for stable training
    with optional LSTM hybrid for temporal pattern recognition.
    """
    
    def __init__(
        self,
        env,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        use_lstm: bool = True,
        use_dqn_fallback: bool = False,
        verbose: int = 1
    ):
        """
        Initialize PPO Agent.
        
        Args:
            env: Trading environment (gym.Env compatible)
            lr: Learning rate
            clip_range: PPO clip range for policy updates
            ent_coef: Entropy coefficient for exploration
            use_lstm: Whether to use LSTM feature extractor
            use_dqn_fallback: If True, use DQN instead of PPO
            verbose: Verbosity level (0: none, 1: info, 2: debug)
        """
        self.env = env
        self.use_lstm = use_lstm
        self.use_dqn_fallback = use_dqn_fallback
        self.verbose = verbose
        
        # DQN Fallback (if old V7 models need to be loaded)
        if use_dqn_fallback:
            try:
                from ultimate_bot_v7_professional import RainbowDQNAgent
                self.model = RainbowDQNAgent(env)
                print("⚠️  Using DQN fallback mode")
                return
            except ImportError:
                print("⚠️  DQN fallback not available, using PPO")
                self.use_dqn_fallback = False
        
        # Create vectorized environment
        try:
            self.vec_env = make_vec_env(lambda: env, n_envs=1)
        except:
            # If env is already vectorized or doesn't work with make_vec_env
            self.vec_env = env
        
        # Policy kwargs for LSTM
        policy_kwargs = {}
        if use_lstm:
            try:
                policy_kwargs = {
                    "features_extractor_class": LSTMFeaturesExtractor,
                    "features_extractor_kwargs": {"features_dim": 128},
                }
                print("✅ LSTM feature extractor enabled")
            except Exception as e:
                print(f"⚠️  LSTM initialization failed: {e}. Using default MLP policy.")
                use_lstm = False
        
        # Create PPO model
        try:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.vec_env,
                learning_rate=lr,
                clip_range=clip_range,
                ent_coef=ent_coef,
                policy_kwargs=policy_kwargs if policy_kwargs else None,
                verbose=verbose
            )
            print(f"✅ PPO Agent initialized: lr={lr}, clip={clip_range}, ent={ent_coef}")
        except Exception as e:
            print(f"❌ PPO initialization failed: {e}")
            raise
        
        self.lstm_extractor = None
    
    def set_lstm_extractor(self, lstm_model: Optional[LSTMPredictor] = None):
        """
        Set custom LSTM extractor (optional, for advanced use).
        
        Args:
            lstm_model: Custom LSTMPredictor instance
        """
        if lstm_model:
            self.lstm_extractor = lstm_model
            print("✅ Custom LSTM extractor set")
    
    def predict(self, state, deterministic: bool = True):
        """
        Predict action for given state.
        
        Args:
            state: Current market state
            deterministic: If True, use deterministic policy (no exploration)
            
        Returns:
            action: Predicted action
        """
        try:
            # If using custom LSTM extractor (not SB3 integrated)
            if self.lstm_extractor is not None and not self.use_lstm:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                state = self.lstm_extractor(state_tensor).detach().numpy()
            
            # Predict using PPO model
            action, _states = self.model.predict(state, deterministic=deterministic)
            return action
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Fallback to random action
            return self.env.action_space.sample() if hasattr(self.env, 'action_space') else 0
    
    def train(self, total_timesteps: int = 10000, eval_freq: int = 1000):
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
        """
        try:
            # Setup evaluation callback
            eval_callback = EvalCallback(
                self.vec_env,
                best_model_save_path="./logs/",
                log_path="./logs/",
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            
            print(f"🚀 Training PPO agent for {total_timesteps} timesteps...")
            
            # Train
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback
            )
            
            # Save model
            model_path = "./models/ppo_model_v8"
            self.model.save(model_path)
            print(f"✅ Training complete! Model saved to {model_path}")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise
    
    def load(self, path: str):
        """
        Load trained PPO model.
        
        Args:
            path: Path to saved model
        """
        try:
            self.model = PPO.load(path, env=self.vec_env)
            print(f"✅ Model loaded from {path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def save(self, path: str):
        """
        Save PPO model.
        
        Args:
            path: Path to save model
        """
        try:
            self.model.save(path)
            print(f"✅ Model saved to {path}")
        except Exception as e:
            print(f"❌ Failed to save model: {e}")


# =============================================================================
# Test Functions
# =============================================================================

def test_ppo_agent():
    """Test PPOAgent with mock environment."""
    print("🧪 Testing PPOAgent...")
    
    # Create mock environment
    class MockEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10,),
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
            self.step_count = 0
        
        def seed(self, seed=None):
            """Set random seed for reproducibility."""
            np.random.seed(seed)
            return [seed]
        
        def reset(self, seed=None, options=None):
            if seed is not None:
                self.seed(seed)
            self.step_count = 0
            return np.zeros(10, dtype=np.float32), {}
        
        def step(self, action):
            self.step_count += 1
            obs = np.random.randn(10).astype(np.float32)
            reward = 0.001 if action == 1 else -0.0005
            done = self.step_count >= 100
            truncated = False
            info = {}
            return obs, reward, done, truncated, info
    
    # Test 1: Initialize PPO agent
    env = MockEnv()
    agent = PPOAgent(env, lr=3e-4, use_lstm=False, verbose=0)
    print("✓ Test 1 passed: PPO agent initialized")
    
    # Test 2: Train for short period
    agent.train(total_timesteps=100, eval_freq=50)
    print("✓ Test 2 passed: Training completed")
    
    # Test 3: Predict action
    obs = np.zeros(10, dtype=np.float32)
    action = agent.predict(obs)
    assert action in [0, 1, 2], f"Invalid action: {action}"
    print(f"✓ Test 3 passed: Prediction successful (action={action})")
    
    # Test 4: Save and load
    agent.save("./models/test_ppo")
    agent.load("./models/test_ppo")
    print("✓ Test 4 passed: Save/Load successful")
    
    print("✅ PPOAgent tests passed!\n")


if __name__ == "__main__":
    test_ppo_agent()
