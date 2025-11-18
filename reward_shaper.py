#!/usr/bin/env python3
"""
================================================================================
REWARD SHAPER - V8 PPO Enhancement
================================================================================
RewardShaper, trading bot'un reward fonksiyonuna penalty'ler ekleyerek
risk yönetimini güçlendirir. News blackout, volatility guards, correlation
violations gibi durumları tespit eder ve cezalandırır.

Kullanım:
    from reward_shaper import RewardShaper
    shaper = RewardShaper(blackout, guards, correlation, market, logger)
    penalty = shaper.compute_penalty(state, action, context)
    
Author: E1 AI Agent + Grok Integration
Date: January 2025
Version: 8.0
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


class RewardShaper:
    """
    Reward shaping için penalty hesaplama sınıfı.
    
    Attributes:
        blackout: NewsBlackout instance (haber filtresi)
        guards: VolatilityGuards instance (volatilite koruması)
        correlation: CorrelationControl instance (korelasyon kontrolü)
        market: MarketState or data provider (ATR, spread bilgisi)
        logger: EnhancedTradeLogger instance (loglama)
    """
    
    def __init__(self, blackout, guards, correlation, market, logger):
        """
        Initialize RewardShaper.
        
        Args:
            blackout: News blackout manager
            guards: Volatility guards manager
            correlation: Correlation control manager
            market: Market state provider (ATR, spread, etc.)
            logger: Trade logger for penalty tracking
        """
        self.blackout = blackout
        self.guards = guards
        self.correlation = correlation
        self.market = market
        self.logger = logger
        
    def compute_penalty(self, state, action, context: Dict[str, Any]) -> float:
        """
        Compute penalty based on trading context and violations.
        
        Args:
            state: Current market state
            action: Proposed trading action
            context: Dictionary containing:
                - timestamp: Current time
                - open_positions: List of open positions
                - symbol: Trading symbol (optional)
                
        Returns:
            float: Penalty value (negative number)
        """
        timestamp = context.get('timestamp', pd.Timestamp.now())
        open_positions = context.get('open_positions', [])
        symbol = context.get('symbol', 'EURUSD')
        
        # Try to get market metrics, fallback to safe defaults
        try:
            atr = self._get_atr(timestamp, symbol)
            avg_atr = self._get_average_atr(symbol)
            slippage_est = self._estimate_slippage(timestamp, symbol)
        except Exception as e:
            # Safe fallback values
            atr = 0.001
            avg_atr = 0.001
            slippage_est = 0.0002
            self.logger.log_error(f"MarketState error in RewardShaper: {e}")
        
        penalty = 0.0
        breakdown = {}
        
        # 1. News Blackout Penalty
        if self._check_blackout(timestamp):
            val = -0.5 * atr
            penalty += val
            breakdown['blackout_penalty'] = val
        
        # 2. Volatility Guards Penalty
        try:
            guard_violations = self._check_guards(timestamp, symbol)
            for guard_name, violated in guard_violations.items():
                if violated:
                    val = -0.3 * slippage_est
                    penalty += val
                    breakdown[f'{guard_name}_penalty'] = val
        except Exception as e:
            self.logger.log_error(f"Guards check error: {e}")
        
        # 3. Correlation Violation Penalty
        if self._check_correlation(open_positions):
            total_size = sum(abs(pos.get('size', 0)) for pos in open_positions)
            val = -0.4 * total_size
            penalty += val
            breakdown['correlation_penalty'] = val
        
        # 4. Risk Scaling (adjust penalty based on market volatility)
        risk_scale = self._compute_risk_scale(atr, avg_atr)
        penalty *= risk_scale
        breakdown['risk_scale'] = risk_scale
        
        # 5. Cap penalty to avoid overwhelming base reward
        base_reward_estimate = 0.001
        penalty = max(penalty, -base_reward_estimate * 0.5)
        
        # Log penalty breakdown
        self._log_penalty(timestamp, penalty, breakdown)
        
        return penalty
    
    def _get_atr(self, timestamp, symbol: str) -> float:
        """Get ATR value at timestamp."""
        try:
            if hasattr(self.market, 'get_atr'):
                return self.market.get_atr(timestamp, symbol)
            elif hasattr(self.market, 'atr_series'):
                return self.market.atr_series.get(symbol, {}).get(timestamp, 0.001)
            else:
                return 0.001
        except:
            return 0.001
    
    def _get_average_atr(self, symbol: str) -> float:
        """Get average ATR value."""
        try:
            if hasattr(self.market, 'get_average_atr'):
                return self.market.get_average_atr(symbol)
            else:
                return 0.001
        except:
            return 0.001
    
    def _estimate_slippage(self, timestamp, symbol: str) -> float:
        """Estimate slippage at timestamp."""
        try:
            if hasattr(self.market, 'estimate_slippage'):
                return self.market.estimate_slippage(timestamp, symbol)
            elif hasattr(self.market, 'current_spread'):
                spread = self.market.current_spread(timestamp, symbol)
                return spread * 1.2  # 20% extra for slippage
            else:
                return 0.0002  # Default 2 pips
        except:
            return 0.0002
    
    def _check_blackout(self, timestamp) -> bool:
        """Check if in news blackout period."""
        try:
            if hasattr(self.blackout, 'is_active'):
                return self.blackout.is_active(timestamp)
            elif hasattr(self.blackout, 'is_blackout'):
                return self.blackout.is_blackout(timestamp)
            else:
                return False
        except:
            return False
    
    def _check_guards(self, timestamp, symbol: str) -> Dict[str, bool]:
        """Check volatility guards violations."""
        try:
            if hasattr(self.guards, 'check_all'):
                return self.guards.check_all(timestamp, symbol)
            elif hasattr(self.guards, 'check'):
                return {'volatility': self.guards.check(timestamp, symbol)}
            else:
                return {}
        except:
            return {}
    
    def _check_correlation(self, open_positions: List[Dict]) -> bool:
        """Check correlation violations."""
        try:
            if hasattr(self.correlation, 'is_violation'):
                return self.correlation.is_violation(open_positions)
            elif hasattr(self.correlation, 'check'):
                return self.correlation.check(open_positions)
            else:
                return False
        except:
            return False
    
    def _compute_risk_scale(self, atr: float, avg_atr: float) -> float:
        """Compute risk scaling factor based on ATR."""
        try:
            if avg_atr > 0:
                scale = atr / avg_atr
                return min(max(scale, 0.8), 1.5)  # Clamp between 0.8 and 1.5
            else:
                return 1.0
        except:
            return 1.0
    
    def _log_penalty(self, timestamp, total_penalty: float, breakdown: Dict[str, float]):
        """Log penalty breakdown for analysis."""
        try:
            if hasattr(self.logger, 'log_penalty_breakdown'):
                self.logger.log_penalty_breakdown(timestamp, total_penalty, breakdown)
            else:
                # Fallback to console
                msg = f"⚠️  Penalty @ {timestamp}: {total_penalty:.4f}\n"
                for k, v in breakdown.items():
                    msg += f"   • {k}: {v:.4f}\n"
                print(msg)
        except Exception as e:
            print(f"Error logging penalty: {e}")


# =============================================================================
# Test Functions
# =============================================================================

def test_reward_shaper():
    """Test RewardShaper with mock objects."""
    print("🧪 Testing RewardShaper...")
    
    # Mock objects
    class MockBlackout:
        def is_active(self, timestamp):
            return True  # Always in blackout for test
    
    class MockGuards:
        def check_all(self, timestamp, symbol):
            return {'high_volatility': True, 'low_liquidity': False}
    
    class MockCorrelation:
        def is_violation(self, positions):
            return len(positions) > 2  # Violation if > 2 positions
    
    class MockMarket:
        def get_atr(self, timestamp, symbol):
            return 0.0015
        def get_average_atr(self, symbol):
            return 0.0012
        def estimate_slippage(self, timestamp, symbol):
            return 0.0002
    
    class MockLogger:
        def log_error(self, msg):
            print(f"ERROR: {msg}")
        def log_penalty_breakdown(self, timestamp, penalty, breakdown):
            print(f"PENALTY: {penalty:.4f} - {breakdown}")
    
    # Create RewardShaper
    shaper = RewardShaper(
        blackout=MockBlackout(),
        guards=MockGuards(),
        correlation=MockCorrelation(),
        market=MockMarket(),
        logger=MockLogger()
    )
    
    # Test 1: With violations
    context = {
        'timestamp': pd.Timestamp.now(),
        'open_positions': [{'size': 0.1}, {'size': 0.05}],
        'symbol': 'EURUSD'
    }
    
    penalty = shaper.compute_penalty(None, None, context)
    print(f"✓ Test 1 passed: penalty = {penalty:.6f}")
    assert penalty < 0, "Penalty should be negative"
    
    # Test 2: No violations
    context2 = {
        'timestamp': pd.Timestamp.now(),
        'open_positions': [],
        'symbol': 'GBPUSD'
    }
    
    penalty2 = shaper.compute_penalty(None, None, context2)
    print(f"✓ Test 2 passed: penalty = {penalty2:.6f}")
    
    print("✅ RewardShaper tests passed!\n")


if __name__ == "__main__":
    test_reward_shaper()
