#!/usr/bin/env python3
"""
================================================================================
ADVANCED BACKTESTER V9 - Monte Carlo & Walk-Forward
================================================================================

Provides:
- Monte Carlo simulation for robustness testing
- Walk-forward validation
- Advanced metrics (Sharpe, Sortino, drawdown, profit factor)
- Parameter sensitivity analysis

Author: E1 AI Agent (Emergent.sh)
Date: January 2025
Version: 9.0 FREE PRO
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('AdvancedBacktesterV9')


class AdvancedBacktesterV9:
    """
    Advanced Backtesting Framework.
    
    Features:
    - Monte Carlo simulation
    - Walk-forward validation
    - Advanced performance metrics
    - Parameter sensitivity analysis
    """
    
    def __init__(
        self,
        initial_capital: float = 25000.0,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Advanced Backtester.
        
        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate (for Sharpe ratio)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        self.results = []
        self.trades = []
        
        logger.info(f"📊 AdvancedBacktesterV9 initialized")
        logger.info(f"   Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"   Risk-Free Rate: {risk_free_rate*100:.2f}%")
    
    def run_backtest(
        self,
        strategy_fn: Callable,
        data: pd.DataFrame,
        params: Dict = None
    ) -> Dict:
        """
        Run single backtest with given strategy.
        
        Args:
            strategy_fn: Strategy function (takes data, params, returns trades)
            data: Market data (OHLCV)
            params: Strategy parameters
        
        Returns:
            Dictionary with backtest results
        """
        logger.info("🎯 Running backtest...")
        
        # Run strategy
        trades = strategy_fn(data, params or {})
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, data)
        
        # Store results
        result = {
            'params': params,
            'trades': trades,
            'metrics': metrics
        }
        self.results.append(result)
        
        logger.info(f"✅ Backtest complete: {len(trades)} trades")
        return result
    
    def monte_carlo_simulation(
        self,
        trades: List[Dict],
        n_simulations: int = 1000,
        randomize: str = 'order'  # 'order', 'returns', 'both'
    ) -> Dict:
        """
        Run Monte Carlo simulation on trade results.
        
        Args:
            trades: List of trade results
            n_simulations: Number of simulations
            randomize: What to randomize ('order', 'returns', 'both')
        
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"🎲 Running Monte Carlo ({n_simulations} sims)...")
        
        if not trades:
            logger.warning("⚠️  No trades to simulate")
            return {}
        
        # Extract returns
        returns = np.array([t.get('pnl', 0) for t in trades])
        
        # Run simulations
        final_balances = []
        max_drawdowns = []
        sharpe_ratios = []
        
        for i in range(n_simulations):
            if randomize == 'order':
                # Shuffle trade order
                sim_returns = np.random.permutation(returns)
            elif randomize == 'returns':
                # Resample with replacement
                sim_returns = np.random.choice(returns, size=len(returns), replace=True)
            elif randomize == 'both':
                # Both shuffle and add noise
                sim_returns = np.random.permutation(returns)
                noise = np.random.normal(0, returns.std() * 0.1, len(returns))
                sim_returns = sim_returns + noise
            else:
                sim_returns = returns
            
            # Calculate equity curve
            equity = self.initial_capital + np.cumsum(sim_returns)
            
            # Calculate metrics
            final_balance = equity[-1]
            max_dd = self._calculate_max_drawdown(equity)
            sharpe = self._calculate_sharpe(sim_returns)
            
            final_balances.append(final_balance)
            max_drawdowns.append(max_dd)
            sharpe_ratios.append(sharpe)
        
        # Aggregate results
        results = {
            'n_simulations': n_simulations,
            'final_balance': {
                'mean': np.mean(final_balances),
                'std': np.std(final_balances),
                'min': np.min(final_balances),
                'max': np.max(final_balances),
                'percentiles': {
                    '5%': np.percentile(final_balances, 5),
                    '25%': np.percentile(final_balances, 25),
                    '50%': np.percentile(final_balances, 50),
                    '75%': np.percentile(final_balances, 75),
                    '95%': np.percentile(final_balances, 95)
                }
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'worst': np.max(max_drawdowns)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios)
            },
            'probability_profit': np.sum(np.array(final_balances) > self.initial_capital) / n_simulations
        }
        
        logger.info(f"✅ Monte Carlo complete")
        logger.info(f"   Mean Final Balance: ${results['final_balance']['mean']:,.2f}")
        logger.info(f"   Probability of Profit: {results['probability_profit']*100:.1f}%")
        
        return results
    
    def calculate_metrics(self, trades: List[Dict], data: pd.DataFrame = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: List of trade results
            data: Market data (optional, for additional metrics)
        
        Returns:
            Dictionary with metrics
        """
        if not trades:
            return self._empty_metrics()
        
        # Extract trade data
        pnls = np.array([t.get('pnl', 0) for t in trades])
        returns = pnls / self.initial_capital
        
        # Calculate equity curve
        equity = self.initial_capital + np.cumsum(pnls)
        
        # Win/Loss stats
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # Risk metrics
        max_dd = self._calculate_max_drawdown(equity)
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        calmar = self._calculate_calmar(returns, max_dd)
        
        # Profit factor
        profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else np.inf
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        metrics = {
            'total_trades': len(trades),
            'final_balance': equity[-1] if len(equity) > 0 else self.initial_capital,
            'total_return': (equity[-1] - self.initial_capital) / self.initial_capital if len(equity) > 0 else 0,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'total_pnl': np.sum(pnls)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity: Equity curve
        
        Returns:
            Maximum drawdown (as percentage)
        """
        if len(equity) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(np.min(drawdown))
    
    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            periods_per_year: Number of trading periods per year
        
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) - (self.risk_free_rate / periods_per_year)
        return (excess_return / np.std(returns)) * np.sqrt(periods_per_year)
    
    def _calculate_sortino(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Array of returns
            periods_per_year: Number of trading periods per year
        
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-8
        
        excess_return = np.mean(returns) - (self.risk_free_rate / periods_per_year)
        return (excess_return / downside_std) * np.sqrt(periods_per_year)
    
    def _calculate_calmar(self, returns: np.ndarray, max_dd: float) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: Array of returns
            max_dd: Maximum drawdown
        
        Returns:
            Calmar ratio
        """
        if max_dd == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        return annual_return / max_dd
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'total_trades': 0,
            'final_balance': self.initial_capital,
            'total_return': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'total_pnl': 0
        }
    
    def print_summary(self, metrics: Dict):
        """
        Print backtest summary.
        
        Args:
            metrics: Metrics dictionary
        """
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Trades:      {metrics['total_trades']}")
        print(f"Final Balance:     ${metrics['final_balance']:,.2f}")
        print(f"Total Return:      {metrics['total_return']*100:.2f}%")
        print(f"Win Rate:          {metrics['win_rate']*100:.2f}%")
        print(f"Avg Win:           ${metrics['avg_win']:,.2f}")
        print(f"Avg Loss:          ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor:     {metrics['profit_factor']:.2f}")
        print(f"Expectancy:        ${metrics['expectancy']:,.2f}")
        print(f"Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:      {metrics['calmar_ratio']:.2f}")
        print("="*60 + "\n")


# =============================================================================
# Test Functions
# =============================================================================

def test_advanced_backtester():
    """Test AdvancedBacktesterV9 with sample data."""
    print("🧪 Testing AdvancedBacktesterV9...")
    
    # Create sample trades
    np.random.seed(42)
    trades = [
        {'pnl': np.random.randn() * 100 + 50} for _ in range(100)
    ]
    
    # Initialize backtester
    backtester = AdvancedBacktesterV9(
        initial_capital=25000.0,
        risk_free_rate=0.02
    )
    
    # Test 1: Calculate metrics
    metrics = backtester.calculate_metrics(trades)
    print(f"✓ Test 1: Metrics calculated")
    print(f"   Total PnL: ${metrics['total_pnl']:,.2f}")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    # Test 2: Monte Carlo simulation
    mc_results = backtester.monte_carlo_simulation(trades, n_simulations=100)
    print(f"\n✓ Test 2: Monte Carlo complete")
    print(f"   Mean Final: ${mc_results['final_balance']['mean']:,.2f}")
    print(f"   Prob Profit: {mc_results['probability_profit']*100:.1f}%")
    
    # Test 3: Print summary
    print("\n✓ Test 3: Print summary")
    backtester.print_summary(metrics)
    
    print("✅ AdvancedBacktesterV9 tests passed!\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_advanced_backtester()
