#!/usr/bin/env python3
"""
================================================================================
UTILS ADVANCED - Gelişmiş Yardımcı Fonksiyonlar
================================================================================

Gelişmiş yardımcı fonksiyonlar, sınıflar ve utilities:
- Portfolio optimization (CVXPY)
- Stress testing
- Sentiment analysis
- Economic calendar integration
- COT data processing
- Advanced visualization

Yazar: E1 AI Agent
Version: 6.0

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

# Optional imports
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger('UtilsAdvanced')

# ============================================================================
# PORTFOLIO OPTIMIZATION
# ============================================================================

class PortfolioOptimizer:
    """
    Portfolio Optimization using Modern Portfolio Theory
    
    Features:
    - Mean-Variance Optimization
    - Risk Parity
    - Maximum Sharpe Ratio
    - Minimum Variance
    """
    
    def __init__(self):
        self.logger = logging.getLogger('PortfolioOptimizer')
    
    def optimize_weights(self, returns: pd.DataFrame, method: str = 'max_sharpe',
                         risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Optimize portfolio weights
        
        Args:
            returns: DataFrame of returns for each symbol
            method: 'max_sharpe', 'min_variance', 'risk_parity'
            risk_free_rate: Risk-free rate
        
        Returns:
            Dict of {symbol: weight}
        """
        if not HAS_CVXPY:
            self.logger.warning("CVXPY not available, using equal weights")
            symbols = returns.columns.tolist()
            return {symbol: 1.0/len(symbols) for symbol in symbols}
        
        # Calculate expected returns and covariance
        mu = returns.mean()  # Expected returns
        Sigma = returns.cov()  # Covariance matrix
        
        n_assets = len(mu)
        
        # Define optimization variable
        w = cp.Variable(n_assets)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # No short selling
        ]
        
        try:
            if method == 'max_sharpe':
                # Maximize Sharpe ratio: (mu^T * w - rf) / sqrt(w^T * Sigma * w)
                # Equivalent to: maximize mu^T * w / sqrt(w^T * Sigma * w)
                portfolio_return = mu.values @ w
                portfolio_variance = cp.quad_form(w, Sigma.values)
                
                # Maximize return for given risk or minimize risk for given return
                # Here we minimize variance for return >= target
                objective = cp.Minimize(portfolio_variance)
                constraints.append(portfolio_return >= mu.mean())  # Target return
                
            elif method == 'min_variance':
                # Minimize variance
                portfolio_variance = cp.quad_form(w, Sigma.values)
                objective = cp.Minimize(portfolio_variance)
                
            elif method == 'risk_parity':
                # Risk parity: equal risk contribution
                # Simplified: inverse volatility weighting
                vol = np.sqrt(np.diag(Sigma))
                inv_vol = 1 / vol
                target_weights = inv_vol / inv_vol.sum()
                
                # Minimize distance to target weights
                objective = cp.Minimize(cp.sum_squares(w - target_weights))
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                weights = w.value
                weight_dict = {symbol: float(weights[i]) 
                               for i, symbol in enumerate(returns.columns)}
                
                self.logger.info(f"Optimization successful: {method}")
                return weight_dict
            else:
                self.logger.warning(f"Optimization failed: {problem.status}")
                # Return equal weights
                symbols = returns.columns.tolist()
                return {symbol: 1.0/len(symbols) for symbol in symbols}
                
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            # Return equal weights
            symbols = returns.columns.tolist()
            return {symbol: 1.0/len(symbols) for symbol in symbols}

# ============================================================================
# STRESS TESTING
# ============================================================================

class StressTester:
    """Stress Testing for Portfolio"""
    
    def __init__(self):
        self.logger = logging.getLogger('StressTester')
    
    def run_stress_test(self, returns: pd.DataFrame, n_scenarios: int = 100) -> Dict:
        """
        Run Monte Carlo stress test
        
        Args:
            returns: Historical returns
            n_scenarios: Number of scenarios to simulate
        
        Returns:
            Dict with stress test results
        """
        self.logger.info(f"Running stress test with {n_scenarios} scenarios...")
        
        # Calculate statistics
        mu = returns.mean()
        sigma = returns.std()
        
        # Historical worst-case scenarios
        worst_daily = returns.min(axis=0)
        worst_weekly = returns.rolling(5).sum().min(axis=0)
        worst_monthly = returns.rolling(20).sum().min(axis=0)
        
        # Simulate scenarios
        simulated_returns = []
        
        for _ in range(n_scenarios):
            # Random scenario based on historical distribution
            scenario = np.random.normal(mu, sigma * 2)  # 2x volatility
            simulated_returns.append(scenario.sum())
        
        simulated_returns = np.array(simulated_returns)
        
        # Calculate stress metrics
        var_95 = np.percentile(simulated_returns, 5)
        cvar_95 = simulated_returns[simulated_returns <= var_95].mean()
        
        results = {
            'worst_daily': worst_daily.to_dict(),
            'worst_weekly': worst_weekly.to_dict(),
            'worst_monthly': worst_monthly.to_dict(),
            'simulated_var_95': var_95,
            'simulated_cvar_95': cvar_95,
            'simulated_worst': simulated_returns.min(),
            'simulated_best': simulated_returns.max()
        }
        
        self.logger.info(f"Stress test complete: VaR={var_95:.4f}, CVaR={cvar_95:.4f}")
        
        return results

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

class SentimentAnalyzer:
    """Sentiment Analysis for News and Social Media"""
    
    def __init__(self):
        if HAS_VADER:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
        
        self.logger = logging.getLogger('SentimentAnalyzer')
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment score (-1 to 1)
        """
        if not self.analyzer:
            return 0.0
        
        try:
            scores = self.analyzer.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def analyze_news(self, news_list: List[str]) -> Dict:
        """
        Analyze sentiment of multiple news articles
        
        Args:
            news_list: List of news texts
        
        Returns:
            Dict with sentiment statistics
        """
        if not news_list:
            return {
                'avg_sentiment': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        sentiments = [self.analyze_text(text) for text in news_list]
        
        positive = sum(1 for s in sentiments if s > 0.05)
        negative = sum(1 for s in sentiments if s < -0.05)
        neutral = len(sentiments) - positive - negative
        
        return {
            'avg_sentiment': np.mean(sentiments),
            'positive_ratio': positive / len(sentiments),
            'negative_ratio': negative / len(sentiments),
            'neutral_ratio': neutral / len(sentiments),
            'sentiment_std': np.std(sentiments)
        }

# ============================================================================
# VISUALIZATION
# ============================================================================

class TradingVisualizer:
    """Advanced Trading Visualization"""
    
    def __init__(self):
        self.logger = logging.getLogger('TradingVisualizer')
    
    def plot_equity_curve(self, trades: List[Dict], initial_capital: float = 25000.0) -> Optional[go.Figure]:
        """
        Plot equity curve
        
        Args:
            trades: List of trades with 'timestamp' and 'pnl'
            initial_capital: Initial capital
        
        Returns:
            Plotly figure or None
        """
        if not HAS_PLOTLY:
            self.logger.warning("Plotly not available")
            return None
        
        if not trades:
            return None
        
        # Calculate cumulative P&L
        timestamps = [t['timestamp'] if 'exit_time' not in t else t['exit_time'] for t in trades]
        pnls = [t.get('pnl', 0) for t in trades]
        
        cumulative_pnl = np.cumsum(pnls)
        equity = initial_capital + cumulative_pnl
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig
    
    def plot_drawdown(self, trades: List[Dict], initial_capital: float = 25000.0) -> Optional[go.Figure]:
        """
        Plot drawdown
        
        Args:
            trades: List of trades
            initial_capital: Initial capital
        
        Returns:
            Plotly figure or None
        """
        if not HAS_PLOTLY:
            return None
        
        if not trades:
            return None
        
        # Calculate equity curve
        pnls = [t.get('pnl', 0) for t in trades]
        cumulative_pnl = np.cumsum(pnls)
        equity = initial_capital + cumulative_pnl
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        timestamps = [t['timestamp'] if 'exit_time' not in t else t['exit_time'] for t in trades]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Time',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig

# ============================================================================
# ECONOMIC CALENDAR
# ============================================================================

class EconomicCalendar:
    """Economic Calendar Data Manager"""
    
    def __init__(self, calendar_path: str = None):
        self.calendar_path = calendar_path
        self.calendar_data = None
        self.logger = logging.getLogger('EconomicCalendar')
        
        if calendar_path:
            self.load_calendar(calendar_path)
    
    def load_calendar(self, path: str):
        """Load economic calendar from CSV"""
        try:
            self.calendar_data = pd.read_csv(path)
            
            # Parse datetime
            if 'datetime' in self.calendar_data.columns:
                self.calendar_data['datetime'] = pd.to_datetime(
                    self.calendar_data['datetime'], 
                    utc=True, 
                    errors='coerce'
                )
            elif 'date' in self.calendar_data.columns:
                self.calendar_data['datetime'] = pd.to_datetime(
                    self.calendar_data['date'], 
                    utc=True, 
                    errors='coerce'
                )
            
            self.logger.info(f"Loaded {len(self.calendar_data)} economic events")
            
        except Exception as e:
            self.logger.warning(f"Could not load calendar: {e}")
    
    def get_events(self, start_date: datetime, end_date: datetime, 
                   impact: str = None) -> pd.DataFrame:
        """
        Get events in date range
        
        Args:
            start_date: Start date
            end_date: End date
            impact: Filter by impact ('High', 'Medium', 'Low')
        
        Returns:
            DataFrame of events
        """
        if self.calendar_data is None:
            return pd.DataFrame()
        
        # Filter by date
        mask = (self.calendar_data['datetime'] >= start_date) & \
               (self.calendar_data['datetime'] <= end_date)
        
        events = self.calendar_data[mask]
        
        # Filter by impact
        if impact and 'impact' in events.columns:
            events = events[events['impact'] == impact]
        
        return events
    
    def is_high_impact_event(self, current_time: datetime, window_hours: int = 1) -> bool:
        """
        Check if there's a high-impact event nearby
        
        Args:
            current_time: Current time
            window_hours: Window around event (hours)
        
        Returns:
            True if high-impact event nearby
        """
        if self.calendar_data is None:
            return False
        
        start_time = current_time - timedelta(hours=window_hours)
        end_time = current_time + timedelta(hours=window_hours)
        
        events = self.get_events(start_time, end_time, impact='High')
        
        return len(events) > 0

# ============================================================================
# COT DATA
# ============================================================================

class COTDataManager:
    """Commitments of Traders Data Manager"""
    
    def __init__(self, cot_path: str = None):
        self.cot_path = cot_path
        self.cot_data = None
        self.logger = logging.getLogger('COTDataManager')
        
        if cot_path:
            self.load_cot_data(cot_path)
    
    def load_cot_data(self, path: str):
        """Load COT data from CSV"""
        try:
            self.cot_data = pd.read_csv(path)
            
            # Parse date
            if 'date' in self.cot_data.columns:
                self.cot_data['date'] = pd.to_datetime(
                    self.cot_data['date'], 
                    utc=True, 
                    errors='coerce'
                )
            
            self.logger.info(f"Loaded {len(self.cot_data)} COT reports")
            
        except Exception as e:
            self.logger.warning(f"Could not load COT data: {e}")
    
    def get_net_position(self, symbol: str, date: datetime) -> float:
        """
        Get net position for symbol at date
        
        Args:
            symbol: Trading symbol
            date: Date
        
        Returns:
            Net position ratio
        """
        if self.cot_data is None:
            return 0.0
        
        # Find closest report
        mask = self.cot_data['date'] <= date
        
        if mask.sum() == 0:
            return 0.0
        
        latest = self.cot_data[mask].iloc[-1]
        
        # Calculate net position
        # This is simplified - real implementation would need proper column names
        long_pos = latest.get('long', 0)
        short_pos = latest.get('short', 0)
        
        if long_pos + short_pos > 0:
            net_ratio = (long_pos - short_pos) / (long_pos + short_pos)
        else:
            net_ratio = 0.0
        
        return net_ratio

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    
    if np.std(returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    return sharpe

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility)
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
    
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    return sortino

def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        (max_drawdown, start_index, end_index)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    
    max_dd = np.min(drawdown)
    end_idx = np.argmin(drawdown)
    
    # Find start of drawdown
    start_idx = 0
    for i in range(end_idx, -1, -1):
        if equity_curve[i] == running_max[i]:
            start_idx = i
            break
    
    return abs(max_dd), start_idx, end_idx

def format_performance_report(performance: Dict) -> str:
    """
    Format performance dictionary to readable report
    
    Args:
        performance: Performance dictionary
    
    Returns:
        Formatted string
    """
    report = "\n" + "="*70 + "\n"
    report += "PERFORMANCE REPORT\n"
    report += "="*70 + "\n\n"
    
    report += f"Total Trades: {performance.get('total_trades', 0)}\n"
    report += f"Winning Trades: {performance.get('winning_trades', 0)}\n"
    report += f"Losing Trades: {performance.get('losing_trades', 0)}\n"
    report += f"Win Rate: {performance.get('win_rate', 0):.1%}\n\n"
    
    report += f"Total P&L: ${performance.get('total_pnl', 0):,.2f}\n"
    report += f"Average Win: ${performance.get('avg_win', 0):,.2f}\n"
    report += f"Average Loss: ${performance.get('avg_loss', 0):,.2f}\n\n"
    
    report += f"Final Capital: ${performance.get('final_capital', 0):,.2f}\n"
    report += f"Return: {performance.get('return_pct', 0):.2f}%\n"
    
    report += "\n" + "="*70 + "\n"
    
    return report

# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'PortfolioOptimizer',
    'StressTester',
    'SentimentAnalyzer',
    'TradingVisualizer',
    'EconomicCalendar',
    'COTDataManager',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'format_performance_report'
]
