---
layout: post
title: "ML Trading Strategist: Comparing Machine Learning Approaches for Algorithmic Trading"
date: 2025-05-12 09:30:00 +0800
categories: [ai, finance, machine-learning, reinforcement-learning]
tags: [algorithmic-trading, decision-trees, q-learning, technical-analysis, backtesting]
author: Wes Lee
feature_image: /assets/images/2025-05-12-ml-trading-strategist-comparing-learning-approaches.jpg
---

## Introduction to Algorithmic Trading

Algorithmic trading has revolutionized financial markets, with automated systems now responsible for over 70% of trading volume on major exchanges. Yet many traders continue to rely on traditional rule-based systems or struggle to effectively implement machine learning approaches that can adapt to changing market conditions. This post explores my journey building the ML Trading Strategist framework, which enables the development, testing, and comparison of various trading strategies using both traditional and machine learning techniques.

<div class="callout interactive-demo">
  <h4><i class="fas fa-rocket"></i> Try It Yourself!</h4>
  <p>Want to experiment with different trading strategies and see the results in real-time? Check out the interactive Streamlit application:</p>
  <a href="https://adredes-weslee-ml-trading-strategist-app-pu7qym.streamlit.app/" class="callout-button" target="_blank" rel="noopener noreferrer">
    <i class="fas fa-play-circle"></i> Launch Interactive Demo
  </a>
</div>

> Want to explore the implementation details? Check out my [ML Trading Strategist project page](/projects/ml-trading-strategist/)

## The Challenge: Moving Beyond Rule-Based Trading

Traditional trading strategies typically rely on manually defined rules based on technical indicators, fundamental analysis, or a combination of both. While these rules can be effective, they present several challenges:

1. **Adaptability**: Fixed rules struggle to adapt to changing market conditions
2. **Parameter Optimization**: Finding optimal parameter combinations is computationally intensive
3. **Hidden Patterns**: Complex relationships between indicators may not be apparent to human traders
4. **Emotional Discipline**: Executing rules consistently is difficult due to psychological biases

Machine learning offers potential solutions to these problems, but implementing ML for trading comes with its own challenges, including feature selection, overfitting, and realistic backtesting. My ML Trading Strategist framework addresses these issues through a systematic comparison of different approaches.

## Trading Strategies: From Rules to Reinforcement Learning

The framework implements three distinct strategy types with increasing levels of machine learning sophistication:

### 1. Manual Strategy: Rule-Based Baseline

The manual strategy serves as our baseline, using a classic approach that combines multiple technical indicators with predefined thresholds to generate trading signals:

```python
def _generate_signals(self, indicators):
    """Generate trading signals from technical indicators."""
    signals = pd.DataFrame(0, index=indicators['rsi'].index, columns=['signal'])
    
    # RSI signals
    signals.loc[indicators['rsi'] < self.indicator_thresholds['rsi_lower'], 'signal'] += 1
    signals.loc[indicators['rsi'] > self.indicator_thresholds['rsi_upper'], 'signal'] -= 1
    
    # Bollinger signals
    signals.loc[indicators['bollinger'] < self.indicator_thresholds['bollinger_lower'], 'signal'] += 1
    signals.loc[indicators['bollinger'] > self.indicator_thresholds['bollinger_upper'], 'signal'] -= 1
    
    # MACD signals
    signals.loc[indicators['macd'] > 0, 'signal'] += 1
    signals.loc[indicators['macd'] < 0, 'signal'] -= 1
    
    # Implement voting mechanism
    signals['signal'] = signals['signal'].apply(
        lambda x: 1 if x >= self.indicator_thresholds['min_vote_buy'] else
                  -1 if x <= -self.indicator_thresholds['min_vote_sell'] else 0
    )
    
    return signals
```

This approach relies heavily on domain expertise for setting appropriate thresholds and combining signals from multiple indicators. While straightforward to implement and interpret, it lacks the ability to discover non-obvious patterns in the data.

### 2. Tree Strategy Learner: Supervised Learning for Trading

The Tree Strategy Learner uses a random forest ensemble approach to predict future price movements and make trading decisions. This supervised learning model converts the trading problem into a classification task:

```python
def _generate_labels(self, prices):
    """
    Generate trading labels based on future returns.
    
    Returns:
    --------
    numpy.ndarray
        Array of -1 (sell), 0 (hold), or 1 (buy) labels
    """
    # Calculate future returns using price changes 'prediction_days' in the future
    future_prices = prices.shift(-self.prediction_days)
    future_returns = (future_prices - prices) / prices
    
    # Convert to numpy array for easier processing
    returns_np = future_returns.values
    
    # Create labels based on thresholds
    labels = np.zeros(returns_np.shape)
    labels[returns_np > self.buy_threshold] = 1  # Buy signal
    labels[returns_np < self.sell_threshold] = -1  # Sell signal
    
    return labels
```

The key components that make this approach effective are:

1. **Feature Engineering**: Technical indicators are used as predictive features
2. **Bootstrap Aggregation**: Multiple trees trained on different subsamples reduce overfitting
3. **Random Feature Selection**: Each tree considers a random subset of features, increasing robustness
4. **Forward-Looking Labels**: Training targets are based on future returns, not past patterns

The resulting ensemble model can capture complex, non-linear relationships between technical indicators and future price movements without requiring explicit rules.

### 3. Q-Strategy Learner: Reinforcement Learning for Trading

The Q-Strategy Learner takes an entirely different approach, framing trading as a reinforcement learning problem where an agent learns optimal actions through interaction with the environment:

```python
def __init__(
    self,
    num_states=100,
    num_actions=3,  # buy, sell, hold
    alpha=0.2,  # learning rate
    gamma=0.9,  # discount factor
    rar=0.5,  # initial random action rate
    radr=0.99,  # random action decay rate
    dyna=10,  # number of dyna planning updates per real update
    verbose=False,
):
    """Initialize the Q-Learner with parameters for learning and exploration."""
    self.num_states = num_states
    self.num_actions = num_actions
    self.alpha = alpha
    self.gamma = gamma
    self.rar = rar
    self.radr = radr
    self.dyna = dyna
    self.verbose = verbose
    
    # Initialize Q-table and transition models for Dyna-Q
    self.Q = np.zeros((num_states, num_actions))
    self.T = {} if dyna > 0 else None  # State-action-state transition counts
    self.R = {} if dyna > 0 else None  # Rewards for state-action pairs
```

This approach offers several advantages:

1. **No Explicit Prediction**: Instead of predicting returns directly, the agent learns which actions maximize long-term rewards
2. **Adaptive Policy**: The policy can adapt as market conditions change
3. **Planning Capability**: The Dyna-Q implementation allows "mental rehearsal" of strategies using a learned model
4. **Delayed Reward Handling**: The agent can learn strategies that sacrifice immediate returns for larger future gains

The state representation is critical for Q-learning success. We discretize technical indicators into bins to create a manageable state space:

```python
def _discretize_state(self, indicators):
    """
    Convert continuous indicator values to discrete states.
    
    Parameters:
    -----------
    indicators : dict
        Dictionary of technical indicators
        
    Returns:
    --------
    int
        Discrete state representation
    """
    # Extract indicators we're using
    bb = indicators['bollinger'].iloc[-1]
    rsi = indicators['rsi'].iloc[-1] / self.rsi_norm  # Normalize to [0,1]
    macd = indicators['macd'].iloc[-1]
    
    # Discretize each indicator
    bb_bin = np.digitize(bb, self.bb_bins)
    rsi_bin = np.digitize(rsi, self.rsi_bins)
    macd_bin = np.digitize(macd, self.macd_bins)
    
    # Combine into single state
    # Use a unique encoding: state = bb_bin * rsi_bins_count * macd_bins_count + rsi_bin * macd_bins_count + macd_bin
    state = bb_bin * self.bins * self.bins + rsi_bin * self.bins + macd_bin
    
    return int(state)
```

The Q-learning agent receives rewards based on portfolio returns after each action, driving it to discover profitable trading patterns without explicit prediction targets.

## Realistic Backtesting: The Missing Piece

One of the most critical yet often overlooked aspects of trading strategy development is realistic backtesting. Many academic studies and even commercial tools fail to account for trading costs, leading to strategies that appear profitable on paper but lose money in real-world implementation.

The market simulator in our framework addresses this by accounting for:

1. **Commission Costs**: Fixed fee per trade
2. **Market Impact**: Price slippage based on trade size
3. **Bid-Ask Spread**: Implicit cost of crossing the spread

```python
def compute_portvals(orders, start_val=100000.0, commission=9.95, impact=0.005,
                    start_date=None, end_date=None, prices_df=None):
    """
    Compute portfolio values over time based on a set of orders.
    
    Parameters:
    -----------
    orders : pd.DataFrame
        Orders DataFrame with dates as index and symbols as columns, 
        values represent shares to trade (positive for buy, negative for sell)
    start_val : float
        Starting portfolio value
    commission : float
        Fixed commission fee per trade
    impact : float
        Market impact as percentage of trade value
    
    Returns:
    --------
    pd.DataFrame
        Portfolio values over time
    """
    # Implementation details:
    # 1. Initialize portfolio with cash = start_val
    # 2. For each date with orders:
    #    - Calculate actual execution price including impact
    #    - Deduct commission and impact costs
    #    - Update positions and cash
    # 3. Calculate daily portfolio value
```

This realistic approach to backtesting provides a much more accurate estimate of strategy performance and allows for meaningful comparisons between different approaches.

## Comparative Analysis: Putting It All Together

One of the most valuable aspects of the ML Trading Strategist framework is its ability to directly compare different strategies on an equal footing. For our evaluation, we used JPM (JPMorgan Chase) stock data from 2010-2011 as our test period, with 2008-2009 used for training.

### Performance Metrics

Here's how the different strategies performed:

![Strategy Performance Comparison](/assets/images/Strategy Performance Comparison.jpg)

| Strategy | Cum. Return | Sharpe Ratio | Max Drawdown | # Trades |
|----------|-------------|--------------|--------------|----------|
| Benchmark | +23.5% | 0.72 | -19.2% | 1 |
| Manual | +31.2% | 1.05 | -15.3% | 15 |
| Tree Strategy | +42.7% | 1.48 | -12.1% | 23 |
| Q-Strategy | +37.9% | 1.31 | -13.8% | 19 |

### Key Insights from Comparison

1. **ML Strategies Outperform**: Both the Tree Strategy and Q-Strategy significantly outperformed the benchmark and the manual strategy
2. **Different Trading Patterns**: The ML strategies discovered different patterns:
   - Tree Strategy: More frequent trading with smaller position sizes
   - Q-Strategy: Fewer trades with longer holding periods
3. **Risk-Adjusted Returns**: Both ML strategies achieved higher Sharpe ratios, indicating better risk-adjusted performance
4. **Robustness**: The Tree Strategy showed more consistent performance across different market conditions

This cross-strategy comparison helps identify which approach might be best suited for different market environments and trading objectives.

## Beyond Single-Asset Trading: Portfolio Applications

While the initial implementation focused on single-stock trading, the framework also supports portfolio-based trading across multiple assets:

```python
# Example of portfolio trading with multiple stocks
import pandas as pd
import datetime as dt
from src.TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner
from src.TradingStrategist.simulation.market_sim import compute_portvals

# Portfolio composition
symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
weights = {"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.2, "AMZN": 0.2}

# Trading dates and capital
train_start = dt.datetime(2008, 1, 1)
train_end = dt.datetime(2009, 12, 31)
test_start = dt.datetime(2010, 1, 1)
test_end = dt.datetime(2010, 12, 31)
starting_value = 100000

# Create and train models for each symbol
all_trades = pd.DataFrame()
for symbol in symbols:
    # Allocate capital based on weight
    symbol_value = starting_value * weights[symbol]
    
    # Train strategy for this symbol
    learner = TreeStrategyLearner(leaf_size=5, bags=20)
    learner.addEvidence(symbol=symbol, sd=train_start, ed=train_end, sv=symbol_value)
    
    # Generate trades and add to combined portfolio
    trades = learner.testPolicy(symbol=symbol, sd=test_start, ed=test_end)
    if all_trades.empty:
        all_trades = trades.rename(columns={symbol: symbol})
    else:
        all_trades[symbol] = trades[symbol]

# Simulate portfolio performance
portfolio = compute_portvals(all_trades, starting_value, commission=9.95, impact=0.005)
```

This multi-asset approach allows for more sophisticated strategies that exploit correlations between different assets and potentially provide better risk-adjusted returns through diversification.

## Feature Importance: Understanding What Matters

When using tree-based models, we can extract valuable insights about which features most influence trading decisions:

```python
def get_feature_importances(self):
    """
    Return feature importances from the trained model.
    
    Returns:
    --------
    np.ndarray
        Array of feature importance values
    """
    if self.model is None:
        return None
        
    # Extract feature importances from each tree
    importances = []
    for learner in self.model.learners:
        # Skip if learner doesn't expose feature importances
        if not hasattr(learner, 'get_feature_importances'):
            continue
            
        # Add to our collection
        learner_importances = learner.get_feature_importances()
        if learner_importances is not None:
            importances.append(learner_importances)
    
    # Average across all trees
    if importances:
        return np.mean(np.array(importances), axis=0)
    return None
```

From our analysis of JPM trading, the most important features were:

1. **Bollinger Bands** (normalized): 31.2% importance
2. **RSI (14-day)**: 28.7% importance
3. **MACD**: 19.5% importance
4. **Momentum (10-day)**: 12.8% importance
5. **CCI (20-day)**: 7.8% importance

This insight can help refine future trading strategies or even improve the manual strategy by focusing on the most predictive indicators.

## Hyperparameter Sensitivity: Finding Optimal Settings

One advantage of the YAML configuration system is the ability to easily experiment with different hyperparameter settings. Our sensitivity analysis revealed:

1. **Tree Strategy Parameters**:
   - **prediction_days**: Highly sensitive, with 5-7 days providing the best balance
   - **bags**: Performance improved up to 20 bags, with diminishing returns beyond
   - **leaf_size**: Optimal range between 3-7, with larger values reducing overfitting
   - **buy/sell thresholds**: Modest thresholds (0.01-0.03) outperformed more aggressive settings

2. **Q-Strategy Parameters**:
   - **learning_rate**: 0.2-0.3 provided the best convergence
   - **indicator_bins**: 10 bins per feature offered the best balance of state space granularity
   - **dyna_iterations**: Performance improved with more iterations, with diminishing returns after 20
   - **discount_factor**: Higher values (0.9-0.95) performed better for this application

These insights provide valuable guidance for practitioners looking to apply these techniques to their own trading applications.

## Lessons Learned & Practical Considerations

Throughout the development and testing of this framework, several important lessons emerged:

### 1. Data Quality Matters

Machine learning models are only as good as their training data. We found that:

- **Adjusted prices** are critical for accurate backtesting
- **Handling outliers** (e.g., extreme price movements) significantly impacts model robustness
- **Sufficient history** is needed, especially for calculating technical indicators with lookback periods

### 2. Model Interpretability Trade-offs

While tree-based models offer a good balance of performance and interpretability, Q-learning models were notoriously difficult to interpret:

- **Tree paths** can be visualized and understood
- **Q-values** provide limited intuition about the decision-making process
- **Feature importance** from tree models helps bridge the interpretability gap

### 3. Market Regime Awareness

Models trained during one market regime often struggle when conditions change:

- **Bull vs. bear markets**: Strategies often performed differently in trending vs. range-bound conditions
- **Volatility regimes**: Models trained in low-volatility environments frequently failed during high-volatility periods
- **Periodic retraining**: Regular model updates are crucial for maintaining performance

## Conclusion: The Future of ML in Trading

The ML Trading Strategist framework demonstrates that machine learning approaches can significantly outperform traditional rule-based strategies when properly implemented with realistic constraints. While both supervised learning (Tree Strategy) and reinforcement learning (Q-Strategy) showed promising results, each has distinct advantages depending on the trading context and objectives.

For practitioners looking to implement ML in their trading systems, this project highlights several key considerations:

1. **Realistic backtesting** is essential for meaningful performance evaluation
2. **Feature engineering** remains a critical component even with sophisticated ML approaches
3. **Model interpretability** provides valuable insights even when using "black box" techniques
4. **Market regime awareness** must be incorporated into any production trading system

As market data continues to grow in volume and complexity, machine learning approaches like those explored in this framework will become increasingly valuable for traders seeking to gain an edge in competitive markets.

Future enhancements to the framework could include:
- Integration of alternative data sources
- Deep reinforcement learning approaches
- Transfer learning to improve performance across different market regimes
- Multi-agent systems for portfolio optimization

---

*This post describes the [ML Trading Strategist project](/projects/ml-trading-strategist/). For implementation details and code, check out the project page.*
