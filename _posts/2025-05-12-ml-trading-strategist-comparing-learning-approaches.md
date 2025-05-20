---
layout: post
title: "Developing ML Trading Strategies: From Rule-Based Systems to Reinforcement Learning"
date: 2025-05-12 09:30:00 +0800
categories: [ai, finance, machine-learning, reinforcement-learning]
tags: [algorithmic-trading, decision-trees, q-learning, technical-analysis, backtesting, python, data-science]
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


## The Core Challenge: Moving Beyond Fixed Rules

Traditional trading strategies often rely on manually defined rules based on technical indicators or fundamental analysis. While straightforward, these methods face hurdles:

1.  **Adaptability**: Fixed rules struggle with dynamic market conditions.
2.  **Optimization**: Finding optimal parameters is a complex, often manual task.
3.  **Hidden Patterns**: Human traders might miss subtle, complex relationships between indicators.
4.  **Emotional Discipline**: Consistent execution can be compromised by psychological biases.

Machine learning (ML) offers a path to address these, but ML in trading isn't without its own complexities, such as feature selection, overfitting, and the need for truly realistic backtesting. Our ML Trading Strategist framework was built to systematically explore and compare different solutions.

## Strategy Implementation Deep Dive: From Rules to Reinforcement

The framework was designed to implement and compare three distinct strategy types, each with increasing ML sophistication.

### 1. The Baseline: Manual Rule-Based Strategy

We started by implementing a manual strategy as a baseline. This classic approach uses a combination of technical indicators (RSI, Bollinger Bands, MACD) with predefined thresholds to generate trading signals. A voting mechanism consolidates these signals.

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
This method is interpretable but relies heavily on domain expertise for threshold setting and lacks adaptability.

### 2. Supervised Learning: The Tree Strategy Learner

Next, we implemented a Tree Strategy Learner using a random forest ensemble. This converts the trading problem into a classification task: predicting future price movements.

**Feature Engineering**: Technical indicators (RSI, Bollinger Bands, MACD, Momentum, CCI) serve as the input features for the model.

**Label Generation**: Trading labels (buy, sell, hold) are generated based on future returns. If the price is expected to rise above a `buy_threshold` within a certain number of `prediction_days`, it's a buy signal, and vice-versa for sell.

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
    labels[returns_np > self.buy_threshold] = 1  # Buy signal
    labels[returns_np < self.sell_threshold] = -1  # Sell signal
    
    return labels
```

**Key Ensemble Techniques**:
* **Bootstrap Aggregation (Bagging)**: Multiple decision trees are trained on different subsamples of the training data, reducing overfitting.
* **Random Feature Selection**: Each tree considers only a random subset of features, increasing model robustness.

This approach allows the model to capture complex, non-linear relationships without explicit rule definition.

### 3. Reinforcement Learning: The Q-Strategy Learner

The Q-Strategy Learner frames trading as a reinforcement learning problem. An agent learns an optimal trading policy by interacting with the market environment and maximizing long-term rewards.

```python
def __init__(
    self,
    num_states=100,  # Derived from discretized indicator bins
    num_actions=3,  # buy, sell, hold
    alpha=0.2,  # learning rate
    gamma=0.9,  # discount factor
    rar=0.5,  # initial random action rate
    radr=0.99,  # random action decay rate
    dyna=10,  # number of dyna planning updates per real update
    verbose=False,
):
    """Initialize the Q-Learner with parameters for learning and exploration."""
    self.num_states = num_states 
    self.num_actions = num_actions
    self.alpha = alpha
    self.gamma = gamma
    self.rar = rar # Random Action Rate
    self.radr = radr # Random Action Decay Rate
    self.dyna = dyna # Dyna-Q iterations
    self.verbose = verbose
    
    self.Q = np.zeros((num_states, num_actions)) # Q-table
    # For Dyna-Q: T tracks transitions, R tracks rewards
    self.T = {} if dyna > 0 else None 
    self.R = {} if dyna > 0 else None
```

**Advantages of RL**:
* **No Explicit Prediction**: Learns actions to maximize rewards, not direct price prediction.
* **Adaptive Policy**: Can adapt to changing market dynamics.
* **Planning (Dyna-Q)**: Allows "mental rehearsals" using a learned model of the environment, improving sample efficiency.
* **Delayed Reward Handling**: Can learn strategies that optimize for long-term gains.

**State Representation**: A critical aspect is defining the state. We discretized continuous technical indicator values (like Bollinger Bands, RSI, MACD) into bins to create a manageable, discrete state space.

```python
def _discretize_state(self, indicators):
    """Convert continuous indicator values to discrete states."""
    # Extract current indicator values
    bb_val = indicators['bollinger'].iloc[-1]
    rsi_val = indicators['rsi'].iloc[-1] / self.rsi_norm # Normalize RSI
    macd_val = indicators['macd'].iloc[-1]
    
    # Digitize each indicator based on predefined bins
    bb_bin = np.digitize(bb_val, self.bb_bins)
    rsi_bin = np.digitize(rsi_val, self.rsi_bins)
    macd_bin = np.digitize(macd_val, self.macd_bins)
    
    # Combine bin indices into a single unique state identifier
    state = bb_bin * (self.bins_per_indicator ** 2) + \
            rsi_bin * self.bins_per_indicator + \
            macd_bin
    return int(state)
```
The agent receives rewards based on portfolio returns after each action, guiding its learning process.

## Ensuring Realism: The Backtesting Engine

Effective strategy evaluation hinges on realistic backtesting. Many strategies look good on paper but fail when real-world costs are considered. Our market simulator incorporates:

1.  **Commission Costs**: A fixed fee per trade.
2.  **Market Impact (Slippage)**: The effect of a trade on execution price, proportional to trade size.
3.  **Bid-Ask Spread**: Implicit cost of executing trades.

The `compute_portvals` function in our simulator meticulously tracks cash, positions, and deducts these costs to provide an accurate portfolio valuation over time.

## Putting the Strategies to the Test

We evaluated these strategies using JPMorgan Chase (JPM) stock data, training on 2008-2009 and testing on 2010-2011.

Both the Tree Strategy and Q-Strategy significantly outperformed the benchmark and the manual strategy in terms of cumulative return and Sharpe ratio. Interestingly, they exhibited different trading patterns: the Tree Strategy tended towards more frequent trades with smaller positions, while the Q-Strategy favored fewer trades with longer holding periods. 

## Uncovering Insights: Feature Importance and Hyperparameters

### Feature Importance (Tree Strategy)

For the JPM dataset, the Tree Strategy identified the following technical indicators as most influential:
1.  **Bollinger Bands** (normalized): 31.2% importance
2.  **RSI (14-day)**: 28.7% importance
3.  **MACD**: 19.5% importance
4.  **Momentum (10-day)**: 12.8% importance
5.  **CCI (20-day)**: 7.8% importance

This information can be used to refine features for future models or even improve manual strategies.

### Hyperparameter Sensitivity

Through experimentation (facilitated by our YAML configuration system):
* **Tree Strategy**: `prediction_days` (5-7 optimal), `bags` (up to 20), and `leaf_size` (3-7) were key. Modest `buy/sell_thresholds` (0.01-0.03) worked best.
* **Q-Strategy**: `learning_rate` (0.2-0.3), `indicator_bins` (10 per feature), `dyna_iterations` (diminishing returns after 20), and a high `discount_factor` (0.9-0.95) were found to be effective.

## Expanding Capabilities: Portfolio Trading

The framework isn't limited to single-asset trading. It supports portfolio-based strategies across multiple assets, allowing for capital allocation based on predefined weights and training individual models for each asset.

```python
# Conceptual example of portfolio trading setup
# symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
# weights = {"AAPL": 0.3, "MSFT": 0.3, "GOOG": 0.2, "AMZN": 0.2}
# starting_value = 100000
# all_trades = pd.DataFrame()

# for symbol in symbols:
#     symbol_value = starting_value * weights[symbol]
#     learner = TreeStrategyLearner(...) # Configure learner
#     learner.addEvidence(symbol=symbol, sd=train_start, ed=train_end, sv=symbol_value)
#     trades = learner.testPolicy(symbol=symbol, sd=test_start, ed=test_end)
#     # Combine trades for portfolio simulation
#     ...

# portfolio_values = compute_portvals(all_trades, ...)
```
This enables more sophisticated strategies that can leverage inter-asset correlations and diversification.

## Lessons Learned on the Journey

1.  **Data Quality is King**: Adjusted prices are essential. Outlier handling and sufficient historical data (especially for indicators with lookback periods) are critical.
2.  **Interpretability vs. Performance**: Tree-based models offer a good balance. Q-learning models, while powerful, can be harder to interpret directly, though analyzing the learned Q-values can offer some insight. Feature importance from trees helps bridge this.
3.  **Market Regime Awareness**: Models trained in one market regime (e.g., bull market, low volatility) may not perform well when conditions change. Periodic retraining and regime-aware modeling are crucial.

## Conclusion: The Future of ML in Trading

The ML Trading Strategist framework demonstrates that machine learning, when implemented with realistic constraints and rigorous backtesting, can offer significant advantages over traditional rule-based trading. Both supervised learning and reinforcement learning approaches showed promise, each with unique characteristics.

Key takeaways for practitioners:
* Realistic backtesting is non-negotiable.
* Thoughtful feature engineering remains vital.
* Strive for model interpretability where possible.
* Incorporate market regime awareness into live systems.

As financial data grows in complexity, ML will become increasingly indispensable. Future work on this framework could include integrating alternative data, exploring deep reinforcement learning, and applying transfer learning across market regimes.

> For a complete overview of the ML Trading Strategist framework, including system architecture, component details, and deployment instructions, please see the [ML Trading Strategist: Advanced Algorithmic Trading Framework](/projects/ml-trading-strategist/) project page.

---
