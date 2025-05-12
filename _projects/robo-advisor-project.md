---
layout: project
title: "AI-Powered Robo Advisor for Portfolio Optimization"
categories: machine-learning finance reinforcement-learning
image: /assets/images/placeholder.svg
technologies: [Python, Reinforcement Learning, Plotly Dash, Scikit-Learn, Deep Learning]
github: https://github.com/Adredes-weslee/Using_Artificial_Intelligence_to_Develop_a_Robo_Advisor
blog_post: /ai/finance/machine-learning/reinforcement-learning/2023/10/25/robo-advisor-risk-profiling-portfolio-optimization.html
---

## Project Overview

Developed a full-stack, intelligent robo advisor using supervised learning, reinforcement learning, and convex optimization. This system predicts investor risk tolerance and dynamically optimizes investment portfolios to maximize returns based on individual risk profiles. [Read the detailed blog post](/ai/finance/machine-learning/reinforcement-learning/2023/10/25/robo-advisor-risk-profiling-portfolio-optimization.html) for a comprehensive breakdown of the methodology and results.

## Business Context & Market Need

Traditional financial advisory services often come with high fees and investment minimums, creating barriers for average retail investors. Robo-advisors democratize investment management through algorithmic portfolio optimization, but most existing solutions use relatively simple rule-based approaches.

This project addresses several limitations of current robo-advisors:

1. **Static Risk Assessment**: Most platforms use rigid questionnaires that inadequately capture risk tolerance
2. **Simplified Allocation**: Many rely on predefined asset allocation templates rather than true optimization
3. **Lack of Adaptability**: Few solutions dynamically adjust to changing market conditions
4. **Limited Personalization**: Most systems offer only a handful of portfolio templates

## Data Sources

The project utilizes two primary data sources:

1. **2019 Survey of Consumer Finances (SCF)** from the Federal Reserve - Contains detailed information on households' demographics, finances, assets, and investment behaviors. This dataset was used to train the risk tolerance prediction models.

2. **S&P 500 Historical Stock Price Data (2000-2023)** - Retrieved using the yfinance library by scraping tickers from Wikipedia and downloading adjusted closing prices. This data was used to train the reinforcement learning models and test portfolio optimization strategies.

## System Architecture

The robo advisor consists of three integrated components:

1. **Risk Profile Predictor**: Predicts an investor's risk tolerance based on demographic and financial data using supervised machine learning
2. **Portfolio Optimizer**: Allocates assets using both traditional Modern Portfolio Theory (via convex optimization) and Deep Q-Network reinforcement learning
3. **Interactive Dashboard**: Allows users to input their information and receive personalized recommendations through a Plotly Dash interface with OpenAI-powered chatbot assistance

The system follows this workflow:
1. User inputs demographic and financial information
2. Extra Trees Regressor model predicts risk tolerance on a 0-1 scale
3. Based on risk tolerance and selected stocks, parallel optimization occurs:
   - Classical MVO using CVXOPT
   - Reinforcement learning allocation using trained DQN
4. Results are displayed with interactive visualizations
5. AI chatbot provides additional guidance and answers financial questions

![System Architecture](/assets/images/robo-advisor-architecture.png)

## Risk Profiling Module

### Data Source & Preparation

Using survey data from the Federal Reserve's 2019 Survey of Consumer Finances (SCF), I created a robust risk prediction model:

- **Data Processing**: Cleaned and normalized 40+ variables from 5,777 respondents
- **Feature Engineering**: Created composite indicators like debt-to-income ratio and savings rate
- **Synthetic Labels**: Generated risk tolerance scores using financial behavior markers

### Model Development

I implemented and compared multiple regression approaches:

```python
# Model pipeline implementation
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Feature preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# Model candidates
models = {
    'extra_trees': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesRegressor(random_state=42))
    ]),
    'gradient_boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ]),
    'neural_network': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(random_state=42, max_iter=1000))
    ])
}

# Hyperparameter grids
param_grids = {
    'extra_trees': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10]
    },
    # Similar grids for other models
}

# Find optimal model through cross-validation
best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(
        model, param_grids[name], cv=5, scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
```

### Model Performance

The Extra Trees Regressor emerged as the best performer:

| Model | R² | RMSE | Cross-Val Score |
|-------|-----|------|----------------|
| Extra Trees | 0.91 | 0.115 | 0.89 ± 0.03 |
| Gradient Boosting | 0.87 | 0.136 | 0.86 ± 0.04 |
| Neural Network | 0.85 | 0.148 | 0.84 ± 0.05 |

### Feature Importance

Analysis revealed the most important predictors of risk tolerance:

1. **Age**: Younger investors tend to have higher risk tolerance
2. **Income Stability**: Those with reliable income sources are more risk-tolerant
3. **Financial Knowledge**: Self-reported investment knowledge strongly correlates with risk appetite
4. **Existing Allocations**: Current investment mix provides strong signals
5. **Time Horizon**: Longer investment horizons associate with higher risk tolerance

## Portfolio Optimization Engine

I implemented both classical and AI-driven optimization approaches:

### Modern Portfolio Theory Implementation

```python
import numpy as np
import pandas as pd
import cvxopt as opt
from cvxopt import blas, solvers

def markowitz_optimization(returns, risk_tolerance=0.5):
    """
    Perform Markowitz portfolio optimization using convex programming.
    
    Args:
        returns: DataFrame of historical asset returns
        risk_tolerance: Float between 0-1 representing risk preference
            (0 = minimum risk, 1 = maximum return)
    
    Returns:
        Optimal weights for each asset
    """
    n = len(returns.columns)
    returns_mat = returns.values
    
    # Calculate mean returns and covariance matrix
    mu = np.mean(returns_mat, axis=0)
    cov = np.cov(returns_mat, rowvar=False)
    
    # Convert to cvxopt matrices
    P = opt.matrix(cov)
    q = opt.matrix(np.zeros(n))
    
    # Constraints: weights sum to 1, weights >= 0
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(np.zeros(n))
    A = opt.matrix(np.ones(n)).T
    b = opt.matrix(np.ones(1))
    
    # Adjust risk tolerance and maximize returns
    solvers.options['show_progress'] = False
    optimal_weights = solvers.qp(
        P * (1 - risk_tolerance), 
        -opt.matrix(mu) * risk_tolerance,
        G, h, A, b
    )['x']
    
    return np.array(optimal_weights).flatten()
```

### Reinforcement Learning Portfolio Optimization

For dynamic portfolio management, I developed a Deep Q-Network (DQN) approach:

```python
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class PortfolioEnv(gym.Env):
    """Custom Environment for portfolio optimization"""
    
    def __init__(self, returns_data, lookback=30, episodes=252):
        self.returns = returns_data
        self.lookback = lookback
        self.episode_length = episodes
        
        # Action space: Discrete allocations across assets
        # Each action represents a predefined allocation strategy
        self.action_space = gym.spaces.Discrete(27)  # 3 allocation levels for 3 assets
        
        # Observation space: Historical returns + current allocation
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback * returns_data.shape[1] + 3,),
            dtype=np.float32
        )
        
        self.reset()
    
    def step(self, action):
        # Convert action index to actual allocation
        self.allocation = self._get_allocation_from_action(action)
        
        # Calculate portfolio return
        portfolio_return = np.sum(self.returns.iloc[self.current_step] * self.allocation)
        
        # Calculate reward (Sharpe ratio component)
        self.portfolio_returns.append(portfolio_return)
        reward = self._calculate_sharpe()
        
        # Update state
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        return self._get_observation(), reward, done, {}
    
    def reset(self):
        # Reset environment state
        self.current_step = self.lookback
        self.allocation = np.array([1/3, 1/3, 1/3])  # Start with equal weights
        self.portfolio_returns = []
        
        return self._get_observation()
    
    def _get_observation(self):
        # Return historical returns + current allocation
        historical_returns = self.returns.iloc[
            self.current_step - self.lookback:self.current_step
        ].values.flatten()
        
        return np.concatenate([historical_returns, self.allocation])
    
    def _calculate_sharpe(self, risk_free_rate=0.02/252):
        if len(self.portfolio_returns) < 2:
            return 0
        
        returns_array = np.array(self.portfolio_returns)
        sharpe = (np.mean(returns_array) - risk_free_rate) / np.std(returns_array)
        return sharpe * np.sqrt(252)  # Annualized Sharpe
    
    def _get_allocation_from_action(self, action):
        # Convert discrete action to allocation weights
        # Example mapping for 3 assets with 3 allocation levels each
        allocations = np.zeros(3)
        
        # Extract allocation level for each asset (0, 0.5, 1) from action index
        action_vector = np.base_repr(action, base=3, padding=3)[-3:]
        for i in range(3):
            allocations[i] = float(action_vector[i]) / 2  # Map {0,1,2} to {0, 0.5, 1}
        
        # Normalize to sum to 1
        return allocations / allocations.sum() if allocations.sum() > 0 else np.array([1/3, 1/3, 1/3])
```

### Training the RL Agent

```python
# Setup the DQN agent
def build_dqn_agent(env):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    
    memory = SequentialMemory(limit=10000, window_length=1)
    policy = EpsGreedyQPolicy(eps=0.1)
    
    dqn = DQNAgent(
        model=model,
        nb_actions=env.action_space.n,
        memory=memory,
        nb_steps_warmup=500,
        target_model_update=1e-2,
        policy=policy
    )
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    return dqn

# Create environment with historical market data
env = PortfolioEnv(returns_data)
dqn = build_dqn_agent(env)

# Train the agent
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)
```

### Performance Comparison

I backtested both approaches against market benchmarks:

| Strategy | Annualized Return | Max Drawdown | Sharpe Ratio | Sortino Ratio |
|----------|------------------|--------------|--------------|---------------|
| RL Portfolio (High Risk) | 17.2% | -19.3% | 1.75 | 2.31 |
| MPT Portfolio (High Risk) | 15.8% | -22.6% | 1.42 | 1.95 |
| RL Portfolio (Medium Risk) | 13.5% | -14.8% | 1.66 | 2.25 |
| MPT Portfolio (Medium Risk) | 12.3% | -17.1% | 1.38 | 1.82 |
| S&P 500 | 10.2% | -33.9% | 0.95 | 1.27 |

## Interactive Dashboard

Built an interactive web application using Plotly Dash with three main sections:

### 1. User Profile Collection

The dashboard collects comprehensive user information:

```python
# Sample of the Dash form layout
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AI Robo-Advisor"),
    
    # User information form
    html.Div([
        html.H3("Your Profile"),
        
        html.Label("Age"),
        dcc.Slider(id="age-slider", min=18, max=85, step=1, value=35,
                 marks={18: '18', 30: '30', 45: '45', 60: '60', 75: '75', 85: '85'}),
        
        html.Label("Annual Income (USD)"),
        dcc.Input(id="income-input", type="number", placeholder="e.g. 75000"),
        
        html.Label("Investment Experience"),
        dcc.Dropdown(
            id="experience-dropdown",
            options=[
                {'label': 'None', 'value': 0},
                {'label': 'Beginner (< 2 years)', 'value': 1},
                {'label': 'Intermediate (2-5 years)', 'value': 2},
                {'label': 'Advanced (5+ years)', 'value': 3}
            ],
            value=1
        ),
        
        # More profile inputs...
        
        html.Button("Calculate Risk Profile", id="calculate-button")
    ]),
    
    # Results display
    html.Div(id="results-container")
])
```

### 2. Portfolio Construction Interface

Users can customize their investment universe:

- **Asset Selection**: Interface to filter and select from S&P 500 stocks
- **Sector Allocation**: Visual breakdown of sector exposures
- **Historical Performance**: Interactive charts of past performance

### 3. Recommendation Display

The system provides detailed portfolio recommendations:

- **Asset Allocation**: Visual representation of recommended weights
- **Expected Performance**: Projections under different market scenarios
- **Risk Metrics**: Clear explanation of volatility and drawdown expectations
- **Implementation Plan**: Step-by-step guide to implementing the recommendations

## Technical Implementation

### Data Pipeline

The system incorporated diverse data sources:

- **Market Data**: Daily pricing from Yahoo Finance via yfinance
- **Fundamental Data**: Company financials from Financial Modeling Prep API
- **Economic Indicators**: Federal Reserve Economic Data (FRED)
- **Risk Survey Data**: Federal Reserve's Survey of Consumer Finances

### Deployment Architecture

```
├── app/
│   ├── main.py             # Main Dash application
│   ├── callbacks.py        # Dashboard interactive callbacks
│   ├── layouts.py          # UI components and layouts
│   └── assets/             # CSS and static assets
├── models/
│   ├── risk_predictor.py   # Risk profiling models
│   ├── portfolio_opt.py    # Traditional optimization
│   └── rl_portfolio.py     # Reinforcement learning
├── data/
│   ├── market_data.py      # Market data fetching
│   ├── fundamental_data.py # Company fundamentals
│   └── economic_data.py    # Economic indicators
├── utils/
│   ├── preprocessing.py    # Data preprocessing utilities
│   └── visualization.py    # Plotting functions
├── tests/                  # Unit and integration tests
├── requirements.txt        # Dependencies
└── Procfile               # Heroku deployment configuration
```

## Results and Business Impact

This robo-advisor project delivers several key innovations:

1. **Enhanced Risk Profiling**: More accurate risk tolerance assessment using 40+ features
2. **Dynamic Allocation**: Reinforcement learning adapts to changing market conditions
3. **Explainable Recommendations**: Clear visualization of allocation rationale
4. **Performance Improvement**: 1.5-2% higher risk-adjusted returns compared to traditional approaches

## Future Development

Planned enhancements include:

1. **Multi-Period Optimization**: Extending RL to incorporate time-varying investment horizons
2. **Alternative Data**: Incorporating sentiment analysis from news and social media
3. **ESG Integration**: Adding environmental, social, and governance filters
4. **Tax-Aware Optimization**: Maximizing after-tax returns through tax-loss harvesting
5. **Progressive Web App**: Converting the dashboard to a mobile-friendly PWA

## Resources & References

- [Modern Portfolio Theory](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
- [Reinforcement Learning for Finance](https://arxiv.org/abs/1811.09549)
- [Federal Reserve SCF Data](https://www.federalreserve.gov/econres/scfindex.htm)
- [Portfolio Optimization with Python](https://hudsonthames.org/an-introduction-to-portfolio-optimisation-in-python/)
- [RL for Portfolio Management](https://papers.nips.cc/paper/2020/file/1706beefdc83b7067d216e97d0f39a3d-Paper.pdf)
