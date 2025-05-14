---
layout: post
title: "Engineering an AI Robo-Advisor: A Technical Deep Dive into Risk Profiling and RL-Based Portfolio Optimization"
date: 2023-10-25 09:30:00 +0800 # Retaining original date
categories: [ai, finance, machine-learning, reinforcement-learning, tutorial]
tags: [robo-advisor, portfolio-optimization, risk-profiling, supervised-learning, deep-q-network, python, scikit-learn, tensorflow, keras, cvxopt, plotly-dash]
author: Wes Lee
feature_image: /assets/images/2023-10-25-robo-advisor-risk-profiling-portfolio-optimization.jpg # Or a new, more technical image
---

## Introduction: Building a Smarter Financial Advisor

Traditional financial advice often comes with high barriers to entry. Robo-advisors aim to democratize wealth management, but many existing solutions rely on simplistic rule-based systems. This post details the technical journey of building an advanced AI-powered robo-advisor. We'll explore the two core AI components: a supervised learning model for predicting investor risk tolerance and a reinforcement learning (Deep Q-Network) agent for dynamic portfolio optimization, comparing its output to traditional Modern Portfolio Theory (MPT).

> For a higher-level overview of this project's business context, system architecture, and overall impact, please see the [Intelligent Investing Project Page](/projects/ai-robo-advisor-project-page/).

## Data Foundation: Fueling the AI

Two primary datasets powered this project:

1.  **2019 Survey of Consumer Finances (SCF):** Provided by the U.S. Federal Reserve, this rich dataset contains detailed demographic, financial, and investment behavior information for thousands of households. It was the cornerstone for training our risk tolerance prediction model.
2.  **S&P 500 Historical Stock Prices (2000-2023):** Retrieved using the `yfinance` library (by scraping tickers from Wikipedia), this data was crucial for training and backtesting the portfolio optimization algorithms, especially the reinforcement learning agent.

## Component 1: Predicting Investor Risk Tolerance with Supervised Learning

Accurately assessing an investor's risk tolerance is the first step in personalized financial advice.

### SCF Data Preparation and Feature Engineering
The SCF dataset (5,777 respondents, 40+ variables) required significant preprocessing:
-   Cleaning and normalization.
-   Feature Engineering: Creating composite indicators like debt-to-income ratio, savings rate, and critically, a synthetic risk tolerance score.

### Defining Risk Tolerance (The Target Variable)
We defined risk tolerance as the ratio of an individual's risky assets to their total investment assets (risky + risk-free).

```python
# Python: Calculating Risk Tolerance from SCF data
# Assume 'working_df' is the preprocessed SCF DataFrame

# Define categories of risk-free assets
working_df['RiskFreeAssets'] = (
    working_df['LIQ'] + working_df['MMA'] + 
    working_df['CHECKING'] + working_df['SAVING'] + 
    working_df['CALL'] + working_df['CDS'] +
    working_df['PREPAID'] + working_df['SAVBND'] + 
    working_df['CASHLI']
)

# Define categories of risky assets
working_df['RiskyAssets'] = (
    working_df['NMMF'] + working_df['STOCKS'] + 
    working_df['BOND'] + working_df['OTHMA'] + 
    working_df['OTHFIN'] + working_df['RETQLIQ']
)

# Calculate risk tolerance score (0 to 1)
# Handle potential division by zero if total assets are zero
total_investments = working_df['RiskyAssets'] + working_df['RiskFreeAssets']
working_df['risk_tolerance_score'] = working_df['RiskyAssets'].divide(total_investments).fillna(0)

# Remove rows where total_investments is zero or negative to avoid issues
working_df = working_df[total_investments > 0]
```

### Model Development and Selection for Risk Prediction
We evaluated several regression models. Ensemble methods, particularly Extra Trees Regressor, showed superior performance.

```python
# Python: Model training pipeline for risk tolerance prediction
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assume X contains features from SCF, y contains 'risk_tolerance_score'
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler()) # Standardize features
])

# Define model candidates and their hyperparameter grids
models_to_tune = {
    'extra_trees': (ExtraTreesRegressor(random_state=42), {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }),
    'gradient_boosting': (GradientBoostingRegressor(random_state=42), {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5]
    }),
    'neural_network': (MLPRegressor(random_state=42, max_iter=1000, early_stopping=True), {
        'regressor__hidden_layer_sizes': [(50,), (100,), (50, 25)],
        'regressor__activation': ['relu', 'tanh'],
        'regressor__alpha': [0.0001, 0.001]
    })
}

best_estimators = {}
# Fit and tune models
# for name, (model_instance, params) in models_to_tune.items():
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', model_instance)
#     ])
#     grid_search = GridSearchCV(pipeline, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     best_estimators[name] = grid_search.best_estimator_
#     print(f"Best parameters for {name}: {grid_search.best_params_}")
    
    # Evaluate the best model for each type
    # y_pred = grid_search.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # r2 = r2_score(y_test, y_pred)
    # print(f"{name} - Test RMSE: {rmse:.4f}, Test R2: {r2:.4f}")

# The Extra Trees Regressor was selected (R²=0.91, RMSE=0.115).
# Key features: Age, Income Stability, Financial Knowledge, Existing Asset Allocations, Time Horizon.
```
The Extra Trees Regressor achieved an R² of 0.91 and an RMSE of 0.115, proving effective for predicting risk tolerance.

## Component 2: Portfolio Optimization Engine - Classical vs. AI

We implemented two distinct methods for optimizing asset allocation.

### Approach A: Modern Portfolio Theory (MPT) with Convex Optimization
MPT aims to maximize expected return for a given level of risk. We used `cvxopt` for this.

```python
# Python: Markowitz Portfolio Optimization using cvxopt
import numpy as np
import pandas as pd # Assuming returns_df is a Pandas DataFrame
import cvxopt as opt
from cvxopt import blas, solvers

solvers.options['show_progress'] = False # Suppress solver output

def optimize_portfolio_mpt(returns_df, target_risk_level=None, target_return_level=None, risk_tolerance_factor=0.5):
    """
    Performs Markowitz portfolio optimization.
    Args:
        returns_df (pd.DataFrame): DataFrame where columns are assets and rows are historical returns.
        target_risk_level (float, optional): Desired portfolio volatility (std dev).
        target_return_level (float, optional): Desired portfolio expected return.
        risk_tolerance_factor (float): Value from 0 (min risk) to 1 (max return focus).
                                      Used if target_risk/return are not specified.
    Returns:
        np.array: Optimal asset weights.
    """
    returns_matrix = np.array(returns_df).T # Assets as rows, observations as columns
    num_assets = returns_matrix.shape[0]
    
    mean_returns = np.array([np.mean(asset_returns) for asset_returns in returns_matrix])
    cov_matrix = np.cov(returns_matrix)

    P = opt.matrix(cov_matrix)
    q = opt.matrix(np.zeros(num_assets)) # No linear term in objective for min variance

    # Constraints: sum(weights) = 1 and weights >= 0
    G = opt.matrix(np.vstack((-np.eye(num_assets), np.eye(num_assets)))) # For w_i >= 0 and w_i <= 1 (implicitly by sum=1)
    h_lower = np.zeros(num_assets)
    h_upper = np.ones(num_assets) 
    h = opt.matrix(np.concatenate([h_lower, h_upper]))
    
    A = opt.matrix(1.0, (1, num_assets))
    b = opt.matrix(1.0)

    # Adjust objective based on risk tolerance if specific targets aren't given
    # Maximize (risk_tolerance_factor * mu'w) - ((1-risk_tolerance_factor)/2 * w'Pw)
    # This is equivalent to minimizing ((1-risk_tolerance_factor)/2 * w'Pw) - (risk_tolerance_factor * mu'w)
    # q_objective = -opt.matrix(mean_returns * risk_tolerance_factor)
    # P_objective = P * (1 - risk_tolerance_factor) # Scale covariance by risk aversion

    # For a simpler QP formulation: min (1/2)x'Px + q'x
    # To maximize mu'w - lambda * w'Sigma w (lambda is risk aversion)
    # We solve min lambda * w'Sigma w - mu'w
    # Here, lambda is related to (1-risk_tolerance_factor)
    lambda_risk_aversion = 50 * (1 - risk_tolerance_factor + 1e-6) # Heuristic scaling for lambda
    
    P_objective = opt.matrix(lambda_risk_aversion * cov_matrix)
    q_objective = -opt.matrix(mean_returns)


    solution = solvers.qp(P_objective, q_objective, G, h, A, b)
    
    optimal_weights = np.array(solution['x']).flatten()
    return optimal_weights / np.sum(optimal_weights) # Ensure sum to 1 due to potential numerical precision

# Example usage:
# historical_returns_example_df = pd.DataFrame(np.random.rand(100, 3), columns=['AssetA', 'AssetB', 'AssetC'])
# weights = optimize_portfolio_mpt(historical_returns_example_df, risk_tolerance_factor=0.7)
# print("MPT Optimal Weights:", weights)
```
This provides a baseline, theoretically sound allocation based on historical data and the user's predicted risk tolerance.

### Approach B: Reinforcement Learning (Deep Q-Network - DQN)
To create a more adaptive system, we developed a DQN agent.

**1. Custom Gym Environment (`PortfolioEnv`):**
A custom environment was built using OpenAI Gym to simulate portfolio management.
-   **State Space:** Historical price movements (e.g., last 30 days of returns for selected assets) and current portfolio allocation.
-   **Action Space:** Discrete set of possible portfolio reallocations (e.g., for 3 assets, 3 allocation levels each -> 3^3 = 27 actions). Each action maps to a specific weight distribution.
-   **Reward Function:** Annualized Sharpe ratio of the portfolio's returns during a step. This encourages high risk-adjusted returns.

```python
import gym
import numpy as np

class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, daily_returns_df, initial_capital=100000, lookback_window=30, episode_length=252):
        super(PortfolioEnv, self).__init__()

        self.daily_returns_df = daily_returns_df # DataFrame of daily returns, columns are assets
        self.num_assets = daily_returns_df.shape[1]
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.episode_length = episode_length # Number of trading days in an episode

        # Action space: discrete actions representing different allocations
        # Example: 3 assets, 3 levels of allocation (e.g., 0%, 50%, 100% of a part of portfolio)
        # Total actions = levels ^ num_assets. For simplicity, let's use a fixed number.
        self.num_allocation_levels = 3 
        self.action_space = gym.spaces.Discrete(self.num_allocation_levels**self.num_assets) # e.g., 3^3 = 27 for 3 assets

        # Observation space: lookback_window days of returns for each asset + current weights
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback_window * self.num_assets + self.num_assets,),
            dtype=np.float32
        )
        self.current_episode_step = 0
        self.current_data_start_idx = 0 # To vary starting point of data for each episode
        self.reset()

    def _action_to_weights(self, action_idx):
        """Converts a discrete action index to portfolio weights."""
        weights = np.zeros(self.num_assets)
        # Example: action_idx for 3 assets, 3 levels (0,1,2)
        # action_idx = w1*3^0 + w2*3^1 + w3*3^2
        # This needs a robust mapping. A simpler way for N assets, K levels:
        # Each digit in base K representation of action_idx corresponds to an asset's level.
        
        # Simplified example: if action_idx is a tuple of weights directly
        # For discrete actions, map index to predefined weight sets or use a base-N system
        temp_action = action_idx
        for i in range(self.num_assets):
            level = temp_action % self.num_allocation_levels
            weights[i] = level / (self.num_allocation_levels - 1.0) # Scale to 0-1
            temp_action //= self.num_allocation_levels
        
        if np.sum(weights) == 0: # Avoid all-zero weights if mapping results in it
            return np.full(self.num_assets, 1.0 / self.num_assets)
        return weights / np.sum(weights) # Normalize to sum to 1

    def step(self, action_idx):
        self.current_weights = self._action_to_weights(action_idx)
        
        # Get returns for the current day
        current_day_returns = self.episode_data.iloc[self.current_episode_step]
        portfolio_return_today = np.dot(current_day_returns, self.current_weights)
        self.portfolio_daily_returns.append(portfolio_return_today)
        
        # Calculate reward (e.g., Sharpe ratio of returns so far in episode)
        reward = self._calculate_sharpe_ratio()

        self.current_episode_step += 1
        done = self.current_episode_step >= self.episode_length
        
        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        # Observation is past 'lookback_window' returns and current weights
        start_idx = self.current_data_start_idx + self.current_episode_step - self.lookback_window
        end_idx = self.current_data_start_idx + self.current_episode_step
        
        historical_returns = self.daily_returns_df.iloc[start_idx:end_idx].values.flatten()
        obs = np.concatenate((historical_returns, self.current_weights))
        return obs

    def reset(self):
        # Reset for a new episode: pick a random start point in data
        self.current_data_start_idx = np.random.randint(
            0, len(self.daily_returns_df) - self.episode_length - self.lookback_window
        )
        self.episode_data = self.daily_returns_df.iloc[
            self.current_data_start_idx : self.current_data_start_idx + self.episode_length + self.lookback_window
        ]

        self.current_episode_step = self.lookback_window 
        self.current_weights = np.full(self.num_assets, 1.0 / self.num_assets) # Start with equal weights
        self.portfolio_daily_returns = []
        return self._get_observation()

    def _calculate_sharpe_ratio(self, risk_free_rate_daily=0.0): # Assuming 0 risk-free for simplicity
        if len(self.portfolio_daily_returns) < self.lookback_window // 2: # Need some returns to calculate std
            return 0.0
        
        returns_arr = np.array(self.portfolio_daily_returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr)
        
        if std_return == 0: # Avoid division by zero
            return 0.0
            
        sharpe_ratio = (mean_return - risk_free_rate_daily) / std_return
        return sharpe_ratio * np.sqrt(252) # Annualized
```

**2. DQN Agent Architecture (Keras):**
A neural network was defined to approximate the Q-value function.
```python
# Python: Building the DQN Agent Model using TensorFlow/Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout # Added Dropout
from tensorflow.keras.optimizers import Adam
# from rl.agents.dqn import DQNAgent # From keras-rl or similar library
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory

def build_dqn_model(input_shape_tuple, num_actions):
    """Builds a Keras Sequential model for the DQN agent."""
    model = Sequential()
    model.add(Flatten(input_shape=input_shape_tuple)) # Input shape e.g., (1, obs_space_dim) for keras-rl
    model.add(Dense(128, activation='relu')) # Reduced complexity
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_actions, activation='linear')) # Output Q-values for each action
    return model

# Example usage:
# env = PortfolioEnv(historical_returns_example_df) # Assuming historical_returns_example_df is loaded
# obs_space_shape = (1,) + env.observation_space.shape # Add batch dimension for Keras-RL
# dqn_keras_model = build_dqn_model(obs_space_shape, env.action_space.n)
# print(dqn_keras_model.summary())

# memory = SequentialMemory(limit=50000, window_length=1)
# policy = EpsGreedyQPolicy(eps=0.1) # Epsilon for exploration

# dqn_agent = DQNAgent(
#     model=dqn_keras_model,
#     nb_actions=env.action_space.n,
#     memory=memory,
#     nb_steps_warmup=1000,       # Steps before learning starts
#     target_model_update=1e-2, # Soft update parameter for target network
#     policy=policy,
#     gamma=0.99                 # Discount factor for future rewards
# )
# dqn_agent.compile(Adam(learning_rate=1e-4), metrics=['mae'])

# Train the agent
# dqn_agent.fit(env, nb_steps=100000, visualize=False, verbose=1) # nb_steps is total training steps
```

**3. Training Process:**
The agent was trained over many episodes, interacting with the `PortfolioEnv`. It used an epsilon-greedy policy for exploration and experience replay to learn efficiently. The goal was to learn a policy (mapping from states to actions) that maximizes the cumulative discounted reward (Sharpe ratio).

```python
# Conceptual RL training loop (simplified from blog)
# episode_count = 100 # Number of episodes to train for
# batch_size = 32
# data_length = len(env.daily_returns_df)
# window_size = env.lookback_window
# rebalance_period = 5 # Rebalance every 5 days

# for e in range(episode_count):
#     current_state = env.reset() # Get initial state
#     total_episode_reward = 0
    
#     for t in range(window_size, data_length - rebalance_period, rebalance_period):
#         action_idx = dqn_agent.forward(current_state) # Get action from agent
#         # For keras-rl, agent.act(state) is used in training, forward is for testing
        
#         # Environment step (simplified)
#         # next_state, reward, done, _ = env.step_simulation(t, action_idx, rebalance_period) 
#         # The PortfolioEnv.step method handles this logic internally when dqn_agent.fit is called.

#         # dqn_agent.memory.append(current_state, action_idx, reward, next_state, done) # Keras-RL handles this
#         # if len(dqn_agent.memory) > batch_size:
#         #    dqn_agent.backward(batch_size) # Keras-RL handles this during fit
        
#         current_state = next_state
#         total_episode_reward += reward
#         if done:
#             break
#     print(f"Episode {e+1}/{episode_count}, Total Reward: {total_episode_reward:.2f}")
```
Backtesting showed the DQN approach could achieve better risk-adjusted returns (Sharpe ratio 1.75 for high risk) and lower drawdowns (-19.3%) compared to MPT (Sharpe 1.42, Drawdown -22.6%) and the S&P 500 benchmark, especially during volatile periods like the COVID-19 crash.

## Component 3: Interactive Dashboard with Plotly Dash

A Plotly Dash application serves as the user interface.

### User Profile Input
The dashboard collects 13 key demographic and financial data points from the user (age, income, experience, etc.) via sliders, dropdowns, and input fields.

### Risk Tolerance Prediction Callback
A Dash callback takes these user inputs, feeds them into the pre-trained Extra Trees Regressor model, and displays the predicted risk tolerance score (0-1 scale).

```python
# Dash Callback for Risk Tolerance Prediction (Conceptual)
# from dash.dependencies import Input, Output, State
# import joblib # Assuming model is saved

# risk_model = joblib.load('path_to_your_trained_extra_trees_model.pkl')
# input_scaler = joblib.load('path_to_your_fitted_scaler.pkl')

# @app.callback(
#     Output('risk-score-display', 'children'), # Component to display score
#     [Input('submit-profile-button', 'n_clicks')],
#     [State('age-slider', 'value'),
#      State('income-input', 'value'),
#      # ... other State inputs for all 13 features
#      ]
# )
# def update_risk_score(n_clicks, age, income, ...other_feature_values):
#     if n_clicks is None or n_clicks == 0:
#         return "Please submit your profile."
    
#     # Create feature vector from inputs in the correct order model expects
#     user_features = np.array([[age, income, ...]]).reshape(1, -1)
    
#     # Scale features using the same scaler used during training
#     scaled_user_features = input_scaler.transform(user_features)
    
#     # Predict risk tolerance
#     predicted_risk_score = risk_model.predict(scaled_user_features)[0]
    
#     return f"Your Predicted Risk Tolerance Score: {predicted_risk_score:.2f} (out of 1.0)"
```

### Portfolio Recommendation and Chatbot
The dashboard then presents optimized portfolio allocations from both MPT and the RL agent, tailored to the predicted risk score. An OpenAI-powered chatbot provides additional financial guidance.

```python
# Dash Callback for Chatbot (Conceptual)
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from langchain_openai import OpenAI # Updated import

# openai_api_key = "YOUR_OPENAI_API_KEY"
# llm_chat = OpenAI(temperature=0, openai_api_key=openai_api_key)
# conversation_memory = ConversationBufferMemory()
# chat_conversation_chain = ConversationChain(
#     llm=llm_chat,
#     verbose=False, # Set to True for debugging
#     memory=conversation_memory
# )

# @app.callback(
#     Output('chatbot-output-div', 'children'),
#     [Input('submit-chat-button', 'n_clicks')],
#     [State('chat-input-field', 'value'),
#      State('chatbot-history-store', 'data')] # dcc.Store to hold history
# )
# def update_chatbot_conversation(n_clicks, user_input, chat_history):
#     if n_clicks is None or n_clicks == 0 or not user_input:
#         return chat_history or [] # Display existing history or nothing

#     # Update conversation memory from stored history if needed
#     # For simplicity, this example doesn't fully rehydrate Langchain's memory from dcc.Store
    
#     ai_response = chat_conversation_chain.predict(input=user_input)
    
#     new_entry_user = html.P(f"You: {user_input}")
#     new_entry_ai = html.P(f"RoboChat: {ai_response}")
    
#     updated_history = (chat_history or []) + [new_entry_user, new_entry_ai]
#     return updated_history
```

## Technical Challenges & Learnings

-   **SCF Data Nuances:** Cleaning and interpreting the complex SCF data, then engineering a meaningful synthetic risk score, was a significant undertaking.
-   **RL Environment Design:** Crafting an RL environment that accurately reflects market dynamics while providing a stable learning signal (reward function) is challenging. The choice of state representation, action space discretization, and reward shaping are critical.
-   **Computational Resources:** Training DQN agents can be computationally intensive and time-consuming.
-   **MPT Sensitivity:** MPT results are highly sensitive to the estimates of expected returns and covariances, which are hard to predict accurately. RL offers a way to learn policies less dependent on these explicit forecasts.

## Conclusion

This project successfully demonstrated the engineering of an AI-powered robo-advisor by integrating supervised learning for nuanced risk profiling and reinforcement learning for dynamic, adaptive portfolio optimization. The technical deep dive shows how these advanced AI techniques can be practically applied to build financial tools that offer more personalized and potentially more effective investment strategies than traditional methods or simpler robo-advisors. The combination of a data-driven risk assessment with an intelligent allocation engine represents a significant step towards democratizing sophisticated wealth management.

---

*This post details the technical engineering of the AI-Powered Robo Advisor. For more on the project's overall architecture, business impact, and future directions, please visit the [project page](/projects/ai-robo-advisor-project-page/). The source code is available on [GitHub](https://github.com/Adredes-weslee/Using_Artificial_Intelligence_to_Develop_a_Robo_Advisor).*
