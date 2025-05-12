---
layout: project
title: "AI-Powered Robo Advisor for Portfolio Optimization"
categories: machine-learning finance reinforcement-learning
image: /assets/images/placeholder.svg
technologies: [Python, Reinforcement Learning, Plotly Dash, Scikit-Learn, Deep Learning]
github: https://github.com/Adredes-weslee/Using_Artificial_Intelligence_to_Develop_a_Robo_Advisor
---

## Project Overview

Developed a full-stack, intelligent robo advisor using supervised learning, reinforcement learning, and convex optimization. This system predicts investor risk tolerance and dynamically optimizes investment portfolios to maximize returns.

## System Architecture

The robo advisor consists of three main components:

1. **Risk Profile Predictor**: Predicts an investor's risk tolerance based on demographic and financial data
2. **Portfolio Optimizer**: Allocates assets using both traditional methods and reinforcement learning
3. **Interactive Dashboard**: Allows users to input their information and receive personalized recommendations

## Risk Profiling

Using survey data from the Federal Reserve's 2019 Survey of Consumer Finances (SCF), I trained multiple regression models to predict an investor's risk tolerance score:

- Tested approaches including Extra Trees, Neural Networks, and Gradient Boosting
- Achieved R² = 0.91 and RMSE = 0.115 using Extra Trees Regressor
- Features included age, income, education level, net worth, and financial behavior indicators

## Portfolio Optimization

Implemented two different strategies for optimal weight allocation:

### Traditional Approach
- Mean-Variance Optimization (MVO) using convex programming (CVXOPT)
- Optimized for expected return given a risk constraint based on the user's profile

### Reinforcement Learning Approach
- Trained a Deep Q-Network (DQN) to dynamically adjust portfolio weights
- Optimized for maximum Sharpe Ratio (achieving ~1.75)
- Outperformed static allocation in backtesting scenarios

## Interactive Dashboard

Built an interactive web application using Plotly Dash:

- User input form for collecting demographic and financial information
- S&P 500 stock selector for portfolio construction
- Visualization of optimized portfolios and performance metrics
- Deployed on Heroku for public access

![Robo Advisor Dashboard](/assets/images/robo-dashboard.png)

## Code Sample

```python
# Deep Q-Network (DQN) for portfolio optimization
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
```

## Results

The robo advisor achieved impressive results across multiple metrics:

- **Risk Tolerance Prediction**: R² = 0.91, RMSE = 0.115
- **Portfolio Optimization**: RL-based portfolio achieved Sharpe Ratio of 1.75
- **Evaluation Methods**: Cross-validation, out-of-sample testing, and portfolio backtests

## Technologies Used

- **Python** - Core programming language
- **TensorFlow/Keras** - For the DQN implementation
- **Scikit-learn** - For supervised learning models
- **CVXOPT** - For convex optimization
- **Pandas** - For data manipulation
- **Plotly Dash** - For the interactive web interface
- **Heroku** - For deployment
