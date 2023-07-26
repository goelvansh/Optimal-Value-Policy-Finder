# Optimal Value Policy Finder

## Description

The `Optimal Value Policy Finder` is a Python script that demonstrates different methods for finding the optimal value and policy for an agent faced with a series of questions, each with two possible actions: "CONTINUE" or "QUIT." The agent receives rewards for each "CONTINUE" action but risks losing all accumulated rewards if it answers incorrectly. The objective is to determine the best policy (sequence of actions) that maximizes the total reward for the agent over time.

The script implements three different algorithms:
1. **Value Iteration with Non-Deterministic Actions (ValueIteration):** This algorithm assumes that the agent takes non-deterministic actions at each state and iteratively computes the optimal value and policy for each state.
2. **Value Iteration with Deterministic Actions (ValueIteration1):** This version assumes that the agent takes only deterministic "CONTINUE" actions at each state.
3. **Monte Carlo Simulation (SARS):** This method uses a Monte Carlo approach to simulate the agent's actions for multiple episodes and calculate the average total reward.

## How to Use

1. Open the script `optimal_value_policy_finder.py`.
2. Set the rewards and probabilities for each question in the `rewards` and `probabilities` arrays, respectively.
3. For the `ValueIteration` and `ValueIteration1` methods, set the `future_decay` and `theta` parameters.
4. Run the script.

## Output

The script will display the optimal policy obtained by each algorithm and compare the optimal value calculated by `ValueIteration` and the average total reward from the `SARS` method.

## Requirements

This code requires Python and the NumPy library.

## Sample Usage

```python
# Sample data for rewards and probabilities
rewards = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000])
probabilities = np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

# Create instances of the classes
agent_value_iteration = ValueIteration(states=np.arange(10), rewards=rewards, probabilities=probabilities, future_decay=0.7, theta=1e-12)
agent_sars = SARS(rewards=rewards, probabilities=probabilities)

# Run Value Iteration and obtain the optimal policy
agent_value_iteration.run()
policy_value_iteration = agent_value_iteration.get_policy()
print("For max reward, the agent should follow the policy:", policy_value_iteration)

# Run SARS and obtain the average total reward
average_reward_sars = agent_sars.run()
# Obtain the optimal value from Value Iteration
optimal_value_value_iteration = agent_value_iteration.values[policy_value_iteration == "CONTINUE"].max()
print(f"The optimal value using SARS is {average_reward_sars} and using value iteration it is {optimal_value_value_iteration}")
```
# Note:

_The code can be customized with different reward distributions, probabilities, and parameters to explore various scenarios and compare different algorithms for finding optimal value and policies in reinforcement learning problems._
