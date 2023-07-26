import numpy as np

class ValueIteration:
    def __init__(self, states, rewards, probabilities, future_decay, theta) -> None:
        self.states = states
        self.values = np.zeros(len(states))
        self.rewards = rewards
        self.probabilities = probabilities
        self.future_decay = future_decay
        self.theta = theta

    def run(self):
        while True:
            prior_values = self.values.copy()
            for state in self.states:
                # Calculate the expected rewards for both "CONTINUE" and "QUIT" actions at the current state
                action_continue_reward = self.probabilities[state] * (self.rewards[state] + self.future_decay * self.values[state + 1])
                action_quit_reward = np.sum(self.rewards[:state])
                # Update the value of the current state to the maximum of the two expected rewards
                self.values[state] = max(action_continue_reward, action_quit_reward)
            # Check for convergence based on the specified threshold 'theta'
            if np.allclose(self.values, prior_values, atol=self.theta):
                break

    def get_policy(self):
        action_array = np.empty(len(self.states), dtype=object)
        for state in self.states:
            # Calculate the expected rewards for both "CONTINUE" and "QUIT" actions at the current state
            action_continue_reward = self.probabilities[state] * (self.rewards[state] + self.future_decay * self.values[state + 1])
            action_quit_reward = np.sum(self.rewards[:state])
            # Choose the action that leads to the higher expected reward
            action_array[state] = "CONTINUE" if action_continue_reward >= action_quit_reward else "QUIT"
        return action_array

class SARS:
    def __init__(self, rewards, probabilities) -> None:
        self.rewards = rewards
        self.probabilities = probabilities

    def run(self):
        total_reward = 0
        num_episodes = 10000
        for _ in range(num_episodes):
            curr_reward = 0
            for ques in range(len(self.rewards)):
                # Simulate the agent's actions for a single episode (question)
                if np.random.random() > self.probabilities[ques]:
                    break
                else:
                    curr_reward += self.rewards[ques]
            total_reward += curr_reward
        # Calculate the average total reward obtained over all episodes
        return total_reward / num_episodes

# Sample data for rewards and probabilities
rewards = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000])
probabilities = np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

# Create instances of the classes
agent_value_iteration = ValueIteration(states=np.arange(10), rewards=rewards, probabilities=probabilities, future_decay=0.7, theta=1e-12)
agent_sars = SARS(rewards=rewards, probabilities=probabilities)

print("========================================================================================")
# Run Value Iteration and obtain the optimal policy
agent_value_iteration.run()
policy_value_iteration = agent_value_iteration.get_policy()
print("For max reward, the agent should follow the policy:", policy_value_iteration)
print("========================================================================================")
# Run SARS and obtain the average total reward
average_reward_sars = agent_sars.run()
# Obtain the optimal value from Value Iteration
optimal_value_value_iteration = agent_value_iteration.values[policy_value_iteration == "CONTINUE"].max()
print(f"The optimal value using SARS is {average_reward_sars} and using value iteration it is {optimal_value_value_iteration}")
print("========================================================================================")
