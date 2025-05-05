import pickle
import matplotlib.pyplot as plt
import numpy as np

# Helper: moving average smoothing
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Load reward data
path = '/content/drive/MyDrive/rl_logs/reward_history.pkl'

with open(path, 'rb') as f:
    reward_data = pickle.load(f)

rewards_reinforce = reward_data['reinforce']
rewards_actor_critic = reward_data['actor_critic']

# Apply smoothing
reinforce_smoothed = moving_average(rewards_reinforce)
actor_critic_smoothed = moving_average(rewards_actor_critic)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(reinforce_smoothed, label='REINFORCE (smoothed)', color='blue')
plt.plot(actor_critic_smoothed, label='Actor-Critic (smoothed)', color='green')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Comparison of REINFORCE vs Actor-Critic')
plt.legend()
plt.grid(True)
plt.show()
