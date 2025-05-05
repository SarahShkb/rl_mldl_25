import pickle
import matplotlib.pyplot as plt
import numpy as np

# Helper: Moving average for smoothing
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Load REINFORCE rewards
with open('reinforce_rewards.pkl', 'rb') as f:
    reinforce_rewards = pickle.load(f)

# Load Actor-Critic rewards
with open('actor_critic_rewards.pkl', 'rb') as f:
    actor_critic_rewards = pickle.load(f)

# Apply smoothing (optional but recommended)
reinforce_smoothed = moving_average(reinforce_rewards)
actor_critic_smoothed = moving_average(actor_critic_rewards)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(reinforce_smoothed, label='REINFORCE (smoothed)', color='blue')
plt.plot(actor_critic_smoothed, label='Actor-Critic (smoothed)', color='green')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Comparison: REINFORCE vs Actor-Critic')
plt.grid(True)
plt.legend()
plt.show()
