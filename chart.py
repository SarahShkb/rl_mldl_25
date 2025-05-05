import pickle
import matplotlib.pyplot as plt
import numpy as np



# Helper: moving average smoothing
def moving_average_fast(x, w):
    x = np.array(x)
    if len(x) < w:
        return x  # Return as-is if not enough data
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[w:] - cumsum[:-w]) / float(w)

# Load reward data
path = 'reward_history.pkl'

with open(path, 'rb') as f:
    reward_data = pickle.load(f)

print("file loaded")

rewards_reinforce = reward_data['reinforce']
rewards_actor_critic = reward_data['actor_critic']

rewards_reinforce = rewards_reinforce[:1000]
rewards_actor_critic = rewards_actor_critic[:1000]
print(rewards_reinforce[0])

print("arrays defined!")


# Fast smoothing
# window = 100
# reinforce_smoothed = moving_average_fast(rewards_reinforce, window)
# actor_critic_smoothed = moving_average_fast(rewards_actor_critic, window)

print("plotting...")


# Plot raw data
plt.figure(figsize=(12, 6))
plt.plot(rewards_reinforce, label='REINFORCE (raw)', color='blue')
plt.plot(rewards_actor_critic, label='Actor-Critic (raw)', color='green')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('REINFORCE vs Actor-Critic (No Smoothing)')
plt.legend()
plt.grid(True)
plt.show()