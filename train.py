"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import matplotlib.pyplot as plt

import torch
import gym
import pickle
import os

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #
	save_dir = "/content/drive/MyDrive/rl_logs"
	os.makedirs(save_dir, exist_ok=True)

	# File to save both reward histories
	save_path = os.path.join(save_dir, "reward_history.pkl")

	# Initialize or load rewards
	if os.path.exists(save_path):
		with open(save_path, 'rb') as f:
			reward_data = pickle.load(f)
			rewards_reinforce = reward_data.get('reinforce', [])
			rewards_actor_critic = reward_data.get('actor_critic', [])
	else:
		rewards_reinforce = []
		rewards_actor_critic = []

	for episode in range(10000):
		reward1, reward2 = agent.update_policy(env)
		print(f"Episode {episode} - REINFORCE: {reward1} , ACTOR_CRITIC: {reward2}")
		rewards_reinforce.append(reward_reinforce)
		rewards_actor_critic.append(reward_actor_critic)
		
		with open(save_path, 'wb') as f:
			pickle.dump({
                'reinforce': rewards_reinforce,
                'actor_critic': rewards_actor_critic
            }, f)
	# 	done = False
	# 	train_reward = 0
	# 	state = env.reset()  # Reset the environment and observe the initial state

	# 	while not done:  # Loop until the episode is over

	# 		action, action_probabilities = agent.get_action(state)
	# 		previous_state = state

	# 		state, reward, done, info = env.step(action.detach().cpu().numpy())

	# 		agent.store_outcome(previous_state, state, action_probabilities, reward, done)

	# 		train_reward += reward
		
	# 	if (episode+1)%args.print_every == 0:
	# 		print('Training episode:', episode)
	# 		print('Episode return:', train_reward)

	#torch.save(agent.policy.state_dict(), "model.mdl")

	

if __name__ == '__main__':
	main()