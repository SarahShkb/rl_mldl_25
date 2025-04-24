"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os
os.environ['MUJOCO_GL'] = 'osmesa'
print("About to import gym...")
import gym
from gym.wrappers import RecordVideo
print("Successfully imported gym!")
print("About to env.custom_hopper...")
from env.custom_hopper import *
print("Successfully imported env.custom_hopper!")


def main():
    try:
        print("About to call gym.make...")
        train_env = gym.make('CustomHopper-source-v0')

        print('State space:', train_env.observation_space)  # state-space
        print('Action space:', train_env.action_space)  # action-space
        print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

        #
        # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
        #
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()