import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import gym
from env.custom_hopper import *


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        # I decided to use action-value function (Q-function) as critic
        self.fc1_critic = torch.nn.Linear(state_space + action_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_q_value = torch.nn.Linear(self.hidden, 1)


        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist
    

    def critic_forward(self, x, y):
        # x is the state, y is the action
        """
            Critic
        """
        # TASK 3: forward in the critic network
        state_action = torch.cat([x, y], dim=-1)
        q = self.tanh(self.fc1_critic(state_action))
        q = self.tanh(self.fc2_critic(q))
        q_value = self.fc3_critic_q_value(q)

        return q_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.actor_params = list(self.policy.fc1_actor.parameters()) + \
            list(self.policy.fc2_actor.parameters()) + \
            list(self.policy.fc3_actor_mean.parameters()) + \
            [self.policy.sigma]  
        self.critic_params = list(self.policy.fc1_critic.parameters()) + \
                list(self.policy.fc2_critic.parameters()) + \
                list(self.policy.fc3_critic_q_value.parameters())
        
        self.optimizer_critic = torch.optim.Adam(self.critic_params, lr=1e-3)
        self.optimizer_actor = torch.optim.Adam(self.actor_params, lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
   
    

    def update_policy(self, env):
        # action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        # states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        # next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        # rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        # done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #
        # reward_reinforce = self.REINFORCE(env, maxSteps=1000, baseline=20)  # Call the REINFORCE method to update the policy


        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #
        reward_actorCritic = self.ActorCritic(env, maxSteps=10000)  # Call the ActorCritic method to update the policy

        return reward_actorCritic


    def REINFORCE(self,env,maxSteps, baseline):
        done = False
        state = env.reset()	# Reset environment to initial state
        steps = 0
        episode_reward = 0
        
        #generate a trajectory for a single episode
        while not done and steps < maxSteps:  # Until the episode is over

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
            distribution = self.policy(state_tensor) 

            action = distribution.sample()
            action_log_prob = distribution.log_prob(action).sum()  # Sum across dimensions if multidimensional

            next_state, reward, done, _ = env.step(action.cpu().numpy())  # Send tensor to numpy

            episode_reward += reward
            self.states.append(state_tensor.squeeze(0))
            self.action_log_probs.append(action_log_prob)
            self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.train_device))
            state = next_state

        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = [Gt - baseline for Gt in returns]


        self.optimizer.zero_grad()  
        loss = 0
        for log_prob, Gt in zip(self.action_log_probs, returns):
            loss += -log_prob * Gt 

        loss.backward()           
        self.optimizer.step()       

        return episode_reward     


    def ActorCritic(self, env, maxSteps):
        done = False
        state = env.reset()	# Reset environment to initial state
        steps = 0
        episode_reward = 0
        
        while not done and steps < maxSteps:
            # sample action from the policy pi
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
            distribution = self.policy(state_tensor) 

            action = distribution.sample()
            action_log_prob = distribution.log_prob(action).sum()  # Sum across dimensions if multidimensional

            Q_s_a = self.policy.critic_forward(state_tensor, action)  # Q-value for the current state-action pair

            next_state, reward, done, _ = env.step(action.cpu().numpy())  # Send tensor to numpy

            # computing TD error (Advantage)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.train_device)
            next_distribution = self.policy(next_state_tensor)
            next_action = next_distribution.sample()
            Q_s_a_next = self.policy.critic_forward(next_state_tensor, next_action)
            # TD error
            target = reward + self.gamma * Q_s_a_next.detach()
            delta = target - Q_s_a  # Q_s_a still connected to critic network  


            # update theta (actor parameters)
            actor_loss = -Q_s_a.detach() * action_log_prob  # Negative because we want to maximize
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()



            # update weights of the critic network (w)
            critic_loss = delta.pow(2)  # MSE loss
            self.optimizer_critic.zero_grad()
            critic_loss.backward()  # retain_graph=True to keep the graph for the actor update
            self.optimizer_critic.step()

            state = next_state
            steps += 1
            episode_reward += reward

        return episode_reward



    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x).normal_dist

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

