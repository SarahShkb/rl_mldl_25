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
        # I decided to use action-value function (Q-function) as critic at first,
        # but it didn't work well, so I changed it to state-value function (V-function)
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)  
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)  


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
    

    def critic_forward(self, x):
        """
            Critic: now takes only state
        """
        v = self.tanh(self.fc1_critic(x))
        v = self.tanh(self.fc2_critic(v))
        v_value = self.fc3_critic_value(v)

        return v_value


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
                list(self.policy.fc3_critic_value.parameters())
        
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

        self.states, self.next_states, self.action_log_probs, self.actions, self.values, self.rewards, self.done = [], [], [], [], [], [], []

        #
        # TASK 2:
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #
        reward_reinforce = self.REINFORCE(env, maxSteps=1000000, baseline=20)  # Call the REINFORCE method to update the policy

        self.states, self.next_states, self.action_log_probs, self.actions, self.values, self.rewards, self.done = [], [], [], [], [], [], []

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #
        reward_actorCritic = self.ActorCritic(env, maxSteps=1000000)  # Call the ActorCritic method to update the policy

        return reward_reinforce, reward_actorCritic
        #return reward_reinforce


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

        # returns = [Gt - baseline for Gt in returns]


        self.optimizer.zero_grad()  
        loss = 0
        for log_prob, Gt in zip(self.action_log_probs, returns):
            loss += -log_prob * Gt 

        loss.backward()           
        self.optimizer.step()       

        return episode_reward     


    def ActorCritic(self, env, maxSteps):
        done = False
        state = env.reset()
        steps = 0
        episode_reward = 0

        while not done and steps < maxSteps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
            distribution = self.policy(state_tensor)
            action = distribution.sample()
            action_log_prob = distribution.log_prob(action).sum()

            V_s = self.policy.critic_forward(state_tensor)

            next_state, reward, done, _ = env.step(action.cpu().numpy())

            # Store the outcome
            self.states.append(state_tensor)
            self.actions.append(action)
            self.action_log_probs.append(action_log_prob)
            self.rewards.append(reward)
            self.done.append(done)
            self.values.append(V_s)

            state = next_state
            steps += 1
            episode_reward += reward


        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.train_device)
        V_s_next = self.policy.critic_forward(next_state_tensor).detach()

        # first, normalize the rewards
        rewards = np.array(self.rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        self.rewards = rewards.tolist()

        returns = []
        R = V_s_next
        for r, d in zip(reversed(self.rewards), reversed(self.done)):
            R = r + self.gamma * R * (1-d)
            returns.insert(0, R)
        
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.train_device)
        values_tensor = torch.cat(self.values)

        # delta (Advantage)
        delta = returns_tensor - values_tensor
        delta = (delta - delta.mean()) / (delta.std() + 1e-8)

        # Actor loss: policy gradient
        actor_log_probs_tensor = torch.stack(self.action_log_probs).to(self.train_device)
        # Actor loss (policy gradient with advantage)
        actor_loss = -(actor_log_probs_tensor * delta.detach()).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Critic loss: regression to target
        critic_loss = delta.pow(2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        return episode_reward


    def PPO(self, env, maxSteps=10000):
        done = False
        state = env.reset()
        steps = 0
        episode_reward = 0

        ### 1. collect trajectory (experience)
        while not done and steps < maxSteps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.train_device)
            distribution = self.policy(state_tensor)
            action = distribution.sample()
            action_log_prob = distribution.log_prob(action).sum()

            V_s = self.policy.critic_forward(state_tensor)

            next_state, reward, done, _ = env.step(action.cpu().numpy())

            # Store the outcome of the stocastic astion and corresponding reward and value fnunction
            self.states.append(state_tensor)
            self.actions.append(action)
            self.action_log_probs.append(action_log_prob)
            self.rewards.append(reward)
            self.done.append(done)
            self.values.append(V_s)

            state = next_state
            steps += 1
            episode_reward += reward

        ### 2. evaluate the trajectory (experience)
        #   -> we want to calculate 
        #       A_{t} = r + \gamma*V(s+1) - V(s)

        #   2.1. compute gamma*V(s+1)
        returns = []
        V_s_next = self.policy.critic_forward(next_state_tensor).detach()
        for r, d in zip(reversed(self.rewards), reversed(self.done)):
            V_s_next = r + self.gamma * V_s_next * (1-d)
            returns.insert(0, V_s_next)

        #   2.2. compute tensors
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.train_device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.train_device)
        values_tensor = torch.cat(self.values)

        #   2.3. compute advantage and normalize it
        advantage = returns_tensor - values_tensor
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)


        ### 3. Calculate improvement with clipping (Heart of PPO <3)
        

            


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

