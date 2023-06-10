import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, num_agent):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear((state_dim + action_dim)*num_agent, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear((state_dim + action_dim)*num_agent, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class Agent():
	def __init__(self, state_dim, action_dim, max_action, num_agent, name):
		self.name = name
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.critic = Critic(state_dim, action_dim, num_agent).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.critic_target = copy.deepcopy(self.critic)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		num_adversaries = 1
		num_agent = 2
		self.eps = 0.1
		self.adv_eps = 0.1
		self.agents = []
		for i in range(num_adversaries):
			self.agents.append(Agent(state_dim, action_dim, max_action, num_agent, f'good_{i}'))
		for i in range(num_adversaries, num_agent):
			self.agents.append(Agent(state_dim, action_dim, max_action, num_agent, f'bad_{i}'))
 
		self.state_dim = state_dim
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

	def sample_action(self, states):
		actions = []
		for i, agent in enumerate(self.agents):
			state = states[i] 
			action = agent.select_action(state).tolist()
			actions.append(action)
		actions = np.array(actions)
		return actions
	


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state_n, action_n, next_state_n, reward_n, not_done_n = replay_buffer.sample(batch_size)

		# state_n # batch_size, num_agent, state_dim
		# next_state_n  # batch_size, num_agent, state_dim
		# action_n # batch_size, num_agent, action_dim
		

		for i, agent in enumerate(self.agents):
			if 'good' in agent.name:
				eps = self.eps
			else:
				eps = self.adv_eps
		
			reward_batch = reward_n[:, i]
			not_done_batch = not_done_n[:,i]
 

			_next_actions = [self.agents[j].actor(next_state_n[:, j, :]).requires_grad_().unsqueeze(0) for j in range(len(self.agents))] # [num_agent, batch_size, action_size] 

			_next_action_n_batch_critic = torch.cat([_next_action if j != i else _next_action   for j, _next_action in enumerate(_next_actions)],axis=0).transpose(0, 1) #  [batch_size, num_agent,  action_size]
			_critic_target1, _critic_target2 = self.agents[i].critic_target(next_state_n.reshape(batch_size, -1), _next_action_n_batch_critic.reshape(batch_size, -1))

			_critic_target_loss =  _critic_target1.mean() + _critic_target2.mean()
			for _next_action in _next_actions:
				_next_action.retain_grad()
			_critic_target_loss.backward()
			next_action_n_batch = torch.cat([_next_action - eps * torch.nn.functional.normalize(_next_action.grad) if j != i else _next_action for j, _next_action in enumerate(_next_actions)], axis=0).transpose(0, 1)

			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action_n) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)


				next_action_n = (
					next_action_n_batch + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.agents[i].critic_target(next_state_n.reshape(batch_size, -1), next_action_n.reshape(batch_size, -1))
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward_batch + not_done_batch * self.discount * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.agents[i].critic(state_n.reshape(batch_size, -1), action_n.reshape(batch_size, -1))

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.agents[i].critic_optimizer.zero_grad()
			critic_loss.backward()
			self.agents[i].critic_optimizer.step()
		
 
			# Delayed policy updates
			if self.total_it % self.policy_freq == 0:
				_actions = [self.agents[j].actor(state_n[:, j, :]).requires_grad_().unsqueeze(0) for j in range(len(self.agents))] # [num_agent, batch_size, action_size] 
				_action_n_batch_actor = torch.cat([_action if j != i else _action  for j, _action in enumerate(_actions)],axis=0).transpose(0, 1) #  [batch_size, num_agent,  action_size]
				_actor_target1, _actor_target2  =   self.agents[i].critic(state_n.reshape(batch_size, -1), _action_n_batch_actor.reshape(batch_size, -1)) 
				_actor_target_loss =   _actor_target1.mean()  + _actor_target2.mean()
				for _action in _actions:
					_action.retain_grad()
				_actor_target_loss.backward(retain_graph=True)
				action_n_batch_actor = torch.cat([_action - eps * torch.nn.functional.normalize(_action.grad) if j != i else _action for j, _action in enumerate(_actions)], axis=0).transpose(0, 1) #  [batch_size, num_agent,  action_size]

				# Compute actor losse
				actor_loss = -self.agents[i].critic.Q1(state_n.reshape(batch_size, -1), action_n_batch_actor.reshape(batch_size, -1)).mean() 

				# Optimize the actor 
				self.agents[i].actor_optimizer.zero_grad()
				actor_loss.backward()
				self.agents[i].actor_optimizer.step()
 

				# Update the frozen target models
				for param, target_param in zip(self.agents[i].critic.parameters(), self.agents[i].critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.agents[i].actor.parameters(), self.agents[i].actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		