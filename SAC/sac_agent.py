import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class ReplayMemory:
    def __init__(self, memo_capacity, state_dim, action_dim):
        self.memory_size = memo_capacity
        self.state_memo = np.zeros((self.memory_size, state_dim))
        self.next_state_memo = np.zeros((self.memory_size, state_dim))
        self.action_memo = np.zeros((self.memory_size, action_dim))
        self.reward_memo = np.zeros(self.memory_size)
        self.done_memo = np.zeros(self.memory_size)
        self.memo_counter = 0

    def add_memory(self, state, action, reward, next_state, done):
        index = self.memo_counter % self.memory_size
        self.state_memo[index] = state
        self.next_state_memo[index] = next_state
        self.action_memo[index] = action
        self.reward_memo[index] = reward
        self.done_memo[index] = done

        self.memo_counter += 1
        
    def sample_memory(self, batch_size):
        current_memo_size = np.minimum(self.memory_size, self.memo_counter)
        index = np.random.choice(current_memo_size, batch_size, replace=False)
        batch_state = self.state_memo[index]
        batch_next_state = self.next_state_memo[index]
        batch_action = self.action_memo[index]
        batch_reward = self.reward_memo[index]
        batch_done = self.done_memo[index]

        return batch_state,batch_next_state,batch_action,batch_reward,batch_done


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        
        self.fc1 = nn.Linear(self.state_dim + self.action_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.q = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim = 1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q
    
class ValueNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        
        self.fc1 = nn.Linear(self.state_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.v = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = beta)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)

        return v
    
class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.max_action = max_action

        self.fc1 = nn.Linear(self.state_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        
        self.mu = nn.Linear(self.fc2_dim, self.action_dim)
        self.sigma = nn.Linear(self.fc2_dim, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.tiny_positive = 1e-6

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x)) * self.max_action

        sigma = self.sigma(x)
        sigma = F.softplus(sigma) + self.tiny_positive
        sigma = torch.clamp(sigma, min=self.tiny_positive, max=1.0)

        return mu, sigma
    
    def sample_normal(self, state, reparameterize):
        mu, sigma = self.forward(state)
        probability = Normal(mu, sigma)

        if reparameterize:
            raw_action = probability.rsample()
        else:
            raw_action = probability.sample()

        tanh_action = torch.tanh(raw_action)
        scaled_action = tanh_action * self.max_action
        log_prob = probability.log_prob(raw_action)
        log_prob -= torch.log(1-tanh_action.pow(2) + self.tiny_positive)

        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(0)
        log_prob = log_prob.sum(1, keepdim=True)

        return scaled_action, log_prob


class SACAgent():
    def __init__(self, beta, gamma, tau, alpha, state_dim, action_dim, 
                 layer1_dim, layer2_dim, memo_capacity, batch_size):
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memo_capacity = memo_capacity

        self.CriticNetwork1 = CriticNetwork(beta,state_dim,action_dim,
                                            layer1_dim,layer2_dim).to(device)
        self.CriticNetwork2 = CriticNetwork(beta,state_dim,action_dim,
                                            layer1_dim,layer2_dim).to(device)
        self.ActionNetwork = ActorNetwork(alpha, state_dim, action_dim, 
                                          layer1_dim, layer2_dim, action_dim).to(device)
        self.ValueNetwork = ValueNetwork(beta, state_dim, action_dim, 
                                         layer1_dim, layer2_dim).to(device)
        self.TargetValueNetwork = ValueNetwork(beta, state_dim, action_dim, 
                                         layer1_dim, layer2_dim).to(device)
        self.ReplayMemory = ReplayMemory(memo_capacity, state_dim, action_dim)

    def get_action(self, state):
        
        state = torch.tensor(state, dtype= torch.float).to(device)
        action, _ = self.ActionNetwork.sample_normal(state, reparameterize=False)
        return action.cpu().detach().numpy()
    
    def add_memory(self, state, action, reward, next_state, done):
        self.ReplayMemory.add_memory(state, action, reward, next_state, done)

    def update(self):
        if self.ReplayMemory.memo_counter < self.batch_size:
            return 
        
        state, next_state, action_batch, reward, done = self.ReplayMemory.sample_memory(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action_batch = torch.tensor(action_batch, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device).view(-1, 1)
        done = torch.tensor(done, dtype=torch.float).to(device).view(-1, 1)

        # value function update
        new_actions, log_prob = self.ActionNetwork.sample_normal(state, reparameterize=True)
        value = self.ValueNetwork.forward(state)

        with torch.no_grad():
            q1_new_policy = self.CriticNetwork1.forward(state, new_actions)
            q2_new_policy = self.CriticNetwork2.forward(state, new_actions)
            critical_value = torch.min(q1_new_policy, q2_new_policy)
            v_target = critical_value - log_prob
            
        v_loss = 0.5 * F.mse_loss(value, v_target.detach())
        self.ValueNetwork.optimizer.zero_grad()
        v_loss.backward()
        self.ValueNetwork.optimizer.step()

        #TargetValueNetwork update
        for target_param, param in zip(self.TargetValueNetwork.parameters(), self.ValueNetwork.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ActionNetwork update
        new_actions_2, log_prob = self.ActionNetwork.sample_normal(state, reparameterize=True)
        q1_new_policy = self.CriticNetwork1.forward(state, new_actions_2)
        q2_new_policy = self.CriticNetwork2.forward(state, new_actions_2)
        critical_value = torch.min(q1_new_policy, q2_new_policy)
 
        Actor_loss = (log_prob - critical_value).mean() #todo
        self.ActionNetwork.optimizer.zero_grad()
        Actor_loss.backward()
        self.ActionNetwork.optimizer.step()

        #QNetwork update
        with torch.no_grad():
            value_next = self.TargetValueNetwork.forward(next_state)
            q_target = reward + self.gamma * (1 - done) * value_next

        q1_pred = self.CriticNetwork1.forward(state, action_batch)
        q2_pred = self.CriticNetwork2.forward(state, action_batch)

        q1_loss = 0.5 * F.mse_loss(q1_pred.view(-1), q_target.view(-1))
        q2_loss = 0.5 * F.mse_loss(q2_pred.view(-1), q_target.view(-1))

        self.CriticNetwork1.optimizer.zero_grad()
        self.CriticNetwork2.optimizer.zero_grad()
        q_loss = q1_loss + q2_loss
        q_loss.backward()
        self.CriticNetwork1.optimizer.step()
        self.CriticNetwork2.optimizer.step()
