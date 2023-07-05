from memory import Memory
from model import DQN
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Agent:    
    def __init__(self, batch_size=64, memory_size=100000, gamma=1.0, eps_start=1.0, 
                  eps_end=0.01, eps_decay=0.995, tau=0.001, lr=0.001):
        self.batch_size = batch_size
        self.gamma =  gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.t_step = 0    

        self.memory = Memory(memory_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=self.lr)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()
        if np.random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice([i for i in range(4)])

    def step(self, state, action, reward, next_state, terminated):
        self.memory.store(state, action, reward, next_state, terminated)
        
        if len(self.memory.memory) > self.batch_size:
            self.train(2)

    def train(self, method=1):
        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)
        
        # Best q values for each experience of next states
        best_q_value = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        q_current = self.policy_network(states).gather(1, actions)
        q_target = rewards + (self.gamma * best_q_value * (1 - terminals))

        # Calculate loss
        loss = F.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if method == 1:
            torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        if method == 2:
            # Train network
            self.soft_update(self.policy_network, self.target_network)

    def step_soft(self, state, action, reward, next_state, terminated, update_every=4):
        # Save experience in replay memory
        self.memory.store(state, action, reward, next_state, terminated)
        
        # Learn every 4 time steps.
        self.t_step = (self.t_step + 1) % update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory.memory) > self.batch_size:
                self.train(2)                

    def soft_update(self, local_model, target_model, tau=0.001):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, path):
        torch.save(self.policy_network.state_dict(), path)

    def load_model(self, path):
        self.eps = 0
        self.policy_network = DQN().to(self.device)
        self.policy_network.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        self.policy_network.eval()