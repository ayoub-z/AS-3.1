from collections import deque, namedtuple
import numpy as np
from random import sample
import torch

class Memory:
    """
    Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)
        self.transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def store(self, *args):
        # exp = self.transition(state, action, reward, next_state, terminated)
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        assert len(self.memory) >= batch_size
        transitions = sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in transitions if e is not None])).float().to(self.device)
        terminated = torch.from_numpy(np.vstack([e.terminated for e in transitions if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, terminated)