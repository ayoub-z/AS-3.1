import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    def __init__(self, n_states=8, n_actions=4):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        actions = self.layer3(x)
        return actions