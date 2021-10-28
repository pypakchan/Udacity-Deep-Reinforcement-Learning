import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dueling):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling
            
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        
        if dueling:
            self.value_fc1  = nn.Linear(128, 64)
            self.value_fc2  = nn.Linear(64, 1)
            self.action_fc1 = nn.Linear(128, 64)
            self.action_fc2 = nn.Linear(64, action_size)
        else:
            self.fc4 = nn.Linear(128, 64)
            self.fc5 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        
        if self.dueling:
            # better to have two layers here so that we can insert a relu for some non-linearity
            v = self.value_fc2(F.relu(self.value_fc1(x)))
            a = self.action_fc2(F.relu(self.action_fc1(x)))
            out = v + a - torch.mean(a, dim=1).unsqueeze(1)
        else:
            out = self.fc5(F.relu(self.fc4(x)))
        return out
