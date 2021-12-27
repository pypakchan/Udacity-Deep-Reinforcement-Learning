import torch
import torch.nn as nn
import torch.nn.functional as F

class actorNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_dim, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(actorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
            
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_dim)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = F.relu( self.fc4(x) )
        
        # action is between -1 and 1
        x = torch.tanh(self.fc5(x))
        
        return x


class criticNet(nn.Module):
    """Critic (Q) Model."""

    def __init__(self, state_size, action_dim, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(criticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
            
        self.bn = nn.BatchNorm1d(64)
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(action_dim+64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1) 

    def forward(self, state, action):
        """Build a network that maps state + action to Q value."""
        
        # we add an "embedding" layer for the state input before we concatenate with the action input (next step)
        # this allows the network to learn how best to combine the two inputs
        x = F.relu( self.bn( self.fc1(state) ) )
        
        x = torch.cat((x, action), dim=1)
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = F.relu( self.fc4(x) )
        
        out = torch.sigmoid(self.fc5(x))

        return out