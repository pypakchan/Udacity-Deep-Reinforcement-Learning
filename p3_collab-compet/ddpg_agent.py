import numpy as np
import random
from collections import namedtuple, deque
from numpy.random import choice
from scipy.stats import rankdata

from model import criticNet, actorNet
 
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
MAX_TD_DIFF = 1000      # TD diff for new experience, supposed to be very large

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            
            DDPG paper: https://arxiv.org/abs/1509.02971
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # actor network
        self.actor_local  = actorNet(state_size, action_size, seed).to(device)
        self.actor_target = actorNet(state_size, action_size, seed+1).to(device)
        
        # critic network
        self.critic_local  = criticNet(state_size, action_size, seed+2).to(device)
        self.critic_target = criticNet(state_size, action_size, seed+3).to(device)
        
        # initialize the target net to be same as local net
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.noise = 1.
        
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.noise = max( 0.01, self.noise * .99 )
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state)
        self.actor_local.train()

        # Add some noise to the aciton
        actions = actions + self.noise*torch.randn(actions.shape[0],actions.shape[1]).to(device)
        
        # conver to numpy
        actions = actions.cpu().detach().numpy()
        
        # cap and floor to action space per dimension
        actions = np.clip(actions, -1, 1)
        
        # we added one more dimension to the output assuming there is only one agent
        return actions
        

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        
        """ Double DQN
            Use the local network to drive the best next action,
            but use the target network to look up the Q value.

            Paper here: https://arxiv.org/abs/1509.06461
        """
       
        # update critic networks
        Q_expected = self.critic_local(states, actions)
        next_actions   = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()        
        critic_loss.backward()
        self.critic_optimizer.step()
        
        
        # update actor network
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
                
        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        
        Copying the local network parameters to the target network, only after we learnt a batch of data
        This provides stability as the target would not change every step
        
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            sampling_mode (string): "Uniform" to sample experience uniformly; "Ranked" to use ranked replay and importance sampling
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "TD_diff"])
        self.seed = random.seed(seed)
        
        self.choices       = []
        self.probs         = []
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # initialize experience with MAX_TD_DIFF
        e = self.experience(state, action, reward, next_state, done, MAX_TD_DIFF)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
            
        states  = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones   = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)