import numpy as np
import random
from collections import namedtuple, deque
from numpy.random import choice
from scipy.stats import rankdata

from model import QNetwork

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

    def __init__(self, state_size, action_size, seed, ddqn, sampling_mode, dueling):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            ddqn (int): True to use Double DQN; False to use standard (single) DQN
            sampling_mode (string): "Uniform" to sample experience uniformly; "Ranked" to use ranked replay and importance sampling
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.ddqn = ddqn
        self.sampling_mode = sampling_mode

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, dueling).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, sampling_mode)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
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
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())           

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
        
        if self.ddqn:
            """ Double DQN
                Use the local network to drive the best next action,
                but use the target network to look up the Q value.
                
                Paper here: https://arxiv.org/abs/1509.06461
            """
            # get next action using the local network
            next_actions = self.qnetwork_local(next_states).detach().max(1).indices.unsqueeze(1)
            # look up the next Q value using the target Q network
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,next_actions)
        else:    
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)           

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.sampling_mode == "Uniform":
            loss = F.mse_loss(Q_expected, Q_targets)
            self.optimizer.zero_grad()        
            loss.backward()
            self.optimizer.step()
        elif self.sampling_mode == "Ranked":
            # TO DO: add a beta schedule here! it should be increasing from initial value to 1
            beta = 0
            IS_weights = self.memory.get_importance_sampling_weights(beta)
            # scale the loss to compensate for importance sampling            
            TD_diff = Q_expected - Q_targets
            weighted_loss = torch.mean(IS_weights * IS_weights * TD_diff * TD_diff)
            
            self.optimizer.zero_grad()        
            weighted_loss.backward()
            self.optimizer.step()
            
            self.memory.update_TD_diff(TD_diff.cpu().detach().numpy())
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

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

    def __init__(self, action_size, buffer_size, batch_size, seed, sampling_mode):
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
        
        self.sampling_mode = sampling_mode
        self.choices       = []
        self.probs         = []
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # initialize experience with MAX_TD_DIFF
        e = self.experience(state, action, reward, next_state, done, MAX_TD_DIFF)
        self.memory.append(e)
    
    def sample_uniform(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
            
        states  = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones   = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
  
    def sample_ranked(self):
        """Ranked sampling a batch of experiences from memory"""
        # rankdata ranks in ascending order, that's why we multiply by -1 here to get descending ranking
        # TO DO: check if this ranking logic works
        # TO DO: check the None cases does it actually work?
        # remove None, only happens at initial phase
        
        alpha     = 0.5
        memory    = [e for e in self.memory if e is not None]
        ranks     = rankdata([-e.TD_diff for e in self.memory if e is not None])
        all_probs = [pow(1./k, alpha) for k in ranks]
        #all_probs = [1. for k in ranks]
        all_probs = all_probs / np.sum(all_probs)
        choices   = choice(len(memory), self.batch_size, p=all_probs, replace=False)
        experiences = [memory[k] for k in choices]
            
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        
        self.choices = choices
        self.probs   = [all_probs[k] for k in choices]
                                 
        return (states, actions, rewards, next_states, dones)
                                        
    def update_TD_diff(self, diffs):             
        for i, choice in enumerate(self.choices):
            exp = self.memory[choice]
            self.memory[choice] = self.experience(exp[0], exp[1], exp[2], exp[3], exp[4], abs(diffs[i]))
        
    def get_importance_sampling_weights(self, beta):
        weights = [pow( p * self.buffer_size, -beta) for p in self.probs]
        weights = weights / np.max(weights)
        return torch.Tensor(weights).float().to(device)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.sampling_mode == "Uniform":
            return self.sample_uniform()
        elif self.sampling_mode == "Ranked":
            return self.sample_ranked()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)