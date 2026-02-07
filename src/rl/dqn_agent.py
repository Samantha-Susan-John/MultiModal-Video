"""Deep Q-Network (DQN) agent for frame sampling."""
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class DQNetwork(nn.Module):
    """Deep Q-Network for frame sampling decisions."""
    
    def __init__(
        self,
        state_dim: int,
        action_space: int,
        hidden_dims: List[int] = [256, 128]
    ):
        """
        Initialize DQN.
        
        Args:
            state_dim: State feature dimension
            action_space: Number of possible actions (frame indices)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output Q-values for each action
        layers.append(nn.Linear(prev_dim, action_space))
        
        self.network = nn.Sequential(*layers)
        
        # Dueling DQN architecture (optional)
        self.use_dueling = False
        if self.use_dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dims[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dims[-1], 128),
                nn.ReLU(),
                nn.Linear(128, action_space)
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (B, state_dim)
            
        Returns:
            Q-values for each action (B, action_space)
        """
        if self.use_dueling:
            features = self.network[:-1](state)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.network(state)
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(
        self,
        capacity: int = 10000,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta_start: float = 0.4
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            prioritized: Whether to use prioritized experience replay
            alpha: Prioritization exponent
            beta_start: Initial importance sampling weight
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta_start
        
        if prioritized:
            self.priorities = deque(maxlen=capacity)
            self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        if self.prioritized:
            self.priorities.append(self.max_priority)
    
    def sample(
        self,
        batch_size: int,
        beta: Optional[float] = None
    ) -> Tuple:
        """
        Sample batch of experiences.
        
        Args:
            batch_size: Batch size
            beta: Importance sampling weight (for prioritized replay)
            
        Returns:
            Tuple of batched experiences and optional importance weights
        """
        if self.prioritized:
            return self._prioritized_sample(batch_size, beta or self.beta)
        else:
            return self._uniform_sample(batch_size)
    
    def _uniform_sample(self, batch_size: int) -> Tuple:
        """Sample uniformly from buffer."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
            None,  # No importance weights
            None   # No indices
        )
    
    def _prioritized_sample(
        self,
        batch_size: int,
        beta: float
    ) -> Tuple:
        """Sample with priorities."""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
            weights.astype(np.float32),
            indices
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)


class DQNAgent:
    """DQN agent for optimal frame sampling."""
    
    def __init__(self, config: Dict, device: str = "mps"):
        """
        Initialize DQN agent.
        
        Args:
            config: Agent configuration
            device: Device to run on
        """
        self.config = config
        self.device = device
        
        # Extract config
        self.state_dim = config.get('state_dim', 512)
        self.action_space = config.get('action_space', 300)
        hidden_dims = config.get('hidden_dims', [256, 128])
        
        # Create networks
        self.policy_net = DQNetwork(
            self.state_dim,
            self.action_space,
            hidden_dims
        ).to(device)
        
        self.target_net = DQNetwork(
            self.state_dim,
            self.action_space,
            hidden_dims
        ).to(device)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        lr = config.get('learning_rate', 0.0001)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        buffer_config = config.get('replay_buffer', {})
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_config.get('capacity', 10000),
            prioritized=buffer_config.get('prioritized', True),
            alpha=buffer_config.get('alpha', 0.6),
            beta_start=buffer_config.get('beta_start', 0.4)
        )
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update_freq = config.get('target_update_frequency', 10)
        
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(
        self,
        state: torch.Tensor,
        valid_actions: Optional[List[int]] = None,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (1, state_dim)
            valid_actions: List of valid action indices
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random exploration
            if valid_actions:
                return random.choice(valid_actions)
            else:
                return random.randint(0, self.action_space - 1)
        
        # Greedy action selection
        with torch.no_grad():
            q_values = self.policy_net(state)
            
            # Mask invalid actions if provided
            if valid_actions:
                mask = torch.ones(self.action_space, device=self.device) * float('-inf')
                mask[valid_actions] = 0
                q_values = q_values + mask
            
            action = q_values.max(1)[1].item()
        
        return action
    
    def train_step(self, batch_size: int) -> float:
        """
        Perform one training step.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        beta = self.replay_buffer.beta + self.steps_done * (1.0 - self.replay_buffer.beta) / 100000
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(
            batch_size, beta
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)
            loss = (weights * (current_q_values.squeeze() - target_q_values) ** 2).mean()
            
            # Update priorities
            with torch.no_grad():
                td_errors = torch.abs(current_q_values.squeeze() - target_q_values).cpu().numpy()
                self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        else:
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
