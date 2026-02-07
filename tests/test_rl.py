"""Test RL components."""
import pytest
import torch
import numpy as np

from src.rl.dqn_agent import DQNAgent, DQNetwork, ReplayBuffer
from src.rl.environment import FrameSamplingEnv


class TestDQNetwork:
    """Test DQN architecture."""
    
    def test_dqn_forward(self):
        """Test DQN forward pass."""
        network = DQNetwork(
            state_dim=512,
            action_space=300,
            hidden_dims=[256, 128]
        )
        
        # Create dummy state
        state = torch.randn(4, 512)
        
        # Forward pass
        q_values = network(state)
        
        assert q_values.shape == (4, 300)


class TestReplayBuffer:
    """Test replay buffer."""
    
    def test_buffer_push_and_sample(self):
        """Test pushing and sampling from buffer."""
        buffer = ReplayBuffer(capacity=100, prioritized=False)
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(512)
            action = np.random.randint(0, 300)
            reward = np.random.randn()
            next_state = np.random.randn(512)
            done = False
            
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 50
        
        # Sample batch
        batch = buffer.sample(32)
        states, actions, rewards, next_states, dones, _, _ = batch
        
        assert states.shape == (32, 512)
        assert actions.shape == (32,)


class TestDQNAgent:
    """Test DQN agent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        config = {
            'state_dim': 512,
            'action_space': 300,
            'hidden_dims': [256, 128],
            'learning_rate': 0.001,
            'gamma': 0.99
        }
        
        agent = DQNAgent(config, device='cpu')
        
        assert agent.state_dim == 512
        assert agent.action_space == 300
    
    def test_action_selection(self):
        """Test action selection."""
        config = {
            'state_dim': 512,
            'action_space': 300,
            'epsilon_start': 0.0  # Greedy for testing
        }
        
        agent = DQNAgent(config, device='cpu')
        
        state = torch.randn(1, 512)
        action = agent.select_action(state, training=False)
        
        assert isinstance(action, int)
        assert 0 <= action < 300


if __name__ == '__main__':
    pytest.main([__file__])
