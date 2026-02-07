"""Frame sampling environment for RL agent."""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class FrameSamplingEnv:
    """Environment for learning optimal frame sampling."""
    
    def __init__(
        self,
        video_model,
        max_frames: int = 300,
        target_frames: int = 32,
        min_frames: int = 8,
        reward_config: Optional[Dict] = None
    ):
        """
        Initialize frame sampling environment.
        
        Args:
            video_model: Pretrained video understanding model
            max_frames: Maximum frames available
            target_frames: Target number of frames to sample
            min_frames: Minimum frames required
            reward_config: Reward function configuration
        """
        self.video_model = video_model
        self.max_frames = max_frames
        self.target_frames = target_frames
        self.min_frames = min_frames
        
        # Reward configuration
        reward_config = reward_config or {}
        self.accuracy_weight = reward_config.get('accuracy_weight', 1.0)
        self.efficiency_weight = reward_config.get('efficiency_weight', 0.3)
        self.diversity_weight = reward_config.get('diversity_weight', 0.2)
        
        # State
        self.current_video = None
        self.current_label = None
        self.selected_frames = []
        self.available_frames = None
        self.frame_features = None
        
    def reset(
        self,
        video_frames: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment with new video.
        
        Args:
            video_frames: Full video frames (T, C, H, W)
            label: Ground truth label
            
        Returns:
            Initial state and info dict
        """
        self.current_video = video_frames
        self.current_label = label
        self.selected_frames = []
        self.available_frames = list(range(len(video_frames)))
        
        # Extract frame features using video model
        with torch.no_grad():
            # Add batch dimension
            video_batch = video_frames.unsqueeze(0)
            features = self.video_model.vision_encoder(video_batch)
            self.frame_features = features.squeeze(0)  # (T, D)
        
        # Initial state is mean of all frame features
        initial_state = torch.mean(self.frame_features, dim=0).cpu().numpy()
        
        info = {
            'total_frames': len(video_frames),
            'target_frames': self.target_frames
        }
        
        return initial_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step by selecting a frame.
        
        Args:
            action: Frame index to select
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Add selected frame
        if action in self.available_frames:
            self.selected_frames.append(action)
            self.available_frames.remove(action)
        
        # Compute next state
        if len(self.selected_frames) > 0:
            selected_features = self.frame_features[self.selected_frames]
            current_state = torch.mean(selected_features, dim=0)
            
            # Concatenate with history statistics
            state_features = current_state.cpu().numpy()
        else:
            state_features = torch.mean(self.frame_features, dim=0).cpu().numpy()
        
        # Check if done
        done = len(self.selected_frames) >= self.target_frames
        
        # Compute reward
        reward = 0.0
        info = {}
        
        if done or len(self.selected_frames) >= self.min_frames:
            # Evaluate sampled frames
            reward, info = self._compute_reward()
        
        return state_features, reward, done, info
    
    def _compute_reward(self) -> Tuple[float, Dict]:
        """
        Compute reward based on sampled frames.
        
        Returns:
            Reward value and info dict
        """
        if len(self.selected_frames) == 0:
            return -1.0, {'accuracy': 0.0, 'efficiency': 0.0}
        
        # Get sampled frames
        sampled_indices = sorted(self.selected_frames)
        sampled_frames = self.current_video[sampled_indices]
        
        # Evaluate with video model
        with torch.no_grad():
            # Add batch dimension
            sampled_batch = sampled_frames.unsqueeze(0)
            outputs = self.video_model(video=sampled_batch, audio=None)
            
            # Get prediction
            logits = outputs['class_logits']
            pred = torch.argmax(logits, dim=-1)
            
            # Accuracy reward
            correct = (pred == self.current_label).float().item()
            accuracy_reward = correct * self.accuracy_weight
        
        # Efficiency reward (using fewer frames is better)
        efficiency = 1.0 - (len(self.selected_frames) / self.target_frames)
        efficiency_reward = efficiency * self.efficiency_weight
        
        # Diversity reward (encourage frames spread across video)
        diversity = self._compute_diversity(sampled_indices)
        diversity_reward = diversity * self.diversity_weight
        
        total_reward = accuracy_reward + efficiency_reward + diversity_reward
        
        info = {
            'accuracy': correct,
            'efficiency': efficiency,
            'diversity': diversity,
            'num_frames': len(self.selected_frames),
            'sampled_indices': sampled_indices
        }
        
        return total_reward, info
    
    def _compute_diversity(self, indices: List[int]) -> float:
        """
        Compute diversity score for sampled frames.
        
        Args:
            indices: List of frame indices
            
        Returns:
            Diversity score in [0, 1]
        """
        if len(indices) <= 1:
            return 0.0
        
        # Compute temporal spread
        indices = sorted(indices)
        gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        
        # Ideal gap for uniform sampling
        ideal_gap = self.max_frames / len(indices)
        
        # Measure deviation from ideal
        gap_variance = np.var(gaps) if len(gaps) > 0 else 0
        diversity_score = 1.0 / (1.0 + gap_variance / (ideal_gap ** 2))
        
        return diversity_score
    
    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions (available frames).
        
        Returns:
            List of valid frame indices
        """
        return self.available_frames.copy()


class RLTrainer:
    """Trainer for RL-based frame sampling."""
    
    def __init__(
        self,
        agent,
        env: FrameSamplingEnv,
        config: Dict
    ):
        """
        Initialize RL trainer.
        
        Args:
            agent: DQN agent
            env: Frame sampling environment
            config: Training configuration
        """
        self.agent = agent
        self.env = env
        self.config = config
        
        self.num_episodes = config.get('num_episodes', 1000)
        self.batch_size = config.get('batch_size', 64)
        self.warmup_episodes = config.get('warmup_episodes', 50)
        self.eval_frequency = config.get('eval_frequency', 10)
        
    def train_episode(
        self,
        video: torch.Tensor,
        label: torch.Tensor
    ) -> Dict:
        """
        Train on one video episode.
        
        Args:
            video: Video frames
            label: Ground truth label
            
        Returns:
            Episode statistics
        """
        state, info = self.env.reset(video, label)
        
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0
        
        done = False
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            
            # Get valid actions
            valid_actions = self.env.get_valid_actions()
            
            # Select action
            action = self.agent.select_action(
                state_tensor,
                valid_actions=valid_actions,
                training=True
            )
            
            # Take step
            next_state, reward, done, step_info = self.env.step(action)
            
            # Store transition
            self.agent.replay_buffer.push(
                state, action, reward, next_state, done
            )
            
            # Train if enough samples
            if len(self.agent.replay_buffer) >= self.batch_size:
                loss = self.agent.train_step(self.batch_size)
                episode_loss += loss
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Prevent infinite loops
            if steps >= self.env.target_frames:
                break
        
        # Episode statistics
        stats = {
            'reward': episode_reward,
            'loss': episode_loss / max(steps, 1),
            'steps': steps,
            'epsilon': self.agent.epsilon
        }
        stats.update(step_info)
        
        return stats
    
    def evaluate(
        self,
        eval_videos: List[torch.Tensor],
        eval_labels: List[torch.Tensor]
    ) -> Dict:
        """
        Evaluate agent on validation set.
        
        Args:
            eval_videos: List of validation videos
            eval_labels: List of validation labels
            
        Returns:
            Evaluation metrics
        """
        total_accuracy = 0.0
        total_efficiency = 0.0
        total_diversity = 0.0
        
        for video, label in zip(eval_videos, eval_labels):
            state, _ = self.env.reset(video, label)
            done = False
            steps = 0
            
            while not done and steps < self.env.target_frames:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                valid_actions = self.env.get_valid_actions()
                
                # Greedy action selection
                action = self.agent.select_action(
                    state_tensor,
                    valid_actions=valid_actions,
                    training=False
                )
                
                state, reward, done, info = self.env.step(action)
                steps += 1
            
            total_accuracy += info.get('accuracy', 0.0)
            total_efficiency += info.get('efficiency', 0.0)
            total_diversity += info.get('diversity', 0.0)
        
        n = len(eval_videos)
        return {
            'accuracy': total_accuracy / n,
            'efficiency': total_efficiency / n,
            'diversity': total_diversity / n
        }
