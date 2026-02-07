"""Logging utilities."""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import wandb


class ExperimentLogger:
    """Logger for experiment tracking."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = './logs',
        use_wandb: bool = False,
        wandb_project: str = None
    ):
        """
        Initialize logger.
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory for logs
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging
        self.setup_file_logging()
        
        # W&B logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project or "multimodal-video-rl",
                name=experiment_name,
                dir=str(self.log_dir)
            )
        
        # Metrics history
        self.history = {
            'train': [],
            'val': [],
            'test': []
        }
    
    def setup_file_logging(self):
        """Set up file logging."""
        log_file = self.log_dir / 'experiment.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.experiment_name)
    
    def log_config(self, config: Dict):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info("Configuration saved")
        
        if self.use_wandb:
            wandb.config.update(config)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            phase: Phase (train, val, test)
        """
        # Add to history
        metrics_with_step = {'step': step, **metrics}
        self.history[phase].append(metrics_with_step)
        
        # Log to file
        self.logger.info(f"{phase.upper()} - Step {step}: {metrics}")
        
        # Log to W&B
        if self.use_wandb:
            wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)
    
    def log_message(self, message: str, level: str = 'info'):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
    
    def save_history(self):
        """Save metrics history to file."""
        history_file = self.log_dir / 'history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info("History saved")
    
    def finish(self):
        """Finish logging."""
        self.save_history()
        
        if self.use_wandb:
            wandb.finish()
        
        self.logger.info("Experiment complete")


def create_experiment_name(prefix: str = "exp") -> str:
    """
    Create unique experiment name with timestamp.
    
    Args:
        prefix: Prefix for experiment name
        
    Returns:
        Experiment name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter
        
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars."""
        self.writer.add_scalars(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image."""
        self.writer.add_image(tag, image, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close writer."""
        self.writer.close()
