"""Visualization utilities."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Optional
from pathlib import Path


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot training curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if 'train' in history and len(history['train']) > 0:
        train_steps = [h['step'] for h in history['train']]
        train_losses = [h.get('loss', 0) for h in history['train']]
        
        axes[0, 0].plot(train_steps, train_losses, label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Validation loss
    if 'val' in history and len(history['val']) > 0:
        val_steps = [h['step'] for h in history['val']]
        val_losses = [h.get('loss', 0) for h in history['val']]
        
        axes[0, 1].plot(val_steps, val_losses, label='Val Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Accuracies
    if 'train' in history and len(history['train']) > 0:
        train_acc = [h.get('accuracy', 0) for h in history['train']]
        axes[1, 0].plot(train_steps, train_acc, label='Train Acc')
    
    if 'val' in history and len(history['val']) > 0:
        val_acc = [h.get('accuracy', 0) for h in history['val']]
        axes[1, 0].plot(val_steps, val_acc, label='Val Acc', color='orange')
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    if 'train' in history and len(history['train']) > 0:
        lrs = [h.get('lr', 0) for h in history['train']]
        axes[1, 1].plot(train_steps, lrs)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=False,
        cmap='Blues',
        xticklabels=class_names if class_names else [],
        yticklabels=class_names if class_names else [],
        vmin=0,
        vmax=1
    )
    
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_frame_sampling(
    video_frames: np.ndarray,
    selected_indices: List[int],
    save_path: Optional[str] = None
):
    """
    Visualize selected frames from video.
    
    Args:
        video_frames: Video frames array (T, H, W, C)
        selected_indices: Indices of selected frames
        save_path: Path to save figure
    """
    num_frames = len(selected_indices)
    cols = min(8, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        row = i // cols
        col = i % cols
        
        frame = video_frames[idx]
        
        # Normalize if needed
        if frame.max() > 1:
            frame = frame / 255.0
        
        axes[row, col].imshow(frame)
        axes[row, col].set_title(f'Frame {idx}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_frames, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Attention weight matrix
        save_path: Path to save figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar=True
    )
    
    plt.title('Attention Weights')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(
    results: List[dict],
    metric: str = 'accuracy',
    save_path: Optional[str] = None
):
    """
    Plot comparison of different models.
    
    Args:
        results: List of result dictionaries
        metric: Metric to compare
        save_path: Path to save figure
    """
    model_names = [r['name'] for r in results]
    values = [r[metric] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(model_names, values)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{value:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_rl_training(
    episode_rewards: List[float],
    episode_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Plot RL training progress.
    
    Args:
        episode_rewards: List of episode rewards
        episode_losses: List of episode losses
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Rewards
    axes[0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    
    # Moving average
    if len(episode_rewards) >= 10:
        window = 10
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        axes[0].plot(
            range(window - 1, len(episode_rewards)),
            moving_avg,
            label='Moving Average',
            linewidth=2
        )
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('RL Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Losses
    axes[1].plot(episode_losses, alpha=0.6, label='Episode Loss')
    
    if len(episode_losses) >= 10:
        moving_avg_loss = np.convolve(
            episode_losses,
            np.ones(window) / window,
            mode='valid'
        )
        axes[1].plot(
            range(window - 1, len(episode_losses)),
            moving_avg_loss,
            label='Moving Average',
            linewidth=2
        )
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('RL Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
