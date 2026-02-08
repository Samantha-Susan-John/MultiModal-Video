"""Multi-task trainer for video understanding model."""
import os
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


class MultiTaskTrainer:
    """Trainer for multi-task video understanding."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "mps"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Multi-modal video model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training parameters
        training_config = config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 100)
        self.gradient_clip = training_config.get('gradient_clip', 1.0)
        self.mixed_precision = training_config.get('mixed_precision', True)
        
        # Optimizer
        optimizer_config = config.get('optimizer', {})
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Scheduler
        scheduler_config = config.get('scheduler', {})
        self.scheduler = self._create_scheduler(scheduler_config)
        
        # Loss weights
        loss_config = config.get('loss', {})
        self.classification_weight = loss_config.get('classification_weight', 1.0)
        self.captioning_weight = loss_config.get('captioning_weight', 0.5)
        self.sync_weight = loss_config.get('sync_weight', 0.3)
        self.label_smoothing = loss_config.get('label_smoothing', 0.1)
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
        )
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.sync_loss = nn.BCELoss()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Checkpointing
        checkpoint_config = config.get('checkpointing', {})
        self.save_dir = Path(checkpoint_config.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = checkpoint_config.get('save_frequency', 5)
        self.keep_last_n = checkpoint_config.get('keep_last_n', 3)
        
        # Logging
        logging_config = config.get('logging', {})
        self.use_wandb = logging_config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=logging_config.get('wandb_project', 'multimodal-video-rl'),
                config=config
            )
        
        # Metrics
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
    def _create_optimizer(self, config: Dict) -> torch.optim.Optimizer:
        """Create optimizer."""
        name = config.get('name', 'adamw')
        lr = config.get('learning_rate', 0.0001)
        weight_decay = config.get('weight_decay', 0.01)
        
        if name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=config.get('betas', [0.9, 0.999])
            )
        elif name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _create_scheduler(self, config: Dict):
        """Create learning rate scheduler."""
        name = config.get('name', 'cosine')
        
        if name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=config.get('min_lr', 1e-6)
            )
        elif name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        elif name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=config.get('patience', 10),
                factor=config.get('factor', 0.1)
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_caption_loss = 0.0
        total_sync_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            video = batch['video'].to(self.device)
            audio = batch.get('audio', None)
            if audio is not None:
                audio = audio.to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(video, audio)
                    loss = self._compute_loss(outputs, labels, batch)
            else:
                outputs = self.model(video, audio)
                loss = self._compute_loss(outputs, labels, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Classification accuracy
            _, predicted = outputs['class_logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100. * correct / total
        }
        
        return metrics
    
    def _compute_loss(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        batch: Dict
    ) -> torch.Tensor:
        """Compute multi-task loss."""
        losses = {}
        
        # Classification loss
        class_loss = self.classification_loss(outputs['class_logits'], labels)
        losses['classification'] = class_loss * self.classification_weight
        
        # Caption loss (if captions available)
        if 'caption_logits' in outputs and 'captions' in batch:
            captions = batch['captions'].to(self.device)
            caption_logits = outputs['caption_logits']
            # Reshape for loss computation
            caption_loss = self.caption_loss(
                caption_logits.view(-1, caption_logits.size(-1)),
                captions.view(-1)
            )
            losses['captioning'] = caption_loss * self.captioning_weight
        
        # Sync loss (if audio available)
        if 'sync_score' in outputs:
            # Create sync labels (assume aligned by default)
            sync_labels = torch.ones_like(outputs['sync_score'])
            sync_loss = self.sync_loss(outputs['sync_score'], sync_labels)
            losses['sync'] = sync_loss * self.sync_weight
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                video = batch['video'].to(self.device)
                audio = batch.get('audio', None)
                if audio is not None:
                    audio = audio.to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(video, audio)
                loss = self._compute_loss(outputs, labels, batch)
                
                total_loss += loss.item()
                
                _, predicted = outputs['class_logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': 100. * correct / total
        }
        
        return metrics
    
    def train(self):
        """Full training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_acc'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            metrics['epoch'] = epoch
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch}: {metrics}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self.save_checkpoint(epoch, metrics)
            
            # Save best model
            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                self.save_checkpoint(epoch, metrics, is_best=True)
        
        print("Training complete!")
        
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Auto-download in Colab after each epoch
        try:
            import sys
            if 'google.colab' in sys.modules:
                from google.colab import files
                print(f"Downloading checkpoint to your Mac: {path.name}")
                files.download(str(path))
                print(f"✓ Download started for {path.name}")
        except Exception as e:
            # Not in Colab or download failed, skip silently
            pass
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
