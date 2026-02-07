"""Knowledge distillation for model compression."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm


class KnowledgeDistillation:
    """Knowledge distillation trainer."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Dict,
        device: str = "mps"
    ):
        """
        Initialize knowledge distillation.
        
        Args:
            teacher_model: Pretrained teacher model
            student_model: Student model to train
            config: Distillation configuration
            device: Device to train on
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Distillation parameters
        self.temperature = config.get('temperature', 4.0)
        self.alpha = config.get('alpha', 0.7)  # Weight for distillation loss
        
        # Optimizer
        lr = config.get('learning_rate', 0.0001)
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=lr
        )
        
        # Losses
        self.hard_loss = nn.CrossEntropyLoss()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            
        Returns:
            Combined loss
        """
        # Soft targets loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        soft_targets_loss = F.kl_div(
            soft_prob,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard targets loss
        hard_targets_loss = self.hard_loss(student_logits, labels)
        
        # Combined loss
        loss = (
            self.alpha * soft_targets_loss +
            (1 - self.alpha) * hard_targets_loss
        )
        
        return loss
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train student for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics
        """
        self.student.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Distillation Training")
        
        for batch in pbar:
            video = batch['video'].to(self.device)
            audio = batch.get('audio', None)
            if audio is not None:
                audio = audio.to(self.device)
            labels = batch['label'].to(self.device)
            
            # Teacher predictions
            with torch.no_grad():
                teacher_outputs = self.teacher(video, audio)
                teacher_logits = teacher_outputs['class_logits']
            
            # Student predictions
            student_outputs = self.student(video, audio)
            student_logits = student_outputs['class_logits']
            
            # Compute loss
            loss = self.distillation_loss(
                student_logits,
                teacher_logits,
                labels
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (total / labels.size(0)),
                'acc': 100. * correct / total
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate student model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.student.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                video = batch['video'].to(self.device)
                audio = batch.get('audio', None)
                if audio is not None:
                    audio = audio.to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.student(video, audio)
                _, predicted = outputs['class_logits'].max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'accuracy': 100. * correct / total
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50
    ) -> Dict:
        """
        Full distillation training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(
                    self.student.state_dict(),
                    'best_student_model.pth'
                )
                print(f"Saved best student model (acc: {best_val_acc:.2f}%)")
        
        print(f"\nDistillation complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        return history


def create_student_model(teacher_model, scale_factor: float = 0.5):
    """
    Create a smaller student model based on teacher architecture.
    
    Args:
        teacher_model: Teacher model
        scale_factor: Size reduction factor
        
    Returns:
        Student model with reduced capacity
    """
    # This is a placeholder - would need to be implemented based on
    # specific model architecture
    
    # Example: reduce hidden dimensions
    from ..models import MultiModalVideoModel
    
    teacher_config = teacher_model.config
    student_config = teacher_config.copy()
    
    # Scale down dimensions
    for key in ['vision_encoder', 'audio_encoder', 'temporal_encoder', 'fusion']:
        if key in student_config:
            if 'hidden_dim' in student_config[key]:
                student_config[key]['hidden_dim'] = int(
                    student_config[key]['hidden_dim'] * scale_factor
                )
            if 'output_dim' in student_config[key]:
                student_config[key]['output_dim'] = int(
                    student_config[key]['output_dim'] * scale_factor
                )
            if 'num_layers' in student_config[key]:
                student_config[key]['num_layers'] = max(
                    1,
                    int(student_config[key]['num_layers'] * scale_factor)
                )
    
    student_model = MultiModalVideoModel(student_config)
    
    return student_model
