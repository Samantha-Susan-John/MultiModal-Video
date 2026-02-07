"""Model pruning utilities."""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple


class ModelPruner:
    """Neural network pruning for model compression."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        """
        Initialize pruner.
        
        Args:
            model: Model to prune
            config: Pruning configuration
        """
        self.model = model
        self.config = config
        
        self.method = config.get('method', 'magnitude')
        self.target_sparsity = config.get('target_sparsity', 0.5)
        self.schedule = config.get('schedule', 'gradual')
        self.num_iterations = config.get('num_iterations', 10)
        
        # Layers to prune
        self.layers_to_prune = config.get('layers_to_prune', [])
        self.layer_wise_sparsity = config.get('layer_wise_sparsity', {})
        
    def prune_model(self) -> Dict[str, float]:
        """
        Prune the model.
        
        Returns:
            Dictionary with pruning statistics
        """
        if self.method == 'magnitude':
            return self._magnitude_pruning()
        elif self.method == 'structured':
            return self._structured_pruning()
        elif self.method == 'lottery_ticket':
            return self._lottery_ticket_pruning()
        else:
            raise ValueError(f"Unknown pruning method: {self.method}")
    
    def _magnitude_pruning(self) -> Dict[str, float]:
        """Magnitude-based pruning."""
        parameters_to_prune = []
        
        # Collect parameters to prune
        for name, module in self.model.named_modules():
            if self._should_prune_layer(name, module):
                if isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, 'weight'))
                elif isinstance(module, nn.Conv2d):
                    parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        if self.schedule == 'one_shot':
            # Prune all at once
            for module, param_name in parameters_to_prune:
                sparsity = self._get_layer_sparsity(module)
                prune.l1_unstructured(module, param_name, amount=sparsity)
        
        elif self.schedule == 'gradual':
            # Gradual pruning
            current_sparsity = 0.0
            sparsity_increment = self.target_sparsity / self.num_iterations
            
            for iteration in range(self.num_iterations):
                current_sparsity = min(
                    current_sparsity + sparsity_increment,
                    self.target_sparsity
                )
                
                for module, param_name in parameters_to_prune:
                    prune.l1_unstructured(
                        module,
                        param_name,
                        amount=current_sparsity
                    )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Calculate statistics
        stats = self._compute_pruning_stats()
        
        return stats
    
    def _structured_pruning(self) -> Dict[str, float]:
        """Structured pruning (prune entire neurons/filters)."""
        parameters_to_prune = []
        
        for name, module in self.model.named_modules():
            if self._should_prune_layer(name, module):
                if isinstance(module, nn.Linear):
                    # Prune output neurons
                    sparsity = self._get_layer_sparsity(module)
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=sparsity,
                        n=2,
                        dim=0
                    )
                    parameters_to_prune.append((module, 'weight'))
                    
                elif isinstance(module, nn.Conv2d):
                    # Prune filters
                    sparsity = self._get_layer_sparsity(module)
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=sparsity,
                        n=2,
                        dim=0
                    )
                    parameters_to_prune.append((module, 'weight'))
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        stats = self._compute_pruning_stats()
        return stats
    
    def _lottery_ticket_pruning(self) -> Dict[str, float]:
        """Lottery Ticket Hypothesis pruning."""
        # Save initial weights
        initial_state = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Train model (assumed to be done externally)
        # Then prune based on magnitude
        stats = self._magnitude_pruning()
        
        # Reset remaining weights to initial values
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in initial_state:
                    # Keep pruned mask but reset values
                    mask = (param != 0).float()
                    param.copy_(initial_state[name] * mask)
        
        return stats
    
    def _should_prune_layer(self, name: str, module: nn.Module) -> bool:
        """Check if layer should be pruned."""
        # Check if layer type is prunable
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            return False
        
        # Check if in layers to prune
        if self.layers_to_prune:
            for layer_name in self.layers_to_prune:
                if layer_name in name:
                    return True
            return False
        
        return True
    
    def _get_layer_sparsity(self, module: nn.Module) -> float:
        """Get sparsity for specific layer."""
        # Check layer-wise config
        for name, mod in self.model.named_modules():
            if mod is module:
                for key, sparsity in self.layer_wise_sparsity.items():
                    if key in name:
                        return sparsity
        
        # Default to target sparsity
        return self.target_sparsity
    
    def _compute_pruning_stats(self) -> Dict[str, float]:
        """Compute pruning statistics."""
        total_params = 0
        pruned_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                pruned_params += (param == 0).sum().item()
        
        sparsity = pruned_params / total_params if total_params > 0 else 0
        
        # Compute model size
        model_size_mb = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        ) / (1024 ** 2)
        
        return {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'sparsity': sparsity,
            'model_size_mb': model_size_mb
        }
    
    def evaluate_pruned_model(
        self,
        dataloader,
        device: str = 'mps'
    ) -> float:
        """
        Evaluate pruned model accuracy.
        
        Args:
            dataloader: Validation data loader
            device: Device to evaluate on
            
        Returns:
            Accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                video = batch['video'].to(device)
                audio = batch.get('audio', None)
                if audio is not None:
                    audio = audio.to(device)
                labels = batch['label'].to(device)
                
                outputs = self.model(video, audio)
                _, predicted = outputs['class_logits'].max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy


class IterativePruning:
    """Iterative pruning with fine-tuning."""
    
    def __init__(
        self,
        model: nn.Module,
        trainer,
        config: Dict
    ):
        """
        Initialize iterative pruning.
        
        Args:
            model: Model to prune
            trainer: Trainer instance
            config: Pruning configuration
        """
        self.model = model
        self.trainer = trainer
        self.config = config
        
        self.pruner = ModelPruner(model, config)
        self.pruning_frequency = config.get('pruning_frequency', 5)
        self.finetune_epochs = config.get('finetune_epochs', 3)
    
    def prune_and_finetune(
        self,
        num_iterations: int = 10
    ) -> List[Dict]:
        """
        Iteratively prune and fine-tune model.
        
        Args:
            num_iterations: Number of pruning iterations
            
        Returns:
            List of stats for each iteration
        """
        results = []
        
        for iteration in range(num_iterations):
            print(f"\nPruning iteration {iteration + 1}/{num_iterations}")
            
            # Prune model
            prune_stats = self.pruner.prune_model()
            print(f"Sparsity: {prune_stats['sparsity']:.2%}")
            
            # Fine-tune
            original_epochs = self.trainer.num_epochs
            self.trainer.num_epochs = self.finetune_epochs
            self.trainer.train()
            self.trainer.num_epochs = original_epochs
            
            # Evaluate
            val_metrics = self.trainer.validate()
            
            stats = {
                **prune_stats,
                **val_metrics,
                'iteration': iteration
            }
            results.append(stats)
            
            print(f"Validation accuracy: {val_metrics['val_acc']:.2f}%")
        
        return results
