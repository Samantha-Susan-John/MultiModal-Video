"""Evaluation metrics for video understanding."""
import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score
)


class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    def __init__(self, num_classes: int = 400):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.all_predictions = []
        self.all_labels = []
        self.all_logits = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor = None
    ):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted class indices
            labels: Ground truth labels
            logits: Raw logits (for top-k accuracy)
        """
        self.all_predictions.extend(predictions.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        
        if logits is not None:
            self.all_logits.append(logits.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        # Accuracy
        accuracy = accuracy_score(labels, predictions) * 100
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average='macro',
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100
        }
        
        # Top-k accuracy
        if len(self.all_logits) > 0:
            logits = np.concatenate(self.all_logits, axis=0)
            
            for k in [1, 5]:
                top_k_acc = top_k_accuracy_score(
                    labels,
                    logits,
                    k=k,
                    labels=np.arange(self.num_classes)
                ) * 100
                metrics[f'top_{k}_accuracy'] = top_k_acc
        
        return metrics
    
    def confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix
        """
        return confusion_matrix(
            self.all_labels,
            self.all_predictions,
            labels=np.arange(self.num_classes)
        )


class EfficiencyMetrics:
    """Metrics for model efficiency."""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Tuple of (total params, trainable params)
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    @staticmethod
    def count_flops(
        model: torch.nn.Module,
        input_shape: Tuple
    ) -> float:
        """
        Estimate FLOPs (simplified).
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            
        Returns:
            Estimated FLOPs
        """
        # This is a simplified estimation
        # For accurate FLOPs, use libraries like thop or fvcore
        
        total_flops = 0
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Conv2d FLOPs
                kernel_ops = (
                    module.kernel_size[0] * module.kernel_size[1] *
                    module.in_channels * module.out_channels
                )
                # Approximate output size
                out_h = input_shape[2] // module.stride[0]
                out_w = input_shape[3] // module.stride[1]
                flops = kernel_ops * out_h * out_w
                total_flops += flops
                
            elif isinstance(module, torch.nn.Linear):
                # Linear FLOPs
                flops = module.in_features * module.out_features
                total_flops += flops
        
        return total_flops
    
    @staticmethod
    def measure_inference_time(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10,
        device: str = 'mps'
    ) -> Dict[str, float]:
        """
        Measure inference time.
        
        Args:
            model: PyTorch model
            input_tensor: Sample input
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            device: Device to run on
            
        Returns:
            Timing statistics
        """
        import time
        
        model.eval()
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device == 'mps':
                    # MPS doesn't have synchronize
                    start = time.time()
                    _ = model(input_tensor)
                    end = time.time()
                elif device == 'cuda':
                    torch.cuda.synchronize()
                    start = time.time()
                    _ = model(input_tensor)
                    torch.cuda.synchronize()
                    end = time.time()
                else:
                    start = time.time()
                    _ = model(input_tensor)
                    end = time.time()
                
                times.append(end - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }
    
    @staticmethod
    def model_size_mb(model: torch.nn.Module) -> float:
        """
        Calculate model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


def print_model_summary(model: torch.nn.Module):
    """
    Print comprehensive model summary.
    
    Args:
        model: PyTorch model
    """
    total_params, trainable_params = EfficiencyMetrics.count_parameters(model)
    size_mb = EfficiencyMetrics.model_size_mb(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {size_mb:.2f} MB")
    print("=" * 60)
    
    # Print layer-wise parameters
    print("\nLayer-wise parameters:")
    print("-" * 60)
    
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:30s}: {num_params:>15,}")
    
    print("=" * 60)


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    dataloader,
    device: str = 'mps',
    verbose: bool = True
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: PyTorch model
        dataloader: Evaluation data loader
        device: Device to evaluate on
        verbose: Whether to print results
        
    Returns:
        Dictionary of all metrics
    """
    model.eval()
    model = model.to(device)
    
    metrics_calc = MetricsCalculator()
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            audio = batch.get('audio', None)
            if audio is not None:
                audio = audio.to(device)
            labels = batch['label'].to(device)
            
            outputs = model(video, audio)
            logits = outputs['class_logits']
            predictions = logits.argmax(dim=-1)
            
            metrics_calc.update(predictions, labels, logits)
    
    # Compute metrics
    metrics = metrics_calc.compute()
    
    # Add efficiency metrics
    total_params, trainable_params = EfficiencyMetrics.count_parameters(model)
    metrics['total_params'] = total_params
    metrics['trainable_params'] = trainable_params
    metrics['model_size_mb'] = EfficiencyMetrics.model_size_mb(model)
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:>10.2f}")
            else:
                print(f"{key:30s}: {value:>10,}")
        print("=" * 60)
    
    return metrics
