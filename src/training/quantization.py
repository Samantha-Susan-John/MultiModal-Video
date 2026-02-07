"""Model quantization utilities."""
import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Optional
import copy


class ModelQuantizer:
    """Model quantization for compression and speedup."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        """
        Initialize quantizer.
        
        Args:
            model: Model to quantize
            config: Quantization configuration
        """
        self.model = model
        self.config = config
        
        self.method = config.get('method', 'post_training')
        self.backend = config.get('backend', 'qnnpack')
        self.dtype = config.get('dtype', 'qint8')
        
        # Set backend
        torch.backends.quantized.engine = self.backend
    
    def quantize(self, calibration_loader=None) -> nn.Module:
        """
        Quantize the model.
        
        Args:
            calibration_loader: Data loader for calibration (PTQ only)
            
        Returns:
            Quantized model
        """
        if self.method == 'post_training':
            return self._post_training_quantization(calibration_loader)
        elif self.method == 'quantization_aware':
            return self._quantization_aware_training()
        elif self.method == 'dynamic':
            return self._dynamic_quantization()
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")
    
    def _post_training_quantization(
        self,
        calibration_loader
    ) -> nn.Module:
        """
        Post-training static quantization.
        
        Args:
            calibration_loader: Data loader for calibration
            
        Returns:
            Quantized model
        """
        # Make a copy for quantization
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        
        # Fuse modules
        model_copy = self._fuse_modules(model_copy)
        
        # Prepare for quantization
        model_copy.qconfig = quant.get_default_qconfig(self.backend)
        quant.prepare(model_copy, inplace=True)
        
        # Calibrate
        if calibration_loader:
            print("Calibrating quantized model...")
            self._calibrate(model_copy, calibration_loader)
        
        # Convert to quantized model
        quant.convert(model_copy, inplace=True)
        
        return model_copy
    
    def _quantization_aware_training(self) -> nn.Module:
        """
        Quantization-aware training.
        
        Returns:
            Model prepared for QAT
        """
        model_copy = copy.deepcopy(self.model)
        
        # Fuse modules
        model_copy = self._fuse_modules(model_copy)
        
        # Prepare for QAT
        model_copy.qconfig = quant.get_default_qat_qconfig(self.backend)
        quant.prepare_qat(model_copy, inplace=True)
        
        print("Model prepared for quantization-aware training")
        print("Train the model and then call convert() to get quantized model")
        
        return model_copy
    
    def _dynamic_quantization(self) -> nn.Module:
        """
        Dynamic quantization (weights only).
        
        Returns:
            Quantized model
        """
        model_copy = copy.deepcopy(self.model)
        
        # Specify layers to quantize
        layers_to_quantize = {nn.Linear, nn.LSTM, nn.GRU}
        
        quantized_model = quant.quantize_dynamic(
            model_copy,
            layers_to_quantize,
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse consecutive modules for better quantization.
        
        Args:
            model: Model to fuse
            
        Returns:
            Fused model
        """
        # Common fusion patterns
        # Conv + BN + ReLU
        # Linear + ReLU
        
        # This is model-specific and would need to be customized
        # For now, return model as-is
        return model
    
    def _calibrate(
        self,
        model: nn.Module,
        calibration_loader,
        num_batches: int = 100
    ):
        """
        Calibrate quantized model.
        
        Args:
            model: Model to calibrate
            calibration_loader: Calibration data
            num_batches: Number of batches for calibration
        """
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                
                video = batch['video']
                audio = batch.get('audio', None)
                
                # Forward pass for calibration
                _ = model(video, audio)
    
    def compare_models(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_loader,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Compare original and quantized models.
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_loader: Test data loader
            device: Device for evaluation
            
        Returns:
            Comparison metrics
        """
        import time
        
        # Evaluate original model
        original_model.eval()
        original_correct = 0
        original_total = 0
        original_time = 0
        
        with torch.no_grad():
            for batch in test_loader:
                video = batch['video'].to(device)
                audio = batch.get('audio', None)
                if audio is not None:
                    audio = audio.to(device)
                labels = batch['label'].to(device)
                
                start = time.time()
                outputs = original_model(video, audio)
                original_time += time.time() - start
                
                _, predicted = outputs['class_logits'].max(1)
                original_total += labels.size(0)
                original_correct += predicted.eq(labels).sum().item()
        
        # Evaluate quantized model (on CPU)
        quantized_model.eval()
        quantized_correct = 0
        quantized_total = 0
        quantized_time = 0
        
        with torch.no_grad():
            for batch in test_loader:
                video = batch['video']  # Keep on CPU
                audio = batch.get('audio', None)
                labels = batch['label']
                
                start = time.time()
                outputs = quantized_model(video, audio)
                quantized_time += time.time() - start
                
                _, predicted = outputs['class_logits'].max(1)
                quantized_total += labels.size(0)
                quantized_correct += predicted.eq(labels).sum().item()
        
        # Model sizes
        def get_model_size(model):
            torch.save(model.state_dict(), '/tmp/temp_model.pth')
            size = os.path.getsize('/tmp/temp_model.pth') / (1024 ** 2)
            os.remove('/tmp/temp_model.pth')
            return size
        
        import os
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        return {
            'original_accuracy': 100. * original_correct / original_total,
            'quantized_accuracy': 100. * quantized_correct / quantized_total,
            'accuracy_drop': (original_correct - quantized_correct) / original_total * 100,
            'original_time': original_time,
            'quantized_time': quantized_time,
            'speedup': original_time / quantized_time if quantized_time > 0 else 0,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0
        }


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Simple helper to quantize model to INT8.
    
    Args:
        model: Model to quantize
        
    Returns:
        Quantized model
    """
    quantized = quant.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )
    
    print("Model quantized to INT8")
    return quantized
