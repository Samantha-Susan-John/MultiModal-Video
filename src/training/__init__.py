"""Training module initialization."""
from .trainer import MultiTaskTrainer
from .pruning import ModelPruner, IterativePruning
from .quantization import ModelQuantizer, quantize_model_int8
from .distillation import KnowledgeDistillation, create_student_model

__all__ = [
    'MultiTaskTrainer',
    'ModelPruner',
    'IterativePruning',
    'ModelQuantizer',
    'quantize_model_int8',
    'KnowledgeDistillation',
    'create_student_model'
]
