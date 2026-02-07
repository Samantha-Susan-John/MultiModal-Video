# Multi-Modal Video Understanding with Reinforcement Learning

A comprehensive video understanding system that combines computer vision, audio processing, and reinforcement learning for optimal frame sampling. Built with PyTorch and optimized for Apple Silicon (MPS).

## 🎯 Project Overview

This project implements a state-of-the-art multi-modal video understanding system featuring:

- **Multi-Modal Architecture**: Vision + Audio + Text fusion
- **Reinforcement Learning**: DQN agent for intelligent frame sampling
- **Multi-Task Learning**: Action classification, video captioning, and audio-visual sync detection
- **Model Optimization**: Neural network pruning, quantization, and knowledge distillation
- **Production-Ready**: FastAPI serving, comprehensive logging, and deployment tools

## 🏗️ Architecture

```
Video Input → Vision Encoder (EfficientNet/MobileViT)
                    ↓
              Temporal Modeling (Transformer/LSTM)
                    ↓
Audio Input → Audio Encoder (Wav2Vec2)
                    ↓
              Multi-Modal Fusion (Cross-Attention)
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
Classification Head    Captioning Head
```

**RL Agent**: DQN agent learns optimal frame sampling to maximize classification accuracy while minimizing computational cost.

## 📁 Project Structure

```
multimode_rl/
├── configs/                  # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── rl_config.yaml
│   ├── training_config.yaml
│   └── optimization_config.yaml
├── src/
│   ├── data/                # Data pipeline
│   │   ├── video_loader.py
│   │   ├── audio_extractor.py
│   │   ├── augmentation.py
│   │   └── dataset.py
│   ├── models/              # Model architectures
│   │   ├── vision_encoder.py
│   │   ├── audio_encoder.py
│   │   ├── fusion.py
│   │   ├── task_heads.py
│   │   └── multimodal_model.py
│   ├── rl/                  # Reinforcement learning
│   │   ├── dqn_agent.py
│   │   └── environment.py
│   ├── training/            # Training and optimization
│   │   ├── trainer.py
│   │   ├── pruning.py
│   │   ├── quantization.py
│   │   └── distillation.py
│   └── utils/               # Utilities
│       ├── metrics.py
│       ├── logger.py
│       ├── visualization.py
│       └── config.py
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd multimode_rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download Kinetics-400 subset:
```bash
# Instructions for downloading Kinetics-400
# Store videos in data/kinetics400/train and data/kinetics400/val
```

2. Create annotations:
```python
# annotations format: 
# {
#   "video_path": "path/to/video.mp4",
#   "label": 0,
#   "class_name": "class_name",
#   "caption": "description"
# }
```

### Training

#### Phase 1: Pretrain Encoders

```python
from src.utils.config import load_all_configs
from src.data.dataset import Kinetics400Dataset, create_dataloader
from src.models.multimodal_model import MultiModalVideoModel
from src.training.trainer import MultiTaskTrainer

# Load configs
configs = load_all_configs('./configs')

# Create dataset
train_dataset = Kinetics400Dataset(
    root_dir='./data/kinetics400',
    split='train',
    config=configs['data_config']
)

train_loader = create_dataloader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# Create model
model = MultiModalVideoModel(configs['model_config'])

# Create trainer
trainer = MultiTaskTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=configs['training_config'],
    device='mps'
)

# Train
trainer.train()
```

#### Phase 2: RL Training

```python
from src.rl.dqn_agent import DQNAgent
from src.rl.environment import FrameSamplingEnv, RLTrainer

# Create RL environment
env = FrameSamplingEnv(
    video_model=model,
    max_frames=300,
    target_frames=32,
    reward_config=configs['rl_config']['reward']
)

# Create DQN agent
agent = DQNAgent(
    config=configs['rl_config']['agent'],
    device='mps'
)

# Train RL agent
rl_trainer = RLTrainer(agent, env, configs['rl_config']['training'])
# Training loop handled in separate script
```

#### Phase 3: Model Optimization

```python
from src.training.pruning import ModelPruner
from src.training.quantization import ModelQuantizer
from src.training.distillation import KnowledgeDistillation

# Pruning
pruner = ModelPruner(model, configs['optimization_config']['pruning'])
prune_stats = pruner.prune_model()

# Quantization
quantizer = ModelQuantizer(model, configs['optimization_config']['quantization'])
quantized_model = quantizer.quantize(calibration_loader)

# Knowledge Distillation
student_model = create_student_model(model, scale_factor=0.5)
distiller = KnowledgeDistillation(model, student_model, configs['optimization_config']['distillation'])
distiller.train(train_loader, val_loader, num_epochs=50)
```

## 📊 Evaluation

```python
from src.utils.metrics import evaluate_model_comprehensive

# Comprehensive evaluation
results = evaluate_model_comprehensive(
    model=model,
    dataloader=test_loader,
    device='mps',
    verbose=True
)

print(results)
# Output:
# - accuracy, top_5_accuracy
# - precision, recall, f1_score  
# - model_size_mb, total_params
# - inference_time
```

## 🎯 Expected Results

| Metric | Target | Description |
|--------|--------|-------------|
| Top-1 Accuracy | >70% | Classification accuracy on Kinetics-400 |
| Top-5 Accuracy | >90% | Top-5 classification accuracy |
| Model Size | <50 MB | After pruning and quantization |
| Inference Time | <100ms | Per video on MPS |
| RL Improvement | 2-3x | Speedup vs. uniform sampling |

## 🔬 Advanced Features

### Attention Visualization

```python
from src.utils.visualization import plot_attention_weights

# Extract and visualize attention
outputs = model(video, audio, return_features=True)
attention = outputs['attention_weights']
plot_attention_weights(attention, save_path='attention.png')
```

### Frame Sampling Visualization

```python
from src.utils.visualization import visualize_frame_sampling

# Visualize RL agent's frame selection
visualize_frame_sampling(
    video_frames=video,
    selected_indices=agent_selected_frames,
    save_path='sampling.png'
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_data_pipeline.py

# With coverage
pytest --cov=src tests/
```

## 📈 Experiment Tracking

The project supports both local logging and Weights & Biases:

```python
# Enable W&B in training_config.yaml
logging:
  use_wandb: true
  wandb_project: "multimodal-video-rl"
  wandb_entity: "your-username"
```

## 🚢 Deployment

```python
# Export model for deployment
torch.save(model.state_dict(), 'model_production.pth')

# Quantized model for edge devices
quantized_model = quantize_model_int8(model)
torch.save(quantized_model.state_dict(), 'model_quantized.pth')
```

## 📚 Key Technologies

- **PyTorch**: Deep learning framework
- **Transformers**: Pretrained models (Wav2Vec2, GPT-2)
- **OpenCV**: Video processing
- **Librosa**: Audio processing
- **Weights & Biases**: Experiment tracking
- **FastAPI**: Model serving (planned)

## 🎓 Learning Resources

This project demonstrates:
- Multi-modal deep learning
- Reinforcement learning (DQN)
- Model optimization techniques
- Production ML best practices
- Software engineering principles

## 📝 Citation

```bibtex
@software{multimodal_video_rl,
  title={Multi-Modal Video Understanding with Reinforcement Learning},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/multimode_rl}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📧 Contact

For questions or collaborations, reach out to: your.email@example.com

---

**Built for Tesla AI Internship Application** | Demonstrating Foundation Models, Multi-Modal Learning, RL, and Production ML
