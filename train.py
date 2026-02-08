"""Main training script."""
import argparse
from pathlib import Path

import torch
from src.utils.config import load_all_configs
from src.dataloaders.dataset import Kinetics400Dataset, create_dataloader
from src.dataloaders.augmentation import MultiModalAugmentation
from src.models.multimodal_model import MultiModalVideoModel
from src.training.trainer import MultiTaskTrainer
from src.utils.logger import ExperimentLogger, create_experiment_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-modal video model')
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default='./configs',
        help='Directory containing config files'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data/kinetics400',
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configurations
    print("Loading configurations...")
    configs = load_all_configs(args.config_dir)
    
    # Create experiment name
    experiment_name = args.experiment_name or create_experiment_name('multimodal_train')
    
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        use_wandb=configs['training_config']['logging']['use_wandb'],
        wandb_project=configs['training_config']['logging']['wandb_project']
    )
    
    logger.log_config(configs)
    logger.log_message(f"Starting experiment: {experiment_name}")
    
    # Create datasets
    logger.log_message("Creating datasets...")
    
    train_transform = MultiModalAugmentation(configs['data_config'])
    
    train_dataset = Kinetics400Dataset(
        root_dir=args.data_root,
        split='train',
        config=configs['data_config'],
        transform=train_transform,
        load_audio=False,
        load_video=True
    )
    
    val_dataset = Kinetics400Dataset(
        root_dir=args.data_root,
        split='val',
        config=configs['data_config'],
        transform=None,
        load_audio=False,
        load_video=True
    )
    
    # Create dataloaders
    dataloader_config = configs['data_config']['dataloader']
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=dataloader_config['batch_size'],
        num_workers=dataloader_config['num_workers'],
        shuffle=True,
        pin_memory=dataloader_config['pin_memory']
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=dataloader_config['batch_size'],
        num_workers=dataloader_config['num_workers'],
        shuffle=False,
        pin_memory=dataloader_config['pin_memory']
    )
    
    logger.log_message(f"Train samples: {len(train_dataset)}")
    logger.log_message(f"Val samples: {len(val_dataset)}")
    
    # Create model
    logger.log_message("Creating model...")
    model = MultiModalVideoModel(configs['model_config'])
    
    total_params, trainable_params = model.get_num_parameters()
    logger.log_message(f"Total parameters: {total_params:,}")
    logger.log_message(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.log_message("Creating trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=configs['training_config'],
        device=args.device
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.log_message(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    logger.log_message("Starting training...")
    trainer.train()
    
    logger.log_message("Training complete!")
    logger.finish()


if __name__ == '__main__':
    main()
