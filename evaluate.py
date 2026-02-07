"""Main evaluation script."""
import argparse
import torch

from src.utils.config import load_all_configs
from src.data.dataset import Kinetics400Dataset, create_dataloader
from src.models.multimodal_model import MultiModalVideoModel
from src.utils.metrics import evaluate_model_comprehensive, print_model_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate multi-modal video model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
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
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configurations
    print("Loading configurations...")
    configs = load_all_configs(args.config_dir)
    
    # Create dataset
    print(f"Loading {args.split} dataset...")
    dataset = Kinetics400Dataset(
        root_dir=args.data_root,
        split=args.split,
        config=configs['data_config'],
        transform=None,
        load_audio=True,
        load_video=True
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = MultiModalVideoModel(configs['model_config'])
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print model summary
    print_model_summary(model)
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    results = evaluate_model_comprehensive(
        model=model,
        dataloader=dataloader,
        device=args.device,
        verbose=True
    )
    
    # Save results
    import json
    results_file = f'evaluation_results_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
