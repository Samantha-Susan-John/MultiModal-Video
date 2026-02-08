"""Download checkpoint from HuggingFace Hub."""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_checkpoint(
    repo_id: str = "Samantha-Susan-John/MultiModal-Video-Checkpoints",
    filename: str = "best_model.pth",
    output_dir: str = "./checkpoints"
):
    """
    Download checkpoint from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: Checkpoint filename to download
        output_dir: Local directory to save checkpoint
    """
    print(f"Downloading from HuggingFace Hub: {repo_id}/{filename}")
    
    # Download
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=output_dir
    )
    
    print(f"✓ Downloaded to: {local_path}")
    print(f"\nTo load checkpoint:")
    print(f"  import torch")
    print(f"  checkpoint = torch.load('{local_path}')")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
    
    return local_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download checkpoint from HuggingFace')
    parser.add_argument(
        '--repo-id',
        type=str,
        default='Samantha-Susan-John/MultiModal-Video-Checkpoints',
        help='HuggingFace repository ID'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='best_model.pth',
        help='Checkpoint filename'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Output directory'
    )
    
    args = parser.parse_args()
    download_checkpoint(args.repo_id, args.filename, args.output_dir)
