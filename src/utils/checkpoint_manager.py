"""Checkpoint manager with GitHub auto-push support."""
import os
import subprocess
from pathlib import Path
from typing import Optional
import torch


class GitCheckpointManager:
    """Manage checkpoints with automatic GitHub push."""
    
    def __init__(
        self,
        repo_path: str = ".",
        checkpoint_branch: str = "checkpoints",
        auto_push: bool = True,
        keep_local: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            repo_path: Path to git repository
            checkpoint_branch: Branch name for checkpoints
            auto_push: Automatically push after saving
            keep_local: Keep local copy after pushing
        """
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint_branch = checkpoint_branch
        self.auto_push = auto_push
        self.keep_local = keep_local
        
    def save_and_push(
        self,
        checkpoint: dict,
        filename: str,
        commit_message: Optional[str] = None
    ):
        """
        Save checkpoint and push to GitHub.
        
        Args:
            checkpoint: Checkpoint dictionary to save
            filename: Checkpoint filename (e.g., 'checkpoint_epoch_5.pth')
            commit_message: Git commit message
        """
        # Save checkpoint to temp location
        temp_path = Path('/tmp') / filename
        torch.save(checkpoint, temp_path)
        
        print(f"Saved checkpoint: {temp_path}")
        print(f"Size: {temp_path.stat().st_size / (1024**2):.2f} MB")
        
        if self.auto_push:
            try:
                self._push_to_github(temp_path, filename, commit_message)
                print(f"✓ Pushed to GitHub: {self.checkpoint_branch}/{filename}")
                
                if not self.keep_local:
                    temp_path.unlink()
                    print(f"✓ Cleaned up local checkpoint")
                    
            except Exception as e:
                print(f"✗ Failed to push to GitHub: {e}")
                print(f"Checkpoint saved locally at: {temp_path}")
    
    def _push_to_github(self, checkpoint_path: Path, filename: str, message: Optional[str]):
        """Push checkpoint to GitHub branch."""
        original_branch = self._get_current_branch()
        
        try:
            # Stash any changes on main branch
            subprocess.run(['git', 'stash'], cwd=self.repo_path, check=False)
            
            # Switch to checkpoint branch
            result = subprocess.run(
                ['git', 'checkout', self.checkpoint_branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Branch doesn't exist, create it
                subprocess.run(
                    ['git', 'checkout', '--orphan', self.checkpoint_branch],
                    cwd=self.repo_path,
                    check=True
                )
                subprocess.run(['git', 'rm', '-rf', '.'], cwd=self.repo_path, check=False)
            
            # Copy checkpoint to repo
            dest_path = self.repo_path / filename
            subprocess.run(['cp', str(checkpoint_path), str(dest_path)], check=True)
            
            # Git add
            subprocess.run(['git', 'add', filename], cwd=self.repo_path, check=True)
            
            # Commit
            if message is None:
                epoch = filename.split('_')[-1].replace('.pth', '')
                message = f"Add checkpoint: {filename} (epoch {epoch})"
            
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.repo_path,
                check=True
            )
            
            # Push
            subprocess.run(
                ['git', 'push', '-u', 'origin', self.checkpoint_branch],
                cwd=self.repo_path,
                check=True
            )
            
        finally:
            # Return to original branch
            subprocess.run(
                ['git', 'checkout', original_branch],
                cwd=self.repo_path,
                check=False
            )
            subprocess.run(['git', 'stash', 'pop'], cwd=self.repo_path, check=False)
    
    def _get_current_branch(self) -> str:
        """Get current git branch name."""
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    
    @staticmethod
    def download_checkpoint(branch: str = "checkpoints", filename: str = "best_model.pth"):
        """
        Download checkpoint from GitHub.
        
        Args:
            branch: Branch containing checkpoints
            filename: Checkpoint filename to download
            
        Returns:
            Path to downloaded checkpoint
        """
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp())
        
        # Clone only checkpoint branch
        subprocess.run([
            'git', 'clone',
            '--branch', branch,
            '--single-branch',
            '--depth', '1',
            'https://github.com/Samantha-Susan-John/MultiModal-Video.git',
            str(temp_dir)
        ], check=True)
        
        # Pull LFS files
        subprocess.run(['git', 'lfs', 'pull'], cwd=temp_dir, check=True)
        
        checkpoint_path = temp_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filename}")
        
        return checkpoint_path
