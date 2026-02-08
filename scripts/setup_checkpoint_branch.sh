#!/bin/bash
# Setup separate branch for checkpoints with Git LFS

set -e

echo "Setting up checkpoint storage branch..."

# Install Git LFS (if not already installed)
git lfs install

# Track .pth files with LFS
git lfs track "*.pth"
git add .gitattributes

# Commit LFS config
git commit -m "Configure Git LFS for checkpoint files" || true

# Create orphan branch for checkpoints (no history to save space)
git checkout --orphan checkpoints
git rm -rf .
git clean -fdx

# Create README
cat > README.md << 'EOF'
# Model Checkpoints

This branch stores trained model checkpoints.

## Files

- `checkpoint_epoch_N.pth` - Checkpoint from epoch N
- `best_model.pth` - Best performing checkpoint

## File Size Warning

Checkpoint files are large (~600MB each). GitHub LFS has a 1GB/month bandwidth limit on free tier.

## Download Checkpoint

```bash
git clone --branch checkpoints --single-branch https://github.com/Samantha-Susan-John/MultiModal-Video.git checkpoints
cd checkpoints
git lfs pull  # Download actual checkpoint files
```
EOF

git add README.md
git commit -m "Initialize checkpoints branch"
git push -u origin checkpoints

# Return to main branch
git checkout main

echo "✓ Checkpoints branch created!"
echo "✓ Git LFS configured for .pth files"
echo ""
echo "Checkpoints will auto-push to: origin/checkpoints"
