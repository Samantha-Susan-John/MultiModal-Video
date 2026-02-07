"""Generate dummy video data for testing."""
import json
import os
from pathlib import Path
import numpy as np
import cv2

def generate_video(output_path: str, duration: int = 10, fps: int = 30, 
                   width: int = 224, height: int = 224, pattern: str = 'random'):
    """
    Generate a synthetic video file.
    
    Args:
        output_path: Path to save the video
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        pattern: Pattern type ('random', 'gradient', 'motion')
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration * fps
    
    for i in range(num_frames):
        if pattern == 'random':
            # Random colored noise
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        elif pattern == 'gradient':
            # Moving gradient
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            shift = int((i / num_frames) * 255)
            frame[:, :, 0] = (np.arange(width) + shift) % 256
            frame[:, :, 1] = (np.arange(height).reshape(-1, 1) + shift) % 256
            frame[:, :, 2] = 128
        
        elif pattern == 'motion':
            # Moving circle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cx = int(width / 2 + (width / 4) * np.sin(2 * np.pi * i / num_frames))
            cy = int(height / 2 + (height / 4) * np.cos(2 * np.pi * i / num_frames))
            cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        
        else:  # solid color
            frame = np.full((height, width, 3), i % 256, dtype=np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Generated: {output_path}")


def create_dummy_dataset(root_dir: str = './data/kinetics400', 
                        num_train: int = 80, 
                        num_val: int = 10, 
                        num_test: int = 10,
                        num_classes: int = 20):
    """
    Create a complete dummy dataset.
    
    Args:
        root_dir: Root directory for dataset
        num_train: Number of training videos
        num_val: Number of validation videos
        num_test: Number of test videos
        num_classes: Number of action classes (subset of 400)
    """
    root_path = Path(root_dir)
    
    # Class names (subset of real Kinetics-400 actions)
    all_classes = [
        "abseiling", "air_drumming", "answering_questions", "applauding", "applying_cream",
        "archery", "arm_wrestling", "arranging_flowers", "assembling_computer", "auctioning",
        "baby_waking_up", "baking_cookies", "balloon_blowing", "bandaging", "barbequing",
        "bartending", "beatboxing", "bending_back", "bending_metal", "biking_through_snow"
    ]
    classes = all_classes[:num_classes]
    
    # Save classes
    with open(root_path / 'classes.json', 'w') as f:
        json.dump(classes, f, indent=2)
    print(f"Created classes.json with {num_classes} classes")
    
    # Generate videos and annotations for each split
    for split, num_videos in [('train', num_train), ('val', num_val), ('test', num_test)]:
        split_dir = root_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        annotations = []
        patterns = ['random', 'gradient', 'motion', 'solid']
        
        for i in range(num_videos):
            # Random class
            class_idx = np.random.randint(0, num_classes)
            class_name = classes[class_idx]
            
            # Video filename
            video_name = f"{split}_video_{i:04d}.mp4"
            video_path = split_dir / video_name
            
            # Generate video with varying pattern
            pattern = patterns[i % len(patterns)]
            generate_video(str(video_path), duration=10, fps=30, pattern=pattern)
            
            # Create annotation
            annotations.append({
                'video_path': str(video_path),
                'label': class_idx,
                'class_name': class_name,
                'caption': f"A person performing {class_name.replace('_', ' ')}"
            })
        
        # Save annotations
        annotation_file = root_path / f'{split}_annotations.json'
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"Created {split}_annotations.json with {num_videos} videos")
    
    print("\n✅ Dummy dataset created successfully!")
    print(f"📁 Location: {root_path}")
    print(f"📊 Classes: {num_classes}")
    print(f"🎬 Videos: {num_train + num_val + num_test} total")
    print(f"   - Train: {num_train}")
    print(f"   - Val: {num_val}")
    print(f"   - Test: {num_test}")


if __name__ == '__main__':
    # Create dataset with 100 total videos (80 train, 10 val, 10 test)
    # Using 20 classes for faster training
    create_dummy_dataset(
        root_dir='./data/kinetics400',
        num_train=80,
        num_val=10,
        num_test=10,
        num_classes=20
    )
