#!/usr/bin/env python3
"""
Script to download required model checkpoints
"""

import os
import urllib.request
import sys


MODELS = {
    'sam_vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'filename': 'sam_vit_h_4b8939.pth',
        'size': '2.4GB',
        'description': 'SAM ViT-H (Best quality, slowest)'
    },
    'sam_vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'filename': 'sam_vit_l_0b3195.pth',
        'size': '1.2GB',
        'description': 'SAM ViT-L (Balanced)'
    },
    'sam_vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'filename': 'sam_vit_b_01ec64.pth',
        'size': '375MB',
        'description': 'SAM ViT-B (Fastest, good quality)'
    }
}


def download_with_progress(url: str, filepath: str):
    """Download file with progress bar"""
    
    def report_progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        sys.stdout.write(f'\r  [{bar}] {percent}%')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
    print()  # New line after progress bar


def main():
    print("\n" + "=" * 60)
    print("MODEL CHECKPOINT DOWNLOADER")
    print("=" * 60)
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\nModels will be saved to: {models_dir}\n")
    
    # Show available models
    print("Available SAM models:")
    print("-" * 60)
    for key, info in MODELS.items():
        print(f"  {key}: {info['description']} ({info['size']})")
    print()
    
    # Default to vit_h if no argument
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
    else:
        model_key = 'sam_vit_h'
        print(f"No model specified, using default: {model_key}")
    
    if model_key not in MODELS:
        print(f"\n❌ Unknown model: {model_key}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return
    
    model = MODELS[model_key]
    filepath = os.path.join(models_dir, model['filename'])
    
    # Check if already exists
    if os.path.exists(filepath):
        print(f"\n✓ Model already exists: {filepath}")
        response = input("Download again? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    # Download
    print(f"\nDownloading {model_key}...")
    print(f"  URL: {model['url']}")
    print(f"  Size: {model['size']}")
    print()
    
    try:
        download_with_progress(model['url'], filepath)
        print(f"\n✓ Downloaded successfully: {filepath}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return
    
    # Verify file
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.1f} MB")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nYou can now run the analyzer with:")
    print(f"  python examples/demo.py your_image.jpg")


if __name__ == "__main__":
    main()
