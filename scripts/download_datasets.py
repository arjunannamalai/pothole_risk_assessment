#!/usr/bin/env python3
"""
Script to download pothole datasets for Indian roads
"""

import os
import urllib.request
import zipfile
import tarfile
import sys
import shutil


DATASETS = {
    'rdd2022_india': {
        'url': 'https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip',
        'filename': 'RDD2022_India.zip',
        'description': 'RDD2022 India - 7,706 road damage images from Indian roads',
        'extract': True
    },
    'pothole_kaggle': {
        'url': 'https://github.com/arjunannamalai/pothole_risk_assessment/releases/download/v0.1/pothole_sample.zip',
        'filename': 'pothole_sample.zip',
        'description': 'Sample pothole images (for testing)',
        'extract': True,
        'note': 'For full Kaggle dataset, use: kaggle datasets download -d sachinpatel21/pothole-image-dataset'
    }
}


def download_with_progress(url: str, filepath: str):
    """Download file with progress bar"""
    
    def report_progress(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            percent = min(percent, 100)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            size_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f'\r  [{bar}] {percent}% ({size_mb:.1f}/{total_mb:.1f} MB)')
        else:
            size_mb = count * block_size / (1024 * 1024)
            sys.stdout.write(f'\r  Downloaded: {size_mb:.1f} MB')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
    print()


def extract_archive(filepath: str, extract_to: str):
    """Extract zip or tar archive"""
    print(f"  Extracting to: {extract_to}")
    
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif filepath.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif filepath.endswith('.tar'):
        with tarfile.open(filepath, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print("  ✓ Extraction complete")


def download_rdd2022_india(data_dir: str):
    """Download RDD2022 India dataset"""
    print("\n" + "=" * 60)
    print("Downloading RDD2022 India Dataset")
    print("=" * 60)
    print("Source: CRDDC2022 (Crowdsensing-based Road Damage Detection)")
    print("Images: ~7,706 road damage images from Indian roads")
    print("Labels: Bounding boxes in COCO format")
    print("Categories: D00 (Longitudinal), D10 (Transverse), D20 (Alligator), D40 (Pothole)")
    print()
    
    dataset_dir = os.path.join(data_dir, 'rdd2022_india')
    os.makedirs(dataset_dir, exist_ok=True)
    
    url = DATASETS['rdd2022_india']['url']
    filename = DATASETS['rdd2022_india']['filename']
    filepath = os.path.join(dataset_dir, filename)
    
    if os.path.exists(os.path.join(dataset_dir, 'India')):
        print("✓ Dataset already exists!")
        return dataset_dir
    
    print(f"Downloading from: {url}")
    try:
        download_with_progress(url, filepath)
        print(f"✓ Downloaded: {filepath}")
        
        # Extract
        extract_archive(filepath, dataset_dir)
        
        # Clean up zip file
        os.remove(filepath)
        print(f"✓ Cleaned up archive")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://crddc2022.sekilab.global/download/")
        return None
    
    return dataset_dir


def download_sample_images(data_dir: str):
    """Download sample pothole images for testing"""
    print("\n" + "=" * 60)
    print("Downloading Sample Pothole Images")
    print("=" * 60)
    
    sample_dir = os.path.join(data_dir, 'sample_images')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Download a few sample images from public sources
    sample_urls = [
        ('pothole1.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Pot_hole_road.jpg/1280px-Pot_hole_road.jpg'),
        ('pothole2.jpg', 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Pothole.jpg/1280px-Pothole.jpg'),
    ]
    
    for filename, url in sample_urls:
        filepath = os.path.join(sample_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename} already exists")
            continue
        
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ Downloaded {filename}")
        except Exception as e:
            print(f"  ⚠ Failed to download {filename}: {e}")
    
    return sample_dir


def show_kaggle_instructions():
    """Show instructions for Kaggle dataset"""
    print("\n" + "=" * 60)
    print("KAGGLE DATASET INSTRUCTIONS")
    print("=" * 60)
    print("""
To download the full Kaggle Pothole Dataset:

1. Install Kaggle CLI:
   pip install kaggle

2. Setup Kaggle credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save kaggle.json to ~/.kaggle/kaggle.json
   - chmod 600 ~/.kaggle/kaggle.json

3. Download dataset:
   kaggle datasets download -d sachinpatel21/pothole-image-dataset -p data/kaggle_pothole
   unzip data/kaggle_pothole/pothole-image-dataset.zip -d data/kaggle_pothole/

Available datasets on Kaggle:
- sachinpatel21/pothole-image-dataset (1,200+ images)
- sovitrath/road-pothole-images-for-pothole-detection
- chitholian/annotated-potholes-dataset
""")


def main():
    print("\n" + "=" * 60)
    print("POTHOLE DATASET DOWNLOADER")
    print("=" * 60)
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\nDatasets will be saved to: {data_dir}\n")
    
    # Parse arguments
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
    else:
        dataset = 'all'
    
    print("Available datasets:")
    print("  1. rdd2022    - RDD2022 India (7,706 images)")
    print("  2. sample     - Sample images for testing")
    print("  3. all        - Download all available")
    print("  4. kaggle     - Show Kaggle download instructions")
    print()
    
    if dataset in ['rdd2022', 'all', '1']:
        download_rdd2022_india(data_dir)
    
    if dataset in ['sample', 'all', '2']:
        download_sample_images(data_dir)
    
    if dataset in ['kaggle', '4']:
        show_kaggle_instructions()
    
    if dataset == 'all':
        show_kaggle_instructions()
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item != '.gitkeep':
            # Count files
            file_count = sum(len(files) for _, _, files in os.walk(item_path))
            print(f"  {item}: {file_count} files")
    
    print("\n✓ Dataset download complete!")
    print(f"  Data location: {data_dir}")


if __name__ == "__main__":
    main()
