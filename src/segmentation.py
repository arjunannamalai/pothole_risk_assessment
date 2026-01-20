"""
Stage 1: Pothole Segmentation using Segment Anything Model (SAM)

This module provides pixel-level segmentation of potholes from road images.
SAM provides precise boundaries which are essential for accurate depth and
size measurements.
"""

import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional
import os


class PotholeSegmenter:
    """
    Segment potholes using Meta's Segment Anything Model (SAM)
    
    SAM provides:
    - Zero-shot segmentation (no fine-tuning needed)
    - Precise pixel-level masks
    - Multiple mask proposals per object
    """
    
    def __init__(
        self,
        checkpoint_path: str = "models/sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: str = "cuda"
    ):
        """
        Initialize SAM model
        
        Args:
            checkpoint_path: Path to SAM checkpoint
            model_type: Model variant (vit_h, vit_l, vit_b)
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Import SAM
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment-anything not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint_path}. "
                f"Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        
        print(f"Loading SAM model ({model_type}) on {self.device}...")
        
        # Load model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        # Create mask generator with optimized parameters
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,          # Grid density for point prompts
            pred_iou_thresh=0.80,        # Minimum predicted IoU
            stability_score_thresh=0.85, # Mask stability threshold
            crop_n_layers=1,             # Number of crop layers
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1000    # Minimum mask area in pixels
        )
        
        print("✓ SAM model loaded successfully")
    
    def segment_image(
        self,
        image: np.ndarray,
        min_area: int = 1000,
        max_area: int = 500000
    ) -> List[Dict]:
        """
        Segment all objects in image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            min_area: Minimum mask area in pixels
            max_area: Maximum mask area in pixels
            
        Returns:
            List of mask dictionaries with segmentation info
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with shape (H, W, 3)")
        
        # Generate all masks
        masks = self.mask_generator.generate(image)
        
        # Filter by area
        filtered_masks = [
            mask for mask in masks
            if min_area < mask['area'] < max_area
        ]
        
        return filtered_masks
    
    def filter_pothole_masks(
        self,
        masks: List[Dict],
        image: np.ndarray,
        darkness_threshold: float = 0.5,
        road_region: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Filter masks to identify likely potholes
        
        Heuristics:
        1. Potholes are typically darker than surrounding road
        2. Potholes are in the lower portion of image (road surface)
        3. Potholes have irregular/rough edges
        
        Args:
            masks: List of SAM masks
            image: Original RGB image
            darkness_threshold: How much darker than surroundings (0-1)
            road_region: Optional mask of road surface
            
        Returns:
            Filtered list of potential pothole masks
        """
        pothole_candidates = []
        
        # Convert to grayscale for darkness analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_mean = np.mean(gray)
        
        for mask in masks:
            segmentation = mask['segmentation']
            bbox = mask['bbox']  # [x, y, w, h]
            area = mask['area']
            
            # Get mask center
            y_indices, x_indices = np.where(segmentation)
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
            
            # Heuristic 1: Must be in lower 70% of image (road surface)
            if center_y < image.shape[0] * 0.3:
                continue
            
            # Heuristic 2: Darker than surrounding
            mask_pixels = gray[segmentation]
            mask_mean = np.mean(mask_pixels)
            
            # Create dilated mask for surrounding area
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(segmentation.astype(np.uint8), kernel, iterations=2)
            surrounding = dilated.astype(bool) & ~segmentation
            
            if np.sum(surrounding) > 0:
                surrounding_mean = np.mean(gray[surrounding])
                darkness_ratio = mask_mean / surrounding_mean
                
                # Pothole should be darker than surroundings
                if darkness_ratio > darkness_threshold + 0.3:
                    continue
            
            # Heuristic 3: Check if within road region (if provided)
            if road_region is not None:
                overlap = np.sum(segmentation & road_region) / area
                if overlap < 0.8:  # At least 80% should be on road
                    continue
            
            # Calculate shape features
            contours, _ = cv2.findContours(
                segmentation.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # Add shape info to mask
                mask['circularity'] = circularity
                mask['center'] = (center_x, center_y)
                mask['darkness_ratio'] = darkness_ratio if 'darkness_ratio' in dir() else 1.0
                
                pothole_candidates.append(mask)
        
        # Sort by area (larger potholes first)
        pothole_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        return pothole_candidates
    
    def extract_mask_info(self, mask: Dict, image_shape: Tuple[int, int]) -> Dict:
        """
        Extract detailed information from a mask
        
        Args:
            mask: SAM mask dictionary
            image_shape: (height, width) of original image
            
        Returns:
            Dictionary with detailed mask information
        """
        segmentation = mask['segmentation']
        bbox = mask['bbox']
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(
            segmentation.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Fit ellipse for orientation (if enough points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            orientation = ellipse[2]  # Angle in degrees
        else:
            orientation = 0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'segmentation': segmentation,
            'bbox': [x, y, w, h],
            'area_pixels': area,
            'perimeter_pixels': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'orientation': orientation,
            'contour': contour,
            'stability_score': mask.get('stability_score', 0),
            'predicted_iou': mask.get('predicted_iou', 0)
        }
    
    def visualize_masks(
        self,
        image: np.ndarray,
        masks: List[Dict],
        alpha: float = 0.5,
        show_bbox: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize masks on image
        
        Args:
            image: RGB image
            masks: List of mask dictionaries
            alpha: Transparency for mask overlay
            show_bbox: Whether to draw bounding boxes
            show_labels: Whether to show mask labels
            
        Returns:
            Annotated image
        """
        output = image.copy()
        
        # Color palette for different masks
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            segmentation = mask['segmentation']
            
            # Create colored overlay
            overlay = output.copy()
            overlay[segmentation] = color
            output = cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)
            
            # Draw bounding box
            if show_bbox:
                bbox = mask['bbox']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            if show_labels:
                label = f"Pothole #{idx + 1}"
                area = mask.get('area', mask.get('area_pixels', 0))
                label += f" ({area} px)"
                
                cv2.putText(
                    output, label,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2
                )
        
        return output


def download_sam_checkpoint(
    model_type: str = "vit_h",
    save_dir: str = "models"
) -> str:
    """
    Download SAM checkpoint
    
    Args:
        model_type: vit_h, vit_l, or vit_b
        save_dir: Directory to save checkpoint
        
    Returns:
        Path to downloaded checkpoint
    """
    import urllib.request
    
    urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    if model_type not in urls:
        raise ValueError(f"Unknown model type: {model_type}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    filename = urls[model_type].split("/")[-1]
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Checkpoint already exists: {filepath}")
        return filepath
    
    print(f"Downloading SAM {model_type} checkpoint...")
    print(f"URL: {urls[model_type]}")
    print("This may take a while (2.4GB for vit_h)...")
    
    urllib.request.urlretrieve(urls[model_type], filepath)
    
    print(f"✓ Downloaded to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Example usage
    print("Testing PotholeSegmenter...")
    
    # Download checkpoint if needed
    checkpoint = download_sam_checkpoint("vit_h", "models")
    
    # Initialize segmenter
    segmenter = PotholeSegmenter(
        checkpoint_path=checkpoint,
        model_type="vit_h",
        device="cuda"
    )
    
    # Test with sample image
    # image = cv2.imread("sample_pothole.jpg")
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # masks = segmenter.segment_image(image_rgb)
    # pothole_masks = segmenter.filter_pothole_masks(masks, image_rgb)
    # print(f"Found {len(pothole_masks)} potential potholes")
