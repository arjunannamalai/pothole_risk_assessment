"""
Stage 2: Depth Estimation using Depth Anything V2

This module provides monocular depth estimation to measure pothole depth
from a single image. Depth Anything V2 is a state-of-the-art model that
provides accurate relative depth maps.
"""

import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional, Union
import os


class DepthEstimator:
    """
    Estimate depth from single images using Depth Anything V2
    
    Depth Anything V2 provides:
    - Zero-shot depth estimation
    - High-quality depth maps
    - Works on any image without calibration
    
    Note: Output is relative depth. Calibration is needed for absolute measurements.
    """
    
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Large-hf",
        device: str = "cuda"
    ):
        """
        Initialize Depth Anything V2 model
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
        
        print(f"Loading Depth Anything V2 on {self.device}...")
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Depth Anything V2 loaded successfully")
    
    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Estimate relative depth from image
        
        Args:
            image: RGB image (numpy array or PIL Image)
            
        Returns:
            Depth map as numpy array (H, W), values are relative depth
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Prepare input
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False
        )
        
        # Convert to numpy
        depth_map = prediction.squeeze().cpu().numpy()
        
        return depth_map
    
    def calibrate_depth(
        self,
        depth_map: np.ndarray,
        calibration_method: str = "camera_height",
        **kwargs
    ) -> np.ndarray:
        """
        Convert relative depth to absolute depth (meters)
        
        Calibration Methods:
        1. camera_height: Use known camera height from ground
        2. reference_object: Use known object size in scene
        3. lane_width: Use standard lane width as reference
        
        Args:
            depth_map: Relative depth map
            calibration_method: Method for calibration
            **kwargs: Method-specific parameters
            
        Returns:
            Calibrated depth map in meters
        """
        if calibration_method == "camera_height":
            # Assume camera is at known height from ground
            camera_height = kwargs.get('camera_height', 1.5)  # meters
            
            # Ground is typically at the 95th percentile of depth values
            ground_depth = np.percentile(depth_map, 95)
            
            # Scale factor: camera_height corresponds to ground_depth
            scale_factor = camera_height / ground_depth
            
        elif calibration_method == "reference_object":
            # Use known object in scene
            actual_size = kwargs.get('actual_size')  # meters
            pixel_size = kwargs.get('pixel_size')    # pixels
            reference_depth = kwargs.get('reference_depth')  # depth value at reference
            
            if actual_size is None or pixel_size is None:
                raise ValueError("reference_object requires actual_size and pixel_size")
            
            # Approximate scale
            scale_factor = actual_size / (pixel_size * np.mean(depth_map))
            
        elif calibration_method == "lane_width":
            # Use standard lane width
            lane_width_m = kwargs.get('lane_width', 3.5)  # meters
            lane_width_px = kwargs.get('lane_width_pixels')
            
            if lane_width_px is None:
                # Estimate from image (assume road fills 60% of width at bottom)
                lane_width_px = depth_map.shape[1] * 0.6
            
            # Rough approximation
            avg_depth = np.mean(depth_map[int(depth_map.shape[0]*0.7):, :])
            scale_factor = lane_width_m / (lane_width_px * avg_depth / depth_map.shape[1])
            
        else:
            raise ValueError(f"Unknown calibration method: {calibration_method}")
        
        # Apply calibration
        calibrated_depth = depth_map * scale_factor
        
        return calibrated_depth
    
    def analyze_pothole_depth(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        calibrated: bool = True
    ) -> Dict:
        """
        Analyze depth within a pothole mask
        
        Args:
            depth_map: Depth map (calibrated or relative)
            mask: Binary mask of pothole (same size as depth_map)
            calibrated: Whether depth_map is in meters
            
        Returns:
            Dictionary with depth statistics
        """
        if depth_map.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch: depth {depth_map.shape} vs mask {mask.shape}"
            )
        
        # Extract depth values within mask
        pothole_depths = depth_map[mask]
        
        if len(pothole_depths) == 0:
            return {
                'error': 'No pixels in mask',
                'valid': False
            }
        
        # Get surrounding road surface for reference
        kernel = np.ones((25, 25), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
        surrounding = dilated.astype(bool) & ~mask
        
        if np.sum(surrounding) > 0:
            road_depth = np.median(depth_map[surrounding])
        else:
            road_depth = np.percentile(depth_map, 50)
        
        # Calculate relative depth (pothole depth relative to road surface)
        relative_depths = pothole_depths - road_depth
        
        # Pothole should be deeper (higher depth value) than road
        # If using inverse depth, flip the sign
        if np.mean(relative_depths) < 0:
            relative_depths = -relative_depths
        
        unit = "m" if calibrated else "relative"
        
        return {
            'valid': True,
            'depth_max': float(np.max(relative_depths)),
            'depth_min': float(np.min(relative_depths)),
            'depth_mean': float(np.mean(relative_depths)),
            'depth_median': float(np.median(relative_depths)),
            'depth_std': float(np.std(relative_depths)),
            'road_surface_depth': float(road_depth),
            'depth_profile': relative_depths.tolist(),
            'unit': unit,
            'calibrated': calibrated
        }
    
    def get_depth_cross_section(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        direction: str = 'horizontal'
    ) -> Dict:
        """
        Get cross-section profile through pothole
        
        Args:
            depth_map: Depth map
            mask: Pothole mask
            direction: 'horizontal' or 'vertical'
            
        Returns:
            Cross-section profile data
        """
        # Find mask center and bounds
        y_indices, x_indices = np.where(mask)
        
        if len(y_indices) == 0:
            return {'error': 'Empty mask'}
        
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        # Add some padding
        padding = 20
        
        if direction == 'horizontal':
            start_x = max(0, min_x - padding)
            end_x = min(depth_map.shape[1], max_x + padding)
            profile = depth_map[center_y, start_x:end_x]
            positions = np.arange(start_x, end_x)
            
        else:  # vertical
            start_y = max(0, min_y - padding)
            end_y = min(depth_map.shape[0], max_y + padding)
            profile = depth_map[start_y:end_y, center_x]
            positions = np.arange(start_y, end_y)
        
        return {
            'profile': profile.tolist(),
            'positions': positions.tolist(),
            'center': (center_x, center_y),
            'direction': direction,
            'pothole_bounds': (min_x, min_y, max_x, max_y)
        }
    
    def visualize_depth(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_MAGMA,
        mask: Optional[np.ndarray] = None,
        show_scale: bool = True
    ) -> np.ndarray:
        """
        Visualize depth map as colored image
        
        Args:
            depth_map: Depth map
            colormap: OpenCV colormap
            mask: Optional mask to highlight
            show_scale: Whether to add depth scale bar
            
        Returns:
            Colored depth visualization
        """
        # Normalize depth for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)
        
        # Highlight mask if provided
        if mask is not None:
            # Draw contour around mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(depth_colored, contours, -1, (0, 255, 0), 3)
        
        # Add scale bar
        if show_scale:
            h, w = depth_colored.shape[:2]
            bar_width = 30
            bar_height = int(h * 0.7)
            bar_x = w - bar_width - 20
            bar_y = int(h * 0.15)
            
            # Draw gradient bar
            for i in range(bar_height):
                color_val = int(255 * (1 - i / bar_height))
                color = cv2.applyColorMap(
                    np.array([[color_val]], dtype=np.uint8),
                    colormap
                )[0][0].tolist()
                cv2.line(
                    depth_colored,
                    (bar_x, bar_y + i),
                    (bar_x + bar_width, bar_y + i),
                    color,
                    1
                )
            
            # Add labels
            cv2.putText(depth_colored, "Deeper", (bar_x - 10, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(depth_colored, "Shallower", (bar_x - 30, bar_y + bar_height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return depth_colored
    
    def compute_volume_estimate(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        pixel_to_meter: float = 0.01
    ) -> Dict:
        """
        Estimate pothole volume
        
        Args:
            depth_map: Calibrated depth map in meters
            mask: Pothole mask
            pixel_to_meter: Scale factor for pixel to meter conversion
            
        Returns:
            Volume estimate in cubic meters
        """
        # Get depth within mask
        pothole_depths = depth_map[mask]
        
        # Get road surface level
        kernel = np.ones((25, 25), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
        surrounding = dilated.astype(bool) & ~mask
        
        if np.sum(surrounding) > 0:
            road_level = np.median(depth_map[surrounding])
        else:
            road_level = np.percentile(depth_map, 50)
        
        # Calculate depth below road surface
        relative_depths = pothole_depths - road_level
        relative_depths = np.abs(relative_depths)  # Ensure positive
        
        # Area per pixel (in square meters)
        pixel_area_m2 = pixel_to_meter ** 2
        
        # Total area
        total_pixels = np.sum(mask)
        area_m2 = total_pixels * pixel_area_m2
        
        # Volume = sum of (depth * pixel_area)
        volume_m3 = np.sum(relative_depths) * pixel_area_m2
        
        # Volume in liters (for intuitive understanding)
        volume_liters = volume_m3 * 1000
        
        return {
            'volume_m3': float(volume_m3),
            'volume_liters': float(volume_liters),
            'area_m2': float(area_m2),
            'mean_depth_m': float(np.mean(relative_depths)),
            'max_depth_m': float(np.max(relative_depths)),
            'pixel_count': int(total_pixels)
        }


if __name__ == "__main__":
    # Example usage
    print("Testing DepthEstimator...")
    
    # Initialize estimator
    estimator = DepthEstimator(
        model_name="depth-anything/Depth-Anything-V2-Large-hf",
        device="cuda"
    )
    
    # Test with sample image
    # image = Image.open("sample_pothole.jpg")
    # depth_map = estimator.estimate_depth(image)
    # calibrated = estimator.calibrate_depth(depth_map, "camera_height", camera_height=1.5)
    # print(f"Depth map shape: {depth_map.shape}")
    # print(f"Depth range: {depth_map.min():.3f} to {depth_map.max():.3f}")
