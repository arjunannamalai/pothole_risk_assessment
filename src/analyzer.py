"""
Main Analyzer Module

Integrates all three stages:
1. Segmentation (SAM)
2. Depth Estimation (Depth Anything V2)
3. Risk Assessment + Qdrant Storage

This is the main entry point for analyzing pothole images.
"""

import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import uuid
import os
import yaml

from .segmentation import PotholeSegmenter, download_sam_checkpoint
from .depth_estimation import DepthEstimator
from .risk_assessment import RiskAssessor, QdrantRiskStorage


class PotholeRiskAnalyzer:
    """
    Complete pothole risk assessment pipeline
    
    Combines:
    - SAM for precise segmentation
    - Depth Anything V2 for depth estimation
    - Multi-factor risk scoring
    - Qdrant for storage and similarity search
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        sam_checkpoint: str = "models/sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf",
        embedding_model: str = "facebook/dinov2-base",
        qdrant_path: str = "./data/qdrant_db",
        device: str = "cuda"
    ):
        """
        Initialize the analyzer
        
        Args:
            config_path: Path to config.yaml (optional)
            sam_checkpoint: Path to SAM checkpoint
            sam_model_type: SAM model variant
            depth_model: Depth estimation model name
            embedding_model: Embedding model for Qdrant
            qdrant_path: Path to Qdrant database
            device: Device (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            sam_checkpoint = config.get('models', {}).get('sam', {}).get('checkpoint', sam_checkpoint)
            sam_model_type = config.get('models', {}).get('sam', {}).get('model_type', sam_model_type)
            depth_model = config.get('models', {}).get('depth', {}).get('model_name', depth_model)
            embedding_model = config.get('models', {}).get('embedding', {}).get('model_name', embedding_model)
            qdrant_path = config.get('qdrant', {}).get('path', qdrant_path)
        
        print("=" * 60)
        print("Initializing Pothole Risk Assessment System")
        print("=" * 60)
        
        # Initialize Stage 1: Segmentation
        print("\n[Stage 1] Loading Segmentation Model (SAM)...")
        if not os.path.exists(sam_checkpoint):
            print(f"SAM checkpoint not found. Downloading...")
            sam_checkpoint = download_sam_checkpoint(sam_model_type, "models")
        
        self.segmenter = PotholeSegmenter(
            checkpoint_path=sam_checkpoint,
            model_type=sam_model_type,
            device=self.device
        )
        
        # Initialize Stage 2: Depth Estimation
        print("\n[Stage 2] Loading Depth Estimation Model...")
        self.depth_estimator = DepthEstimator(
            model_name=depth_model,
            device=self.device
        )
        
        # Initialize Stage 3: Risk Assessment + Storage
        print("\n[Stage 3] Setting up Risk Assessment and Storage...")
        self.risk_assessor = RiskAssessor()
        self.storage = QdrantRiskStorage(
            qdrant_path=qdrant_path,
            collection_name="pothole_risk_assessment"
        )
        
        # Initialize embedding model for Qdrant
        print("\nLoading Embedding Model for Qdrant...")
        self._init_embedding_model(embedding_model)
        
        # Calibration settings
        self.calibration = {
            'camera_height': 1.5,
            'pixel_to_meter': 0.01,
            'lane_width': 3.5
        }
        
        print("\n" + "=" * 60)
        print("✓ System Initialized Successfully!")
        print("=" * 60 + "\n")
    
    def _init_embedding_model(self, model_name: str):
        """Initialize embedding model for Qdrant storage"""
        from transformers import AutoImageProcessor, AutoModel
        
        self.embedding_processor = AutoImageProcessor.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        print(f"✓ Embedding model loaded: {model_name}")
    
    def set_calibration(
        self,
        camera_height: Optional[float] = None,
        pixel_to_meter: Optional[float] = None,
        lane_width: Optional[float] = None
    ):
        """
        Set calibration parameters for accurate measurements
        
        Args:
            camera_height: Camera height from ground in meters
            pixel_to_meter: Scale factor for pixel to meter
            lane_width: Lane width for reference
        """
        if camera_height:
            self.calibration['camera_height'] = camera_height
        if pixel_to_meter:
            self.calibration['pixel_to_meter'] = pixel_to_meter
        if lane_width:
            self.calibration['lane_width'] = lane_width
        
        print(f"Calibration updated: {self.calibration}")
    
    def _generate_embedding(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Generate embedding for Qdrant storage"""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        inputs = self.embedding_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].squeeze().cpu().numpy()
        
        return embedding
    
    def analyze(
        self,
        image_path: str,
        location: Optional[Dict] = None,
        save_visualization: bool = True,
        output_dir: str = "./outputs"
    ) -> Dict:
        """
        Complete pothole analysis pipeline
        
        Args:
            image_path: Path to road image
            location: GPS coordinates {'lat': float, 'lon': float, 'address': str}
            save_visualization: Whether to save visualization images
            output_dir: Directory to save outputs
            
        Returns:
            Complete analysis results
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING: {image_path}")
        print(f"{'='*80}")
        
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Segmentation
        # ═══════════════════════════════════════════════════════════════
        print("\n[STAGE 1] Segmenting potholes...")
        
        # Get all masks
        all_masks = self.segmenter.segment_image(image_rgb)
        print(f"  Found {len(all_masks)} segments")
        
        # Filter for potholes
        pothole_masks = self.segmenter.filter_pothole_masks(all_masks, image_rgb)
        print(f"  Identified {len(pothole_masks)} potential potholes")
        
        if not pothole_masks:
            print("  ⚠️ No potholes detected in image")
            return {
                'status': 'no_potholes_detected',
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Depth Estimation
        # ═══════════════════════════════════════════════════════════════
        print("\n[STAGE 2] Estimating depth...")
        
        # Get depth map
        depth_map_relative = self.depth_estimator.estimate_depth(pil_image)
        
        # Calibrate to absolute depth
        depth_map = self.depth_estimator.calibrate_depth(
            depth_map_relative,
            calibration_method='camera_height',
            camera_height=self.calibration['camera_height']
        )
        print(f"  Depth map computed: {depth_map.shape}")
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: Analysis and Risk Assessment
        # ═══════════════════════════════════════════════════════════════
        print("\n[STAGE 3] Computing measurements and risk assessment...")
        
        pothole_analyses = []
        
        for idx, mask in enumerate(pothole_masks):
            print(f"\n  Processing Pothole #{idx + 1}...")
            
            segmentation = mask['segmentation']
            bbox = mask['bbox']
            
            # Ensure mask matches depth map shape
            if segmentation.shape != depth_map.shape:
                segmentation = cv2.resize(
                    segmentation.astype(np.uint8),
                    (depth_map.shape[1], depth_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            # Analyze depth within pothole
            depth_analysis = self.depth_estimator.analyze_pothole_depth(
                depth_map, segmentation, calibrated=True
            )
            
            if not depth_analysis.get('valid', False):
                print(f"    ⚠️ Invalid depth analysis, skipping")
                continue
            
            # Calculate volume
            volume_analysis = self.depth_estimator.compute_volume_estimate(
                depth_map,
                segmentation,
                pixel_to_meter=self.calibration['pixel_to_meter']
            )
            
            # Extract mask info for shape analysis
            mask_info = self.segmenter.extract_mask_info(mask, image_rgb.shape[:2])
            
            # Calculate physical dimensions
            width_px = bbox[2]
            height_px = bbox[3]
            width_m = width_px * self.calibration['pixel_to_meter']
            length_m = height_px * self.calibration['pixel_to_meter']
            area_m2 = volume_analysis['area_m2']
            
            # Compile measurements
            measurements = {
                'depth_max_m': depth_analysis['depth_max'],
                'depth_mean_m': depth_analysis['depth_mean'],
                'depth_std_m': depth_analysis['depth_std'],
                'width_m': width_m,
                'length_m': length_m,
                'area_m2': area_m2,
                'volume_m3': volume_analysis['volume_m3'],
                'volume_liters': volume_analysis['volume_liters'],
                'circularity': mask_info.get('circularity', 0.5) if mask_info else 0.5,
                'pixel_count': volume_analysis['pixel_count']
            }
            
            # Calculate risk
            risk_assessment = self.risk_assessor.calculate_total_risk(
                depth_m=measurements['depth_max_m'],
                area_m2=measurements['area_m2'],
                volume_m3=measurements['volume_m3'],
                depth_std=measurements['depth_std_m'],
                circularity=measurements['circularity']
            )
            
            # Generate cost estimate
            cost_estimate = self.risk_assessor.estimate_repair_cost(
                measurements['volume_m3']
            )
            
            pothole_analyses.append({
                'id': idx + 1,
                'bbox': bbox,
                'segmentation_shape': segmentation.shape,
                'measurements': measurements,
                'risk_assessment': risk_assessment,
                'cost_estimate': cost_estimate
            })
            
            print(f"    Depth: {measurements['depth_max_m']*100:.1f} cm")
            print(f"    Area: {measurements['area_m2']:.2f} m²")
            print(f"    Risk Score: {risk_assessment['total_score']}/100 ({risk_assessment['risk_level']})")
        
        if not pothole_analyses:
            return {
                'status': 'analysis_failed',
                'image_path': image_path,
                'timestamp': datetime.now().isoformat()
            }
        
        # ═══════════════════════════════════════════════════════════════
        # Store in Qdrant
        # ═══════════════════════════════════════════════════════════════
        print("\n[STORAGE] Saving to Qdrant...")
        
        # Generate image embedding
        embedding = self._generate_embedding(image_rgb)
        
        # Find highest risk pothole for main record
        highest_risk = max(pothole_analyses, key=lambda x: x['risk_assessment']['total_score'])
        
        report_id = self.storage.store_assessment(
            embedding=embedding,
            measurements=highest_risk['measurements'],
            risk_assessment=highest_risk['risk_assessment'],
            image_path=image_path,
            location=location,
            metadata={
                'total_potholes': len(pothole_analyses),
                'all_analyses': pothole_analyses
            }
        )
        
        print(f"  Stored with Report ID: {report_id}")
        
        # ═══════════════════════════════════════════════════════════════
        # Visualizations
        # ═══════════════════════════════════════════════════════════════
        visualizations = {}
        
        if save_visualization:
            print("\n[VISUALIZATION] Generating outputs...")
            os.makedirs(output_dir, exist_ok=True)
            
            # Segmentation visualization
            seg_vis = self.segmenter.visualize_masks(
                image_rgb, pothole_masks, alpha=0.4
            )
            seg_path = os.path.join(output_dir, f"{report_id[:8]}_segmentation.jpg")
            cv2.imwrite(seg_path, cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))
            visualizations['segmentation'] = seg_path
            
            # Depth visualization
            depth_vis = self.depth_estimator.visualize_depth(
                depth_map,
                mask=pothole_masks[0]['segmentation'] if pothole_masks else None
            )
            depth_path = os.path.join(output_dir, f"{report_id[:8]}_depth.jpg")
            cv2.imwrite(depth_path, depth_vis)
            visualizations['depth'] = depth_path
            
            print(f"  Saved visualizations to: {output_dir}")
        
        # ═══════════════════════════════════════════════════════════════
        # Generate Report
        # ═══════════════════════════════════════════════════════════════
        print("\n[REPORT] Generating assessment report...")
        
        report = self.risk_assessor.generate_report(
            pothole_id=report_id,
            measurements=highest_risk['measurements'],
            risk_assessment=highest_risk['risk_assessment'],
            location=location,
            image_path=image_path
        )
        
        print(report)
        
        # Save report
        if save_visualization:
            report_path = os.path.join(output_dir, f"{report_id[:8]}_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
            visualizations['report'] = report_path
        
        # ═══════════════════════════════════════════════════════════════
        # Return Results
        # ═══════════════════════════════════════════════════════════════
        return {
            'status': 'success',
            'report_id': report_id,
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'total_potholes': len(pothole_analyses),
            'highest_risk': {
                'pothole_id': highest_risk['id'],
                'measurements': highest_risk['measurements'],
                'risk_assessment': highest_risk['risk_assessment'],
                'cost_estimate': highest_risk['cost_estimate']
            },
            'all_potholes': pothole_analyses,
            'visualizations': visualizations
        }
    
    def find_similar_cases(
        self,
        image_path: str,
        limit: int = 5,
        risk_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Find similar historical pothole cases
        
        Args:
            image_path: Query image path
            limit: Number of results
            risk_filter: Filter by risk level (CRITICAL, HIGH, MEDIUM, LOW)
            
        Returns:
            List of similar cases
        """
        # Load and generate embedding
        image = Image.open(image_path)
        embedding = self._generate_embedding(image)
        
        # Query Qdrant
        results = self.storage.find_similar(
            query_embedding=embedding,
            limit=limit,
            risk_filter=risk_filter
        )
        
        print(f"\n🔍 Found {len(results)} similar cases:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            payload = result['payload']
            print(f"\n{i}. Similarity: {result['score']:.4f}")
            print(f"   Report ID: {payload['report_id'][:8]}...")
            print(f"   Date: {payload['timestamp']}")
            print(f"   Risk: {payload['risk_assessment']['risk_level']} ({payload['risk_assessment']['total_score']}/100)")
            print(f"   Location: {payload.get('location', {})}")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about stored assessments"""
        return self.storage.get_statistics()


def main():
    """Example usage"""
    print("\n" + "=" * 80)
    print("POTHOLE RISK ASSESSMENT SYSTEM")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = PotholeRiskAnalyzer(
        sam_model_type="vit_h",
        device="cuda"
    )
    
    # Set calibration (adjust based on your camera setup)
    analyzer.set_calibration(
        camera_height=1.5,      # Camera 1.5m from ground
        pixel_to_meter=0.01    # 1 pixel ≈ 1cm at typical distance
    )
    
    # Analyze image
    # results = analyzer.analyze(
    #     image_path="path/to/pothole_image.jpg",
    #     location={
    #         'lat': 28.6139,
    #         'lon': 77.2090,
    #         'address': 'New Delhi, India'
    #     },
    #     save_visualization=True,
    #     output_dir="./outputs"
    # )
    
    # Find similar cases
    # similar = analyzer.find_similar_cases(
    #     "path/to/pothole_image.jpg",
    #     limit=5,
    #     risk_filter="HIGH"
    # )
    
    # Get statistics
    # stats = analyzer.get_statistics()
    # print(f"\nDatabase Statistics: {stats}")


if __name__ == "__main__":
    main()
