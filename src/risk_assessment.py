"""
Stage 3: Risk Assessment System

This module provides multi-factor risk scoring for potholes based on:
- Depth (danger to vehicles)
- Size/Area (extent of damage)
- Volume (repair material needed)
- Shape irregularity (tire damage risk)

Results are stored in Qdrant for similarity search and historical analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import json


class RiskAssessor:
    """
    Multi-factor pothole risk assessment
    
    Scoring System (0-100):
    - Depth:        0-40 points (most critical for vehicle damage)
    - Area:         0-30 points (extent of road affected)
    - Volume:       0-20 points (repair complexity)
    - Irregularity: 0-10 points (edge sharpness, tire damage risk)
    """
    
    # Default thresholds (in meters for depth/volume, m² for area)
    DEFAULT_THRESHOLDS = {
        'depth': {
            'critical': 0.15,    # >15cm - immediate danger
            'high': 0.10,        # >10cm - significant risk
            'medium': 0.05,      # >5cm - moderate risk
        },
        'area': {
            'very_large': 2.0,   # >2m² - affects multiple vehicles
            'large': 1.0,        # >1m²
            'medium': 0.5,       # >0.5m²
        },
        'volume': {
            'extensive': 0.30,   # >300 liters
            'major': 0.15,       # >150 liters
            'moderate': 0.05,    # >50 liters
        }
    }
    
    # Point allocations
    POINT_WEIGHTS = {
        'depth': {'critical': 40, 'high': 30, 'medium': 20, 'low': 10},
        'area': {'very_large': 30, 'large': 20, 'medium': 15, 'small': 10},
        'volume': {'extensive': 20, 'major': 15, 'moderate': 10, 'minor': 5},
        'irregularity': {'sharp': 10, 'moderate': 7, 'smooth': 3}
    }
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize risk assessor
        
        Args:
            thresholds: Custom thresholds (uses defaults if not provided)
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
    
    def assess_depth_risk(self, depth_m: float) -> Dict:
        """
        Assess risk based on pothole depth
        
        Args:
            depth_m: Maximum depth in meters
            
        Returns:
            Risk assessment for depth factor
        """
        th = self.thresholds['depth']
        
        if depth_m > th['critical']:
            level = 'CRITICAL'
            score = self.POINT_WEIGHTS['depth']['critical']
            description = "Extremely deep - immediate vehicle damage risk"
        elif depth_m > th['high']:
            level = 'HIGH'
            score = self.POINT_WEIGHTS['depth']['high']
            description = "Deep pothole - significant damage risk"
        elif depth_m > th['medium']:
            level = 'MEDIUM'
            score = self.POINT_WEIGHTS['depth']['medium']
            description = "Moderate depth - potential damage"
        else:
            level = 'LOW'
            score = self.POINT_WEIGHTS['depth']['low']
            description = "Shallow - minor risk"
        
        return {
            'factor': 'depth',
            'value': depth_m,
            'value_display': f"{depth_m*100:.1f} cm",
            'level': level,
            'score': score,
            'max_score': 40,
            'description': description
        }
    
    def assess_area_risk(self, area_m2: float) -> Dict:
        """
        Assess risk based on pothole area
        
        Args:
            area_m2: Area in square meters
            
        Returns:
            Risk assessment for area factor
        """
        th = self.thresholds['area']
        
        if area_m2 > th['very_large']:
            level = 'VERY_LARGE'
            score = self.POINT_WEIGHTS['area']['very_large']
            description = "Massive area - affects multiple vehicles"
        elif area_m2 > th['large']:
            level = 'LARGE'
            score = self.POINT_WEIGHTS['area']['large']
            description = "Large pothole - wide impact zone"
        elif area_m2 > th['medium']:
            level = 'MEDIUM'
            score = self.POINT_WEIGHTS['area']['medium']
            description = "Medium size - localized impact"
        else:
            level = 'SMALL'
            score = self.POINT_WEIGHTS['area']['small']
            description = "Small pothole"
        
        return {
            'factor': 'area',
            'value': area_m2,
            'value_display': f"{area_m2:.2f} m²",
            'level': level,
            'score': score,
            'max_score': 30,
            'description': description
        }
    
    def assess_volume_risk(self, volume_m3: float) -> Dict:
        """
        Assess risk based on pothole volume (repair complexity)
        
        Args:
            volume_m3: Volume in cubic meters
            
        Returns:
            Risk assessment for volume factor
        """
        th = self.thresholds['volume']
        volume_liters = volume_m3 * 1000
        
        if volume_m3 > th['extensive']:
            level = 'EXTENSIVE'
            score = self.POINT_WEIGHTS['volume']['extensive']
            description = "Major repair needed - road closure likely"
        elif volume_m3 > th['major']:
            level = 'MAJOR'
            score = self.POINT_WEIGHTS['volume']['major']
            description = "Significant repair work required"
        elif volume_m3 > th['moderate']:
            level = 'MODERATE'
            score = self.POINT_WEIGHTS['volume']['moderate']
            description = "Standard repair"
        else:
            level = 'MINOR'
            score = self.POINT_WEIGHTS['volume']['minor']
            description = "Quick patch possible"
        
        return {
            'factor': 'volume',
            'value': volume_m3,
            'value_display': f"{volume_liters:.1f} liters",
            'level': level,
            'score': score,
            'max_score': 20,
            'description': description
        }
    
    def assess_irregularity_risk(self, depth_std: float, circularity: float) -> Dict:
        """
        Assess risk based on shape irregularity (edge sharpness)
        
        Sharp edges can cause:
        - Tire damage
        - Loss of vehicle control
        - Rim damage
        
        Args:
            depth_std: Standard deviation of depth within pothole
            circularity: How circular the shape is (0-1, 1=perfect circle)
            
        Returns:
            Risk assessment for irregularity factor
        """
        # Combine factors: high std + low circularity = sharp irregular edges
        irregularity_score = (depth_std * 10) + (1 - circularity) * 0.5
        
        if irregularity_score > 0.05 or circularity < 0.3:
            level = 'SHARP_EDGES'
            score = self.POINT_WEIGHTS['irregularity']['sharp']
            description = "Sharp/irregular edges - tire damage risk"
        elif irregularity_score > 0.03 or circularity < 0.5:
            level = 'MODERATE_EDGES'
            score = self.POINT_WEIGHTS['irregularity']['moderate']
            description = "Moderately irregular edges"
        else:
            level = 'SMOOTH'
            score = self.POINT_WEIGHTS['irregularity']['smooth']
            description = "Relatively smooth edges"
        
        return {
            'factor': 'irregularity',
            'value': {'depth_std': depth_std, 'circularity': circularity},
            'value_display': f"Circularity: {circularity:.2f}",
            'level': level,
            'score': score,
            'max_score': 10,
            'description': description
        }
    
    def calculate_total_risk(
        self,
        depth_m: float,
        area_m2: float,
        volume_m3: float,
        depth_std: float = 0.0,
        circularity: float = 0.5
    ) -> Dict:
        """
        Calculate comprehensive risk assessment
        
        Args:
            depth_m: Maximum depth in meters
            area_m2: Area in square meters
            volume_m3: Volume in cubic meters
            depth_std: Depth variation
            circularity: Shape circularity (0-1)
            
        Returns:
            Complete risk assessment
        """
        # Assess each factor
        depth_risk = self.assess_depth_risk(depth_m)
        area_risk = self.assess_area_risk(area_m2)
        volume_risk = self.assess_volume_risk(volume_m3)
        irregularity_risk = self.assess_irregularity_risk(depth_std, circularity)
        
        # Calculate total
        total_score = (
            depth_risk['score'] +
            area_risk['score'] +
            volume_risk['score'] +
            irregularity_risk['score']
        )
        
        # Determine overall risk level
        if total_score >= 80:
            risk_level = 'CRITICAL'
            priority = 1
            recommended_action = "IMMEDIATE CLOSURE - Emergency repair required"
            response_time = "Within 24 hours"
        elif total_score >= 60:
            risk_level = 'HIGH'
            priority = 2
            recommended_action = "URGENT REPAIR - Deploy crew within 48 hours"
            response_time = "Within 48 hours"
        elif total_score >= 40:
            risk_level = 'MEDIUM'
            priority = 3
            recommended_action = "SCHEDULED REPAIR - Include in weekly maintenance"
            response_time = "Within 1 week"
        else:
            risk_level = 'LOW'
            priority = 4
            recommended_action = "MONITOR - Include in routine inspection"
            response_time = "Within 2 weeks"
        
        return {
            'total_score': total_score,
            'max_possible_score': 100,
            'risk_level': risk_level,
            'priority': priority,
            'factors': {
                'depth': depth_risk,
                'area': area_risk,
                'volume': volume_risk,
                'irregularity': irregularity_risk
            },
            'recommended_action': recommended_action,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def estimate_repair_cost(
        self,
        volume_m3: float,
        material_cost_per_liter: float = 5.0,  # INR per liter
        labor_cost_per_hour: float = 500.0,     # INR per hour
        equipment_cost: float = 2000.0          # Fixed equipment cost
    ) -> Dict:
        """
        Estimate repair cost (for Indian context)
        
        Args:
            volume_m3: Volume to fill
            material_cost_per_liter: Cost of filling material per liter
            labor_cost_per_hour: Labor cost per hour
            equipment_cost: Fixed equipment deployment cost
            
        Returns:
            Cost estimate breakdown
        """
        volume_liters = volume_m3 * 1000
        
        # Material cost
        material_cost = volume_liters * material_cost_per_liter
        
        # Estimate labor hours based on volume
        if volume_liters > 300:
            labor_hours = 8  # Full day
        elif volume_liters > 150:
            labor_hours = 4  # Half day
        elif volume_liters > 50:
            labor_hours = 2
        else:
            labor_hours = 1
        
        labor_cost = labor_hours * labor_cost_per_hour
        
        # Total
        total_cost = material_cost + labor_cost + equipment_cost
        
        return {
            'material_cost_inr': material_cost,
            'labor_cost_inr': labor_cost,
            'equipment_cost_inr': equipment_cost,
            'total_cost_inr': total_cost,
            'estimated_labor_hours': labor_hours,
            'material_liters': volume_liters
        }
    
    def generate_report(
        self,
        pothole_id: str,
        measurements: Dict,
        risk_assessment: Dict,
        location: Optional[Dict] = None,
        image_path: Optional[str] = None
    ) -> str:
        """
        Generate human-readable report
        
        Args:
            pothole_id: Unique identifier
            measurements: Physical measurements
            risk_assessment: Risk assessment results
            location: GPS coordinates and address
            image_path: Path to original image
            
        Returns:
            Formatted report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        POTHOLE RISK ASSESSMENT REPORT                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Report ID: {pothole_id[:36]:<56} ║
║ Generated: {risk_assessment['timestamp'][:19]:<56} ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─ LOCATION ───────────────────────────────────────────────────────────────────┐
"""
        if location:
            report += f"│ GPS: {location.get('lat', 'N/A')}, {location.get('lon', 'N/A')}\n"
            report += f"│ Address: {location.get('address', 'Not specified')}\n"
        else:
            report += "│ Location: Not specified\n"
        
        report += f"""└──────────────────────────────────────────────────────────────────────────────┘

┌─ PHYSICAL MEASUREMENTS ──────────────────────────────────────────────────────┐
│                                                                              │
│   Depth (Maximum):    {measurements.get('depth_max_m', 0)*100:>6.1f} cm                                    │
│   Depth (Average):    {measurements.get('depth_mean_m', 0)*100:>6.1f} cm                                    │
│   Width:              {measurements.get('width_m', 0):>6.2f} m                                     │
│   Length:             {measurements.get('length_m', 0):>6.2f} m                                     │
│   Area:               {measurements.get('area_m2', 0):>6.2f} m²                                    │
│   Volume:             {measurements.get('volume_m3', 0)*1000:>6.1f} liters                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ RISK ASSESSMENT ────────────────────────────────────────────────────────────┐
│                                                                              │
│   ╔═══════════════════════════════════════════════════════════════════╗     │
│   ║  OVERALL RISK SCORE:  {risk_assessment['total_score']:>3}/100                                  ║     │
│   ║  RISK LEVEL:          {risk_assessment['risk_level']:<15}                          ║     │
│   ║  PRIORITY:            {risk_assessment['priority']} of 4                                   ║     │
│   ╚═══════════════════════════════════════════════════════════════════╝     │
│                                                                              │
│   Risk Factor Breakdown:                                                     │
│   ───────────────────────────────────────────────────────────────────────    │
"""
        
        for factor_name, factor_data in risk_assessment['factors'].items():
            bar_filled = int(factor_data['score'] / factor_data['max_score'] * 20)
            bar = '█' * bar_filled + '░' * (20 - bar_filled)
            report += f"│   • {factor_name.capitalize():<12} [{bar}] {factor_data['score']:>2}/{factor_data['max_score']:<2} - {factor_data['level']:<15}│\n"
        
        report += f"""│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ RECOMMENDED ACTION ─────────────────────────────────────────────────────────┐
│                                                                              │
│   {risk_assessment['recommended_action']:<64} │
│   Response Time: {risk_assessment['response_time']:<50} │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
        
        # Add cost estimate
        if 'volume_m3' in measurements:
            cost = self.estimate_repair_cost(measurements['volume_m3'])
            report += f"""
┌─ REPAIR COST ESTIMATE (INR) ─────────────────────────────────────────────────┐
│                                                                              │
│   Material ({cost['material_liters']:.0f} liters):     ₹{cost['material_cost_inr']:>8,.0f}                              │
│   Labor ({cost['estimated_labor_hours']} hours):            ₹{cost['labor_cost_inr']:>8,.0f}                              │
│   Equipment:                ₹{cost['equipment_cost_inr']:>8,.0f}                              │
│   ─────────────────────────────────────                                      │
│   TOTAL ESTIMATE:           ₹{cost['total_cost_inr']:>8,.0f}                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
"""
        
        return report


class QdrantRiskStorage:
    """
    Store and retrieve risk assessments using Qdrant vector database
    """
    
    def __init__(
        self,
        qdrant_path: str = "./data/qdrant_db",
        collection_name: str = "pothole_risk_assessment",
        embedding_dim: int = 768
    ):
        """
        Initialize Qdrant storage
        
        Args:
            qdrant_path: Path to Qdrant database
            collection_name: Name of collection
            embedding_dim: Dimension of embeddings (DINOv2-base = 768)
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
        
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Create collection if it doesn't exist
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created Qdrant collection: {collection_name}")
        except Exception:
            print(f"✓ Using existing Qdrant collection: {collection_name}")
    
    def store_assessment(
        self,
        embedding: np.ndarray,
        measurements: Dict,
        risk_assessment: Dict,
        image_path: str,
        location: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store pothole assessment in Qdrant
        
        Args:
            embedding: Image embedding from DINOv2
            measurements: Physical measurements
            risk_assessment: Risk assessment results
            image_path: Path to original image
            location: GPS coordinates
            metadata: Additional metadata
            
        Returns:
            Report ID
        """
        from qdrant_client.models import PointStruct
        
        report_id = str(uuid.uuid4())
        
        payload = {
            'report_id': report_id,
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'measurements': measurements,
            'risk_assessment': {
                'total_score': risk_assessment['total_score'],
                'risk_level': risk_assessment['risk_level'],
                'priority': risk_assessment['priority'],
                'recommended_action': risk_assessment['recommended_action']
            },
            'location': location or {},
            'metadata': metadata or {},
            'status': 'pending_review'
        }
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=report_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            ]
        )
        
        return report_id
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        risk_filter: Optional[str] = None,
        min_score: Optional[int] = None
    ) -> List[Dict]:
        """
        Find similar pothole cases
        
        Args:
            query_embedding: Query image embedding
            limit: Number of results
            risk_filter: Filter by risk level
            min_score: Minimum risk score
            
        Returns:
            List of similar cases
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        
        # Build filter
        filter_conditions = []
        
        if risk_filter:
            filter_conditions.append(
                FieldCondition(
                    key="risk_assessment.risk_level",
                    match=MatchValue(value=risk_filter)
                )
            )
        
        if min_score is not None:
            filter_conditions.append(
                FieldCondition(
                    key="risk_assessment.total_score",
                    range=Range(gte=min_score)
                )
            )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=query_filter,
            limit=limit
        ).points
        
        return [
            {
                'id': r.id,
                'score': r.score,
                'payload': r.payload
            }
            for r in results
        ]
    
    def get_statistics(self) -> Dict:
        """
        Get collection statistics
        
        Returns:
            Statistics about stored assessments
        """
        collection_info = self.client.get_collection(self.collection_name)
        
        # Get all points for analysis
        all_points = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000
        )[0]
        
        if not all_points:
            return {
                'total_reports': 0,
                'risk_distribution': {},
                'average_score': 0
            }
        
        # Analyze risk distribution
        risk_levels = {}
        scores = []
        
        for point in all_points:
            level = point.payload.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
            risk_levels[level] = risk_levels.get(level, 0) + 1
            
            score = point.payload.get('risk_assessment', {}).get('total_score', 0)
            scores.append(score)
        
        return {
            'total_reports': len(all_points),
            'risk_distribution': risk_levels,
            'average_score': np.mean(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0
        }


if __name__ == "__main__":
    # Example usage
    print("Testing RiskAssessor...")
    
    assessor = RiskAssessor()
    
    # Test assessment
    risk = assessor.calculate_total_risk(
        depth_m=0.12,
        area_m2=0.8,
        volume_m3=0.096,
        depth_std=0.03,
        circularity=0.4
    )
    
    print(f"\nRisk Assessment Results:")
    print(f"Total Score: {risk['total_score']}/100")
    print(f"Risk Level: {risk['risk_level']}")
    print(f"Recommended Action: {risk['recommended_action']}")
    
    # Generate report
    measurements = {
        'depth_max_m': 0.12,
        'depth_mean_m': 0.08,
        'width_m': 0.9,
        'length_m': 0.9,
        'area_m2': 0.8,
        'volume_m3': 0.096
    }
    
    report = assessor.generate_report(
        pothole_id="test-123",
        measurements=measurements,
        risk_assessment=risk,
        location={'lat': 28.6139, 'lon': 77.2090, 'address': 'New Delhi'}
    )
    
    print(report)
