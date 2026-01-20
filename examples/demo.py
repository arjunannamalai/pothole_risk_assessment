#!/usr/bin/env python3
"""
Example script demonstrating the Pothole Risk Assessment System

This script shows how to:
1. Initialize the analyzer
2. Analyze a pothole image
3. Find similar historical cases
4. Generate reports
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyzer import PotholeRiskAnalyzer


def main():
    """Main example function"""
    
    print("\n" + "=" * 80)
    print("POTHOLE RISK ASSESSMENT SYSTEM - EXAMPLE")
    print("=" * 80 + "\n")
    
    # Check for sample image
    sample_image = "sample_pothole.jpg"
    if not os.path.exists(sample_image):
        print(f"⚠️  Sample image not found: {sample_image}")
        print("Please provide a road image with potholes to analyze.")
        print("\nExample usage:")
        print("  python examples/demo.py path/to/your/image.jpg")
        
        if len(sys.argv) > 1:
            sample_image = sys.argv[1]
            if not os.path.exists(sample_image):
                print(f"\n❌ Image not found: {sample_image}")
                return
        else:
            print("\nYou can download sample images from:")
            print("  - RDD2022: https://github.com/sekilab/RoadDamageDetector")
            print("  - Kaggle: https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset")
            return
    
    print(f"📷 Input Image: {sample_image}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Initialize Analyzer
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/4] Initializing analyzer...")
    
    try:
        analyzer = PotholeRiskAnalyzer(
            sam_model_type="vit_h",  # Use vit_b for faster inference
            device="cuda"
        )
    except FileNotFoundError as e:
        print(f"\n❌ Model checkpoint not found!")
        print("Please download SAM checkpoint first:")
        print("  mkdir -p models")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth")
        return
    
    # Set calibration for typical dashcam setup
    analyzer.set_calibration(
        camera_height=1.5,      # Camera 1.5m from ground
        pixel_to_meter=0.01    # 1 pixel ≈ 1cm at typical distance
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Analyze Image
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/4] Analyzing pothole image...")
    
    results = analyzer.analyze(
        image_path=sample_image,
        location={
            'lat': 28.6139,
            'lon': 77.2090,
            'address': 'Sample Location, New Delhi, India'
        },
        save_visualization=True,
        output_dir="./outputs"
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Display Results Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    if results['status'] == 'success':
        risk = results['highest_risk']['risk_assessment']
        measurements = results['highest_risk']['measurements']
        cost = results['highest_risk']['cost_estimate']
        
        print(f"""
📊 SUMMARY
─────────────────────────────────────────────────────────
Report ID:        {results['report_id'][:8]}...
Total Potholes:   {results['total_potholes']}
Highest Risk:     Pothole #{results['highest_risk']['pothole_id']}

📏 MEASUREMENTS (Highest Risk Pothole)
─────────────────────────────────────────────────────────
Depth:            {measurements['depth_max_m']*100:.1f} cm
Area:             {measurements['area_m2']:.2f} m²
Volume:           {measurements['volume_liters']:.1f} liters

⚠️  RISK ASSESSMENT
─────────────────────────────────────────────────────────
Risk Score:       {risk['total_score']}/100
Risk Level:       {risk['risk_level']}
Priority:         {risk['priority']} of 4
Response Time:    {risk['response_time']}

💰 REPAIR COST ESTIMATE
─────────────────────────────────────────────────────────
Material:         ₹{cost['material_cost_inr']:,.0f}
Labor:            ₹{cost['labor_cost_inr']:,.0f}
Equipment:        ₹{cost['equipment_cost_inr']:,.0f}
TOTAL:            ₹{cost['total_cost_inr']:,.0f}

📁 OUTPUT FILES
─────────────────────────────────────────────────────────
""")
        for viz_type, path in results.get('visualizations', {}).items():
            print(f"  {viz_type}: {path}")
    
    else:
        print(f"\n⚠️  Analysis status: {results['status']}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Find Similar Cases
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/4] Searching for similar historical cases...")
    
    similar = analyzer.find_similar_cases(
        sample_image,
        limit=5
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Database Statistics
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/4] Database statistics...")
    
    stats = analyzer.get_statistics()
    print(f"""
📈 DATABASE STATISTICS
─────────────────────────────────────────────────────────
Total Reports:    {stats['total_reports']}
Average Score:    {stats['average_score']:.1f}
Risk Distribution: {stats['risk_distribution']}
""")
    
    print("\n✅ Demo complete!")
    print("Check the ./outputs directory for visualizations and reports.")


if __name__ == "__main__":
    main()
