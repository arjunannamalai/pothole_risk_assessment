# Pothole Risk Assessment System

A comprehensive AI-powered system for detecting, measuring, and assessing road potholes on Indian roads.

## 🎯 Features

- **Precise Segmentation**: Uses Meta's Segment Anything Model (SAM) for pixel-level pothole detection
- **Depth Estimation**: Leverages Depth Anything V2 for monocular depth estimation
- **Multi-Factor Risk Scoring**: Comprehensive risk assessment based on depth, area, volume, and shape
- **Similarity Search**: Qdrant vector database for finding similar historical cases
- **Cost Estimation**: Repair cost estimation tailored for Indian context
- **Detailed Reports**: Auto-generated reports for city officials

## 🏗️ Architecture

```
Input Image → Stage 1: Segmentation (SAM) → Stage 2: Depth Estimation → Stage 3: Risk Assessment
                     ↓                              ↓                           ↓
              Pixel-level mask              Depth map (meters)          Priority Score (0-100)
                                                                               ↓
                                                                    Qdrant Storage + Reports
```

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pothole_risk_assessment.git
cd pothole_risk_assessment
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download SAM checkpoint
```bash
mkdir -p models
# Download SAM ViT-H (2.4GB) - highest quality
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth

# Or use smaller models:
# ViT-L (1.2GB): wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# ViT-B (375MB): wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 🚀 Quick Start

### Python API

```python
from src.analyzer import PotholeRiskAnalyzer

# Initialize analyzer
analyzer = PotholeRiskAnalyzer(
    sam_model_type="vit_h",
    device="cuda"  # or "cpu"
)

# Set camera calibration
analyzer.set_calibration(
    camera_height=1.5,    # Camera height from ground (meters)
    pixel_to_meter=0.01   # Scale factor
)

# Analyze an image
results = analyzer.analyze(
    image_path="road_image.jpg",
    location={
        'lat': 28.6139,
        'lon': 77.2090,
        'address': 'New Delhi, India'
    },
    save_visualization=True,
    output_dir="./outputs"
)

# Print results
print(f"Risk Score: {results['highest_risk']['risk_assessment']['total_score']}/100")
print(f"Risk Level: {results['highest_risk']['risk_assessment']['risk_level']}")
print(f"Estimated Depth: {results['highest_risk']['measurements']['depth_max_m']*100:.1f} cm")

# Find similar historical cases
similar = analyzer.find_similar_cases("road_image.jpg", limit=5)
```

### Using Individual Modules

```python
# Stage 1: Segmentation only
from src.segmentation import PotholeSegmenter

segmenter = PotholeSegmenter(
    checkpoint_path="models/sam_vit_h_4b8939.pth",
    model_type="vit_h"
)
masks = segmenter.segment_image(image)
pothole_masks = segmenter.filter_pothole_masks(masks, image)

# Stage 2: Depth estimation only
from src.depth_estimation import DepthEstimator

depth_estimator = DepthEstimator()
depth_map = depth_estimator.estimate_depth(image)
calibrated_depth = depth_estimator.calibrate_depth(depth_map, "camera_height", camera_height=1.5)

# Stage 3: Risk assessment only
from src.risk_assessment import RiskAssessor

assessor = RiskAssessor()
risk = assessor.calculate_total_risk(
    depth_m=0.12,
    area_m2=0.8,
    volume_m3=0.096
)
```

## 📊 Risk Scoring System

| Factor | Max Points | Description |
|--------|------------|-------------|
| Depth | 40 | Deeper potholes cause more vehicle damage |
| Area | 30 | Larger potholes affect more road users |
| Volume | 20 | Indicates repair complexity |
| Irregularity | 10 | Sharp edges increase tire damage risk |

### Risk Levels

| Score | Level | Response Time |
|-------|-------|---------------|
| 80-100 | CRITICAL | Within 24 hours |
| 60-79 | HIGH | Within 48 hours |
| 40-59 | MEDIUM | Within 1 week |
| 0-39 | LOW | Within 2 weeks |

## 📁 Project Structure

```
pothole_risk_assessment/
├── src/
│   ├── __init__.py
│   ├── segmentation.py      # Stage 1: SAM segmentation
│   ├── depth_estimation.py  # Stage 2: Depth Anything V2
│   ├── risk_assessment.py   # Stage 3: Risk scoring + Qdrant
│   └── analyzer.py          # Main pipeline
├── models/                  # Model checkpoints
├── data/
│   └── qdrant_db/          # Qdrant database
├── outputs/                 # Generated reports and visualizations
├── config.yaml              # Configuration
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
models:
  sam:
    checkpoint: "models/sam_vit_h_4b8939.pth"
    model_type: "vit_h"
  depth:
    model_name: "depth-anything/Depth-Anything-V2-Large-hf"

calibration:
  camera_height_m: 1.5
  pixel_to_meter: 0.01

risk_assessment:
  depth_thresholds:
    critical: 0.15  # 15cm
    high: 0.10      # 10cm
    medium: 0.05    # 5cm
```

## 📱 Integration Ideas

- **Mobile App**: Citizens photograph potholes → instant risk assessment
- **Dashboard**: City officials monitor high-risk areas
- **API**: Integrate with existing civic tech platforms
- **Alert System**: Automatic notifications for critical potholes

## 🗃️ Datasets for Training/Testing

- [RDD2022 India Subset](https://github.com/sekilab/RoadDamageDetector) - 7,706 images
- [Indian Road Damage Dataset](https://data.mendeley.com/datasets/t576ydh9v8/1) - 5,000+ images
- [Kaggle Pothole Dataset](https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [Segment Anything (Meta)](https://github.com/facebookresearch/segment-anything)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Qdrant Vector Database](https://qdrant.tech/)
- [DINOv2 (Meta)](https://github.com/facebookresearch/dinov2)
