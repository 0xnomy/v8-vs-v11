# YOLOv11 vs YOLOv8 Performance Analysis

A comparative study of YOLOv11 and YOLOv8 object detection models, evaluating their performance on the COCO dataset and real-time applications.

## Overview

This project evaluates and compares the nano variants of YOLOv11 and YOLOv8, focusing on accuracy, speed, and practical deployment considerations.

## Key Findings

### Performance Metrics (COCO Dataset)

| Metric | YOLOv8n | YOLOv11n | Improvement |
|--------|---------|-----------|-------------|
| mAP@0.5 | 0.607 | 0.671 | +10.6% |
| mAP@0.5:0.95 | 0.448 | 0.505 | +12.8% |
| Precision | 0.639 | 0.663 | +3.9% |
| Recall | 0.536 | 0.589 | +9.9% |
| Speed (ms/img) | 48.0 | 48.8 | +1.5% |
| FPS | 21 | 21 | -1.5% |

### Key Improvements in YOLOv11

- Enhanced feature extraction with lighter convolutional structures
- Improved small and overlapping object detection
- Better regularization and augmentation techniques
- Deployment-ready design for edge hardware

## Repository Structure

```
├── run.py           # Main execution script
├── evaluate.py      # Evaluation script
├── coco.yaml        # COCO dataset configuration
├── yolov8n.pt      # YOLOv8 nano model weights
├── yolo11n.pt      # YOLOv11 nano model weights
└── annotations/     # COCO dataset annotations
```

## Conclusion

YOLOv11n shows significant improvements in accuracy (+10-13% mAP) while maintaining comparable speed to YOLOv8n. It's recommended for most production scenarios with GPU support, while YOLOv8n remains viable for resource-constrained deployments.
