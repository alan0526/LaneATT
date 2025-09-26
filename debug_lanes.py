#!/usr/bin/env python3
"""
Debug script to examine lane detection coordinates
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from inference import LaneATTInference


def debug_lane_coordinates():
    print("🔍 Debugging Lane Detection Coordinates")
    print("=" * 50)
    
    # Paths
    base_dir = Path(__file__).parent
    config_path = base_dir / "experiments" / "laneatt_r18_culane" / "config.yaml"
    model_path = base_dir / "experiments" / "laneatt_r18_culane" / "models" / "model_0015.pt"
    test_image = base_dir / "test_frame_example.jpg"
    
    # Check if files exist
    if not all([config_path.exists(), model_path.exists(), test_image.exists()]):
        print("❌ Required files not found")
        return
    
    # Initialize inference
    try:
        print("🔄 Loading model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        inferencer = LaneATTInference(
            str(config_path), 
            str(model_path), 
            device=device
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and process image
    import cv2
    image = cv2.imread(str(test_image))
    original_h, original_w = image.shape[:2]
    print(f"\\n📐 Image dimensions: {original_w}x{original_h}")
    
    # Preprocess
    img_tensor = inferencer.preprocess_image(image)
    print(f"📐 Model input size: {inferencer.img_w}x{inferencer.img_h}")
    
    # Run inference
    with torch.no_grad():
        predictions = inferencer.model(img_tensor, **inferencer.test_params)
        lanes = inferencer.model.decode(predictions, as_lanes=True)[0]
    
    print(f"\\n🛣️  Detected {len(lanes)} lanes")
    
    # Examine each lane's coordinates
    for i, lane in enumerate(lanes):
        print(f"\\n--- Lane {i+1} ---")
        points = lane.points
        print(f"Points shape: {points.shape}")
        print(f"X range: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
        print(f"Y range: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
        print(f"Confidence: {lane.metadata.get('conf', 'N/A')}")
        
        # Show first few points
        print("First 5 points (normalized):")
        for j in range(min(5, len(points))):
            x_norm, y_norm = points[j]
            x_pixel = int(x_norm * original_w)
            y_pixel = int(y_norm * original_h)
            print(f"  Point {j}: ({x_norm:.4f}, {y_norm:.4f}) -> ({x_pixel}, {y_pixel})")
    
    # Test drawing with debug info
    result_img = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, lane in enumerate(lanes):
        color = colors[i % len(colors)]
        points = lane.points
        
        if len(points) < 2:
            continue
        
        print(f"\\n🎨 Drawing lane {i+1} with color {color}")
        
        # Scale points to original image size
        points_scaled = points.copy()
        points_scaled[:, 0] *= original_w  # x coordinates
        points_scaled[:, 1] *= original_h  # y coordinates
        
        # Convert to integer coordinates
        points_int = points_scaled.astype(np.int32)
        
        print(f"   Scaled points range: X[{points_int[:, 0].min()}, {points_int[:, 0].max()}], Y[{points_int[:, 1].min()}, {points_int[:, 1].max()}]")
        
        # Filter out invalid points and draw
        valid_points = []
        for pt in points_int:
            x, y = pt
            if 0 <= x < original_w and 0 <= y < original_h:
                valid_points.append((x, y))
        
        print(f"   Valid points: {len(valid_points)}/{len(points_int)}")
        
        # Draw lane
        if len(valid_points) >= 2:
            for j in range(len(valid_points) - 1):
                pt1 = valid_points[j]
                pt2 = valid_points[j + 1]
                cv2.line(result_img, pt1, pt2, color, 3)
                
                # Add point markers for first few points
                if j < 3:
                    cv2.circle(result_img, pt1, 5, (255, 255, 255), -1)
                    cv2.putText(result_img, f"L{i+1}P{j}", 
                               (pt1[0]+10, pt1[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
    
    # Save debug result
    debug_output = base_dir / "debug_lanes.jpg"
    cv2.imwrite(str(debug_output), result_img)
    print(f"\\n💾 Debug image saved to: {debug_output}")
    
    # Display result
    cv2.imshow('Debug Lane Detection', result_img)
    print("\\n👁️  Press any key to close the debug image...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    debug_lane_coordinates()