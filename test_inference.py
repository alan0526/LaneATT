#!/usr/bin/env python3
"""
Test script for LaneATT inference - tests with individual images first
"""

import os
import sys
import cv2
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from inference import LaneATTInference


def extract_frame_from_video(video_path, frame_number=0, output_path=None):
    """
    Extract a single frame from video for testing
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None
    
    # Go to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Cannot read frame {frame_number} from video")
        return None
    
    if output_path:
        cv2.imwrite(str(output_path), frame)
        print(f"Frame saved to: {output_path}")
    
    return frame


def main():
    print("=== LaneATT Test Script ===")
    print()
    
    # Paths
    base_dir = Path(__file__).parent
    config_path = base_dir / "experiments" / "laneatt_r18_culane" / "config.yaml"
    model_path = base_dir / "experiments" / "laneatt_r18_culane" / "models" / "model_0015.pt"
    
    # Check if files exist
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
        
    print(f"✅ Config: {config_path}")
    print(f"✅ Model: {model_path}")
    print()
    
    # Initialize inference
    try:
        print("🔄 Initializing LaneATT model...")
        inferencer = LaneATTInference(
            str(config_path), 
            str(model_path), 
            device='cuda'
        )
        print("✅ Model loaded successfully!")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with extracted frames first
    test_images = []
    
    # Extract frame from example.mp4 if it exists
    example_video = base_dir / "example.mp4"
    if example_video.exists():
        print("🔄 Extracting test frame from example.mp4...")
        test_frame_path = base_dir / "test_frame_example.jpg"
        frame = extract_frame_from_video(example_video, frame_number=10, output_path=test_frame_path)
        if frame is not None:
            test_images.append(("example_frame", test_frame_path))
    
    # Look for any existing image files for testing
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for ext in image_extensions:
        for img_file in base_dir.glob(f"*{ext}"):
            if img_file.name.startswith("test_frame"):
                continue  # Skip our generated test frames
            test_images.append((img_file.stem, img_file))
            break  # Just take the first one found
    
    if not test_images:
        print("⚠️  No test images found. Creating a test frame from video...")
        # Try to extract from any video file
        for video_file in base_dir.glob("*.mp4"):
            test_frame_path = base_dir / f"test_frame_{video_file.stem}.jpg"
            frame = extract_frame_from_video(video_file, frame_number=10, output_path=test_frame_path)
            if frame is not None:
                test_images.append((f"{video_file.stem}_frame", test_frame_path))
                break
    
    if not test_images:
        print("❌ No test images available")
        return
    
    print(f"📸 Found {len(test_images)} test images:")
    for name, path in test_images:
        print(f"   - {name}: {path}")
    print()
    
    # Test inference on images
    for i, (name, image_path) in enumerate(test_images, 1):
        print(f"🔄 Testing inference {i}/{len(test_images)}: {name}")
        
        try:
            # Run inference
            lanes, result_img = inferencer.infer_image(
                str(image_path),
                save_path=str(base_dir / f"result_{name}.jpg"),
                show_result=True  # Set to False if you don't want to see the image popup
            )
            
            print(f"✅ Successfully processed: {name}")
            print(f"   Detected {len(lanes)} lanes")
            
            # Print lane details
            for j, lane in enumerate(lanes):
                conf = lane.metadata.get('conf', 0)
                print(f"   Lane {j+1}: {len(lane.points)} points, confidence: {conf:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("🎉 Image testing complete!")
    print("\nIf image inference works, you can now test video inference with:")
    print("python run_inference.py")


if __name__ == '__main__':
    main()