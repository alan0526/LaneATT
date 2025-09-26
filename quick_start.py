#!/usr/bin/env python3
"""
Quick start script for LaneATT inference on your specific videos
"""

import os
import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from inference import LaneATTInference


def main():
    print("🚗 LaneATT Local Inference - Quick Start")
    print("=" * 50)
    
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
    
    # Initialize inference
    try:
        print("🔄 Loading LaneATT model...")
        inferencer = LaneATTInference(
            str(config_path), 
            str(model_path), 
            device='cuda' if torch.cuda.is_available() else 'cpu'  # Auto-detect CUDA
        )
        print("✅ Model loaded successfully!")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Ask user what they want to do
    print("Select mode:")
    print("1. Test with extracted frame from example.mp4")
    print("2. Process full example.mp4 video")
    print("3. Process image")
    print("4. Exit")
    
    while True:
        choice = input("\\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Test with frame
            test_frame = base_dir / "test_frame_example.jpg"
            if test_frame.exists():
                print(f"\\n🔄 Processing test frame: {test_frame}")
                try:
                    lanes, result_img = inferencer.infer_image(
                        str(test_frame),
                        save_path=str(base_dir / "result_test_frame.jpg"),
                        show_result=True
                    )
                    print(f"✅ Detected {len(lanes)} lanes!")
                    print(f"📁 Result saved to: result_test_frame.jpg")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Test frame not found. Run test_inference.py first.")
            
        elif choice == '2':
            # Process video
            video_path = base_dir / "example.mp4"
            if video_path.exists():
                output_path = base_dir / "output_example_with_lanes.mp4"
                print(f"\\n🔄 Processing video: {video_path}")
                print("📝 This may take a while... Press 'q' during playback to stop")
                try:
                    inferencer.infer_video(
                        str(video_path),
                        str(output_path),
                        show_result=True,
                        skip_frames=1  # Process every other frame for speed
                    )
                    print(f"✅ Video processing complete!")
                    print(f"📁 Output saved to: {output_path}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ example.mp4 not found in project directory")
        
        elif choice == '3':
            # Process custom image
            image_path = input("Enter path to image file: ").strip()
            if os.path.exists(image_path):
                output_name = f"result_{Path(image_path).stem}.jpg"
                output_path = base_dir / output_name
                print(f"\\n🔄 Processing image: {image_path}")
                try:
                    lanes, result_img = inferencer.infer_image(
                        image_path,
                        save_path=str(output_path),
                        show_result=True
                    )
                    print(f"✅ Detected {len(lanes)} lanes!")
                    print(f"📁 Result saved to: {output_path}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print(f"❌ Image file not found: {image_path}")
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")
        
        print()
        continue_choice = input("Do you want to continue? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("👋 Goodbye!")
            break


if __name__ == '__main__':
    main()