#!/usr/bin/env python3
"""
Simple usage script for LaneATT inference with your specific video files
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from inference import LaneATTInference


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LaneATT Inference with lane data export')
    parser.add_argument('--mode', choices=['video', 'image'], default='video', help='Processing mode')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--export_format', choices=['json', 'yaml', 'both'], default='both', help='Export format for lane data')
    parser.add_argument('--skip_frames', type=int, default=0, help='Skip frames for faster processing')
    args = parser.parse_args()
    
    # If arguments provided, use them directly
    if args.input and args.output:
        run_with_args(args)
        return
    print("=== LaneATT Local Inference ===")
    print()
    
    # Paths
    base_dir = Path(__file__).parent
    config_path = base_dir / "experiments" / "laneatt_r18_culane" / "config.yaml"
    model_path = base_dir / "experiments" / "laneatt_r18_culane" / "models" / "model_0015.pt"
    
    # Test videos  
    example_video = base_dir / "example.mp4"
    driver_video = base_dir / "dataset" / "driver_23_30frame" / "05151640_0419.MP4"
    
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
    
    # Process videos
    videos_to_process = []
    
    if example_video.exists():
        videos_to_process.append(("example.mp4", example_video))
    else:
        print(f"⚠️  Example video not found: {example_video}")
    
    # Note: The driver video path structure looks like it contains frames, not the actual video
    # Let's check for the actual video file
    possible_driver_paths = [
        base_dir / "dataset" / "driver_23_30frame" / "05151640_0419.MP4",
        base_dir / "dataset" / "05151640_0419.MP4", 
        base_dir / "05151640_0419.MP4"
    ]
    
    driver_video_found = False
    for path in possible_driver_paths:
        if path.exists() and path.is_file() and path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            videos_to_process.append(("05151640_0419.MP4", path))
            driver_video_found = True
            break
    
    if not driver_video_found:
        print(f"⚠️  Driver video file not found in expected locations")
        print("    Checked paths:")
        for path in possible_driver_paths:
            print(f"    - {path}")
    
    if not videos_to_process:
        print("❌ No valid video files found to process")
        return
    
    print(f"📹 Found {len(videos_to_process)} videos to process:")
    for name, path in videos_to_process:
        print(f"   - {name}: {path}")
    print()
    
    # Process each video
    for i, (name, video_path) in enumerate(videos_to_process, 1):
        print(f"🔄 Processing video {i}/{len(videos_to_process)}: {name}")
        
        # Create output path
        output_path = base_dir / f"output_{name}"
        
        try:
            inferencer.infer_video(
                str(video_path),
                str(output_path),
                show_result=True,  # Set to False if you don't want to see the video playback
                skip_frames=0,  # Process every frame, set to 1 to process every other frame for speed
                export_format='both'  # Export lane data in both JSON and YAML
            )
            print(f"✅ Successfully processed: {name}")
            print(f"   Output saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
        
        print()
    
    print("🎉 Processing complete!")
    print("\nUsage tips:")
    print("- Press 'q' while video is playing to stop processing")
    print("- Output videos are saved in the same directory")
    print("- You can modify skip_frames parameter to process faster (skip frames)")


def run_with_args(args):
    """Run inference with command line arguments"""
    base_dir = Path(__file__).parent
    config_path = base_dir / "experiments" / "laneatt_r18_culane" / "config.yaml"
    model_path = base_dir / "experiments" / "laneatt_r18_culane" / "models" / "model_0015.pt"
    
    print(f"=== LaneATT {args.mode.upper()} Inference ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Export format: {args.export_format}")
    print()
    
    # Initialize inference
    try:
        inferencer = LaneATTInference(str(config_path), str(model_path), device='cuda')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    try:
        if args.mode == 'video':
            inferencer.infer_video(
                args.input,
                args.output,
                show_result=True,
                skip_frames=args.skip_frames,
                export_format=args.export_format
            )
        else:  # image mode
            lanes, result_img = inferencer.infer_image(
                args.input,
                save_path=args.output,
                show_result=True,
                export_format=args.export_format
            )
            print(f"Detected {len(lanes)} lanes")
        
        print(f"✅ Processing complete! Output saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")


if __name__ == '__main__':
    main()