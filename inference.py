#!/usr/bin/env python3
"""
LaneATT Local Inference Script
Supports both video and image inference modes
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import time

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.config import Config
from lib.models.laneatt import LaneATT
from lib.lane import Lane


class LaneATTInference:
    def __init__(self, config_path, model_path, device='cuda'):
        """
        Initialize LaneATT inference
        
        Args:
            config_path: Path to model configuration file  
            model_path: Path to trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.cfg = Config(config_path)
        print(f"Loaded config from: {config_path}")
        
        # Load model
        self.model = self.cfg.get_model()
        print(f"Loading model weights from: {model_path}")
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model state dict if it's a training checkpoint
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get test parameters
        self.test_params = self.cfg.get_test_parameters()
        print(f"Test parameters: {self.test_params}")
        
        # Image preprocessing parameters
        self.img_h = self.cfg['model']['parameters']['img_h']
        self.img_w = self.cfg['model']['parameters']['img_w']
        print(f"Model input size: {self.img_w}x{self.img_h}")
        
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Resize image
        img_resized = cv2.resize(image, (self.img_w, self.img_h))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def draw_lanes_on_image(self, image, lanes, thickness=3):
        """
        Draw detected lanes on image
        
        Args:
            image: Original image
            lanes: List of detected lanes
            thickness: Line thickness for drawing
            
        Returns:
            Image with lanes drawn
        """
        if not lanes:
            return image
            
        # Create a copy of the image
        result_img = image.copy()
        
        # Define colors for different lanes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw each lane
        for i, lane in enumerate(lanes):
            color = colors[i % len(colors)]
            points = lane.points
            
            if len(points) < 2:
                continue
                
            # The points are already in normalized coordinates [0, 1]
            # Scale them to original image size
            original_h, original_w = image.shape[:2]
            points_scaled = points.copy()
            points_scaled[:, 0] *= original_w  # x coordinates
            points_scaled[:, 1] *= original_h  # y coordinates
            
            # Convert to integer coordinates
            points_int = points_scaled.astype(np.int32)
            
            # Filter out invalid points (outside image bounds)
            valid_points = []
            for pt in points_int:
                x, y = pt
                if 0 <= x < original_w and 0 <= y < original_h:
                    valid_points.append((x, y))
            
            # Draw lane as connected line segments
            if len(valid_points) >= 2:
                for j in range(len(valid_points) - 1):
                    pt1 = valid_points[j]
                    pt2 = valid_points[j + 1]
                    cv2.line(result_img, pt1, pt2, color, thickness)
        
        return result_img
    
    def infer_image(self, image_path, save_path=None, show_result=True):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save result (optional)
            show_result: Whether to display result
            
        Returns:
            Detected lanes and result image
        """
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_h, original_w = image.shape[:2]
        print(f"Original image size: {original_w}x{original_h}")
        
        # Preprocess
        img_tensor = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = self.model(img_tensor, **self.test_params)
            lanes = self.model.decode(predictions, as_lanes=True)[0]
        
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f}s")
        print(f"Detected {len(lanes)} lanes")
        
        # Draw results
        result_img = self.draw_lanes_on_image(image, lanes)
        
        # Add info text
        cv2.putText(result_img, f"Lanes: {len(lanes)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_img, f"Time: {inference_time:.3f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save result
        if save_path:
            cv2.imwrite(str(save_path), result_img)
            print(f"Result saved to: {save_path}")
        
        # Show result
        if show_result:
            cv2.imshow('LaneATT Result', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return lanes, result_img
    
    def infer_video(self, video_path, output_path=None, show_result=True, skip_frames=0):
        """
        Run inference on video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_result: Whether to display results
            skip_frames: Number of frames to skip between processing
        """
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        # Process video
        frame_count = 0
        processed_count = 0
        
        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames if specified
                    if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                        frame_count += 1
                        pbar.update(1)
                        continue
                    
                    # Run inference
                    img_tensor = self.preprocess_image(frame)
                    
                    with torch.no_grad():
                        predictions = self.model(img_tensor, **self.test_params)
                        lanes = self.model.decode(predictions, as_lanes=True)[0]
                    
                    # Draw results
                    result_frame = self.draw_lanes_on_image(frame, lanes)
                    
                    # Add frame info
                    cv2.putText(result_frame, f"Frame: {frame_count+1}/{total_frames}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Lanes: {len(lanes)}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Write frame to output video
                    if writer:
                        writer.write(result_frame)
                    
                    # Show result
                    if show_result:
                        cv2.imshow('LaneATT Video', result_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Processing interrupted by user")
                            break
                    
                    frame_count += 1
                    processed_count += 1
                    pbar.update(1)
                    
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if show_result:
                cv2.destroyAllWindows()
        
        print(f"Processed {processed_count} frames")
        if output_path:
            print(f"Output video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LaneATT Inference Script')
    parser.add_argument('--mode', choices=['image', 'video'], required=True,
                       help='Inference mode: image or video')
    parser.add_argument('--input', required=True,
                       help='Path to input image or video')
    parser.add_argument('--config', required=True,
                       help='Path to model config file')
    parser.add_argument('--model', required=True,
                       help='Path to model weights file')
    parser.add_argument('--output', 
                       help='Path to save output (optional)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display results')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Number of frames to skip in video processing (for speed)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
        
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Initialize inference
    try:
        inferencer = LaneATTInference(args.config, args.model, args.device)
    except Exception as e:
        print(f"Error initializing inference: {e}")
        return
    
    # Run inference
    try:
        if args.mode == 'image':
            lanes, result_img = inferencer.infer_image(
                args.input, 
                args.output, 
                show_result=not args.no_show
            )
            print(f"Inference completed successfully!")
            
        elif args.mode == 'video':
            inferencer.infer_video(
                args.input,
                args.output,
                show_result=not args.no_show,
                skip_frames=args.skip_frames
            )
            print(f"Video processing completed successfully!")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return


if __name__ == '__main__':
    main()