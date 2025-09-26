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
import json
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
    
    def filter_duplicate_lanes(self, lanes, min_distance=150):
        """
        Filter out duplicate/overlapping lanes based on spatial distance
        
        Args:
            lanes: List of detected lanes
            min_distance: Minimum pixel distance between lane centers
            
        Returns:
            Filtered list of lanes
        """
        if len(lanes) <= 1:
            return lanes
        
        # Calculate lane center positions
        lane_centers = []
        for lane in lanes:
            points = lane.points
            if len(points) > 0:
                # Calculate average x position across all points
                avg_x = np.mean(points[:, 0])
                lane_centers.append(avg_x)
            else:
                lane_centers.append(0)
        
        # Sort lanes by confidence (highest first)
        lane_conf_pairs = [(i, lanes[i].metadata.get('conf', 0)) for i in range(len(lanes))]
        lane_conf_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter duplicates
        filtered_indices = []
        for i, conf in lane_conf_pairs:
            current_center = lane_centers[i]
            
            # Check if this lane is too close to any already selected lane
            is_duplicate = False
            for j in filtered_indices:
                existing_center = lane_centers[j]
                distance = abs(current_center - existing_center)
                
                # Convert normalized distance to pixels for comparison
                if distance < (min_distance / 1280):  # Assuming 1280 pixel width
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_indices.append(i)
        
        # Return filtered lanes sorted by position (left to right)
        filtered_lanes = [lanes[i] for i in filtered_indices]
        filtered_lanes.sort(key=lambda lane: np.mean(lane.points[:, 0]))
        
        return filtered_lanes
        
    def smart_lane_selection(self, input_lanes, target_count=3):
        """
        Intelligently select lanes to better cover the road width
        
        Args:
            lanes: List of detected lanes
            target_count: Target number of lanes to return
            
        Returns:
            Selected lanes with better distribution
        """
        if len(input_lanes) <= target_count:
            return input_lanes
        
        # Calculate lane positions and confidence
        lane_info = []
        for i, lane in enumerate(input_lanes):
            points = lane.points
            if len(points) > 0:
                x_center = np.mean(points[:, 0])
                x_min = np.min(points[:, 0])
                x_max = np.max(points[:, 0])
                conf = lane.metadata.get('conf', 0)
                lane_info.append({
                    'index': i,
                    'center': x_center,
                    'min': x_min, 
                    'max': x_max,
                    'conf': conf,
                    'lane': lane
                })
        
        if len(lane_info) == 0:
            return input_lanes
        
        # Sort by position (left to right)
        lane_info.sort(key=lambda x: x['center'])
        
        # Select lanes with better distribution
        selected = []
        
        # Always take the highest confidence lane
        best_conf_lane = max(lane_info, key=lambda x: x['conf'])
        selected.append(best_conf_lane)
        
        # Try to find lanes that are sufficiently different
        for candidate in lane_info:
            if candidate in selected:
                continue
                
            # Check distance from already selected lanes
            too_close = False
            for selected_lane in selected:
                distance = abs(candidate['center'] - selected_lane['center'])
                if distance < 0.15:  # Minimum 15% of image width separation
                    too_close = True
                    break
            
            if not too_close:
                selected.append(candidate)
                if len(selected) >= target_count:
                    break
        
        # If we need more lanes and have space, add the next best by confidence
        while len(selected) < target_count and len(selected) < len(lane_info):
            remaining = [l for l in lane_info if l not in selected]
            if remaining:
                best_remaining = max(remaining, key=lambda x: x['conf'])
                selected.append(best_remaining)
        
        # Return the lanes sorted by position
        selected.sort(key=lambda x: x['center'])
        return [item['lane'] for item in selected]
    
    def export_lane_info(self, lanes, image_path, original_size, format='json'):
        """
        Export lane information to JSON or YAML format
        
        Args:
            lanes: List of detected lanes
            image_path: Path to source image
            original_size: (width, height) of original image
            format: 'json' or 'yaml'
            
        Returns:
            Dictionary containing lane information
        """
        lane_data = {
            'image_info': {
                'path': str(image_path),
                'width': original_size[0],
                'height': original_size[1],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'detection_info': {
                'total_lanes': len(lanes),
                'model_input_size': [self.img_w, self.img_h],
                'confidence_threshold': self.test_params.get('conf_threshold', 0.5),
                'nms_threshold': self.test_params.get('nms_thres', 50.0)
            },
            'lanes': []
        }
        
        for i, lane in enumerate(lanes):
            points = lane.points
            confidence = lane.metadata.get('conf', 0)
            start_x = lane.metadata.get('start_x', 0)
            start_y = lane.metadata.get('start_y', 0)
            
            # Convert normalized coordinates to pixel coordinates
            pixel_points = []
            for point in points:
                x_pixel = int(point[0] * original_size[0])
                y_pixel = int(point[1] * original_size[1])
                pixel_points.append([x_pixel, y_pixel])
            
            # Calculate lane statistics
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            lane_info = {
                'lane_id': i + 1,
                'confidence': float(confidence),
                'total_points': len(points),
                'coordinates': {
                    'normalized': points.tolist(),  # Normalized [0,1] coordinates
                    'pixel': pixel_points           # Pixel coordinates
                },
                'statistics': {
                    'x_range_normalized': [float(x_coords.min()), float(x_coords.max())],
                    'y_range_normalized': [float(y_coords.min()), float(y_coords.max())],
                    'x_range_pixel': [int(x_coords.min() * original_size[0]), int(x_coords.max() * original_size[0])],
                    'y_range_pixel': [int(y_coords.min() * original_size[1]), int(y_coords.max() * original_size[1])],
                    'center_x_normalized': float(np.mean(x_coords)),
                    'center_y_normalized': float(np.mean(y_coords)),
                    'center_x_pixel': int(np.mean(x_coords) * original_size[0]),
                    'center_y_pixel': int(np.mean(y_coords) * original_size[1]),
                    'length_pixels': len(points)
                },
                'metadata': {
                    'start_x': float(start_x),
                    'start_y': float(start_y)
                }
            }
            
            lane_data['lanes'].append(lane_info)
        
        return lane_data
    
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
    
    def infer_image(self, image_path, save_path=None, show_result=True, export_format=None):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save result (optional)
            show_result: Whether to display result
            export_format: Export lane data format ('json', 'yaml', 'both', or None)
            
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
        
        # Filter duplicate lanes and apply smart selection
        filtered_lanes = self.filter_duplicate_lanes(lanes, min_distance=150)  # Increased from 40 to 150
        lanes = self.smart_lane_selection(filtered_lanes, target_count=2)  # Reduced from 3 to 2
        
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
            
            # Export lane information if requested
            if export_format in ['json', 'yaml', 'both']:
                base_path = Path(save_path)
                
                # Export lane data
                lane_data = self.export_lane_info(lanes, image_path, (original_w, original_h))
                
                if export_format in ['json', 'both']:
                    json_path = base_path.with_suffix('.json')
                    with open(json_path, 'w') as f:
                        json.dump(lane_data, f, indent=2)
                    print(f"Lane data (JSON) saved to: {json_path}")
                
                if export_format in ['yaml', 'both']:
                    yaml_path = base_path.with_suffix('.yaml')
                    with open(yaml_path, 'w') as f:
                        yaml.dump(lane_data, f, default_flow_style=False, indent=2)
                    print(f"Lane data (YAML) saved to: {yaml_path}")
        
        # Show result
        if show_result:
            cv2.imshow('LaneATT Result', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return lanes, result_img
    
    def infer_video(self, video_path, output_path=None, show_result=True, skip_frames=0, export_format=None):
        """
        Run inference on video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_result: Whether to display results
            skip_frames: Number of frames to skip between processing
            export_format: Export lane data format ('json', 'yaml', 'both', or None)
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
                    
                    # Filter duplicate lanes and apply smart selection
                    filtered_lanes = self.filter_duplicate_lanes(lanes, min_distance=150)  # Increased filtering
                    lanes = self.smart_lane_selection(filtered_lanes, target_count=2)  # Reduced target lanes
                    
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
            
            # Export lane information for the last frame if requested
            if export_format in ['json', 'yaml', 'both'] and processed_count > 0:
                print("Exporting sample lane data from last processed frame...")
                # Re-read the last frame for lane data export
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, last_frame = cap.read()
                if ret:
                    # Run inference on last frame
                    img_tensor = self.preprocess_image(last_frame)
                    with torch.no_grad():
                        predictions = self.model(img_tensor, **self.test_params)
                        lanes = self.model.decode(predictions, as_lanes=True)[0]
                    
                    # Filter lanes
                    filtered_lanes = self.filter_duplicate_lanes(lanes, min_distance=150)
                    lanes = self.smart_lane_selection(filtered_lanes, target_count=2)
                    
                    # Export lane data
                    base_path = Path(output_path)
                    sample_frame_name = f"{base_path.stem}_sample_frame"
                    
                    lane_data = self.export_lane_info(lanes, f"{video_path}_frame_{frame_count}", (width, height))
                    
                    if export_format in ['json', 'both']:
                        json_path = base_path.parent / f"{sample_frame_name}.json"
                        with open(json_path, 'w') as f:
                            json.dump(lane_data, f, indent=2)
                        print(f"Sample lane data (JSON) saved to: {json_path}")
                    
                    if export_format in ['yaml', 'both']:
                        yaml_path = base_path.parent / f"{sample_frame_name}.yaml"
                        with open(yaml_path, 'w') as f:
                            yaml.dump(lane_data, f, default_flow_style=False, indent=2)
                        print(f"Sample lane data (YAML) saved to: {yaml_path}")


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