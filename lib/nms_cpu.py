"""
CPU-compatible NMS implementation for LaneATT
"""

import torch
import numpy as np


def nms_cpu(proposals, scores, overlap=0.5, top_k=200):
    """
    CPU-based Non-Maximum Suppression for lane proposals
    
    Args:
        proposals: tensor of shape [N, >=5] containing proposals
        scores: tensor of shape [N] containing confidence scores
        overlap: IoU threshold for suppression
        top_k: maximum number of detections to keep
    
    Returns:
        keep: indices of kept proposals
        num_to_keep: number of proposals to keep
        _: placeholder for compatibility
    """
    if proposals.shape[0] == 0:
        return torch.empty(0, dtype=torch.long), 0, None
    
    # Convert to numpy for easier processing
    proposals_np = proposals.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Sort by score in descending order
    order = scores_np.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0 and len(keep) < top_k:
        # Take the proposal with highest score
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
            
        # Compute IoU with remaining proposals
        ious = compute_lane_iou(proposals_np[i], proposals_np[order[1:]])
        
        # Keep only proposals with IoU < threshold
        inds = np.where(ious <= overlap)[0]
        order = order[inds + 1]  # +1 because we excluded the first element
    
    keep = torch.tensor(keep, dtype=torch.long)
    return keep, len(keep), None


def compute_lane_iou(lane1, lane2_array):
    """
    Compute IoU between one lane and an array of lanes
    
    For lane detection, we use a simplified IoU based on x-coordinate overlap
    at different y positions along the lane
    """
    if len(lane2_array.shape) == 1:
        lane2_array = lane2_array.reshape(1, -1)
    
    n_lanes = lane2_array.shape[0]
    ious = np.zeros(n_lanes)
    
    # Extract x-coordinates (assuming format: [conf1, conf2, start_y, start_x, length, x1, x2, ...])
    if lane1.shape[0] < 6:  # Not enough points
        return ious
    
    lane1_xs = lane1[5:]  # x-coordinates
    lane1_start = int(lane1[2] * (len(lane1_xs) - 1))  # start position
    lane1_length = int(lane1[4])  # length
    lane1_end = min(lane1_start + lane1_length, len(lane1_xs))
    
    for i in range(n_lanes):
        lane2 = lane2_array[i]
        if lane2.shape[0] < 6:
            continue
            
        lane2_xs = lane2[5:]
        lane2_start = int(lane2[2] * (len(lane2_xs) - 1))
        lane2_length = int(lane2[4])
        lane2_end = min(lane2_start + lane2_length, len(lane2_xs))
        
        # Compute overlap in the y-direction
        y_start = max(lane1_start, lane2_start)
        y_end = min(lane1_end, lane2_end)
        
        if y_start >= y_end:
            ious[i] = 0.0
            continue
        
        # Compute x-coordinate differences in overlapping region
        lane1_x_segment = lane1_xs[y_start:y_end]
        lane2_x_segment = lane2_xs[y_start:y_end]
        
        # Simple IoU approximation based on x-coordinate similarity
        x_diff = np.abs(lane1_x_segment - lane2_x_segment)
        
        # Consider lanes similar if x-coordinates are close (within some threshold)
        threshold = 50  # pixels
        overlap_points = np.sum(x_diff < threshold)
        union_points = len(lane1_x_segment)  # simplified union
        
        if union_points > 0:
            ious[i] = overlap_points / union_points
        else:
            ious[i] = 0.0
    
    return ious


def nms(proposals, scores, overlap=0.5, top_k=200):
    """
    Main NMS function - wrapper that chooses CPU implementation
    """
    return nms_cpu(proposals, scores, overlap, top_k)