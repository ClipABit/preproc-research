"""
Frame Selection Algorithms
==========================
Different strategies for selecting representative frames from video chunks.
This is the most critical component for semantic search quality.
"""

import cv2
import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate visual difference between two frames using histogram comparison.
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
    
    Returns:
        Difference score (0-1, higher = more different)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Calculate correlation (1 - correlation gives difference)
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return 1.0 - correlation


def extract_keyframes(cap: cv2.VideoCapture, start_time: float, end_time: float, 
                     fps: float, num_keyframes: int = 1) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract keyframes by selecting most representative/diverse frames.
    
    Strategy: Extract frames at regular intervals, then select the most visually
    diverse subset using clustering-based selection.
    
    Args:
        cap: OpenCV video capture object
        start_time: Chunk start time in seconds
        end_time: Chunk end time in seconds
        fps: Video frame rate
        num_keyframes: Number of keyframes to extract
    
    Returns:
        Tuple of (frames, timestamps)
    """
    frames = []
    timestamps = []
    
    # Get all frames from the chunk
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    all_frames = []
    all_timestamps = []
    
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            all_frames.append(frame)
            all_timestamps.append(frame_idx / fps)
    
    if not all_frames:
        return [], []
    
    if len(all_frames) <= num_keyframes:
        return all_frames, all_timestamps
    
    # Select diverse keyframes
    if num_keyframes == 1:
        # For single keyframe, take the middle frame
        mid_idx = len(all_frames) // 2
        return [all_frames[mid_idx]], [all_timestamps[mid_idx]]
    
    # For multiple keyframes, use diversity-based selection
    selected_indices = [0]  # Always include first frame
    
    for _ in range(num_keyframes - 1):
        max_min_distance = -1
        best_idx = -1
        
        # Find frame most different from already selected frames
        for i, frame in enumerate(all_frames):
            if i in selected_indices:
                continue
            
            # Calculate minimum distance to selected frames
            min_distance = float('inf')
            for selected_idx in selected_indices:
                distance = calculate_frame_difference(frame, all_frames[selected_idx])
                min_distance = min(min_distance, distance)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)
    
    # Sort indices and extract frames
    selected_indices.sort()
    frames = [all_frames[i] for i in selected_indices]
    timestamps = [all_timestamps[i] for i in selected_indices]
    
    return frames, timestamps


def extract_dense_frames(cap: cv2.VideoCapture, start_time: float, end_time: float,
                        fps: float, sample_fps: float = 1.0) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract frames at regular intervals (dense sampling).
    
    Args:
        cap: OpenCV video capture object
        start_time: Chunk start time in seconds
        end_time: Chunk end time in seconds
        fps: Video frame rate
        sample_fps: Desired sampling rate in frames per second
    
    Returns:
        Tuple of (frames, timestamps)
    """
    frames = []
    timestamps = []
    
    # Calculate frame interval
    frame_interval = max(1, int(fps / sample_fps))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    current_frame = start_frame
    
    while current_frame < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)
            timestamps.append(current_frame / fps)
        
        current_frame += frame_interval
    
    return frames, timestamps


def extract_adaptive_frames(cap: cv2.VideoCapture, start_time: float, end_time: float,
                           fps: float, motion_threshold: float = 15.0,
                           min_fps: float = 0.5, max_fps: float = 2.0) -> Tuple[List[np.ndarray], List[float]]:
    """
    Adaptive frame sampling based on motion/change detection.
    
    Static scenes get low sampling rate, dynamic scenes get higher sampling rate.
    
    Args:
        cap: OpenCV video capture object
        start_time: Chunk start time in seconds
        end_time: Chunk end time in seconds
        fps: Video frame rate
        motion_threshold: Threshold for considering content as dynamic (0-100)
        min_fps: Minimum sampling rate for static scenes
        max_fps: Maximum sampling rate for dynamic scenes
    
    Returns:
        Tuple of (frames, timestamps)
    """
    frames = []
    timestamps = []
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # First, analyze motion across the chunk
    motion_scores = []
    prev_frame = None
    
    for frame_idx in range(start_frame, end_frame, max(1, int(fps / 10))):  # Sample every 0.1s for analysis
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        if prev_frame is not None:
            # Calculate motion using frame difference
            diff_score = calculate_frame_difference(prev_frame, frame) * 100
            motion_scores.append(diff_score)
        
        prev_frame = frame
    
    # Determine if chunk is static or dynamic
    avg_motion = np.mean(motion_scores) if motion_scores else 0
    
    # Interpolate sampling rate based on motion
    if avg_motion < motion_threshold:
        # Static scene - use minimum sampling
        sampling_fps = min_fps
    else:
        # Dynamic scene - scale sampling rate
        motion_factor = min(1.0, avg_motion / (motion_threshold * 2))
        sampling_fps = min_fps + (max_fps - min_fps) * motion_factor
    
    # Extract frames at determined rate
    frame_interval = max(1, int(fps / sampling_fps))
    current_frame = start_frame
    
    while current_frame < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)
            timestamps.append(current_frame / fps)
        
        current_frame += frame_interval
    
    return frames, timestamps


def calculate_motion_score(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate motion score between two frames using optical flow magnitude.
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
    
    Returns:
        Motion score (higher = more motion)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Calculate magnitude
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Return average magnitude as motion score
    return np.mean(mag)


def visualize_frame_importance(frames: List[np.ndarray]) -> np.ndarray:
    """
    Create a heatmap showing visual importance/diversity of frames.
    
    Args:
        frames: List of frames
    
    Returns:
        Array of importance scores (0-1)
    """
    if len(frames) <= 1:
        return np.array([1.0] * len(frames))
    
    importance_scores = []
    
    for i, frame in enumerate(frames):
        # Calculate difference from previous and next frames
        score = 0.0
        count = 0
        
        if i > 0:
            score += calculate_frame_difference(frames[i-1], frame)
            count += 1
        
        if i < len(frames) - 1:
            score += calculate_frame_difference(frame, frames[i+1])
            count += 1
        
        importance_scores.append(score / count if count > 0 else 0.5)
    
    # Normalize scores
    scores = np.array(importance_scores)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    return scores


def extract_action_frames(cap: cv2.VideoCapture, start_time: float, end_time: float,
                         fps: float, num_frames: int = 5) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract frames during action peaks (high motion moments).
    
    Useful for sports, action sequences, or dynamic content.
    
    Args:
        cap: OpenCV video capture object
        start_time: Chunk start time in seconds
        end_time: Chunk end time in seconds
        fps: Video frame rate
        num_frames: Number of action frames to extract
    
    Returns:
        Tuple of (frames, timestamps)
    """
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Calculate motion for all frames
    motion_scores = []
    all_frames = []
    all_timestamps = []
    
    prev_frame = None
    
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        all_frames.append(frame)
        timestamp = frame_idx / fps
        all_timestamps.append(timestamp)
        
        if prev_frame is not None:
            motion = calculate_frame_difference(prev_frame, frame)
            motion_scores.append(motion)
        else:
            motion_scores.append(0.0)
        
        prev_frame = frame
    
    if len(all_frames) <= num_frames:
        return all_frames, all_timestamps
    
    # Find peaks in motion
    motion_array = np.array(motion_scores)
    peaks, _ = find_peaks(motion_array, distance=max(1, len(motion_array) // (num_frames * 2)))
    
    # If not enough peaks, take top motion frames
    if len(peaks) < num_frames:
        top_indices = np.argsort(motion_array)[-num_frames:]
        top_indices = sorted(top_indices)
    else:
        # Take top peaks
        peak_heights = motion_array[peaks]
        top_peak_indices = np.argsort(peak_heights)[-num_frames:]
        top_indices = sorted(peaks[top_peak_indices])
    
    selected_frames = [all_frames[i] for i in top_indices]
    selected_timestamps = [all_timestamps[i] for i in top_indices]
    
    return selected_frames, selected_timestamps
