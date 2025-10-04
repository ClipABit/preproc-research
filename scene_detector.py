"""
Scene Detection Module
======================
Implements scene detection for semantic video chunking using PySceneDetect.
"""

import cv2
import numpy as np
from typing import List, Tuple
from scenedetect import detect, ContentDetector
import tempfile


def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """
    Detect scene boundaries in a video using content-based detection.
    
    Args:
        video_path: Path to video file
        threshold: Detection threshold (lower = more sensitive, default=27)
    
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    try:
        # Use modern scenedetect API (VideoManager is deprecated)
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        
        # Convert to (start_time, end_time) tuples
        scenes = []
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append((start_time, end_time))
        
        # If no scenes detected, return entire video as one scene
        if not scenes:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            scenes = [(0.0, duration)]
        
        return scenes
    
    except Exception as e:
        print(f"Scene detection failed: {e}")
        # Fallback: return entire video as single scene
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return [(0.0, duration)]


def detect_scenes_simple(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """
    Simple scene detection using frame difference (fallback if PySceneDetect unavailable).
    
    Args:
        video_path: Path to video file
        threshold: Difference threshold for scene change (0-100)
    
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return [(0.0, 0.0)]
    
    scene_boundaries = [0]  # Start with first frame
    prev_frame = None
    
    for frame_idx in range(0, frame_count, max(1, int(fps))):  # Sample every second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        if prev_frame is not None:
            # Calculate histogram difference
            diff = calculate_histogram_difference(prev_frame, frame)
            
            if diff * 100 > threshold:
                scene_boundaries.append(frame_idx)
        
        prev_frame = frame
    
    scene_boundaries.append(frame_count - 1)  # End with last frame
    cap.release()
    
    # Convert frame indices to time ranges
    scenes = []
    for i in range(len(scene_boundaries) - 1):
        start_time = scene_boundaries[i] / fps
        end_time = scene_boundaries[i + 1] / fps
        
        # Filter out very short scenes (< 0.5 seconds)
        if end_time - start_time >= 0.5:
            scenes.append((start_time, end_time))
    
    return scenes if scenes else [(0.0, frame_count / fps)]


def calculate_histogram_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate histogram difference between frames in HSV color space.
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
    
    Returns:
        Difference score (0-1)
    """
    # Convert to HSV
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for each channel
    total_diff = 0.0
    
    for i in range(3):
        hist1 = cv2.calcHist([hsv1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([hsv2], [i], None, [256], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Bhattacharyya distance
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        total_diff += corr
    
    return total_diff / 3.0


def visualize_scene_timeline(scenes: List[Tuple[float, float]], total_duration: float) -> str:
    """
    Create ASCII visualization of scene timeline.
    
    Args:
        scenes: List of (start_time, end_time) tuples
        total_duration: Total video duration in seconds
    
    Returns:
        ASCII timeline string
    """
    width = 80
    timeline = ['-'] * width
    
    for scene in scenes:
        start_pos = int((scene[0] / total_duration) * width)
        end_pos = int((scene[1] / total_duration) * width)
        
        start_pos = max(0, min(width - 1, start_pos))
        end_pos = max(0, min(width - 1, end_pos))
        
        timeline[start_pos] = '['
        if end_pos < width:
            timeline[end_pos] = ']'
        
        for i in range(start_pos + 1, end_pos):
            if i < width:
                timeline[i] = '='
    
    return ''.join(timeline)


def merge_short_scenes(scenes: List[Tuple[float, float]], 
                      min_duration: float = 3.0) -> List[Tuple[float, float]]:
    """
    Merge scenes that are shorter than minimum duration.
    
    Args:
        scenes: List of (start_time, end_time) tuples
        min_duration: Minimum scene duration in seconds
    
    Returns:
        Merged scene list
    """
    if not scenes:
        return []
    
    merged = []
    current_start = scenes[0][0]
    current_end = scenes[0][1]
    
    for i in range(1, len(scenes)):
        scene_start, scene_end = scenes[i]
        
        # Check if current accumulated duration is below minimum
        if current_end - current_start < min_duration:
            # Merge with next scene
            current_end = scene_end
        else:
            # Save current scene and start new one
            merged.append((current_start, current_end))
            current_start = scene_start
            current_end = scene_end
    
    # Add last scene
    merged.append((current_start, current_end))
    
    return merged


def split_long_scenes(scenes: List[Tuple[float, float]], 
                     max_duration: float = 20.0) -> List[Tuple[float, float]]:
    """
    Split scenes that exceed maximum duration.
    
    Args:
        scenes: List of (start_time, end_time) tuples
        max_duration: Maximum scene duration in seconds
    
    Returns:
        Split scene list
    """
    split_scenes = []
    
    for start_time, end_time in scenes:
        duration = end_time - start_time
        
        if duration <= max_duration:
            split_scenes.append((start_time, end_time))
        else:
            # Split into equal chunks
            num_chunks = int(np.ceil(duration / max_duration))
            chunk_duration = duration / num_chunks
            
            for i in range(num_chunks):
                chunk_start = start_time + (i * chunk_duration)
                chunk_end = min(chunk_start + chunk_duration, end_time)
                split_scenes.append((chunk_start, chunk_end))
    
    return split_scenes


def apply_scene_constraints(scenes: List[Tuple[float, float]], 
                           min_duration: float = 3.0,
                           max_duration: float = 20.0) -> List[Tuple[float, float]]:
    """
    Apply both minimum and maximum duration constraints to scenes.
    
    This implements the "Hybrid" chunking strategy from the report.
    
    Args:
        scenes: List of (start_time, end_time) tuples
        min_duration: Minimum scene duration in seconds
        max_duration: Maximum scene duration in seconds
    
    Returns:
        Constrained scene list
    """
    # First merge short scenes
    merged = merge_short_scenes(scenes, min_duration)
    
    # Then split long scenes
    constrained = split_long_scenes(merged, max_duration)
    
    return constrained


def get_scene_statistics(scenes: List[Tuple[float, float]]) -> dict:
    """
    Calculate statistics about detected scenes.
    
    Args:
        scenes: List of (start_time, end_time) tuples
    
    Returns:
        Dictionary with scene statistics
    """
    if not scenes:
        return {
            'count': 0,
            'avg_duration': 0.0,
            'min_duration': 0.0,
            'max_duration': 0.0,
            'total_duration': 0.0
        }
    
    durations = [end - start for start, end in scenes]
    
    return {
        'count': len(scenes),
        'avg_duration': np.mean(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        'total_duration': sum(durations),
        'std_duration': np.std(durations)
    }
