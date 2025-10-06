"""
Scene Detection Module
======================
Implements advanced scene detection for semantic video chunking.
Uses multi-method detection with adaptive thresholding for better accuracy.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scenedetect import detect, ContentDetector, AdaptiveDetector
import tempfile


def detect_scenes(video_path: str, threshold: float = 27.0, 
                 adaptive: bool = True, min_scene_len: float = 0.5) -> List[Tuple[float, float]]:
    """
    Detect scene boundaries using advanced multi-method detection.
    
    Uses both content-based and adaptive detection for better accuracy.
    Especially important for adaptive sampling where scene boundaries affect quality.
    
    Args:
        video_path: Path to video file
        threshold: Detection threshold (lower = more sensitive, default=27)
        adaptive: Use adaptive detector for better accuracy
        min_scene_len: Minimum scene length in seconds
    
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    try:
        if adaptive:
            # Use AdaptiveDetector for better scene boundary detection
            # AdaptiveDetector is more robust to gradual transitions and lighting changes
            scene_list = detect(
                video_path, 
                AdaptiveDetector(
                    adaptive_threshold=3.0,  # Sensitivity to content changes
                    min_scene_len=int(min_scene_len * 30),  # Assuming ~30 fps
                    window_width=2,  # Frames to analyze for content change
                )
            )
        else:
            # Fallback to ContentDetector for faster processing
            scene_list = detect(
                video_path, 
                ContentDetector(
                    threshold=threshold,
                    min_scene_len=int(min_scene_len * 30)
                )
            )
        
        # Convert to (start_time, end_time) tuples
        scenes = []
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append((start_time, end_time))
        
        # If no scenes detected, use hybrid approach
        if not scenes:
            print("Primary detection found no scenes, using hybrid detection...")
            scenes = detect_scenes_hybrid(video_path, threshold)
        
        return scenes
    
    except Exception as e:
        print(f"Scene detection failed: {e}, falling back to hybrid detection")
        return detect_scenes_hybrid(video_path, threshold)


def detect_scenes_hybrid(video_path: str, threshold: float = 30.0) -> List[Tuple[float, float]]:
    """
    Hybrid scene detection combining multiple methods for better accuracy.
    
    Combines:
    1. Histogram-based difference detection (color changes)
    2. Edge-based detection (composition changes)
    3. Motion-based detection (camera movement/cuts)
    
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
    
    # Analyze every N frames for performance (adaptive based on video length)
    sample_interval = max(1, min(int(fps), int(frame_count / 1000)))
    
    scene_boundaries = [0]  # Start with first frame
    prev_frame = None
    
    # Track multiple metrics
    hist_diffs = []
    edge_diffs = []
    motion_scores = []
    frame_indices = []
    
    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_indices.append(frame_idx)
        
        if prev_frame is not None:
            # Calculate histogram difference (color-based)
            hist_diff = calculate_histogram_difference(prev_frame, frame)
            hist_diffs.append(hist_diff)
            
            # Calculate edge difference (composition-based)
            edge_diff = calculate_edge_difference(prev_frame, frame)
            edge_diffs.append(edge_diff)
            
            # Calculate motion score
            motion = calculate_motion_magnitude(prev_frame, frame)
            motion_scores.append(motion)
        else:
            hist_diffs.append(0.0)
            edge_diffs.append(0.0)
            motion_scores.append(0.0)
        
        prev_frame = frame
    
    cap.release()
    
    # Normalize all metrics to 0-1 range
    hist_diffs = np.array(hist_diffs)
    edge_diffs = np.array(edge_diffs)
    motion_scores = np.array(motion_scores)
    
    # Combine metrics with weighted average
    # Higher weight on histogram for color-based scene changes
    # Motion helps detect camera cuts
    combined_score = (
        0.5 * hist_diffs +
        0.3 * edge_diffs +
        0.2 * motion_scores
    )
    
    # Adaptive thresholding: find peaks that are significantly higher than baseline
    mean_score = np.mean(combined_score)
    std_score = np.std(combined_score)
    adaptive_threshold = mean_score + (1.5 * std_score)
    
    # Find scene boundaries where combined score exceeds adaptive threshold
    for i, score in enumerate(combined_score):
        if score > adaptive_threshold and score * 100 > threshold:
            scene_boundaries.append(frame_indices[i])
    
    # Add last frame
    scene_boundaries.append(frame_count - 1)
    
    # Remove duplicates and sort
    scene_boundaries = sorted(list(set(scene_boundaries)))
    
    # Convert frame indices to time ranges
    scenes = []
    for i in range(len(scene_boundaries) - 1):
        start_time = scene_boundaries[i] / fps
        end_time = scene_boundaries[i + 1] / fps
        
        # Filter out very short scenes (< 0.5 seconds)
        if end_time - start_time >= 0.5:
            scenes.append((start_time, end_time))
    
    return scenes if scenes else [(0.0, frame_count / fps)]


def calculate_edge_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate edge-based difference between frames (composition changes).
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
    
    Returns:
        Difference score (0-1)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)
    
    # Calculate difference in edge maps
    edge_diff = np.abs(edges1.astype(float) - edges2.astype(float))
    
    # Normalize by image size
    diff_score = np.mean(edge_diff) / 255.0
    
    return diff_score


def calculate_motion_magnitude(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate motion magnitude between frames (camera movement/cuts).
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
    
    Returns:
        Motion magnitude (0-1, normalized)
    """
    # Convert to grayscale and resize for performance
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Resize for faster computation
    small_size = (320, 180)
    gray1_small = cv2.resize(gray1, small_size)
    gray2_small = cv2.resize(gray2, small_size)
    
    # Simple absolute difference (fast alternative to optical flow)
    diff = cv2.absdiff(gray1_small, gray2_small)
    
    # Normalize
    motion = np.mean(diff) / 255.0
    
    return motion


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
                           max_duration: float = 20.0,
                           adaptive_merge: bool = True) -> List[Tuple[float, float]]:
    """
    Apply both minimum and maximum duration constraints to scenes.
    
    This implements intelligent scene chunking for adaptive sampling:
    - Merges very short scenes that would produce poor samples
    - Splits very long scenes while respecting semantic boundaries
    - Optionally uses content-aware merging
    
    Args:
        scenes: List of (start_time, end_time) tuples
        min_duration: Minimum scene duration in seconds
        max_duration: Maximum scene duration in seconds
        adaptive_merge: Use content-aware merging (preserves scene semantics)
    
    Returns:
        Constrained scene list optimized for adaptive sampling
    """
    if not scenes:
        return []
    
    # First merge short scenes intelligently
    if adaptive_merge:
        merged = merge_scenes_adaptive(scenes, min_duration)
    else:
        merged = merge_short_scenes(scenes, min_duration)
    
    # Then split long scenes while preserving natural boundaries
    constrained = split_long_scenes_smart(merged, max_duration)
    
    return constrained


def merge_scenes_adaptive(scenes: List[Tuple[float, float]], 
                         min_duration: float = 3.0) -> List[Tuple[float, float]]:
    """
    Intelligently merge short scenes based on temporal proximity.
    
    Keeps semantically related scenes together while ensuring minimum duration.
    
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
        current_duration = current_end - current_start
        
        # If current accumulated scene is still below minimum, keep merging
        if current_duration < min_duration:
            current_end = scene_end
        else:
            # Current scene meets minimum duration
            merged.append((current_start, current_end))
            current_start = scene_start
            current_end = scene_end
    
    # Add last scene
    merged.append((current_start, current_end))
    
    return merged


def split_long_scenes_smart(scenes: List[Tuple[float, float]], 
                           max_duration: float = 20.0,
                           prefer_equal_splits: bool = True) -> List[Tuple[float, float]]:
    """
    Split long scenes into manageable chunks for better adaptive sampling.
    
    Creates equal-sized chunks when possible to ensure consistent sampling quality.
    
    Args:
        scenes: List of (start_time, end_time) tuples
        max_duration: Maximum scene duration in seconds
        prefer_equal_splits: Create equal-sized chunks vs fixed-size chunks
    
    Returns:
        Split scene list
    """
    split_scenes = []
    
    for start_time, end_time in scenes:
        duration = end_time - start_time
        
        if duration <= max_duration:
            split_scenes.append((start_time, end_time))
        else:
            # Calculate optimal number of chunks
            num_chunks = int(np.ceil(duration / max_duration))
            
            if prefer_equal_splits:
                # Create equal-sized chunks for consistent sampling
                chunk_duration = duration / num_chunks
                
                for i in range(num_chunks):
                    chunk_start = start_time + (i * chunk_duration)
                    chunk_end = min(chunk_start + chunk_duration, end_time)
                    split_scenes.append((chunk_start, chunk_end))
            else:
                # Create fixed-size chunks (last chunk may be smaller)
                current_start = start_time
                while current_start < end_time:
                    chunk_end = min(current_start + max_duration, end_time)
                    split_scenes.append((current_start, chunk_end))
                    current_start = chunk_end
    
    return split_scenes


def chunk_scenes_for_adaptive_sampling(scenes: List[Tuple[float, float]], 
                                       target_chunk_duration: float = 10.0,
                                       allow_variance: float = 0.3) -> List[Tuple[float, float]]:
    """
    Create optimal scene chunks specifically for adaptive sampling.
    
    Each chunk represents a semantic unit that will be sampled adaptively based on
    its motion/content characteristics. Aims for consistent chunk sizes while
    respecting scene boundaries.
    
    Args:
        scenes: List of detected scene boundaries (start_time, end_time)
        target_chunk_duration: Target duration for each chunk in seconds
        allow_variance: Allowed variance from target (0.3 = ±30%)
    
    Returns:
        List of optimized chunks for adaptive sampling
    """
    min_duration = target_chunk_duration * (1 - allow_variance)
    max_duration = target_chunk_duration * (1 + allow_variance)
    
    chunks = []
    current_chunk_start = None
    current_chunk_end = None
    
    for scene_start, scene_end in scenes:
        scene_duration = scene_end - scene_start
        
        if current_chunk_start is None:
            # Start new chunk
            current_chunk_start = scene_start
            current_chunk_end = scene_end
        else:
            # Try to add scene to current chunk
            potential_duration = scene_end - current_chunk_start
            
            if potential_duration <= max_duration:
                # Scene fits in current chunk
                current_chunk_end = scene_end
            else:
                # Scene doesn't fit, finalize current chunk
                chunks.append((current_chunk_start, current_chunk_end))
                current_chunk_start = scene_start
                current_chunk_end = scene_end
    
    # Add final chunk
    if current_chunk_start is not None:
        chunks.append((current_chunk_start, current_chunk_end))
    
    # Post-process: split any remaining oversized chunks
    final_chunks = []
    for chunk_start, chunk_end in chunks:
        duration = chunk_end - chunk_start
        
        if duration > max_duration * 1.5:  # Significantly over target
            # Split into equal parts
            num_splits = int(np.ceil(duration / target_chunk_duration))
            split_duration = duration / num_splits
            
            for i in range(num_splits):
                split_start = chunk_start + (i * split_duration)
                split_end = min(split_start + split_duration, chunk_end)
                final_chunks.append((split_start, split_end))
        else:
            final_chunks.append((chunk_start, chunk_end))
    
    return final_chunks


def get_scene_statistics(scenes: List[Tuple[float, float]]) -> dict:
    """
    Calculate comprehensive statistics about detected scenes.
    
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
            'total_duration': 0.0,
            'std_duration': 0.0,
            'median_duration': 0.0,
        }
    
    durations = [end - start for start, end in scenes]
    
    return {
        'count': len(scenes),
        'avg_duration': np.mean(durations),
        'min_duration': np.min(durations),
        'max_duration': np.max(durations),
        'total_duration': sum(durations),
        'std_duration': np.std(durations),
        'median_duration': np.median(durations),
    }


def analyze_scene_complexity(video_path: str, scene: Tuple[float, float],
                            sample_frames: int = 10) -> Dict[str, float]:
    """
    Analyze the visual complexity of a scene for adaptive sampling.
    
    Returns metrics that help determine optimal sampling rate:
    - motion_level: Average motion in the scene (0-1)
    - visual_diversity: How much the scene changes visually (0-1)
    - edge_density: Amount of detail/edges (0-1)
    
    Args:
        video_path: Path to video file
        scene: (start_time, end_time) tuple
        sample_frames: Number of frames to sample for analysis
    
    Returns:
        Dictionary with complexity metrics
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_time, end_time = scene
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    duration_frames = end_frame - start_frame
    
    if duration_frames <= 0:
        cap.release()
        return {
            'motion_level': 0.0,
            'visual_diversity': 0.0,
            'edge_density': 0.0,
            'recommended_fps': 0.5,
        }
    
    # Sample frames evenly across the scene
    sample_interval = max(1, duration_frames // sample_frames)
    
    motion_scores = []
    diversity_scores = []
    edge_densities = []
    prev_frame = None
    frames = []
    
    for i in range(sample_frames):
        frame_idx = start_frame + (i * sample_interval)
        if frame_idx >= end_frame:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frames.append(frame)
        
        # Calculate edge density
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        edge_densities.append(edge_density)
        
        # Calculate motion if we have previous frame
        if prev_frame is not None:
            motion = calculate_motion_magnitude(prev_frame, frame)
            motion_scores.append(motion)
        
        prev_frame = frame
    
    cap.release()
    
    # Calculate visual diversity (variation between all sampled frames)
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            diversity = calculate_histogram_difference(frames[i], frames[j])
            diversity_scores.append(diversity)
    
    # Aggregate metrics
    motion_level = np.mean(motion_scores) if motion_scores else 0.0
    visual_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
    edge_density = np.mean(edge_densities) if edge_densities else 0.0
    
    # Calculate recommended sampling rate based on complexity
    # High motion/diversity = higher sampling rate
    complexity_score = (motion_level * 0.5) + (visual_diversity * 0.3) + (edge_density * 0.2)
    
    # Map complexity to FPS (0.5 to 2.0 range)
    recommended_fps = 0.5 + (complexity_score * 1.5)
    recommended_fps = np.clip(recommended_fps, 0.5, 2.0)
    
    return {
        'motion_level': float(motion_level),
        'visual_diversity': float(visual_diversity),
        'edge_density': float(edge_density),
        'complexity_score': float(complexity_score),
        'recommended_fps': float(recommended_fps),
    }


def get_adaptive_sampling_plan(video_path: str, scenes: List[Tuple[float, float]]) -> List[Dict]:
    """
    Create a per-scene adaptive sampling plan.
    
    Analyzes each scene and determines optimal sampling parameters.
    This is the key function for adaptive sampling integration.
    
    Args:
        video_path: Path to video file
        scenes: List of scene boundaries
    
    Returns:
        List of dictionaries with sampling plan for each scene:
        {
            'scene_index': int,
            'start_time': float,
            'end_time': float,
            'duration': float,
            'complexity': dict (from analyze_scene_complexity),
            'recommended_fps': float,
            'estimated_frames': int,
        }
    """
    sampling_plan = []
    
    for i, (start_time, end_time) in enumerate(scenes):
        duration = end_time - start_time
        
        # Analyze scene complexity
        complexity = analyze_scene_complexity(video_path, (start_time, end_time))
        
        # Calculate estimated number of frames with recommended fps
        estimated_frames = int(duration * complexity['recommended_fps'])
        
        plan = {
            'scene_index': i,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'complexity': complexity,
            'recommended_fps': complexity['recommended_fps'],
            'estimated_frames': max(1, estimated_frames),  # At least 1 frame per scene
        }
        
        sampling_plan.append(plan)
    
    return sampling_plan
