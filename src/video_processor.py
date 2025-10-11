"""
Video Processing Utilities
===========================
Handles video compression, resizing, and metadata extraction.
"""

import cv2
import subprocess
import os
from typing import Dict, Optional, Tuple
import tempfile


def get_video_info(video_path: str) -> Dict:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'duration': 0.0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    # Get file size
    if os.path.exists(video_path):
        info['file_size_mb'] = os.path.getsize(video_path) / (1024 * 1024)
    
    cap.release()
    
    return info


def compress_video(input_path: str, output_path: str, 
                   crf: int = 23, preset: str = 'medium') -> bool:
    """
    Compress video using H.264 codec with FFmpeg.
    
    Args:
        input_path: Input video file path
        output_path: Output video file path
        crf: Constant Rate Factor (18-28, lower = better quality)
        preset: Encoding speed preset (ultrafast, fast, medium, slow, veryslow)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', preset,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Compression failed: {e}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg.")
        return False


def resize_video(input_path: str, output_path: str, 
                target_resolution: str = "720p") -> bool:
    """
    Resize video to target resolution.
    
    Args:
        input_path: Input video file path
        output_path: Output video file path
        target_resolution: Target resolution (e.g., "720p", "512x512", "1080p")
    
    Returns:
        True if successful, False otherwise
    """
    # Parse resolution
    if target_resolution == "720p":
        scale = "scale=-2:720"
    elif target_resolution == "1080p":
        scale = "scale=-2:1080"
    elif target_resolution == "480p":
        scale = "scale=-2:480"
    elif "x" in target_resolution:
        # Custom resolution like "512x512"
        width, height = target_resolution.split("x")
        scale = f"scale={width}:{height}"
    else:
        print(f"Unknown resolution: {target_resolution}")
        return False
    
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', scale,
            '-c:a', 'copy',
            '-y',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Resize failed: {e}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg.")
        return False


def compress_and_resize(input_path: str, output_path: str,
                       resolution: str = "720p", crf: int = 23) -> bool:
    """
    Compress and resize video in one operation.
    
    Args:
        input_path: Input video file path
        output_path: Output video file path
        resolution: Target resolution
        crf: Compression quality (18-28)
    
    Returns:
        True if successful, False otherwise
    """
    # Parse resolution
    if resolution == "720p":
        scale = "scale=-2:720"
    elif resolution == "1080p":
        scale = "scale=-2:1080"
    elif resolution == "480p":
        scale = "scale=-2:480"
    elif "x" in resolution:
        width, height = resolution.split("x")
        scale = f"scale={width}:{height}"
    elif resolution.lower() == "original":
        scale = None
    else:
        scale = "scale=-2:720"  # Default to 720p
    
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
        ]
        
        if scale:
            cmd.extend(['-vf', scale])
        
        cmd.extend(['-y', output_path])
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Processing failed: {e}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg.")
        return False


def extract_audio(video_path: str, output_path: str) -> bool:
    """
    Extract audio from video file.
    
    Args:
        video_path: Input video file path
        output_path: Output audio file path (e.g., .mp3, .wav)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-q:a', '2',  # Quality (0-9, lower is better)
            '-y',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Audio extraction failed: {e}")
        return False


def create_thumbnail(video_path: str, output_path: str, 
                    timestamp: float = 0.0) -> bool:
    """
    Extract a single frame as thumbnail.
    
    Args:
        video_path: Input video file path
        output_path: Output image file path
        timestamp: Time in seconds to extract frame
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            '-y',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Thumbnail creation failed: {e}")
        return False


def estimate_compression_ratio(original_info: Dict, crf: int = 23, 
                               target_resolution: Optional[str] = None) -> float:
    """
    Estimate compression ratio based on settings.
    
    This is a rough estimation based on typical compression ratios.
    
    Args:
        original_info: Original video metadata from get_video_info()
        crf: Compression quality setting
        target_resolution: Target resolution (if resizing)
    
    Returns:
        Estimated compression ratio (0-1, e.g., 0.5 = 50% of original size)
    """
    # Base compression ratio for CRF
    # CRF 18: ~0.7-0.8, CRF 23: ~0.4-0.5, CRF 28: ~0.2-0.3
    crf_ratios = {
        18: 0.75,
        23: 0.45,
        28: 0.25
    }
    
    # Interpolate for other CRF values
    if crf in crf_ratios:
        compression_ratio = crf_ratios[crf]
    else:
        # Linear interpolation
        if crf < 18:
            compression_ratio = 0.75
        elif crf > 28:
            compression_ratio = 0.25
        else:
            # Between 18 and 23, or 23 and 28
            if crf < 23:
                factor = (crf - 18) / (23 - 18)
                compression_ratio = 0.75 - factor * (0.75 - 0.45)
            else:
                factor = (crf - 23) / (28 - 23)
                compression_ratio = 0.45 - factor * (0.45 - 0.25)
    
    # Adjust for resolution change
    if target_resolution:
        original_pixels = original_info['width'] * original_info['height']
        
        if target_resolution == "720p":
            target_pixels = 1280 * 720
        elif target_resolution == "1080p":
            target_pixels = 1920 * 1080
        elif target_resolution == "480p":
            target_pixels = 854 * 480
        elif "x" in target_resolution:
            w, h = map(int, target_resolution.split("x"))
            target_pixels = w * h
        else:
            target_pixels = original_pixels
        
        resolution_ratio = target_pixels / original_pixels
        compression_ratio *= resolution_ratio
    
    return compression_ratio


def compare_quality(original_path: str, compressed_path: str) -> Dict:
    """
    Compare quality metrics between original and compressed video.
    
    Args:
        original_path: Path to original video
        compressed_path: Path to compressed video
    
    Returns:
        Dictionary with comparison metrics
    """
    original_info = get_video_info(original_path)
    compressed_info = get_video_info(compressed_path)
    
    size_reduction = 1.0 - (compressed_info['file_size_mb'] / original_info['file_size_mb'])
    
    return {
        'original_size_mb': original_info['file_size_mb'],
        'compressed_size_mb': compressed_info['file_size_mb'],
        'size_reduction_percent': size_reduction * 100,
        'original_resolution': f"{original_info['width']}x{original_info['height']}",
        'compressed_resolution': f"{compressed_info['width']}x{compressed_info['height']}",
        'compression_ratio': compressed_info['file_size_mb'] / original_info['file_size_mb']
    }
