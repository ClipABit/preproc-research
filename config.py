"""
Configuration File for Video Preprocessing Demo
================================================
Centralized settings for easy customization
"""

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

CHUNKING_DEFAULTS = {
    'static_interval_duration': 10,  # seconds
    'scene_detection_threshold': 27,  # lower = more sensitive (10-50)
    'hybrid_min_duration': 5,  # seconds
    'hybrid_max_duration': 20,  # seconds
}

# =============================================================================
# FRAME SELECTION CONFIGURATION
# =============================================================================

FRAME_SELECTION_DEFAULTS = {
    'keyframe_count': 1,  # number of keyframes per chunk
    'dense_sampling_fps': 1.0,  # frames per second
    'adaptive_motion_threshold': 15.0,  # motion threshold for adaptive
    'adaptive_min_fps': 0.5,  # minimum sampling rate
    'adaptive_max_fps': 2.0,  # maximum sampling rate
}

# =============================================================================
# COMPRESSION CONFIGURATION
# =============================================================================

COMPRESSION_DEFAULTS = {
    'enabled': True,
    'target_resolution': '720p',  # '720p', '1080p', '480p', '512x512', 'original'
    'crf': 23,  # Constant Rate Factor (18=high quality, 28=low quality)
    'preset': 'medium',  # 'ultrafast', 'fast', 'medium', 'slow', 'veryslow'
}

# CRF Quality Presets
CRF_PRESETS = {
    'high': 18,
    'medium': 23,
    'low': 28,
}

# =============================================================================
# VIDEO PROCESSING CONFIGURATION
# =============================================================================

VIDEO_CONFIG = {
    'max_upload_size_mb': 500,  # maximum video file size
    'supported_formats': ['mp4', 'avi', 'mov', 'mkv', 'webm'],
    'temp_dir': '/tmp/video_preprocessing',  # temporary storage
}

# =============================================================================
# STREAMLIT UI CONFIGURATION
# =============================================================================

UI_CONFIG = {
    'page_title': 'Video Preprocessing Demo',
    'page_icon': '🎬',
    'layout': 'wide',
    'sidebar_width': 300,
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

VIZ_CONFIG = {
    'timeline_height': 400,  # pixels
    'chart_height': 400,
    'frames_per_row': 4,  # in chunk preview
    'color_scheme': 'Viridis',  # Plotly color scheme
}

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

PERFORMANCE_CONFIG = {
    'scene_detection_downscale': 1,  # downscale factor for scene detection
    'max_frames_in_memory': 100,  # maximum frames to keep in memory
    'parallel_processing': True,  # enable parallel chunk processing
    'cache_enabled': True,  # enable Streamlit caching
}

# =============================================================================
# PRESET CONFIGURATIONS FOR DIFFERENT USE CASES
# =============================================================================

PRESETS = {
    'development': {
        'chunking': {
            'method': 'hybrid',
            'min_duration': 5,
            'max_duration': 20,
            'threshold': 27,
        },
        'frame_selection': {
            'method': 'dense',
            'fps': 1.0,
        },
        'compression': {
            'resolution': '720p',
            'crf': 23,
        }
    },
    
    'production_optimized': {
        'chunking': {
            'method': 'hybrid',
            'min_duration': 5,
            'max_duration': 20,
            'threshold': 27,
        },
        'frame_selection': {
            'method': 'adaptive',
            'motion_threshold': 15.0,
            'min_fps': 0.5,
            'max_fps': 2.0,
        },
        'compression': {
            'resolution': '720p',
            'crf': 23,
        }
    },
    
    'high_quality': {
        'chunking': {
            'method': 'scene_detection',
            'threshold': 27,
        },
        'frame_selection': {
            'method': 'dense',
            'fps': 2.0,
        },
        'compression': {
            'resolution': '1080p',
            'crf': 18,
        }
    },
    
    'fast_preview': {
        'chunking': {
            'method': 'static',
            'duration': 10,
        },
        'frame_selection': {
            'method': 'keyframe',
            'count': 1,
        },
        'compression': {
            'resolution': '480p',
            'crf': 28,
        }
    },
    
    'storage_optimized': {
        'chunking': {
            'method': 'static',
            'duration': 15,
        },
        'frame_selection': {
            'method': 'adaptive',
            'motion_threshold': 20.0,
            'min_fps': 0.3,
            'max_fps': 1.0,
        },
        'compression': {
            'resolution': '512x512',
            'crf': 26,
        }
    }
}

# =============================================================================
# RECOMMENDED SETTINGS FROM PREPROCESSING REPORT
# =============================================================================

RECOMMENDED_MVP_SETTINGS = {
    'compression': {
        'codec': 'H.264',
        'crf': 23,
        'resolution': '720p',
        'rationale': 'Balance of quality and storage. H.264 mature ecosystem, CRF 23 is sweet spot.'
    },
    'chunking': {
        'method': 'hybrid',
        'min_duration': 5,
        'max_duration': 20,
        'scene_threshold': 27,
        'rationale': 'Semantic coherence without complexity of arbitrary lengths.'
    },
    'frame_selection': {
        'method': 'adaptive',  # Start with dense for MVP, optimize to adaptive later
        'initial_fps': 1.0,  # Dense sampling for validation
        'optimized_min_fps': 0.5,  # Adaptive settings for production
        'optimized_max_fps': 2.0,
        'motion_threshold': 15.0,
        'rationale': 'Dense initially for quality validation, adaptive for optimization.'
    }
}

# =============================================================================
# COST ESTIMATION PARAMETERS
# =============================================================================

COST_CONFIG = {
    'aws_s3_storage_per_gb': 0.023,  # $/GB/month
    'aws_s3_egress_per_gb': 0.09,  # $/GB (after first 100GB free)
    'aws_opensearch_free_tier_gb': 10,  # GB
    'embedding_size_kb': 10,  # KB per chunk (metadata + embedding)
}

# =============================================================================
# DATASET ASSUMPTIONS (from FLIGHT plan)
# =============================================================================

DATASET_ASSUMPTIONS = {
    'vlog_dataset': {
        'total_clips': 300,
        'avg_clip_length_sec': 30,
        'avg_clip_size_mb': 10,
        'clip_fps': 30,
        'clip_resolution': '1080p',
    },
    'team_storage': {
        'members': 10,
        'storage_per_member_gb': 20,  # per month
        'queries_per_day_per_member': 5,
        'clips_per_query': 12,  # average
    }
}

# =============================================================================
# TESTING CONFIGURATION
# =============================================================================

TEST_CONFIG = {
    'test_video_duration': 20,  # seconds
    'test_video_fps': 30,
    'test_video_resolution': (640, 480),
    'test_scenes': [
        {'type': 'static', 'duration_ratio': 0.25},
        {'type': 'slow_change', 'duration_ratio': 0.25},
        {'type': 'fast_action', 'duration_ratio': 0.25},
        {'type': 'static', 'duration_ratio': 0.25},
    ]
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'preprocessing.log',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset(preset_name: str) -> dict:
    """Get a preset configuration by name."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    return PRESETS[preset_name]


def estimate_storage(num_videos: int, avg_duration: float, 
                     preset_name: str = 'production_optimized') -> dict:
    """
    Estimate storage requirements for a given number of videos.
    
    Args:
        num_videos: Number of videos to process
        avg_duration: Average video duration in seconds
        preset_name: Configuration preset to use
    
    Returns:
        Dictionary with storage estimates
    """
    preset = get_preset(preset_name)
    
    # Estimate chunks per video
    if preset['chunking']['method'] == 'static':
        chunk_duration = preset['chunking'].get('duration', 10)
    else:
        chunk_duration = (preset['chunking']['min_duration'] + 
                         preset['chunking']['max_duration']) / 2
    
    chunks_per_video = avg_duration / chunk_duration
    
    # Estimate frames per chunk
    if preset['frame_selection']['method'] == 'keyframe':
        frames_per_chunk = preset['frame_selection'].get('count', 1)
    elif preset['frame_selection']['method'] == 'dense':
        fps = preset['frame_selection']['fps']
        frames_per_chunk = chunk_duration * fps
    else:  # adaptive
        avg_fps = (preset['frame_selection']['min_fps'] + 
                  preset['frame_selection']['max_fps']) / 2
        frames_per_chunk = chunk_duration * avg_fps
    
    total_chunks = num_videos * chunks_per_video
    total_frames = total_chunks * frames_per_chunk
    
    # Estimate storage (rough)
    # Assume ~500KB per frame compressed at 720p
    frame_size_mb = 0.5
    total_storage_mb = total_frames * frame_size_mb
    
    return {
        'num_videos': num_videos,
        'avg_duration_sec': avg_duration,
        'chunks_per_video': chunks_per_video,
        'total_chunks': total_chunks,
        'frames_per_chunk': frames_per_chunk,
        'total_frames': total_frames,
        'estimated_storage_mb': total_storage_mb,
        'estimated_storage_gb': total_storage_mb / 1024,
        'preset_used': preset_name,
    }
