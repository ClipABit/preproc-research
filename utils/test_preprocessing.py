"""
Test Suite for Video Preprocessing Demo
========================================
Run this to validate all preprocessing algorithms work correctly.
"""

import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

from frame_selector import (
    extract_keyframes,
    extract_dense_frames,
    extract_adaptive_frames,
    calculate_frame_difference
)
from scene_detector import detect_scenes, apply_scene_constraints
from video_processor import get_video_info, compress_video
from generate_test_videos import generate_test_video


class TestPreprocessing:
    """Test suite for preprocessing algorithms."""
    
    def __init__(self):
        self.test_video_path = None
        self.setup()
    
    def setup(self):
        """Create a test video."""
        print("Setting up test environment...")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        self.test_video_path = temp_file.name
        temp_file.close()
        
        generate_test_video(self.test_video_path, duration=10, fps=30)
        print(f"✅ Test video created: {self.test_video_path}\n")
    
    def teardown(self):
        """Clean up test files."""
        if self.test_video_path and os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
            print(f"\n🧹 Cleaned up test video")
    
    def test_video_info(self):
        """Test video metadata extraction."""
        print("TEST: Video Info Extraction")
        print("-" * 50)
        
        info = get_video_info(self.test_video_path)
        
        assert info['duration'] > 0, "Duration should be positive"
        assert info['fps'] > 0, "FPS should be positive"
        assert info['width'] > 0, "Width should be positive"
        assert info['height'] > 0, "Height should be positive"
        
        print(f"Duration: {info['duration']:.2f}s")
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"FPS: {info['fps']:.2f}")
        print(f"Frames: {info['frame_count']}")
        print("✅ PASSED\n")
        
        return info
    
    def test_keyframe_extraction(self, video_info):
        """Test keyframe extraction."""
        print("TEST: Keyframe Extraction")
        print("-" * 50)
        
        cap = cv2.VideoCapture(self.test_video_path)
        fps = video_info['fps']
        
        # Test with different keyframe counts
        for num_keyframes in [1, 2, 3]:
            frames, timestamps = extract_keyframes(
                cap, 0.0, 5.0, fps, num_keyframes=num_keyframes
            )
            
            assert len(frames) <= num_keyframes, \
                f"Should extract at most {num_keyframes} keyframes"
            assert len(frames) == len(timestamps), \
                "Frames and timestamps should match"
            
            print(f"Keyframes={num_keyframes}: Extracted {len(frames)} frames")
        
        cap.release()
        print("✅ PASSED\n")
    
    def test_dense_sampling(self, video_info):
        """Test dense frame sampling."""
        print("TEST: Dense Sampling")
        print("-" * 50)
        
        cap = cv2.VideoCapture(self.test_video_path)
        fps = video_info['fps']
        
        # Test with different sampling rates
        for sample_fps in [0.5, 1.0, 2.0]:
            frames, timestamps = extract_dense_frames(
                cap, 0.0, 5.0, fps, sample_fps=sample_fps
            )
            
            expected_frames = int(5.0 * sample_fps)
            tolerance = 2  # Allow some tolerance
            
            assert abs(len(frames) - expected_frames) <= tolerance, \
                f"Should extract ~{expected_frames} frames at {sample_fps} fps"
            
            print(f"Sample FPS={sample_fps}: Extracted {len(frames)} frames "
                  f"(expected ~{expected_frames})")
        
        cap.release()
        print("✅ PASSED\n")
    
    def test_adaptive_sampling(self, video_info):
        """Test adaptive frame sampling."""
        print("TEST: Adaptive Sampling")
        print("-" * 50)
        
        cap = cv2.VideoCapture(self.test_video_path)
        fps = video_info['fps']
        
        # Test on different parts of video
        # First part should be more static, later parts more dynamic
        test_segments = [
            (0.0, 2.5, "Static scene"),
            (5.0, 7.5, "Dynamic scene"),
        ]
        
        for start, end, description in test_segments:
            frames, timestamps = extract_adaptive_frames(
                cap, start, end, fps,
                motion_threshold=15.0,
                min_fps=0.5,
                max_fps=2.0
            )
            
            assert len(frames) > 0, "Should extract at least one frame"
            print(f"{description} ({start}-{end}s): {len(frames)} frames")
        
        cap.release()
        print("✅ PASSED\n")
    
    def test_scene_detection(self, video_info):
        """Test scene detection."""
        print("TEST: Scene Detection")
        print("-" * 50)
        
        scenes = detect_scenes(self.test_video_path, threshold=27)
        
        assert len(scenes) > 0, "Should detect at least one scene"
        
        # Verify scene times are valid
        for i, (start, end) in enumerate(scenes):
            assert start < end, f"Scene {i}: start should be before end"
            assert start >= 0, f"Scene {i}: start should be non-negative"
            assert end <= video_info['duration'] + 1, \
                f"Scene {i}: end should not exceed video duration"
            
            print(f"Scene {i+1}: {start:.2f}s - {end:.2f}s "
                  f"(duration: {end-start:.2f}s)")
        
        print(f"\nTotal scenes detected: {len(scenes)}")
        print("✅ PASSED\n")
        
        return scenes
    
    def test_scene_constraints(self, scenes):
        """Test scene constraint application."""
        print("TEST: Scene Constraints (Hybrid Chunking)")
        print("-" * 50)
        
        min_duration = 3.0
        max_duration = 8.0
        
        constrained = apply_scene_constraints(
            scenes, 
            min_duration=min_duration,
            max_duration=max_duration
        )
        
        # Verify all constraints are met
        for i, (start, end) in enumerate(constrained):
            duration = end - start
            
            assert duration >= min_duration - 0.1, \
                f"Scene {i}: duration {duration:.2f}s below minimum {min_duration}s"
            assert duration <= max_duration + 0.1, \
                f"Scene {i}: duration {duration:.2f}s exceeds maximum {max_duration}s"
            
            print(f"Chunk {i+1}: {start:.2f}s - {end:.2f}s "
                  f"(duration: {duration:.2f}s) ✓")
        
        print(f"\nOriginal scenes: {len(scenes)}")
        print(f"Constrained chunks: {len(constrained)}")
        print("✅ PASSED\n")
    
    def test_frame_difference(self):
        """Test frame difference calculation."""
        print("TEST: Frame Difference Calculation")
        print("-" * 50)
        
        # Create two different frames
        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 150
        
        # Identical frames should have 0 difference
        diff_same = calculate_frame_difference(frame1, frame1)
        assert diff_same < 0.01, "Identical frames should have ~0 difference"
        print(f"Identical frames difference: {diff_same:.4f} ✓")
        
        # Different frames should have non-zero difference
        diff_different = calculate_frame_difference(frame1, frame2)
        assert diff_different > 0, "Different frames should have >0 difference"
        print(f"Different frames difference: {diff_different:.4f} ✓")
        
        print("✅ PASSED\n")
    
    def test_compression(self, video_info):
        """Test video compression (if FFmpeg available)."""
        print("TEST: Video Compression")
        print("-" * 50)
        
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            ffmpeg_available = False
        
        if not ffmpeg_available:
            print("⚠️  FFmpeg not available, skipping compression test")
            print("SKIPPED\n")
            return
        
        output_path = self.test_video_path.replace('.mp4', '_compressed.mp4')
        
        success = compress_video(
            self.test_video_path,
            output_path,
            crf=23,
            preset='fast'
        )
        
        assert success, "Compression should succeed"
        assert os.path.exists(output_path), "Compressed file should exist"
        
        # Check compressed file is smaller
        original_size = os.path.getsize(self.test_video_path)
        compressed_size = os.path.getsize(output_path)
        
        print(f"Original size: {original_size / 1024:.2f} KB")
        print(f"Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"Compression ratio: {compressed_size / original_size:.2%}")
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
        
        print("✅ PASSED\n")
    
    def run_all_tests(self):
        """Run all tests."""
        print("=" * 60)
        print("RUNNING VIDEO PREPROCESSING TEST SUITE")
        print("=" * 60)
        print()
        
        try:
            # Test 1: Video Info
            video_info = self.test_video_info()
            
            # Test 2: Frame Difference
            self.test_frame_difference()
            
            # Test 3: Keyframe Extraction
            self.test_keyframe_extraction(video_info)
            
            # Test 4: Dense Sampling
            self.test_dense_sampling(video_info)
            
            # Test 5: Adaptive Sampling
            self.test_adaptive_sampling(video_info)
            
            # Test 6: Scene Detection
            scenes = self.test_scene_detection(video_info)
            
            # Test 7: Scene Constraints
            self.test_scene_constraints(scenes)
            
            # Test 8: Compression
            self.test_compression(video_info)
            
            print("=" * 60)
            print("✅ ALL TESTS PASSED!")
            print("=" * 60)
            
        except AssertionError as e:
            print("\n" + "=" * 60)
            print(f"❌ TEST FAILED: {e}")
            print("=" * 60)
            raise
        
        finally:
            self.teardown()


if __name__ == "__main__":
    tester = TestPreprocessing()
    tester.run_all_tests()
