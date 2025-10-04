"""
Sample Video Generator
======================
Creates sample videos for testing the preprocessing demo.
Useful when you don't have test footage readily available.
"""

import cv2
import numpy as np
import tempfile
import os


def generate_test_video(output_path: str, duration: int = 30, fps: int = 30):
    """
    Generate a test video with various scenes for preprocessing demo.
    
    The video contains:
    - Static scene (solid color)
    - Slow motion scene (gradually changing)
    - Fast action scene (rapid changes)
    - Another static scene
    
    Args:
        output_path: Path where to save the video
        duration: Duration in seconds
        fps: Frames per second
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    frames_per_scene = total_frames // 4
    
    print(f"Generating {duration}s test video at {fps} fps...")
    
    for frame_idx in range(total_frames):
        # Determine which scene we're in
        scene = frame_idx // frames_per_scene
        frame_in_scene = frame_idx % frames_per_scene
        
        if scene == 0:
            # Static scene - solid blue
            frame = np.ones((height, width, 3), dtype=np.uint8) * [200, 100, 50]
            cv2.putText(frame, "Scene 1: Static", (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        elif scene == 1:
            # Slow motion - gradually changing color
            progress = frame_in_scene / frames_per_scene
            color = int(255 * progress)
            frame = np.ones((height, width, 3), dtype=np.uint8) * [50, 100, color]
            cv2.putText(frame, "Scene 2: Slow Change", (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        elif scene == 2:
            # Fast action - rapidly changing patterns
            pattern = frame_in_scene % 10
            if pattern < 5:
                frame = np.ones((height, width, 3), dtype=np.uint8) * [255, 100, 100]
            else:
                frame = np.ones((height, width, 3), dtype=np.uint8) * [100, 255, 100]
            
            # Add moving circle
            x = int((frame_in_scene / frames_per_scene) * width)
            y = height // 2
            cv2.circle(frame, (x, y), 30, (255, 255, 255), -1)
            
            cv2.putText(frame, "Scene 3: Fast Action", (50, height//4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        else:
            # Another static scene - solid green
            frame = np.ones((height, width, 3), dtype=np.uint8) * [100, 200, 100]
            cv2.putText(frame, "Scene 4: Static", (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames")
    
    out.release()
    print(f"✅ Test video saved to: {output_path}")
    print(f"Video info: {duration}s, {fps} fps, {width}x{height}")


def generate_action_video(output_path: str, duration: int = 20, fps: int = 30):
    """
    Generate a test video with lots of action/motion.
    
    Args:
        output_path: Path where to save the video
        duration: Duration in seconds
        fps: Frames per second
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Generating {duration}s action video at {fps} fps...")
    
    # Multiple moving objects
    objects = [
        {'x': 0, 'y': 100, 'vx': 5, 'vy': 2, 'color': (255, 0, 0)},
        {'x': width, 'y': 200, 'vx': -4, 'vy': 3, 'color': (0, 255, 0)},
        {'x': width//2, 'y': height, 'vx': 3, 'vy': -4, 'color': (0, 0, 255)},
    ]
    
    for frame_idx in range(total_frames):
        # Black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update and draw each object
        for obj in objects:
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # Bounce off walls
            if obj['x'] < 0 or obj['x'] > width:
                obj['vx'] *= -1
            if obj['y'] < 0 or obj['y'] > height:
                obj['vy'] *= -1
            
            # Keep in bounds
            obj['x'] = max(0, min(width, obj['x']))
            obj['y'] = max(0, min(height, obj['y']))
            
            # Draw circle
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), 20, obj['color'], -1)
        
        cv2.putText(frame, "High Motion Scene", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames")
    
    out.release()
    print(f"✅ Action video saved to: {output_path}")


def generate_static_video(output_path: str, duration: int = 15, fps: int = 30):
    """
    Generate a mostly static video with minimal motion.
    
    Args:
        output_path: Path where to save the video
        duration: Duration in seconds
        fps: Frames per second
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Generating {duration}s static video at {fps} fps...")
    
    # Create base frame
    base_frame = np.ones((height, width, 3), dtype=np.uint8) * [150, 150, 150]
    cv2.putText(base_frame, "Static Scene Demo", (width//4, height//2), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 3)
    
    for frame_idx in range(total_frames):
        frame = base_frame.copy()
        
        # Add very subtle noise to simulate camera sensor noise
        noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add frame counter
        cv2.putText(frame, f"{frame_idx}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames")
    
    out.release()
    print(f"✅ Static video saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test videos for preprocessing demo")
    parser.add_argument("--type", choices=["test", "action", "static", "all"], 
                       default="test", help="Type of video to generate")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Duration in seconds")
    parser.add_argument("--output", type=str, default="test_video.mp4",
                       help="Output file path")
    
    args = parser.parse_args()
    
    if args.type == "all":
        print("Generating all test video types...\n")
        generate_test_video("test_mixed.mp4", args.duration)
        print()
        generate_action_video("test_action.mp4", args.duration)
        print()
        generate_static_video("test_static.mp4", args.duration)
        print("\n✅ All test videos generated!")
    elif args.type == "test":
        generate_test_video(args.output, args.duration)
    elif args.type == "action":
        generate_action_video(args.output, args.duration)
    elif args.type == "static":
        generate_static_video(args.output, args.duration)
