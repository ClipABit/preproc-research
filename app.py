"""
Video Preprocessing Demo for Semantic Search
============================================
Interactive Streamlit app demonstrating different preprocessing strategies
for video semantic search, with emphasis on frame selection techniques.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import List, Tuple, Dict
import time
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.frame_selector import (
    extract_keyframes,
    extract_dense_frames,
    extract_adaptive_frames,
    calculate_frame_difference
)
from src.scene_detector import detect_scenes, visualize_scene_timeline
from src.video_processor import compress_video, get_video_info, resize_video


@dataclass
class ChunkInfo:
    """Information about a video chunk"""
    start_time: float
    end_time: float
    duration: float
    frames: List[np.ndarray]
    frame_timestamps: List[float]
    chunk_id: int


def main():
    st.set_page_config(
        page_title="Video Preprocessing Demo",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Preprocessing for Semantic Search")
    st.markdown("""
    This demo showcases different preprocessing strategies for video semantic search.
    **Focus: Frame Selection** - The most critical component for search quality.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Preprocessing Configuration")
    
    # Preset selector
    preset = st.sidebar.selectbox(
        "Quick Preset",
        ["Custom", "Recommended (Best)", "High Quality", "Fast Preview", "Storage Optimized"],
        index=1,  # Default to "Recommended (Best)"
        help="Select a preset configuration or customize your own"
    )
    
    # Video upload
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to test preprocessing strategies"
    )
    
    if uploaded_file is None:
        st.info("👆 Upload a video file to get started")
        display_methodology()
        return
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Get video info
    video_info = get_video_info(video_path)
    
    # Display video info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{video_info['duration']:.1f}s")
    with col2:
        st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
    with col3:
        st.metric("FPS", f"{video_info['fps']:.1f}")
    with col4:
        st.metric("Total Frames", video_info['frame_count'])
    
    # Preprocessing settings
    st.sidebar.subheader("1. Chunking Strategy")
    
    # Set defaults based on preset
    if preset == "Recommended (Best)":
        default_chunking = 2  # Hybrid
        default_frame_method = 2  # Adaptive
        default_threshold = 20
    elif preset == "High Quality":
        default_chunking = 1  # Scene Detection
        default_frame_method = 1  # Dense
        default_threshold = 27
    elif preset == "Fast Preview":
        default_chunking = 0  # Static
        default_frame_method = 0  # Keyframe
        default_threshold = 27
    elif preset == "Storage Optimized":
        default_chunking = 2  # Hybrid
        default_frame_method = 2  # Adaptive
        default_threshold = 27
    else:
        default_chunking = 2
        default_frame_method = 2
        default_threshold = 20
    
    chunking_method = st.sidebar.radio(
        "Method",
        ["Static Interval", "Scene Detection", "Hybrid (Scene + Constraints)"],
        index=default_chunking,
        help="How to divide video into searchable segments"
    )
    
    if chunking_method == "Static Interval":
        chunk_duration = st.sidebar.slider(
            "Chunk Duration (seconds)",
            min_value=3,
            max_value=30,
            value=10,
            step=1
        )
    elif chunking_method == "Scene Detection":
        scene_threshold = st.sidebar.slider(
            "Scene Detection Sensitivity",
            min_value=10,
            max_value=50,
            value=default_threshold,
            step=1,
            help="Lower = more sensitive to changes (20 recommended for fast-moving clips)"
        )
    else:  # Hybrid
        min_chunk_duration = st.sidebar.slider(
            "Min Chunk Duration (seconds)",
            min_value=3,
            max_value=10,
            value=5,
            step=1
        )
        max_chunk_duration = st.sidebar.slider(
            "Max Chunk Duration (seconds)",
            min_value=15,
            max_value=60,
            value=20,
            step=5
        )
        scene_threshold = st.sidebar.slider(
            "Scene Detection Sensitivity",
            min_value=10,
            max_value=50,
            value=default_threshold,
            step=1
        )
    
    st.sidebar.subheader("2. Frame Selection Strategy")
    frame_method = st.sidebar.radio(
        "Method",
        ["Keyframe Only", "Dense Sampling", "Adaptive Sampling"],
        index=default_frame_method,
        help="How to select representative frames from each chunk"
    )
    
    if frame_method == "Keyframe Only":
        num_keyframes = st.sidebar.slider(
            "Keyframes per Chunk",
            min_value=1,
            max_value=5,
            value=1,
            step=1
        )
    elif frame_method == "Dense Sampling":
        sample_fps = st.sidebar.slider(
            "Sampling Rate (fps)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5
        )
    else:  # Adaptive
        static_threshold = st.sidebar.slider(
            "Motion Threshold (static vs dynamic)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
            help="Higher = more tolerant of motion before increasing sampling"
        )
        min_fps = st.sidebar.slider("Min FPS (static scenes)", 0.5, 2.0, 0.5, 0.5)
        max_fps = st.sidebar.slider("Max FPS (dynamic scenes)", 1.0, 5.0, 2.0, 0.5)
    
    st.sidebar.subheader("3. Compression Settings")
    enable_compression = st.sidebar.checkbox("Enable Compression", value=True)
    
    if enable_compression:
        target_resolution = st.sidebar.selectbox(
            "Target Resolution",
            ["512x512", "720p", "Original"],
            index=1
        )
        quality_preset = st.sidebar.select_slider(
            "Quality",
            options=["Low (CRF 28)", "Medium (CRF 23)", "High (CRF 18)"],
            value="Medium (CRF 23)"
        )
    
    # Process button
    if st.sidebar.button("🚀 Process Video", type="primary"):
        process_video(
            video_path,
            video_info,
            chunking_method,
            frame_method,
            locals()  # Pass all local variables as settings
        )
    
    # Display original video
    st.subheader("📹 Original Video")
    st.video(video_path)


def process_video(video_path: str, video_info: Dict, chunking_method: str, 
                  frame_method: str, settings: Dict):
    """Process video with selected settings and display results"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Chunking
    status_text.text("Step 1/3: Chunking video...")
    progress_bar.progress(0.1)
    
    chunks = create_chunks(video_path, video_info, chunking_method, settings)
    
    st.success(f"✅ Created {len(chunks)} chunks")
    progress_bar.progress(0.3)
    
    # Step 2: Frame Selection
    status_text.text("Step 2/3: Selecting frames...")
    
    chunks_with_frames = select_frames_for_chunks(
        video_path, chunks, frame_method, settings, progress_bar
    )
    
    progress_bar.progress(0.7)
    
    # Step 3: Visualization
    status_text.text("Step 3/3: Generating visualizations...")
    
    visualize_results(chunks_with_frames, video_info, settings)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()


def create_chunks(video_path: str, video_info: Dict, method: str, 
                  settings: Dict) -> List[Tuple[float, float]]:
    """Create video chunks based on selected method"""
    
    if method == "Static Interval":
        chunk_duration = settings['chunk_duration']
        chunks = []
        current_time = 0
        total_duration = video_info['duration']
        
        while current_time < total_duration:
            end_time = min(current_time + chunk_duration, total_duration)
            chunks.append((current_time, end_time))
            current_time = end_time
        
        return chunks
    
    elif method == "Scene Detection":
        scene_threshold = settings['scene_threshold']
        scenes = detect_scenes(video_path, threshold=scene_threshold)
        return scenes
    
    else:  # Hybrid
        min_dur = settings['min_chunk_duration']
        max_dur = settings['max_chunk_duration']
        scene_threshold = settings['scene_threshold']
        
        scenes = detect_scenes(video_path, threshold=scene_threshold)
        
        # Apply constraints
        constrained_chunks = []
        i = 0
        while i < len(scenes):
            start, end = scenes[i]
            duration = end - start
            
            # Too short - merge with next
            while duration < min_dur and i + 1 < len(scenes):
                i += 1
                end = scenes[i][1]
                duration = end - start
            
            # Too long - split
            if duration > max_dur:
                current = start
                while current < end:
                    chunk_end = min(current + max_dur, end)
                    constrained_chunks.append((current, chunk_end))
                    current = chunk_end
            else:
                constrained_chunks.append((start, end))
            
            i += 1
        
        return constrained_chunks


def select_frames_for_chunks(video_path: str, chunks: List[Tuple[float, float]], 
                             method: str, settings: Dict, progress_bar) -> List[ChunkInfo]:
    """Select frames for each chunk based on method"""
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    chunks_with_frames = []
    
    for idx, (start_time, end_time) in enumerate(chunks):
        # Update progress
        progress = 0.3 + (0.4 * (idx / len(chunks)))
        progress_bar.progress(progress)
        
        duration = end_time - start_time
        
        if method == "Keyframe Only":
            frames, timestamps = extract_keyframes(
                cap, start_time, end_time, fps,
                num_keyframes=settings['num_keyframes']
            )
        
        elif method == "Dense Sampling":
            frames, timestamps = extract_dense_frames(
                cap, start_time, end_time, fps,
                sample_fps=settings['sample_fps']
            )
        
        else:  # Adaptive
            frames, timestamps = extract_adaptive_frames(
                cap, start_time, end_time, fps,
                motion_threshold=settings['static_threshold'],
                min_fps=settings['min_fps'],
                max_fps=settings['max_fps']
            )
        
        chunk_info = ChunkInfo(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            frames=frames,
            frame_timestamps=timestamps,
            chunk_id=idx
        )
        chunks_with_frames.append(chunk_info)
    
    cap.release()
    return chunks_with_frames


def visualize_results(chunks: List[ChunkInfo], video_info: Dict, settings: Dict):
    """Display comprehensive visualization of preprocessing results"""
    
    st.header("📊 Preprocessing Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_frames = sum(len(chunk.frames) for chunk in chunks)
    avg_frames_per_chunk = total_frames / len(chunks) if chunks else 0
    avg_chunk_duration = np.mean([chunk.duration for chunk in chunks])
    
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        st.metric("Total Frames Selected", total_frames)
    with col3:
        st.metric("Avg Frames/Chunk", f"{avg_frames_per_chunk:.1f}")
    with col4:
        st.metric("Avg Chunk Duration", f"{avg_chunk_duration:.1f}s")
    
    # Timeline visualization
    st.subheader("📅 Chunk Timeline")
    fig = create_timeline_chart(chunks, video_info['duration'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Frame distribution chart
    st.subheader("📈 Frame Distribution")
    fig = create_frame_distribution_chart(chunks)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display all chunks with frames
    st.subheader("🎞️ Chunk Preview")
    
    # Option to show all or select one
    view_mode = st.radio(
        "View mode",
        ["Show All Chunks", "Select Individual Chunk"],
        horizontal=True
    )
    
    if view_mode == "Select Individual Chunk":
        # Chunk selector
        chunk_idx = st.selectbox(
            "Select chunk to preview",
            range(len(chunks)),
            format_func=lambda x: f"Chunk {x+1} ({chunks[x].start_time:.1f}s - {chunks[x].end_time:.1f}s, {len(chunks[x].frames)} frames)"
        )
        
        selected_chunk = chunks[chunk_idx]
        
        st.write(f"**Chunk {chunk_idx + 1} Details:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Start: {selected_chunk.start_time:.2f}s")
        with col2:
            st.write(f"End: {selected_chunk.end_time:.2f}s")
        with col3:
            st.write(f"Frames: {len(selected_chunk.frames)}")
        
        # Display frames in a grid
        if selected_chunk.frames:
            st.write("**Selected Frames:**")
            
            # Determine grid layout
            num_frames = len(selected_chunk.frames)
            cols_per_row = min(4, num_frames)
            
            for i in range(0, num_frames, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < num_frames:
                        frame = selected_chunk.frames[i + j]
                        timestamp = selected_chunk.frame_timestamps[i + j]
                        
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        with col:
                            st.image(frame_rgb, caption=f"t={timestamp:.2f}s", use_container_width=True)
    
    else:  # Show All Chunks
        st.info(f"Showing all {len(chunks)} chunks with their selected frames")
        
        for chunk_idx, chunk in enumerate(chunks):
            with st.expander(
                f"📦 Chunk {chunk_idx + 1}: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s ({len(chunk.frames)} frames)",
                expanded=(chunk_idx < 3)  # Expand first 3 chunks by default
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Start", f"{chunk.start_time:.2f}s")
                with col2:
                    st.metric("Duration", f"{chunk.duration:.2f}s")
                with col3:
                    st.metric("Frames", len(chunk.frames))
                
                # Display frames in a grid
                if chunk.frames:
                    num_frames = len(chunk.frames)
                    cols_per_row = min(6, num_frames)  # Use 6 columns for compact view
                    
                    for i in range(0, num_frames, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < num_frames:
                                frame = chunk.frames[i + j]
                                timestamp = chunk.frame_timestamps[i + j]
                                
                                # Convert BGR to RGB for display
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                with col:
                                    st.image(frame_rgb, caption=f"{timestamp:.1f}s", use_container_width=True)


def create_timeline_chart(chunks: List[ChunkInfo], total_duration: float):
    """Create interactive timeline showing chunks and frame selections"""
    
    fig = go.Figure()
    
    # Add chunks as bars
    for chunk in chunks:
        fig.add_trace(go.Bar(
            x=[chunk.duration],
            y=[f"Chunk {chunk.chunk_id + 1}"],
            orientation='h',
            name=f"Chunk {chunk.chunk_id + 1}",
            text=f"{len(chunk.frames)} frames",
            textposition='inside',
            hovertemplate=(
                f"Chunk {chunk.chunk_id + 1}<br>"
                f"Start: {chunk.start_time:.2f}s<br>"
                f"End: {chunk.end_time:.2f}s<br>"
                f"Duration: {chunk.duration:.2f}s<br>"
                f"Frames: {len(chunk.frames)}<br>"
                "<extra></extra>"
            ),
            marker=dict(
                color=len(chunk.frames),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Frames")
            )
        ))
    
    fig.update_layout(
        title="Chunk Timeline (color intensity = frame count)",
        xaxis_title="Duration (seconds)",
        yaxis_title="Chunks",
        showlegend=False,
        height=max(400, len(chunks) * 30)
    )
    
    return fig


def create_frame_distribution_chart(chunks: List[ChunkInfo]):
    """Create chart showing frame count distribution across chunks"""
    
    chunk_labels = [f"Chunk {c.chunk_id + 1}" for c in chunks]
    frame_counts = [len(c.frames) for c in chunks]
    durations = [c.duration for c in chunks]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Frames per Chunk", "Chunk Durations")
    )
    
    # Frames per chunk
    fig.add_trace(
        go.Bar(
            x=chunk_labels,
            y=frame_counts,
            name="Frame Count",
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # Chunk durations
    fig.add_trace(
        go.Bar(
            x=chunk_labels,
            y=durations,
            name="Duration (s)",
            marker_color='lightcoral'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Chunk", row=1, col=1)
    fig.update_xaxes(title_text="Chunk", row=1, col=2)
    fig.update_yaxes(title_text="Frame Count", row=1, col=1)
    fig.update_yaxes(title_text="Duration (s)", row=1, col=2)
    
    return fig


def display_methodology():
    """Display methodology explanation when no video is uploaded"""
    
    st.header("📚 Methodology")
    
    tab1, tab2, tab3 = st.tabs(["Chunking", "Frame Selection", "Compression"])
    
    with tab1:
        st.subheader("Chunking Strategies")
        
        st.markdown("""
        **Static Interval:**
        - Divides video into fixed-duration segments
        - ✅ Simple, predictable, parallelizable
        - ❌ Ignores content, may split scenes awkwardly
        
        **Scene Detection:**
        - Uses shot boundary detection to find natural transitions
        - ✅ Semantically coherent chunks
        - ❌ Variable lengths, computational overhead
        
        **Hybrid (Recommended):**
        - Scene detection with min/max duration constraints
        - ✅ Balances semantic coherence with system requirements
        - ⚠️ More complex implementation
        """)
    
    with tab2:
        st.subheader("Frame Selection Strategies")
        
        st.markdown("""
        **Keyframe Only:**
        - Selects 1-3 most representative frames per chunk
        - ✅ Minimal storage and compute
        - ❌ May miss brief but important content
        
        **Dense Sampling:**
        - Extracts frames at regular intervals (e.g., 1 fps)
        - ✅ Comprehensive coverage of visual content
        - ❌ Higher storage and processing costs
        
        **Adaptive Sampling (Recommended):**
        - Varies sampling rate based on motion/change detection
        - ✅ Optimizes storage vs. quality tradeoff
        - ⚠️ Requires content analysis during preprocessing
        """)
        
        st.info("💡 **Frame selection is the most critical component** - it directly impacts search recall and precision.")
    
    with tab3:
        st.subheader("Compression Settings")
        
        st.markdown("""
        **Codec:** H.264/AVC
        - Industry standard, broad compatibility
        - Mature tooling (FFmpeg)
        
        **Quality Levels (CRF):**
        - CRF 18: High quality, larger files
        - CRF 23: Recommended balance
        - CRF 28: Lower quality, smaller files
        
        **Resolution:**
        - Match AI model input requirements
        - 512x512 or 720p recommended
        - Reduces processing by order of magnitude
        """)


if __name__ == "__main__":
    main()
