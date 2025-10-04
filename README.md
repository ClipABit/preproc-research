# Video Preprocessing Research

Demo for video preprocessing strategies in semantic search. Part of [ClipABit](https://github.com/ClipABit) @ WAT.ai.

## Quick Start

```bash
uv sync
uv run streamlit run app.py
```

**Requires:** Python 3.9+, FFmpeg (optional)

Experiment with chunking strategies, frame selection algorithms (keyframe, dense, adaptive), and compression settings through an interactive UI.

## 📖 Usage Guide

### 1. Upload a Video

- Click "Browse files" and select a video (MP4, AVI, MOV, MKV)
- The app will display video metadata (duration, resolution, FPS, etc.)

### 2. Configure Preprocessing

**Chunking Strategy:**
- **Static Interval**: Fixed-duration chunks (simple, predictable)
- **Scene Detection**: Natural scene boundaries (semantic coherence)
- **Hybrid**: Scene detection with min/max constraints (recommended)

**Frame Selection Strategy (Most Important!):**
- **Keyframe Only**: 1-3 representative frames per chunk (minimal storage)
- **Dense Sampling**: Frames at regular intervals (comprehensive coverage)
- **Adaptive Sampling**: Varies rate based on motion (optimal balance)

**Compression Settings:**
- Choose target resolution (512x512, 720p, original)
- Select quality preset (CRF 18/23/28)

### 3. Process and Visualize

Click "🚀 Process Video" to run preprocessing with your settings. The app will show:

- **Summary Statistics**: Total chunks, frames selected, averages
- **Timeline Chart**: Visual representation of chunks and frame distribution
- **Frame Distribution**: How frames are distributed across chunks
- **Chunk Preview**: View selected frames from any chunk

## 🧪 Experimentation Guide

### Understanding Frame Selection

Frame selection is the **most critical component** for search quality. Try these experiments:

**Experiment 1: Static vs. Dynamic Content**
- Upload a video with both static shots and action sequences
- Compare "Keyframe Only" vs "Dense Sampling" vs "Adaptive Sampling"
- Notice how adaptive sampling allocates more frames to dynamic scenes

**Experiment 2: Scene Coherence**
- Use "Static Interval" with 10-second chunks
- Switch to "Scene Detection"
- Observe how scene detection produces semantically complete chunks

**Experiment 3: Storage vs. Quality Tradeoff**
- Try "Dense Sampling" at 1 fps (comprehensive but storage-heavy)
- Switch to "Adaptive Sampling" (smart allocation)
- Compare total frame counts - adaptive should be 30-50% fewer frames while capturing key moments

### Recommended Settings for MVP

Based on the preprocessing report, here are recommended starting points:

**For Development/Testing:**
- Chunking: Hybrid (5s min, 20s max)
- Frame Selection: Dense Sampling (1 fps)
- Compression: 720p, CRF 23

**For Production (optimized):**
- Chunking: Hybrid (5s min, 20s max)
- Frame Selection: Adaptive Sampling (0.5-2 fps range)
- Compression: 512x512 or 720p, CRF 23

## 📊 Understanding the Visualizations

### Timeline Chart
- Each bar represents a chunk
- Color intensity = number of frames selected
- Hover to see detailed chunk info

### Frame Distribution
- Left chart: Frame count per chunk
- Right chart: Chunk durations
- Helps identify if chunks are balanced

### Chunk Preview
- Shows actual frames selected from a chunk
- Timestamps indicate when each frame occurs
- Use this to verify frame selection quality

## 🏗️ Project Context

This demo is part of the **ClipABit** semantic video search project at WAT.ai. The preprocessing stage serves as the foundation for:

1. **Information Extraction**: Video/image models process selected frames
2. **Embedding Generation**: Create semantic embeddings from frames
3. **Search**: Natural language queries find relevant moments

**Key Insight**: Poor frame selection cascades through the entire pipeline. Missing a critical frame means it won't be searchable. Over-sampling wastes storage and compute.

## 🔧 Technical Architecture

```
app.py                  # Main Streamlit application
├── frame_selector.py   # Frame selection algorithms
│   ├── extract_keyframes()
│   ├── extract_dense_frames()
│   └── extract_adaptive_frames()
├── scene_detector.py   # Scene detection utilities
│   ├── detect_scenes()
│   └── apply_scene_constraints()
└── video_processor.py  # Compression & metadata
    ├── get_video_info()
    ├── compress_video()
    └── resize_video()
```

## 📝 Implementation Details

### Frame Selection Algorithms

**Keyframe Selection:**
- Extracts all frames from chunk
- Uses diversity-based selection (maximizes visual difference)
- Returns N most representative frames

**Dense Sampling:**
- Fixed sampling rate (e.g., 1 frame per second)
- Guarantees comprehensive coverage
- Parallelizable extraction

**Adaptive Sampling:**
1. Analyzes motion/change across chunk
2. Calculates average motion score
3. Interpolates sampling rate: low motion = 0.5 fps, high motion = 2 fps
4. Extracts frames at determined rate

### Scene Detection

Uses PySceneDetect with ContentDetector:
- Analyzes HSV histogram differences between frames
- Threshold controls sensitivity (lower = more scenes)
- Falls back to simple histogram difference if PySceneDetect unavailable

### Hybrid Chunking

1. Run scene detection
2. Merge scenes shorter than minimum duration
3. Split scenes longer than maximum duration
4. Results in bounded, semantically coherent chunks

## 📚 References

- **PySceneDetect**: [scenedetect.com](https://www.scenedetect.com/)
- **FFmpeg**: [ffmpeg.org](https://ffmpeg.org/)
- **Preprocessing Report**: See `preprocessing_report.md` for comprehensive analysis

## 🐛 Troubleshooting

**"FFmpeg not found" error:**
- Ensure FFmpeg is installed and in your PATH
- Compression features require FFmpeg

**Slow processing:**
- Scene detection is compute-intensive
- Try reducing video resolution first
- Use static chunking for faster preview

**Memory issues:**
- Large videos may require significant RAM
- Try processing shorter clips first
- Reduce dense sampling rate


