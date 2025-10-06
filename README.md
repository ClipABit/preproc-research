# Video Preprocessing Research

Comparing multimodal embedding strategies for semantic video search. Part of [ClipABit](https://github.com/ClipABit) @ WAT.ai.

## Quick Start

```bash
uv sync
uv run streamlit run app.py
```

**Requires:** Python 3.9+, FFmpeg (optional)

## Research Goals

Compare approaches for creating searchable video embeddings:

1. **Scene-based LLM descriptions**: PySceneDetect → keyframes → LLM → text embeddings
2. **Dense visual embeddings**: Fixed interval frame extraction → CLIP/image model
3. **Adaptive visual embeddings**: Motion-based sampling → CLIP/image model
4. **Multimodal fusion**: Combine video, audio transcription, and text descriptions
5. **Reranking strategies**: Vector search vs. reranking with larger models

**Key Question**: Which modality combination provides best accuracy/cost tradeoff?

##  Preprocessing Techniques

### Chunking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Static Interval** | Fixed-duration chunks (e.g., 10s) | Simple baseline, predictable |
| **Scene Detection** | PySceneDetect content detector | Semantic boundaries, variable length |
| **Hybrid** | Scene detection + min/max constraints | **Recommended**: Balances semantics & consistency |

### Frame Selection Algorithms

| Algorithm | Sampling Rate | Method | Best For |
|-----------|---------------|--------|----------|
| **Keyframe** | 1-3 frames/chunk | Diversity-based selection | LLM descriptions (minimize API costs) |
| **Dense Sampling** | 0.5-2 fps fixed | Regular intervals | Baseline comparison, comprehensive coverage |
| **Adaptive Sampling** | 0.5-2 fps variable | Motion-based rate adjustment | Production (balances quality/storage) |
| **Action Frames** | N peak moments | Optical flow peak detection | Sports, dynamic content |

**Implementation Details:**
- **Keyframe**: Maximizes visual diversity using histogram correlation
- **Dense**: Uniform temporal coverage, parallelizable
- **Adaptive**: Analyzes motion scores, adjusts rate dynamically (static=0.5fps, dynamic=2fps)
- **Action**: Uses optical flow to find motion peaks

### Recommended Settings

**For LLM Text Embedding Pipeline:**
- Chunking: Scene Detection (semantic coherence)
- Frames: Keyframe (1-2 per scene)
- Rationale: Minimize expensive LLM API calls

**For CLIP Image Embedding:**
- Chunking: Hybrid (5-20s)
- Frames: Adaptive Sampling (0.5-2 fps)
- Rationale: Balance coverage with storage

**For Multimodal (Video + Audio + Text):**
- Chunking: Hybrid (5-20s)
- Video: Adaptive frames for CLIP
- Text: Keyframes for LLM descriptions
- Audio: Fixed 1fps timestamps for alignment
- Rationale: Optimize each modality independently

##  Comparison Metrics

Track these for each approach:
- **Processing time**: Frame extraction, model inference
- **Storage**: Total frames, compressed size
- **Model costs**: API calls (LLM), inference time (CLIP)
- **Search accuracy**: Precision/recall on test queries
- **Resource usage**: Memory, GPU utilization

## Technical Architecture

```
app.py                  # Streamlit demo UI
├── frame_selector.py   # Frame selection algorithms
│   ├── extract_keyframes()        # Diversity-based selection
│   ├── extract_dense_frames()     # Fixed interval sampling
│   ├── extract_adaptive_frames()  # Motion-based variable rate
│   └── extract_action_frames()    # Optical flow peak detection
├── scene_detector.py   # Scene detection utilities
│   ├── detect_scenes()            # PySceneDetect wrapper
│   └── apply_scene_constraints()  # Min/max duration constraints
└── video_processor.py  # Compression & metadata
    ├── get_video_info()
    ├── compress_video()
    └── resize_video()
```

## References

- **PySceneDetect**: Scene boundary detection via content analysis
- **OpenCV**: Optical flow, histogram comparison, frame extraction
- **FFmpeg**: Video compression and format conversion

