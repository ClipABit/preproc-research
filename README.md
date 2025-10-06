# Video Preprocessing Research

Comparing multimodal embedding strategies for semantic video search. Part of [ClipABit](https://github.com/ClipABit) @ WAT.ai.

## Quick Start

```bash
uv sync
uv run streamlit run app.py
```

**Requires:** Python 3.9+, FFmpeg (optional)

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


- **PySceneDetect**: Scene boundary detection via content analysis
- **OpenCV**: Optical flow, histogram comparison, frame extraction
- **FFmpeg**: Video compression and format conversion

