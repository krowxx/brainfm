# BrainFM

Apply Brain.fm style frequency modulation and effects to audio files.

## Installation

```bash
# Install from current directory
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Basic usage
brainfm input.mp3 output.mp3

# With parameters
brainfm input.mp3 output.mp3 \
  --tempo 0.75 \
  --mod-freq 18 \
  --depth 0.30 \
  --pan-rate 0.2 \
  --noise-db -32 \
  --reverb 0.3 \
  --soften 0.5 \
  --multiband
```

### Python API

```python
from brainfm import process_audio

# Process an audio file
process_audio(
    "input.mp3", 
    "output.mp3",
    mod_freq=17.0,
    depth=0.35,
    tempo=1.0,
    reverb=0.3,
    soften=0.2,
    pan_rate=0.2,
    noise_db=-32,
    multiband=True
)
```

## Available Effects

- **Frequency Modulation**: Apply amplitude modulation at specified frequency
- **Multiband Modulation**: Apply different modulation parameters to different frequency bands
- **Tempo Adjustment**: Speed up or slow down audio
- **Reverb**: Add spaciousness and depth
- **Transient Softening**: Smooth out percussive elements
- **Autopanning**: Slowly shift audio between stereo channels
- **Noise Layer**: Add background pink noise at specified level

## Extensions

The modular design makes it easy to add new effects. To add a new effect:

1. Create a new module in the `effects` directory
2. Implement your effect function(s)
3. Update the `core.py` pipeline to include your effect
4. Add CLI parameters in `cli.py` if needed 