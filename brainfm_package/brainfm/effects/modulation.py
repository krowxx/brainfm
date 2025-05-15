"""Amplitude modulation effects."""

import numpy as np
from pydub.effects import normalize, low_pass_filter, high_pass_filter

from brainfm.effects.filters import band_pass_filter
from brainfm.utils.conversion import segment_to_samples, samples_to_segment

def apply_modulation(samples, sample_rate, mod_freq, depth):
    """Apply simple amplitude modulation to samples.
    
    Args:
        samples: numpy array of shape [channels, n_samples]
        sample_rate: audio sample rate in Hz
        mod_freq: modulation frequency in Hz
        depth: modulation depth (0-1)
        
    Returns:
        Modulated samples array
    """
    if depth <= 0:
        return samples
        
    t = np.arange(samples.shape[1]) / sample_rate
    envelope = 1.0 + depth * np.sin(2 * np.pi * mod_freq * t)
    envelope = np.tile(envelope, (samples.shape[0], 1))  # same envelope for all channels

    # Apply modulation
    return samples * envelope

def _amp_modulate(samples, sr, freq, depth):
    """Amplitude‑modulate a numpy sample array in‑place."""
    if depth <= 0:
        return samples
    t = np.arange(samples.shape[1]) / sr
    env = 1.0 + depth * np.sin(2 * np.pi * freq * t)
    env = np.tile(env, (samples.shape[0], 1))
    return samples * env

def multiband_modulate(audio, mod_freq, depth):
    """Apply Brain.fm‑style multi‑band modulation.
    
    Modulation characteristics by frequency band:
      • sub (<120 Hz): shallow, ≤8 Hz
      • low‑mid (120‑500 Hz): moderate at mod_freq
      • high‑mid (500‑4 kHz): strongest at mod_freq
      • air (>4 kHz): very light
      
    Args:
        audio: PyDub AudioSegment
        mod_freq: base modulation frequency in Hz
        depth: overall modulation depth (0-1)
        
    Returns:
        PyDub AudioSegment with multiband modulation applied
    """
    # Split bands
    sub      = low_pass_filter(audio, 120)
    low_mid  = band_pass_filter(audio, 120, 500)
    high_mid = band_pass_filter(audio, 500, 4000)
    air      = high_pass_filter(audio, 4000)

    # Convert to numpy arrays
    sub_samples, sr, sw, ch = segment_to_samples(sub)
    low_mid_samples, _, _, _ = segment_to_samples(low_mid)
    high_mid_samples, _, _, _ = segment_to_samples(high_mid)
    air_samples, _, _, _ = segment_to_samples(air)

    # Process each band
    sub_mod      = _amp_modulate(sub_samples, sr, min(mod_freq, 8), min(depth * 0.4, 0.10))
    low_mid_mod  = _amp_modulate(low_mid_samples, sr, mod_freq, max(0.15, depth * 0.7))
    high_mid_mod = _amp_modulate(high_mid_samples, sr, mod_freq, depth)
    air_mod      = _amp_modulate(air_samples, sr, mod_freq, min(depth * 0.3, 0.10))

    # Convert back to AudioSegments
    sub_seg      = samples_to_segment(sub_mod, sub)
    low_mid_seg  = samples_to_segment(low_mid_mod, low_mid)
    high_mid_seg = samples_to_segment(high_mid_mod, high_mid)
    air_seg      = samples_to_segment(air_mod, air)

    # Mix bands and normalize
    combined = sub_seg.overlay(low_mid_seg).overlay(high_mid_seg).overlay(air_seg)
    return normalize(combined) 