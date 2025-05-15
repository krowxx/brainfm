"""Panning effects for stereo audio."""

import numpy as np

def apply_autopan(samples, sample_rate, pan_rate):
    """Apply a stereo autopan effect that slowly shifts audio between channels.
    
    Args:
        samples: numpy array of shape [channels, samples]
        sample_rate: audio sample rate in Hz
        pan_rate: frequency of the autopan in Hz (0 disables the effect)
        
    Returns:
        Modified samples array
    """
    if pan_rate <= 0 or samples.shape[0] < 2:
        return samples  # Do nothing if pan_rate is 0 or mono audio
        
    # Create a panning envelope that oscillates between -1 and 1
    t = np.arange(samples.shape[1]) / sample_rate
    pan_envelope = np.sin(2 * np.pi * pan_rate * t)
    
    # Apply panning
    left_gain = (1 + pan_envelope) / 2  # Ranges from 0 to 1
    right_gain = (1 - pan_envelope) / 2  # Opposite of left
    
    # Apply gains to channels
    samples[0, :] *= left_gain  # Left channel
    samples[1, :] *= right_gain  # Right channel
    
    return samples 