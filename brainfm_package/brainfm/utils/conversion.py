"""Audio format conversion utilities."""

import numpy as np

def segment_to_samples(segment):
    """Convert a PyDub AudioSegment to numpy sample array.
    
    Args:
        segment: PyDub AudioSegment
        
    Returns:
        Tuple of (samples, sample_rate, sample_width, channels)
        where samples is a numpy array of shape [channels, n_samples]
    """
    sample_rate = segment.frame_rate
    sample_width = segment.sample_width
    channels = segment.channels
    
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    if channels == 2:
        samples = samples.reshape((-1, 2)).T  # shape: [2, n]
    else:
        samples = samples[np.newaxis, :]  # shape: [1, n]
        
    return samples, sample_rate, sample_width, channels

def samples_to_segment(samples, template_segment):
    """Convert numpy sample array back to PyDub AudioSegment.
    
    Args:
        samples: numpy array of shape [channels, n_samples]
        template_segment: PyDub AudioSegment to use as template
        
    Returns:
        PyDub AudioSegment with the provided samples
    """
    channels = template_segment.channels
    sample_width = template_segment.sample_width
    
    # Clip to int16 range
    max_val = float(2 ** (8 * sample_width - 1) - 1)
    samples = np.clip(samples, -max_val, max_val)
    samples = samples.astype(np.int16)

    if channels == 2:
        samples = samples.T.reshape((-1,))
    else:
        samples = samples.reshape((-1,))

    # Create AudioSegment from samples
    return template_segment._spawn(samples.tobytes()) 