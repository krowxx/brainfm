"""Audio filter effects."""

from pydub.effects import normalize, low_pass_filter, high_pass_filter

def band_pass_filter(audio, low_cutoff, high_cutoff):
    """Apply a band-pass filter to audio.
    
    Args:
        audio: PyDub AudioSegment
        low_cutoff: Low frequency cutoff in Hz
        high_cutoff: High frequency cutoff in Hz
        
    Returns:
        PyDub AudioSegment with band-pass filter applied
    """
    bp = high_pass_filter(audio, low_cutoff)
    bp = low_pass_filter(bp, high_cutoff)
    return bp

def soften_transients(audio, amount):
    """Apply effects to soften percussive sounds like drums.
    
    Args:
        audio: PyDub AudioSegment
        amount: Float between 0 and 1 controlling softening intensity
        
    Returns:
        PyDub AudioSegment with softened transients
    """
    if amount <= 0:
        return audio
    
    print(f"Softening percussive elements: {amount}")
    result = audio
    
    # 1. Apply a low-pass filter to reduce high frequencies (transients)
    cutoff = 8000 - (amount * 3000)  # Reduce cutoff as amount increases
    result = low_pass_filter(result, cutoff)
    
    # 2. Add a subtle "attack softener" by mixing in a slightly delayed version
    # This simulates a slower attack time
    delay_ms = int(5 + (15 * amount))  # 5-20ms delay based on amount
    volume_reduction = amount * 6  # dB reduction for the original signal
    
    # Create delayed copy (attack portion only)
    delayed = audio[delay_ms:]
    
    # Mix original (reduced) with delayed copy
    softened = (audio - volume_reduction).overlay(delayed, position=0)
    
    # Mix with result based on amount
    result = result.overlay(softened - 3, position=0)
    
    # Normalize to avoid clipping
    return normalize(result) 