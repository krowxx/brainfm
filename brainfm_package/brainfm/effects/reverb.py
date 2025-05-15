"""Reverb audio effect."""

from pydub.effects import normalize

def apply_reverb(audio, reverb_amount):
    """Apply a simple reverb effect to the audio.
    
    Args:
        audio: PyDub AudioSegment
        reverb_amount: Float between 0 and 1 controlling reverb intensity
        
    Returns:
        PyDub AudioSegment with reverb effect applied
    """
    if reverb_amount <= 0:
        return audio
    
    print(f"Adding reverb: {reverb_amount}")
    # Create a reverb effect by adding delayed and attenuated copies of the original audio
    result = audio
    
    # Add 3 echoes with decreasing volume
    for i in range(1, 4):
        delay = int(i * 100 * reverb_amount)  # Delay in milliseconds
        attenuation = 1.0 - (0.15 * i * reverb_amount)  # Volume reduction factor
        
        # Add the delayed and attenuated copy
        delayed_audio = audio[0:].overlay(
            audio - (10 * i * reverb_amount), 
            position=delay
        )
        
        # Mix with result
        result = result.overlay(
            delayed_audio - (6 * i * reverb_amount), 
            position=0
        )
    
    # Normalize the result to avoid clipping
    return normalize(result) 