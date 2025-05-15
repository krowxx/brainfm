"""Noise generation effects."""

from pydub.generators import WhiteNoise

def add_pink_noise(audio, noise_db):
    """Overlay low‑level pink‑style noise under the whole track.
    
    Args:
        audio: PyDub AudioSegment
        noise_db: Negative number for noise level in dB (e.g. -32). None or 0 disables.
        
    Returns:
        PyDub AudioSegment with noise added
    """
    if noise_db is None or noise_db >= 0:
        return audio

    print(f"Adding pink noise bed at {noise_db} dBFS")
    # Pydub has WhiteNoise; approximate pink by low‑passing it.
    noise = WhiteNoise().to_audio_segment(duration=len(audio)).low_pass_filter(8000)
    noise = noise.apply_gain(noise_db)  # set level
    return audio.overlay(noise) 