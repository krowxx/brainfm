"""Core audio processing functions."""

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

from brainfm.effects.reverb import apply_reverb
from brainfm.effects.filters import soften_transients
from brainfm.effects.modulation import apply_modulation, multiband_modulate
from brainfm.effects.panning import apply_autopan
from brainfm.effects.noise import add_pink_noise
from brainfm.utils.conversion import segment_to_samples, samples_to_segment

def process_audio(mp3_path, output_path, mod_freq=17.0, depth=0.35,
                 tempo=1.0, reverb=0, soften=0,
                 pan_rate=0.0, noise_db=None, multiband=False):
    """Apply BrainFM-style modulation and effects to audio.
    
    Args:
        mp3_path: Path to input MP3 file
        output_path: Path to save processed MP3 file
        mod_freq: Modulation frequency in Hz (default: 17.0)
        depth: Modulation depth (default: 0.35)
        tempo: Tempo adjustment factor (default: 1.0)
        reverb: Reverb amount (0-1, default: 0)
        soften: Soften percussive elements (0-1, default: 0)
        pan_rate: Slow autopan rate in Hz (0 disables, default: 0)
        noise_db: Level (dBFS) of pink noise bed (negative value, e.g. -32; None disables)
        multiband: Enable multi-band modulation (default: False)
    
    Returns:
        None. Exports the processed audio to output_path.
    """
    print(f"Processing: {mp3_path}")
    audio = AudioSegment.from_file(mp3_path)
    
    # Apply tempo change if different from default
    if tempo != 1.0:
        print(f"Adjusting tempo: {tempo}x")
        # For tempo adjustments, we'll multiply the frame rate
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * tempo)
        })
        # Converting back to the original frame rate (this changes the sound)
        audio = audio.set_frame_rate(audio.frame_rate)
    
    # Apply reverb if specified
    if reverb > 0:
        audio = apply_reverb(audio, reverb)
    
    # Apply transient softening if specified
    if soften > 0:
        audio = soften_transients(audio, soften)
    
    # --- optional multiband modulation ---
    if multiband:
        audio = multiband_modulate(audio, mod_freq, depth)
        # Modulation is done, skip single-band envelope later
        modulated_audio = audio
    else:
        # Convert to numpy samples for processing
        samples, sample_rate, sample_width, channels = segment_to_samples(audio)

        # Apply single-band modulation
        modulated = apply_modulation(samples, sample_rate, mod_freq, depth)
        
        # Optional slow stereo autopan
        apply_autopan(modulated, sample_rate, pan_rate)
        
        # Convert back to AudioSegment
        modulated_audio = samples_to_segment(modulated, audio)

    # Add pink/brown noise bed if requested
    modulated_audio = add_pink_noise(modulated_audio, noise_db)

    # Export final audio
    modulated_audio.export(output_path, format="mp3")
    print(f"✅ Done: {output_path}") 