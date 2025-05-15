import sys
import argparse
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter

def apply_reverb(audio, reverb_amount):
    """Apply a simple reverb effect to the audio."""
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

def soften_transients(audio, amount):
    """Apply effects to soften percussive sounds like drums."""
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

def apply_brainfm_modulation(mp3_path, output_path, mod_freq=17.0, depth=0.35, tempo=1.0, reverb=0, soften=0):
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
    
    sample_rate = audio.frame_rate
    sample_width = audio.sample_width
    channels = audio.channels

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if channels == 2:
        samples = samples.reshape((-1, 2)).T  # shape: [2, n]
    else:
        samples = samples[np.newaxis, :]  # shape: [1, n]

    # Create modulation envelope
    t = np.arange(samples.shape[1]) / sample_rate
    envelope = 1.0 + depth * np.sin(2 * np.pi * mod_freq * t)
    envelope = np.tile(envelope, (channels, 1))  # same envelope for both channels

    # Apply modulation
    modulated = samples * envelope

    # Clip to int16 range
    max_val = float(2 ** (8 * sample_width - 1) - 1)
    modulated = np.clip(modulated, -max_val, max_val)
    modulated = modulated.astype(np.int16)

    if channels == 2:
        modulated = modulated.T.reshape((-1,))
    else:
        modulated = modulated.reshape((-1,))

    # Create AudioSegment from modulated samples
    modulated_audio = audio._spawn(modulated.tobytes())
    modulated_audio.export(output_path, format="mp3")
    print(f"✅ Done: {output_path}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply BrainFM-like modulation to audio files")
    parser.add_argument("input_mp3", help="Input MP3 file path")
    parser.add_argument("output_mp3", help="Output MP3 file path")
    parser.add_argument("--tempo", type=float, default=1.0, 
                        help="Tempo adjustment factor (0.5 = half speed, 2.0 = double speed)")
    parser.add_argument("--reverb", type=float, default=0.0,
                        help="Reverb amount (0-1, default: 0)")
    parser.add_argument("--soften", type=float, default=0.0,
                        help="Soften percussive elements like drums (0-1, default: 0)")
    parser.add_argument("--mod-freq", type=float, default=17.0,
                        help="Modulation frequency in Hz (default: 17.0)")
    parser.add_argument("--depth", type=float, default=0.35,
                        help="Modulation depth (default: 0.35)")
    
    args = parser.parse_args()
    
    apply_brainfm_modulation(
        args.input_mp3, 
        args.output_mp3,
        mod_freq=args.mod_freq,
        depth=args.depth,
        tempo=args.tempo,
        reverb=args.reverb,
        soften=args.soften
    )