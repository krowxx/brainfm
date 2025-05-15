import sys
import numpy as np
from pydub import AudioSegment

def apply_brainfm_modulation(mp3_path, output_path, mod_freq=16.0, depth=0.3):
    print(f"Processing: {mp3_path}")
    audio = AudioSegment.from_file(mp3_path)
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
    if len(sys.argv) != 3:
        print("Usage: python brainfm_modulator.py input.mp3 output.mp3")
        sys.exit(1)
    
    input_mp3 = sys.argv[1]
    output_mp3 = sys.argv[2]
    apply_brainfm_modulation(input_mp3, output_mp3)