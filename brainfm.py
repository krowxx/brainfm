import sys
import argparse
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
from pydub.generators import WhiteNoise
import gc
import tempfile
import os


CHUNK_SIZE_MS = 60000

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

def add_pink_noise(audio, noise_db):
    """
    Overlay low‑level pink‑style noise under the whole track.
    `noise_db` should be a negative number (e.g. -32). 0 disables the layer.
    """
    if noise_db is None or noise_db >= 0:
        return audio

    print(f"Adding pink noise bed at {noise_db} dBFS")
    # Pydub has WhiteNoise; approximate pink by low‑passing it.
    noise = WhiteNoise().to_audio_segment(duration=len(audio)).low_pass_filter(8000)
    noise = noise.apply_gain(noise_db)  # set level
    return audio.overlay(noise)

def band_pass_filter(audio, low_cutoff, high_cutoff):
    """Return a simple band‑pass (high‑pass then low‑pass)."""
    bp = high_pass_filter(audio, low_cutoff)
    bp = low_pass_filter(bp, high_cutoff)
    return bp

def _amp_modulate(samples, sr, freq, depth):
    """Amplitude‑modulate a numpy sample array in‑place."""
    if depth <= 0:
        return samples
    t = np.arange(samples.shape[1]) / sr
    env = 1.0 + depth * np.sin(2 * np.pi * freq * t)
    env = np.tile(env, (samples.shape[0], 1))
    # Modify samples in-place to save memory
    samples *= env
    return samples

def multiband_modulate(audio, mod_freq, depth):
    """
    Apply Brain.fm‑style multi‑band modulation:
      • sub (<120 Hz): shallow, ≤8 Hz
      • low‑mid (120‑500 Hz): moderate at mod_freq
      • high‑mid (500‑4 kHz): strongest at mod_freq
      • air (>4 kHz): very light
    """
    sr = audio.frame_rate
    sw = audio.sample_width
    ch = audio.channels

    # Split bands
    sub = low_pass_filter(audio, 120)
    low_mid = band_pass_filter(audio, 120, 500)
    high_mid = band_pass_filter(audio, 500, 4000)
    air = high_pass_filter(audio, 4000)

    def seg2np(seg):
        arr = np.array(seg.get_array_of_samples()).astype(np.float32)
        return arr.reshape((-1, ch)).T if ch == 2 else arr[np.newaxis, :]

    def np2seg(arr, template):
        if ch == 2:
            arr = arr.T.reshape((-1,))
        else:
            arr = arr.reshape((-1,))
        return template._spawn(arr.astype(np.int16).tobytes())

    # Process each band
    sub_mod = _amp_modulate(seg2np(sub), sr, min(mod_freq, 8), min(depth * 0.4, 0.10))
    gc.collect()  # Collect garbage to free memory
    
    low_mid_mod = _amp_modulate(seg2np(low_mid), sr, mod_freq, max(0.15, depth * 0.7))
    gc.collect()
    
    high_mid_mod = _amp_modulate(seg2np(high_mid), sr, mod_freq, depth)
    gc.collect()
    
    air_mod = _amp_modulate(seg2np(air), sr, mod_freq, min(depth * 0.3, 0.10))
    gc.collect()

    # Convert back to AudioSegments
    sub_seg = np2seg(sub_mod, sub)
    del sub_mod
    gc.collect()
    
    low_mid_seg = np2seg(low_mid_mod, low_mid)
    del low_mid_mod
    gc.collect()
    
    high_mid_seg = np2seg(high_mid_mod, high_mid)
    del high_mid_mod
    gc.collect()
    
    air_seg = np2seg(air_mod, air)
    del air_mod
    gc.collect()

    # Mix bands and normalize
    combined = sub_seg.overlay(low_mid_seg)
    del sub_seg
    gc.collect()
    
    combined = combined.overlay(high_mid_seg)
    del low_mid_seg
    gc.collect()
    
    combined = combined.overlay(air_seg)
    del high_mid_seg, air_seg
    gc.collect()
    
    return normalize(combined)

def apply_autopan(samples, sample_rate, pan_rate, pan_depth=0.5):
    """Apply a stereo autopan effect that slowly shifts audio between channels."""
    if pan_rate <= 0 or samples.shape[0] < 2:
        return samples
        
    # Create a panning envelope that oscillates between -1 and 1
    t = np.arange(samples.shape[1]) / sample_rate
    # Scale down the envelope by pan_depth to limit range
    pan_envelope = pan_depth * np.sin(2 * np.pi * pan_rate * t)
    
    # Apply panning
    left_gain = (1 + pan_envelope) / 2  # Now ranges from (1-depth)/2 to (1+depth)/2
    right_gain = (1 - pan_envelope) / 2  # Opposite of left
    
    # Apply gains to channels
    samples[0, :] *= left_gain
    samples[1, :] *= right_gain
    
    return samples

def process_audio_chunk(chunk, mod_freq, depth, reverb, soften, pan_rate, noise_db, multiband):
    """Process a single chunk of audio with all effects."""
    # Apply reverb if specified
    if reverb > 0:
        chunk = apply_reverb(chunk, reverb)
    
    # Apply transient softening if specified
    if soften > 0:
        chunk = soften_transients(chunk, soften)
    
    # Apply multi-band or single-band modulation
    if multiband:
        chunk = multiband_modulate(chunk, mod_freq, depth)
    else:
        sample_rate = chunk.frame_rate
        sample_width = chunk.sample_width
        channels = chunk.channels

        samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
        if channels == 2:
            samples = samples.reshape((-1, 2)).T
        else:
            samples = samples[np.newaxis, :]

        # Apply modulation
        t = np.arange(samples.shape[1]) / sample_rate
        envelope = 1.0 + depth * np.sin(2 * np.pi * mod_freq * t)
        envelope = np.tile(envelope, (channels, 1))
        samples *= envelope

        # Apply autopan
        if pan_rate > 0 and channels == 2:
            apply_autopan(samples, sample_rate, pan_rate)

        # Clip to int16 range
        max_val = float(2 ** (8 * sample_width - 1) - 1)
        samples = np.clip(samples, -max_val, max_val)
        samples = samples.astype(np.int16)

        # Convert back to AudioSegment
        if channels == 2:
            samples = samples.T.reshape((-1,))
        else:
            samples = samples.reshape((-1,))
        chunk = chunk._spawn(samples.tobytes())

    # Add pink noise
    if noise_db is not None and noise_db < 0:
        chunk = add_pink_noise(chunk, noise_db)
    
    return chunk

def apply_brainfm_modulation(mp3_path, output_path, mod_freq=17.0, depth=0.35,
                             tempo=1.0, reverb=0, soften=0,
                             pan_rate=0.0, noise_db=None, multiband=False):
    try:
        print(f"Processing: {mp3_path}")

        # Get audio info first without loading full content
        info = AudioSegment.from_file(mp3_path, duration=100).set_sample_width(2)
        sample_rate = info.frame_rate
        channels = info.channels
        
        # Apply tempo change to frame rate if needed
        if tempo != 1.0:
            print(f"Adjusting tempo: {tempo}x")
            target_rate = int(sample_rate * tempo)
        else:
            target_rate = sample_rate
        
        # Create a temporary directory for processed chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            # Check if the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Process the file in chunks
            input_audio = AudioSegment.from_file(mp3_path)
            total_length = len(input_audio)
            chunks = []
            
            for i in range(0, total_length, CHUNK_SIZE_MS):
                print(f"Processing chunk {i//CHUNK_SIZE_MS + 1}/{(total_length+CHUNK_SIZE_MS-1)//CHUNK_SIZE_MS}...")
                # Extract chunk
                end = min(i + CHUNK_SIZE_MS, total_length)
                chunk = input_audio[i:end]
                
                # Apply tempo change at chunk level
                if tempo != 1.0:
                    chunk = chunk._spawn(chunk.raw_data, overrides={"frame_rate": target_rate})
                    chunk = chunk.set_frame_rate(sample_rate)
                
                # Process this chunk
                processed_chunk = process_audio_chunk(chunk, mod_freq, depth, reverb, 
                                                    soften, pan_rate, noise_db, multiband)
                
                # Save processed chunk to temporary file
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                processed_chunk.export(chunk_path, format="wav")
                chunks.append(chunk_path)
                
                # Free memory
                del chunk, processed_chunk
                gc.collect()
            
            # Combine all processed chunks
            print("Combining processed chunks...")
            combined = AudioSegment.empty()
            for chunk_path in chunks:
                chunk = AudioSegment.from_file(chunk_path)
                combined += chunk
                del chunk
                gc.collect()
            
            # Export final combined audio
            combined.export(output_path, format="mp3")
            print(f"✅ Done: {output_path}")
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        sys.exit(1)

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
    parser.add_argument("--pan-rate", type=float, default=0.0,
                        help="Slow autopan rate in Hz (0 disables, e.g. 0.2)")
    parser.add_argument("--noise-db", type=float, default=None,
                        help="Level (dBFS) of pink noise bed (negative value, e.g. -32; omit to disable)")
    parser.add_argument("--multiband", action="store_true",
                        help="Enable multi‑band Brain.fm‑style modulation")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_MS,
                        help="Size of audio chunks to process in milliseconds (default: 10000)")
    
    args = parser.parse_args()
    
    # Update chunk size if specified
    if args.chunk_size > 0:
        CHUNK_SIZE_MS = args.chunk_size
    
    apply_brainfm_modulation(
        args.input_mp3, 
        args.output_mp3,
        mod_freq=args.mod_freq,
        depth=args.depth,
        tempo=args.tempo,
        reverb=args.reverb,
        soften=args.soften,
        pan_rate=args.pan_rate,
        noise_db=args.noise_db,
        multiband=args.multiband
    )