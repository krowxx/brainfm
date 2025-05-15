"""Command-line interface for BrainFM."""

import argparse
import sys

from brainfm.core import process_audio

def parse_args():
    """Parse command-line arguments."""
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
    
    return parser.parse_args()

def main():
    """Main entry point for the command-line interface."""
    args = parse_args()
    
    process_audio(
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

if __name__ == "__main__":
    main() 