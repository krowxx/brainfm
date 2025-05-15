"""Audio effects modules for BrainFM."""

from brainfm.effects.modulation import apply_modulation, multiband_modulate
from brainfm.effects.reverb import apply_reverb
from brainfm.effects.filters import band_pass_filter, soften_transients
from brainfm.effects.panning import apply_autopan
from brainfm.effects.noise import add_pink_noise 