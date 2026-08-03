"""Round-trip check: modulate a tone onto an FM carrier, then recover it.

No radio hardware required — the IQ is synthesised here. If the demodulator is
correct, the recovered audio peaks at the same frequency we started with.

    python examples/demodulate_fm.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdr_core import AppConfig, ModulationType, SignalProcessor


def synthesise_fm(tone_hz: float, sample_rate: float, duration_s: float,
                  deviation_hz: float = 5000.0) -> np.ndarray:
    """Generate complex IQ for a single tone frequency-modulated onto a carrier."""
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    message = np.sin(2 * np.pi * tone_hz * t)
    # FM: instantaneous phase is the integral of the message
    phase = 2 * np.pi * deviation_hz * np.cumsum(message) / sample_rate
    return np.exp(1j * phase).astype(np.complex128)


def dominant_frequency(x: np.ndarray, sample_rate: float) -> float:
    """Frequency of the strongest spectral component, ignoring DC."""
    spectrum = np.abs(np.fft.rfft(x - np.mean(x)))
    freqs = np.fft.rfftfreq(len(x), d=1 / sample_rate)
    spectrum[0] = 0.0
    return float(freqs[int(np.argmax(spectrum))])


def main() -> int:
    sample_rate = 240_000.0
    tone_hz = 1_000.0
    duration_s = 0.25

    config = AppConfig()
    config.fft_size = 4096
    processor = SignalProcessor(config)
    processor.current_filter_bw = 15_000

    iq = synthesise_fm(tone_hz, sample_rate, duration_s)
    print(f"synthesised   {len(iq):,} IQ samples "
          f"({duration_s}s @ {sample_rate/1e3:.0f} kHz), {tone_hz:.0f} Hz tone")

    audio = processor.demodulate(iq, sample_rate, ModulationType.FM_NARROW)
    recovered = dominant_frequency(audio, sample_rate)
    error_hz = abs(recovered - tone_hz)

    print(f"demodulated   {len(audio):,} samples")
    print(f"recovered     {recovered:.1f} Hz  (expected {tone_hz:.0f} Hz, "
          f"error {error_hz:.1f} Hz)")

    spectrum = processor.spectrum(iq)
    print(f"spectrum      {len(spectrum)} bins, "
          f"peak {spectrum.max():.1f} dB, floor {spectrum.min():.1f} dB")

    # 1% of the tone frequency is a generous bound for a 0.25 s block
    if error_hz > tone_hz * 0.01:
        print(f"\nFAIL — recovered tone is off by {error_hz:.1f} Hz")
        return 1

    print("\nOK — FM demodulation round-trips within 1%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
