# sdr-signal-processing

**Demodulation, spectrum analysis, modulation classification and protocol decoding for
software-defined radio — as a library, independent of any radio hardware or GUI.**

```python
from sdr_core import AppConfig, ModulationType, SignalProcessor

processor = SignalProcessor(AppConfig())
audio = processor.demodulate(iq_samples, sample_rate, ModulationType.FM_NARROW)
```

## What's in it

| Module | Contents |
|---|---|
| `sdr_core.dsp` | `SignalProcessor` — AM, narrow/wide FM, SSB (upper and lower), FSK and BPSK/QPSK/8PSK demodulation, plus a windowed averaged spectrum. `ModulationType` enumerates 14 schemes. |
| `sdr_core.classifier` | `SignalClassifier` — modulation recognition over the live spectrum. |
| `sdr_core.decoder` | `ProtocolDecoder` — pluggable decoders, resolved by name at runtime. |
| `sdr_core.recorder` | `SignalRecorder` — IQ capture to HDF5 with compression. |
| `sdr_core.config` | `AppConfig` — device, DSP and recording settings, loadable from JSON. |

Processing runs on a background thread and publishes results as Qt signals
(`spectrum_ready`, `audio_ready`, `demodulated_ready`), which is what a live radio front
end wants. The `demodulate`, `demodulate_bits` and `spectrum` methods expose the same
maths synchronously, for offline work and testing.

## Try it

```bash
pip install -r requirements.txt
python examples/demodulate_fm.py
```

The example synthesises a 1 kHz tone frequency-modulated onto a carrier, demodulates it,
and checks that the recovered audio peaks at the frequency it started from:

```
synthesised   60,000 IQ samples (0.25s @ 240 kHz), 1000 Hz tone
demodulated   60,000 samples
recovered     1000.0 Hz  (expected 1000 Hz, error 0.0 Hz)
spectrum      4096 bins, peak 55.0 dB, floor -97.8 dB

OK — FM demodulation round-trips within 1%
```

No SDR hardware required — the IQ is generated in the example.

## Implementation notes

**FM** differentiates unwrapped phase and scales by `sample_rate / (2π·deviation)`. Wide FM
additionally applies a 75 µs de-emphasis filter.

**AM** takes the envelope, removes the DC term, then low-passes at the configured
bandwidth. The DC removal matters — a carrier offset otherwise dominates the recovered
audio.

**SSB** band-passes and then takes the analytic signal via a Hilbert transform; lower
sideband is handled by conjugating the input rather than writing a second filter path.

**Spectrum** uses a Blackman–Harris window — −92 dB sidelobes, worth the wider main lobe
when hunting for weak signals beside strong ones — with exponential averaging across
frames.

## Scope

This repository is the signal-processing core. Radio hardware drivers and the desktop UI
are not part of it.

## Licence

All rights reserved. Published for reading, not for reuse.
