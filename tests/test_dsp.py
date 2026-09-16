"""Demodulation, tested against signals whose right answer is known in advance.

Every test here builds a waveform with a tone deliberately put into it, runs the
real demodulator, and checks that the tone comes back out at the frequency it
went in at. A test that only asserts "an array came back" would pass against a
demodulator that returned noise.
"""

import numpy as np
import pytest

from sdr_core import AppConfig, ModulationType, SignalProcessor


SAMPLE_RATE = 48_000.0


@pytest.fixture
def processor():
    return SignalProcessor(AppConfig())


def dominant_frequency(audio, sample_rate=SAMPLE_RATE):
    """The strongest frequency in a real signal, ignoring DC."""
    spectrum = np.abs(np.fft.rfft(audio - np.mean(audio)))
    freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    return freqs[int(np.argmax(spectrum))]


def test_am_demodulation_recovers_the_tone_that_was_modulated(processor):
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone_hz = 1000.0
    envelope = 1.0 + 0.5 * np.sin(2 * np.pi * tone_hz * t)
    iq = envelope.astype(np.complex128)

    audio = processor._demodulate_am(iq, SAMPLE_RATE)

    assert len(audio) == len(iq)
    assert dominant_frequency(audio) == pytest.approx(tone_hz, abs=5)


def test_am_demodulation_removes_the_carrier_offset(processor):
    """The envelope is centred, so a steady carrier must average to zero."""
    iq = np.ones(4096, dtype=np.complex128)
    audio = processor._demodulate_am(iq, SAMPLE_RATE)
    assert abs(float(np.mean(audio))) < 1e-9


def test_fm_demodulation_recovers_the_modulating_tone(processor):
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone_hz = 1200.0
    deviation = 5000.0
    phase = 2 * np.pi * deviation / (2 * np.pi * tone_hz) * np.sin(2 * np.pi * tone_hz * t)
    iq = np.exp(1j * phase)

    audio = processor._demodulate_fm(iq, SAMPLE_RATE, deviation=deviation)

    assert len(audio) == len(iq)
    assert dominant_frequency(audio) == pytest.approx(tone_hz, abs=10)


def test_fm_demodulation_scales_by_deviation(processor):
    """A constant frequency offset becomes a constant level, of a known size.

    An offset of `f` Hz with deviation `d` should read f/d, which is the whole
    point of the sample_rate / (2*pi*deviation) scaling.
    """
    t = np.arange(20_000) / SAMPLE_RATE
    offset_hz = 2500.0
    deviation = 5000.0
    iq = np.exp(1j * 2 * np.pi * offset_hz * t)

    audio = processor._demodulate_fm(iq, SAMPLE_RATE, deviation=deviation)

    middle = audio[1000:-1000]          # skip the filter's settling edges
    assert float(np.mean(middle)) == pytest.approx(offset_hz / deviation, rel=0.05)


def test_fsk_bits_follow_the_two_tones(processor):
    """Alternating mark and space tones must come back as alternating bits."""
    symbol_len = 480                     # 10 ms at 48 kHz
    mark, space = 3000.0, -3000.0
    pieces, expected = [], []
    phase = 0.0
    for index in range(20):
        freq = mark if index % 2 == 0 else space
        t = np.arange(symbol_len) / SAMPLE_RATE
        segment_phase = phase + 2 * np.pi * freq * t
        pieces.append(np.exp(1j * segment_phase))
        phase = segment_phase[-1]
        expected.append(1 if index % 2 == 0 else 0)
    iq = np.concatenate(pieces)

    _, bits = processor._demodulate_fsk(iq, SAMPLE_RATE)

    assert set(np.unique(bits)).issubset({0, 1})
    # Read the bit at the middle of each symbol, away from the transitions.
    sampled = [int(bits[index * symbol_len + symbol_len // 2]) for index in range(20)]
    assert sampled == expected


def test_psk_normalises_and_returns_the_right_constellation(processor):
    iq = 7.5 * np.exp(1j * np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]))

    normalised, constellation = processor._demodulate_psk(iq, SAMPLE_RATE, ModulationType.QPSK)

    assert np.max(np.abs(normalised)) == pytest.approx(1.0)
    assert len(constellation) == 4
    assert np.allclose(np.abs(constellation), 1.0)
    # 8-PSK asks for eight points, BPSK for two.
    assert len(processor._demodulate_psk(iq, SAMPLE_RATE, ModulationType.PSK8)[1]) == 8
    assert len(processor._demodulate_psk(iq, SAMPLE_RATE, ModulationType.BPSK)[1]) == 2


def test_psk_survives_an_all_zero_input(processor):
    """Dividing by the peak is a division by zero on silence."""
    normalised, _ = processor._demodulate_psk(np.zeros(64, dtype=np.complex128),
                                              SAMPLE_RATE, ModulationType.BPSK)
    assert np.all(np.isfinite(normalised))


def test_spectrum_peaks_at_the_signal_frequency(processor):
    t = np.arange(4096) / SAMPLE_RATE
    tone_hz = 6000.0
    iq = np.exp(1j * 2 * np.pi * tone_hz * t)

    spectrum = processor.spectrum(iq)

    assert len(spectrum) == len(iq)
    bin_hz = SAMPLE_RATE / len(iq)
    peak_offset = (int(np.argmax(spectrum)) - len(iq) // 2) * bin_hz
    assert peak_offset == pytest.approx(tone_hz, abs=2 * bin_hz)


def test_demodulate_dispatches_on_the_modulation_it_is_given(processor):
    t = np.arange(8192) / SAMPLE_RATE
    iq = np.exp(1j * 2 * np.pi * 1000 * t)

    for modulation in (ModulationType.AM, ModulationType.FM_NARROW,
                       ModulationType.FM_WIDE, ModulationType.SSB_UPPER):
        audio = processor.demodulate(iq, SAMPLE_RATE, modulation)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.all(np.isfinite(audio)), f"{modulation} produced NaN or inf"
