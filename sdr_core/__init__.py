"""Signal-processing core for software-defined radio.

Demodulation, spectrum analysis, modulation classification, protocol decoding
and recording — independent of any particular radio hardware or user interface.

    from sdr_core import AppConfig, ModulationType, SignalProcessor

    processor = SignalProcessor(AppConfig())
    audio = processor.demodulate(iq_samples, sample_rate, ModulationType.FM_NARROW)
"""

from .config import AppConfig
from .dsp import ModulationType, SignalProcessor

__all__ = ["AppConfig", "ModulationType", "SignalProcessor"]
__version__ = "0.1.0"
