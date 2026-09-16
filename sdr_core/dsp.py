import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
from scipy import signal
from ._qt import QObject, pyqtSignal, pyqtSlot
import threading
import queue

logger = logging.getLogger(__name__)

class ModulationType(Enum):
    UNKNOWN = 0
    AM = 1
    FM_NARROW = 2
    FM_WIDE = 3
    SSB_UPPER = 4
    SSB_LOWER = 5
    CW = 6
    FSK = 7
    GFSK = 8
    BPSK = 9
    QPSK = 10
    PSK8 = 11
    QAM16 = 12
    QAM64 = 13

class SignalProcessor(QObject):
    spectrum_ready = pyqtSignal(np.ndarray)
    audio_ready = pyqtSignal(np.ndarray, int)
    demodulated_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_processing = False
        self.fft_size = config.fft_size
        self.window = signal.windows.blackmanharris(self.fft_size)
        self.spectrum_averaging = 0.5
        self.last_spectrum = None
        self.current_modulation = ModulationType.FM_NARROW
        self.current_filter_bw = 12500
        self.current_decimation = 1
        self.processing_queue = queue.Queue(maxsize=10)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    @pyqtSlot(np.ndarray, float, float)
    def process_samples(self, samples: np.ndarray, center_freq: float, sample_rate: float):
        try:
            self.processing_queue.put_nowait({
                'samples': samples.copy(),
                'center_freq': center_freq,
                'sample_rate': sample_rate,
                'timestamp': np.datetime64('now')
            })
        except queue.Full:
            logger.warning("Processing queue full, dropping samples")

    def _processing_loop(self):
        self.is_processing = True
        while self.is_processing:
            try:
                data = self.processing_queue.get(timeout=0.5)
                self._process_spectrum(data['samples'], data['sample_rate'])
                if self.current_modulation != ModulationType.UNKNOWN:
                    self._demodulate_signal(data)
                self.processing_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")

    def _process_spectrum(self, samples: np.ndarray, sample_rate: float):
        if len(samples) < self.fft_size:
            samples = np.pad(samples, (0, self.fft_size - len(samples)))
        if len(samples) > self.fft_size:
            start_idx = (len(samples) - self.fft_size) // 2
            samples = samples[start_idx:start_idx + self.fft_size]
        windowed = samples * self.window
        fft_result = np.fft.fftshift(np.fft.fft(windowed))
        spectrum = 20 * np.log10(np.abs(fft_result) + 1e-10)
        if self.last_spectrum is not None:
            spectrum = self.spectrum_averaging * spectrum + (1 - self.spectrum_averaging) * self.last_spectrum
        self.last_spectrum = spectrum.copy()
        frequencies = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/sample_rate))
        spectrum_data = {
            'spectrum': spectrum,
            'frequencies': frequencies,
            'sample_rate': sample_rate
        }
        self.spectrum_ready.emit(spectrum_data)

    def _demodulate_signal(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        center_freq = data['center_freq']
        if self.current_modulation == ModulationType.AM:
            demodulated = self._demodulate_am(samples, sample_rate)
        elif self.current_modulation == ModulationType.FM_NARROW:
            demodulated = self._demodulate_fm(samples, sample_rate, deviation=5000)
        elif self.current_modulation == ModulationType.FM_WIDE:
            demodulated = self._demodulate_fm(samples, sample_rate, deviation=75000)
        elif self.current_modulation == ModulationType.SSB_UPPER:
            demodulated = self._demodulate_ssb(samples, sample_rate, upper=True)
        elif self.current_modulation == ModulationType.SSB_LOWER:
            demodulated = self._demodulate_ssb(samples, sample_rate, upper=False)
        elif self.current_modulation == ModulationType.FSK:
            demodulated, bits = self._demodulate_fsk(samples, sample_rate)
        elif self.current_modulation in [ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8]:
            demodulated, symbols = self._demodulate_psk(samples, sample_rate, self.current_modulation)
        else:
            demodulated = samples
        demod_data = {
            'samples': demodulated,
            'sample_rate': sample_rate,
            'center_freq': center_freq,
            'modulation': self.current_modulation,
            'timestamp': data['timestamp']
        }
        self.demodulated_ready.emit(demod_data)
        if self.current_modulation in [ModulationType.AM, ModulationType.FM_NARROW,
                                     ModulationType.FM_WIDE, ModulationType.SSB_UPPER,
                                     ModulationType.SSB_LOWER]:
            if np.iscomplexobj(demodulated):
                audio = demodulated.real
            else:
                audio = demodulated
            if len(audio) > 0:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val
            self.audio_ready.emit(audio, int(sample_rate))

    def _demodulate_am(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        envelope = np.abs(samples)
        envelope = envelope - np.mean(envelope)
        if self.current_filter_bw < sample_rate / 2:
            nyquist = sample_rate / 2
            cutoff = self.current_filter_bw / nyquist
            b, a = signal.butter(5, cutoff, btype='low')
            envelope = signal.filtfilt(b, a, envelope)
        return envelope

    def _demodulate_fm(self, samples: np.ndarray, sample_rate: float, deviation: float = 5000) -> np.ndarray:
        phase = np.angle(samples)
        unwrapped_phase = np.unwrap(phase)
        demodulated = np.diff(unwrapped_phase)
        demodulated = np.append(demodulated, demodulated[-1])
        demodulated = demodulated * (sample_rate / (2 * np.pi * deviation))
        if self.current_modulation == ModulationType.FM_WIDE:
            tau = 75e-6
            alpha = 1.0 / (1.0 + tau * sample_rate)
            demodulated = signal.lfilter([alpha], [1, -(1-alpha)], demodulated)
        if self.current_filter_bw < sample_rate / 2:
            nyquist = sample_rate / 2
            cutoff = self.current_filter_bw / nyquist
            b, a = signal.butter(5, cutoff, btype='low')
            demodulated = signal.filtfilt(b, a, demodulated)
        return demodulated

    def _demodulate_ssb(self, samples: np.ndarray, sample_rate: float, upper: bool = True) -> np.ndarray:
        nyquist = sample_rate / 2
        if upper:
            cutoff = [100 / nyquist, self.current_filter_bw / nyquist]
        else:
            cutoff = [100 / nyquist, self.current_filter_bw / nyquist]
            samples = np.conjugate(samples)
        b, a = signal.butter(5, cutoff, btype='bandpass')
        filtered = signal.filtfilt(b, a, samples)
        analytic = signal.hilbert(filtered.real)
        demodulated = analytic.real
        return demodulated

    def _demodulate_fsk(self, samples: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        demodulated = self._demodulate_fm(samples, sample_rate, deviation=self.current_filter_bw/2)
        threshold = np.mean(demodulated)
        bits = (demodulated > threshold).astype(np.int8)
        return demodulated, bits

    def _demodulate_psk(self, samples: np.ndarray, sample_rate: float, mode: ModulationType) -> Tuple[np.ndarray, np.ndarray]:
        if mode == ModulationType.BPSK:
            M = 2
        elif mode == ModulationType.QPSK:
            M = 4
        elif mode == ModulationType.PSK8:
            M = 8
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            normalized = samples / max_val
        else:
            normalized = samples
        constellation = np.exp(1j * 2 * np.pi * np.arange(M) / M)
        return normalized, constellation

    def configure_for_modulation(self, modulation_info: Dict[str, Any]):
        if 'type' in modulation_info:
            mod_str = modulation_info['type'].upper()
            if mod_str == 'AM':
                self.current_modulation = ModulationType.AM
                self.current_filter_bw = 10000
            elif mod_str == 'FM_N' or mod_str == 'NFM':
                self.current_modulation = ModulationType.FM_NARROW
                self.current_filter_bw = 12500
            elif mod_str == 'FM_W' or mod_str == 'WFM':
                self.current_modulation = ModulationType.FM_WIDE
                self.current_filter_bw = 200000
            elif mod_str == 'USB':
                self.current_modulation = ModulationType.SSB_UPPER
                self.current_filter_bw = 2700
            elif mod_str == 'LSB':
                self.current_modulation = ModulationType.SSB_LOWER
                self.current_filter_bw = 2700
            elif mod_str == 'FSK':
                self.current_modulation = ModulationType.FSK
                self.current_filter_bw = 15000
            elif mod_str == 'GFSK':
                self.current_modulation = ModulationType.GFSK
                self.current_filter_bw = 15000
            elif mod_str == 'BPSK':
                self.current_modulation = ModulationType.BPSK
                self.current_filter_bw = 10000
            elif mod_str == 'QPSK':
                self.current_modulation = ModulationType.QPSK
                self.current_filter_bw = 20000
            elif mod_str == '8PSK':
                self.current_modulation = ModulationType.PSK8
                self.current_filter_bw = 30000
            else:
                logger.warning(f"Unknown modulation type: {mod_str}")
                self.current_modulation = ModulationType.UNKNOWN
        if 'bandwidth' in modulation_info:
            self.current_filter_bw = modulation_info['bandwidth']
        if 'decimation' in modulation_info:
            self.current_decimation = modulation_info['decimation']
        logger.info(f"Configured for {self.current_modulation.name} with BW {self.current_filter_bw/1000:.1f} kHz")
        self.status_changed.emit({
            'modulation': self.current_modulation.name,
            'filter_bw': self.current_filter_bw,
            'decimation': self.current_decimation
        })
    # ------------------------------------------------------------------
    # Public, synchronous API.
    #
    # The queue/thread path above is how a live radio feeds this class: samples
    # arrive, results leave as Qt signals. That is awkward to call directly, so
    # these two methods expose the same maths as ordinary function calls.
    # ------------------------------------------------------------------

    def demodulate(self, samples: np.ndarray, sample_rate: float,
                   modulation: Optional[ModulationType] = None) -> np.ndarray:
        """Demodulate a block of complex IQ samples and return the baseband result.

        `modulation` defaults to whatever the processor is currently configured
        for. FSK returns the demodulated waveform; use `demodulate_bits` for the
        recovered bitstream.
        """
        mode = modulation if modulation is not None else self.current_modulation

        if mode == ModulationType.AM:
            return self._demodulate_am(samples, sample_rate)
        if mode == ModulationType.FM_NARROW:
            return self._demodulate_fm(samples, sample_rate, deviation=5000)
        if mode == ModulationType.FM_WIDE:
            return self._demodulate_fm(samples, sample_rate, deviation=75000)
        if mode == ModulationType.SSB_UPPER:
            return self._demodulate_ssb(samples, sample_rate, upper=True)
        if mode == ModulationType.SSB_LOWER:
            return self._demodulate_ssb(samples, sample_rate, upper=False)
        if mode in (ModulationType.FSK, ModulationType.GFSK):
            waveform, _bits = self._demodulate_fsk(samples, sample_rate)
            return waveform
        if mode in (ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8):
            symbols, _constellation = self._demodulate_psk(samples, sample_rate, mode)
            return symbols

        logger.warning("No demodulator for %s; returning samples unchanged", mode.name)
        return samples

    def demodulate_bits(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """Recover a hard-decision bitstream from an FSK-modulated block."""
        _waveform, bits = self._demodulate_fsk(samples, sample_rate)
        return bits

    def spectrum(self, samples: np.ndarray) -> np.ndarray:
        """Averaged power spectrum in dB, using the configured FFT size and window."""
        block = samples[: self.fft_size]
        if len(block) < self.fft_size:
            block = np.pad(block, (0, self.fft_size - len(block)))
        windowed = block * self.window
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

        if self.last_spectrum is not None and len(self.last_spectrum) == len(power_db):
            a = self.spectrum_averaging
            power_db = a * self.last_spectrum + (1 - a) * power_db
        self.last_spectrum = power_db
        return power_db
