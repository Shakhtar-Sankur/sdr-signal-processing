import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import threading
import time
import os
from datetime import datetime
from ._qt import QObject, pyqtSignal, pyqtSlot
import json

logger = logging.getLogger(__name__)

class SignalRecorder(QObject):
    recording_status_changed = pyqtSignal(dict)
    recording_completed = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_recording = False
        self.current_recording = None
        self.record_queue = []
        self.recording_thread = None
        os.makedirs(self.config.recording_dir, exist_ok=True)

    def start_recording(self, metadata: Dict[str, Any] = None) -> bool:
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        try:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            # Fall back to the configured tuning rather than 0, so recordings are
            # not all named 0.000MHz and indistinguishable from one another.
            center_freq = (metadata or {}).get('center_freq') or getattr(
                self.config, 'center_frequency', 0)
            freq_mhz = (center_freq or 0) / 1e6
            filename = f"recording_{timestamp_str}_{freq_mhz:.3f}MHz"
            recording_path = os.path.join(self.config.recording_dir, filename)
            os.makedirs(recording_path, exist_ok=True)
            self.current_recording = {
                'path': recording_path,
                'filename': filename,
                'format': self.config.recording_format,
                'timestamp': timestamp,
                'metadata': metadata or {},
                'files': [],
                'sample_count': 0,
                'duration': 0.0
            }
            metadata_path = os.path.join(recording_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp.isoformat(),
                    'format': self.config.recording_format,
                    'center_frequency': center_freq,
                    'sample_rate': metadata.get('sample_rate', 0) if metadata else 0,
                    'metadata': metadata or {}
                }, f, indent=2)
            self.is_recording = True
            if not self.recording_thread or not self.recording_thread.is_alive():
                self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
                self.recording_thread.start()
            self.recording_status_changed.emit({
                'status': 'recording',
                'path': recording_path,
                'filename': filename
            })
            logger.info(f"Started recording to {recording_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            self.current_recording = None
            return False

    def stop_recording(self) -> bool:
        if not self.is_recording:
            logger.warning("No recording in progress")
            return False
        try:
            self.is_recording = False
            timeout = 5.0
            start_time = time.time()
            while self.record_queue and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            if self.current_recording:
                summary_path = os.path.join(self.current_recording['path'], 'summary.json')
                with open(summary_path, 'w') as f:
                    json.dump({
                        'duration': self.current_recording['duration'],
                        'sample_count': self.current_recording['sample_count'],
                        'files': self.current_recording['files'],
                        'completed': datetime.now().isoformat()
                    }, f, indent=2)
                self.recording_completed.emit(self.current_recording)
            recording_info = self.current_recording
            self.current_recording = None
            self.recording_status_changed.emit({
                'status': 'stopped'
            })
            logger.info("Recording stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False

    @pyqtSlot(np.ndarray, float, float)
    def record_iq_samples(self, samples: np.ndarray, center_freq: float, sample_rate: float):
        if not self.is_recording or not self.current_recording:
            return
        self.record_queue.append({
            'type': 'iq',
            'samples': samples.copy(),
            'center_freq': center_freq,
            'sample_rate': sample_rate,
            'timestamp': datetime.now()
        })

    @pyqtSlot(dict)
    def record_demodulated(self, demod_data: Dict[str, Any]):
        if not self.is_recording or not self.current_recording:
            return
        self.record_queue.append({
            'type': 'demodulated',
            'data': demod_data,
            'timestamp': datetime.now()
        })

    @pyqtSlot(dict)
    def record_decoded_message(self, message_data: Dict[str, Any]):
        if not self.is_recording or not self.current_recording:
            return
        self.record_queue.append({
            'type': 'message',
            'data': message_data,
            'timestamp': datetime.now()
        })

    def _recording_worker(self):
        try:
            import h5py
            H5PY_AVAILABLE = True
        except ImportError:
            H5PY_AVAILABLE = False
            logger.warning("h5py not available, using NumPy format for recordings")
        while True:
            while self.record_queue:
                try:
                    item = self.record_queue.pop(0)
                    if not self.is_recording or not self.current_recording:
                        continue
                    if item['type'] == 'iq':
                        self._save_iq_samples(item)
                    elif item['type'] == 'demodulated':
                        self._save_demodulated(item)
                    elif item['type'] == 'message':
                        self._save_message(item)
                except Exception as e:
                    logger.error(f"Error in recording worker: {e}")
            time.sleep(0.05)
            if not self.is_recording and not self.record_queue:
                break

    def _save_iq_samples(self, item: Dict[str, Any]):
        try:
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
            if self.config.recording_format == 'hdf5':
                try:
                    import h5py
                    filename = f'iq_{timestamp}.h5'
                    filepath = os.path.join(self.current_recording['path'], filename)
                    with h5py.File(filepath, 'w') as f:
                        f.create_dataset('samples', data=item['samples'],
                                        compression=self.config.recording_compression)
                        f.attrs['center_frequency'] = item['center_freq']
                        f.attrs['sample_rate'] = item['sample_rate']
                        f.attrs['timestamp'] = str(item['timestamp'])
                        f.attrs['samples_count'] = len(item['samples'])
                except ImportError:
                    filename = f'iq_{timestamp}.npz'
                    filepath = os.path.join(self.current_recording['path'], filename)
                    np.savez_compressed(filepath,
                                      samples=item['samples'],
                                      center_freq=item['center_freq'],
                                      sample_rate=item['sample_rate'],
                                      timestamp=str(item['timestamp']))
            else:
                filename = f'iq_{timestamp}.npz'
                filepath = os.path.join(self.current_recording['path'], filename)
                np.savez_compressed(filepath,
                                  samples=item['samples'],
                                  center_freq=item['center_freq'],
                                  sample_rate=item['sample_rate'],
                                  timestamp=str(item['timestamp']))
            self.current_recording['files'].append(filename)
            self.current_recording['sample_count'] += len(item['samples'])
            if item['sample_rate'] > 0:
                self.current_recording['duration'] += len(item['samples']) / item['sample_rate']
        except Exception as e:
            logger.error(f"Failed to save IQ samples: {e}")

    def _save_demodulated(self, item: Dict[str, Any]):
        try:
            data = item['data']
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
            filename = f'demod_{timestamp}.npz'
            filepath = os.path.join(self.current_recording['path'], filename)
            np.savez_compressed(filepath,
                              samples=data['samples'],
                              sample_rate=data['sample_rate'],
                              center_freq=data.get('center_freq', 0),
                              modulation=str(data.get('modulation', 'UNKNOWN')),
                              timestamp=str(item['timestamp']))
            self.current_recording['files'].append(filename)
        except Exception as e:
            logger.error(f"Failed to save demodulated signal: {e}")

    def _save_message(self, item: Dict[str, Any]):
        try:
            data = item['data']
            timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
            filename = f'message_{timestamp}.json'
            filepath = os.path.join(self.current_recording['path'], filename)
            with open(filepath, 'w') as f:
                json.dump({
                    'decoder': data.get('decoder', 'UNKNOWN'),
                    'messages': data.get('messages', []),
                    'timestamp': str(item['timestamp'])
                }, f, indent=2)
            self.current_recording['files'].append(filename)
        except Exception as e:
            logger.error(f"Failed to save decoded message: {e}")