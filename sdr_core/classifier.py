import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import threading
import time
from ._qt import QObject, pyqtSignal, pyqtSlot
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Machine learning features will be limited.")

logger = logging.getLogger(__name__)

#: Number of spectrum bins the classifier feeds the model. Fixed, because a model
#: is trained at one input width.
FEATURE_BINS = 2048


class DummyModel:
    """Stand-in used when no trained model is available.

    Defined at module level rather than inside `_create_dummy_model` so callers
    can recognise it with `isinstance`. It returns plain numpy, not tensors.
    """

    def __call__(self, inputs):
        inputs = np.asarray(inputs)
        batch_size = inputs.shape[0]
        num_classes = len(SignalClassifier.SIGNAL_CLASSES)
        logits = np.random.exponential(0.5, (batch_size, num_classes))
        logits = logits - np.min(logits, axis=1, keepdims=True)
        total = np.sum(logits, axis=1, keepdims=True)
        total[total == 0] = 1.0
        return {'logits': logits, 'probabilities': logits / total}

class SignalClassifier(QObject):
    SIGNAL_CLASSES = [
        'NOISE', 'AM', 'FM_NARROW', 'FM_WIDE', 'SSB_UPPER', 'SSB_LOWER',
        'CW', 'FSK', 'GFSK', 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64',
        'NOAA_APT', 'METEOR_M2', 'GSM', 'TETRA', 'DMR', 'ADS_B', 'ACARS',
        'AIS', 'NAVTEX', 'LORA', 'SIGFOX', 'BLE', 'ZIGBEE'
    ]

    classification_ready = pyqtSignal(list)
    model_status_changed = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.is_classifying = False
        self.last_classification_time = 0
        self.classification_interval = config.classification_interval
        self.classification_threshold = config.classification_threshold
        self.signal_classes = list(self.SIGNAL_CLASSES)
        self.load_model()

    def load_model(self) -> bool:
        if not TF_AVAILABLE:
            # Fall back to the dummy rather than leaving self.model as None.
            # With no model at all, classify_spectrum quietly returned without
            # emitting anything: no output, no error, nothing to debug.
            logger.warning("TensorFlow not available; using the dummy classifier")
            self._create_dummy_model()
            self.model_status_changed.emit(
                {'status': 'dummy', 'message': 'TensorFlow not available; predictions are random'})
            return False
        try:
            model_path = self.config.classifier_model_path
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                self.model_status_changed.emit({'status': 'error', 'message': 'Model file not found'})
                self._create_dummy_model()
                self.model_status_changed.emit({'status': 'dummy', 'message': 'Using dummy model'})
                return False
            self.model = tf.saved_model.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            self.model_status_changed.emit({'status': 'loaded', 'message': 'Model loaded successfully'})
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_status_changed.emit({'status': 'error', 'message': f'Failed to load model: {str(e)}'})
            self._create_dummy_model()
            self.model_status_changed.emit({'status': 'dummy', 'message': 'Using dummy model due to load error'})
            return False

    def _create_dummy_model(self):
        self.model = DummyModel()
        logger.warning("Created dummy classification model")

    @pyqtSlot(dict)
    def classify_spectrum(self, spectrum_data):
        current_time = time.time()
        if (current_time - self.last_classification_time) < self.classification_interval:
            return
        self.last_classification_time = current_time
        self.is_classifying = True
        try:
            spectrum = spectrum_data['spectrum']
            features = self._preprocess_spectrum(spectrum)
            if self.model is not None:
                # `isinstance(self.model, type(lambda: None).__class__)` reduced to
                # `isinstance(model, type)` - "is the model a class object?" - which
                # is false for any instance. The dummy therefore went down the
                # TensorFlow path, where `.numpy()` on its plain ndarray raised
                # AttributeError. Ask the question directly instead.
                if TF_AVAILABLE and not isinstance(self.model, DummyModel):
                    features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
                    predictions = self.model(features_tf)
                    probabilities = np.asarray(predictions['probabilities'])[0]
                else:
                    predictions = self.model(features)
                    probabilities = np.asarray(predictions['probabilities'])[0]
                classifications = self._process_classification_results(probabilities)
                self.classification_ready.emit(classifications)
        except Exception as e:
            logger.error(f"Classification error: {e}")
        finally:
            self.is_classifying = False

    def _preprocess_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        if len(spectrum.shape) == 1:
            features = spectrum.reshape(1, -1)
        else:
            features = spectrum
        features = (features - np.mean(features)) / (np.std(features) + 1e-10)
        # Always resample to one width. The previous test let 4096-bin spectra
        # through untouched while sending everything else to 2048, so the model
        # received two different input sizes depending on the FFT setting.
        if features.shape[1] != FEATURE_BINS:
            from scipy import signal
            features = signal.resample(features, FEATURE_BINS, axis=1)
        return features

    def _process_classification_results(self, probabilities: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            prob = probabilities[idx]
            if prob >= self.classification_threshold:
                if idx < len(self.signal_classes):
                    class_name = self.signal_classes[idx]
                else:
                    class_name = f"UNKNOWN_{idx}"
                results.append({
                    'class': class_name,
                    'probability': float(prob),
                    'index': int(idx)
                })
                protocol = self._map_class_to_protocol(class_name)
                if protocol:
                    results[-1]['protocol'] = protocol
                modulation = self._map_class_to_modulation(class_name)
                if modulation:
                    results[-1]['modulation'] = modulation
        return results

    def _map_class_to_protocol(self, class_name: str) -> Optional[str]:
        protocol_map = {
            'NOAA_APT': 'weather.noaa_apt',
            'METEOR_M2': 'weather.meteor_m2',
            'GSM': 'telecom.gsm',
            'TETRA': 'telecom.tetra',
            'DMR': 'telecom.dmr',
            'ADS_B': 'aviation.ads_b',
            'ACARS': 'aviation.acars',
            'AIS': 'maritime.ais',
            'NAVTEX': 'maritime.navtex',
            'LORA': 'iot.lora',
            'SIGFOX': 'iot.sigfox',
            'BLE': 'iot.ble',
            'ZIGBEE': 'iot.zigbee'
        }
        return protocol_map.get(class_name)

    def _map_class_to_modulation(self, class_name: str) -> Optional[Dict[str, Any]]:
        modulation_map = {
            'AM': {'type': 'AM', 'bandwidth': 10000},
            'FM_NARROW': {'type': 'FM_N', 'bandwidth': 12500},
            'FM_WIDE': {'type': 'FM_W', 'bandwidth': 200000},
            'SSB_UPPER': {'type': 'USB', 'bandwidth': 2700},
            'SSB_LOWER': {'type': 'LSB', 'bandwidth': 2700},
            'CW': {'type': 'CW', 'bandwidth': 500},
            'FSK': {'type': 'FSK', 'bandwidth': 15000},
            'GFSK': {'type': 'GFSK', 'bandwidth': 15000},
            'BPSK': {'type': 'BPSK', 'bandwidth': 10000},
            'QPSK': {'type': 'QPSK', 'bandwidth': 20000},
            '8PSK': {'type': '8PSK', 'bandwidth': 30000},
            'QAM16': {'type': 'QAM16', 'bandwidth': 30000},
            'QAM64': {'type': 'QAM64', 'bandwidth': 40000},
            'NOAA_APT': {'type': 'FM_W', 'bandwidth': 40000},
            'METEOR_M2': {'type': 'QPSK', 'bandwidth': 120000},
            'GSM': {'type': 'GMSK', 'bandwidth': 270000},
            'TETRA': {'type': 'PI/4-DQPSK', 'bandwidth': 25000},
            'DMR': {'type': '4FSK', 'bandwidth': 12500},
            'ADS_B': {'type': 'PPM', 'bandwidth': 1000000},
            'ACARS': {'type': 'AM', 'bandwidth': 10000},
            'AIS': {'type': 'GMSK', 'bandwidth': 14000},
            'NAVTEX': {'type': 'BFSK', 'bandwidth': 500},
            'LORA': {'type': 'CSS', 'bandwidth': 125000},
            'SIGFOX': {'type': 'BPSK', 'bandwidth': 100},
            'BLE': {'type': 'GFSK', 'bandwidth': 2000000},
            'ZIGBEE': {'type': 'OQPSK', 'bandwidth': 2000000}
        }
        return modulation_map.get(class_name)