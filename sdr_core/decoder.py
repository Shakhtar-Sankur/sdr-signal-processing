import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import threading
import importlib
from ._qt import QObject, pyqtSignal, pyqtSlot
import sys
import os
import json

logger = logging.getLogger(__name__)

class ProtocolDecoder(QObject):
    message_decoded = pyqtSignal(dict)
    decoder_status_changed = pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.active_decoder = None
        self.available_decoders = {}
        self.decoder_instances = {}
        self._load_available_decoders()

    def _load_available_decoders(self):
        decoders_path = os.path.join(os.path.dirname(__file__), '..', 'decoders')
        if not os.path.exists(decoders_path):
            logger.warning(f"Decoders directory not found: {decoders_path}")
            self._register_dummy_decoders()
            return
        if decoders_path not in sys.path:
            sys.path.append(str(decoders_path))
        metadata_files = []
        for root, _, files in os.walk(decoders_path):
            for file in files:
                if file == 'decoder.json':
                    metadata_files.append(os.path.join(root, file))
        if not metadata_files:
            logger.warning("No decoder metadata files found")
            self._register_dummy_decoders()
            return
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                rel_path = os.path.relpath(os.path.dirname(metadata_file), decoders_path)
                decoder_id = rel_path.replace(os.path.sep, '.')
                self.available_decoders[decoder_id] = {
                    'id': decoder_id,
                    'name': metadata.get('name', decoder_id),
                    'description': metadata.get('description', ''),
                    'supported_signals': metadata.get('supported_signals', []),
                    'parameters': metadata.get('parameters', {}),
                    'module_path': decoder_id,
                    'metadata_path': metadata_file
                }
                logger.debug(f"Registered decoder: {decoder_id}")
            except Exception as e:
                logger.error(f"Failed to load decoder metadata from {metadata_file}: {e}")
        if not self.available_decoders:
            self._register_dummy_decoders()
        self.decoder_status_changed.emit({
            'available_decoders': list(self.available_decoders.keys()),
            'active_decoder': self.active_decoder
        })

    def _register_dummy_decoders(self):
        self.available_decoders = {
            'weather.noaa_apt': {
                'id': 'weather.noaa_apt',
                'name': 'NOAA APT Weather Satellite',
                'description': 'Decoder for NOAA weather satellite APT transmissions',
                'supported_signals': ['NOAA_APT', 'FM_WIDE'],
                'parameters': {},
                'is_dummy': True
            },
            'aviation.ads_b': {
                'id': 'aviation.ads_b',
                'name': 'ADS-B Aircraft Transponder',
                'description': 'Decoder for ADS-B aircraft tracking transmissions',
                'supported_signals': ['ADS_B', 'PPM'],
                'parameters': {},
                'is_dummy': True
            },
            'telecom.dmr': {
                'id': 'telecom.dmr',
                'name': 'Digital Mobile Radio',
                'description': 'Decoder for DMR digital voice transmissions',
                'supported_signals': ['DMR', '4FSK'],
                'parameters': {},
                'is_dummy': True
            }
        }
        logger.warning("Using dummy decoders")

    @pyqtSlot(dict)
    def process_demodulated(self, demod_data):
        if not self.active_decoder or self.active_decoder not in self.available_decoders:
            return
        decoder_info = self.available_decoders[self.active_decoder]
        try:
            if self.active_decoder not in self.decoder_instances:
                self._create_decoder_instance(self.active_decoder)
            decoder = self.decoder_instances.get(self.active_decoder)
            if not decoder:
                return
            if decoder_info.get('is_dummy', False):
                messages = self._dummy_decode(demod_data, decoder_info)
            else:
                messages = decoder.process(demod_data['samples'],
                                          demod_data['sample_rate'],
                                          demod_data.get('center_freq'))
            if messages:
                result = {
                    'decoder': self.active_decoder,
                    'messages': messages,
                    'timestamp': demod_data.get('timestamp')
                }
                self.message_decoded.emit(result)
        except Exception as e:
            logger.error(f"Error in protocol decoder: {e}")

    def _create_decoder_instance(self, decoder_id: str):
        decoder_info = self.available_decoders.get(decoder_id)
        if not decoder_info:
            logger.error(f"Decoder not found: {decoder_id}")
            return None
        try:
            if decoder_info.get('is_dummy', False):
                self.decoder_instances[decoder_id] = DummyDecoder(decoder_id, decoder_info)
                return
            module_name = decoder_info['module_path']
            module = importlib.import_module(module_name)
            decoder_class = getattr(module, 'Decoder')
            decoder = decoder_class()
            self.decoder_instances[decoder_id] = decoder
            logger.info(f"Created decoder instance: {decoder_id}")
        except Exception as e:
            logger.error(f"Failed to create decoder instance for {decoder_id}: {e}")

    def set_active_decoder(self, decoder_id: str) -> bool:
        if decoder_id not in self.available_decoders:
            logger.error(f"Decoder not found: {decoder_id}")
            return False
        self.active_decoder = decoder_id
        if decoder_id not in self.decoder_instances:
            self._create_decoder_instance(decoder_id)
        self.decoder_status_changed.emit({
            'active_decoder': self.active_decoder
        })
        logger.info(f"Active decoder set to: {decoder_id}")
        return True

    def clear_active_decoder(self):
        self.active_decoder = None
        self.decoder_status_changed.emit({
            'active_decoder': None
        })
        logger.info("Active decoder cleared")

    def get_decoder_info(self, decoder_id: str) -> Optional[Dict[str, Any]]:
        return self.available_decoders.get(decoder_id)

    def get_recommended_decoder(self, signal_class: str) -> Optional[str]:
        for decoder_id, decoder_info in self.available_decoders.items():
            supported_signals = decoder_info.get('supported_signals', [])
            if signal_class in supported_signals:
                return decoder_id
        return None

    def _dummy_decode(self, demod_data: Dict[str, Any], decoder_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        import random
        import datetime
        num_messages = random.randint(0, 3)
        if num_messages == 0:
            return []
        messages = []
        decoder_id = decoder_info['id']
        for i in range(num_messages):
            if decoder_id == 'weather.noaa_apt':
                message = {
                    'satellite': random.choice(['NOAA-15', 'NOAA-18', 'NOAA-19']),
                    'frequency': demod_data.get('center_freq'),
                    'signal_quality': random.uniform(0.4, 0.95),
                    'timestamp': datetime.datetime.now().isoformat()
                }
            elif decoder_id == 'aviation.ads_b':
                message = {
                    'icao': hex(random.randint(0, 16777215))[2:].upper().zfill(6),
                    'callsign': ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6)),
                    'altitude': random.randint(1000, 40000),
                    'speed': random.randint(100, 600),
                    'heading': random.randint(0, 359),
                    'timestamp': datetime.datetime.now().isoformat()
                }
            elif decoder_id == 'telecom.dmr':
                message = {
                    'source_id': random.randint(1000, 9999),
                    'target_id': random.randint(1000, 9999),
                    'talker_alias': f"User-{random.randint(100, 999)}",
                    'color_code': random.randint(1, 15),
                    'timeslot': random.randint(1, 2),
                    'timestamp': datetime.datetime.now().isoformat()
                }
            else:
                message = {
                    'type': 'UNKNOWN',
                    'data': f"Message {i+1}",
                    'timestamp': datetime.datetime.now().isoformat()
                }
            messages.append(message)
        return messages

class DummyDecoder:
    def __init__(self, decoder_id: str, decoder_info: Dict[str, Any]):
        self.decoder_id = decoder_id
        self.decoder_info = decoder_info

    def process(self, samples: np.ndarray, sample_rate: float, center_freq: Optional[float] = None) -> List[Dict[str, Any]]:
        return []