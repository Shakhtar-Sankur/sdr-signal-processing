import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AppConfig:
    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        self.sdr_device = "rtlsdr"
        self.device_serial = None
        self.device_index = 0
        self.center_frequency = 100e6
        self.sample_rate = 2.4e6
        self.gain = 'auto'
        self.ppm_error = 0
        self.bandwidth = None
        self.fft_size = 4096
        self.decimation = 1
        self.frame_rate = 30
        self.theme = "dark"
        self.waterfall_history = 600
        self.spectrum_min_db = -120
        self.spectrum_max_db = -20
        self.auto_select_decoder = True
        self.recording_dir = str(Path.home() / "SDR_Recordings")
        self.recording_format = "hdf5"
        self.recording_compression = "lzf"
        self.enabled_decoders = [
            "weather.noaa_apt", "weather.meteor_m2",
            "telecom.gsm", "telecom.tetra", "telecom.dmr",
            "aviation.ads_b", "aviation.acars",
            "maritime.ais", "maritime.navtex",
            "iot.lora", "iot.sigfox", "iot.ble", "iot.zigbee",
            "satellite.noaa", "satellite.inmarsat", "satellite.iridium"
        ]
        self.classifier_model_path = "models/signal_classifier_v2.3.pb"
        self.classification_threshold = 0.65
        self.classification_interval = 1.0

    def load_from_file(self, config_path: Path) -> bool:
        try:
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return False
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            logger.info(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def save_to_file(self, config_path: Path) -> bool:
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_data = {key: value for key, value in self.__dict__.items()
                          if not key.startswith('_') and not callable(value)}
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> bool:
        if hasattr(self, key):
            setattr(self, key, value)
            return True
        else:
            logger.warning(f"Attempted to set unknown configuration key: {key}")
            return False

    def as_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_') and not callable(value)}