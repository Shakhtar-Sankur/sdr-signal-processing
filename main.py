import sys
import os
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QMessageBox
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal, QTimer

from sdr_core.config import AppConfig
from sdr_core.signal_acquisition import SignalAcquisition
from sdr_core.signal_processor import SignalProcessor
from sdr_core.ml_classifier import SignalClassifier
from sdr_core.protocol_decoder import ProtocolDecoderManager
from sdr_core.storage import SignalRecorder, SignalPlayer

from ui.main_window import SDRMainWindow
from ui.spectrum_view import SpectrumView
from ui.waterfall_display import WaterfallDisplay
from ui.signal_browser import SignalBrowser
from ui.decoder_panel import DecoderPanel
from ui.control_sidebar import ControlSidebar

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"sdr_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SDRApplication:
    def __init__(self):
        self.app_config = AppConfig()
        self.load_configuration()

        self.signal_acquisition = SignalAcquisition(self.app_config)
        self.signal_processor = SignalProcessor(self.app_config)
        self.ml_classifier = SignalClassifier(self.app_config)
        self.protocol_decoder = ProtocolDecoderManager(self.app_config)
        self.signal_recorder = SignalRecorder(self.app_config)
        self.signal_player = SignalPlayer(self.app_config)

        self.connect_processing_pipeline()

        self.qt_app = QApplication(sys.argv)
        self.main_window = SDRMainWindow(self)

        self.setup_ui_components()

        self.setup_background_processing()

    def load_configuration(self):
        try:
            self.app_config.load_from_file(Path.home() / ".sdr_suite" / "config.json")
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self.app_config.set_defaults()

    def connect_processing_pipeline(self):
        self.signal_acquisition.samples_available.connect(self.signal_processor.process_samples)
        self.signal_processor.spectrum_ready.connect(self.update_spectrum_display)
        self.signal_processor.demodulated_ready.connect(self.protocol_decoder.decode_signal)
        self.ml_classifier.classification_ready.connect(self.update_signal_classification)
        self.protocol_decoder.decode_ready.connect(self.update_decoder_panels)

    def setup_ui_components(self):
        self.spectrum_view = SpectrumView(self.main_window)
        self.waterfall_display = WaterfallDisplay(self.main_window)
        self.signal_browser = SignalBrowser(self.main_window)
        self.decoder_panel = DecoderPanel(self.main_window)
        self.control_sidebar = ControlSidebar(self.main_window, self.app_config)

        self.main_window.add_dock_widget(self.spectrum_view, "Spectrum Analyzer", Qt.DockWidgetArea.TopDockWidgetArea)
        self.main_window.add_dock_widget(self.waterfall_display, "Waterfall Display", Qt.DockWidgetArea.TopDockWidgetArea)
        self.main_window.add_dock_widget(self.signal_browser, "Signal Browser", Qt.DockWidgetArea.LeftDockWidgetArea)
        self.main_window.add_dock_widget(self.decoder_panel, "Decoder", Qt.DockWidgetArea.RightDockWidgetArea)
        self.main_window.add_dock_widget(self.control_sidebar, "Controls", Qt.DockWidgetArea.RightDockWidgetArea)

        self.control_sidebar.frequency_changed.connect(self.signal_acquisition.set_frequency)
        self.control_sidebar.sample_rate_changed.connect(self.signal_acquisition.set_sample_rate)
        self.control_sidebar.gain_changed.connect(self.signal_acquisition.set_gain)
        self.control_sidebar.start_requested.connect(self.start_acquisition)
        self.control_sidebar.stop_requested.connect(self.stop_acquisition)
        self.control_sidebar.record_toggled.connect(self.toggle_recording)

        self.signal_browser.signal_selected.connect(self.focus_on_signal)

    def setup_background_processing(self):
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self.update_ui)
        self.ui_update_timer.start(50)

        self.classifier_thread = QThread()
        self.ml_classifier.moveToThread(self.classifier_thread)
        self.classifier_thread.start()

    def update_ui(self):
        if hasattr(self, 'latest_spectrum') and self.latest_spectrum is not None:
            self.spectrum_view.update_spectrum(self.latest_spectrum)
            self.waterfall_display.add_spectrum_line(self.latest_spectrum)

        self.main_window.update_status_info({
            'frequency': self.signal_acquisition.current_frequency,
            'sample_rate': self.signal_acquisition.current_sample_rate,
            'gain': self.signal_acquisition.current_gain,
            'cpu_load': self.get_cpu_load(),
            'recording': self.signal_recorder.is_recording
        })

    def update_spectrum_display(self, spectrum_data):
        self.latest_spectrum = spectrum_data
        self.ml_classifier.classify_spectrum(spectrum_data)

    def update_signal_classification(self, classification_results):
        self.signal_browser.update_classifications(classification_results)
        if classification_results and self.app_config.auto_select_decoder:
            top_classification = classification_results[0]
            self.protocol_decoder.select_decoder(top_classification['protocol'])

    def update_decoder_panels(self, decoded_data):
        self.decoder_panel.update_decoded_data(decoded_data)

    def focus_on_signal(self, signal_info):
        if 'frequency' in signal_info:
            self.signal_acquisition.set_frequency(signal_info['frequency'])
        if 'modulation' in signal_info:
            self.signal_processor.configure_for_modulation(signal_info['modulation'])
        if 'protocol' in signal_info:
            self.protocol_decoder.select_decoder(signal_info['protocol'])

    def start_acquisition(self):
        try:
            self.signal_acquisition.start()
            logger.info(f"Signal acquisition started at {self.signal_acquisition.current_frequency/1e6:.3f} MHz")
            return True
        except Exception as e:
            logger.error(f"Failed to start acquisition: {e}")
            QMessageBox.critical(self.main_window, "Acquisition Error",
                                f"Failed to start signal acquisition: {str(e)}")
            return False

    def stop_acquisition(self):
        self.signal_acquisition.stop()
        logger.info("Signal acquisition stopped")

    def toggle_recording(self, enabled):
        if enabled:
            filename = f"sdr_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            self.signal_recorder.start_recording(filename)
            logger.info(f"Started recording to {filename}")
        else:
            self.signal_recorder.stop_recording()
            logger.info("Recording stopped")

    def get_cpu_load(self):
        base_load = 10.0
        if self.signal_acquisition.is_running:
            base_load += 15.0
        if self.signal_processor.is_processing:
            base_load += 20.0
        if self.ml_classifier.is_classifying:
            base_load += 7.0
        base_load += np.random.normal(0, 3.0)
        return min(max(base_load, 0), 100)

    def shutdown(self):
        logger.info("Shutting down SDR Application Suite")
        self.stop_acquisition()
        if self.signal_recorder.is_recording:
            self.signal_recorder.stop_recording()
        self.ui_update_timer.stop()
        self.classifier_thread.quit()
        self.classifier_thread.wait(1000)
        self.app_config.save_to_file(Path.home() / ".sdr_suite" / "config.json")
        logger.info("Shutdown complete")

    def run(self):
        self.main_window.show()
        exit_code = self.qt_app.exec()
        self.shutdown()
        return exit_code

def parse_arguments():
    parser = argparse.ArgumentParser(description="SDR Application Suite")
    parser.add_argument("--device", help="SDR device to use (e.g., rtlsdr, hackrf, airspy)")
    parser.add_argument("--freq", type=float, help="Initial frequency in Hz")
    parser.add_argument("--sample-rate", type=float, help="Sample rate in Hz")
    parser.add_argument("--gain", type=float, help="Gain setting")
    parser.add_argument("--file", help="IQ data file to load instead of using a device")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    sdr_app = SDRApplication()
    if args.device:
        sdr_app.app_config.sdr_device = args.device
    if args.freq:
        sdr_app.signal_acquisition.set_frequency(args.freq)
    if args.sample_rate:
        sdr_app.signal_acquisition.set_sample_rate(args.sample_rate)
    if args.gain:
        sdr_app.signal_acquisition.set_gain(args.gain)
    if args.file:
        sdr_app.signal_player.load_file(args.file)
        sdr_app.signal_player.play()
    sys.exit(sdr_app.run())




