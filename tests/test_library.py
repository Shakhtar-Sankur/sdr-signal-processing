"""Config, classification results and the decoder registry.

None of this needs TensorFlow or a radio: the classifier's preprocessing and
result handling are ordinary array work, and the decoders under test are the
built-in ones.
"""

import numpy as np
import pytest

from sdr_core import AppConfig
from sdr_core.classifier import SignalClassifier
from sdr_core.decoder import ProtocolDecoder


# ── config ────────────────────────────────────────────────────────────────
def test_defaults_are_a_working_configuration():
    config = AppConfig()
    assert config.sample_rate > 0
    assert config.fft_size > 0 and config.fft_size & (config.fft_size - 1) == 0, "FFT size should be a power of two"
    assert 0 < config.classification_threshold <= 1
    assert config.enabled_decoders, "no decoders enabled by default"


def test_config_round_trips_through_a_file(tmp_path):
    config = AppConfig()
    config.set("center_frequency", 144.8e6)
    config.set("theme", "light")
    path = tmp_path / "sdr.json"

    assert config.save_to_file(path) is True
    assert path.exists()

    reloaded = AppConfig()
    assert reloaded.load_from_file(path) is True
    assert reloaded.get("center_frequency") == 144.8e6
    assert reloaded.get("theme") == "light"


def test_loading_a_missing_file_fails_without_raising(tmp_path):
    assert AppConfig().load_from_file(tmp_path / "nope.json") is False


def test_get_returns_the_default_for_an_unknown_key():
    assert AppConfig().get("no_such_setting", "fallback") == "fallback"


# ── classifier ────────────────────────────────────────────────────────────
@pytest.fixture
def classifier():
    return SignalClassifier(AppConfig())


def test_preprocessing_gives_one_width_whatever_the_fft_size(classifier):
    """Two different FFT sizes must not reach the model as two input shapes."""
    widths = {classifier._preprocess_spectrum(np.random.rand(size)).shape[1]
              for size in (512, 1024, 2048, 4096)}
    assert len(widths) == 1, f"model would see {widths}"


def test_preprocessing_standardises_the_signal(classifier):
    features = classifier._preprocess_spectrum(np.random.rand(2048) * 50 + 20)
    assert abs(float(np.mean(features))) < 1e-6
    assert float(np.std(features)) == pytest.approx(1.0, rel=1e-3)


def test_a_flat_spectrum_does_not_divide_by_zero(classifier):
    features = classifier._preprocess_spectrum(np.ones(2048))
    assert np.all(np.isfinite(features))


def test_results_are_ordered_and_thresholded(classifier):
    classifier.classification_threshold = 0.3
    probabilities = np.zeros(len(classifier.signal_classes))
    probabilities[2], probabilities[5], probabilities[7] = 0.8, 0.5, 0.1

    results = classifier._process_classification_results(probabilities)

    assert [r["index"] for r in results] == [2, 5], "below-threshold class should be dropped"
    assert results[0]["probability"] > results[1]["probability"], "not sorted by confidence"
    assert all("class" in r for r in results)


def test_a_known_class_carries_its_protocol_and_modulation(classifier):
    classifier.classification_threshold = 0.1
    for name in ("ADS_B", "AIS", "LORA"):
        if name not in classifier.signal_classes:
            continue
        probabilities = np.zeros(len(classifier.signal_classes))
        probabilities[classifier.signal_classes.index(name)] = 0.9
        result = classifier._process_classification_results(probabilities)[0]
        assert result["class"] == name
        assert "protocol" in result, f"{name} has no protocol mapping"


# ── decoders ──────────────────────────────────────────────────────────────
@pytest.fixture
def decoder():
    return ProtocolDecoder(AppConfig())


def test_the_enabled_decoders_are_registered(decoder):
    info = [decoder.get_decoder_info(d) for d in AppConfig().enabled_decoders]
    assert all(i is not None for i in info), "a decoder listed in the config is not registered"


def test_selecting_and_clearing_a_decoder(decoder):
    first = AppConfig().enabled_decoders[0]
    assert decoder.set_active_decoder(first) is True
    assert decoder.active_decoder is not None
    decoder.clear_active_decoder()
    assert decoder.active_decoder is None


def test_an_unknown_decoder_is_refused(decoder):
    assert decoder.set_active_decoder("not.a.decoder") is False


def test_a_signal_class_recommends_its_decoder(decoder):
    assert decoder.get_recommended_decoder("ADS_B") == "aviation.ads_b"
    assert decoder.get_recommended_decoder("NOT_A_SIGNAL") is None
