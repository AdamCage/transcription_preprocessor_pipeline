import numpy as np
import pytest

from audio_asr_pipeline.errors import AudioLoadError
from audio_asr_pipeline.io import split_stereo_channels


def test_split_samples_channels() -> None:
    y = np.zeros((100, 2), dtype=np.float32)
    y[:, 0] = 1.0
    c0, c1, n = split_stereo_channels(y)
    assert n == 100
    assert c0.shape == (100,) and np.allclose(c0, 1.0)
    assert np.allclose(c1, 0.0)


def test_split_channels_samples() -> None:
    y = np.zeros((2, 50), dtype=np.float32)
    y[1, :] = -1.0
    c0, c1, n = split_stereo_channels(y)
    assert n == 50
    assert np.allclose(c0, 0.0) and np.allclose(c1, -1.0)


def test_split_rejects_wrong_shape() -> None:
    with pytest.raises(AudioLoadError):
        split_stereo_channels(np.zeros((10, 3)))
    with pytest.raises(AudioLoadError):
        split_stereo_channels(np.zeros((3, 10)))
