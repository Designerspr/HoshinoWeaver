"""RGB padding exclusion is independent of alignment and frame-buffer metadata."""
import numpy as np
import pytest

from hoshicore._custom_op import _dispatch
from hoshicore._custom_op.ops import fgp
from hoshicore.component.data_container import HuberMeanParam


@pytest.fixture(params=['numpy', 'cpu', 'auto'])
def backend(request, monkeypatch):
    monkeypatch.setattr(_dispatch, '_backend_preference_override', request.param)


def _huber_inputs(dtype, channels):
    stack = np.zeros((4, 1, 3, channels), dtype=dtype)
    for index, value in enumerate((90, 100, 110), start=1):
        stack[index, 0, 0, :3] = value
        stack[index, 0, 1, :3] = [0, value, value]
    if channels == 4:
        stack[..., 3] = 100
    mean = np.full(stack.shape[1:], 100, dtype=np.float32)
    std = np.full_like(mean, 10)
    return stack, mean, std


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('channels', [3, 4])
@pytest.mark.parametrize('array_weight', [False, True])
def test_huber_accumulation_excludes_padding_from_both_sums(
        backend, dtype, channels, array_weight):
    stack, mean, std = _huber_inputs(dtype, channels)
    result = HuberMeanParam(np.zeros_like(mean, np.float64),
                            np.zeros_like(mean, np.float64), source_dtype=stack.dtype)
    for frame in stack:
        weight = np.ones_like(mean) if array_weight else 1.0
        fgp.huber_weighted_accumulate(result, frame, mean, std, 1.345,
                                      weight, skip_zero_rgb=True)
    np.testing.assert_allclose(result.weighted_sum[0, 0, :3], 300)
    np.testing.assert_allclose(result.weight_total[0, 0, :3], 3)
    # A zero in just one color channel is retained, including its weight.
    assert result.weighted_sum[0, 1, 0] == 0
    assert result.weight_total[0, 1, 0] > 0
    np.testing.assert_allclose(result.weight_total[0, 1, 1:3], 3)
    np.testing.assert_array_equal(result.weighted_sum[0, 2], 0)
    np.testing.assert_array_equal(result.weight_total[0, 2], 0)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [3, 4])
@pytest.mark.parametrize('weighted', [False, True])
def test_huber_chunk_excludes_padding_with_scalar_frame_weights(
        backend, dtype, channels, weighted):
    stack, mean, std = _huber_inputs(dtype, channels)
    weights = np.array([1., 1., 2., .5]) if weighted else None
    kwargs = dict(stack=stack.reshape(4, -1), ref_mean=mean.ravel(),
                  ref_std=std.ravel(), huber_c=1.345, weights=weights,
                  skip_zero_rgb=True, channels=channels)
    actual_sum, actual_weight = fgp.huber_weighted_chunk(**kwargs)
    expected_sum = 345 if weighted else 300
    expected_weight = 3.5 if weighted else 3
    np.testing.assert_allclose(actual_sum[:3], expected_sum)
    np.testing.assert_allclose(actual_weight[:3], expected_weight)
    assert actual_weight[channels] > 0
    np.testing.assert_array_equal(actual_sum[2 * channels:], 0)
    np.testing.assert_array_equal(actual_weight[2 * channels:], 0)


@pytest.mark.parametrize('channels', [1, 2, 4])
def test_huber_chunk_validates_rgb_channel_layout(channels):
    with pytest.raises(ValueError, match='channels'):
        fgp.huber_weighted_chunk_numpy(
            np.zeros((2, 9), np.uint16), np.zeros(9), np.ones(9), 1.345,
            skip_zero_rgb=True, channels=channels)
