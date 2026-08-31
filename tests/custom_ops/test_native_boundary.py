import unittest

import numpy as np

import hoshicore._custom_op.ops.max as max_ops


class TestNativeMutableArrayBoundary(unittest.TestCase):
    @staticmethod
    def _compiled_module():
        module, error = max_ops._load_compiled_module_result()
        if module is None:
            raise unittest.SkipTest(error or "compiled custom ops unavailable")
        return module

    def test_fgp_accumulate_rejects_noncontiguous_mutable_outputs(self) -> None:
        module = self._compiled_module()
        fresh = np.ones((2, 2), dtype=np.uint16)

        for mutable_index in range(3):
            outputs = [
                np.zeros((2, 2), dtype=np.uint32),
                np.zeros((2, 2), dtype=np.uint64),
                np.zeros((2, 2), dtype=np.uint16),
            ]
            dtype = outputs[mutable_index].dtype
            storage = np.zeros((2, 4), dtype=dtype)
            outputs[mutable_index] = storage[:, ::2]
            with self.subTest(mutable_index=mutable_index):
                with self.assertRaisesRegex(ValueError, "must be C-contiguous"):
                    module.fgp_accumulate(*outputs, fresh)
                np.testing.assert_array_equal(storage, 0)

    def test_fgp_accumulate_rejects_readonly_mutable_outputs(self) -> None:
        module = self._compiled_module()
        fresh = np.ones((2, 2), dtype=np.uint16)

        for mutable_index in range(3):
            outputs = [
                np.zeros((2, 2), dtype=np.uint32),
                np.zeros((2, 2), dtype=np.uint64),
                np.zeros((2, 2), dtype=np.uint16),
            ]
            outputs[mutable_index].setflags(write=False)
            with self.subTest(mutable_index=mutable_index):
                with self.assertRaisesRegex(ValueError, "must be writeable"):
                    module.fgp_accumulate(*outputs, fresh)
                np.testing.assert_array_equal(outputs[mutable_index], 0)

    def test_huber_accumulate_rejects_invalid_mutable_outputs(self) -> None:
        module = self._compiled_module()
        fresh = np.ones((2, 2), dtype=np.uint16)
        ref_mean = np.ones((2, 2), dtype=np.float32)
        ref_std = np.ones((2, 2), dtype=np.float32)
        other_output = np.zeros((2, 2), dtype=np.float64)
        noncontiguous_storage = np.zeros((2, 4), dtype=np.float64)
        noncontiguous = noncontiguous_storage[:, ::2]
        readonly = np.zeros((2, 2), dtype=np.float64)
        readonly.setflags(write=False)

        for mutable_index in range(2):
            outputs = [other_output.copy(), other_output.copy()]
            outputs[mutable_index] = noncontiguous
            with self.subTest(kind="noncontiguous", mutable_index=mutable_index):
                with self.assertRaisesRegex(ValueError, "must be C-contiguous"):
                    module.huber_weighted_accumulate(
                        *outputs, fresh, ref_mean, ref_std, 1.5
                    )

            outputs = [other_output.copy(), other_output.copy()]
            outputs[mutable_index] = readonly
            with self.subTest(kind="readonly", mutable_index=mutable_index):
                with self.assertRaisesRegex(ValueError, "must be writeable"):
                    module.huber_weighted_accumulate(
                        *outputs, fresh, ref_mean, ref_std, 1.5
                    )

        np.testing.assert_array_equal(noncontiguous_storage, 0)
        np.testing.assert_array_equal(readonly, 0)

    def test_max_kernels_reject_invalid_mutable_outputs(self) -> None:
        module = self._compiled_module()
        fresh = np.ones((2, 2), dtype=np.float32)
        noncontiguous_storage = np.zeros((2, 4), dtype=np.float32)
        noncontiguous = noncontiguous_storage[:, ::2]
        readonly = np.zeros((2, 2), dtype=np.float32)
        readonly.setflags(write=False)

        with self.assertRaisesRegex(ValueError, "base must be C-contiguous"):
            module.max_combine(noncontiguous, fresh)
        with self.assertRaisesRegex(ValueError, "base must be writeable"):
            module.max_combine(readonly, fresh)
        with self.assertRaisesRegex(ValueError, "result must be C-contiguous"):
            module.threshold_max_merge(noncontiguous, fresh, fresh, fresh, 2.0)
        with self.assertRaisesRegex(ValueError, "result must be writeable"):
            module.threshold_max_merge(readonly, fresh, fresh, fresh, 2.0)
        np.testing.assert_array_equal(noncontiguous_storage, 0)
        np.testing.assert_array_equal(readonly, 0)
