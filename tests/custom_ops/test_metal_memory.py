from unittest import mock

from hoshicore._custom_op._dispatch import (
    CustomOpMetalRuntimeUnavailableError,
    CustomOpResourceExhaustedError,
)
import hoshicore._custom_op.metal_memory as metal_memory

from tests.custom_ops._base import CustomOpsTestCase


class TestMetalMemory(CustomOpsTestCase):
    def tearDown(self) -> None:
        metal_memory._reset_metal_reservations_for_tests()
        super().tearDown()

    def test_star_shrink_estimator_matches_workspace_buffers(self) -> None:
        estimate = metal_memory.estimate_star_shrink_process(
            height=11,
            width=13,
            channels=3,
            dtype_bytes=2,
        )
        pixels = 11 * 13
        total = pixels * 3
        expected = 2 * total * 2 + pixels + 4 * pixels * 4 + 3 * total * 4

        self.assertEqual(estimate.logical_op, "star_shrink_process")
        self.assertEqual(estimate.peak_device_bytes, expected)
        self.assertEqual(estimate.confidence, "exact")

    def test_usable_memory_reserves_headroom_and_promises(self) -> None:
        gib = 1024**3

        usable = metal_memory.metal_usable_memory_bytes(
            8 * gib,
            2 * gib,
            gib,
        )

        self.assertEqual(usable, 8 * gib - 2 * gib - gib - int(0.4 * gib))

    def test_admission_reservation_is_visible_to_another_worker(self) -> None:
        gib = 1024**3
        estimate = metal_memory.MetalMemoryEstimate("first", gib)
        second = metal_memory.MetalMemoryEstimate("second", 7 * gib)
        payload = {
            "available": True,
            "status": "available",
            "recommended_max_working_set_bytes": 8 * gib,
            "current_allocated_bytes": 0,
        }

        with mock.patch.object(
            metal_memory,
            "metal_device_info",
            return_value=payload,
        ):
            with metal_memory.metal_memory_admission(estimate) as first:
                with metal_memory.metal_memory_admission(
                    second,
                    evict_cache_once=False,
                ) as denied:
                    self.assertTrue(first.granted)
                    self.assertEqual(denied.reserved_bytes, gib)
                    self.assertFalse(denied.granted)

        with mock.patch.object(
            metal_memory,
            "metal_device_info",
            return_value=payload,
        ):
            with metal_memory.metal_memory_admission(second) as released:
                self.assertTrue(released.granted)

    def test_admission_evicts_cache_once_and_reprobes(self) -> None:
        gib = 1024**3
        estimate = metal_memory.MetalMemoryEstimate("sample", 2 * gib)
        payloads = [
            {
                "available": True,
                "status": "available",
                "recommended_max_working_set_bytes": 8 * gib,
                "current_allocated_bytes": 6 * gib,
            },
            {
                "available": True,
                "status": "available",
                "recommended_max_working_set_bytes": 8 * gib,
                "current_allocated_bytes": gib,
            },
        ]

        with (
            mock.patch.object(
                metal_memory,
                "metal_device_info",
                side_effect=payloads,
            ),
            mock.patch.object(
                metal_memory,
                "_clear_current_thread_metal_cache",
                return_value=True,
            ) as clear_cache,
        ):
            with metal_memory.metal_memory_admission(estimate) as decision:
                self.assertTrue(decision.granted)
                self.assertTrue(decision.cache_evicted)

        clear_cache.assert_called_once()

    def test_explicitly_unavailable_probe_raises_typed_error(self) -> None:
        estimate = metal_memory.MetalMemoryEstimate("sample", 1024)
        with mock.patch.object(
            metal_memory,
            "metal_device_info",
            return_value={
                "available": False,
                "status": "explicitly_unavailable",
                "reason_code": "metal_unified_memory_required",
                "reason": "unified memory required",
            },
        ):
            with self.assertRaisesRegex(
                CustomOpMetalRuntimeUnavailableError,
                "unified memory required",
            ) as caught:
                with metal_memory.metal_memory_admission(estimate):
                    pass

        self.assertEqual(
            caught.exception.reason_code,
            "metal_unified_memory_required",
        )

    def test_run_admitted_metal_denial_stops_before_kernel(self) -> None:
        estimate = metal_memory.MetalMemoryEstimate("sample", 1024)
        kernel = mock.Mock()
        admission = mock.MagicMock()
        admission.__enter__.return_value = metal_memory.MetalAdmissionDecision(
            logical_op="sample",
            granted=False,
            checked=True,
            reason_code="insufficient_working_set_estimate",
            estimated_peak_bytes=1024,
            recommended_max_working_set_bytes=2048,
            current_allocated_bytes=1024,
            headroom_bytes=512,
        )

        with mock.patch.object(
            metal_memory,
            "metal_memory_admission",
            return_value=admission,
        ):
            with self.assertRaisesRegex(
                CustomOpResourceExhaustedError,
                "estimated peak",
            ):
                metal_memory.run_admitted_metal(estimate, kernel)

        kernel.assert_not_called()
