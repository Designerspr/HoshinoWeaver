# Custom-op tests

Tests are grouped by logical op rather than by implementation backend. Keep the
NumPy reference, compiled CPU/CUDA correctness, fallback, dtype, mask, and edge
cases for one logical op in the same module.

Shared infrastructure has dedicated modules:

- `test_dispatch.py` — thread policy and CUDA unavailable classification
- `test_backend_registry.py` — backend candidate selection
- `test_native_boundary.py` — direct `_C` mutable-array contracts
- `test_cuda_workspace.py` — device/pinned cache reuse and worker lifecycle
- `_base.py` — common selector-cache cleanup for wrapper tests

Run the suite with:

```bash
conda run -n astro-dev python -m pytest tests/custom_ops/ -v
```
