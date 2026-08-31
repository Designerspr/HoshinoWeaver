# Debug tools

This directory contains developer-facing diagnostic entrypoints. They call
current HoshinoWeaver production code, and sometimes private implementation
APIs, but they are not stable application interfaces.

Run them from the repository root with Python's module form so the project
package and sibling debug modules resolve consistently:

```powershell
python -m tools.debug.debug_norma_ba <image-folder> [options]
python -m tools.debug.debug_norma_ba_sampling <image-folder> [options]
python -m tools.debug.debug_fisheye_two_frame_align --reference <ref> --source <src> --output <image> [options]
python -m tools.debug.debug_manual_fisheye_pairs --pairs <csv> [options]
python -m tools.debug.debug_star_detection <image> [options]
python -m tools.debug.debug_star_shrink <image> [options]
```

The scripts may change together with internal APIs. Keep reusable algorithmic
behavior in `hoshicore/` and test it there; the tracked test suite must not
import these tools. Local datasets and generated results remain outside version
control.
