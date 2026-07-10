#include "fgp_ops.h"

#include "fgp_internal.h"

void bind_fgp_ops(py::module_& m) {
    bind_fgp_accumulate_ops(m);
    bind_fgp_merge_ops(m);
    bind_sigma_clip_merge_ops(m);
    bind_huber_accumulate_ops(m);
}
