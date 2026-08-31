#include <metal_stdlib>

using namespace metal;

// ksize doubles as the Gaussian radius or the morphology kernel size, and
// threshold is only meaningful for the mask kernel; each dispatch sets what it
// needs. Layout must stay in sync with StarMaskParams in star_mask_ops.mm.
struct StarMaskParams {
    uint height;
    uint width;
    uint channels;
    uint ksize;
    float threshold;
};

inline float dtype_max_value(device const uchar*) {
    return 255.0f;
}

inline float dtype_max_value(device const ushort*) {
    return 65535.0f;
}

// OpenCV BORDER_REFLECT_101: -1 -> 1, length -> length - 2.
inline int reflect101(int idx, const int length) {
    if (length <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * length - idx - 2;
        }
    }
    return idx;
}

template <typename T>
inline void to_gray(device const T* image, device float* gray, constant StarMaskParams& params,
                    const uint gid) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    const float max_value = dtype_max_value(image);
    if (params.channels == 1) {
        gray[gid] = float(image[gid]) / max_value;
        return;
    }
    // Same formula and operation order as convert_to_gray in
    // csrc/ops/cpu/star_shrink/star_mask_dog_ops.cpp: weight the raw samples and
    // divide once. Normalizing each channel first (as the CUDA kernel does) is a
    // different fp32 result; matching order is the closest we can get, since the
    // Metal and host compilers may still contract to FMA differently.
    const uint base = gid * 3;
    gray[gid] = (0.114f * float(image[base]) + 0.587f * float(image[base + 1]) +
                 0.299f * float(image[base + 2])) /
                max_value;
}

kernel void star_mask_gray_u8(device const uchar* image [[buffer(0)]],
                              device float* gray [[buffer(1)]],
                              constant StarMaskParams& params [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    to_gray(image, gray, params, gid);
}

kernel void star_mask_gray_u16(device const ushort* image [[buffer(0)]],
                               device float* gray [[buffer(1)]],
                               constant StarMaskParams& params [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    to_gray(image, gray, params, gid);
}

kernel void star_mask_gaussian_horizontal(device const float* input [[buffer(0)]],
                                          device float* tmp [[buffer(1)]],
                                          device const float* weights [[buffer(2)]],
                                          constant StarMaskParams& params [[buffer(3)]],
                                          uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    const int radius = int(params.ksize);
    const int width = int(params.width);
    const uint y = gid / params.width;
    const int x = int(gid - y * params.width);
    float sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        const int xx = reflect101(x + dx, width);
        sum += input[y * params.width + uint(xx)] * weights[dx + radius];
    }
    tmp[gid] = sum;
}

kernel void star_mask_gaussian_vertical(device const float* tmp [[buffer(0)]],
                                        device float* output [[buffer(1)]],
                                        device const float* weights [[buffer(2)]],
                                        constant StarMaskParams& params [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    const int radius = int(params.ksize);
    const int height = int(params.height);
    const uint y = gid / params.width;
    const uint x = gid - y * params.width;
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = reflect101(int(y) + dy, height);
        sum += tmp[uint(yy) * params.width + x] * weights[dy + radius];
    }
    output[gid] = sum;
}

// The mean/stddev reduction that turns this into a threshold runs on the host in
// double precision: Metal has no fp64, and the CPU reference reduces in double.
kernel void star_mask_dog_diff(device const float* blur_small [[buffer(0)]],
                               device const float* blur_large [[buffer(1)]],
                               device float* dog [[buffer(2)]],
                               constant StarMaskParams& params [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    dog[gid] = blur_small[gid] - blur_large[gid];
}

kernel void star_mask_threshold(device const float* dog [[buffer(0)]],
                                device uchar* mask [[buffer(1)]],
                                constant StarMaskParams& params [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    mask[gid] = dog[gid] > params.threshold ? 1 : 0;
}

inline bool cross_active(const uint ksize, const uint ky, const uint kx) {
    const int radius = int(ksize / 2);
    return (int(ky) - radius) == 0 || (int(kx) - radius) == 0;
}

// Border pixels sample only the in-bounds arm of the cross, matching the CUDA
// and CPU kernels: out-of-range taps are skipped rather than treated as 0/1.
kernel void star_mask_erode_cross(device const uchar* input [[buffer(0)]],
                                  device uchar* output [[buffer(1)]],
                                  constant StarMaskParams& params [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    const int height = int(params.height);
    const int width = int(params.width);
    const uint ksize = params.ksize;
    const int radius = int(ksize / 2);
    const int y = int(gid / params.width);
    const int x = int(gid - uint(y) * params.width);
    bool keep = true;
    for (uint ky = 0; ky < ksize && keep; ++ky) {
        const int yy = y + int(ky) - radius;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (uint kx = 0; kx < ksize; ++kx) {
            if (!cross_active(ksize, ky, kx)) {
                continue;
            }
            const int xx = x + int(kx) - radius;
            if (xx < 0 || xx >= width) {
                continue;
            }
            if (input[uint(yy) * params.width + uint(xx)] == 0) {
                keep = false;
                break;
            }
        }
    }
    output[gid] = keep ? 1 : 0;
}

kernel void star_mask_dilate_cross(device const uchar* input [[buffer(0)]],
                                   device uchar* output [[buffer(1)]],
                                   constant StarMaskParams& params [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    const int height = int(params.height);
    const int width = int(params.width);
    const uint ksize = params.ksize;
    const int radius = int(ksize / 2);
    const int y = int(gid / params.width);
    const int x = int(gid - uint(y) * params.width);
    bool keep = false;
    for (uint ky = 0; ky < ksize && !keep; ++ky) {
        const int yy = y + int(ky) - radius;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (uint kx = 0; kx < ksize; ++kx) {
            if (!cross_active(ksize, ky, kx)) {
                continue;
            }
            const int xx = x + int(kx) - radius;
            if (xx < 0 || xx >= width) {
                continue;
            }
            if (input[uint(yy) * params.width + uint(xx)] != 0) {
                keep = true;
                break;
            }
        }
    }
    output[gid] = keep ? 1 : 0;
}
