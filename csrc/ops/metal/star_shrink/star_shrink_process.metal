#include <metal_stdlib>

using namespace metal;

struct StarShrinkParams {
    uint height;
    uint width;
    uint channels;
    uint shrink_ksize;
    uint shrink_shape;
    uint deringing_ksize;
    float shrink_ratio;
};

inline float normalized_sample(device const uchar* image, const uint index) {
    return float(image[index]) / 255.0f;
}

inline float normalized_sample(device const ushort* image, const uint index) {
    return float(image[index]) / 65535.0f;
}

inline uchar cast_u8(const float value) {
    const float scaled = rint(clamp(value, 0.0f, 1.0f) * 255.0f);
    return uchar(clamp(scaled, 0.0f, 255.0f));
}

inline ushort cast_u16(const float value) {
    const float scaled = rint(clamp(value, 0.0f, 1.0f) * 65535.0f);
    return ushort(clamp(scaled, 0.0f, 65535.0f));
}

inline float srgb_to_linear(const float value) {
    const float x = clamp(value, 0.0f, 1.0f);
    if (x <= 0.04045f) {
        return x / 12.92f;
    }
    return pow((x + 0.055f) / 1.055f, 2.4f);
}

inline float linear_to_srgb(const float value) {
    const float x = clamp(value, 0.0f, 1.0f);
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * pow(x, 1.0f / 2.4f) - 0.055f;
}

inline float lab_f(const float value) {
    constexpr float delta = 6.0f / 29.0f;
    constexpr float delta3 = delta * delta * delta;
    if (value > delta3) {
        return pow(value, 1.0f / 3.0f);
    }
    return value / (3.0f * delta * delta) + 4.0f / 29.0f;
}

inline float lab_f_inv(const float value) {
    constexpr float delta = 6.0f / 29.0f;
    if (value > delta) {
        return value * value * value;
    }
    return 3.0f * delta * delta * (value - 4.0f / 29.0f);
}

template <typename T>
inline void bgr_to_lab(device const T* image, device float* luma, device float* lab_a,
                       device float* lab_b, constant StarShrinkParams& params, const uint gid) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    if (params.channels == 1) {
        luma[gid] = normalized_sample(image, gid);
        return;
    }
    const uint base = gid * 3;
    const float b = normalized_sample(image, base);
    const float g = normalized_sample(image, base + 1);
    const float r = normalized_sample(image, base + 2);
    const float rl = srgb_to_linear(r);
    const float gl = srgb_to_linear(g);
    const float bl = srgb_to_linear(b);
    const float x = 0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl;
    const float y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl;
    const float z = 0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl;
    const float fx = lab_f(x / 0.95047f);
    const float fy = lab_f(y);
    const float fz = lab_f(z / 1.08883f);
    luma[gid] = 116.0f * fy - 16.0f;
    lab_a[gid] = 500.0f * (fx - fy);
    lab_b[gid] = 200.0f * (fy - fz);
}

kernel void star_shrink_bgr_to_lab_u8(device const uchar* image [[buffer(0)]],
                                      device float* luma [[buffer(1)]],
                                      device float* lab_a [[buffer(2)]],
                                      device float* lab_b [[buffer(3)]],
                                      constant StarShrinkParams& params [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]]) {
    bgr_to_lab(image, luma, lab_a, lab_b, params, gid);
}

kernel void star_shrink_bgr_to_lab_u16(device const ushort* image [[buffer(0)]],
                                       device float* luma [[buffer(1)]],
                                       device float* lab_a [[buffer(2)]],
                                       device float* lab_b [[buffer(3)]],
                                       constant StarShrinkParams& params [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]]) {
    bgr_to_lab(image, luma, lab_a, lab_b, params, gid);
}

inline bool kernel_active(const uint shape, const uint ksize, const uint ky, const uint kx) {
    const int radius = int(ksize / 2);
    const int dy = int(ky) - radius;
    const int dx = int(kx) - radius;
    if (shape == 0) {
        return true;
    }
    if (shape == 1) {
        return dx == 0 || dy == 0;
    }
    return dx * dx + dy * dy <= radius * radius;
}

kernel void star_shrink_erode_luma(device const float* current [[buffer(0)]],
                                   device float* next [[buffer(1)]],
                                   constant StarShrinkParams& params [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    const int y = int(gid / params.width);
    const int x = int(gid - uint(y) * params.width);
    const int radius = int(params.shrink_ksize / 2);
    float minimum = INFINITY;
    for (uint ky = 0; ky < params.shrink_ksize; ++ky) {
        const int yy = y + int(ky) - radius;
        if (yy < 0 || yy >= int(params.height)) {
            continue;
        }
        for (uint kx = 0; kx < params.shrink_ksize; ++kx) {
            if (!kernel_active(params.shrink_shape, params.shrink_ksize, ky, kx)) {
                continue;
            }
            const int xx = x + int(kx) - radius;
            if (xx < 0 || xx >= int(params.width)) {
                continue;
            }
            minimum = min(minimum, current[uint(yy) * params.width + uint(xx)]);
        }
    }
    next[gid] = minimum * params.shrink_ratio + current[gid] * (1.0f - params.shrink_ratio);
}

kernel void star_shrink_lab_to_bgr(device const float* luma [[buffer(0)]],
                                   device const float* lab_a [[buffer(1)]],
                                   device const float* lab_b [[buffer(2)]],
                                   device float* shrunk [[buffer(3)]],
                                   constant StarShrinkParams& params [[buffer(4)]],
                                   uint gid [[thread_position_in_grid]]) {
    const uint plane_size = params.height * params.width;
    if (gid >= plane_size) {
        return;
    }
    if (params.channels == 1) {
        shrunk[gid] = luma[gid];
        return;
    }
    const float l = luma[gid];
    const float a = lab_a[gid];
    const float b_lab = lab_b[gid];
    const float fy = (l + 16.0f) / 116.0f;
    const float fx = fy + a / 500.0f;
    const float fz = fy - b_lab / 200.0f;
    const float x = 0.95047f * lab_f_inv(fx);
    const float y = lab_f_inv(fy);
    const float z = 1.08883f * lab_f_inv(fz);
    const float rl = 3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    const float gl = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    const float bl = 0.0556434f * x - 0.2040259f * y + 1.0572252f * z;
    const uint base = gid * 3;
    shrunk[base] = clamp(linear_to_srgb(bl), 0.0f, 1.0f);
    shrunk[base + 1] = clamp(linear_to_srgb(gl), 0.0f, 1.0f);
    shrunk[base + 2] = clamp(linear_to_srgb(rl), 0.0f, 1.0f);
}

inline int reflect101(int index, const int length) {
    if (length <= 1) {
        return 0;
    }
    while (index < 0 || index >= length) {
        if (index < 0) {
            index = -index;
        } else {
            index = 2 * length - index - 2;
        }
    }
    return index;
}

template <typename T>
inline void horizontal_blur(device const T* image, device float* tmp,
                            constant StarShrinkParams& params, const uint gid) {
    const uint total = params.height * params.width * params.channels;
    if (gid >= total) {
        return;
    }
    const uint channel = gid % params.channels;
    const uint pixel = gid / params.channels;
    const int y = int(pixel / params.width);
    const int x = int(pixel - uint(y) * params.width);
    const int radius = int(params.deringing_ksize / 2);
    float sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        const int xx = reflect101(x + dx, int(params.width));
        const uint index = (uint(y) * params.width + uint(xx)) * params.channels + channel;
        sum += normalized_sample(image, index);
    }
    tmp[gid] = sum / float(params.deringing_ksize);
}

kernel void star_shrink_horizontal_blur_u8(device const uchar* image [[buffer(0)]],
                                           device float* tmp [[buffer(1)]],
                                           constant StarShrinkParams& params [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]]) {
    horizontal_blur(image, tmp, params, gid);
}

kernel void star_shrink_horizontal_blur_u16(device const ushort* image [[buffer(0)]],
                                            device float* tmp [[buffer(1)]],
                                            constant StarShrinkParams& params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]]) {
    horizontal_blur(image, tmp, params, gid);
}

kernel void star_shrink_vertical_blur(device const float* tmp [[buffer(0)]],
                                      device float* blurred [[buffer(1)]],
                                      constant StarShrinkParams& params [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    const uint total = params.height * params.width * params.channels;
    if (gid >= total) {
        return;
    }
    const uint channel = gid % params.channels;
    const uint pixel = gid / params.channels;
    const int y = int(pixel / params.width);
    const int x = int(pixel - uint(y) * params.width);
    const int radius = int(params.deringing_ksize / 2);
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int yy = reflect101(y + dy, int(params.height));
        const uint index = (uint(yy) * params.width + uint(x)) * params.channels + channel;
        sum += tmp[index];
    }
    blurred[gid] = sum / float(params.deringing_ksize);
}

inline bool final_mask_value(device const uchar* mask, device const float* shrunk,
                             device const float* blurred, constant StarShrinkParams& params,
                             const uint gid, thread float& value) {
    const uint total = params.height * params.width * params.channels;
    if (gid >= total) {
        return false;
    }
    const uint pixel = gid / params.channels;
    if (mask[pixel] == 0) {
        return false;
    }
    value = max(shrunk[gid], blurred[gid]);
    return true;
}

kernel void star_shrink_final_mask_u8(device const uchar* image [[buffer(0)]],
                                      device const uchar* mask [[buffer(1)]],
                                      device const float* shrunk [[buffer(2)]],
                                      device const float* blurred [[buffer(3)]],
                                      device uchar* output [[buffer(4)]],
                                      constant StarShrinkParams& params [[buffer(5)]],
                                      uint gid [[thread_position_in_grid]]) {
    const uint total = params.height * params.width * params.channels;
    if (gid >= total) {
        return;
    }
    float value = 0.0f;
    output[gid] =
        final_mask_value(mask, shrunk, blurred, params, gid, value) ? cast_u8(value) : image[gid];
}

kernel void star_shrink_final_mask_u16(device const ushort* image [[buffer(0)]],
                                       device const uchar* mask [[buffer(1)]],
                                       device const float* shrunk [[buffer(2)]],
                                       device const float* blurred [[buffer(3)]],
                                       device ushort* output [[buffer(4)]],
                                       constant StarShrinkParams& params [[buffer(5)]],
                                       uint gid [[thread_position_in_grid]]) {
    const uint total = params.height * params.width * params.channels;
    if (gid >= total) {
        return;
    }
    float value = 0.0f;
    output[gid] =
        final_mask_value(mask, shrunk, blurred, params, gid, value) ? cast_u16(value) : image[gid];
}
