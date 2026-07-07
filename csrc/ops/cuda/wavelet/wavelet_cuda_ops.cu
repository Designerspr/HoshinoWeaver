#include "common/compat.h"

#include "wavelet_device.cuh"

void launch_wavelet_dec_rec_cuda_core(const double* image_host,
                                      double* out_host,
                                      const int height,
                                      const int width,
                                      const int level) {
    DeviceBuffer current;
    const int threads = 256;

    size_t current_size = static_cast<size_t>(height) * static_cast<size_t>(width);
    current.allocate(current_size, "wavelet_dec_rec_cuda_core cudaMalloc input");
    throw_if_cuda_failed(cudaMemcpy(current.get(),
                                    image_host,
                                    current_size * sizeof(double),
                                    cudaMemcpyHostToDevice),
                         "wavelet_dec_rec_cuda_core cudaMemcpy input");

    DeviceImage result =
        wavelet_dec_rec_device(std::move(current), height, width, level, threads);
    throw_if_cuda_failed(cudaMemcpy(out_host,
                                    result.data.get(),
                                    result.size * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "wavelet_dec_rec_cuda_core cudaMemcpy output");
}
