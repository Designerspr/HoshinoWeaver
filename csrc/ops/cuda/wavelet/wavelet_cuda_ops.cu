#include "common/compat.h"
#include "wavelet_device.cuh"

void launch_wavelet_dec_rec_cuda_core(const double* image_host, double* out_host, const int height,
                                      const int width, const int level) {
    auto workspace =
        hnw::cuda::acquire_host_io_workspace("wavelet_dec_rec_cuda_core cudaGetDevice");
    try {
        DeviceBuffer current;
        const int threads = 256;
        cudaStream_t stream = workspace.stream();

        const size_t current_size = static_cast<size_t>(height) * static_cast<size_t>(width);
        current.allocate(current_size, "wavelet_dec_rec_cuda_core cudaMalloc input", &workspace);
        throw_if_cuda_failed(cudaMemcpyAsync(current.get(), image_host,
                                             current_size * sizeof(double), cudaMemcpyHostToDevice,
                                             stream),
                             "wavelet_dec_rec_cuda_core cudaMemcpy input");

        DeviceImage result = wavelet_dec_rec_device(std::move(current), height, width, level,
                                                    threads, &workspace, stream);
        throw_if_cuda_failed(cudaMemcpyAsync(out_host, result.data.get(),
                                             result.size * sizeof(double), cudaMemcpyDeviceToHost,
                                             stream),
                             "wavelet_dec_rec_cuda_core cudaMemcpy output");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "wavelet_dec_rec_cuda_core synchronize");
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}
