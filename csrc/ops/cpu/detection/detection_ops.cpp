#include "detection_ops.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ComponentStats {
    ssize_t count = 0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    double sum_xy = 0.0;
    double sum_intensity = 0.0;
};

int32_t find_root(std::vector<int32_t>& parent, int32_t label) {
    int32_t root = label;
    while (parent[static_cast<size_t>(root)] != root) {
        root = parent[static_cast<size_t>(root)];
    }
    while (parent[static_cast<size_t>(label)] != label) {
        const int32_t next = parent[static_cast<size_t>(label)];
        parent[static_cast<size_t>(label)] = root;
        label = next;
    }
    return root;
}

void union_labels(std::vector<int32_t>& parent, const int32_t a, const int32_t b) {
    const int32_t root_a = find_root(parent, a);
    const int32_t root_b = find_root(parent, b);
    if (root_a == root_b) {
        return;
    }
    if (root_a < root_b) {
        parent[static_cast<size_t>(root_b)] = root_a;
    } else {
        parent[static_cast<size_t>(root_a)] = root_b;
    }
}

void validate_inputs(const py::buffer_info& image_info, const py::buffer_info& bw_info) {
    if (image_info.ndim != 2) {
        throw std::invalid_argument(
            "star_detect_connected_components_candidates: image must be 2D");
    }
    if (bw_info.ndim != 2) {
        throw std::invalid_argument("star_detect_connected_components_candidates: bw must be 2D");
    }
    if (image_info.shape[0] <= 0 || image_info.shape[1] <= 0) {
        throw std::invalid_argument(
            "star_detect_connected_components_candidates: image height and width must be positive");
    }
    if (image_info.shape[0] != bw_info.shape[0] || image_info.shape[1] != bw_info.shape[1]) {
        throw std::invalid_argument(
            "star_detect_connected_components_candidates: image and bw shapes must match");
    }
}

py::tuple star_detect_connected_components_candidates_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& bw) {
    const auto image_info = image.request();
    const auto bw_info = bw.request();
    validate_inputs(image_info, bw_info);

    const ssize_t h = image_info.shape[0];
    const ssize_t w = image_info.shape[1];
    const ssize_t plane_size = h * w;
    const auto* image_ptr = static_cast<const double*>(image_info.ptr);
    const auto* bw_ptr = static_cast<const uint8_t*>(bw_info.ptr);

    std::vector<int32_t> labels(static_cast<size_t>(plane_size), 0);
    std::vector<int32_t> parent;
    parent.reserve(static_cast<size_t>(plane_size / 8 + 1));
    parent.push_back(0);
    std::vector<ComponentStats> stats;

    {
        py::gil_scoped_release release;

        for (ssize_t y = 0; y < h; ++y) {
            for (ssize_t x = 0; x < w; ++x) {
                const ssize_t idx = y * w + x;
                if (bw_ptr[idx] == 0) {
                    continue;
                }

                int32_t chosen = 0;
                int32_t neighbors[4] = {0, 0, 0, 0};
                int neighbor_count = 0;
                if (x > 0) {
                    neighbors[neighbor_count++] = labels[static_cast<size_t>(idx - 1)];
                }
                if (y > 0 && x > 0) {
                    neighbors[neighbor_count++] = labels[static_cast<size_t>(idx - w - 1)];
                }
                if (y > 0) {
                    neighbors[neighbor_count++] = labels[static_cast<size_t>(idx - w)];
                }
                if (y > 0 && x + 1 < w) {
                    neighbors[neighbor_count++] = labels[static_cast<size_t>(idx - w + 1)];
                }

                for (int n = 0; n < neighbor_count; ++n) {
                    const int32_t label = neighbors[n];
                    if (label == 0) {
                        continue;
                    }
                    if (chosen == 0 || label < chosen) {
                        chosen = label;
                    }
                }

                if (chosen == 0) {
                    chosen = static_cast<int32_t>(parent.size());
                    parent.push_back(chosen);
                } else {
                    for (int n = 0; n < neighbor_count; ++n) {
                        const int32_t label = neighbors[n];
                        if (label != 0 && label != chosen) {
                            union_labels(parent, chosen, label);
                        }
                    }
                }
                labels[static_cast<size_t>(idx)] = chosen;
            }
        }

        std::vector<ssize_t> root_to_index(parent.size(), -1);
        for (ssize_t idx = 0; idx < plane_size; ++idx) {
            const int32_t label = labels[static_cast<size_t>(idx)];
            if (label == 0) {
                continue;
            }
            const int32_t root = find_root(parent, label);
            ssize_t stats_idx = root_to_index[static_cast<size_t>(root)];
            if (stats_idx < 0) {
                stats_idx = static_cast<ssize_t>(stats.size());
                root_to_index[static_cast<size_t>(root)] = stats_idx;
                stats.push_back(ComponentStats{});
            }

            const double x = static_cast<double>(idx % w);
            const double y = static_cast<double>(idx / w);
            ComponentStats& item = stats[static_cast<size_t>(stats_idx)];
            item.count += 1;
            item.sum_x += x;
            item.sum_y += y;
            item.sum_x2 += x * x;
            item.sum_y2 += y * y;
            item.sum_xy += x * y;
            item.sum_intensity += image_ptr[idx];
        }

        labels.clear();
        labels.shrink_to_fit();
        parent.clear();
        parent.shrink_to_fit();
    }

    ssize_t out_count = 0;
    for (const ComponentStats& item : stats) {
        if (item.count > 5) {
            ++out_count;
        }
    }

    py::array_t<double> positions(std::vector<ssize_t>{out_count, 2});
    py::array_t<double> areas(std::vector<ssize_t>{out_count});
    py::array_t<double> intensities(std::vector<ssize_t>{out_count});
    py::array_t<double> eccentricities(std::vector<ssize_t>{out_count});

    auto positions_info = positions.request();
    auto areas_info = areas.request();
    auto intensities_info = intensities.request();
    auto eccentricities_info = eccentricities.request();
    auto* positions_ptr = static_cast<double*>(positions_info.ptr);
    auto* areas_ptr = static_cast<double*>(areas_info.ptr);
    auto* intensities_ptr = static_cast<double*>(intensities_info.ptr);
    auto* eccentricities_ptr = static_cast<double*>(eccentricities_info.ptr);

    ssize_t out_idx = 0;
    for (const ComponentStats& item : stats) {
        if (item.count <= 5) {
            continue;
        }
        const double inv_count = 1.0 / static_cast<double>(item.count);
        const double cx = item.sum_x * inv_count;
        const double cy = item.sum_y * inv_count;
        const double cov_xx = item.sum_x2 * inv_count - cx * cx;
        const double cov_yy = item.sum_y2 * inv_count - cy * cy;
        const double cov_xy = item.sum_xy * inv_count - cx * cy;
        const double trace = cov_xx + cov_yy;
        const double det_term =
            std::sqrt(std::max(0.0, (cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy));
        const double lambda_max = 0.5 * (trace + det_term);
        const double lambda_min = 0.5 * (trace - det_term);
        double eccentricity = 0.0;
        if (lambda_max > 1e-12) {
            eccentricity = std::sqrt(std::max(0.0, 1.0 - lambda_min / lambda_max));
        }

        positions_ptr[out_idx * 2] = cx;
        positions_ptr[out_idx * 2 + 1] = cy;
        areas_ptr[out_idx] = static_cast<double>(item.count);
        intensities_ptr[out_idx] = item.sum_intensity * inv_count;
        eccentricities_ptr[out_idx] = eccentricity;
        ++out_idx;
    }
    return py::make_tuple(positions, areas, intensities, eccentricities);
}

} // namespace

void bind_detection_ops(py::module_& m) {
    m.def("star_detect_connected_components_candidates",
          &star_detect_connected_components_candidates_impl, py::arg("image"), py::arg("bw"),
          "Extract connected-component star candidates from a binary detection mask.");
}
