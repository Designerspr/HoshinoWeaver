#pragma once

#include "common/compat.h"

#include <cmath>

#if defined(__CUDACC__)
#define HNW_CAMERA_HD __host__ __device__
#else
#define HNW_CAMERA_HD
#endif

namespace hnw::camera {

constexpr int PROJECTION_PERSPECTIVE = 0;
constexpr int PROJECTION_FISHEYE = 1;

HNW_CAMERA_HD inline bool valid_projection(const int projection) {
    return projection == PROJECTION_PERSPECTIVE || projection == PROJECTION_FISHEYE;
}

HNW_CAMERA_HD inline double solve_fisheye_theta(const double radius_distorted,
                                                const bool has_distortion, const double* coeffs) {
    double theta = radius_distorted;
    if (!has_distortion) {
        return theta;
    }

    const double k1 = coeffs[0];
    const double k2 = coeffs[1];
    const double k3 = coeffs[2];
    const double k4 = coeffs[3];
    for (int iter = 0; iter < 10; ++iter) {
        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta4 * theta4;
        const double value = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) -
                             radius_distorted;
        const double derivative =
            1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8;
        const double safe_derivative = ::fabs(derivative) > 1.0e-12 ? derivative : 1.0e-12;
        theta -= value / safe_derivative;
    }
    return theta;
}

HNW_CAMERA_HD inline void unproject(const double x_distorted, const double y_distorted,
                                    const int projection, const bool has_distortion,
                                    const double* coeffs, double* ray_x, double* ray_y,
                                    double* ray_z) {
    if (projection == PROJECTION_FISHEYE) {
        const double radius_distorted =
            ::sqrt(x_distorted * x_distorted + y_distorted * y_distorted);
        const double theta = solve_fisheye_theta(radius_distorted, has_distortion, coeffs);
        const double safe_radius = radius_distorted > 0.0 ? radius_distorted : 1.0;
        const double sin_theta = ::sin(theta);
        *ray_x = sin_theta * x_distorted / safe_radius;
        *ray_y = sin_theta * y_distorted / safe_radius;
        *ray_z = ::cos(theta);
        return;
    }

    double x = x_distorted;
    double y = y_distorted;
    if (has_distortion) {
        const double k1 = coeffs[0];
        const double k2 = coeffs[1];
        const double p1 = coeffs[2];
        const double p2 = coeffs[3];
        const double k3 = coeffs[4];
        for (int iter = 0; iter < 5; ++iter) {
            const double radius2 = x * x + y * y;
            const double radius4 = radius2 * radius2;
            const double radius6 = radius4 * radius2;
            const double radial = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6;
            const double xy2 = 2.0 * x * y;
            const double delta_x = p1 * xy2 + p2 * (radius2 + 2.0 * x * x);
            const double delta_y = p1 * (radius2 + 2.0 * y * y) + p2 * xy2;
            x = (x_distorted - delta_x) / radial;
            y = (y_distorted - delta_y) / radial;
        }
    }
    *ray_x = x;
    *ray_y = y;
    *ray_z = 1.0;
}

HNW_CAMERA_HD inline bool project(const double ray_x, const double ray_y, const double ray_z,
                                  const int projection, const bool has_distortion,
                                  const double* coeffs, double* x_distorted, double* y_distorted) {
    if (projection == PROJECTION_FISHEYE) {
        const double norm = ::sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
        if (!(norm > 1.0e-12)) {
            return false;
        }
        const double x = ray_x / norm;
        const double y = ray_y / norm;
        const double z = ray_z / norm;
        const double radius_xy = ::sqrt(x * x + y * y);
        const double theta = ::atan2(radius_xy, z);
        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        const double theta8 = theta4 * theta4;
        double theta_distorted = theta;
        if (has_distortion) {
            theta_distorted *= 1.0 + coeffs[0] * theta2 + coeffs[1] * theta4 + coeffs[2] * theta6 +
                               coeffs[3] * theta8;
        }
        const double safe_radius = radius_xy > 1.0e-12 ? radius_xy : 1.0;
        *x_distorted = radius_xy > 1.0e-12 ? theta_distorted * x / safe_radius : 0.0;
        *y_distorted = radius_xy > 1.0e-12 ? theta_distorted * y / safe_radius : 0.0;
        return true;
    }

    if (!(ray_z > 0.0)) {
        return false;
    }
    double x = ray_x / ray_z;
    double y = ray_y / ray_z;
    if (has_distortion) {
        const double radius2 = x * x + y * y;
        const double radius4 = radius2 * radius2;
        const double radius6 = radius4 * radius2;
        const double radial = 1.0 + coeffs[0] * radius2 + coeffs[1] * radius4 + coeffs[4] * radius6;
        const double xy2 = 2.0 * x * y;
        const double x_with_distortion =
            x * radial + coeffs[2] * xy2 + coeffs[3] * (radius2 + 2.0 * x * x);
        const double y_with_distortion =
            y * radial + coeffs[2] * (radius2 + 2.0 * y * y) + coeffs[3] * xy2;
        x = x_with_distortion;
        y = y_with_distortion;
    }
    *x_distorted = x;
    *y_distorted = y;
    return true;
}

} // namespace hnw::camera

#undef HNW_CAMERA_HD
