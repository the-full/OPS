#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include "interpolation_cuda_kernel.h"


void nearestneighbor_cuda(int b, int n, int m, at::Tensor unknown_tensor, at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor)
{
    const float *unknown = unknown_tensor.data_ptr<float>();
    const float *known = known_tensor.data_ptr<float>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    nearestneighbor_cuda_launcher(b, n, m, unknown, known, dist2, idx);
}

void interpolation_forward_cuda(int b, int c, int m, int n, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor)
{
    const float *points = points_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *out = out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    interpolation_forward_cuda_launcher(b, c, m, n, points, idx, weight, out);
}

void interpolation_backward_cuda(int b, int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_points_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *grad_points = grad_points_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    interpolation_backward_cuda_launcher(b, c, n, m, grad_out, idx, weight, grad_points);
}

void nearestneighbor_cuda_fast(int b, int n, int m, at::Tensor unknown_tensor, at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data_ptr<float>();
    const float *known = known_tensor.data_ptr<float>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    nearestneighbor_cuda_launcher_fast(b, n, m, unknown, known, dist2, idx);
}

void interpolation_forward_cuda_fast(int b, int c, int m, int n, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor) {

    const float *points = points_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *out = out_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    interpolation_forward_cuda_launcher_fast(b, c, m, n, points, idx, weight, out);
}
