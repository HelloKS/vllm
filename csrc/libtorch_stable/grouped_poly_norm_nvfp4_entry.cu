/*
 * Always-compiled entry point for the motif3 grouped-PolyNorm NVFP4 quant op.
 *
 * The kernel itself (grouped_poly_norm_nvfp4_quant_kernel.cu) is only compiled
 * when FP4_ARCHS is non-empty, so its symbol is absent on builds without an
 * NVFP4-capable arch. This file is always compiled and always registers the op,
 * failing with a clear message at call time instead of at link time — the same
 * split nvfp4_quant_entry.cu uses for scaled_fp4_quant.
 */

#include <torch/csrc/stable/tensor.h>

#include "torch_utils.h"

#include "cutlass_extensions/common.hpp"

#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
void grouped_poly_norm_nvfp4_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input, torch::stable::Tensor const& mul,
    torch::stable::Tensor const& weight, torch::stable::Tensor const& bias,
    torch::stable::Tensor const& expert_offsets,
    torch::stable::Tensor const& blockscale_offsets,
    torch::stable::Tensor const& input_global_scale, double eps,
    double hidden_clamp, double polynorm_output_scale);
#endif

void grouped_poly_norm_nvfp4_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input, torch::stable::Tensor const& mul,
    torch::stable::Tensor const& weight, torch::stable::Tensor const& bias,
    torch::stable::Tensor const& expert_offsets,
    torch::stable::Tensor const& blockscale_offsets,
    torch::stable::Tensor const& input_global_scale, double eps,
    double hidden_clamp, double polynorm_output_scale) {
#if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  const int32_t sm = get_sm_version_num();
  STD_TORCH_CHECK(sm >= 100 && sm < 120,
                  "No compiled grouped_poly_norm_nvfp4_quant kernel for SM ",
                  sm, ". Recompile with the appropriate CUDA arch.");
  return grouped_poly_norm_nvfp4_quant_sm1xxa(
      output, output_scale, input, mul, weight, bias, expert_offsets,
      blockscale_offsets, input_global_scale, eps, hidden_clamp,
      polynorm_output_scale);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled grouped_poly_norm_nvfp4_quant kernel");
}
