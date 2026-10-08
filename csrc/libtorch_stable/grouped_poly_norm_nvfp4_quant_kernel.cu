/*
 * Motif ModelOpt NVFP4 MoE: grouped PolyNorm fused with the expert-aware
 * activation quantization consumed by the second W4A4 grouped GEMM.
 *
 * This file is compiled for the architecture-specific SM100 family target
 * because native E2M1 conversion is not available in portable sm_100 PTX.
 */

#include "torch_utils.h"

#include "cub_helpers.h"
#include "cuda_vec_utils.cuh"
#include "dispatch_utils.h"
#include "quantization/fp4/nvfp4_utils.cuh"
#include "quantization/vectorization.cuh"

#include <cstdint>

namespace vllm {

template <typename scalar_t>
__global__ void grouped_poly_norm_nvfp4_quant_kernel(
    uint32_t* __restrict__ output, uint32_t* __restrict__ output_scale,
    const scalar_t* __restrict__ input, const scalar_t* __restrict__ mul,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const int64_t* __restrict__ expert_offsets,
    const int* __restrict__ blockscale_offsets,
    const float* __restrict__ input_global_scale, const int num_experts,
    const int64_t input_row_stride, const int64_t mul_row_stride,
    const int hidden_size, const float eps, const float hidden_clamp,
    const float polynorm_output_scale) {
  const int row = blockIdx.x;
  __shared__ int shared_local_expert;
  if (threadIdx.x == 0) {
    if (row >= expert_offsets[num_experts]) {
      shared_local_expert = -1;
    } else {
      int low = 0;
      int high = num_experts;
      while (low < high) {
        const int mid = (low + high) / 2;
        if (expert_offsets[mid + 1] <= row) {
          low = mid + 1;
        } else {
          high = mid;
        }
      }
      shared_local_expert = low;
    }
  }
  __syncthreads();
  const int local_expert = shared_local_expert;
  if (local_expert < 0) {
    return;
  }

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const scalar_t* input_row =
      input + static_cast<size_t>(row) * input_row_stride;
  const scalar_t* mul_row = mul + static_cast<size_t>(row) * mul_row_stride;

  const float w0 = weight[local_expert * 3 + 0];
  const float w1 = weight[local_expert * 3 + 1];
  const float w2 = weight[local_expert * 3 + 2];
  const float b0 = bias[local_expert];
  const bool do_clamp = hidden_clamp > 0.0f;
  const float inv_d = 1.0f / static_cast<float>(hidden_size);

  using VecT = vec_n_t<scalar_t, CVT_FP4_ELTS_PER_THREAD>;
  const auto* vector_input = reinterpret_cast<const VecT*>(input_row);
  const auto* vector_mul = reinterpret_cast<const VecT*>(mul_row);
  const int vector_count = hidden_size / CVT_FP4_ELTS_PER_THREAD;

  float s2 = 0.0f;
  float s4 = 0.0f;
  float s6 = 0.0f;
  for (int i = tid; i < vector_count; i += nthreads) {
    const VecT value = vector_input[i];
#pragma unroll
    for (int j = 0; j < CVT_FP4_ELTS_PER_THREAD; ++j) {
      float x = static_cast<float>(value.val[j]);
      if (do_clamp) {
        x = fmaxf(-hidden_clamp, fminf(hidden_clamp, x));
      }
      const float x2 = x * x;
      const float x4 = x2 * x2;
      s2 += x2;
      s4 += x4;
      s6 += x4 * x2;
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_tmp;
  __shared__ float shared_rms1;
  __shared__ float shared_rms2;
  __shared__ float shared_rms3;

  const float r2 = BlockReduce(reduce_tmp).Reduce(s2, CubAddOp{}, nthreads);
  __syncthreads();
  const float r4 = BlockReduce(reduce_tmp).Reduce(s4, CubAddOp{}, nthreads);
  __syncthreads();
  const float r6 = BlockReduce(reduce_tmp).Reduce(s6, CubAddOp{}, nthreads);
  __syncthreads();
  if (tid == 0) {
    shared_rms1 = rsqrtf(r2 * inv_d + eps);
    shared_rms2 = rsqrtf(r4 * inv_d + eps);
    shared_rms3 = rsqrtf(r6 * inv_d + eps);
  }
  __syncthreads();

  using cuda_type = typename CUDATypeConverter<scalar_t>::Type;
  using QuantVec = PackedVec<cuda_type, CVT_FP4_PACK16>;
  static_assert(sizeof(QuantVec) == sizeof(VecT));
  constexpr int NUM_THREADS_PER_SF =
      CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  const int row_in_expert =
      row - static_cast<int>(expert_offsets[local_expert]);
  const int num_k_tiles = (hidden_size + 63) / 64;
  uint32_t* scale_out_in_expert =
      output_scale + blockscale_offsets[local_expert] * num_k_tiles;
  const float global_scale = input_global_scale[local_expert];

  // Quantization uses a warp shuffle between each adjacent thread pair. Make
  // the branch warp-uniform when the hidden size occupies a partial warp.
  for (int base = 0; base < vector_count; base += nthreads) {
    const int remaining = min(vector_count - base, nthreads);
    const int quant_threads = ((remaining + 31) / 32) * 32;
    if (tid < quant_threads) {
      const int i = base + tid;
      const bool valid = tid < remaining;
      VecT activation_vector{};
      if (valid) {
        const VecT x_vector = vector_input[i];
        const VecT m_vector = vector_mul[i];
#pragma unroll
        for (int j = 0; j < CVT_FP4_ELTS_PER_THREAD; ++j) {
          float x = static_cast<float>(x_vector.val[j]);
          float m = static_cast<float>(m_vector.val[j]);
          if (do_clamp) {
            x = fmaxf(-hidden_clamp, fminf(hidden_clamp, x));
            m = fmaxf(-hidden_clamp, fminf(hidden_clamp, m));
          }
          const float x2 = x * x;
          const float x3 = x2 * x;
          const float poly = w0 * (x3 * shared_rms3) + w1 * (x2 * shared_rms2) +
                             w2 * (x * shared_rms1) + b0;
          scalar_t activation = static_cast<scalar_t>(poly * m);
          if (do_clamp) {
            float activation_fp32 = static_cast<float>(activation);
            activation_fp32 =
                fmaxf(-hidden_clamp, fminf(hidden_clamp, activation_fp32));
            activation = static_cast<scalar_t>(activation_fp32);
          }
          if (polynorm_output_scale != 1.0f) {
            activation = static_cast<scalar_t>(static_cast<float>(activation) *
                                               polynorm_output_scale);
          }
          activation_vector.val[j] = activation;
        }
      }

      QuantVec quant_input;
      *reinterpret_cast<int4*>(&quant_input) =
          *reinterpret_cast<const int4*>(&activation_vector);
      uint8_t* sf_out = nullptr;
      if (valid) {
        sf_out =
            cvt_quant_to_fp4_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(
                row_in_expert, i, num_k_tiles, scale_out_in_expert);
      }
      const auto packed = cvt_warp_fp16_to_fp4<cuda_type, NUM_THREADS_PER_SF>(
          quant_input, valid ? global_scale : 1.0f, sf_out);
      if (valid) {
        output[static_cast<size_t>(row) * vector_count + i] = packed;
      }
    }
  }
}

}  // namespace vllm

// Arch-gated implementation: this translation unit is only compiled when
// FP4_ARCHS is non-empty (see CMakeLists.txt). The always-compiled entry point
// lives in grouped_poly_norm_nvfp4_entry.cu, mirroring nvfp4_quant_entry.cu.
void grouped_poly_norm_nvfp4_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input, torch::stable::Tensor const& mul,
    torch::stable::Tensor const& weight, torch::stable::Tensor const& bias,
    torch::stable::Tensor const& expert_offsets,
    torch::stable::Tensor const& blockscale_offsets,
    torch::stable::Tensor const& input_global_scale, double eps,
    double hidden_clamp, double polynorm_output_scale) {
  const int32_t dev = input.get_device_index();
  STD_TORCH_CHECK(input.dim() == 2, "input must be 2D [N_total, D]");
  STD_TORCH_CHECK(mul.dim() == 2, "mul must be 2D [N_total, D]");
  STD_TORCH_CHECK(input.size(0) == mul.size(0) && input.size(1) == mul.size(1),
                  "input/mul shape mismatch");
  STD_TORCH_CHECK(mul.get_device_index() == dev,
                  "input/mul must be on the same device");
  STD_TORCH_CHECK(input.scalar_type() == mul.scalar_type(),
                  "input/mul must have the same dtype");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "input/mul must be fp16 or bf16");
  STD_TORCH_CHECK(input.stride(1) == 1 && mul.stride(1) == 1,
                  "input/mul must have a unit-stride hidden dimension");
  STD_TORCH_CHECK(
      input.stride(0) >= input.size(1) && mul.stride(0) >= mul.size(1),
      "input/mul rows must not overlap");

  const int N_total = input.size(0);
  const int D = input.size(1);
  STD_TORCH_CHECK(D % CVT_FP4_SF_VEC_SIZE == 0,
                  "hidden size must be a multiple of 16");
  // input/mul are fp16 or bf16 (checked above), so the element size is 2.
  const size_t alignment = sizeof(uint16_t) * CVT_FP4_ELTS_PER_THREAD;
  STD_TORCH_CHECK(
      input.stride(0) % CVT_FP4_ELTS_PER_THREAD == 0 &&
          mul.stride(0) % CVT_FP4_ELTS_PER_THREAD == 0 &&
          reinterpret_cast<uintptr_t>(input.const_data_ptr()) % alignment ==
              0 &&
          reinterpret_cast<uintptr_t>(mul.const_data_ptr()) % alignment == 0,
      "input/mul must support aligned eight-element vector access");

  STD_TORCH_CHECK(
      weight.get_device_index() == dev && bias.get_device_index() == dev,
      "weight/bias must be on the input device");
  STD_TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 3,
                  "weight must be [E_local, 3]");
  STD_TORCH_CHECK(
      bias.dim() == 2 && bias.size(0) == weight.size(0) && bias.size(1) == 1,
      "bias must be [E_local, 1]");
  STD_TORCH_CHECK(
      weight.scalar_type() == torch::headeronly::ScalarType::Float &&
          bias.scalar_type() == torch::headeronly::ScalarType::Float,
      "weight/bias must be fp32");
  STD_TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(),
                  "weight/bias must be contiguous");

  const int num_experts = weight.size(0);
  STD_TORCH_CHECK(
      expert_offsets.get_device_index() == dev && expert_offsets.dim() == 1 &&
          expert_offsets.size(0) == num_experts + 1 &&
          expert_offsets.scalar_type() == torch::headeronly::ScalarType::Long &&
          expert_offsets.is_contiguous(),
      "expert_offsets must be contiguous CUDA int64 [E_local+1]");
  STD_TORCH_CHECK(
      blockscale_offsets.get_device_index() == dev &&
          blockscale_offsets.dim() == 1 &&
          blockscale_offsets.size(0) == num_experts + 1 &&
          blockscale_offsets.scalar_type() ==
              torch::headeronly::ScalarType::Int &&
          blockscale_offsets.is_contiguous(),
      "blockscale_offsets must be contiguous CUDA int32 [E_local+1]");
  STD_TORCH_CHECK(input_global_scale.get_device_index() == dev &&
                      input_global_scale.dim() == 1 &&
                      input_global_scale.size(0) == num_experts &&
                      input_global_scale.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      input_global_scale.is_contiguous(),
                  "input_global_scale must be contiguous CUDA fp32 [E_local]");

  STD_TORCH_CHECK(
      output.get_device_index() == dev &&
          output.scalar_type() == torch::headeronly::ScalarType::Byte &&
          output.is_contiguous() && output.dim() == 2 &&
          output.size(0) == N_total && output.size(1) == D / 2,
      "output must be contiguous CUDA uint8 [N_total, D/2]");
  const int scale_cols = ((D / CVT_FP4_SF_VEC_SIZE) + 3) / 4;
  STD_TORCH_CHECK(
      output_scale.get_device_index() == dev &&
          output_scale.scalar_type() == torch::headeronly::ScalarType::Int &&
          output_scale.is_contiguous() && output_scale.dim() == 2 &&
          output_scale.size(0) >= N_total && output_scale.size(1) == scale_cols,
      "output_scale must be contiguous CUDA int32 with compatible "
      "swizzled scale-factor shape");

  if (N_total == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(dev);
  const cudaStream_t stream = get_current_cuda_stream(dev);
  int block = (D < 1024) ? D : 1024;
  block = ((block + 31) / 32) * 32;
  if (block > 1024) block = 1024;
  if (block < 32) block = 32;

  dim3 grid(N_total);
  dim3 block_dim(block);
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "grouped_poly_norm_nvfp4_quant", [&] {
        vllm::grouped_poly_norm_nvfp4_quant_kernel<scalar_t>
            <<<grid, block_dim, 0, stream>>>(
                reinterpret_cast<uint32_t*>(output.mutable_data_ptr<uint8_t>()),
                reinterpret_cast<uint32_t*>(
                    output_scale.mutable_data_ptr<int>()),
                input.const_data_ptr<scalar_t>(),
                mul.const_data_ptr<scalar_t>(), weight.const_data_ptr<float>(),
                bias.const_data_ptr<float>(),
                expert_offsets.const_data_ptr<int64_t>(),
                blockscale_offsets.const_data_ptr<int>(),
                input_global_scale.const_data_ptr<float>(), num_experts,
                input.stride(0), mul.stride(0), D, static_cast<float>(eps),
                static_cast<float>(hidden_clamp),
                static_cast<float>(polynorm_output_scale));
      });
}
