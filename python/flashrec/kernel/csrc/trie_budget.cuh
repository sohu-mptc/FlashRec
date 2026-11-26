/*
 * Fused sparse lm_head scoring for trie decode.
 *
 * One kernel, no host sync and no [R, D, H] temporary:
 *
 *   tbb_sparse_score  one block per beam row. Walks the row's CSR edge slice,
 *                     computes one dot(hidden, W[token]) per edge (a gather-
 *                     GEMV: the weight rows are read straight out of the
 *                     lm_head, never gathered into a tensor), log-softmaxes
 *                     over the legal children only, and bitonic-sorts the row
 *                     descending.
 *
 * Scores are fp32; token / node / child ids int64; hidden and lm_head weight
 * share one floating dtype (bf16 / fp16 / fp32).
 */
#pragma once

#include <dlpack/dlpack.h>
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>
#include <sgl_kernel/type.cuh>   // For dtype_trait<fp32_t>::from
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE
#include <sgl_kernel/warp.cuh>   // For warp shuffle helpers

namespace {

// Kernel A: one block per beam row, one warp per CSR edge. 256 keeps 8 edges
// in flight per row without oversubscribing the smem hidden vector.
constexpr uint32_t kTbbScoreBlock = 256u;
// Padded out-degree ceiling. The bitonic sort keeps (key, index) for the
// power-of-two padding of D in smem: 2048 * 8B = 16KiB on top of the hidden
// vector. Wider tries stay on the PyTorch path.
constexpr uint32_t kTbbMaxDeg = 2048u;

// Descending order on the key, ties broken by the smaller index so the result
// is independent of how the bitonic network happened to route equal scores.
SGL_DEVICE bool tbb_worse(float ak, int ai, float bk, int bi) {
  if (ak != bk) return ak < bk;
  return ai > bi;
}

extern __shared__ char tbb_dyn_smem[];

// ---------------------------------------------------------------------------
// Kernel A: CSR gather-GEMV -> log-softmax over legal children -> sort desc
// ---------------------------------------------------------------------------
template <typename WeightT>
__global__ void tbb_sparse_score_kernel(
    float* __restrict__ out_lp,              // [R, D]
    int64_t* __restrict__ out_tok,           // [R, D]
    int64_t* __restrict__ out_child,         // [R, D]
    int32_t* __restrict__ out_deg,           // [R]
    const WeightT* __restrict__ weight,      // [vocab, H] restricted lm_head
    const WeightT* __restrict__ hidden,      // [R, H]
    const int64_t* __restrict__ node_ids,    // [R]
    const int32_t* __restrict__ rowptr,      // [n_nodes + 1]
    const int64_t* __restrict__ edge_tok,    // [E]
    const int64_t* __restrict__ edge_child,  // [E]
    int64_t token_base,
    int64_t vocab,
    int64_t n_nodes,
    int64_t invalid_node,
    float neg_inf,
    uint32_t R,
    uint32_t H,
    uint32_t D,
    uint32_t Dp) {
  const uint32_t r = blockIdx.x;
  if (r >= R) return;

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5;
  const uint32_t n_warps = blockDim.x >> 5;

  float* h_s = reinterpret_cast<float*>(tbb_dyn_smem);
  float* key_s = h_s + H;
  int* idx_s = reinterpret_cast<int*>(key_s + Dp);
  __shared__ float red_s[kTbbScoreBlock / 32];

  const int64_t node = node_ids[r];
  int64_t start = 0;
  uint32_t d = 0u;
  // node < 0 marks an unconstrained row (past the trie's max depth): it has no
  // legal-child set to score, so the caller keeps it on the dense head.
  if (node >= 0 && node < n_nodes) {
    start = static_cast<int64_t>(rowptr[node]);
    const int64_t end = static_cast<int64_t>(rowptr[node + 1]);
    const int64_t deg = end - start;
    d = static_cast<uint32_t>(deg < 0 ? 0 : (deg > static_cast<int64_t>(D) ? static_cast<int64_t>(D) : deg));
  }
  if (tid == 0u) out_deg[r] = static_cast<int32_t>(d);

  if (d == 0u) {
    for (uint32_t j = tid; j < D; j += blockDim.x) {
      const size_t o = static_cast<size_t>(r) * D + j;
      out_lp[o] = neg_inf;
      out_tok[o] = 0;
      out_child[o] = invalid_node;
    }
    return;
  }

  const WeightT* h_row = hidden + static_cast<size_t>(r) * H;
  for (uint32_t i = tid; i < H; i += blockDim.x) {
    h_s[i] = dtype_trait<fp32_t>::from(h_row[i]);
  }
  __syncthreads();

  // One warp per edge: lanes stride the hidden dim, then a shuffle reduce.
  // Reading weight[rel] in place is what avoids the [R, D, H] gather.
  for (uint32_t e = warp; e < d; e += n_warps) {
    const int64_t rel = edge_tok[start + e] - token_base;
    const bool ok = (rel >= 0 && rel < vocab);
    float acc = 0.0f;
    if (ok) {
      const WeightT* w = weight + static_cast<size_t>(rel) * H;
      for (uint32_t i = lane; i < H; i += 32u) {
        acc += h_s[i] * dtype_trait<fp32_t>::from(w[i]);
      }
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, m, 32);
    }
    if (lane == 0u) {
      key_s[e] = ok ? acc : neg_inf;
      idx_s[e] = static_cast<int>(e);
    }
  }
  for (uint32_t j = d + tid; j < Dp; j += blockDim.x) {
    key_s[j] = neg_inf;
    idx_s[j] = static_cast<int>(j);
  }
  __syncthreads();

  // log_softmax over the d legal logits.
  float m_part = neg_inf;
  for (uint32_t j = tid; j < d; j += blockDim.x) {
    m_part = fmaxf(m_part, key_s[j]);
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    m_part = fmaxf(m_part, __shfl_xor_sync(0xffffffffu, m_part, m, 32));
  }
  if (lane == 0u) red_s[warp] = m_part;
  __syncthreads();
  if (tid == 0u) {
    float mx = red_s[0];
    for (uint32_t w = 1; w < n_warps; ++w)
      mx = fmaxf(mx, red_s[w]);
    red_s[0] = mx;
  }
  __syncthreads();
  const float row_max = red_s[0];

  float s_part = 0.0f;
  for (uint32_t j = tid; j < d; j += blockDim.x) {
    s_part += __expf(key_s[j] - row_max);
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    s_part += __shfl_xor_sync(0xffffffffu, s_part, m, 32);
  }
  if (lane == 0u) red_s[warp] = s_part;
  __syncthreads();
  if (tid == 0u) {
    float sm = red_s[0];
    for (uint32_t w = 1; w < n_warps; ++w)
      sm += red_s[w];
    red_s[0] = sm;
  }
  __syncthreads();
  const float log_z = row_max + __logf(red_s[0]);

  for (uint32_t j = tid; j < d; j += blockDim.x) {
    key_s[j] -= log_z;
  }
  __syncthreads();

  // Bitonic sort descending on (key, idx) over the power-of-two padding.
  for (uint32_t k = 2u; k <= Dp; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
      for (uint32_t i = tid; i < Dp; i += blockDim.x) {
        const uint32_t l = i ^ j;
        if (l > i) {
          const bool up = ((i & k) == 0u);
          const bool i_worse = tbb_worse(key_s[i], idx_s[i], key_s[l], idx_s[l]);
          if (up == i_worse) {
            const float tk = key_s[i];
            const int ti = idx_s[i];
            key_s[i] = key_s[l];
            idx_s[i] = idx_s[l];
            key_s[l] = tk;
            idx_s[l] = ti;
          }
        }
      }
      __syncthreads();
    }
  }

  for (uint32_t j = tid; j < D; j += blockDim.x) {
    const size_t o = static_cast<size_t>(r) * D + j;
    if (j < Dp && key_s[j] > neg_inf) {
      const int64_t e = start + static_cast<int64_t>(idx_s[j]);
      out_lp[o] = key_s[j];
      out_tok[o] = edge_tok[e];
      out_child[o] = edge_child[e];
    } else {
      out_lp[o] = neg_inf;
      out_tok[o] = 0;
      out_child[o] = invalid_node;
    }
  }
}

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

// Cache the opt-in dynamic-smem budget per (device, kernel): the static
// __shared__ arrays already eat into the per-block limit, and calling
// cudaFuncSetAttribute on every decode step would be a host call on the
// critical path.
template <typename KernelT>
size_t tbb_max_dyn_smem(KernelT kernel, int device_id, const void* tag) {
  static constexpr int kMaxDevices = 16;
  static const void* tags[kMaxDevices] = {};
  static size_t cached[kMaxDevices] = {};
  const bool cacheable = device_id >= 0 && device_id < kMaxDevices;
  if (cacheable && tags[device_id] == tag) return cached[device_id];

  size_t budget = 0;
  int opt_in = 0;
  cudaFuncAttributes attr{};
  if (::cudaFuncGetAttributes(&attr, kernel) == cudaSuccess &&
      ::cudaDeviceGetAttribute(&opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) == cudaSuccess) {
    const size_t stat = static_cast<size_t>(attr.sharedSizeBytes);
    const size_t total = static_cast<size_t>(opt_in);
    // Drivers reject a request landing exactly on the limit on some
    // architectures; keep one alignment unit in reserve.
    if (total > stat + 128u) {
      budget = total - stat - 128u;
      if (::cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(budget)) !=
          cudaSuccess) {
        budget = 0;
        ::cudaGetLastError();
      }
    }
  } else {
    ::cudaGetLastError();
  }
  if (cacheable) {
    tags[device_id] = tag;
    cached[device_id] = budget;
  }
  return budget;
}

template <typename WeightT>
void tbb_launch_score(
    void* out_lp,
    void* out_tok,
    void* out_child,
    void* out_deg,
    const void* weight,
    const void* hidden,
    const void* node_ids,
    const void* rowptr,
    const void* edge_tok,
    const void* edge_child,
    int64_t token_base,
    int64_t vocab,
    int64_t n_nodes,
    int64_t invalid_node,
    uint32_t R,
    uint32_t H,
    uint32_t D,
    DLDevice device) {
  uint32_t dp = 1u;
  while (dp < D)
    dp <<= 1;
  const size_t dyn = static_cast<size_t>(H) * sizeof(float) + static_cast<size_t>(dp) * (sizeof(float) + sizeof(int));
  auto kernel = tbb_sparse_score_kernel<WeightT>;
  static const char tag_score = 0;
  const size_t cap = tbb_max_dyn_smem(kernel, device.device_id, &tag_score);
  host::RuntimeCheck(dyn <= cap, "trie_budget_score: hidden + sorted candidates exceed shared memory");
  host::LaunchKernel(R, kTbbScoreBlock, device, dyn)(
      kernel,
      static_cast<float*>(out_lp),
      static_cast<int64_t*>(out_tok),
      static_cast<int64_t*>(out_child),
      static_cast<int32_t*>(out_deg),
      static_cast<const WeightT*>(weight),
      static_cast<const WeightT*>(hidden),
      static_cast<const int64_t*>(node_ids),
      static_cast<const int32_t*>(rowptr),
      static_cast<const int64_t*>(edge_tok),
      static_cast<const int64_t*>(edge_child),
      token_base,
      vocab,
      n_nodes,
      invalid_node,
      -std::numeric_limits<float>::infinity(),
      R,
      H,
      D,
      dp);
}

/// \brief Sparse trie-neighbor scoring: log_softmax over legal children only.
///
/// Writes ``[R, D]`` planes sorted descending per row, plus each row's true
/// out-degree. ``hidden`` and ``weight`` must share dtype.
void trie_budget_score(
    tvm::ffi::TensorView out_logprobs,  // [R, D] fp32
    tvm::ffi::TensorView out_tokens,    // [R, D] int64
    tvm::ffi::TensorView out_children,  // [R, D] int64
    tvm::ffi::TensorView out_deg,       // [R] int32
    tvm::ffi::TensorView weight,        // [vocab, H]
    tvm::ffi::TensorView hidden,        // [R, H]
    tvm::ffi::TensorView node_ids,      // [R] int64
    tvm::ffi::TensorView edge_rowptr,   // [n_nodes + 1] int32
    tvm::ffi::TensorView edge_token,    // [E] int64
    tvm::ffi::TensorView edge_child,    // [E] int64
    int64_t token_base,
    int64_t invalid_node) {
  using namespace host;

  SymbolicSize R = {"rows"};
  SymbolicSize D = {"max_deg"};
  SymbolicSize Hd = {"hidden_dim"};
  SymbolicSize Vv = {"vocab"};
  SymbolicSize Np1 = {"n_nodes_p1"};
  SymbolicSize E = {"n_edges"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({R, D}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(out_logprobs);
  TensorMatcher({R, D}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(out_tokens).verify(out_children);
  TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(out_deg);
  TensorMatcher({R}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(node_ids);
  TensorMatcher({Np1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(edge_rowptr);
  TensorMatcher({E}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(edge_token).verify(edge_child);
  TensorMatcher({Vv, Hd}).with_device<kDLCUDA>(device_).verify(weight);
  TensorMatcher({R, Hd}).with_device<kDLCUDA>(device_).verify(hidden);

  const uint32_t rows = static_cast<uint32_t>(R.unwrap());
  const uint32_t d = static_cast<uint32_t>(D.unwrap());
  const uint32_t h = static_cast<uint32_t>(Hd.unwrap());
  RuntimeCheck(rows > 0 && d > 0 && h > 0, "trie_budget_score: empty");
  RuntimeCheck(d <= kTbbMaxDeg, "trie_budget_score: max_deg above kTbbMaxDeg");
  RuntimeCheck(Np1.unwrap() >= 2, "trie_budget_score: edge_rowptr too short");
  const DLDataType wt = weight.dtype();
  const DLDataType ht = hidden.dtype();
  RuntimeCheck(
      wt.code == ht.code && wt.bits == ht.bits && wt.lanes == ht.lanes,
      "trie_budget_score: hidden and weight dtype differ");

  const int64_t n_nodes = Np1.unwrap() - 1;
  auto dispatch = [&](auto tag) {
    using WeightT = decltype(tag);
    tbb_launch_score<WeightT>(
        out_logprobs.data_ptr(),
        out_tokens.data_ptr(),
        out_children.data_ptr(),
        out_deg.data_ptr(),
        weight.data_ptr(),
        hidden.data_ptr(),
        node_ids.data_ptr(),
        edge_rowptr.data_ptr(),
        edge_token.data_ptr(),
        edge_child.data_ptr(),
        token_base,
        Vv.unwrap(),
        n_nodes,
        invalid_node,
        rows,
        h,
        d,
        device_.unwrap());
  };
  if (wt.code == kDLFloat && wt.bits == 32) {
    dispatch(fp32_t{});
  } else if (wt.code == kDLFloat && wt.bits == 16) {
    dispatch(fp16_t{});
  } else if (wt.code == kDLBfloat && wt.bits == 16) {
    dispatch(bf16_t{});
  } else {
    RuntimeCheck(false, "trie_budget_score: weight dtype must be fp32 / fp16 / bf16");
  }
}

}  // namespace
