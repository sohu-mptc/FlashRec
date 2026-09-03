/*
 * Fused GenRec / beam-search trie helpers.
 *
 * Collapses the multi-kernel PyTorch chains in beam_valid_path.py:
 *   mask:     sub + clamp + gather + and + masked_fill
 *   advance:  clamp + gather(parent) + clamp + gather(next) + where×2
 *   expand:   gather(parent rows) + write new token column
 *
 * All tensors are contiguous CUDA; node / token ids are int64; scores fp32;
 * allow_table is uint8 (view of torch.bool).
 */
#pragma once

#include <dlpack/dlpack.h>
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For device::warp shuffle helpers / kWarpThreads

namespace {

// 512 threads/block so genrec_fused covers beam widths up to 512; static smem
// (cum_s/nodes_s/sel_* below) stays ~14KiB, and the wide-beam score matrix
// already spills to the global scratch via kMaxSmemScoresBytes.
constexpr uint32_t kBeamTrieBlock = 512;
constexpr uint32_t kMaxSmemScoresBytes = 48u * 1024u;

SGL_DEVICE int64_t clamp64(int64_t x, int64_t lo, int64_t hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

SGL_DEVICE bool trie_cand_ok(
    int64_t node,
    int64_t tok,
    const uint8_t* __restrict__ allow_table,
    int64_t token_base,
    int64_t n_nodes,
    int64_t vsz) {
  if (node < 0) return true;
  const int64_t rel = tok - token_base;
  if (rel < 0 || rel >= vsz || n_nodes <= 0) return false;
  const int64_t safe = clamp64(node, 0, n_nodes - 1);
  return __ldg(allow_table + safe * vsz + rel) != 0;
}

SGL_DEVICE int64_t trie_transition(
    int64_t old,
    int64_t tok,
    const int64_t* __restrict__ next_node,
    int64_t token_base,
    int64_t n_nodes,
    int64_t vsz,
    int64_t invalid_node) {
  if (old < 0) return old;
  const int64_t rel = tok - token_base;
  if (rel < 0 || rel >= vsz || n_nodes <= 0) return invalid_node;
  const int64_t safe = clamp64(old, 0, n_nodes - 1);
  return __ldg(next_node + safe * vsz + rel);
}

// ---------------------------------------------------------------------------
// mask_candidates: scores_out[r,c] = ok ? scores_in[r,c] : neg_inf
// ---------------------------------------------------------------------------
__global__ void trie_mask_kernel(
    float* __restrict__ scores_out,
    const float* __restrict__ scores_in,
    const int64_t* __restrict__ cand_tokens,
    const int64_t* __restrict__ node_ids,
    const uint8_t* __restrict__ allow_table,
    int64_t token_base,
    int64_t n_nodes,
    int64_t vsz,
    float neg_inf,
    uint32_t rows,
    uint32_t cols) {
  const uint32_t total = rows * cols;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    const uint32_t r = idx / cols;
    const int64_t node = node_ids[r];
    const int64_t tok = cand_tokens[idx];
    const bool ok = trie_cand_ok(node, tok, allow_table, token_base, n_nodes, vsz);
    scores_out[idx] = ok ? scores_in[idx] : neg_inf;
  }
}

void trie_mask_candidates(
    tvm::ffi::TensorView scores_out,
    tvm::ffi::TensorView scores_in,
    tvm::ffi::TensorView cand_tokens,
    tvm::ffi::TensorView node_ids,
    tvm::ffi::TensorView allow_table,
    int64_t token_base,
    double neg_inf) {
  using namespace host;

  SymbolicSize R = {"rows"};
  SymbolicSize C = {"cols"};
  SymbolicSize Nn = {"n_nodes"};
  SymbolicSize V = {"vsz"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({R, C})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(scores_out)
      .verify(scores_in);
  TensorMatcher({R, C})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(cand_tokens);
  TensorMatcher({R})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(node_ids);
  TensorMatcher({Nn, V})  //
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(allow_table);

  const uint32_t rows = static_cast<uint32_t>(R.unwrap());
  const uint32_t cols = static_cast<uint32_t>(C.unwrap());
  const int64_t n_nodes = Nn.unwrap();
  const int64_t vsz = V.unwrap();
  RuntimeCheck(rows > 0 && cols > 0, "trie_mask: empty scores");
  RuntimeCheck(vsz > 0, "trie_mask: empty allow vocab");

  const uint32_t total = rows * cols;
  const uint32_t grid = div_ceil(total, kBeamTrieBlock);
  LaunchKernel(grid, kBeamTrieBlock, device_.unwrap())(
      trie_mask_kernel,
      static_cast<float*>(scores_out.data_ptr()),
      static_cast<const float*>(scores_in.data_ptr()),
      static_cast<const int64_t*>(cand_tokens.data_ptr()),
      static_cast<const int64_t*>(node_ids.data_ptr()),
      static_cast<const uint8_t*>(allow_table.data_ptr()),
      token_base,
      n_nodes,
      vsz,
      static_cast<float>(neg_inf),
      rows,
      cols);
}

// ---------------------------------------------------------------------------
// advance_node_ids (1D): out[i] = transition(node_ids[parents[i]], tokens[i])
// ---------------------------------------------------------------------------
__global__ void trie_advance_kernel(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ node_ids,
    const int64_t* __restrict__ parents,
    const int64_t* __restrict__ tokens,
    const int64_t* __restrict__ next_node,
    int64_t token_base,
    int64_t n_nodes,
    int64_t vsz,
    int64_t invalid_node,
    int64_t max_parent,
    uint32_t n) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const int64_t p = clamp64(parents[i], 0, max_parent);
    const int64_t old = node_ids[p];
    out[i] = trie_transition(old, tokens[i], next_node, token_base, n_nodes, vsz, invalid_node);
  }
}

void trie_advance_nodes(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView node_ids,
    tvm::ffi::TensorView parents,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView next_node,
    int64_t token_base,
    int64_t invalid_node) {
  using namespace host;

  SymbolicSize N = {"n"};
  SymbolicSize Bw = {"bw"};
  SymbolicSize Nn = {"n_nodes"};
  SymbolicSize V = {"vsz"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(out)
      .verify(parents)
      .verify(tokens);
  TensorMatcher({Bw})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(node_ids);
  TensorMatcher({Nn, V})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(next_node);

  const uint32_t n = static_cast<uint32_t>(N.unwrap());
  const int64_t bw = Bw.unwrap();
  const int64_t n_nodes = Nn.unwrap();
  const int64_t vsz = V.unwrap();
  RuntimeCheck(n > 0, "trie_advance: empty");
  RuntimeCheck(bw > 0 && vsz > 0, "trie_advance: bad table");

  const uint32_t grid = div_ceil(n, kBeamTrieBlock);
  LaunchKernel(grid, kBeamTrieBlock, device_.unwrap())(
      trie_advance_kernel,
      static_cast<int64_t*>(out.data_ptr()),
      static_cast<const int64_t*>(node_ids.data_ptr()),
      static_cast<const int64_t*>(parents.data_ptr()),
      static_cast<const int64_t*>(tokens.data_ptr()),
      static_cast<const int64_t*>(next_node.data_ptr()),
      token_base,
      n_nodes,
      vsz,
      invalid_node,
      bw - 1,
      n);
}

// ---------------------------------------------------------------------------
// advance_node_ids batched: node_ids/parents/tokens/out are [B, BW]
// ---------------------------------------------------------------------------
__global__ void trie_advance_batched_kernel(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ node_ids,
    const int64_t* __restrict__ parents,
    const int64_t* __restrict__ tokens,
    const int64_t* __restrict__ next_node,
    int64_t token_base,
    int64_t n_nodes,
    int64_t vsz,
    int64_t invalid_node,
    int64_t max_parent,
    uint32_t batch,
    uint32_t bw) {
  const uint32_t total = batch * bw;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    const uint32_t b = idx / bw;
    const int64_t p = clamp64(parents[idx], 0, max_parent);
    const int64_t old = node_ids[b * bw + static_cast<uint32_t>(p)];
    out[idx] = trie_transition(old, tokens[idx], next_node, token_base, n_nodes, vsz, invalid_node);
  }
}

void trie_advance_nodes_batched(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView node_ids,
    tvm::ffi::TensorView parents,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView next_node,
    int64_t token_base,
    int64_t invalid_node) {
  using namespace host;

  SymbolicSize B = {"batch"};
  SymbolicSize Bw = {"bw"};
  SymbolicSize Nn = {"n_nodes"};
  SymbolicSize V = {"vsz"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({B, Bw})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(out)
      .verify(node_ids)
      .verify(parents)
      .verify(tokens);
  TensorMatcher({Nn, V})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(next_node);

  const uint32_t batch = static_cast<uint32_t>(B.unwrap());
  const uint32_t bw = static_cast<uint32_t>(Bw.unwrap());
  const int64_t n_nodes = Nn.unwrap();
  const int64_t vsz = V.unwrap();
  RuntimeCheck(batch > 0 && bw > 0, "trie_advance_batched: empty");
  RuntimeCheck(vsz > 0, "trie_advance_batched: empty vocab");

  const uint32_t total = batch * bw;
  const uint32_t grid = div_ceil(total, kBeamTrieBlock);
  LaunchKernel(grid, kBeamTrieBlock, device_.unwrap())(
      trie_advance_batched_kernel,
      static_cast<int64_t*>(out.data_ptr()),
      static_cast<const int64_t*>(node_ids.data_ptr()),
      static_cast<const int64_t*>(parents.data_ptr()),
      static_cast<const int64_t*>(tokens.data_ptr()),
      static_cast<const int64_t*>(next_node.data_ptr()),
      token_base,
      n_nodes,
      vsz,
      invalid_node,
      static_cast<int64_t>(bw) - 1,
      batch,
      bw);
}

SGL_DEVICE void beam_expand_write(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ token_ids,
    const int64_t* __restrict__ parents,
    const int64_t* __restrict__ new_tokens,
    int64_t max_parent,
    uint32_t width,
    uint32_t col,
    uint32_t b,
    uint32_t c) {
  if (c == col) {
    out[b * width + c] = new_tokens[b];
    return;
  }
  const int64_t p = clamp64(parents[b], 0, max_parent);
  out[b * width + c] = token_ids[static_cast<uint32_t>(p) * width + c];
}

// ---------------------------------------------------------------------------
// expand token_ids [BW, L] -> out [BW, L] (L must already have room for col)
// out[b, c] = (c == col) ? new_tokens[b] : token_ids[parents[b], c]
// ---------------------------------------------------------------------------
__global__ void beam_expand_kernel(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ token_ids,
    const int64_t* __restrict__ parents,
    const int64_t* __restrict__ new_tokens,
    int64_t max_parent,
    uint32_t bw,
    uint32_t width,
    uint32_t col) {
  const uint32_t total = bw * width;
  if ((width % 2u) == 0u) {
    using vec_t = device::AlignedVector<int64_t, 2>;
    const uint32_t n_vecs = total / 2u;
    for (uint32_t vi = blockIdx.x * blockDim.x + threadIdx.x; vi < n_vecs; vi += blockDim.x * gridDim.x) {
      const uint32_t idx = vi * 2u;
      const uint32_t b = idx / width;
      const uint32_t c = idx - b * width;
      const int64_t p = clamp64(parents[b], 0, max_parent);
      vec_t v;
      v.load(token_ids + static_cast<uint32_t>(p) * width, c / 2u);
      if (c == col) v[0] = new_tokens[b];
      if (c + 1u == col) v[1] = new_tokens[b];
      v.store(out + b * width, c / 2u);
    }
    return;
  }
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    const uint32_t b = idx / width;
    const uint32_t c = idx - b * width;
    beam_expand_write(out, token_ids, parents, new_tokens, max_parent, width, col, b, c);
  }
}

void beam_expand_token_ids(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView token_ids,
    tvm::ffi::TensorView parents,
    tvm::ffi::TensorView new_tokens,
    int64_t col) {
  using namespace host;

  SymbolicSize Bw = {"bw"};
  SymbolicSize L = {"width"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({Bw, L})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(out)
      .verify(token_ids);
  TensorMatcher({Bw})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(parents)
      .verify(new_tokens);

  const uint32_t bw = static_cast<uint32_t>(Bw.unwrap());
  const uint32_t width = static_cast<uint32_t>(L.unwrap());
  RuntimeCheck(bw > 0 && width > 0, "beam_expand: empty");
  RuntimeCheck(col >= 0 && static_cast<uint32_t>(col) < width, "beam_expand: col OOB");

  const uint32_t n_items = ((width % 2u) == 0u) ? (bw * width / 2u) : (bw * width);
  const uint32_t grid = div_ceil(n_items, kBeamTrieBlock);
  LaunchKernel(grid, kBeamTrieBlock, device_.unwrap())(
      beam_expand_kernel,
      static_cast<int64_t*>(out.data_ptr()),
      static_cast<const int64_t*>(token_ids.data_ptr()),
      static_cast<const int64_t*>(parents.data_ptr()),
      static_cast<const int64_t*>(new_tokens.data_ptr()),
      static_cast<int64_t>(bw) - 1,
      bw,
      width,
      static_cast<uint32_t>(col));
}

// ---------------------------------------------------------------------------
// batched expand: token_ids/out [N, BW, L], parents/new_tokens [N, BW]
// ---------------------------------------------------------------------------
__global__ void beam_expand_batched_kernel(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ token_ids,
    const int64_t* __restrict__ parents,
    const int64_t* __restrict__ new_tokens,
    int64_t max_parent,
    uint32_t batch,
    uint32_t bw,
    uint32_t width,
    uint32_t col) {
  const uint32_t plane = bw * width;
  const uint32_t total = batch * plane;
  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    const uint32_t n = idx / plane;
    const uint32_t rem = idx - n * plane;
    const uint32_t b = rem / width;
    const uint32_t c = rem - b * width;
    const uint32_t base = n * plane;
    if (c == col) {
      out[idx] = new_tokens[n * bw + b];
    } else {
      const int64_t p = clamp64(parents[n * bw + b], 0, max_parent);
      out[idx] = token_ids[base + static_cast<uint32_t>(p) * width + c];
    }
  }
}

void beam_expand_token_ids_batched(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView token_ids,
    tvm::ffi::TensorView parents,
    tvm::ffi::TensorView new_tokens,
    int64_t col) {
  using namespace host;

  SymbolicSize N = {"batch"};
  SymbolicSize Bw = {"bw"};
  SymbolicSize L = {"width"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N, Bw, L})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(out)
      .verify(token_ids);
  TensorMatcher({N, Bw})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(parents)
      .verify(new_tokens);

  const uint32_t batch = static_cast<uint32_t>(N.unwrap());
  const uint32_t bw = static_cast<uint32_t>(Bw.unwrap());
  const uint32_t width = static_cast<uint32_t>(L.unwrap());
  RuntimeCheck(batch > 0 && bw > 0 && width > 0, "beam_expand_batched: empty");
  RuntimeCheck(col >= 0 && static_cast<uint32_t>(col) < width, "beam_expand_batched: col OOB");

  const uint32_t total = batch * bw * width;
  const uint32_t grid = div_ceil(total, kBeamTrieBlock);
  LaunchKernel(grid, kBeamTrieBlock, device_.unwrap())(
      beam_expand_batched_kernel,
      static_cast<int64_t*>(out.data_ptr()),
      static_cast<const int64_t*>(token_ids.data_ptr()),
      static_cast<const int64_t*>(parents.data_ptr()),
      static_cast<const int64_t*>(new_tokens.data_ptr()),
      static_cast<int64_t>(bw) - 1,
      batch,
      bw,
      width,
      static_cast<uint32_t>(col));
}

// ---------------------------------------------------------------------------
// GenRec fused step: cum+logprob mask → sorted top-K → optional expand/advance
//
// One CUDA block per request. Masked scores stay in dynamic smem when they
// fit (GenRec BW=50, C=100 → 20 KiB). Top-K merges BW descending lists with
// warp-shuffle reduce (1 syncthreads per pick when BW>32).
// ---------------------------------------------------------------------------
struct GenrecCand {
  float v;
  int idx;
  int b;
  int p;
  int valid;
};

SGL_DEVICE GenrecCand genrec_cand_better(GenrecCand a, GenrecCand b) {
  if (!a.valid) return b;
  if (!b.valid) return a;
  if (b.v > a.v || (b.v == a.v && b.idx < a.idx)) return b;
  return a;
}

SGL_DEVICE GenrecCand genrec_shfl_xor(GenrecCand c, int mask, uint32_t activemask) {
  GenrecCand o;
  o.v = __shfl_xor_sync(activemask, c.v, mask, 32);
  o.idx = __shfl_xor_sync(activemask, c.idx, mask, 32);
  o.b = __shfl_xor_sync(activemask, c.b, mask, 32);
  o.p = __shfl_xor_sync(activemask, c.p, mask, 32);
  o.valid = __shfl_xor_sync(activemask, c.valid, mask, 32);
  return o;
}

SGL_DEVICE GenrecCand genrec_warp_best(GenrecCand mine) {
  constexpr uint32_t kFull = 0xffffffffu;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    mine = genrec_cand_better(mine, genrec_shfl_xor(mine, mask, kFull));
  }
  return mine;
}

// Warp-shuffle reduce; only thread 0's return value is defined for n_warps>1.
SGL_DEVICE GenrecCand genrec_block_best(GenrecCand mine, GenrecCand* __restrict__ warp_best, uint32_t n_active) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5;
  const uint32_t n_warps = (n_active + 31u) >> 5;
  const GenrecCand wbest = genrec_warp_best(mine);
  if (n_warps <= 1u) {
    return wbest;
  }
  if (lane == 0u && warp < n_warps) {
    warp_best[warp] = wbest;
  }
  __syncthreads();
  if (tid == 0u) {
    GenrecCand acc = warp_best[0];
    for (uint32_t w = 1; w < n_warps; ++w) {
      acc = genrec_cand_better(acc, warp_best[w]);
    }
    warp_best[0] = acc;
  }
  __syncthreads();
  return warp_best[0];
}

extern __shared__ char genrec_dyn_smem[];

__global__ void genrec_mask_topk_expand_kernel(
    float* __restrict__ out_vals,              // [N, K]
    int64_t* __restrict__ out_parents,         // [N, K]
    int64_t* __restrict__ out_tokens,          // [N, K]
    int64_t* __restrict__ out_indices,         // [N, K] flat idx into BW*C
    int64_t* __restrict__ token_ids_out,       // [N, BW, L] or nullptr
    int64_t* __restrict__ node_ids_out,        // [N, BW] or nullptr
    float* __restrict__ scratch_scores,        // [N, BW*C]
    const float* __restrict__ cum,             // [N, BW]
    const float* __restrict__ top_logprobs,    // [N, BW, C]
    const int64_t* __restrict__ top_tokens,    // [N, BW, C]
    const int64_t* __restrict__ node_ids_in,   // [N, BW] or nullptr
    const uint8_t* __restrict__ allow_table,   // [n_nodes, vsz] or nullptr
    const int64_t* __restrict__ token_ids_in,  // [N, BW, L] or nullptr
    const int64_t* __restrict__ next_node,     // [n_nodes, vsz] or nullptr
    const uint8_t* __restrict__ do_expand,     // [N] or nullptr (=all)
    const int32_t* __restrict__ col_n,         // [N] per-row write column
    int64_t token_base,
    int64_t n_nodes,
    int64_t vsz,
    int64_t invalid_node,
    float neg_inf,
    uint32_t N,
    uint32_t BW,
    uint32_t C,
    uint32_t K,
    uint32_t L,
    int apply_mask,
    int apply_expand,
    int apply_advance,
    int use_smem_scores) {
  const uint32_t n = blockIdx.x;
  if (n >= N) return;
  uint32_t col = 0;
  if (col_n != nullptr) {
    col = static_cast<uint32_t>(col_n[n] < 0 ? 0 : col_n[n]);
  }

  const uint32_t M = BW * C;
  float* scores =
      use_smem_scores ? reinterpret_cast<float*>(genrec_dyn_smem) : (scratch_scores + static_cast<size_t>(n) * M);
  const float* cum_n = cum + static_cast<size_t>(n) * BW;
  const float* lp_n = top_logprobs + static_cast<size_t>(n) * M;
  const int64_t* tok_n = top_tokens + static_cast<size_t>(n) * M;
  const int64_t* nodes_n = (node_ids_in != nullptr) ? (node_ids_in + static_cast<size_t>(n) * BW) : nullptr;

  __shared__ float cum_s[kBeamTrieBlock];
  __shared__ int64_t nodes_s[kBeamTrieBlock];
  __shared__ GenrecCand warp_best[kBeamTrieBlock / 32];
  __shared__ int64_t sel_parent[kBeamTrieBlock];
  __shared__ int64_t sel_token[kBeamTrieBlock];
  __shared__ int win_b;
  __shared__ int win_p;

  if (threadIdx.x < BW) {
    cum_s[threadIdx.x] = cum_n[threadIdx.x];
    nodes_s[threadIdx.x] = (nodes_n != nullptr) ? nodes_n[threadIdx.x] : static_cast<int64_t>(0);
  }
  __syncthreads();

  const bool do_mask = apply_mask && allow_table != nullptr && nodes_n != nullptr;

  if (do_mask) {
    using f4_t = device::AlignedVector<float, 4>;
    const uint32_t n_vecs = M / 4u;
    for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += blockDim.x) {
      const uint32_t i0 = vi * 4u;
      f4_t v;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const uint32_t i = i0 + static_cast<uint32_t>(t);
        const uint32_t parent = i / C;
        float s = cum_s[parent] + lp_n[i];
        if (!trie_cand_ok(nodes_s[parent], tok_n[i], allow_table, token_base, n_nodes, vsz)) {
          s = neg_inf;
        }
        v[t] = s;
      }
      v.store(scores, vi);
    }
    for (uint32_t i = n_vecs * 4u + threadIdx.x; i < M; i += blockDim.x) {
      const uint32_t parent = i / C;
      float s = cum_s[parent] + lp_n[i];
      if (!trie_cand_ok(nodes_s[parent], tok_n[i], allow_table, token_base, n_nodes, vsz)) {
        s = neg_inf;
      }
      scores[i] = s;
    }
    __syncthreads();
  }

  int my_p = 0;
  for (uint32_t k = 0; k < K; ++k) {
    GenrecCand mine;
    mine.v = neg_inf;
    mine.idx = 0;
    mine.b = 0;
    mine.p = 0;
    mine.valid = 0;

    if (threadIdx.x < BW) {
      const uint32_t b = threadIdx.x;
      int p = my_p;
      if (do_mask) {
        while (p < static_cast<int>(C) && scores[b * C + static_cast<uint32_t>(p)] == neg_inf) {
          ++p;
        }
      }
      my_p = p;
      if (p < static_cast<int>(C)) {
        const int idx = static_cast<int>(b * C) + p;
        const float v = do_mask ? scores[static_cast<uint32_t>(idx)] : (cum_s[b] + lp_n[static_cast<uint32_t>(idx)]);
        mine.v = v;
        mine.idx = idx;
        mine.b = static_cast<int>(b);
        mine.p = p;
        mine.valid = 1;
      }
    }

    const GenrecCand best = genrec_block_best(mine, warp_best, BW);
    if (threadIdx.x == 0) {
      float out_v = neg_inf;
      int best_b = 0;
      int best_i = 0;
      int best_p = 0;
      if (best.valid) {
        out_v = best.v;
        best_b = best.b;
        best_i = best.idx;
        best_p = best.p;
      }
      const size_t ok = static_cast<size_t>(n) * K + k;
      out_vals[ok] = out_v;
      out_parents[ok] = static_cast<int64_t>(best_b);
      out_tokens[ok] = (M > 0) ? tok_n[best_i] : static_cast<int64_t>(0);
      out_indices[ok] = static_cast<int64_t>(best_i);
      sel_parent[k] = static_cast<int64_t>(best_b);
      sel_token[k] = (M > 0) ? tok_n[best_i] : static_cast<int64_t>(0);
      win_b = best.valid ? best_b : -1;
      win_p = best_p;
    }
    __syncthreads();
    if (win_b >= 0 && threadIdx.x == static_cast<uint32_t>(win_b)) {
      my_p = win_p + 1;
    }
  }

  const bool expand_this = apply_expand && token_ids_in != nullptr && token_ids_out != nullptr &&
                           (do_expand == nullptr || do_expand[n] != 0);
  if (!expand_this) return;

  const int64_t* tin = token_ids_in + static_cast<size_t>(n) * BW * L;
  int64_t* tout = token_ids_out + static_cast<size_t>(n) * BW * L;
  const int64_t max_parent = static_cast<int64_t>(BW) - 1;
  const uint32_t plane = BW * L;

  if ((L % 2u) == 0u) {
    using i2_t = device::AlignedVector<int64_t, 2>;
    const uint32_t n_vecs = plane / 2u;
    for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += blockDim.x) {
      const uint32_t idx = vi * 2u;
      const uint32_t b = idx / L;
      const uint32_t c = idx - b * L;
      int64_t p;
      if (b >= K) {
        p = static_cast<int64_t>(b);
      } else {
        p = clamp64(sel_parent[b], 0, max_parent);
      }
      i2_t v;
      v.load(tin + static_cast<uint32_t>(p) * L, c / 2u);
      if (b < K) {
        if (col < L && c == col) v[0] = sel_token[b];
        if (col < L && c + 1u == col) v[1] = sel_token[b];
      }
      v.store(tout + b * L, c / 2u);
    }
  } else {
    for (uint32_t idx = threadIdx.x; idx < plane; idx += blockDim.x) {
      const uint32_t b = idx / L;
      const uint32_t c = idx - b * L;
      if (b >= K) {
        tout[idx] = tin[idx];
        continue;
      }
      if (col < L && c == col) {
        tout[idx] = sel_token[b];
      } else {
        const int64_t p = clamp64(sel_parent[b], 0, max_parent);
        tout[idx] = (p == static_cast<int64_t>(b)) ? tin[idx] : tin[static_cast<uint32_t>(p) * L + c];
      }
    }
  }

  if (apply_advance && next_node != nullptr && node_ids_in != nullptr && node_ids_out != nullptr) {
    __syncthreads();
    for (uint32_t b = threadIdx.x; b < K; b += blockDim.x) {
      const int64_t p = clamp64(sel_parent[b], 0, max_parent);
      const int64_t old = nodes_s[p];
      node_ids_out[static_cast<size_t>(n) * BW + b] =
          trie_transition(old, sel_token[b], next_node, token_base, n_nodes, vsz, invalid_node);
    }
    for (uint32_t b = K + threadIdx.x; b < BW; b += blockDim.x) {
      node_ids_out[static_cast<size_t>(n) * BW + b] = nodes_s[b];
    }
  }
}

void genrec_mask_topk_expand(
    tvm::ffi::TensorView out_vals,
    tvm::ffi::TensorView out_parents,
    tvm::ffi::TensorView out_tokens,
    tvm::ffi::TensorView out_indices,
    tvm::ffi::TensorView scratch_scores,
    tvm::ffi::TensorView cum,
    tvm::ffi::TensorView top_logprobs,
    tvm::ffi::TensorView top_tokens,
    tvm::ffi::TensorView node_ids_in,
    tvm::ffi::TensorView allow_table,
    tvm::ffi::TensorView token_ids_in,
    tvm::ffi::TensorView token_ids_out,
    tvm::ffi::TensorView next_node,
    tvm::ffi::TensorView node_ids_out,
    tvm::ffi::TensorView do_expand,
    tvm::ffi::TensorView col_n,
    int64_t token_base,
    int64_t invalid_node,
    int64_t apply_mask,
    int64_t apply_expand,
    int64_t apply_advance) {
  using namespace host;

  SymbolicSize N = {"batch"};
  SymbolicSize Bw = {"bw"};
  SymbolicSize C = {"cand"};
  SymbolicSize K = {"select_k"};
  SymbolicSize M = {"flat"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N, K})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(out_vals);
  TensorMatcher({N, K})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(out_parents)
      .verify(out_tokens)
      .verify(out_indices);
  TensorMatcher({N, Bw})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(cum);
  TensorMatcher({N, Bw, C})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(top_logprobs);
  TensorMatcher({N, Bw, C})  //
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(top_tokens);
  TensorMatcher({N, M})  //
      .with_dtype<float>()
      .with_device<kDLCUDA>(device_)
      .verify(scratch_scores);

  const uint32_t batch = static_cast<uint32_t>(N.unwrap());
  const uint32_t bw = static_cast<uint32_t>(Bw.unwrap());
  const uint32_t cand = static_cast<uint32_t>(C.unwrap());
  const uint32_t select_k = static_cast<uint32_t>(K.unwrap());
  RuntimeCheck(batch > 0 && bw > 0 && cand > 0 && select_k > 0, "genrec_fused: empty");
  RuntimeCheck(select_k <= bw, "genrec_fused: select_k > bw");
  RuntimeCheck(bw <= kBeamTrieBlock, "genrec_fused: bw too large for list_ptr smem");
  RuntimeCheck(select_k <= kBeamTrieBlock, "genrec_fused: select_k too large");
  RuntimeCheck(
      static_cast<uint64_t>(bw) * cand == static_cast<uint64_t>(M.unwrap()), "genrec_fused: scratch flat != bw*cand");

  const int64_t* node_ids_in_i64 = nullptr;
  const uint8_t* allow_ptr = nullptr;
  int64_t n_nodes = 0;
  int64_t vsz = 0;
  if (apply_mask) {
    SymbolicSize Nn = {"n_nodes"};
    SymbolicSize V = {"vsz"};
    TensorMatcher({N, Bw})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(node_ids_in);
    TensorMatcher({Nn, V})  //
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(allow_table);
    n_nodes = Nn.unwrap();
    vsz = V.unwrap();
    RuntimeCheck(n_nodes > 0 && vsz > 0, "genrec_fused: empty allow_table");
    node_ids_in_i64 = static_cast<const int64_t*>(node_ids_in.data_ptr());
    allow_ptr = static_cast<const uint8_t*>(allow_table.data_ptr());
  }

  const int64_t* token_ids_in_ptr = nullptr;
  int64_t* token_ids_out_ptr = nullptr;
  uint32_t width = 1;
  const uint8_t* do_expand_ptr = nullptr;
  if (apply_expand) {
    SymbolicSize L = {"width"};
    TensorMatcher({N, Bw, L})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(token_ids_in)
        .verify(token_ids_out);
    width = static_cast<uint32_t>(L.unwrap());
    RuntimeCheck(width > 0, "genrec_fused: width=0");
    token_ids_in_ptr = static_cast<const int64_t*>(token_ids_in.data_ptr());
    token_ids_out_ptr = static_cast<int64_t*>(token_ids_out.data_ptr());

    TensorMatcher({N})  //
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(do_expand);
    do_expand_ptr = static_cast<const uint8_t*>(do_expand.data_ptr());
    TensorMatcher({N})  //
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(col_n);
  }

  const int64_t* next_node_ptr = nullptr;
  int64_t* node_ids_out_ptr = nullptr;
  if (apply_advance) {
    RuntimeCheck(apply_mask != 0, "genrec_fused: advance needs mask inputs");
    SymbolicSize Nn2 = {"n_nodes2"};
    SymbolicSize V2 = {"vsz2"};
    TensorMatcher({Nn2, V2})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(next_node);
    TensorMatcher({N, Bw})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(node_ids_out);
    RuntimeCheck(Nn2.unwrap() == n_nodes && V2.unwrap() == vsz, "genrec_fused: next_node shape");
    next_node_ptr = static_cast<const int64_t*>(next_node.data_ptr());
    node_ids_out_ptr = static_cast<int64_t*>(node_ids_out.data_ptr());
  }

  const int32_t* col_ptr = nullptr;
  if (apply_expand) {
    col_ptr = static_cast<const int32_t*>(col_n.data_ptr());
  }

  const size_t score_bytes = static_cast<size_t>(bw) * cand * sizeof(float);
  const int use_smem_scores = (apply_mask != 0 && score_bytes > 0 && score_bytes <= kMaxSmemScoresBytes) ? 1 : 0;
  const size_t dyn_smem = use_smem_scores ? ((score_bytes + 15u) & ~size_t{15}) : 0;

  LaunchKernel(batch, kBeamTrieBlock, device_.unwrap(), dyn_smem)(
      genrec_mask_topk_expand_kernel,
      static_cast<float*>(out_vals.data_ptr()),
      static_cast<int64_t*>(out_parents.data_ptr()),
      static_cast<int64_t*>(out_tokens.data_ptr()),
      static_cast<int64_t*>(out_indices.data_ptr()),
      token_ids_out_ptr,
      node_ids_out_ptr,
      static_cast<float*>(scratch_scores.data_ptr()),
      static_cast<const float*>(cum.data_ptr()),
      static_cast<const float*>(top_logprobs.data_ptr()),
      static_cast<const int64_t*>(top_tokens.data_ptr()),
      node_ids_in_i64,
      allow_ptr,
      token_ids_in_ptr,
      next_node_ptr,
      do_expand_ptr,
      col_ptr,
      token_base,
      n_nodes,
      vsz,
      invalid_node,
      -std::numeric_limits<float>::infinity(),
      batch,
      bw,
      cand,
      select_k,
      width,
      static_cast<int>(apply_mask),
      static_cast<int>(apply_expand),
      static_cast<int>(apply_advance),
      use_smem_scores);
}

}  // namespace
