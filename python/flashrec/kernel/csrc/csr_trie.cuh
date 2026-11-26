// Standard CSR and the two-kernel mask -> Top-K/expand pipeline.
// Adapted from trie-op-learn/exercises/{sparse_trie,csr_mask,opt}.cuh.
#pragma once
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cmath>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <stdexcept>

namespace flashrec_csr {
inline cudaStream_t stream(tvm::ffi::TensorView t) {
  const auto d = t.device();
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(d.device_type, d.device_id));
}
inline void check_launch() {
  const auto e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Trie {
  const int32_t* row;
  const int64_t* col;  // Absolute token IDs, sorted within each row.
  const int64_t* child;
  int64_t nodes, invalid;
  __device__ int64_t edge(int64_t node, int64_t token) const {
    node = node < 0 ? 0 : (node >= nodes ? nodes - 1 : node);
    int64_t lo = row[node], hi = row[node + 1], end = hi;
    while (lo < hi) {
      const int64_t mid = lo + (hi - lo) / 2;
      if (col[mid] < token)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo < end && col[lo] == token ? lo : -1;
  }
  __device__ bool allowed(int64_t node, int64_t token) const {
    return node < 0 || edge(node, token) >= 0;
  }
  __device__ int64_t next(int64_t node, int64_t token) const {
    if (node < 0) return node;
    const int64_t e = edge(node, token);
    return e < 0 ? invalid : child[e];
  }
};

__global__ void
lookup_kernel(Trie trie, const int64_t* nodes, const int64_t* tokens, bool* found, int64_t* children, int64_t count) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
    const int64_t n = nodes[i], e = trie.edge(n, tokens[i]);
    found[i] = n < 0 || e >= 0;
    children[i] = n < 0 ? n : e < 0 ? trie.invalid : trie.child[e];
  }
}

// One block per beam, as in trie-op-learn's split CSR mask.
__global__ void mask_kernel(
    Trie trie, const float* cum, const float* lp, const int64_t* tokens, const int64_t* nodes, float* scores, int C) {
  const int64_t beam = blockIdx.x;
  const int64_t node = nodes[beam];
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const int64_t i = beam * C + c;
    scores[i] = trie.allowed(node, tokens[i]) ? cum[beam] + lp[i] : -INFINITY;
  }
}

// Each sorted beam contributes at most its first 50 finite legal items.
// A warp compacts holes, then CUB selects with score-desc/index-asc ordering.
// -inf remains padding: normalize parents/tokens/indices to survivor 0 just
// like FlashRec's repeat_best_legal, rather than emitting an illegal token.
__global__ void finish50_kernel(
    Trie trie,
    const float* scores,
    const int64_t* candidates,
    const int64_t* nodes,
    const int64_t* tin,
    const int32_t* cols,
    const uint8_t* expand,
    float* values,
    int64_t* parents,
    int64_t* tokens,
    int64_t* indices,
    int64_t* tout,
    int64_t* nout,
    int C,
    int L) {
  using Sort = cub::BlockRadixSort<unsigned long long, 512, 5>;
  __shared__ union {
    int index[2500];
    Sort::TempStorage sort;
  } scratch;
  __shared__ int64_t selected_parent[50], selected_token[50];
  __shared__ int best_index;
  const int n = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int M = 50 * C;
  const float* plane = scores + int64_t(n) * M;
  const int64_t* cand = candidates + int64_t(n) * M;
  for (int beam = warp; beam < 50; beam += 16) {
    int count = 0;
    for (int base = 0; base < C && count < 50; base += 32) {
      const int c = base + lane, index = beam * C + c;
      const bool valid = c < C && plane[index] != -INFINITY && !isnan(plane[index]);
      const unsigned mask = __ballot_sync(0xffffffffu, valid);
      const int rank = __popc(mask & ((1u << lane) - 1u));
      if (valid && count + rank < 50) scratch.index[beam * 50 + count + rank] = index;
      count += __popc(mask);
    }
    for (int k = count + lane; k < 50; k += 32)
      scratch.index[beam * 50 + k] = -1;
  }
  __syncthreads();
  unsigned long long keys[5];
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int p = tid * 5 + j, index = p < 2500 ? scratch.index[p] : -1;
    unsigned long long key = 0;
    if (index >= 0) {
      const float score = plane[index];
      const unsigned bits = __float_as_uint(score == 0.f ? 0.f : score);
      const unsigned ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
      key = (uint64_t(ordered) << 32) | (0xffffffffu - unsigned(index));
    }
    keys[j] = key;
  }
  __syncthreads();  // CUB reuses the index storage only after all reads.
  Sort(scratch.sort).SortDescending(keys);
  if (tid == 0) {
    const int first = keys[0] ? int(0xffffffffu - unsigned(keys[0])) : 0;
    best_index = keys[0] && isfinite(plane[first]) ? first : 0;
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int k = tid * 5 + j;
    if (k < 50) {
      const bool valid = keys[j] != 0;
      const int raw = valid ? int(0xffffffffu - unsigned(keys[j])) : 0;
      const int index = valid && isfinite(plane[raw]) ? raw : best_index;
      const int64_t parent = index / C, token = cand[index], out = int64_t(n) * 50 + k;
      values[out] = valid ? plane[raw] : -INFINITY;
      parents[out] = selected_parent[k] = parent;
      tokens[out] = selected_token[k] = token;
      indices[out] = index;
    }
  }
  __syncthreads();
  const bool do_expand = expand[n] != 0;
  for (int i = tid; i < 50 * L; i += blockDim.x) {
    const int b = i / L, c = i % L;
    const int parent = int(selected_parent[b]);
    if (do_expand)
      tout[int64_t(n) * 50 * L + i] = c == cols[n] ? selected_token[b] : tin[(int64_t(n) * 50 + parent) * L + c];
  }
  for (int b = tid; b < 50; b += blockDim.x) {
    const int64_t old = nodes[int64_t(n) * 50 + selected_parent[b]];
    if (do_expand) nout[int64_t(n) * 50 + b] = trie.next(old, selected_token[b]);
  }
}

__global__ void expand_kernel(
    Trie trie,
    const int64_t* candidates,
    const int64_t* nodes,
    const int64_t* tin,
    const int32_t* cols,
    const uint8_t* expand,
    const float* values,
    int64_t* parents,
    int64_t* tokens,
    int64_t* indices,
    int64_t* tout,
    int64_t* nout,
    int BW,
    int C,
    int K,
    int L) {
  const int n = blockIdx.x;
  const int64_t out = int64_t(n) * K;
  const bool live = isfinite(values[out]);
  const int64_t bp = live ? parents[out] : 0;
  const int64_t bt = live ? tokens[out] : candidates[int64_t(n) * BW * C];
  const int64_t bi = live ? indices[out] : 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    if (!isfinite(values[out + k])) {
      parents[out + k] = bp;
      tokens[out + k] = bt;
      indices[out + k] = bi;
    }
  }
  __syncthreads();
  if (!expand[n]) return;
  for (int i = threadIdx.x; i < BW * L; i += blockDim.x) {
    const int b = i / L, c = i % L;
    const int64_t p = b < K ? parents[out + b] : b;
    tout[int64_t(n) * BW * L + i] = b < K && c == cols[n] ? tokens[out + b] : tin[(int64_t(n) * BW + p) * L + c];
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    nout[int64_t(n) * BW + k] = trie.next(nodes[int64_t(n) * BW + parents[out + k]], tokens[out + k]);
}

inline Trie view(tvm::ffi::TensorView row, tvm::ffi::TensorView col, tvm::ffi::TensorView child, int64_t invalid) {
  return {
      static_cast<const int32_t*>(row.data_ptr()),
      static_cast<const int64_t*>(col.data_ptr()),
      static_cast<const int64_t*>(child.data_ptr()),
      row.numel() - 1,
      invalid};
}
}  // namespace flashrec_csr

void csr_lookup(
    tvm::ffi::TensorView row,
    tvm::ffi::TensorView col,
    tvm::ffi::TensorView child,
    int64_t invalid,
    tvm::ffi::TensorView nodes,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView found,
    tvm::ffi::TensorView out) {
  flashrec_csr::lookup_kernel<<<(nodes.numel() + 255) / 256, 256, 0, flashrec_csr::stream(out)>>>(
      flashrec_csr::view(row, col, child, invalid),
      static_cast<const int64_t*>(nodes.data_ptr()),
      static_cast<const int64_t*>(tokens.data_ptr()),
      static_cast<bool*>(found.data_ptr()),
      static_cast<int64_t*>(out.data_ptr()),
      nodes.numel());
  flashrec_csr::check_launch();
}

void csr_mask(
    tvm::ffi::TensorView row,
    tvm::ffi::TensorView col,
    tvm::ffi::TensorView child,
    int64_t invalid,
    tvm::ffi::TensorView cum,
    tvm::ffi::TensorView lp,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView nodes,
    tvm::ffi::TensorView scores) {
  flashrec_csr::mask_kernel<<<cum.numel(), 128, 0, flashrec_csr::stream(scores)>>>(
      flashrec_csr::view(row, col, child, invalid),
      static_cast<const float*>(cum.data_ptr()),
      static_cast<const float*>(lp.data_ptr()),
      static_cast<const int64_t*>(tokens.data_ptr()),
      static_cast<const int64_t*>(nodes.data_ptr()),
      static_cast<float*>(scores.data_ptr()),
      int(lp.size(-1)));
  flashrec_csr::check_launch();
}

void csr_finish(
    tvm::ffi::TensorView row,
    tvm::ffi::TensorView col,
    tvm::ffi::TensorView child,
    int64_t invalid,
    tvm::ffi::TensorView scores,
    tvm::ffi::TensorView candidates,
    tvm::ffi::TensorView nodes,
    tvm::ffi::TensorView tin,
    tvm::ffi::TensorView cols,
    tvm::ffi::TensorView expand,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView parents,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView tout,
    tvm::ffi::TensorView nout,
    bool sort50) {
  const auto trie = flashrec_csr::view(row, col, child, invalid);
  if (sort50) {
    if (nodes.size(1) != 50 || values.size(1) != 50) throw std::invalid_argument("CSR sort50 requires BW=K=50");
    flashrec_csr::finish50_kernel<<<nodes.size(0), 512, 0, flashrec_csr::stream(tout)>>>(
        trie,
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const int64_t*>(candidates.data_ptr()),
        static_cast<const int64_t*>(nodes.data_ptr()),
        static_cast<const int64_t*>(tin.data_ptr()),
        static_cast<const int32_t*>(cols.data_ptr()),
        static_cast<const uint8_t*>(expand.data_ptr()),
        static_cast<float*>(values.data_ptr()),
        static_cast<int64_t*>(parents.data_ptr()),
        static_cast<int64_t*>(tokens.data_ptr()),
        static_cast<int64_t*>(indices.data_ptr()),
        static_cast<int64_t*>(tout.data_ptr()),
        static_cast<int64_t*>(nout.data_ptr()),
        int(candidates.size(-1)),
        int(tin.size(-1)));
  } else {
    flashrec_csr::expand_kernel<<<nodes.size(0), 256, 0, flashrec_csr::stream(tout)>>>(
        trie,
        static_cast<const int64_t*>(candidates.data_ptr()),
        static_cast<const int64_t*>(nodes.data_ptr()),
        static_cast<const int64_t*>(tin.data_ptr()),
        static_cast<const int32_t*>(cols.data_ptr()),
        static_cast<const uint8_t*>(expand.data_ptr()),
        static_cast<const float*>(values.data_ptr()),
        static_cast<int64_t*>(parents.data_ptr()),
        static_cast<int64_t*>(tokens.data_ptr()),
        static_cast<int64_t*>(indices.data_ptr()),
        static_cast<int64_t*>(tout.data_ptr()),
        static_cast<int64_t*>(nout.data_ptr()),
        int(nodes.size(1)),
        int(candidates.size(-1)),
        int(values.size(1)),
        int(tin.size(-1)));
  }
  flashrec_csr::check_launch();
}
