#ifndef MATMUL
#define MATMUL
#include "qhblas_hvx.h"
#include "math_utils.h"
#include "hexagon_protos.h"
#include <vector>
#include <cmath>
#include <type_traits>
#include <cstdint>
#include <algorithm>
#include "utils.h"
#include "gemm_instrics.h"
namespace nputils {
namespace detail {
template<typename T>
int matmul_general(const FixedPointBlock<T>& input_block,
                   const FixedPointBlock<T>& weight_block,
                   float* output,
                   int m,
                   int k,
                   int padded_n,
                   bool swapped) {
    static_assert(sizeof(T) <= 2, "Only int8 and int16 mantissas supported");
    static_assert(std::is_same<T, int8_t>::value, "Currently only int8_t supported with gemm intrinsic");
    int32_t* output_mantissa = (int32_t*)Scratchpad::alloc(m * padded_n * sizeof(int32_t));
    int err = blas_ll::gemm_int8_int32(input_block.mantissa, weight_block.mantissa, output_mantissa, m, k, padded_n);
    if (err != 0) {
        Scratchpad::free(output_mantissa);
        return err;
    }
    float scale = std::ldexp(1.0f, input_block.exponent + weight_block.exponent);
    if (swapped) {
        for (int j = 0; j < padded_n; ++j) {
            for (int i = 0; i < m; ++i) {
                output[j * m + i] = static_cast<float>(output_mantissa[i * padded_n + j]) * scale;
            }
        }
    } else {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < padded_n; ++j) {
                output[i * padded_n + j] = static_cast<float>(output_mantissa[i * padded_n + j]) * scale;
            }
        }
    }
    Scratchpad::free(output_mantissa);
    return 0;
}
} // namespace detail
namespace blas {
inline int matmul(const float* input, const float* weights, float* output,
                  int m, int k, int n) {
    const float* A = input;
    const float* B = weights;
    int M = m;
    int N = n;
    bool swapped = false;
    if (k * n <= k * m && m % 32 <= n % 32) {
        std::swap(A, B);
        std::swap(M, N);
        swapped = true;
    }
    memory::prefetch_l2(A, M * k * sizeof(float));
    memory::prefetch_l2(B, k * N * sizeof(float));
    FixedPointBlock<int8_t> input_block(M * k);
    FixedPointBlock<int8_t> weight_block(k * N);
    input_block.fit_exponent(A, M * k);
    weight_block.fit_exponent(B, k * N);
    input_block.floats_to_mantissa(A, M * k);
    weight_block.floats_to_mantissa(B, k * N);
    int padded_n = (N + 63) & ~63;
    int8_t* packed_weights = (int8_t*)Scratchpad::alloc(k * padded_n * sizeof(int8_t));
    preprocess::chunkCrouton(weight_block.mantissa, packed_weights, k, N);
    FixedPointBlock<int8_t> packed_weight_block;
    packed_weight_block.mantissa = packed_weights;
    packed_weight_block.exponent = weight_block.exponent;
    int err = detail::matmul_general<int8_t>(input_block, packed_weight_block, output, M, k, padded_n, swapped);
    Scratchpad::free(packed_weights);
    return err;
}
} // namespace blas
} // namespace nputils
#endif