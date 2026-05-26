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
int matmul_general(const float* input, const float* weights, float* output,
                   int m, int k, int n,
                   bool prefetch = true, int layers_prefetch = 1) {
    static_assert(sizeof(T) <= 2, "Only int8 and int16 mantissas supported");
    static_assert(std::is_same<T, int8_t>::value, "Currently only int8_t supported with gemm intrinsic");
    int input_bytes = m * k * sizeof(float);
    int weight_bytes = k * n * sizeof(float);
    memory::prefetch_l2(input, input_bytes);
    if (prefetch) {
        memory::prefetch_l2(weights, weight_bytes * (1 + layers_prefetch));
    }
    FixedPointBlock<T> input_block(m * k);
    FixedPointBlock<T> weight_block(k * n);
    input_block.fit_exponent(input, m * k);
    weight_block.fit_exponent(weights, k * n);
    input_block.floats_to_mantissa(input, m * k);
    weight_block.floats_to_mantissa(weights, k * n);
    int padded_n = (n + 63) & ~63;
    int8_t* packed_weights = (int8_t*)Scratchpad::alloc(k * padded_n * sizeof(int8_t));
    preprocess::chunkCrouton(weight_block.mantissa, packed_weights, k, n);
    int32_t* output_mantissa = (int32_t*)Scratchpad::alloc(m * padded_n * sizeof(int32_t));
    int err = blas_ll::gemm_int8_int32(input_block.mantissa, packed_weights, output_mantissa, m, k, padded_n);
    if (err != 0) {
        Scratchpad::free(packed_weights);
        Scratchpad::free(output_mantissa);
        return err;
    }
    float scale = std::ldexp(1.0f, input_block.exponent + weight_block.exponent);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            output[i * n + j] = static_cast<float>(output_mantissa[i * padded_n + j]) * scale;
        }
    }
    Scratchpad::free(packed_weights);
    Scratchpad::free(output_mantissa);
    return 0;
}
} // namespace detail
namespace blas {
inline int matmul_8bit(const float* input, const float* weights, float* output,
                       int m, int k, int n) {
    return detail::matmul_general<int8_t>(input, weights, output, m, k, n);
}
} // namespace blas
} // namespace nputils
#endif