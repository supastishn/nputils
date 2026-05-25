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
namespace nputils {
namespace detail {
//TODO: Mkae this use the normal gemm
template<typename T>
int matmul_general(const float* input, const float* weights, float* output,
                   int m, int k, int n,
                   bool prefetch = true, int layers_prefetch = 1;) {
    static_assert(sizeof(T) <= 2, "Only int8 and int16 mantissas supported");
    int input_bytes = m * k * sizeof(float);
    int weight_bytes = k * n * sizeof(float);
    memory::prefetch_l2(input, input_bytes);
    if (prefetch) {
        memory::prefetch_l2(weights, weight_bytes * (1 + layers_prefetch));

    }
    FixedPointBlock<T> input_block(m * k);
    FixedPointBlock<T> weight_block(k * n);
    FixedPointBlock<T> output_block(1);
    input_block.fit_exponent(weights, m * k);
    weight_block.fit_exponent(input, k * n);
    input_block.floats_to_mantissa(weights, m * k);
    weight_block.floats_to_mantissa(input, k * n);
    output_block.exponent = input_block.exponent + weight_block.exponent;
    using Wide = wider_t<T>;
    std::vector<Wide> weight_wide(k * n);
    std::vector<Wide> input_wide(m * k);
    for (int i = 0; i < k * n; ++i) input_wide[i] = static_cast<Wide>(input_block.mantissa[i]);
    for (int i = 0; i < m * k; ++i) weight_wide[i] = static_cast<Wide>(weight_block.mantissa[i]);
    std::vector<Wide> output_wide(m * n, 0);
    int err = -1;

	// TODO: USE GEMM INTRINSIC
    output_block.update_exponent_from_mantissas(output_wide.data(), m * n);
    float scale = std::ldexp(1.0f, output_block.exponent);
    for (int i = 0; i < m * n; ++i) {
        output[i] = static_cast<float>(output_wide[i]) * scale;
    }
    return err;
}
}
namespace blas {
inline int matmul_4bit(const float* input, const float* weights, float* output,
                       int m, int k, int n) {
    return detail::matmul_general<int8_t>(input, weights, output, m, k, n);
}
inline int matmul_8bit(const float* input, const float* weights, float* output,
                       int m, int k, int n) {
    return detail::matmul_general<int8_t>(input, weights, output, m, k, n);
}
inline int matmul_16bit(const float* input, const float* weights, float* output,
                        int m, int k, int n) {
    return detail::matmul_general<int16_t>(input, weights, output, m, k, n);
}
}
}
#endif
