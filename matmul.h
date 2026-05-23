#ifndef MATMUL
#define MATMUL
#include "qhblas_hvx.h"
#include "math_utils.h"
#include "hexagon_protos.h"
#include <vector>
#include <cmath>
#include <type_traits>
namespace nputils {
namespace memory {
inline int prefetch_l2(const void* ptr, int bytes) {
    const int LINE = 128;
    const int MAX_HEIGHT = 255;
    const int MAX_FETCH = 256 * 1024;
    const uint8_t* addr = (const uint8_t*)ptr;
    int total_fetched = 0;
    if (bytes > MAX_FETCH) bytes = MAX_FETCH;
    while (bytes > 0) {
        int blocks = (bytes + LINE - 1) / LINE;
        if (blocks > MAX_HEIGHT) blocks = MAX_HEIGHT;
        uint32_t ctrl = blocks | (LINE << 8) | (LINE << 16);
        Q6_l2fetch_AR((void*)addr, ctrl);
        int fetched = blocks * LINE;
        addr += fetched;
        bytes -= fetched;
        total_fetched += fetched;
    }
    return total_fetched;
}
}
namespace detail {
template<typename T>
int matmul_general(const float* input, const float* weights, float* output,
                   int m, int k, int n,
                   bool prefetch_start = false, bool prefetch_next = true) {
    static_assert(sizeof(T) <= 2, "Only int8 and int16 mantissas supported");
    int weight_bytes = m * k * sizeof(float);
    int input_bytes = k * n * sizeof(float);
    if (prefetch_start) {
        memory::prefetch_l2(weights, weight_bytes);
    }
    FixedPointBlock<T> weight_block(m * k);
    FixedPointBlock<T> input_block(k * n);
    FixedPointBlock<T> output_block(1);
    weight_block.fit_exponent(weights, m * k);
    input_block.fit_exponent(input, k * n);
    weight_block.floats_to_mantissa(weights, m * k);
    input_block.floats_to_mantissa(input, k * n);
    output_block.exponent = input_block.exponent + weight_block.exponent;
    using Wide = wider_t<T>;
    std::vector<Wide> input_wide(k * n);
    std::vector<Wide> weight_wide(m * k);
    for (int i = 0; i < k * n; ++i) input_wide[i] = static_cast<Wide>(input_block.mantissa[i]);
    for (int i = 0; i < m * k; ++i) weight_wide[i] = static_cast<Wide>(weight_block.mantissa[i]);
    std::vector<Wide> output_wide(m * n, 0);
    int err = -1;
    if constexpr (sizeof(Wide) == 2) {
        err = qhblas_hvx_matrix_matrix_mpy_ah(
            (const int16_t*)input_wide.data(),
            (const int16_t*)weight_wide.data(),
            (int16_t*)output_wide.data(),
            m, k, n);
    } else if constexpr (sizeof(Wide) == 4) {
        err = qhblas_hvx_matrix_matrix_mpy_aw(
            (const int32_t*)input_wide.data(),
            (const int32_t*)weight_wide.data(),
            (int32_t*)output_wide.data(),
            m, k, n);
    }
    output_block.update_exponent_from_mantissas(output_wide.data(), m * n);
    float scale = std::ldexp(1.0f, output_block.exponent);
    for (int i = 0; i < m * n; ++i) {
        output[i] = static_cast<float>(output_wide[i]) * scale;
    }
    if (prefetch_next) {
        memory::prefetch_l2(weights + m * k, weight_bytes);
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