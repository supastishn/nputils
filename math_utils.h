#ifndef MATH_UTILS
#define MATH_UTILS
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
namespace nputils {
template <typename T>
struct wider;
template <> struct wider<int8_t>  { using type = int16_t; };
template <> struct wider<int16_t> { using type = int32_t; };
template <> struct wider<int32_t> { using type = int64_t; };
template <> struct wider<uint8_t>  { using type = uint16_t; };
template <> struct wider<uint16_t> { using type = uint32_t; };
template <> struct wider<uint32_t> { using type = uint64_t; };
template <typename T>
using wider_t = typename wider<T>::type;
inline uint32_t xor_shift_state = 0x400B4E00;
inline float xorshift_float() {
    xor_shift_state ^= xor_shift_state << 13;
    xor_shift_state ^= xor_shift_state >> 17;
    xor_shift_state ^= xor_shift_state << 5;
    return (float)(xor_shift_state) / (float)(0xFFFFFFFF);
}
inline int stochastic_round(float value) {
    float random = xorshift_float();
    float shifted = value - (random - 0.5f);
    return (int)std::round(shifted);
}
template<typename T>
struct FixedPointBlock {
    T* mantissa;
    int32_t exponent;
    int size;
    bool use_stochastic;
    FixedPointBlock(int sz, bool stochastic = true) {
        size = sz;
        mantissa = new T[size];
        exponent = 0;
        use_stochastic = stochastic;
    }
    ~FixedPointBlock() {
        delete[] mantissa;
    }
    void fit_exponent(const float* values, int count) {
        int min_exp = std::numeric_limits<int>::max();
        int max_exp = std::numeric_limits<int>::min();
        for (int i = 0; i < count; ++i) {
            float val = values[i];
            if (val == 0.0f) continue;
            float abs_val = std::abs(val);
            int exp = (int)std::floor(std::log2(abs_val));
            if (exp < min_exp) min_exp = exp;
            if (exp > max_exp) max_exp = exp;
        }
        if (min_exp == std::numeric_limits<int>::max()) {
            exponent = 0;
        } else {
            exponent = (max_exp + min_exp) / 2;
        }
    }
    void floats_to_mantissa(const float* floats, int count) {
        float scale = std::ldexp(1.0f, -exponent);
        for (int i = 0; i < count; ++i) {
            float scaled = floats[i] * scale;
            if (use_stochastic) {
                mantissa[i] = (T)stochastic_round(scaled);
            } else {
                mantissa[i] = (T)std::round(scaled);
            }
        }
    }
    void mantissa_to_floats(float* output, int count) {
        float scale = std::ldexp(1.0f, exponent);
        for (int i = 0; i < count; ++i) {
            output[i] = (float)mantissa[i] * scale;
        }
    }
    void update_exponent_from_mantissas(const wider_t<T>* wider_mantissas, int count) {
        int max_exp = std::numeric_limits<int>::min();
        for (int i = 0; i < count; ++i) {
            wider_t<T> val = wider_mantissas[i];
            if (val == 0) continue;
            float abs_val = (float)std::abs((double)val);
            int exp = (int)std::ceil(std::log2(abs_val));
            if (exp > max_exp) max_exp = exp;
        }
        if (max_exp != std::numeric_limits<int>::min()) {
            int mantissa_bits = sizeof(T) * 8 - 1;
            exponent += (max_exp - mantissa_bits);
        }
    }
};
}
#endif