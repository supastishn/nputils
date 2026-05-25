#ifndef MATH_UTILS
#define MATH_UTILS
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
namespace nputils {

namespace memory {
	//todo: 8kb blocks insyead of current blocks of 

inline int prefetch_l2(const void* ptr, int bytes, int MAX_FETCH = 256 * 1024) {
    const int LINE = 128;
    const int MAX_HEIGHT = 64;

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
	mantissa = new (std::align_val_t{128}) T[size];
        exponent = 0;
        use_stochastic = stochastic;
    }
    ~FixedPointBlock() {
	::operator delete[](mantissa, std::align_val_t{128});
}
    void fit_exponent(const float* values, int count) {
        float max_abs = 0.0f;
        for (int i = 0; i < count; ++i) {
            float abs_val = std::abs(values[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        if (max_abs == 0.0f) {
            exponent = 0;
            return;
        }
        int mantissa_bits = sizeof(T) * 8 - 1;
        int max_mantissa = (1 << mantissa_bits) - 1;
        float scaled_max = max_abs / (float)max_mantissa;
        exponent = (int)std::ceil(std::log2(scaled_max));
    }
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

	namespace preprocess {
	void chunkCrouton()
	}
};
}
#endif
