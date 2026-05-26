#ifndef MATH_UTILS
#define MATH_UTILS
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <mutex>
namespace nputils {

namespace memory {

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

struct BlockHeader {
    size_t size;
    bool is_free;
    BlockHeader* next;
    BlockHeader* prev;
    uint8_t padding[128 - sizeof(size_t) - sizeof(bool) - 2 * sizeof(BlockHeader*)];
};

class Scratchpad {
private:
    inline static uint8_t* buffer = nullptr;
    inline static size_t total_size = 32 * 1024 * 1024;
    inline static BlockHeader* head = nullptr;
    inline static std::mutex mtx;

    static void initialize_if_needed() {
        if (!buffer) {
            buffer = (uint8_t*)::operator new[](total_size, std::align_val_t{128});
            head = (BlockHeader*)buffer;
            head->size = total_size - sizeof(BlockHeader);
            head->is_free = true;
            head->next = nullptr;
            head->prev = nullptr;
        }
    }

public:
    static void set_size(size_t new_size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer) {
            ::operator delete[](buffer, std::align_val_t{128});
            buffer = nullptr;
            head = nullptr;
        }
        total_size = new_size;
    }

    static void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        initialize_if_needed();
        size = (size + 127) & ~127;
        BlockHeader* curr = head;
        while (curr) {
            if (curr->is_free && curr->size >= size) {
                if (curr->size >= size + sizeof(BlockHeader) + 128) {
                    BlockHeader* next_block = (BlockHeader*)((uint8_t*)curr + sizeof(BlockHeader) + size);
                    next_block->size = curr->size - size - sizeof(BlockHeader);
                    next_block->is_free = true;
                    next_block->next = curr->next;
                    next_block->prev = curr;
                    if (curr->next) {
                        curr->next->prev = next_block;
                    }
                    curr->next = next_block;
                    curr->size = size;
                }
                curr->is_free = false;
                return (void*)((uint8_t*)curr + sizeof(BlockHeader));
            }
            curr = curr->next;
        }
        return ::operator new[](size, std::align_val_t{128});
    }

    static void free(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer && (uint8_t*)ptr >= buffer && (uint8_t*)ptr < buffer + total_size) {
            BlockHeader* curr = (BlockHeader*)((uint8_t*)ptr - sizeof(BlockHeader));
            curr->is_free = true;
            if (curr->next && curr->next->is_free) {
                curr->size += sizeof(BlockHeader) + curr->next->size;
                curr->next = curr->next->next;
                if (curr->next) {
                    curr->next->prev = curr;
                }
            }
            if (curr->prev && curr->prev->is_free) {
                curr->prev->size += sizeof(BlockHeader) + curr->size;
                curr->prev->next = curr->next;
                if (curr->next) {
                    curr->next->prev = curr->prev;
                }
            }
        } else {
            ::operator delete[](ptr, std::align_val_t{128});
        }
    }
};

template<typename T>
struct FixedPointBlock {
    T* mantissa;
    int32_t exponent;
    int size;
    bool use_stochastic;
    FixedPointBlock(int sz, bool stochastic = true) {
        size = sz;
        mantissa = (T*)Scratchpad::alloc(size * sizeof(T));
        exponent = 0;
        use_stochastic = stochastic;
    }
    ~FixedPointBlock() {
        Scratchpad::free(mantissa);
    }
    void fit_exponent(const float* values, int count) {
        memory::prefetch_l2(values, count * sizeof(float));
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
    void floats_to_mantissa(const float* floats, int count) {
        memory::prefetch_l2(floats, count * sizeof(float));
        memory::prefetch_l2(mantissa, count * sizeof(T));
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
        memory::prefetch_l2(mantissa, count * sizeof(T));
        memory::prefetch_l2(output, count * sizeof(float));
        float scale = std::ldexp(1.0f, exponent);
        for (int i = 0; i < count; ++i) {
            output[i] = (float)mantissa[i] * scale;
        }
    }
    void update_exponent_from_mantissas(const wider_t<T>* wider_mantissas, int count) {
        memory::prefetch_l2(wider_mantissas, count * sizeof(wider_t<T>));
        int max_exp = std::numeric_limits<int>::min();
        for (int i = 0; i < count; ++i) {
            wider_t<T> val = wider_mantissas[i];
            if (val == 0) continue;
            float abs_val = (float)std::abs((double)val);
            int exp = (int)std::ceil(std::log2(abs_val));
            if (exp > max_exp) max_exp = exp;
        }
        if (max_exp != std::numeric_limits<int>::min()) {
            exponent += max_exp;
        }
    }
};

namespace preprocess {
inline void chunkCrouton(const int8_t* matrix, int8_t* output, int rows, int cols) {
    memory::prefetch_l2(matrix, rows * cols * sizeof(int8_t));
    int paddedCols = (cols + 63) & ~63;
    memory::prefetch_l2(output, rows * paddedCols * sizeof(int8_t));
    for (int i = 0; i < paddedCols; i += 64) {
        for (int j = 0; j < rows; j += 4) {
            for (int z = 0; z < 4; ++z) {
                for (int k = 0; k < 64; ++k) {
                    int row_idx = j + z;
                    int col_idx = i + k;
                    int8_t val = 0;
                    if (row_idx < rows && col_idx < cols) {
                        val = matrix[row_idx * cols + col_idx];
                    }
                    output[i * rows + j * 64 + k * 4 + z] = val;
                }
            }
        }
    }
}
}
}
#endif