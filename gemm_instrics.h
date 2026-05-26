#ifndef GEMM_INSTRINSICS
#define GEMM_INSTRINSICS
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include "utils.h"
#include <cstring>
namespace nputils {
namespace blas_ll {
inline int gemm_int8_int32(const int8_t* A, const int8_t* B, int32_t* C, int m1rows, int m1colsm2rows, int m2cols) {
    if (reinterpret_cast<uintptr_t>(A) % 128 != 0 || reinterpret_cast<uintptr_t>(B) % 128 != 0 || reinterpret_cast<uintptr_t>(C) % 128 != 0) {
        return -1;
    }
    HVX_Vector register_1 = Q6_V_vsplat_R(0);
    HVX_Vector register_2 = Q6_V_vsplat_R(0);
    if (reinterpret_cast<uintptr_t>(A) % 128 != 0 || reinterpret_cast<uintptr_t>(B) % 128 != 0 || reinterpret_cast<uintptr_t>(C) % 128 != 0) {
        return -1;
    }
    for (int i = 0; i < m1rows; ++i) {
        memory::prefetch_l2(A + i * m1colsm2rows, m1colsm2rows * 3);
        for (int j = 0; j < m2cols / 64; ++j) {
            memory::prefetch_l2(B + j * 64 * m1colsm2rows, (m1colsm2rows * 64) * 4);
            for (int k = 0; k < m1colsm2rows / 4; ++k) {
                HVX_Vector cols_m1 = Q6_V_vmem_adv(B + j * 64 * m1colsm2rows + k * 256);
                HVX_Vector cols_m2 = Q6_V_vmem_adv(B + j * 64 * m1colsm2rows + k * 256 + 128);
                uint32_t splat;
                memcpy(&splat, A + k * 4 + i * m1colsm2rows, sizeof(uint32_t));
                HVX_Vector splatted = Q6_V_vsplat_R(splat);
                register_1 = Q6_Vw_vrmpyacc_VwVbVb(register_1, cols_m1, splatted);
                register_2 = Q6_Vw_vrmpyacc_VwVbVb(register_2, cols_m2, splatted);
            }
            Q6_v_vmem_RV(&C[i * m2cols + j * 64], register_1);
            Q6_v_vmem_RV(&C[i * m2cols + j * 64 + 32], register_2);
            register_1 = Q6_V_vsplat_R(0);
            register_2 = Q6_V_vsplat_R(0);
        }
    }
    return 0;
}
} // namespace blas_ll
} // namespace nputils
#endif