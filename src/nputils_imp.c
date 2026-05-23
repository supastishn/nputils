#include "nputils.h"
#include "math_utils.h"
#include "matmul.h"
AEEResult nputils_matmul_8bit(remote_handle64 h, const float* input, int inputLen, const float* weights, int weightsLen, float* output, int outputLen, int m, int k, int n)
{
    (void)inputLen; (void)weightsLen; (void)outputLen;
    return nputils::blas::matmul_8bit(input, weights, output, m, k, n);
}
AEEResult nputils_matmul_16bit(remote_handle64 h, const float* input, int inputLen, const float* weights, int weightsLen, float* output, int outputLen, int m, int k, int n)
{
    (void)inputLen; (void)weightsLen; (void)outputLen;
    return nputils::blas::matmul_16bit(input, weights, output, m, k, n);
}