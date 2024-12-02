

// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <x86intrin.h>
#include <iostream>
// #include "../logger.h"
namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

float
InnerProductSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return sum;
}

// float 
// SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
//     const uint8_t* x = reinterpret_cast<const uint8_t*>(pVect1v);
//     const uint8_t* y = reinterpret_cast<const uint8_t*>(pVect2v);

//     __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
//     __m256i mask = _mm256_set1_epi8(0xf);

//     for (int i = 0; i < 128; i += 32) { // 每次处理32个uint8（256 bits = 32 bytes）
//         // 加载32个元素（每次处理64个字节，128维数据分为4块处理）
//         auto xx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
//         auto yy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));

//         // 分拆为低四位和高四位
//         auto xx1 = _mm256_and_si256(xx, mask);
//         auto xx2 = _mm256_srli_epi16(xx, 4);
//         xx2 = _mm256_and_si256(xx2, mask);
        
//         auto yy1 = _mm256_and_si256(yy, mask);
//         auto yy2 = _mm256_srli_epi16(yy, 4);
//         yy2 = _mm256_and_si256(yy2, mask);

//         // 计算每部分的差值
//         auto d1 = _mm256_sub_epi8(xx1, yy1);
//         auto d2 = _mm256_sub_epi8(xx2, yy2);

//         // 取差值的绝对值
//         d1 = _mm256_abs_epi8(d1);
//         d2 = _mm256_abs_epi8(d2);

//         // 计算差值平方并累加
//         sum1 = _mm256_add_epi32(sum1, _mm256_maddubs_epi16(d1, d1)); // d1^2
//         sum2 = _mm256_add_epi32(sum2, _mm256_maddubs_epi16(d2, d2)); // d2^2
//     }

//     // 汇总 sum1 和 sum2 中的结果
//     sum1 = _mm256_add_epi32(sum1, sum2);
    
//     // 水平加法，合并sum1中的数据并得到最终结果
//     __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum1), _mm256_extracti128_si256(sum1, 1));
//     sum128 = _mm_hadd_epi32(sum128, sum128); // 水平加法，合并为一个单一的值

//     // 提取最终的 L2 距离
//     return _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1);
// }
float 
SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    uint8_t* x = (uint8_t*)pVect1v; 
    uint8_t* y = (uint8_t*)pVect2v; 
    
    int sum = 0;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 128; ++i) { 
        int diff = static_cast<int>(x[i]) - static_cast<int>(y[i]);  // 计算差值
        sum += diff * diff;  // 累加差值的平方
    }
    
    return static_cast<float>(sum);
}

// float SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
//     const uint8_t* x = reinterpret_cast<const uint8_t*>(pVect1v);
//     const uint8_t* y = reinterpret_cast<const uint8_t*>(pVect2v);
//     uint32_t sum = 0;

//     // 使用SSE2指令集进行优化
//     __m128i sum_vec = _mm_setzero_si128(); // 初始化累加向量为零
//     for (int i = 0; i < 128; i += 16) { // 每次处理16个字节
//         __m128i x_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)); // 加载x的16个字节
//         __m128i y_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + i)); // 加载y的16个字节
//         __m128i diff_vec = _mm_sub_epi8(x_vec, y_vec); // 计算差值
//         __m128i diff_sq_vec = _mm_maddubs_epi16(diff_vec, diff_vec); // 计算差值的平方
//         sum_vec = _mm_add_epi32(sum_vec, _mm_shuffle_epi32(_mm_add_epi32(_mm_add_epi32(diff_sq_vec, _mm_srli_si128(diff_sq_vec, 4)), _mm_srli_si128(diff_sq_vec, 8)), 0x0E)); // 累加差值的平方
//     }

//     // 将累加向量的四个32位整数相加得到最终结果
//     sum = _mm_cvtsi128_si32(sum_vec) + _mm_cvtsi128_si32(_mm_srli_si128(sum_vec, 4)) + _mm_cvtsi128_si32(_mm_srli_si128(sum_vec, 8)) + _mm_cvtsi128_si32(_mm_srli_si128(sum_vec, 12));

//     return static_cast<float>(sum); // 返回 L2 距离的平方
// }
}  // namespace vsag
