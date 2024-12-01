

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
#include "../logger.h"
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

float 
SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    uint8_t* codes1 = (uint8_t*)pVect1v;
    uint8_t* codes2 = (uint8_t*)pVect1v;

    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;
    int dim = 128;
    for (; i + 15 < dim; i += 16) {
        __m128i code1_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes1 + i));
        __m128i code2_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes2 + i));
        __m512i codes1_512 = _mm512_cvtepu8_epi32(code1_values);
        __m512i codes2_512 = _mm512_cvtepu8_epi32(code2_values);
        __m512 code1_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes1_512), _mm512_set1_ps(255.0f));
        __m512 code2_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes2_512), _mm512_set1_ps(255.0f));
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lowerBound_values = _mm512_loadu_ps(lowerBound + i);

        // Perform calculations
        __m512 scaled_codes1 = _mm512_fmadd_ps(code1_floats, diff_values, lowerBound_values);
        __m512 scaled_codes2 = _mm512_fmadd_ps(code2_floats, diff_values, lowerBound_values);
        __m512 val = _mm512_sub_ps(scaled_codes1, scaled_codes2);
        val = _mm512_mul_ps(val, val);
        sum = _mm512_add_ps(sum, val);
    }

    // Horizontal addition
    float result = _mm512_reduce_add_ps(sum);
    // Process the remaining elements
    result += avx2::SQ8ComputeCodesL2Sqr(codes1 + i, codes2 + i, lowerBound + i, diff + i, dim - i);
    return result;
}

// 这个函数起作用了，但是不知道速度是多少
// float 
// SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
//     uint8_t* x = (uint8_t*)pVect1v;
//     uint8_t* y = (uint8_t*)pVect2v;

//     int sum = 0;
//     for (int i = 0; i < 128; ++i) {
//         int diff = static_cast<int>(x[i]) - static_cast<int>(y[i]);  // 计算差值
//         sum += diff * diff;  // 累加差值的平方
//     }
//     return static_cast<float>(sum);
// }

// float SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
//     uint8_t* x = (uint8_t*)pVect1v;
//     uint8_t* y = (uint8_t*)pVect2v;
//     int sum = 0;

//     // 使用SSE2指令集进行优化
//     __m128i sum_vec = _mm_setzero_si128(); // 初始化累加向量为零
//     for (int i = 0; i < 128; i += 16) { // 每次处理16个字节
//         __m128i x_vec = _mm_loadu_si128((__m128i*)(x + i)); // 加载x的16个字节
//         __m128i y_vec = _mm_loadu_si128((__m128i*)(y + i)); // 加载y的16个字节
//         __m128i diff_vec = _mm_sub_epi8(x_vec, y_vec); // 计算差值
//         __m128i diff_sq_vec = _mm_maddubs_epi16(diff_vec, diff_vec); // 计算差值的平方
//         sum_vec = _mm_add_epi32(sum_vec, _mm_shuffle_epi32(_mm_add_epi32(_mm_add_epi32(diff_sq_vec, _mm_srli_si128(diff_sq_vec, 4)), _mm_srli_si128(diff_sq_vec, 8)), 0x0E)); // 累加差值的平方
//     }

//     // 将累加向量的四个32位整数相加得到最终结果
//     sum = _mm_cvtsi128_si32(sum_vec) + _mm_cvtsi128_si32(_mm_srli_si128(sum_vec, 4)) + _mm_cvtsi128_si32(_mm_srli_si128(sum_vec, 8)) + _mm_cvtsi128_si32(_mm_srli_si128(sum_vec, 12));
//     vsag::logger::error("yhh sum = {}", sum);
//     return static_cast<float>(sum); // 返回 L2 距离的平方
// }
}  // namespace vsag
