

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
    uint8_t* x = (uint8_t*)pVect1v;
    uint8_t* y = (uint8_t*)pVect1v;

    __m512 sum = _mm512_setzero_ps(); // 初始化累加向量为零
    uint64_t i = 0;
    int dim = 128;

    for (; i + 31 < dim; i += 32) { // 每次处理32个uint8（256 bits = 32 bytes）
        // 加载32个元素（每次处理32个字节，128维数据分为4块处理）
        __m512i code1_values = _mm512_loadu_epi8(x + i);
        __m512i code2_values = _mm512_loadu_epi8(y + i);

        // 将uint8转换为float32
        __m512 code1_floats = _mm512_cvtepu8_epi32(code1_values);
        __m512 code2_floats = _mm512_cvtepu8_epi32(code2_values);
        code1_floats = _mm512_div_ps(code1_floats, _mm512_set1_ps(255.0f));
        code2_floats = _mm512_div_ps(code2_floats, _mm512_set1_ps(255.0f));

        // 计算差值和差值的平方
        __m512 diff = _mm512_sub_ps(code1_floats, code2_floats);
        diff = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, diff);
    }

    // Horizontal addition to get the final sum
    float result = _mm512_reduce_add_ps(sum);
    return result;
}

// float 
// SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
//     const uint8_t* x = reinterpret_cast<const uint8_t*>(pVect1v);
//     const uint8_t* y = reinterpret_cast<const uint8_t*>(pVect2v);

//     __m256i sum = _mm256_setzero_si256(); // 初始化累加向量为零
//     __m256i mask = _mm256_set1_epi8(0xf); // 用于掩码低4位的掩码

//     for (int i = 0; i < 128; i += 32) { // 每次处理32个uint8（256 bits = 32 bytes）
//         // 加载32个元素（每次处理32个字节，128维数据分为4块处理）
//         __m256i xx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
//         __m256i yy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));

//         // 分拆为低四位和高四位
//         __m256i xx1 = _mm256_and_si256(xx, mask);
//         __m256i xx2 = _mm256_srli_epi16(xx, 4);
//         xx2 = _mm256_and_si256(xx2, mask);

//         __m256i yy1 = _mm256_and_si256(yy, mask);
//         __m256i yy2 = _mm256_srli_epi16(yy, 4);
//         yy2 = _mm256_and_si256(yy2, mask);

//         // 计算每部分的差值
//         __m256i d1 = _mm256_sub_epi8(xx1, yy1);
//         __m256i d2 = _mm256_sub_epi8(xx2, yy2);

//         // 取差值的绝对值
//         d1 = _mm256_abs_epi8(d1);
//         d2 = _mm256_abs_epi8(d2);

//         // 计算差值平方并累加
//         __m256i d1_sq = _mm256_maddubs_epi16(d1, d1); // d1^2
//         __m256i d2_sq = _mm256_maddubs_epi16(d2, d2); // d2^2
//         sum = _mm256_add_epi32(sum, _mm256_add_epi32(d1_sq, d2_sq));
//     }

//     // 将256位向量的8个int32相加得到最终结果
//     __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extractf128_si256(sum, 1));
//     sum128 = _mm_hadd_epi32(sum128, sum128);
//     sum128 = _mm_hadd_epi32(sum128, sum128);

//     // 提取最终的 L2 距离平方
//     return _mm_cvtsi128_si32(sum128);
// }


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
