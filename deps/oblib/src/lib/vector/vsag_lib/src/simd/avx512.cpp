

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

float 
SQ8ComputeCodesL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const int8_t* x = (const int8_t*)pVect1v; 
    const int8_t* y = (const int8_t*)pVect2v; 
 
    size_t dim = 128; 
    __m512i sum = _mm512_setzero_si512(); // 初始化累加向量为零 
 
    size_t i = 0; 
    for (; i < dim; i += 64) { 
        // 加载64个int8元素
        __m512i x_values = _mm512_loadu_epi8(x + i); 
        __m512i y_values = _mm512_loadu_epi8(y + i); 
 
        // 计算差值
        __m512i diff = _mm512_sub_epi8(x_values, y_values); 

        // 将前32个差值扩展到int16并平方
        __m512i diff_lower = _mm512_and_si512(diff, _mm512_set1_epi8(0xFF)); // 提取低32位
        __m512i diff_lower_int16 = _mm512_cvtepi8_epi16(diff_lower); // 扩展到int16
        __m512i diff_lower_squared = _mm512_mullo_epi16(diff_lower_int16, diff_lower_int16); // 计算平方

        // 将后32个差值扩展到int16并平方
        __m512i diff_upper = _mm512_srli_epi64(diff, 32); // 提取高32位
        __m512i diff_upper_int16 = _mm512_cvtepi8_epi16(diff_upper); // 扩展到int16
        __m512i diff_upper_squared = _mm512_mullo_epi16(diff_upper_int16, diff_upper_int16); // 计算平方

        // 将结果扩展到32位
        __m256i lower_256 = _mm512_castsi512_si256(diff_lower_squared); // 低32个元素
        __m256i upper_256 = _mm512_castsi512_si256(diff_upper_squared); // 高32个元素
        
        // 扩展为32位
        lower_256 = _mm256_cvtepi16_epi32(lower_256); 
        upper_256 = _mm256_cvtepi16_epi32(upper_256); 

        // 将32位结果累加
        __m512i lower_sum = _mm512_castsi256_si512(lower_256);
        __m512i upper_sum = _mm512_castsi256_si512(upper_256);
        sum = _mm512_add_epi32(sum, lower_sum);
        sum = _mm512_add_epi32(sum, upper_sum);
    } 
 
    // 将结果水平加和，得到最终的L2距离平方
    int result = 0;
    for (int j = 0; j < 16; ++j) { 
        result += _mm512_extract_epi32(sum, j); // 提取每个32位元素并累加
    } 
 
    return static_cast<float>(result); // 返回L2距离的平方 
}


}  // namespace vsag
