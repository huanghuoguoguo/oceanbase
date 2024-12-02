

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
    int8_t* x = (int8_t*)pVect1v; 
    int8_t* y = (int8_t*)pVect2v; 
    int dim = 128;
    __m512i sum  = _mm512_setzero_si512();  // 32位累加寄存器
    for (int i = 0; i < dim; i += 32) { // 每次处理32个int8（256 bits = 32 bytes） 
        // 加载32个元素 
        __m512i x_values = _mm512_loadu_epi8(x + i); 
        __m512i y_values = _mm512_loadu_epi8(y + i); 
 
        // 计算差值 
        __m512i diff = _mm512_sub_epi8(x_values, y_values); 
 
        // 计算差值的平方 
        __m512i diff_squared = _mm512_maddubs_epi16(diff, diff); // 16位结果
        
        // 将16位的结果转换为32位，分开处理两个256位部分
        __m256i low_256 = _mm512_castsi512_si256(diff_squared); // 低256位
        __m256i high_256 = _mm512_extracti64x4_si512(diff_squared, 1); // 高256位
        
        // 将低高256位的16位差值转换为32位
        __m256i low_32 = _mm256_cvtepi16_epi32(low_256);
        __m256i high_32 = _mm256_cvtepi16_epi32(high_256);

        // 将结果累加到 sum 中
        sum = _mm512_add_epi32(sum, _mm512_castsi256_si512(low_32));
        sum = _mm512_add_epi32(sum, _mm512_castsi256_si512(high_32));
    } 

    // 将结果水平加和，得到最终的 L2 距离平方
    __m256i sum256 = _mm256_add_epi32(
        _mm512_castsi512_si256(sum),
        _mm512_extracti32x8_epi32(sum, 1)
    );
    sum256 = _mm256_hadd_epi32(sum256, sum256);
    sum256 = _mm256_hadd_epi32(sum256, sum256);

    // 提取最终的 L2 距离平方
    int result = _mm_cvtsi128_si32(_mm256_castsi256_si128(sum256)) + 
                 _mm_cvtsi128_si32(_mm256_extracti128_si256(sum256, 1)); 
 
    return static_cast<float>(result); // 返回 L2 距离的平方 
}


}  // namespace vsag
