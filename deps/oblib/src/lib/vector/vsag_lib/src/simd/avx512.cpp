

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

    __m512i sum_low = _mm512_setzero_si512();  // For lower 16 elements
    __m512i sum_high = _mm512_setzero_si512(); // For higher 16 elements
    
    for (int i = 0; i + 31 < 128; i += 32) {   
        __m256i code1_values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));   
        __m256i code2_values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));   
        
        // Convert uint8_t to int16_t 
        __m512i codes1_512 = _mm512_cvtepu8_epi16(code1_values);   
        __m512i codes2_512 = _mm512_cvtepu8_epi16(code2_values);   
        
        // Compute the difference 
        __m512i diff = _mm512_sub_epi16(codes1_512, codes2_512);   
        
        // Compute the square of the differences 
        __m512i diff_squared = _mm512_mullo_epi16(diff, diff);   
        
        // Unpack and accumulate to 32-bit integers to prevent overflow
        __m512i diff_squared_low = _mm512_unpacklo_epi16(diff_squared, _mm512_setzero_si512());
        __m512i diff_squared_high = _mm512_unpackhi_epi16(diff_squared, _mm512_setzero_si512());
        
        sum_low = _mm512_add_epi32(sum_low, diff_squared_low);
        sum_high = _mm512_add_epi32(sum_high, diff_squared_high);
    }   
    
    // Combine both sums and store results
    int result_low[16], result_high[16];
    _mm512_storeu_si512(result_low, sum_low);
    _mm512_storeu_si512(result_high, sum_high);
    
    // Use 64-bit accumulator to prevent overflow
    int total_sum = 0;
    for (int i = 0; i < 16; ++i) {
        total_sum += result_low[i] + result_high[i];
    }
    
    return static_cast<float>(total_sum);
}


}  // namespace vsag
