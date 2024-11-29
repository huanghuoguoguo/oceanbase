

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
    const uint8_t* pVect1 = reinterpret_cast<const uint8_t*>(pVect1v);
    const uint8_t* pVect2 = reinterpret_cast<const uint8_t*>(pVect2v);

    const size_t dim = 128; // We know we are working with 128-dimensional vectors
    __m512i sum = _mm512_setzero_si512(); // accumulator for the sum of squared differences

    size_t i = 0;
    for (; i + 63 < dim; i += 64) {
        // Load 64 bytes (8 uint8_t elements) from both vectors (Note: these are now 512-bit wide)
        __m512i v1 = _mm512_loadu_si512(&pVect1[i]);
        __m512i v2 = _mm512_loadu_si512(&pVect2[i]);

        // Expand uint8_t (8-bit) to 32-bit integers (16 elements, 32-bit each)
        __m512i v1_lo = _mm512_unpacklo_epi8(v1, _mm512_setzero_si512());  // Lower 8 bits of each byte to 32-bit
        __m512i v1_hi = _mm512_unpackhi_epi8(v1, _mm512_setzero_si512());  // Upper 8 bits of each byte to 32-bit
        __m512i v2_lo = _mm512_unpacklo_epi8(v2, _mm512_setzero_si512());
        __m512i v2_hi = _mm512_unpackhi_epi8(v2, _mm512_setzero_si512());

        // Calculate the difference between corresponding elements
        __m512i diff_lo = _mm512_sub_epi32(v1_lo, v2_lo);
        __m512i diff_hi = _mm512_sub_epi32(v1_hi, v2_hi);

        // Calculate the squared difference (using mullo_epi32 to multiply 32-bit integers)
        __m512i squared_diff_lo = _mm512_mullo_epi32(diff_lo, diff_lo);  // Square the differences
        __m512i squared_diff_hi = _mm512_mullo_epi32(diff_hi, diff_hi);

        // Accumulate the squared differences
        sum = _mm512_add_epi32(sum, squared_diff_lo);
        sum = _mm512_add_epi32(sum, squared_diff_hi);
    }

    // Handle remaining elements (less than 64 bytes)
    float result = 0.0f;
    for (; i < dim; ++i) {
        int diff = static_cast<int>(pVect1[i]) - static_cast<int>(pVect2[i]);
        result += diff * diff;
    }

    // Horizontal sum (reducing 16 elements to 1 value)
    uint32_t tmp[16];
    _mm512_storeu_si512(tmp, sum);
    for (int j = 0; j < 16; ++j) {
        result += tmp[j];
    }

    return result;
}

}  // namespace vsag
