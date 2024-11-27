

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
L2SqrSIMD16ExtAVX512_128(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float *pVect1 = (const float *)pVect1v;
    const float *pVect2 = (const float *)pVect2v;
    size_t qty = *((const size_t *)qty_ptr); // qty 应为 128

    // 直接假设 qty 为 128（避免运行时检查），展开主循环
    __m512 sum = _mm512_setzero_ps();

    // 每次加载 16 个浮点数并进行计算，共需 8 次迭代
    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));
    pVect1 += 16; pVect2 += 16;

    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));
    pVect1 += 16; pVect2 += 16;

    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));
    pVect1 += 16; pVect2 += 16;

    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));
    pVect1 += 16; pVect2 += 16;

    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));
    pVect1 += 16; pVect2 += 16;

    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));
    pVect1 += 16; pVect2 += 16;

    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));
    pVect1 += 16; pVect2 += 16;

    sum = _mm512_add_ps(sum, _mm512_mul_ps(
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2)),
        _mm512_sub_ps(_mm512_loadu_ps(pVect1), _mm512_loadu_ps(pVect2))
    ));

    // 利用 AVX-512 指令将结果归约求和
    __m256 sum256 = _mm512_castps512_ps256(sum);
    sum256 = _mm256_add_ps(sum256, _mm512_extractf32x8_ps(sum, 1));
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ps(sum128, _mm_movehdup_ps(sum128));

    return _mm_cvtss_f32(sum128);
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

}  // namespace vsag
