

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

extern float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);

/* L2 Distance */
float
L2SqrSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty4 = qty >> 2;

    const float* pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

float
L2SqrSIMD4ExtSSE_128(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    // 输入是两个 128 维向量
    float* pVect1 = (float*)pVect1v; 
    float* pVect2 = (float*)pVect2v; 

    // 临时数组存储中间结果
    float PORTABLE_ALIGN32 TmpRes[4]; 

    // 初始化 SSE 寄存器
    __m128 diff, v1, v2; 
    __m128 sum = _mm_set1_ps(0); 

    // 全展开：每次处理 4 个浮点数，总共 32 次（128 / 4 = 32）
    v1 = _mm_loadu_ps(pVect1); 
    v2 = _mm_loadu_ps(pVect2); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 4); 
    v2 = _mm_loadu_ps(pVect2 + 4); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 8); 
    v2 = _mm_loadu_ps(pVect2 + 8); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 12); 
    v2 = _mm_loadu_ps(pVect2 + 12); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 16); 
    v2 = _mm_loadu_ps(pVect2 + 16); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 20); 
    v2 = _mm_loadu_ps(pVect2 + 20); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 24); 
    v2 = _mm_loadu_ps(pVect2 + 24); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 28); 
    v2 = _mm_loadu_ps(pVect2 + 28); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 32); 
    v2 = _mm_loadu_ps(pVect2 + 32); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 36); 
    v2 = _mm_loadu_ps(pVect2 + 36); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 40); 
    v2 = _mm_loadu_ps(pVect2 + 40); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 44); 
    v2 = _mm_loadu_ps(pVect2 + 44); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 48); 
    v2 = _mm_loadu_ps(pVect2 + 48); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 52); 
    v2 = _mm_loadu_ps(pVect2 + 52); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 56); 
    v2 = _mm_loadu_ps(pVect2 + 56); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 60); 
    v2 = _mm_loadu_ps(pVect2 + 60); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 64); 
    v2 = _mm_loadu_ps(pVect2 + 64); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 68); 
    v2 = _mm_loadu_ps(pVect2 + 68); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 72); 
    v2 = _mm_loadu_ps(pVect2 + 72); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 76); 
    v2 = _mm_loadu_ps(pVect2 + 76); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 80); 
    v2 = _mm_loadu_ps(pVect2 + 80); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 84); 
    v2 = _mm_loadu_ps(pVect2 + 84); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 88); 
    v2 = _mm_loadu_ps(pVect2 + 88); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 92); 
    v2 = _mm_loadu_ps(pVect2 + 92); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 96); 
    v2 = _mm_loadu_ps(pVect2 + 96); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 100); 
    v2 = _mm_loadu_ps(pVect2 + 100); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 104); 
    v2 = _mm_loadu_ps(pVect2 + 104); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 108); 
    v2 = _mm_loadu_ps(pVect2 + 108); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 112); 
    v2 = _mm_loadu_ps(pVect2 + 112); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 116); 
    v2 = _mm_loadu_ps(pVect2 + 116); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    v1 = _mm_loadu_ps(pVect1 + 120); 
    v2 = _mm_loadu_ps(pVect2 + 120); 
    diff = _mm_sub_ps(v1, v2); 
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff)); 

    // 将结果存储到 TmpRes
    _mm_store_ps(TmpRes, sum); 

    // 汇总所有结果
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3]; 
}

float
L2SqrSIMD4ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4ExtSSE(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float* pVect1 = (float*)pVect1v + qty4;
    float* pVect2 = (float*)pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}

float
L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

extern float (*L2SqrSIMD16Ext)(const void*, const void*, const void*);

float
L2SqrSIMD16ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float* pVect1 = (float*)pVect1v + qty16;
    float* pVect2 = (float*)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

/* IP Distance */
float
InnerProductSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float
InnerProductSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

extern float (*InnerProductSIMD16Ext)(const void*, const void*, const void*);
extern float (*InnerProductSIMD4Ext)(const void*, const void*, const void*);

float
InnerProductDistanceSIMD16ExtResidualsSSE(const void* pVect1v,
                                          const void* pVect2v,
                                          const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float* pVect1 = (float*)pVect1v + qty16;
    float* pVect2 = (float*)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

float
InnerProductDistanceSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return 1.0f - InnerProductSIMD16Ext(pVect1v, pVect2v, qty_ptr);
}

float
InnerProductDistanceSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return 1.0f - InnerProductSIMD4Ext(pVect1v, pVect2v, qty_ptr);
}

float
InnerProductDistanceSIMD4ExtResidualsSSE(const void* pVect1v,
                                         const void* pVect2v,
                                         const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float* pVect1 = (float*)pVect1v + qty4;
    float* pVect2 = (float*)pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}

void
PQDistanceSSEFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    const float* float_centers = (const float*)single_dim_centers;
    float* float_result = (float*)result;
    for (size_t idx = 0; idx < 256; idx += 4) {
        __m128 v_centers_dim = _mm_loadu_ps(float_centers + idx);
        __m128 v_query_vec = _mm_set1_ps(single_dim_val);
        __m128 v_diff = _mm_sub_ps(v_centers_dim, v_query_vec);
        __m128 v_diff_sq = _mm_mul_ps(v_diff, v_diff);
        __m128 v_chunk_dists = _mm_loadu_ps(&float_result[idx]);
        v_chunk_dists = _mm_add_ps(v_chunk_dists, v_diff_sq);
        _mm_storeu_ps(&float_result[idx], v_chunk_dists);
    }
}

}  // namespace vsag
