

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
    
    // 如果可能，确保 pVect1 和 pVect2 都是 64 字节对齐
    // 这样可以直接使用 _mm512_load_ps 而非 _mm512_loadu_ps

    // 128维向量固定路径
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    // 预取数据到L1缓存
    #pragma unroll(8)
    for (int i = 0; i < 8; ++i) { 
        _mm_prefetch((char*)(pVect1 + i * 16), _MM_HINT_T0); 
        _mm_prefetch((char*)(pVect2 + i * 16), _MM_HINT_T0); 
    }

    // 扩展计算循环，每次处理64个元素
    for (int i = 0; i < 2; ++i) { 
        const int offset = i * 64;

        // 使用对齐的加载方式（假设pVect1和pVect2是按64字节对齐）
        __m512 v1_0 = _mm512_load_ps(pVect1 + offset); 
        __m512 v2_0 = _mm512_load_ps(pVect2 + offset); 
        __m512 diff0 = _mm512_sub_ps(v1_0, v2_0); 
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0); 

        __m512 v1_1 = _mm512_load_ps(pVect1 + offset + 16); 
        __m512 v2_1 = _mm512_load_ps(pVect2 + offset + 16); 
        __m512 diff1 = _mm512_sub_ps(v1_1, v2_1); 
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1); 

        __m512 v1_2 = _mm512_load_ps(pVect1 + offset + 32); 
        __m512 v2_2 = _mm512_load_ps(pVect2 + offset + 32); 
        __m512 diff2 = _mm512_sub_ps(v1_2, v2_2); 
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2); 

        __m512 v1_3 = _mm512_load_ps(pVect1 + offset + 48); 
        __m512 v2_3 = _mm512_load_ps(pVect2 + offset + 48); 
        __m512 diff3 = _mm512_sub_ps(v1_3, v2_3); 
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3); 
    }

    // 汇总最终的结果
    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    // 将结果加载到TmpRes并计算L2范数
    float res = 0.0f;
    __m512 tempRes = sum0;
    tempRes = _mm512_hadd_ps(tempRes, tempRes);  // 水平加法
    tempRes = _mm512_hadd_ps(tempRes, tempRes);  // 水平加法

    _mm512_store_ps(&res, tempRes);
    return res;
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
