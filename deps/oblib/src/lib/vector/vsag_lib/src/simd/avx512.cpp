

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

    // 对齐的临时结果数组
    float PORTABLE_ALIGN64 TmpRes[16]; 
    
    // 针对128维向量的优化路径
    if (__builtin_expect(qty == 128, 1)) { 
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

        // 展开的主计算循环，每次处理64个元素
        for (int i = 0; i < 2; ++i) { 
            const int offset = i * 64; 
            
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

        // 使用树形结构归约求和
        sum0 = _mm512_add_ps(sum0, sum1); 
        sum2 = _mm512_add_ps(sum2, sum3); 
        sum0 = _mm512_add_ps(sum0, sum2); 
        
        // 存储结果并计算最终和
        _mm512_store_ps(TmpRes, sum0); 
        float res = 0.0f; 
        for (int i = 0; i < 16; i++) { 
            res += TmpRes[i]; 
        } 
        return res; 
    } 
    
    // 通用路径，处理任意维度
    size_t qty16 = qty >> 4; // 计算可处理的16元素块数
    const float* pEnd1 = pVect1 + (qty16 << 4); 
    __m512 sum = _mm512_setzero_ps(); 
    
    while (pVect1 < pEnd1) { 
        __m512 v1 = _mm512_loadu_ps(pVect1); 
        pVect1 += 16; 
        __m512 v2 = _mm512_loadu_ps(pVect2); 
        pVect2 += 16; 
        __m512 diff = _mm512_sub_ps(v1, v2); 
        sum = _mm512_fmadd_ps(diff, diff, sum); 
    } 
    
    // 如果向量长度不是16的倍数，处理剩余部分
    size_t remaining = qty - (qty16 << 4);
    if (remaining > 0) {
        __m512 v1 = _mm512_setzero_ps();
        __m512 v2 = _mm512_setzero_ps();
        __m512 diff = _mm512_setzero_ps();
        if (remaining > 0) {
            for (size_t i = 0; i < remaining; ++i) {
                v1 = _mm512_set1_ps(pVect1[qty16 * 16 + i]);
                v2 = _mm512_set1_ps(pVect2[qty16 * 16 + i]);
                diff = _mm512_sub_ps(v1, v2);
                sum = _mm512_fmadd_ps(diff, diff, sum);
            }
        }
    }
    
    _mm512_store_ps(TmpRes, sum); 
    float res = 0.0f; 
    for (int i = 0; i < 16; i++) { 
        res += TmpRes[i]; 
    } 
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
