/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Inference-only compatibility implementation for MTGS / gsplat 1.4.
// Retains the canonical input/output ABI and scheduling layout.  It deliberately
// replaces row-level alpha clipping and tile-level early termination with
// per-pixel rules.  Backward state is not provided; the companion patch rejects
// backward calls.  Performance and device correctness require separate testing.
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t LANES = 64;
class MTGSCompatibleRender {
public:
    __aicore__ inline void Init(GM_ADDR gs, GM_ADDR coords, GM_ADDR offsets, GM_ADDR ids,
                               GM_ADDR color, GM_ADDR depth, GM_ADDR last, GM_ADDR error,
                               GM_ADDR gsClip, GM_ADDR alphaClip, int64_t pixels, int64_t tiles)
    {
        pixels_ = pixels;
        tiles_ = tiles;
        vector_ = GetBlockIdx()*GetSubBlockNum()+GetSubBlockIdx();
        vectors_ = GetBlockNum()*GetSubBlockNum();
        gs_.SetGlobalBuffer((__gm__ float*)gs);
        coords_.SetGlobalBuffer((__gm__ float*)coords);
        schedule_.SetGlobalBuffer((__gm__ int64_t*)offsets);
        ids_.SetGlobalBuffer((__gm__ int64_t*)ids);
        color_.SetGlobalBuffer((__gm__ float*)color);
        depth_.SetGlobalBuffer((__gm__ float*)depth);
        last_.SetGlobalBuffer((__gm__ float*)last);
        error_.SetGlobalBuffer((__gm__ float*)error);
        gsClip_.SetGlobalBuffer((__gm__ int64_t*)gsClip);
        alphaClip_.SetGlobalBuffer((__gm__ uint8_t*)alphaClip);
        pipe_.InitBuffer(buffer_, 20*LANES*sizeof(float)+256);
        auto b = buffer_.Get<float>();
        px_=b; py_=b[LANES]; dx_=b[2*LANES]; dy_=b[3*LANES];
        tmp_=b[4*LANES]; tmp2_=b[5*LANES]; sigma_=b[6*LANES];
        alpha_=b[7*LANES]; trans_=b[8*LANES]; next_=b[9*LANES];
        weight_=b[10*LANES]; done_=b[11*LANES]; one_=b[12*LANES];
        red_=b[13*LANES]; green_=b[14*LANES]; blue_=b[15*LANES];
        dep_=b[16*LANES]; zero_=b[17*LANES];
        mask_=b[20*LANES].ReinterpretCast<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        const int64_t start = vector_ ? schedule_.GetValue(vector_-1) : 0;
        const int64_t finish = schedule_.GetValue(vector_);
        for (int64_t index=start; index<finish; ++index) {
            const int64_t tile = schedule_.GetValue(vectors_+index);
            const int64_t first = tile ? schedule_.GetValue(vectors_+tiles_+tile-1) : 0;
            const int64_t end = schedule_.GetValue(vectors_+tiles_+tile);
            for (int64_t pixel=0; pixel<pixels_; pixel+=LANES) {
                DataCopy(px_, coords_[tile*2*pixels_+pixel], LANES);
                DataCopy(py_, coords_[(tile*2+1)*pixels_+pixel], LANES);
                PipeBarrier<PIPE_ALL>();
                Duplicate(trans_,1.0f,LANES); Duplicate(one_,1.0f,LANES);
                Duplicate(done_,0.0f,LANES); Duplicate(zero_,0.0f,LANES);
                Duplicate(red_,0.0f,LANES); Duplicate(green_,0.0f,LANES);
                Duplicate(blue_,0.0f,LANES); Duplicate(dep_,0.0f,LANES);
                PipeBarrier<PIPE_V>();
                for (int64_t item=first; item<end; ++item) {
                    const int64_t g=ids_.GetValue(item)*10;
                    Adds(dx_,px_,-gs_.GetValue(g),LANES);
                    Adds(dy_,py_,-gs_.GetValue(g+1),LANES);
                    PipeBarrier<PIPE_V>();
                    Mul(sigma_,dx_,dx_,LANES); Mul(tmp_,dy_,dy_,LANES);
                    Mul(tmp2_,dx_,dy_,LANES); PipeBarrier<PIPE_V>();
                    Muls(sigma_,sigma_,0.5f*gs_.GetValue(g+2),LANES);
                    Muls(tmp_,tmp_,0.5f*gs_.GetValue(g+4),LANES);
                    Muls(tmp2_,tmp2_,gs_.GetValue(g+3),LANES);
                    PipeBarrier<PIPE_V>();
                    Add(sigma_,sigma_,tmp_,LANES); PipeBarrier<PIPE_V>();
                    Add(sigma_,sigma_,tmp2_,LANES); PipeBarrier<PIPE_V>();
                    Muls(alpha_,sigma_,-1.0f,LANES); PipeBarrier<PIPE_V>();
                    Exp(alpha_,alpha_,LANES); PipeBarrier<PIPE_V>();
                    Muls(alpha_,alpha_,gs_.GetValue(g+5),LANES); PipeBarrier<PIPE_V>();
                    Mins(alpha_,alpha_,0.999f,LANES); PipeBarrier<PIPE_V>();
                    CompareScalar(mask_,sigma_,0.0f,CMPMODE::GE,LANES); PipeBarrier<PIPE_V>();
                    Select(alpha_,mask_,alpha_,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,LANES);
                    PipeBarrier<PIPE_V>();
                    CompareScalar(mask_,alpha_,1.0f/255.0f,CMPMODE::GE,LANES); PipeBarrier<PIPE_V>();
                    Select(alpha_,mask_,alpha_,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,LANES);
                    PipeBarrier<PIPE_V>();
                    Muls(tmp_,alpha_,-1.0f,LANES); PipeBarrier<PIPE_V>();
                    Adds(tmp_,tmp_,1.0f,LANES); PipeBarrier<PIPE_V>();
                    Mul(next_,trans_,tmp_,LANES); PipeBarrier<PIPE_V>();
                    // The splat crossing the transmittance threshold is excluded.
                    CompareScalar(mask_,next_,1e-4f,CMPMODE::LE,LANES); PipeBarrier<PIPE_V>();
                    Select(tmp_,mask_,one_,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,LANES);
                    PipeBarrier<PIPE_V>(); Max(done_,done_,tmp_,LANES); PipeBarrier<PIPE_V>();
                    CompareScalar(mask_,done_,0.5f,CMPMODE::LT,LANES); PipeBarrier<PIPE_V>();
                    Select(alpha_,mask_,alpha_,0.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,LANES);
                    PipeBarrier<PIPE_V>(); Mul(weight_,trans_,alpha_,LANES); PipeBarrier<PIPE_V>();
                    Axpy(red_,weight_,gs_.GetValue(g+6),LANES);
                    Axpy(green_,weight_,gs_.GetValue(g+7),LANES);
                    Axpy(blue_,weight_,gs_.GetValue(g+8),LANES);
                    Axpy(dep_,weight_,gs_.GetValue(g+9),LANES); PipeBarrier<PIPE_V>();
                    Muls(tmp_,alpha_,-1.0f,LANES); PipeBarrier<PIPE_V>();
                    Adds(tmp_,tmp_,1.0f,LANES); PipeBarrier<PIPE_V>();
                    Mul(trans_,trans_,tmp_,LANES); PipeBarrier<PIPE_V>();
                }
                Ln(tmp_,trans_,LANES); PipeBarrier<PIPE_ALL>();
                DataCopy(color_[tile*pixels_+pixel],red_,LANES);
                DataCopy(color_[(tiles_+tile)*pixels_+pixel],green_,LANES);
                DataCopy(color_[(2*tiles_+tile)*pixels_+pixel],blue_,LANES);
                DataCopy(depth_[tile*pixels_+pixel],dep_,LANES);
                DataCopy(last_[tile*pixels_+pixel],tmp_,LANES);
                DataCopy(error_[tile*pixels_+pixel],zero_,LANES);
                PipeBarrier<PIPE_ALL>();
            }
            gsClip_.SetValue(tile,end);
            // Reserved ABI outputs, not valid backward state. Backward is rejected.
            for (int64_t item=first; item<end; ++item) {
                alphaClip_.SetValue(item*2,0);
                alphaClip_.SetValue(item*2+1,0);
            }
        }
    }
private:
    TPipe pipe_; TBuf<QuePosition::VECCALC> buffer_;
    GlobalTensor<float> gs_,coords_,color_,depth_,last_,error_;
    GlobalTensor<int64_t> schedule_,ids_,gsClip_;
    GlobalTensor<uint8_t> alphaClip_;
    LocalTensor<float> px_,py_,dx_,dy_,tmp_,tmp2_,sigma_,alpha_,trans_,next_,weight_,done_,one_;
    LocalTensor<float> red_,green_,blue_,dep_,zero_;
    LocalTensor<uint8_t> mask_;
    int64_t pixels_,tiles_,vector_,vectors_;
};

extern "C" __global__ __aicore__ void calc_render_fwd_double_clip_gsids(
    GM_ADDR gs, GM_ADDR coords, GM_ADDR offsets, GM_ADDR ids, GM_ADDR color,
    GM_ADDR depth, GM_ADDR last, GM_ADDR error, GM_ADDR gsClip, GM_ADDR alphaClip,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetAtomicNone();
    auto values=reinterpret_cast<__gm__ int64_t*>(tiling);
    MTGSCompatibleRender renderer;
    renderer.Init(gs,coords,offsets,ids,color,depth,last,error,gsClip,alphaClip,values[0],values[1]);
    renderer.Process();
}
