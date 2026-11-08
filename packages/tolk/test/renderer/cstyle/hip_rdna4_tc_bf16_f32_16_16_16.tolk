# tolk D105 D261 golden=482689d2956e201aa06d5724a6b8e569
typedef long unsigned int size_t;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);
typedef __bf16 __bf164 __attribute__((ext_vector_type(4)));
static inline __attribute__((device)) __bf164 make___bf164(__bf16 x, __bf16 y, __bf16 z, __bf16 w) { return { x, y, z, w }; }
typedef __bf16 __bf168 __attribute__((ext_vector_type(8)));
static inline __attribute__((device)) __bf168 make___bf168(__bf16 x, __bf16 y, __bf16 z, __bf16 w, __bf16 a, __bf16 b, __bf16 c, __bf16 d) { return { x, y, z, w, a, b, c, d }; }
typedef float float8 __attribute__((ext_vector_type(8)));
static inline __attribute__((device)) float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return { x, y, z, w, a, b, c, d }; }
#define __WMMA_16_16_16_bf16_f32 __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 32))) r_32_2_2_2_2(float* data0_256, __bf16* data1_512, __bf16* data2_512) {
  float buf0[8];
  int lidx0 = __ockl_get_local_id(0); /* 32 */
  int alu0 = (lidx0>>4);
  int alu1 = (lidx0&15);
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+3) = 0.0f;
  *(buf0+4) = 0.0f;
  *(buf0+5) = 0.0f;
  *(buf0+6) = 0.0f;
  *(buf0+7) = 0.0f;
  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {
    int alu10 = ((alu0<<6)+(Lidx0<<8)+alu1);
    __bf16 val0 = (*(data2_512+(alu10+128)));
    __bf16 val1 = (*(data2_512+(alu10+144)));
    __bf16 val2 = (*(data2_512+(alu10+16)));
    __bf16 val3 = (*(data2_512+(alu10+160)));
    __bf16 val4 = (*(data2_512+(alu10+176)));
    __bf16 val5 = (*(data2_512+(alu10+32)));
    __bf16 val6 = (*(data2_512+(alu10+48)));
    __bf16 val7 = (*(data2_512+alu10));
    int alu11 = ((alu0<<2)+(Lidx0<<4)+(alu1<<5));
    __bf164 val8 = (*((__bf164*)((data1_512+(alu11+8)))));
    __bf164 val9 = (*((__bf164*)((data1_512+alu11))));
    float8 wmma0 = __WMMA_16_16_16_bf16_f32(make___bf168(val9.x,val9.y,val9.z,val9.w,val8.x,val8.y,val8.z,val8.w), make___bf168(val7,val2,val5,val6,val0,val1,val3,val4), make_float8((*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3)),(*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7))));
    *(buf0+0) = wmma0[0];
    *(buf0+1) = wmma0[1];
    *(buf0+2) = wmma0[2];
    *(buf0+3) = wmma0[3];
    *(buf0+4) = wmma0[4];
    *(buf0+5) = wmma0[5];
    *(buf0+6) = wmma0[6];
    *(buf0+7) = wmma0[7];
  }
  int alu21 = ((alu0<<7)+alu1);
  *(data0_256+(alu21+112)) = (*(buf0+7));
  *(data0_256+(alu21+16)) = (*(buf0+1));
  *(data0_256+(alu21+32)) = (*(buf0+2));
  *(data0_256+(alu21+48)) = (*(buf0+3));
  *(data0_256+(alu21+64)) = (*(buf0+4));
  *(data0_256+(alu21+80)) = (*(buf0+5));
  *(data0_256+(alu21+96)) = (*(buf0+6));
  *(data0_256+alu21) = (*(buf0+0));
}
