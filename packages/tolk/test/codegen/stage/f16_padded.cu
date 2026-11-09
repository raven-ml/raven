typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
__device__ __forceinline__ void tg_cp_async(void *d, const void *s, int ignore) { asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; cp.async.cg.shared.global [%0], [%1], 16, p; }" :: "r"((unsigned)__cvta_generic_to_shared(d)), "l"(s), "r"(ignore) : "memory"); }
struct __align__(16) half8 { half x, y, z, w, a, b, c, d; }; __device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { half8 r={x, y, z, w, a, b, c, d}; return r; }
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
__device__ float4 __WMMA_8_16_16_f16_f32(half8 a, half4 b, float4 c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7},"
      "{%8, %9}, {%0, %1, %2, %3};"
    : "+r"(c_pk[0]), "+r"(c_pk[1]), "+r"(c_pk[2]), "+r"(c_pk[3])
    : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]), "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]));
  return c;
}
extern "C" __global__ void __launch_bounds__(128) r_2_2_32_2_2_2_2_4_2_2(float* data0_3840, half* data1_3360, half* data2_3584) {
  float buf0[32];
  __shared__ __align__(16) half buf1[4096];
  __shared__ __align__(16) half buf2[4096];
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 2 */
  int lidx2 = threadIdx.z; /* 2 */
  half8 cast0 = make_half8(((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)));
  int alu0 = (lidx0>>2);
  int alu1 = (lidx0&3);
  int alu2 = ((lidx1*448)+(lidx2*896)+(alu0*56)+(alu1<<3));
  bool alu3 = (((lidx2<<4)+alu0+(lidx1<<3))<28);
  half8 val0 = (alu3?*((half8*)((data1_3360+(alu2+1792)))):cast0);
  half8 val1 = (*((half8*)((data1_3360+alu2))));
  int alu4 = (lidx1<<8);
  int alu5 = ((lidx0<<3)+alu4);
  int alu6 = (lidx2<<9);
  int alu7 = (alu5+alu6);
  int alu8 = (alu7+1024);
  half8 val2 = (*((half8*)((data2_3584+alu8))));
  half8 val3 = (*((half8*)((data2_3584+alu7))));
  int alu9 = (alu0<<5);
  int alu10 = (lidx0>>3);
  int alu11 = (alu4+alu6+alu9+((alu10^alu1)<<3));
  tg_cp_async(((half8*)((buf1+(alu11+1024)))), ((half8*)((data1_3360+(alu2+1792)))), !(alu3));
  tg_cp_async(((half8*)((buf1+alu11))), ((half8*)((data1_3360+alu2))), 0);
  tg_cp_async(((half8*)((buf2+alu8))), ((half8*)((data2_3584+alu8))), 0);
  tg_cp_async(((half8*)((buf2+alu7))), ((half8*)((data2_3584+alu7))), 0);
  asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  int alu17 = (alu9+(lidx1<<10));
  int alu18 = (alu1<<1);
  int alu19 = (lidx2<<5);
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+10) = 0.0f;
  *(buf0+11) = 0.0f;
  *(buf0+12) = 0.0f;
  *(buf0+13) = 0.0f;
  *(buf0+14) = 0.0f;
  *(buf0+15) = 0.0f;
  *(buf0+16) = 0.0f;
  *(buf0+17) = 0.0f;
  *(buf0+18) = 0.0f;
  *(buf0+19) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+20) = 0.0f;
  *(buf0+21) = 0.0f;
  *(buf0+22) = 0.0f;
  *(buf0+23) = 0.0f;
  *(buf0+24) = 0.0f;
  *(buf0+25) = 0.0f;
  *(buf0+26) = 0.0f;
  *(buf0+27) = 0.0f;
  *(buf0+28) = 0.0f;
  *(buf0+29) = 0.0f;
  *(buf0+3) = 0.0f;
  *(buf0+30) = 0.0f;
  *(buf0+31) = 0.0f;
  *(buf0+4) = 0.0f;
  *(buf0+5) = 0.0f;
  *(buf0+6) = 0.0f;
  *(buf0+7) = 0.0f;
  *(buf0+8) = 0.0f;
  *(buf0+9) = 0.0f;
  for (int Lidx3 = 0; Lidx3 < 2; Lidx3++) {
    bool alu52 = (Lidx3<1);
    bool alu53 = (alu52&(alu1<3));
    half8 val4 = ((alu53&alu3)?*((half8*)((data1_3360+(alu2+1824)))):cast0);
    half8 val5 = (alu53?*((half8*)((data1_3360+(alu2+32)))):cast0);
    half8 val6 = (alu52?*((half8*)((data2_3584+(alu7+2048)))):cast0);
    half8 val7 = ((alu52&(lidx2<1))?*((half8*)((data2_3584+(alu5+3072)))):cast0);
    int alu54 = (Lidx3*-2048);
    int alu55 = (alu11+alu54);
    int alu56 = (alu7+alu54);
    int alu57 = (Lidx3<<11);
    int alu58 = (alu17+(alu10<<3)+alu18+alu57);
    tg_cp_async(((half8*)((buf1+(alu55+2048)))), ((half8*)((data1_3360+(alu2+32)))), !(alu53));
    tg_cp_async(((half8*)((buf1+(alu55+3072)))), ((half8*)((data1_3360+(alu2+1824)))), !((alu53&alu3)));
    tg_cp_async(((half8*)((buf2+(alu56+2048)))), ((half8*)((data2_3584+(alu7+2048)))), !(alu52));
    tg_cp_async(((half8*)((buf2+(alu56+3072)))), ((half8*)((data2_3584+(alu5+3072)))), !((alu52&(lidx2<1))));
    half val8 = (*(buf1+(alu58+1)));
    half val9 = (*(buf1+(alu58+256)));
    half val10 = (*(buf1+(alu58+257)));
    half val11 = (*(buf1+(alu58+512)));
    half val12 = (*(buf1+(alu58+513)));
    half val13 = (*(buf1+(alu58+768)));
    half val14 = (*(buf1+(alu58+769)));
    int alu63 = (alu17+((alu10^1)<<3)+alu18+alu57);
    half val15 = (*(buf1+(alu63+1)));
    half val16 = (*(buf1+(alu63+256)));
    half val17 = (*(buf1+(alu63+257)));
    half val18 = (*(buf1+(alu63+512)));
    half val19 = (*(buf1+(alu63+513)));
    half val20 = (*(buf1+(alu63+768)));
    half val21 = (*(buf1+(alu63+769)));
    int alu64 = (alu17+((alu10^2)<<3)+alu18+alu57);
    half val22 = (*(buf1+(alu64+1)));
    half val23 = (*(buf1+(alu64+256)));
    half val24 = (*(buf1+(alu64+257)));
    half val25 = (*(buf1+(alu64+512)));
    half val26 = (*(buf1+(alu64+513)));
    half val27 = (*(buf1+(alu64+768)));
    half val28 = (*(buf1+(alu64+769)));
    int alu65 = (alu17+((alu10^3)<<3)+alu18+alu57);
    half val29 = (*(buf1+(alu65+1)));
    half val30 = (*(buf1+(alu65+256)));
    half val31 = (*(buf1+(alu65+257)));
    half val32 = (*(buf1+(alu65+512)));
    half val33 = (*(buf1+(alu65+513)));
    half val34 = (*(buf1+(alu65+768)));
    half val35 = (*(buf1+(alu65+769)));
    half val36 = (*(buf1+alu58));
    half val37 = (*(buf1+alu63));
    half val38 = (*(buf1+alu64));
    half val39 = (*(buf1+alu65));
    int alu66 = ((alu1<<7)+alu19+alu0+alu57);
    half val40 = (*(buf2+(alu66+1024)));
    half val41 = (*(buf2+(alu66+1032)));
    half val42 = (*(buf2+(alu66+1040)));
    half val43 = (*(buf2+(alu66+1048)));
    half val44 = (*(buf2+(alu66+1088)));
    half val45 = (*(buf2+(alu66+1096)));
    half val46 = (*(buf2+(alu66+1104)));
    half val47 = (*(buf2+(alu66+1112)));
    half val48 = (*(buf2+(alu66+1536)));
    half val49 = (*(buf2+(alu66+1544)));
    half val50 = (*(buf2+(alu66+1552)));
    half val51 = (*(buf2+(alu66+1560)));
    half val52 = (*(buf2+(alu66+16)));
    half val53 = (*(buf2+(alu66+1600)));
    half val54 = (*(buf2+(alu66+1608)));
    half val55 = (*(buf2+(alu66+1616)));
    half val56 = (*(buf2+(alu66+1624)));
    half val57 = (*(buf2+(alu66+24)));
    half val58 = (*(buf2+(alu66+512)));
    half val59 = (*(buf2+(alu66+520)));
    half val60 = (*(buf2+(alu66+528)));
    half val61 = (*(buf2+(alu66+536)));
    half val62 = (*(buf2+(alu66+576)));
    half val63 = (*(buf2+(alu66+584)));
    half val64 = (*(buf2+(alu66+592)));
    half val65 = (*(buf2+(alu66+600)));
    half val66 = (*(buf2+(alu66+64)));
    half val67 = (*(buf2+(alu66+72)));
    half val68 = (*(buf2+(alu66+8)));
    half val69 = (*(buf2+(alu66+80)));
    half val70 = (*(buf2+(alu66+88)));
    half val71 = (*(buf2+alu66));
    half8 cast1 = make_half8(val11,val12,val13,val14,val18,val19,val20,val21);
    half8 cast2 = make_half8(val25,val26,val27,val28,val32,val33,val34,val35);
    half8 cast3 = make_half8(val36,val8,val9,val10,val37,val15,val16,val17);
    half8 cast4 = make_half8(val38,val22,val23,val24,val39,val29,val30,val31);
    half4 cast5 = make_half4(val40,val44,val48,val53);
    half4 cast6 = make_half4(val41,val45,val49,val54);
    half4 cast7 = make_half4(val42,val46,val50,val55);
    half4 cast8 = make_half4(val43,val47,val51,val56);
    half4 cast9 = make_half4(val52,val69,val60,val64);
    half4 cast10 = make_half4(val57,val70,val61,val65);
    half4 cast11 = make_half4(val68,val67,val59,val63);
    half4 cast12 = make_half4(val71,val66,val58,val62);
    float4 wmma0 = __WMMA_8_16_16_f16_f32(cast1, cast12, make_float4((*(buf0+16)),(*(buf0+17)),(*(buf0+24)),(*(buf0+25))));
    float4 wmma1 = __WMMA_8_16_16_f16_f32(cast2, cast5, wmma0);
    float4 wmma2 = __WMMA_8_16_16_f16_f32(cast1, cast11, make_float4((*(buf0+18)),(*(buf0+19)),(*(buf0+26)),(*(buf0+27))));
    float4 wmma3 = __WMMA_8_16_16_f16_f32(cast2, cast6, wmma2);
    float4 wmma4 = __WMMA_8_16_16_f16_f32(cast1, cast9, make_float4((*(buf0+20)),(*(buf0+21)),(*(buf0+28)),(*(buf0+29))));
    float4 wmma5 = __WMMA_8_16_16_f16_f32(cast2, cast7, wmma4);
    float4 wmma6 = __WMMA_8_16_16_f16_f32(cast1, cast10, make_float4((*(buf0+22)),(*(buf0+23)),(*(buf0+30)),(*(buf0+31))));
    float4 wmma7 = __WMMA_8_16_16_f16_f32(cast2, cast8, wmma6);
    float4 wmma8 = __WMMA_8_16_16_f16_f32(cast3, cast12, make_float4((*(buf0+0)),(*(buf0+1)),(*(buf0+8)),(*(buf0+9))));
    float4 wmma9 = __WMMA_8_16_16_f16_f32(cast4, cast5, wmma8);
    float4 wmma10 = __WMMA_8_16_16_f16_f32(cast3, cast11, make_float4((*(buf0+2)),(*(buf0+3)),(*(buf0+10)),(*(buf0+11))));
    float4 wmma11 = __WMMA_8_16_16_f16_f32(cast4, cast6, wmma10);
    float4 wmma12 = __WMMA_8_16_16_f16_f32(cast3, cast9, make_float4((*(buf0+4)),(*(buf0+5)),(*(buf0+12)),(*(buf0+13))));
    float4 wmma13 = __WMMA_8_16_16_f16_f32(cast4, cast7, wmma12);
    float4 wmma14 = __WMMA_8_16_16_f16_f32(cast3, cast10, make_float4((*(buf0+6)),(*(buf0+7)),(*(buf0+14)),(*(buf0+15))));
    float4 wmma15 = __WMMA_8_16_16_f16_f32(cast4, cast8, wmma14);
    *(buf0+0) = wmma9.x;
    *(buf0+1) = wmma9.y;
    *(buf0+10) = wmma11.z;
    *(buf0+11) = wmma11.w;
    *(buf0+12) = wmma13.z;
    *(buf0+13) = wmma13.w;
    *(buf0+14) = wmma15.z;
    *(buf0+15) = wmma15.w;
    *(buf0+16) = wmma1.x;
    *(buf0+17) = wmma1.y;
    *(buf0+18) = wmma3.x;
    *(buf0+19) = wmma3.y;
    *(buf0+2) = wmma11.x;
    *(buf0+20) = wmma5.x;
    *(buf0+21) = wmma5.y;
    *(buf0+22) = wmma7.x;
    *(buf0+23) = wmma7.y;
    *(buf0+24) = wmma1.z;
    *(buf0+25) = wmma1.w;
    *(buf0+26) = wmma3.z;
    *(buf0+27) = wmma3.w;
    *(buf0+28) = wmma5.z;
    *(buf0+29) = wmma5.w;
    *(buf0+3) = wmma11.y;
    *(buf0+30) = wmma7.z;
    *(buf0+31) = wmma7.w;
    *(buf0+4) = wmma13.x;
    *(buf0+5) = wmma13.y;
    *(buf0+6) = wmma15.x;
    *(buf0+7) = wmma15.y;
    *(buf0+8) = wmma9.z;
    *(buf0+9) = wmma9.w;
    asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  }
  int alu101 = ((alu0<<6)+(lidx1<<11)+(alu18+alu19));
  bool alu102 = (((lidx1<<5)+alu0)<36);
  *((float2*)((data0_3840+(alu101+1024)))) = make_float2((*(buf0+16)),(*(buf0+17)));
  *((float2*)((data0_3840+(alu101+1032)))) = make_float2((*(buf0+18)),(*(buf0+19)));
  *((float2*)((data0_3840+(alu101+1040)))) = make_float2((*(buf0+20)),(*(buf0+21)));
  *((float2*)((data0_3840+(alu101+1048)))) = make_float2((*(buf0+22)),(*(buf0+23)));
  if (alu102) {
    *((float2*)((data0_3840+(alu101+1536)))) = make_float2((*(buf0+24)),(*(buf0+25)));
  }
  if (alu102) {
    *((float2*)((data0_3840+(alu101+1544)))) = make_float2((*(buf0+26)),(*(buf0+27)));
  }
  if (alu102) {
    *((float2*)((data0_3840+(alu101+1552)))) = make_float2((*(buf0+28)),(*(buf0+29)));
  }
  if (alu102) {
    *((float2*)((data0_3840+(alu101+1560)))) = make_float2((*(buf0+30)),(*(buf0+31)));
  }
  *((float2*)((data0_3840+(alu101+16)))) = make_float2((*(buf0+4)),(*(buf0+5)));
  *((float2*)((data0_3840+(alu101+24)))) = make_float2((*(buf0+6)),(*(buf0+7)));
  *((float2*)((data0_3840+(alu101+512)))) = make_float2((*(buf0+8)),(*(buf0+9)));
  *((float2*)((data0_3840+(alu101+520)))) = make_float2((*(buf0+10)),(*(buf0+11)));
  *((float2*)((data0_3840+(alu101+528)))) = make_float2((*(buf0+12)),(*(buf0+13)));
  *((float2*)((data0_3840+(alu101+536)))) = make_float2((*(buf0+14)),(*(buf0+15)));
  *((float2*)((data0_3840+(alu101+8)))) = make_float2((*(buf0+2)),(*(buf0+3)));
  *((float2*)((data0_3840+alu101))) = make_float2((*(buf0+0)),(*(buf0+1)));
}