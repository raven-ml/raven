typedef unsigned int uint;
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
__device__ __forceinline__ void tg_cp_async(void *d, const void *s, int ignore) { asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; cp.async.cg.shared.global [%0], [%1], 16, p; }" :: "r"((unsigned)__cvta_generic_to_shared(d)), "l"(s), "r"(ignore) : "memory"); }
template <class T, int N, int TR> __device__ __forceinline__ T tg_ldsm(const void *p) { T r; unsigned *u = (unsigned *)&r; unsigned a = (unsigned)__cvta_generic_to_shared(p); if (N == 4) { if (TR) { asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]) : "r"(a)); } else { asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]) : "r"(a)); } } else if (N == 2) { if (TR) { asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(u[0]), "=r"(u[1]) : "r"(a)); } else { asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(u[0]), "=r"(u[1]) : "r"(a)); } } else { if (TR) { asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(u[0]) : "r"(a)); } else { asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(u[0]) : "r"(a)); } } return r; }
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
  int alu9 = (lidx0>>3);
  int alu10 = (alu4+alu6+(alu0<<5)+((alu9^alu1)<<3));
  tg_cp_async(((half8*)((buf1+(alu10+1024)))), ((half8*)((data1_3360+(alu2+1792)))), !(alu3));
  tg_cp_async(((half8*)((buf1+alu10))), ((half8*)((data1_3360+alu2))), 0);
  tg_cp_async(((half8*)((buf2+alu8))), ((half8*)((data2_3584+alu8))), 0);
  tg_cp_async(((half8*)((buf2+alu7))), ((half8*)((data2_3584+alu7))), 0);
  asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  int alu16 = ((lidx0&7)<<2);
  int alu17 = (lidx2<<5);
  int alu18 = ((lidx0>>1)&3);
  int alu19 = (alu16>>3);
  int alu20 = (alu19<<3);
  int alu21 = ((alu19^1)<<3);
  int alu22 = ((alu19^2)<<3);
  int alu23 = ((alu19^3)<<3);
  bool alu24 = ((lidx0&1)!=0);
  bool alu25 = ((alu9&1)!=0);
  bool alu26 = (((lidx0>>4)&1)!=0);
  int alu27 = (alu25?1536:1024);
  int alu28 = (alu25?1600:1088);
  int alu29 = (alu24?alu28:alu27);
  int alu30 = (alu25?1544:1032);
  int alu31 = (alu25?1608:1096);
  int alu32 = (alu24?alu31:alu30);
  int alu33 = (alu25?1552:1040);
  int alu34 = (alu25?1616:1104);
  int alu35 = (alu24?alu34:alu33);
  int alu36 = (alu25?1560:1048);
  int alu37 = (alu25?1624:1112);
  int alu38 = (alu24?alu37:alu36);
  int alu39 = (alu25?512:0);
  int alu40 = (alu25?576:64);
  int alu41 = (alu24?alu40:alu39);
  int alu42 = (alu25?520:8);
  int alu43 = (alu25?584:72);
  int alu44 = (alu24?alu43:alu42);
  int alu45 = (alu25?528:16);
  int alu46 = (alu25?592:80);
  int alu47 = (alu24?alu46:alu45);
  int alu48 = (alu25?536:24);
  int alu49 = (alu25?600:88);
  int alu50 = (alu24?alu49:alu48);
  int alu51 = (alu26?(alu21+256):(alu20+256));
  int alu52 = (alu26?alu21:alu20);
  int alu53 = (alu25?alu51:alu52);
  int alu54 = (alu26?(alu21+512):(alu20+512));
  int alu55 = (alu26?(alu21+768):(alu20+768));
  int alu56 = (alu25?alu55:alu54);
  int alu57 = (alu26?(alu23+256):(alu22+256));
  int alu58 = (alu26?alu23:alu22);
  int alu59 = (alu25?alu57:alu58);
  int alu60 = (alu26?(alu23+512):(alu22+512));
  int alu61 = (alu26?(alu23+768):(alu22+768));
  int alu62 = (alu25?alu61:alu60);
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
    bool alu95 = (Lidx3<1);
    bool alu96 = (alu95&(alu1<3));
    half8 val4 = ((alu96&alu3)?*((half8*)((data1_3360+(alu2+1824)))):cast0);
    half8 val5 = (alu96?*((half8*)((data1_3360+(alu2+32)))):cast0);
    half8 val6 = (alu95?*((half8*)((data2_3584+(alu7+2048)))):cast0);
    half8 val7 = ((alu95&(lidx2<1))?*((half8*)((data2_3584+(alu5+3072)))):cast0);
    int alu97 = (Lidx3*-2048);
    int alu98 = (alu10+alu97);
    int alu99 = (alu7+alu97);
    int alu100 = (Lidx3<<11);
    int alu101 = (((alu16>>2)<<5)+(lidx1<<10)+((alu16&3)<<1)+alu100);
    tg_cp_async(((half8*)((buf1+(alu98+2048)))), ((half8*)((data1_3360+(alu2+32)))), !(alu96));
    tg_cp_async(((half8*)((buf1+(alu98+3072)))), ((half8*)((data1_3360+(alu2+1824)))), !((alu96&alu3)));
    tg_cp_async(((half8*)((buf2+(alu99+2048)))), ((half8*)((data2_3584+(alu7+2048)))), !(alu95));
    tg_cp_async(((half8*)((buf2+(alu99+3072)))), ((half8*)((data2_3584+(alu5+3072)))), !((alu95&(lidx2<1))));
    half8 val8 = tg_ldsm<half8, 4, 0>((buf1+(alu101+alu53)));
    half8 val9 = tg_ldsm<half8, 4, 0>((buf1+(alu101+alu56)));
    half8 val10 = tg_ldsm<half8, 4, 0>((buf1+(alu101+alu59)));
    half8 val11 = tg_ldsm<half8, 4, 0>((buf1+(alu101+alu62)));
    int alu106 = ((alu18<<7)+alu17+(alu18>>2)+alu100);
    half4 val12 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu29)));
    half4 val13 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu32)));
    half4 val14 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu35)));
    half4 val15 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu38)));
    half4 val16 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu41)));
    half4 val17 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu44)));
    half4 val18 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu47)));
    half4 val19 = tg_ldsm<half4, 2, 1>((buf2+(alu106+alu50)));
    float4 wmma0 = __WMMA_8_16_16_f16_f32(val8, val16, make_float4((*(buf0+0)),(*(buf0+1)),(*(buf0+8)),(*(buf0+9))));
    float4 wmma1 = __WMMA_8_16_16_f16_f32(val10, val12, wmma0);
    float4 wmma2 = __WMMA_8_16_16_f16_f32(val8, val17, make_float4((*(buf0+2)),(*(buf0+3)),(*(buf0+10)),(*(buf0+11))));
    float4 wmma3 = __WMMA_8_16_16_f16_f32(val10, val13, wmma2);
    float4 wmma4 = __WMMA_8_16_16_f16_f32(val8, val18, make_float4((*(buf0+4)),(*(buf0+5)),(*(buf0+12)),(*(buf0+13))));
    float4 wmma5 = __WMMA_8_16_16_f16_f32(val10, val14, wmma4);
    float4 wmma6 = __WMMA_8_16_16_f16_f32(val8, val19, make_float4((*(buf0+6)),(*(buf0+7)),(*(buf0+14)),(*(buf0+15))));
    float4 wmma7 = __WMMA_8_16_16_f16_f32(val10, val15, wmma6);
    float4 wmma8 = __WMMA_8_16_16_f16_f32(val9, val16, make_float4((*(buf0+16)),(*(buf0+17)),(*(buf0+24)),(*(buf0+25))));
    float4 wmma9 = __WMMA_8_16_16_f16_f32(val11, val12, wmma8);
    float4 wmma10 = __WMMA_8_16_16_f16_f32(val9, val17, make_float4((*(buf0+18)),(*(buf0+19)),(*(buf0+26)),(*(buf0+27))));
    float4 wmma11 = __WMMA_8_16_16_f16_f32(val11, val13, wmma10);
    float4 wmma12 = __WMMA_8_16_16_f16_f32(val9, val18, make_float4((*(buf0+20)),(*(buf0+21)),(*(buf0+28)),(*(buf0+29))));
    float4 wmma13 = __WMMA_8_16_16_f16_f32(val11, val14, wmma12);
    float4 wmma14 = __WMMA_8_16_16_f16_f32(val9, val19, make_float4((*(buf0+22)),(*(buf0+23)),(*(buf0+30)),(*(buf0+31))));
    float4 wmma15 = __WMMA_8_16_16_f16_f32(val11, val15, wmma14);
    *(buf0+0) = wmma1.x;
    *(buf0+1) = wmma1.y;
    *(buf0+10) = wmma3.z;
    *(buf0+11) = wmma3.w;
    *(buf0+12) = wmma5.z;
    *(buf0+13) = wmma5.w;
    *(buf0+14) = wmma7.z;
    *(buf0+15) = wmma7.w;
    *(buf0+16) = wmma9.x;
    *(buf0+17) = wmma9.y;
    *(buf0+18) = wmma11.x;
    *(buf0+19) = wmma11.y;
    *(buf0+2) = wmma3.x;
    *(buf0+20) = wmma13.x;
    *(buf0+21) = wmma13.y;
    *(buf0+22) = wmma15.x;
    *(buf0+23) = wmma15.y;
    *(buf0+24) = wmma9.z;
    *(buf0+25) = wmma9.w;
    *(buf0+26) = wmma11.z;
    *(buf0+27) = wmma11.w;
    *(buf0+28) = wmma13.z;
    *(buf0+29) = wmma13.w;
    *(buf0+3) = wmma3.y;
    *(buf0+30) = wmma15.z;
    *(buf0+31) = wmma15.w;
    *(buf0+4) = wmma5.x;
    *(buf0+5) = wmma5.y;
    *(buf0+6) = wmma7.x;
    *(buf0+7) = wmma7.y;
    *(buf0+8) = wmma1.z;
    *(buf0+9) = wmma1.w;
    asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  }
  int alu141 = ((alu0<<6)+(lidx1<<11)+((alu1<<1)+alu17));
  bool alu142 = (((lidx1<<5)+alu0)<36);
  *((float2*)((data0_3840+(alu141+1024)))) = make_float2((*(buf0+16)),(*(buf0+17)));
  *((float2*)((data0_3840+(alu141+1032)))) = make_float2((*(buf0+18)),(*(buf0+19)));
  *((float2*)((data0_3840+(alu141+1040)))) = make_float2((*(buf0+20)),(*(buf0+21)));
  *((float2*)((data0_3840+(alu141+1048)))) = make_float2((*(buf0+22)),(*(buf0+23)));
  if (alu142) {
    *((float2*)((data0_3840+(alu141+1536)))) = make_float2((*(buf0+24)),(*(buf0+25)));
  }
  if (alu142) {
    *((float2*)((data0_3840+(alu141+1544)))) = make_float2((*(buf0+26)),(*(buf0+27)));
  }
  if (alu142) {
    *((float2*)((data0_3840+(alu141+1552)))) = make_float2((*(buf0+28)),(*(buf0+29)));
  }
  if (alu142) {
    *((float2*)((data0_3840+(alu141+1560)))) = make_float2((*(buf0+30)),(*(buf0+31)));
  }
  *((float2*)((data0_3840+(alu141+16)))) = make_float2((*(buf0+4)),(*(buf0+5)));
  *((float2*)((data0_3840+(alu141+24)))) = make_float2((*(buf0+6)),(*(buf0+7)));
  *((float2*)((data0_3840+(alu141+512)))) = make_float2((*(buf0+8)),(*(buf0+9)));
  *((float2*)((data0_3840+(alu141+520)))) = make_float2((*(buf0+10)),(*(buf0+11)));
  *((float2*)((data0_3840+(alu141+528)))) = make_float2((*(buf0+12)),(*(buf0+13)));
  *((float2*)((data0_3840+(alu141+536)))) = make_float2((*(buf0+14)),(*(buf0+15)));
  *((float2*)((data0_3840+(alu141+8)))) = make_float2((*(buf0+2)),(*(buf0+3)));
  *((float2*)((data0_3840+alu141))) = make_float2((*(buf0+0)),(*(buf0+1)));
}