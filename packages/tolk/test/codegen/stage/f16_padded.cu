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
extern "C" __global__ void __launch_bounds__(128) r_2_4_2_2_32_2_2_2_2_2(float* data0_3968, half* data1_3472, half* data2_3584) {
  float buf0[4];
  __shared__ __align__(16) half buf1[2048];
  __shared__ __align__(16) half buf2[1024];
  int gidx1 = blockIdx.y; /* 2 */
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 2 */
  int lidx2 = threadIdx.z; /* 2 */
  half8 cast0 = make_half8(((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)),((half)(0.0f)));
  int alu0 = (lidx0>>2);
  int alu1 = (lidx0&3);
  int alu2 = ((lidx1*448)+(lidx2*896)+(alu0*56)+((alu1<<3)+(gidx1*1792)));
  int alu3 = ((gidx1<<5)+alu0);
  bool alu4 = ((alu3+(lidx2<<4)+(lidx1<<3))<62);
  half8 val0 = (alu4?*((half8*)((data1_3472+alu2))):cast0);
  int gidx0 = blockIdx.x; /* 4 */
  int alu5 = (lidx1<<10);
  int alu6 = (lidx0>>1);
  int alu7 = (gidx0<<4);
  int alu8 = (lidx0&1);
  int alu9 = ((alu6<<6)+alu5+((alu8<<3)+alu7));
  bool alu10 = (lidx2<1);
  half8 val1 = (alu10?*((half8*)((data2_3584+alu9))):cast0);
  int alu11 = (lidx1<<8);
  int alu12 = (alu0<<5);
  int alu13 = (lidx0>>3);
  int alu14 = (alu11+(lidx2<<9)+alu12+((alu13^alu1)<<3));
  int alu15 = ((alu6<<4)+alu11+(((alu13&1)^alu8)<<3));
  tg_cp_async(((half8*)((buf1+alu14))), ((half8*)((data1_3472+alu2))), !(alu4));
  if (alu10) {
    tg_cp_async(((half8*)((buf2+alu15))), ((half8*)((data2_3584+alu9))), !(alu10));
  }
  asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  int alu21 = (alu12+(lidx1<<9));
  int alu22 = (alu1<<1);
  int alu23 = (lidx1<<4);
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  *(buf0+2) = 0.0f;
  *(buf0+3) = 0.0f;
  for (int Lidx3 = 0; Lidx3 < 2; Lidx3++) {
    bool alu28 = (Lidx3<1);
    half8 val2 = ((alu28&(alu1<3)&alu4)?*((half8*)((data1_3472+(alu2+32)))):cast0);
    half8 val3 = ((alu28&((alu23+alu6)<24)&alu10)?*((half8*)((data2_3584+(alu9+2048)))):cast0);
    int alu29 = ((((alu6&1)^lidx2)<<3)+(alu1<<5)+alu0+(Lidx3<<9));
    tg_cp_async(((half8*)((buf1+(alu14+(Lidx3*-1024)+1024)))), ((half8*)((data1_3472+(alu2+32)))), !((alu28&(alu1<3)&alu4)));
    if (alu10) {
      tg_cp_async(((half8*)((buf2+(alu15+(Lidx3*-512)+512)))), ((half8*)((data2_3584+(alu9+2048)))), !((alu28&((alu23+alu6)<24)&alu10)));
    }
    half val4 = (*(buf2+(alu29+128)));
    half val5 = (*(buf2+(alu29+144)));
    half val6 = (*(buf2+(alu29+16)));
    half val7 = (*(buf2+(alu29+256)));
    half val8 = (*(buf2+(alu29+272)));
    half val9 = (*(buf2+(alu29+384)));
    half val10 = (*(buf2+(alu29+400)));
    half val11 = (*(buf2+alu29));
    int alu34 = (Lidx3<<10);
    int alu35 = (alu21+(alu13<<3)+alu22+alu34);
    half2 val12 = (*((half2*)((buf1+(alu35+256)))));
    int alu36 = (alu21+((alu13^1)<<3)+alu22+alu34);
    half2 val13 = (*((half2*)((buf1+(alu36+256)))));
    int alu37 = (alu21+((alu13^2)<<3)+alu22+alu34);
    half2 val14 = (*((half2*)((buf1+(alu37+256)))));
    int alu38 = (alu21+((alu13^3)<<3)+alu22+alu34);
    half2 val15 = (*((half2*)((buf1+(alu38+256)))));
    half2 val16 = (*((half2*)((buf1+alu35))));
    half2 val17 = (*((half2*)((buf1+alu36))));
    half2 val18 = (*((half2*)((buf1+alu37))));
    half2 val19 = (*((half2*)((buf1+alu38))));
    float4 wmma0 = __WMMA_8_16_16_f16_f32(make_half8(val16.x,val16.y,val12.x,val12.y,val17.x,val17.y,val13.x,val13.y), make_half4(val11,val6,val4,val5), make_float4((*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))));
    float4 wmma1 = __WMMA_8_16_16_f16_f32(make_half8(val18.x,val18.y,val14.x,val14.y,val19.x,val19.y,val15.x,val15.y), make_half4(val7,val8,val9,val10), wmma0);
    *(buf0+0) = wmma1.x;
    *(buf0+1) = wmma1.y;
    *(buf0+2) = wmma1.z;
    *(buf0+3) = wmma1.w;
    asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();
  }
  int alu45 = ((alu0<<6)+(gidx1<<11)+alu5+(alu22+alu7+(lidx2<<3)));
  if (((alu3+alu23)<54)) {
    *((float2*)((data0_3968+(alu45+512)))) = make_float2((*(buf0+2)),(*(buf0+3)));
  }
  *((float2*)((data0_3968+alu45))) = make_float2((*(buf0+0)),(*(buf0+1)));
}