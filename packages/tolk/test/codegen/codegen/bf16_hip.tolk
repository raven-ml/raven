# tolk D89 D430 golden=87e7c2d35b0d87fb9b544748fd8ee694
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 2
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHL dtypes.i32 [5, 7]
9 Ops.SHRINK dtypes.f32 [2, 8, 4]
10 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=1, dtype=dtypes.bf16)
11 Ops.SHRINK dtypes.bf16 [10, 8, 4]
12 Ops.LOAD dtypes.bf16 [11]
13 Ops.CONST dtypes.weakint [] 0
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.INDEX dtypes.bf16 [12, 14]
16 Ops.BITCAST dtypes.u16 [15] dtypes.u16
17 Ops.CAST dtypes.u32 [16] dtypes.u32
18 Ops.CAST dtypes.u32 [1] dtypes.u32
19 Ops.SHL dtypes.u32 [17, 18]
20 Ops.BITCAST dtypes.f32 [19] dtypes.f32
21 Ops.CONST dtypes.weakfloat [] 1.0
22 Ops.CAST dtypes.f32 [21] dtypes.f32
23 Ops.ADD dtypes.f32 [20, 22]
24 Ops.CMPNE dtypes.bool [23, 23]
25 Ops.BITCAST dtypes.u32 [23] dtypes.u32
26 Ops.CONST dtypes.weakint [] 4194304
27 Ops.CAST dtypes.u32 [26] dtypes.u32
28 Ops.OR dtypes.u32 [25, 27]
29 Ops.SHR dtypes.u32 [25, 18]
30 Ops.CONST dtypes.weakint [] 1
31 Ops.CAST dtypes.u32 [30] dtypes.u32
32 Ops.AND dtypes.u32 [29, 31]
33 Ops.ADD dtypes.u32 [25, 32]
34 Ops.CONST dtypes.weakint [] 32767
35 Ops.CAST dtypes.u32 [34] dtypes.u32
36 Ops.ADD dtypes.u32 [33, 35]
37 Ops.WHERE dtypes.u32 [24, 28, 36]
38 Ops.SHR dtypes.u32 [37, 18]
39 Ops.CAST dtypes.u16 [38] dtypes.u16
40 Ops.CAST dtypes.u32 [39] dtypes.u32
41 Ops.SHL dtypes.u32 [40, 18]
42 Ops.BITCAST dtypes.f32 [41] dtypes.f32
43 Ops.CAST dtypes.i32 [30] dtypes.i32
44 Ops.INDEX dtypes.bf16 [12, 43]
45 Ops.BITCAST dtypes.u16 [44] dtypes.u16
46 Ops.CAST dtypes.u32 [45] dtypes.u32
47 Ops.SHL dtypes.u32 [46, 18]
48 Ops.BITCAST dtypes.f32 [47] dtypes.f32
49 Ops.ADD dtypes.f32 [48, 22]
50 Ops.CMPNE dtypes.bool [49, 49]
51 Ops.BITCAST dtypes.u32 [49] dtypes.u32
52 Ops.OR dtypes.u32 [51, 27]
53 Ops.SHR dtypes.u32 [51, 18]
54 Ops.AND dtypes.u32 [53, 31]
55 Ops.ADD dtypes.u32 [51, 54]
56 Ops.ADD dtypes.u32 [55, 35]
57 Ops.WHERE dtypes.u32 [50, 52, 56]
58 Ops.SHR dtypes.u32 [57, 18]
59 Ops.CAST dtypes.u16 [58] dtypes.u16
60 Ops.CAST dtypes.u32 [59] dtypes.u32
61 Ops.SHL dtypes.u32 [60, 18]
62 Ops.BITCAST dtypes.f32 [61] dtypes.f32
63 Ops.INDEX dtypes.bf16 [12, 7]
64 Ops.BITCAST dtypes.u16 [63] dtypes.u16
65 Ops.CAST dtypes.u32 [64] dtypes.u32
66 Ops.SHL dtypes.u32 [65, 18]
67 Ops.BITCAST dtypes.f32 [66] dtypes.f32
68 Ops.ADD dtypes.f32 [67, 22]
69 Ops.CMPNE dtypes.bool [68, 68]
70 Ops.BITCAST dtypes.u32 [68] dtypes.u32
71 Ops.OR dtypes.u32 [70, 27]
72 Ops.SHR dtypes.u32 [70, 18]
73 Ops.AND dtypes.u32 [72, 31]
74 Ops.ADD dtypes.u32 [70, 73]
75 Ops.ADD dtypes.u32 [74, 35]
76 Ops.WHERE dtypes.u32 [69, 71, 75]
77 Ops.SHR dtypes.u32 [76, 18]
78 Ops.CAST dtypes.u16 [77] dtypes.u16
79 Ops.CAST dtypes.u32 [78] dtypes.u32
80 Ops.SHL dtypes.u32 [79, 18]
81 Ops.BITCAST dtypes.f32 [80] dtypes.f32
82 Ops.CONST dtypes.weakint [] 3
83 Ops.CAST dtypes.i32 [82] dtypes.i32
84 Ops.INDEX dtypes.bf16 [12, 83]
85 Ops.BITCAST dtypes.u16 [84] dtypes.u16
86 Ops.CAST dtypes.u32 [85] dtypes.u32
87 Ops.SHL dtypes.u32 [86, 18]
88 Ops.BITCAST dtypes.f32 [87] dtypes.f32
89 Ops.ADD dtypes.f32 [88, 22]
90 Ops.CMPNE dtypes.bool [89, 89]
91 Ops.BITCAST dtypes.u32 [89] dtypes.u32
92 Ops.OR dtypes.u32 [91, 27]
93 Ops.SHR dtypes.u32 [91, 18]
94 Ops.AND dtypes.u32 [93, 31]
95 Ops.ADD dtypes.u32 [91, 94]
96 Ops.ADD dtypes.u32 [95, 35]
97 Ops.WHERE dtypes.u32 [90, 92, 96]
98 Ops.SHR dtypes.u32 [97, 18]
99 Ops.CAST dtypes.u16 [98] dtypes.u16
100 Ops.CAST dtypes.u32 [99] dtypes.u32
101 Ops.SHL dtypes.u32 [100, 18]
102 Ops.BITCAST dtypes.f32 [101] dtypes.f32
103 Ops.STACK dtypes.f32 [42, 62, 81, 102]
104 Ops.STORE dtypes.void [9, 103]
105 Ops.AFTER dtypes.void [0, 104]
106 Ops.SINK dtypes.void [105] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL))), optimized=True)
107 Ops.SINK dtypes.void [105] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL))), estimates=Estimates(ops=176, lds=96, mem=96), optimized=True)
108 Ops.LINEAR dtypes.void [1, 2, 10, 3, 4, 5, 6, 7, 8, 11, 12, 0, 13, 14, 15, 30, 43, 44, 63, 82, 83, 84, 21, 22, 16, 17, 31, 18, 34, 35, 26, 27, 19, 20, 23, 25, 29, 32, 33, 36, 24, 28, 37, 38, 39, 45, 46, 47, 48, 49, 51, 53, 54, 55, 56, 50, 52, 57, 58, 59, 64, 65, 66, 67, 68, 70, 72, 73, 74, 75, 69, 71, 76, 77, 78, 85, 86, 87, 88, 89, 91, 93, 94, 95, 96, 90, 92, 97, 98, 99, 40, 60, 79, 100, 41, 42, 61, 62, 80, 81, 101, 102, 103, 9, 104, 105, 107]
109 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\ntypedef __bf16 __bf164 __attribute__((ext_vector_type(4)));\nstatic inline __attribute__((device)) __bf164 make___bf164(__bf16 x, __bf16 y, __bf16 z, __bf16 w) { return { x, y, z, w }; }\ntypedef float float4 __attribute__((ext_vector_type(4)));\nstatic inline __attribute__((device)) float4 make_float4(float x, float y, float z, float w) { return { x, y, z, w }; }\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 4))) E_4_4(float* data0_16, __bf16* data1_16) {\n  int lidx0 = __ockl_get_local_id(0); /* 4 */\n  int alu0 = (lidx0<<2);\n  __bf164 val0 = (*((__bf164*)((data1_16+alu0))));\n  float alu1 = (__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (__bf16)(val0.x))))<<16u)))+1.0f);\n  unsigned int cast0 = __builtin_bit_cast(unsigned int, (float)(alu1));\n  unsigned int alu2 = ((alu1!=alu1)?(cast0|4194304u):(cast0+((cast0>>16u)&1u)+32767u));\n  float alu3 = (__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (__bf16)(val0.y))))<<16u)))+1.0f);\n  unsigned int cast1 = __builtin_bit_cast(unsigned int, (float)(alu3));\n  unsigned int alu4 = ((alu3!=alu3)?(cast1|4194304u):(cast1+((cast1>>16u)&1u)+32767u));\n  float alu5 = (__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (__bf16)(val0.z))))<<16u)))+1.0f);\n  unsigned int cast2 = __builtin_bit_cast(unsigned int, (float)(alu5));\n  unsigned int alu6 = ((alu5!=alu5)?(cast2|4194304u):(cast2+((cast2>>16u)&1u)+32767u));\n  float alu7 = (__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (__bf16)(val0.w))))<<16u)))+1.0f);\n  unsigned int cast3 = __builtin_bit_cast(unsigned int, (float)(alu7));\n  unsigned int alu8 = ((alu7!=alu7)?(cast3|4194304u):(cast3+((cast3>>16u)&1u)+32767u));\n  *((float4*)((data0_16+alu0))) = make_float4(__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu2>>16u)))))<<16u))),__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu4>>16u)))))<<16u))),__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu6>>16u)))))<<16u))),__builtin_bit_cast(float, (unsigned int)((((unsigned int)(((unsigned short)((alu8>>16u)))))<<16u))));\n}"
110 Ops.PROGRAM dtypes.void [107, 108, 109] ProgramInfo(local_size=(4, 1, 1), params=(%2, %10), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
111 Ops.SINK dtypes.void [106, 110]

