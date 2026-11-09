# tolk D430 golden=1104cc336eb990eb5a7049728c7b7ba4
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=0, dtype=dtypes.bf16)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.bf16 [1, 3]
5 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] 0.0
8 Ops.CAST dtypes.f32 [7] dtypes.f32
9 Ops.STORE dtypes.void [6, 8]
10 Ops.CONST dtypes.weakint [] 3
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.RANGE dtypes.i32 [11] (AxisType.WEAK, 0)
13 Ops.AFTER dtypes.f32 [5, 9, 12]
14 Ops.INDEX dtypes.f32 [13, 3]
15 Ops.LOAD dtypes.f32 [14]
16 Ops.PARAM dtypes.bf16 [10] ParamArg(slot=1, dtype=dtypes.bf16)
17 Ops.INDEX dtypes.bf16 [16, 12]
18 Ops.LOAD dtypes.bf16 [17]
19 Ops.BITCAST dtypes.u16 [18] dtypes.u16
20 Ops.CAST dtypes.u32 [19] dtypes.u32
21 Ops.CONST dtypes.weakint [] 16
22 Ops.CAST dtypes.u32 [21] dtypes.u32
23 Ops.SHL dtypes.u32 [20, 22]
24 Ops.BITCAST dtypes.f32 [23] dtypes.f32
25 Ops.ADD dtypes.f32 [15, 24]
26 Ops.STORE dtypes.void [14, 25]
27 Ops.END dtypes.void [26, 12]
28 Ops.AFTER dtypes.f32 [5, 27]
29 Ops.INDEX dtypes.f32 [28, 3]
30 Ops.LOAD dtypes.f32 [29]
31 Ops.CMPNE dtypes.bool [30, 30]
32 Ops.BITCAST dtypes.u32 [30] dtypes.u32
33 Ops.CONST dtypes.weakint [] 4194304
34 Ops.CAST dtypes.u32 [33] dtypes.u32
35 Ops.OR dtypes.u32 [32, 34]
36 Ops.SHR dtypes.u32 [32, 22]
37 Ops.CAST dtypes.u32 [0] dtypes.u32
38 Ops.AND dtypes.u32 [36, 37]
39 Ops.ADD dtypes.u32 [32, 38]
40 Ops.CONST dtypes.weakint [] 32767
41 Ops.CAST dtypes.u32 [40] dtypes.u32
42 Ops.ADD dtypes.u32 [39, 41]
43 Ops.WHERE dtypes.u32 [31, 35, 42]
44 Ops.SHR dtypes.u32 [43, 22]
45 Ops.CAST dtypes.u16 [44] dtypes.u16
46 Ops.BITCAST dtypes.bf16 [45] dtypes.bf16
47 Ops.STORE dtypes.void [4, 46]
48 Ops.SINK dtypes.void [47] KernelInfo(name="r_3", optimized=True)
49 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
50 Ops.INDEX dtypes.f32 [49, 3]
51 Ops.STORE dtypes.void [50, 8]
52 Ops.AFTER dtypes.f32 [49, 51, 12]
53 Ops.INDEX dtypes.f32 [52, 3]
54 Ops.LOAD dtypes.f32 [53]
55 Ops.ADD dtypes.f32 [54, 24]
56 Ops.STORE dtypes.void [53, 55]
57 Ops.END dtypes.void [56, 12]
58 Ops.AFTER dtypes.f32 [49, 57]
59 Ops.INDEX dtypes.f32 [58, 3]
60 Ops.LOAD dtypes.f32 [59]
61 Ops.CMPNE dtypes.bool [60, 60]
62 Ops.BITCAST dtypes.u32 [60] dtypes.u32
63 Ops.OR dtypes.u32 [62, 34]
64 Ops.SHR dtypes.u32 [62, 22]
65 Ops.AND dtypes.u32 [64, 37]
66 Ops.ADD dtypes.u32 [62, 65]
67 Ops.ADD dtypes.u32 [66, 41]
68 Ops.WHERE dtypes.u32 [61, 63, 67]
69 Ops.SHR dtypes.u32 [68, 22]
70 Ops.CAST dtypes.u16 [69] dtypes.u16
71 Ops.BITCAST dtypes.bf16 [70] dtypes.bf16
72 Ops.STORE dtypes.void [4, 71]
73 Ops.SINK dtypes.void [72] KernelInfo(name="r_3", estimates=Estimates(ops=14, lds=8, mem=8), optimized=True)
74 Ops.LINEAR dtypes.void [0, 1, 10, 16, 49, 2, 3, 50, 7, 8, 11, 21, 22, 51, 12, 17, 18, 52, 53, 54, 19, 20, 23, 24, 55, 56, 57, 58, 59, 60, 4, 37, 40, 41, 33, 34, 62, 64, 65, 66, 67, 61, 63, 68, 69, 70, 71, 72, 73]
75 Ops.SOURCE dtypes.void [] "\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 1))) r_3(__bf16* data0_1, __bf16* data1_3) {\n  float buf0[1];\n  *(buf0+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 3; Lidx0++) {\n    __bf16 val0 = (*(data1_3+Lidx0));\n    *(buf0+0) = ((*(buf0+0))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (__bf16)(val0))))<<16u))));\n  }\n  float val1 = (*(buf0+0));\n  unsigned int cast0 = __builtin_bit_cast(unsigned int, (float)(val1));\n  unsigned int alu3 = ((val1!=val1)?(cast0|4194304u):(cast0+((cast0>>16u)&1u)+32767u));\n  *(data0_1+0) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu3>>16u)))));\n}"
76 Ops.PROGRAM dtypes.void [73, 74, 75] ProgramInfo(params=(%1, %16), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
77 Ops.SINK dtypes.void [48, 76]

