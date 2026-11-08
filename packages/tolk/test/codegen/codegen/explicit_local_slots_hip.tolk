# tolk D89 D342 golden=4269f7d029b7ca45c1b9876240c07b60
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.i32 [1, 3]
5 Ops.ALLOC dtypes.i32 [0] ParamArg(slot=19, dtype=dtypes.i32, addrspace=AddrSpace.LOCAL)
6 Ops.INDEX dtypes.i32 [5, 3]
7 Ops.ALLOC dtypes.i32 [0] ParamArg(slot=18, dtype=dtypes.i32, addrspace=AddrSpace.REG)
8 Ops.INDEX dtypes.i32 [7, 3]
9 Ops.STORE dtypes.void [8, 3]
10 Ops.CONST dtypes.weakint [] 4
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.RANGE dtypes.i32 [11] (AxisType.WEAK, 0)
13 Ops.AFTER dtypes.i32 [7, 9, 12]
14 Ops.INDEX dtypes.i32 [13, 3]
15 Ops.LOAD dtypes.i32 [14]
16 Ops.CAST dtypes.u32 [15] dtypes.u32
17 Ops.PARAM dtypes.i32 [10] ParamArg(slot=0, dtype=dtypes.i32)
18 Ops.INDEX dtypes.i32 [17, 12]
19 Ops.LOAD dtypes.i32 [18]
20 Ops.CAST dtypes.u32 [19] dtypes.u32
21 Ops.ADD dtypes.u32 [16, 20]
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.STORE dtypes.void [14, 22]
24 Ops.END dtypes.void [23, 12]
25 Ops.AFTER dtypes.i32 [7, 24]
26 Ops.INDEX dtypes.i32 [25, 3]
27 Ops.LOAD dtypes.i32 [26]
28 Ops.STORE dtypes.void [6, 27]
29 Ops.BARRIER dtypes.void [28]
30 Ops.AFTER dtypes.i32 [5, 29]
31 Ops.INDEX dtypes.i32 [30, 3]
32 Ops.LOAD dtypes.i32 [31]
33 Ops.CAST dtypes.u32 [32] dtypes.u32
34 Ops.ALLOC dtypes.i32 [0] ParamArg(slot=17, dtype=dtypes.i32, addrspace=AddrSpace.LOCAL)
35 Ops.INDEX dtypes.i32 [34, 3]
36 Ops.CONST dtypes.weakint [] 3
37 Ops.CAST dtypes.i32 [36] dtypes.i32
38 Ops.STORE dtypes.void [35, 37]
39 Ops.BARRIER dtypes.void [38]
40 Ops.AFTER dtypes.i32 [34, 39]
41 Ops.INDEX dtypes.i32 [40, 3]
42 Ops.LOAD dtypes.i32 [41]
43 Ops.CAST dtypes.u32 [42] dtypes.u32
44 Ops.ADD dtypes.u32 [33, 43]
45 Ops.CAST dtypes.i32 [44] dtypes.i32
46 Ops.STORE dtypes.void [4, 45]
47 Ops.SINK dtypes.void [46] KernelInfo(optimized=True)
48 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=19, dtype=dtypes.i32, addrspace=AddrSpace.LOCAL)
49 Ops.INDEX dtypes.i32 [48, 3]
50 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=18, dtype=dtypes.i32, addrspace=AddrSpace.REG)
51 Ops.INDEX dtypes.i32 [50, 3]
52 Ops.STORE dtypes.void [51, 3]
53 Ops.AFTER dtypes.i32 [50, 52, 12]
54 Ops.INDEX dtypes.i32 [53, 3]
55 Ops.LOAD dtypes.i32 [54]
56 Ops.CAST dtypes.u32 [55] dtypes.u32
57 Ops.ADD dtypes.u32 [56, 20]
58 Ops.CAST dtypes.i32 [57] dtypes.i32
59 Ops.STORE dtypes.void [54, 58]
60 Ops.END dtypes.void [59, 12]
61 Ops.AFTER dtypes.i32 [50, 60]
62 Ops.INDEX dtypes.i32 [61, 3]
63 Ops.LOAD dtypes.i32 [62]
64 Ops.STORE dtypes.void [49, 63]
65 Ops.BARRIER dtypes.void [64]
66 Ops.AFTER dtypes.i32 [48, 65]
67 Ops.INDEX dtypes.i32 [66, 3]
68 Ops.LOAD dtypes.i32 [67]
69 Ops.CAST dtypes.u32 [68] dtypes.u32
70 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=17, dtype=dtypes.i32, addrspace=AddrSpace.LOCAL)
71 Ops.INDEX dtypes.i32 [70, 3]
72 Ops.STORE dtypes.void [71, 37]
73 Ops.BARRIER dtypes.void [72]
74 Ops.AFTER dtypes.i32 [70, 73]
75 Ops.INDEX dtypes.i32 [74, 3]
76 Ops.LOAD dtypes.i32 [75]
77 Ops.CAST dtypes.u32 [76] dtypes.u32
78 Ops.ADD dtypes.u32 [69, 77]
79 Ops.CAST dtypes.i32 [78] dtypes.i32
80 Ops.STORE dtypes.void [4, 79]
81 Ops.SINK dtypes.void [80] KernelInfo(estimates=Estimates(ops=5, lds=36, mem=20), optimized=True)
82 Ops.LINEAR dtypes.void [10, 17, 0, 1, 50, 70, 48, 2, 3, 51, 11, 52, 12, 18, 19, 53, 54, 55, 20, 56, 57, 58, 59, 60, 71, 36, 37, 72, 73, 74, 75, 76, 61, 62, 63, 49, 64, 65, 66, 67, 68, 4, 77, 69, 78, 79, 80, 81]
83 Ops.SOURCE dtypes.void [] "\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 1))) test(int* data0_4, int* data1_1) {\n  int buf0[1];\n  __attribute__((shared, aligned(16)))int buf1[1];\n  __attribute__((shared, aligned(16)))int buf2[1];\n  *(buf0+0) = 0;\n  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {\n    int val0 = (*(data0_4+Lidx0));\n    *(buf0+0) = ((int)((((unsigned int)((*(buf0+0))))+((unsigned int)(val0)))));\n  }\n  *(buf1+0) = 3;\n  __builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\");\n  int val1 = (*(buf1+0));\n  *(buf2+0) = (*(buf0+0));\n  __builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\");\n  int val2 = (*(buf2+0));\n  *(data1_1+0) = ((int)((((unsigned int)(val2))+((unsigned int)(val1)))));\n}"
84 Ops.PROGRAM dtypes.void [81, 82, 83] ProgramInfo(params=(%17, %1), globals=(0, 1), outs=(1,), ins=(0,), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
85 Ops.SINK dtypes.void [47, 84]

