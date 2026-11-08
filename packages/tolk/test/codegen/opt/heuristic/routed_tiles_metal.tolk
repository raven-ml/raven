# tolk D73
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [2] (AxisType.WARP, -1)
7 Ops.CONST dtypes.weakint [] 16
8 Ops.DIV dtypes.weakint [6, 7]
9 Ops.MOD dtypes.weakint [8, 4]
10 Ops.ADD dtypes.weakint [5, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.DIV dtypes.weakint [6, 12]
14 Ops.MOD dtypes.weakint [13, 4]
15 Ops.ADD dtypes.weakint [11, 14]
16 Ops.MUL dtypes.weakint [15, 4]
17 Ops.DIV dtypes.weakint [6, 4]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [16, 18]
20 Ops.CONST dtypes.weakint [] 8
21 Ops.RANGE dtypes.weakint [20] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.DIV dtypes.weakint [6, 20]
24 Ops.MOD dtypes.weakint [23, 4]
25 Ops.ADD dtypes.weakint [22, 24]
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.CONST dtypes.weakint [] 1
28 Ops.DIV dtypes.weakint [6, 27]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [26, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.AFTER dtypes.f32 [1, 19, 33]
35 Ops.CONST dtypes.weakint [] 64
36 Ops.MUL dtypes.weakint [19, 35]
37 Ops.ADD dtypes.weakint [36, 33]
38 Ops.INDEX dtypes.f32 [34, 37]
39 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
40 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0, 0)
41 Ops.MUL dtypes.weakint [40, 4]
42 Ops.ADD dtypes.weakint [41, 24]
43 Ops.MUL dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [43, 29]
45 Ops.MUL dtypes.weakint [44, 4]
46 Ops.ADD dtypes.weakint [45, 32]
47 Ops.MUL dtypes.weakint [46, 4]
48 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0, 1)
49 Ops.ADD dtypes.weakint [47, 48]
50 Ops.ADD dtypes.weakint [49, 36]
51 Ops.INDEX dtypes.bf16 [39, 50]
52 Ops.CAST dtypes.f32 [51] dtypes.f32
53 Ops.CMPLT dtypes.bool [48, 27]
54 Ops.CONST dtypes.weakint [] 8192
55 Ops.PARAM dtypes.u8 [54] ParamArg(slot=2, dtype=dtypes.u8)
56 Ops.MUL dtypes.weakint [33, 2]
57 Ops.ADD dtypes.weakint [41, 9]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.ADD dtypes.weakint [58, 14]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 18]
62 Ops.ADD dtypes.weakint [56, 61]
63 Ops.PARAM dtypes.i32 [12] ParamArg(slot=3, dtype=dtypes.i32)
64 Ops.CONST dtypes.weakint [] 0
65 Ops.ADD dtypes.weakint [5, 64]
66 Ops.MUL dtypes.weakint [65, 4]
67 Ops.ADD dtypes.weakint [66, 64]
68 Ops.MUL dtypes.weakint [67, 4]
69 Ops.ADD dtypes.weakint [68, 64]
70 Ops.DIV dtypes.weakint [69, 35]
71 Ops.INDEX dtypes.i32 [63, 70]
72 Ops.MAX dtypes.i32 [71, 64]
73 Ops.CONST dtypes.weakint [] -1
74 Ops.XOR dtypes.i32 [72, 73]
75 Ops.CONST dtypes.weakint [] -4
76 Ops.MAX dtypes.i32 [74, 75]
77 Ops.XOR dtypes.i32 [76, 73]
78 Ops.CAST dtypes.weakint [77] dtypes.weakint
79 Ops.CONST dtypes.weakint [] 2048
80 Ops.MUL dtypes.weakint [78, 79]
81 Ops.ADD dtypes.weakint [62, 80]
82 Ops.INDEX dtypes.u8 [55, 81]
83 Ops.CONST dtypes.weakint [] 15
84 Ops.AND dtypes.u8 [82, 83]
85 Ops.SHR dtypes.u8 [82, 12]
86 Ops.WHERE dtypes.u8 [53, 84, 85]
87 Ops.CAST dtypes.bf16 [86] dtypes.bf16
88 Ops.CONST dtypes.weakint [] 512
89 Ops.PARAM dtypes.bf16 [88] ParamArg(slot=4, dtype=dtypes.bf16)
90 Ops.MUL dtypes.weakint [33, 4]
91 Ops.DIV dtypes.weakint [61, 7]
92 Ops.ADD dtypes.weakint [90, 91]
93 Ops.CONST dtypes.weakint [] 128
94 Ops.MUL dtypes.weakint [78, 93]
95 Ops.ADD dtypes.weakint [92, 94]
96 Ops.INDEX dtypes.bf16 [89, 95]
97 Ops.MUL dtypes.bf16 [87, 96]
98 Ops.CAST dtypes.f32 [97] dtypes.f32
99 Ops.CONST dtypes.weakfloat [] 0.0
100 Ops.CAST dtypes.f32 [99] dtypes.f32
101 Ops.STACK dtypes.f32 [100, 100]
102 Ops.WMMA dtypes.f32 [52, 98, 101] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
103 Ops.REDUCE dtypes.f32 [102, 40, 48] (Ops.ADD, 0)
104 Ops.STORE dtypes.void [38, 103]
105 Ops.END dtypes.void [104, 3, 6, 21, 32]
106 Ops.SINK dtypes.void [105] KernelInfo(name="r_32_8_32_2_4_2", applied_opts=(Opt(op=OptOps.TC, axis=1, arg=(-1, 0, 1)),), optimized=True)

