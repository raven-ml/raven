# tolk D450 D451 D493 golden=4fbc8a9e2e866edb7bb5d782510e360e
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.CONST dtypes.weakint [] 32
10 Ops.RANGE dtypes.weakint [9] (AxisType.WARP, -1)
11 Ops.CONST dtypes.weakint [] 16
12 Ops.DIV dtypes.weakint [10, 11]
13 Ops.MOD dtypes.weakint [12, 4]
14 Ops.ADD dtypes.weakint [8, 13]
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.CONST dtypes.weakint [] 8
17 Ops.DIV dtypes.weakint [10, 16]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [15, 18]
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.DIV dtypes.weakint [10, 2]
22 Ops.MOD dtypes.weakint [21, 4]
23 Ops.ADD dtypes.weakint [20, 22]
24 Ops.CONST dtypes.weakint [] 64
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.RANGE dtypes.weakint [16] (AxisType.GLOBAL, 3)
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.DIV dtypes.weakint [10, 4]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [27, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.DIV dtypes.weakint [10, 32]
34 Ops.MOD dtypes.weakint [33, 4]
35 Ops.ADD dtypes.weakint [31, 34]
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.ADD dtypes.weakint [25, 38]
40 Ops.INDEX dtypes.f32 [1, 39]
41 Ops.CONST dtypes.weakint [] 16384
42 Ops.PARAM dtypes.bf16 [41] ParamArg(slot=1, dtype=dtypes.bf16)
43 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 1)
44 Ops.MUL dtypes.weakint [43, 9]
45 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 0)
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
48 Ops.ADD dtypes.weakint [46, 47]
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 6]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 29]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 34]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
57 Ops.ADD dtypes.weakint [55, 56]
58 Ops.ADD dtypes.weakint [44, 57]
59 Ops.ADD dtypes.weakint [5, 37]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 13]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 18]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 22]
66 Ops.CONST dtypes.weakint [] 256
67 Ops.MUL dtypes.weakint [65, 66]
68 Ops.ADD dtypes.weakint [58, 67]
69 Ops.INDEX dtypes.bf16 [42, 68]
70 Ops.PARAM dtypes.i8 [41] ParamArg(slot=2, dtype=dtypes.i8)
71 Ops.ADD dtypes.weakint [55, 37]
72 Ops.ADD dtypes.weakint [44, 71]
73 Ops.ADD dtypes.weakint [27, 13]
74 Ops.MUL dtypes.weakint [73, 4]
75 Ops.ADD dtypes.weakint [74, 18]
76 Ops.MUL dtypes.weakint [75, 4]
77 Ops.ADD dtypes.weakint [76, 22]
78 Ops.MUL dtypes.weakint [77, 66]
79 Ops.ADD dtypes.weakint [72, 78]
80 Ops.INDEX dtypes.i8 [70, 79]
81 Ops.CAST dtypes.bf16 [80] dtypes.bf16
82 Ops.CONST dtypes.weakfloat [] 0.0
83 Ops.CAST dtypes.f32 [82] dtypes.f32
84 Ops.STACK dtypes.f32 [83, 83, 83, 83]
85 Ops.WMMA dtypes.f32 [69, 81, 84] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 1)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 1))))
86 Ops.REDUCE dtypes.f32 [85, 45, 47] (Ops.ADD, 0)
87 Ops.CONST dtypes.weakint [] 512
88 Ops.PARAM dtypes.f32 [87] ParamArg(slot=3, dtype=dtypes.f32)
89 Ops.MUL dtypes.weakint [38, 16]
90 Ops.ADD dtypes.weakint [89, 43]
91 Ops.INDEX dtypes.f32 [88, 90]
92 Ops.MUL dtypes.f32 [86, 91]
93 Ops.REDUCE dtypes.f32 [92, 43] (Ops.ADD, 0)
94 Ops.STORE dtypes.void [40, 93]
95 Ops.END dtypes.void [94, 3, 6, 10, 26, 37]
96 Ops.SINK dtypes.void [95] KernelInfo(name="r_4_8_32_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=6, arg=(0, AxisType.UPCAST))), optimized=True)

