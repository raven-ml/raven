# tolk D175 D450 golden=4fbc8a9e2e866edb7bb5d782510e360e
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.RANGE dtypes.weakint [6] (AxisType.WARP, -1)
8 Ops.CONST dtypes.weakint [] 16
9 Ops.DIV dtypes.weakint [7, 8]
10 Ops.MOD dtypes.weakint [9, 4]
11 Ops.ADD dtypes.weakint [5, 10]
12 Ops.MUL dtypes.weakint [11, 4]
13 Ops.CONST dtypes.weakint [] 4
14 Ops.DIV dtypes.weakint [7, 13]
15 Ops.MOD dtypes.weakint [14, 4]
16 Ops.ADD dtypes.weakint [12, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.DIV dtypes.weakint [7, 4]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [17, 19]
21 Ops.CONST dtypes.weakint [] 64
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 3)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.DIV dtypes.weakint [7, 2]
26 Ops.MOD dtypes.weakint [25, 4]
27 Ops.ADD dtypes.weakint [24, 26]
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.CONST dtypes.weakint [] 1
30 Ops.DIV dtypes.weakint [7, 29]
31 Ops.MOD dtypes.weakint [30, 4]
32 Ops.ADD dtypes.weakint [28, 31]
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.ADD dtypes.weakint [22, 35]
37 Ops.INDEX dtypes.f32 [1, 36]
38 Ops.CONST dtypes.weakint [] 16384
39 Ops.PARAM dtypes.bf16 [38] ParamArg(slot=1, dtype=dtypes.bf16)
40 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
41 Ops.MUL dtypes.weakint [40, 6]
42 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0)
43 Ops.MUL dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [43, 26]
45 Ops.MUL dtypes.weakint [44, 4]
46 Ops.ADD dtypes.weakint [45, 31]
47 Ops.MUL dtypes.weakint [46, 4]
48 Ops.ADD dtypes.weakint [47, 34]
49 Ops.ADD dtypes.weakint [41, 48]
50 Ops.CONST dtypes.weakint [] 256
51 Ops.MUL dtypes.weakint [20, 50]
52 Ops.ADD dtypes.weakint [49, 51]
53 Ops.INDEX dtypes.bf16 [39, 52]
54 Ops.CAST dtypes.f32 [53] dtypes.f32
55 Ops.PARAM dtypes.i8 [38] ParamArg(slot=2, dtype=dtypes.i8)
56 Ops.ADD dtypes.weakint [43, 10]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.ADD dtypes.weakint [57, 15]
59 Ops.MUL dtypes.weakint [58, 4]
60 Ops.ADD dtypes.weakint [59, 19]
61 Ops.ADD dtypes.weakint [41, 60]
62 Ops.MUL dtypes.weakint [35, 50]
63 Ops.ADD dtypes.weakint [61, 62]
64 Ops.INDEX dtypes.i8 [55, 63]
65 Ops.CAST dtypes.f32 [64] dtypes.f32
66 Ops.CONST dtypes.weakfloat [] 0.0
67 Ops.CAST dtypes.f32 [66] dtypes.f32
68 Ops.STACK dtypes.f32 [67, 67]
69 Ops.WMMA dtypes.f32 [54, 65, 68] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),)))
70 Ops.REDUCE dtypes.f32 [69, 42] (Ops.ADD, 0)
71 Ops.CONST dtypes.weakint [] 512
72 Ops.PARAM dtypes.f32 [71] ParamArg(slot=3, dtype=dtypes.f32)
73 Ops.MUL dtypes.weakint [35, 2]
74 Ops.ADD dtypes.weakint [73, 40]
75 Ops.INDEX dtypes.f32 [72, 74]
76 Ops.MUL dtypes.f32 [70, 75]
77 Ops.REDUCE dtypes.f32 [76, 40] (Ops.ADD, 0)
78 Ops.STORE dtypes.void [37, 77]
79 Ops.END dtypes.void [78, 3, 7, 23, 34]
80 Ops.SINK dtypes.void [79] KernelInfo(name="r_8_8_32_2_4_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

