# tolk D175 D450 D493 golden=4fbc8a9e2e866edb7bb5d782510e360e
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
42 Ops.RANGE dtypes.weakint [29] (AxisType.WEAK, 0)
43 Ops.MUL dtypes.weakint [42, 13]
44 Ops.RANGE dtypes.weakint [13] (AxisType.UPCAST, 8)
45 Ops.ADD dtypes.weakint [43, 44]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 26]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 31]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 34]
52 Ops.ADD dtypes.weakint [41, 51]
53 Ops.CONST dtypes.weakint [] 256
54 Ops.MUL dtypes.weakint [20, 53]
55 Ops.ADD dtypes.weakint [52, 54]
56 Ops.INDEX dtypes.bf16 [39, 55]
57 Ops.CAST dtypes.f32 [56] dtypes.f32
58 Ops.PARAM dtypes.i8 [38] ParamArg(slot=2, dtype=dtypes.i8)
59 Ops.ADD dtypes.weakint [46, 10]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 15]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 19]
64 Ops.ADD dtypes.weakint [41, 63]
65 Ops.MUL dtypes.weakint [35, 53]
66 Ops.ADD dtypes.weakint [64, 65]
67 Ops.INDEX dtypes.i8 [58, 66]
68 Ops.CAST dtypes.f32 [67] dtypes.f32
69 Ops.CONST dtypes.weakfloat [] 0.0
70 Ops.CAST dtypes.f32 [69] dtypes.f32
71 Ops.STACK dtypes.f32 [70, 70]
72 Ops.WMMA dtypes.f32 [57, 68, 71] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),)))
73 Ops.REDUCE dtypes.f32 [72, 42, 44] (Ops.ADD, 0)
74 Ops.CONST dtypes.weakint [] 512
75 Ops.PARAM dtypes.f32 [74] ParamArg(slot=3, dtype=dtypes.f32)
76 Ops.MUL dtypes.weakint [35, 2]
77 Ops.ADD dtypes.weakint [76, 40]
78 Ops.INDEX dtypes.f32 [75, 77]
79 Ops.MUL dtypes.f32 [73, 78]
80 Ops.REDUCE dtypes.f32 [79, 40] (Ops.ADD, 0)
81 Ops.STORE dtypes.void [37, 80]
82 Ops.END dtypes.void [81, 3, 7, 23, 34]
83 Ops.SINK dtypes.void [82] KernelInfo(name="r_8_8_32_2_8_4", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=4, arg=(0, AxisType.UPCAST))), optimized=True)

