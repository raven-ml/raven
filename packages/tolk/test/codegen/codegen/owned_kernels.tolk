# tolk D89
0 Ops.CONST dtypes.weakint [] 256
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 64
4 Ops.PARAM dtypes.f32 [3] ParamArg(slot=0, dtype=dtypes.f32)
5 Ops.CONST dtypes.weakint [] 32
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 0)
7 Ops.CONST dtypes.weakfloat [] 0.0
8 Ops.CONST dtypes.weakint [] 8
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
10 Ops.PARAM dtypes.f32 [8] ParamArg(slot=3, dtype=dtypes.f32)
11 Ops.INDEX dtypes.f32 [10, 9]
12 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
13 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.CONST dtypes.weakint [] 4
15 Ops.CONST dtypes.weakint [] 2
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 3)
17 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
18 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 1)
19 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 2)
20 Ops.CAST dtypes.i32 [6] dtypes.i32
21 Ops.PARAM dtypes.f32 [15] ParamArg(slot=2, dtype=dtypes.f32)
22 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 2)
23 Ops.MUL dtypes.weakint [18, 5]
24 Ops.ADD dtypes.weakint [23, 19]
25 Ops.INDEX dtypes.f32 [4, 24]
26 Ops.MUL dtypes.weakint [18, 8]
27 Ops.ADD dtypes.weakint [26, 17]
28 Ops.INDEX dtypes.f32 [12, 27]
29 Ops.PARAM dtypes.i32 [15] ParamArg(slot=1, dtype=dtypes.i32)
30 Ops.INDEX dtypes.i32 [29, 18]
31 Ops.CMPEQ dtypes.bool [20, 30]
32 Ops.MUL dtypes.weakint [6, 8]
33 Ops.ADD dtypes.weakint [32, 17]
34 Ops.INDEX dtypes.f32 [2, 33]
35 Ops.WHERE dtypes.f32 [31, 34, 7]
36 Ops.REDUCE dtypes.f32 [35, 6] (Ops.ADD, 0)
37 Ops.STORE dtypes.void [28, 36]
38 Ops.END dtypes.void [37, 18, 17]
39 Ops.SINK dtypes.void [38] KernelInfo()
40 Ops.ADD dtypes.weakint [26, 9]
41 Ops.INDEX dtypes.f32 [13, 40]
42 Ops.MUL dtypes.weakint [16, 15]
43 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 4)
44 Ops.ADD dtypes.weakint [42, 43]
45 Ops.MUL dtypes.weakint [22, 14]
46 Ops.ADD dtypes.weakint [44, 45]
47 Ops.ADD dtypes.weakint [46, 26]
48 Ops.INDEX dtypes.f32 [12, 47]
49 Ops.INDEX dtypes.f32 [21, 18]
50 Ops.RECIP dtypes.f32 [49]
51 Ops.MUL dtypes.f32 [41, 50]
52 Ops.MUL dtypes.f32 [51, 11]
53 Ops.PARAM dtypes.f32 [3] ParamArg(slot=4, dtype=dtypes.f32)
54 Ops.MUL dtypes.weakint [46, 8]
55 Ops.ADD dtypes.weakint [54, 9]
56 Ops.INDEX dtypes.f32 [53, 55]
57 Ops.MUL dtypes.f32 [52, 56]
58 Ops.REDUCE dtypes.f32 [57, 9] (Ops.ADD, 0)
59 Ops.STORE dtypes.void [48, 58]
60 Ops.END dtypes.void [59, 18, 22, 16, 43]
61 Ops.SINK dtypes.void [60] KernelInfo()
62 Ops.MUL dtypes.weakint [19, 8]
63 Ops.ADD dtypes.weakint [62, 9]
64 Ops.INDEX dtypes.f32 [2, 63]
65 Ops.MUL dtypes.f32 [41, 64]
66 Ops.REDUCE dtypes.f32 [65, 9] (Ops.ADD, 0)
67 Ops.STORE dtypes.void [25, 66]
68 Ops.END dtypes.void [67, 18, 19]
69 Ops.SINK dtypes.void [68] KernelInfo()
70 Ops.SINK dtypes.void [39, 61, 69]
