# tolk D73
0 Ops.CONST dtypes.weakint [] 12
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 3
4 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 0)
5 Ops.CONST dtypes.weakint [] 4
6 Ops.MUL dtypes.weakint [4, 5]
7 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 1)
8 Ops.ADD dtypes.weakint [6, 7]
9 Ops.INDEX dtypes.f32 [2, 8]
10 Ops.PARAM dtypes.i32 [3] ParamArg(slot=1, dtype=dtypes.i32)
11 Ops.INDEX dtypes.i32 [10, 4]
12 Ops.BITCAST dtypes.u32 [11] dtypes.u32
13 Ops.CONST dtypes.weakint [] 8
14 Ops.CMPLT dtypes.bool [12, 13]
15 Ops.CONST dtypes.weakint [] 32
16 Ops.PARAM dtypes.f32 [15] ParamArg(slot=2, dtype=dtypes.f32)
17 Ops.CAST dtypes.weakint [11] dtypes.weakint
18 Ops.MUL dtypes.weakint [17, 5]
19 Ops.ADD dtypes.weakint [18, 7]
20 Ops.INDEX dtypes.f32 [16, 19]
21 Ops.CONST dtypes.weakfloat [] 0.0
22 Ops.WHERE dtypes.f32 [14, 20, 21]
23 Ops.STORE dtypes.void [9, 22]
24 Ops.END dtypes.void [23, 4, 7]
25 Ops.SINK dtypes.void [24] KernelInfo()
26 Ops.ALLOC dtypes.i32 [3] ParamArg(slot=0, dtype=dtypes.i32, device="CPU")
27 Ops.PARAM dtypes.i32 [3] ParamArg(slot=0, dtype=dtypes.i32)
28 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 1)
29 Ops.INDEX dtypes.i32 [27, 28]
30 Ops.CONST dtypes.weakint [] 15
31 Ops.PARAM dtypes.i32 [30] ParamArg(slot=1, dtype=dtypes.i32)
32 Ops.CONST dtypes.weakint [] 5
33 Ops.MUL dtypes.weakint [28, 32]
34 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 0)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.INDEX dtypes.i32 [31, 35]
37 Ops.REDUCE dtypes.i32 [36, 34] (Ops.MAX, 0)
38 Ops.CONST dtypes.weakint [] 0
39 Ops.MAX dtypes.i32 [37, 38]
40 Ops.CONST dtypes.weakint [] -1
41 Ops.XOR dtypes.i32 [39, 40]
42 Ops.CONST dtypes.weakint [] -8
43 Ops.MAX dtypes.i32 [41, 42]
44 Ops.XOR dtypes.i32 [43, 40]
45 Ops.STORE dtypes.void [29, 44]
46 Ops.END dtypes.void [45, 28]
47 Ops.SINK dtypes.void [46] KernelInfo()
48 Ops.PARAM dtypes.i32 [30] ParamArg(slot=2, dtype=dtypes.i32, device="CPU")
49 Ops.CALL dtypes.void [47, 26, 48] CallInfo()
50 Ops.AFTER dtypes.i32 [26, 49]
51 Ops.PARAM dtypes.f32 [15] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
52 Ops.CALL dtypes.void [25, 1, 50, 51] CallInfo()
53 Ops.AFTER dtypes.f32 [1, 52]
54 Ops.SINK dtypes.void [53]

