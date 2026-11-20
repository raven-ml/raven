# tolk D73
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
10 Ops.MUL dtypes.weakint [9, 4]
11 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.MUL dtypes.weakint [12, 2]
14 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 4)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.AFTER dtypes.f32 [1, 7, 15]
17 Ops.CONST dtypes.weakint [] 64
18 Ops.MUL dtypes.weakint [7, 17]
19 Ops.ADD dtypes.weakint [18, 15]
20 Ops.INDEX dtypes.f32 [16, 19]
21 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
22 Ops.CONST dtypes.weakint [] 16
23 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0, 0)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.CONST dtypes.weakint [] 1
29 Ops.RANGE dtypes.weakint [28] (AxisType.WEAK, 0, 1)
30 Ops.MUL dtypes.weakint [29, 4]
31 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.ADD dtypes.weakint [27, 32]
34 Ops.ADD dtypes.weakint [33, 18]
35 Ops.INDEX dtypes.f32 [21, 34]
36 Ops.CMPLT dtypes.bool [32, 28]
37 Ops.CONST dtypes.weakint [] 8192
38 Ops.PARAM dtypes.u8 [37] ParamArg(slot=2, dtype=dtypes.u8)
39 Ops.CONST dtypes.weakint [] 32
40 Ops.MUL dtypes.weakint [15, 39]
41 Ops.ADD dtypes.weakint [40, 26]
42 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
43 Ops.DIV dtypes.weakint [7, 4]
44 Ops.INDEX dtypes.i32 [42, 43]
45 Ops.CONST dtypes.weakint [] 0
46 Ops.MAX dtypes.i32 [44, 45]
47 Ops.CONST dtypes.weakint [] -1
48 Ops.XOR dtypes.i32 [46, 47]
49 Ops.CONST dtypes.weakint [] -4
50 Ops.MAX dtypes.i32 [48, 49]
51 Ops.XOR dtypes.i32 [50, 47]
52 Ops.CAST dtypes.weakint [51] dtypes.weakint
53 Ops.CONST dtypes.weakint [] 2048
54 Ops.MUL dtypes.weakint [52, 53]
55 Ops.ADD dtypes.weakint [41, 54]
56 Ops.INDEX dtypes.u8 [38, 55]
57 Ops.CONST dtypes.weakint [] 15
58 Ops.AND dtypes.u8 [56, 57]
59 Ops.SHR dtypes.u8 [56, 8]
60 Ops.WHERE dtypes.u8 [36, 58, 59]
61 Ops.CAST dtypes.f32 [60] dtypes.f32
62 Ops.CONST dtypes.weakint [] 512
63 Ops.PARAM dtypes.f32 [62] ParamArg(slot=4, dtype=dtypes.f32)
64 Ops.MUL dtypes.weakint [15, 4]
65 Ops.DIV dtypes.weakint [26, 22]
66 Ops.ADD dtypes.weakint [64, 65]
67 Ops.CONST dtypes.weakint [] 128
68 Ops.MUL dtypes.weakint [52, 67]
69 Ops.ADD dtypes.weakint [66, 68]
70 Ops.INDEX dtypes.f32 [63, 69]
71 Ops.MUL dtypes.f32 [61, 70]
72 Ops.MUL dtypes.f32 [35, 71]
73 Ops.REDUCE dtypes.f32 [72, 23, 25, 29, 31] (Ops.ADD, 0)
74 Ops.STORE dtypes.void [20, 73]
75 Ops.END dtypes.void [74, 3, 6, 9, 11, 14]
76 Ops.SINK dtypes.void [75] KernelInfo(name="r_8_4_2_8_2_16_2_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(2, AxisType.UPCAST))), optimized=True)

