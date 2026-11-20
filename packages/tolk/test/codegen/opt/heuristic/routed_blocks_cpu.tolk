# tolk D73
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
9 Ops.MUL dtypes.weakint [8, 4]
10 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 4)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.AFTER dtypes.f32 [1, 7, 15]
17 Ops.CONST dtypes.weakint [] 64
18 Ops.MUL dtypes.weakint [7, 17]
19 Ops.ADD dtypes.weakint [18, 15]
20 Ops.INDEX dtypes.f32 [16, 19]
21 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
22 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0, 0)
23 Ops.MUL dtypes.weakint [22, 12]
24 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 7)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.CONST dtypes.weakint [] 1
28 Ops.RANGE dtypes.weakint [27] (AxisType.WEAK, 0, 1)
29 Ops.MUL dtypes.weakint [28, 4]
30 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.ADD dtypes.weakint [26, 31]
33 Ops.ADD dtypes.weakint [32, 18]
34 Ops.INDEX dtypes.f32 [21, 33]
35 Ops.CMPLT dtypes.bool [31, 27]
36 Ops.CONST dtypes.weakint [] 8192
37 Ops.PARAM dtypes.u8 [36] ParamArg(slot=2, dtype=dtypes.u8)
38 Ops.CONST dtypes.weakint [] 32
39 Ops.MUL dtypes.weakint [15, 38]
40 Ops.ADD dtypes.weakint [39, 25]
41 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
42 Ops.DIV dtypes.weakint [7, 4]
43 Ops.INDEX dtypes.i32 [41, 42]
44 Ops.CONST dtypes.weakint [] 0
45 Ops.MAX dtypes.i32 [43, 44]
46 Ops.CONST dtypes.weakint [] -1
47 Ops.XOR dtypes.i32 [45, 46]
48 Ops.CONST dtypes.weakint [] -4
49 Ops.MAX dtypes.i32 [47, 48]
50 Ops.XOR dtypes.i32 [49, 46]
51 Ops.CAST dtypes.weakint [50] dtypes.weakint
52 Ops.CONST dtypes.weakint [] 2048
53 Ops.MUL dtypes.weakint [51, 52]
54 Ops.ADD dtypes.weakint [40, 53]
55 Ops.INDEX dtypes.u8 [37, 54]
56 Ops.CONST dtypes.weakint [] 15
57 Ops.AND dtypes.u8 [55, 56]
58 Ops.SHR dtypes.u8 [55, 12]
59 Ops.WHERE dtypes.u8 [35, 57, 58]
60 Ops.CAST dtypes.f32 [59] dtypes.f32
61 Ops.CONST dtypes.weakint [] 512
62 Ops.PARAM dtypes.f32 [61] ParamArg(slot=4, dtype=dtypes.f32)
63 Ops.MUL dtypes.weakint [15, 4]
64 Ops.CONST dtypes.weakint [] 16
65 Ops.DIV dtypes.weakint [25, 64]
66 Ops.ADD dtypes.weakint [63, 65]
67 Ops.CONST dtypes.weakint [] 128
68 Ops.MUL dtypes.weakint [51, 67]
69 Ops.ADD dtypes.weakint [66, 68]
70 Ops.INDEX dtypes.f32 [62, 69]
71 Ops.MUL dtypes.f32 [60, 70]
72 Ops.MUL dtypes.f32 [34, 71]
73 Ops.REDUCE dtypes.f32 [72, 22, 24, 28, 30] (Ops.ADD, 0)
74 Ops.STORE dtypes.void [20, 73]
75 Ops.END dtypes.void [74, 3, 6, 8, 10, 14]
76 Ops.SINK dtypes.void [75] KernelInfo(name="r_8_8_2_4_2_8_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(4, AxisType.UPCAST))), optimized=True)

