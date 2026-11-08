# tolk D73
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 16
9 Ops.RANGE dtypes.weakint [8] (AxisType.GLOBAL, 2)
10 Ops.CONST dtypes.weakint [] 4
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 5)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 7, 13]
15 Ops.CONST dtypes.weakint [] 64
16 Ops.MUL dtypes.weakint [7, 15]
17 Ops.ADD dtypes.weakint [16, 13]
18 Ops.INDEX dtypes.f32 [14, 17]
19 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.CONST dtypes.weakint [] 1
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0, 0)
22 Ops.CONST dtypes.weakint [] 32
23 Ops.MUL dtypes.weakint [21, 22]
24 Ops.RANGE dtypes.weakint [22] (AxisType.LOCAL, 4)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0, 1)
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.ADD dtypes.weakint [26, 30]
32 Ops.ADD dtypes.weakint [31, 16]
33 Ops.INDEX dtypes.f32 [19, 32]
34 Ops.CMPLT dtypes.bool [30, 20]
35 Ops.CONST dtypes.weakint [] 8192
36 Ops.PARAM dtypes.u8 [35] ParamArg(slot=2, dtype=dtypes.u8)
37 Ops.MUL dtypes.weakint [13, 22]
38 Ops.ADD dtypes.weakint [37, 25]
39 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32)
40 Ops.DIV dtypes.weakint [7, 4]
41 Ops.INDEX dtypes.i32 [39, 40]
42 Ops.CONST dtypes.weakint [] 0
43 Ops.MAX dtypes.i32 [41, 42]
44 Ops.CONST dtypes.weakint [] -1
45 Ops.XOR dtypes.i32 [43, 44]
46 Ops.CONST dtypes.weakint [] -4
47 Ops.MAX dtypes.i32 [45, 46]
48 Ops.XOR dtypes.i32 [47, 44]
49 Ops.CAST dtypes.weakint [48] dtypes.weakint
50 Ops.CONST dtypes.weakint [] 2048
51 Ops.MUL dtypes.weakint [49, 50]
52 Ops.ADD dtypes.weakint [38, 51]
53 Ops.INDEX dtypes.u8 [36, 52]
54 Ops.CONST dtypes.weakint [] 15
55 Ops.AND dtypes.u8 [53, 54]
56 Ops.SHR dtypes.u8 [53, 10]
57 Ops.WHERE dtypes.u8 [34, 55, 56]
58 Ops.CAST dtypes.f32 [57] dtypes.f32
59 Ops.CONST dtypes.weakint [] 512
60 Ops.PARAM dtypes.f32 [59] ParamArg(slot=4, dtype=dtypes.f32)
61 Ops.MUL dtypes.weakint [13, 4]
62 Ops.DIV dtypes.weakint [25, 8]
63 Ops.ADD dtypes.weakint [61, 62]
64 Ops.CONST dtypes.weakint [] 128
65 Ops.MUL dtypes.weakint [49, 64]
66 Ops.ADD dtypes.weakint [63, 65]
67 Ops.INDEX dtypes.f32 [60, 66]
68 Ops.MUL dtypes.f32 [58, 67]
69 Ops.MUL dtypes.f32 [33, 68]
70 Ops.REDUCE dtypes.f32 [69, 21, 24, 27, 29] (Ops.ADD, 0)
71 Ops.STORE dtypes.void [18, 70]
72 Ops.END dtypes.void [71, 3, 6, 9, 12]
73 Ops.SINK dtypes.void [72] KernelInfo(name="r_8_16_32_2_4_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST))), optimized=True)

