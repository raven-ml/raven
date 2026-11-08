# tolk D73
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 16
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
10 Ops.CONST dtypes.weakint [] 4
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 4)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 7, 13]
15 Ops.CONST dtypes.weakint [] 64
16 Ops.MUL dtypes.weakint [7, 15]
17 Ops.ADD dtypes.weakint [16, 13]
18 Ops.INDEX dtypes.f32 [14, 17]
19 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 0, 0)
21 Ops.MUL dtypes.weakint [20, 2]
22 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 6)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.CONST dtypes.weakint [] 1
26 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 0, 1)
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
29 Ops.ADD dtypes.weakint [27, 28]
30 Ops.ADD dtypes.weakint [24, 29]
31 Ops.ADD dtypes.weakint [30, 16]
32 Ops.INDEX dtypes.f32 [19, 31]
33 Ops.CMPLT dtypes.bool [29, 25]
34 Ops.CONST dtypes.weakint [] 8192
35 Ops.PARAM dtypes.u8 [34] ParamArg(slot=2, dtype=dtypes.u8)
36 Ops.CONST dtypes.weakint [] 32
37 Ops.MUL dtypes.weakint [13, 36]
38 Ops.ADD dtypes.weakint [37, 23]
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
53 Ops.INDEX dtypes.u8 [35, 52]
54 Ops.CONST dtypes.weakint [] 15
55 Ops.AND dtypes.u8 [53, 54]
56 Ops.SHR dtypes.u8 [53, 10]
57 Ops.WHERE dtypes.u8 [33, 55, 56]
58 Ops.CAST dtypes.f32 [57] dtypes.f32
59 Ops.CONST dtypes.weakint [] 512
60 Ops.PARAM dtypes.f32 [59] ParamArg(slot=4, dtype=dtypes.f32)
61 Ops.MUL dtypes.weakint [13, 4]
62 Ops.DIV dtypes.weakint [23, 8]
63 Ops.ADD dtypes.weakint [61, 62]
64 Ops.CONST dtypes.weakint [] 128
65 Ops.MUL dtypes.weakint [49, 64]
66 Ops.ADD dtypes.weakint [63, 65]
67 Ops.INDEX dtypes.f32 [60, 66]
68 Ops.MUL dtypes.f32 [58, 67]
69 Ops.MUL dtypes.f32 [32, 68]
70 Ops.REDUCE dtypes.f32 [69, 20, 22, 26, 28] (Ops.ADD, 0)
71 Ops.STORE dtypes.void [18, 70]
72 Ops.END dtypes.void [71, 3, 6, 9, 12]
73 Ops.SINK dtypes.void [72] KernelInfo(name="r_8_16_2_4_4_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(8, AxisType.UPCAST))), optimized=True)

