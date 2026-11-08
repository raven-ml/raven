# tolk D73
0 Ops.CONST dtypes.weakint [] 9216
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 18
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
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
19 Ops.CONST dtypes.weakint [] 36864
20 Ops.PARAM dtypes.bf16 [19] ParamArg(slot=1, dtype=dtypes.bf16)
21 Ops.CONST dtypes.weakint [] 32
22 Ops.RANGE dtypes.weakint [21] (AxisType.WEAK, 0, 0)
23 Ops.MUL dtypes.weakint [22, 10]
24 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 6)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.CONST dtypes.weakint [] 2
27 Ops.MUL dtypes.weakint [25, 26]
28 Ops.CONST dtypes.weakint [] 1
29 Ops.RANGE dtypes.weakint [28] (AxisType.WEAK, 0, 1)
30 Ops.MUL dtypes.weakint [29, 26]
31 Ops.RANGE dtypes.weakint [26] (AxisType.UPCAST, 5)
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.ADD dtypes.weakint [27, 32]
34 Ops.CONST dtypes.weakint [] 256
35 Ops.MUL dtypes.weakint [7, 34]
36 Ops.ADD dtypes.weakint [33, 35]
37 Ops.INDEX dtypes.bf16 [20, 36]
38 Ops.CAST dtypes.f32 [37] dtypes.f32
39 Ops.CMPLT dtypes.bool [32, 28]
40 Ops.CONST dtypes.weakint [] 32768
41 Ops.PARAM dtypes.u8 [40] ParamArg(slot=2, dtype=dtypes.u8)
42 Ops.CONST dtypes.weakint [] 128
43 Ops.MUL dtypes.weakint [13, 42]
44 Ops.ADD dtypes.weakint [43, 25]
45 Ops.CONST dtypes.weakint [] 9
46 Ops.PARAM dtypes.i32 [45] ParamArg(slot=3, dtype=dtypes.i32)
47 Ops.DIV dtypes.weakint [7, 8]
48 Ops.INDEX dtypes.i32 [46, 47]
49 Ops.CONST dtypes.weakint [] 0
50 Ops.MAX dtypes.i32 [48, 49]
51 Ops.CONST dtypes.weakint [] -1
52 Ops.XOR dtypes.i32 [50, 51]
53 Ops.CONST dtypes.weakint [] -4
54 Ops.MAX dtypes.i32 [52, 53]
55 Ops.XOR dtypes.i32 [54, 51]
56 Ops.CAST dtypes.weakint [55] dtypes.weakint
57 Ops.CONST dtypes.weakint [] 8192
58 Ops.MUL dtypes.weakint [56, 57]
59 Ops.ADD dtypes.weakint [44, 58]
60 Ops.INDEX dtypes.u8 [41, 59]
61 Ops.CONST dtypes.weakint [] 15
62 Ops.AND dtypes.u8 [60, 61]
63 Ops.SHR dtypes.u8 [60, 10]
64 Ops.WHERE dtypes.u8 [39, 62, 63]
65 Ops.CAST dtypes.bf16 [64] dtypes.bf16
66 Ops.CONST dtypes.weakint [] 2048
67 Ops.PARAM dtypes.bf16 [66] ParamArg(slot=4, dtype=dtypes.bf16)
68 Ops.MUL dtypes.weakint [13, 4]
69 Ops.DIV dtypes.weakint [25, 8]
70 Ops.ADD dtypes.weakint [68, 69]
71 Ops.CONST dtypes.weakint [] 512
72 Ops.MUL dtypes.weakint [56, 71]
73 Ops.ADD dtypes.weakint [70, 72]
74 Ops.INDEX dtypes.bf16 [67, 73]
75 Ops.MUL dtypes.bf16 [65, 74]
76 Ops.CAST dtypes.f32 [75] dtypes.f32
77 Ops.MUL dtypes.f32 [38, 76]
78 Ops.REDUCE dtypes.f32 [77, 22, 24, 29, 31] (Ops.ADD, 0)
79 Ops.STORE dtypes.void [18, 78]
80 Ops.END dtypes.void [79, 3, 6, 9, 12]
81 Ops.SINK dtypes.void [80] KernelInfo(name="r_18_16_8_4_32_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

