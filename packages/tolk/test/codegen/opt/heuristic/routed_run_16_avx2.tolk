# tolk D73
0 Ops.CONST dtypes.weakint [] 81920
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 160
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
9 Ops.MUL dtypes.weakint [8, 4]
10 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.AFTER dtypes.f32 [1, 7, 11]
13 Ops.CONST dtypes.weakint [] 64
14 Ops.MUL dtypes.weakint [7, 13]
15 Ops.ADD dtypes.weakint [14, 11]
16 Ops.INDEX dtypes.f32 [12, 15]
17 Ops.CONST dtypes.weakint [] 327680
18 Ops.PARAM dtypes.bf16 [17] ParamArg(slot=1, dtype=dtypes.bf16)
19 Ops.CONST dtypes.weakint [] 32
20 Ops.RANGE dtypes.weakint [19] (AxisType.WEAK, 0, 0)
21 Ops.CONST dtypes.weakint [] 4
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [21] (AxisType.UPCAST, 6)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.CONST dtypes.weakint [] 2
26 Ops.MUL dtypes.weakint [24, 25]
27 Ops.CONST dtypes.weakint [] 1
28 Ops.RANGE dtypes.weakint [27] (AxisType.WEAK, 0, 1)
29 Ops.MUL dtypes.weakint [28, 25]
30 Ops.RANGE dtypes.weakint [25] (AxisType.UPCAST, 5)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.ADD dtypes.weakint [26, 31]
33 Ops.CONST dtypes.weakint [] 256
34 Ops.MUL dtypes.weakint [7, 33]
35 Ops.ADD dtypes.weakint [32, 34]
36 Ops.INDEX dtypes.bf16 [18, 35]
37 Ops.CAST dtypes.f32 [36] dtypes.f32
38 Ops.CMPLT dtypes.bool [31, 27]
39 Ops.CONST dtypes.weakint [] 32768
40 Ops.PARAM dtypes.u8 [39] ParamArg(slot=2, dtype=dtypes.u8)
41 Ops.CONST dtypes.weakint [] 128
42 Ops.MUL dtypes.weakint [11, 41]
43 Ops.ADD dtypes.weakint [42, 24]
44 Ops.CONST dtypes.weakint [] 80
45 Ops.PARAM dtypes.i32 [44] ParamArg(slot=3, dtype=dtypes.i32)
46 Ops.CONST dtypes.weakint [] 16
47 Ops.DIV dtypes.weakint [7, 46]
48 Ops.INDEX dtypes.i32 [45, 47]
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
59 Ops.ADD dtypes.weakint [43, 58]
60 Ops.INDEX dtypes.u8 [40, 59]
61 Ops.CONST dtypes.weakint [] 15
62 Ops.AND dtypes.u8 [60, 61]
63 Ops.SHR dtypes.u8 [60, 21]
64 Ops.WHERE dtypes.u8 [38, 62, 63]
65 Ops.CAST dtypes.bf16 [64] dtypes.bf16
66 Ops.CONST dtypes.weakint [] 2048
67 Ops.PARAM dtypes.bf16 [66] ParamArg(slot=4, dtype=dtypes.bf16)
68 Ops.MUL dtypes.weakint [11, 4]
69 Ops.DIV dtypes.weakint [24, 46]
70 Ops.ADD dtypes.weakint [68, 69]
71 Ops.CONST dtypes.weakint [] 512
72 Ops.MUL dtypes.weakint [56, 71]
73 Ops.ADD dtypes.weakint [70, 72]
74 Ops.INDEX dtypes.bf16 [67, 73]
75 Ops.MUL dtypes.bf16 [65, 74]
76 Ops.CAST dtypes.f32 [75] dtypes.f32
77 Ops.MUL dtypes.f32 [37, 76]
78 Ops.REDUCE dtypes.f32 [77, 20, 23, 28, 30] (Ops.ADD, 0)
79 Ops.STORE dtypes.void [16, 78]
80 Ops.END dtypes.void [79, 3, 6, 8, 10]
81 Ops.SINK dtypes.void [80] KernelInfo(name="r_160_8_8_8_32_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

