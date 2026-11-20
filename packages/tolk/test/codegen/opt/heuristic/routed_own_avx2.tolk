# tolk D73
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
8 Ops.MUL dtypes.weakint [7, 2]
9 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 4)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.AFTER dtypes.f32 [1, 6, 10]
12 Ops.CONST dtypes.weakint [] 64
13 Ops.MUL dtypes.weakint [6, 12]
14 Ops.ADD dtypes.weakint [13, 10]
15 Ops.INDEX dtypes.f32 [11, 14]
16 Ops.CONST dtypes.weakint [] 16384
17 Ops.PARAM dtypes.bf16 [16] ParamArg(slot=1, dtype=dtypes.bf16)
18 Ops.CONST dtypes.weakint [] 32
19 Ops.RANGE dtypes.weakint [18] (AxisType.WEAK, 0, 0)
20 Ops.CONST dtypes.weakint [] 4
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.RANGE dtypes.weakint [20] (AxisType.UPCAST, 6)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.CONST dtypes.weakint [] 2
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.CONST dtypes.weakint [] 1
27 Ops.RANGE dtypes.weakint [26] (AxisType.WEAK, 0, 1)
28 Ops.MUL dtypes.weakint [27, 24]
29 Ops.RANGE dtypes.weakint [24] (AxisType.UPCAST, 5)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.ADD dtypes.weakint [25, 30]
32 Ops.CONST dtypes.weakint [] 256
33 Ops.MUL dtypes.weakint [6, 32]
34 Ops.ADD dtypes.weakint [31, 33]
35 Ops.INDEX dtypes.bf16 [17, 34]
36 Ops.CAST dtypes.f32 [35] dtypes.f32
37 Ops.CMPLT dtypes.bool [30, 26]
38 Ops.CONST dtypes.weakint [] 32768
39 Ops.PARAM dtypes.u8 [38] ParamArg(slot=2, dtype=dtypes.u8)
40 Ops.CONST dtypes.weakint [] 128
41 Ops.MUL dtypes.weakint [10, 40]
42 Ops.ADD dtypes.weakint [41, 23]
43 Ops.PARAM dtypes.i32 [20] ParamArg(slot=3, dtype=dtypes.i32)
44 Ops.CONST dtypes.weakint [] 16
45 Ops.DIV dtypes.weakint [6, 44]
46 Ops.INDEX dtypes.i32 [43, 45]
47 Ops.CONST dtypes.weakint [] 0
48 Ops.MAX dtypes.i32 [46, 47]
49 Ops.CONST dtypes.weakint [] -1
50 Ops.XOR dtypes.i32 [48, 49]
51 Ops.CONST dtypes.weakint [] -4
52 Ops.MAX dtypes.i32 [50, 51]
53 Ops.XOR dtypes.i32 [52, 49]
54 Ops.CAST dtypes.weakint [53] dtypes.weakint
55 Ops.CONST dtypes.weakint [] 8192
56 Ops.MUL dtypes.weakint [54, 55]
57 Ops.ADD dtypes.weakint [42, 56]
58 Ops.INDEX dtypes.u8 [39, 57]
59 Ops.CONST dtypes.weakint [] 15
60 Ops.AND dtypes.u8 [58, 59]
61 Ops.SHR dtypes.u8 [58, 20]
62 Ops.WHERE dtypes.u8 [37, 60, 61]
63 Ops.CAST dtypes.bf16 [62] dtypes.bf16
64 Ops.CONST dtypes.weakint [] 2048
65 Ops.PARAM dtypes.bf16 [64] ParamArg(slot=4, dtype=dtypes.bf16)
66 Ops.MUL dtypes.weakint [10, 2]
67 Ops.DIV dtypes.weakint [23, 44]
68 Ops.ADD dtypes.weakint [66, 67]
69 Ops.CONST dtypes.weakint [] 512
70 Ops.MUL dtypes.weakint [54, 69]
71 Ops.ADD dtypes.weakint [68, 70]
72 Ops.INDEX dtypes.bf16 [65, 71]
73 Ops.MUL dtypes.bf16 [63, 72]
74 Ops.CAST dtypes.f32 [73] dtypes.f32
75 Ops.MUL dtypes.f32 [36, 74]
76 Ops.REDUCE dtypes.f32 [75, 19, 22, 27, 29] (Ops.ADD, 0)
77 Ops.STORE dtypes.void [15, 76]
78 Ops.END dtypes.void [77, 3, 5, 7, 9]
79 Ops.SINK dtypes.void [78] KernelInfo(name="r_8_8_8_8_32_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

