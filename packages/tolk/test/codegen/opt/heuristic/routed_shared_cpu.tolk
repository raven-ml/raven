# tolk D73
0 Ops.CONST dtypes.weakint [] 16384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
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
19 Ops.CONST dtypes.weakint [] 65536
20 Ops.PARAM dtypes.bf16 [19] ParamArg(slot=1, dtype=dtypes.bf16)
21 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0, 0)
22 Ops.MUL dtypes.weakint [21, 10]
23 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 6)
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
36 Ops.INDEX dtypes.bf16 [20, 35]
37 Ops.CAST dtypes.f32 [36] dtypes.f32
38 Ops.CMPLT dtypes.bool [31, 27]
39 Ops.PARAM dtypes.u8 [0] ParamArg(slot=2, dtype=dtypes.u8)
40 Ops.CONST dtypes.weakint [] 128
41 Ops.MUL dtypes.weakint [13, 40]
42 Ops.ADD dtypes.weakint [41, 24]
43 Ops.PARAM dtypes.i32 [8] ParamArg(slot=3, dtype=dtypes.i32)
44 Ops.DIV dtypes.weakint [7, 8]
45 Ops.INDEX dtypes.i32 [43, 44]
46 Ops.CONST dtypes.weakint [] 0
47 Ops.MAX dtypes.i32 [45, 46]
48 Ops.CONST dtypes.weakint [] -1
49 Ops.XOR dtypes.i32 [47, 48]
50 Ops.CONST dtypes.weakint [] -2
51 Ops.MAX dtypes.i32 [49, 50]
52 Ops.XOR dtypes.i32 [51, 48]
53 Ops.CAST dtypes.weakint [52] dtypes.weakint
54 Ops.CONST dtypes.weakint [] 8192
55 Ops.MUL dtypes.weakint [53, 54]
56 Ops.ADD dtypes.weakint [42, 55]
57 Ops.INDEX dtypes.u8 [39, 56]
58 Ops.CONST dtypes.weakint [] 15
59 Ops.AND dtypes.u8 [57, 58]
60 Ops.SHR dtypes.u8 [57, 10]
61 Ops.WHERE dtypes.u8 [38, 59, 60]
62 Ops.CAST dtypes.bf16 [61] dtypes.bf16
63 Ops.CONST dtypes.weakint [] 1024
64 Ops.PARAM dtypes.bf16 [63] ParamArg(slot=4, dtype=dtypes.bf16)
65 Ops.MUL dtypes.weakint [13, 4]
66 Ops.DIV dtypes.weakint [24, 8]
67 Ops.ADD dtypes.weakint [65, 66]
68 Ops.CONST dtypes.weakint [] 512
69 Ops.MUL dtypes.weakint [53, 68]
70 Ops.ADD dtypes.weakint [67, 69]
71 Ops.INDEX dtypes.bf16 [64, 70]
72 Ops.MUL dtypes.bf16 [62, 71]
73 Ops.CAST dtypes.f32 [72] dtypes.f32
74 Ops.MUL dtypes.f32 [37, 73]
75 Ops.REDUCE dtypes.f32 [74, 21, 23, 28, 30] (Ops.ADD, 0)
76 Ops.STORE dtypes.void [18, 75]
77 Ops.END dtypes.void [76, 3, 6, 9, 12]
78 Ops.SINK dtypes.void [77] KernelInfo(name="r_32_16_8_4_32_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

