# tolk D101 golden=8d8db871cd3716d1e3b5091ef1e717b9
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 1024
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.CONST dtypes.weakint [] 256
7 Ops.RANGE dtypes.weakint [6] (AxisType.LOCAL, 10)
8 Ops.CONST dtypes.weakint [] 4
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [5, 11]
13 Ops.REDUCE dtypes.f32 [12, 7, 10] (Ops.ADD, 0)
14 Ops.PARAM dtypes.f32 [4] ParamArg(slot=2, dtype=dtypes.f32)
15 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 1)
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [14, 16]
18 Ops.REDUCE dtypes.f32 [17, 7, 15] (Ops.ADD, 0)
19 Ops.ADD dtypes.f32 [13, 18]
20 Ops.PARAM dtypes.f32 [4] ParamArg(slot=3, dtype=dtypes.f32)
21 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
22 Ops.ADD dtypes.weakint [9, 21]
23 Ops.INDEX dtypes.f32 [20, 22]
24 Ops.REDUCE dtypes.f32 [23, 7, 21] (Ops.ADD, 0)
25 Ops.ADD dtypes.f32 [19, 24]
26 Ops.PARAM dtypes.f32 [4] ParamArg(slot=4, dtype=dtypes.f32)
27 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 3)
28 Ops.ADD dtypes.weakint [9, 27]
29 Ops.INDEX dtypes.f32 [26, 28]
30 Ops.REDUCE dtypes.f32 [29, 7, 27] (Ops.ADD, 0)
31 Ops.ADD dtypes.f32 [25, 30]
32 Ops.PARAM dtypes.f32 [4] ParamArg(slot=5, dtype=dtypes.f32)
33 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 4)
34 Ops.ADD dtypes.weakint [9, 33]
35 Ops.INDEX dtypes.f32 [32, 34]
36 Ops.REDUCE dtypes.f32 [35, 7, 33] (Ops.ADD, 0)
37 Ops.ADD dtypes.f32 [31, 36]
38 Ops.PARAM dtypes.f32 [4] ParamArg(slot=6, dtype=dtypes.f32)
39 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 5)
40 Ops.ADD dtypes.weakint [9, 39]
41 Ops.INDEX dtypes.f32 [38, 40]
42 Ops.REDUCE dtypes.f32 [41, 7, 39] (Ops.ADD, 0)
43 Ops.ADD dtypes.f32 [37, 42]
44 Ops.PARAM dtypes.f32 [4] ParamArg(slot=7, dtype=dtypes.f32)
45 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 6)
46 Ops.ADD dtypes.weakint [9, 45]
47 Ops.INDEX dtypes.f32 [44, 46]
48 Ops.REDUCE dtypes.f32 [47, 7, 45] (Ops.ADD, 0)
49 Ops.ADD dtypes.f32 [43, 48]
50 Ops.PARAM dtypes.f32 [4] ParamArg(slot=8, dtype=dtypes.f32)
51 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 7)
52 Ops.ADD dtypes.weakint [9, 51]
53 Ops.INDEX dtypes.f32 [50, 52]
54 Ops.REDUCE dtypes.f32 [53, 7, 51] (Ops.ADD, 0)
55 Ops.ADD dtypes.f32 [49, 54]
56 Ops.PARAM dtypes.f32 [4] ParamArg(slot=9, dtype=dtypes.f32)
57 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 8)
58 Ops.ADD dtypes.weakint [9, 57]
59 Ops.INDEX dtypes.f32 [56, 58]
60 Ops.REDUCE dtypes.f32 [59, 7, 57] (Ops.ADD, 0)
61 Ops.ADD dtypes.f32 [55, 60]
62 Ops.PARAM dtypes.f32 [4] ParamArg(slot=10, dtype=dtypes.f32)
63 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 9)
64 Ops.ADD dtypes.weakint [9, 63]
65 Ops.INDEX dtypes.f32 [62, 64]
66 Ops.REDUCE dtypes.f32 [65, 7, 63] (Ops.ADD, 0)
67 Ops.ADD dtypes.f32 [61, 66]
68 Ops.STORE dtypes.void [3, 67]
69 Ops.SINK dtypes.void [68] KernelInfo(name="r_256_4_4_4_4_4_4_4_4_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(256, AxisType.LOCAL, True)),), optimized=True)

