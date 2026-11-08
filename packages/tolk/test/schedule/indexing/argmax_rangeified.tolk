# tolk D77 golden=c189177af3f08bcb61674716645e9430
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32, device="CPU")
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.INDEX dtypes.i32 [1, 2]
4 Ops.CONST dtypes.weakint [] 8
5 Ops.CONST dtypes.weakint [] 32
6 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
7 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 6)
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 7)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.INDEX dtypes.f32 [6, 10]
12 Ops.STAGE dtypes.f32 [11, 7, 9] BufferizeOpts(device="CPU")
13 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
14 Ops.INDEX dtypes.f32 [12, 2, 13]
15 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 4)
16 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 5)
17 Ops.INDEX dtypes.f32 [12, 15, 16]
18 Ops.REDUCE dtypes.f32 [17, 16] (Ops.MAX, 0)
19 Ops.STAGE dtypes.f32 [18, 15] BufferizeOpts(device="CPU", broadcast=True)
20 Ops.INDEX dtypes.f32 [19, 2]
21 Ops.CMPEQ dtypes.bool [14, 20]
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.CONST dtypes.bool [] True
24 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
25 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 3)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.CONST dtypes.weakint [] 7
28 Ops.CMPLT dtypes.bool [26, 27]
29 Ops.CONST dtypes.bool [] False
30 Ops.CMPEQ dtypes.bool [28, 29]
31 Ops.AND dtypes.bool [23, 30]
32 Ops.CONST dtypes.weakint [] -1
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.CONST dtypes.weakint [] 0
35 Ops.WHERE dtypes.i32 [31, 33, 34]
36 Ops.REDUCE dtypes.i32 [35, 25] (Ops.ADD, 0)
37 Ops.STAGE dtypes.i32 [36, 24] BufferizeOpts(device="CPU", broadcast=True)
38 Ops.INDEX dtypes.i32 [37, 13]
39 Ops.CONST dtypes.weakint [] 9
40 Ops.ADD dtypes.i32 [38, 39]
41 Ops.MUL dtypes.i32 [22, 40]
42 Ops.REDUCE dtypes.i32 [41, 13] (Ops.MAX, 0)
43 Ops.MUL dtypes.i32 [42, 32]
44 Ops.ADD dtypes.i32 [4, 43]
45 Ops.STORE dtypes.void [3, 44]
46 Ops.END dtypes.void [45, 2]
47 Ops.AFTER dtypes.i32 [1, 46]
48 Ops.SINK dtypes.void [47]

