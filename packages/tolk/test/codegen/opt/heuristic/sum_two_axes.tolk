# tolk D280 golden=ffbd2d7fa75873430c1b761edc4747b7
0 Ops.CONST dtypes.weakint [] 112
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 7
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 3)
6 Ops.AFTER dtypes.f32 [1, 3, 5]
7 Ops.MUL dtypes.weakint [3, 4]
8 Ops.ADD dtypes.weakint [7, 5]
9 Ops.INDEX dtypes.f32 [6, 8]
10 Ops.CONST dtypes.weakint [] 17920
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 32
13 Ops.MUL dtypes.weakint [5, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 1)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.CONST dtypes.weakint [] 5
17 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 0)
18 Ops.CONST dtypes.weakint [] 224
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.ADD dtypes.weakint [15, 19]
21 Ops.CONST dtypes.weakint [] 1120
22 Ops.MUL dtypes.weakint [3, 21]
23 Ops.ADD dtypes.weakint [20, 22]
24 Ops.INDEX dtypes.f32 [11, 23]
25 Ops.REDUCE dtypes.f32 [24, 17, 14] (Ops.ADD, 0)
26 Ops.STORE dtypes.void [9, 25]
27 Ops.END dtypes.void [26, 3, 5]
28 Ops.SINK dtypes.void [27] KernelInfo()

