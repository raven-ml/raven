# tolk D280 golden=4f387a1cc588834579fa3a7c1f552ec0
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.i32 [1, 2]
4 Ops.INDEX dtypes.i32 [3, 2]
5 Ops.CONST dtypes.weakint [] 32
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 0)
7 Ops.CMPLT dtypes.bool [6, 0]
8 Ops.CONST dtypes.weakint [] 324
9 Ops.PARAM dtypes.bool [8] ParamArg(slot=1, dtype=dtypes.bool)
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [7, 10]
12 Ops.CONST dtypes.weakint [] 18
13 Ops.MUL dtypes.weakint [6, 12]
14 Ops.ADD dtypes.weakint [13, 2]
15 Ops.CONST dtypes.bool [] Invalid
16 Ops.WHERE dtypes.weakint [11, 14, 15]
17 Ops.INDEX dtypes.bool [9, 16]
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.CONST dtypes.weakint [] 0
20 Ops.CAST dtypes.i32 [19] dtypes.i32
21 Ops.WHERE dtypes.i32 [7, 18, 20]
22 Ops.REDUCE dtypes.i32 [21, 6] (Ops.ADD, 0)
23 Ops.STORE dtypes.void [4, 22]
24 Ops.END dtypes.void [23, 2]
25 Ops.SINK dtypes.void [24] KernelInfo(name="r_17_32", applied_opts=(Opt(op=OptOps.PADTO, axis=1, arg=32),), optimized=True)

