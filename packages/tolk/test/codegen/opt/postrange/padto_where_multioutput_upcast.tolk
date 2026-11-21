# tolk D502 golden=8fa001718414101d6da5cf3915bd3bb8
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 1)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CMPLT dtypes.bool [7, 0]
9 Ops.CONST dtypes.bool [] True
10 Ops.AND dtypes.bool [8, 9]
11 Ops.CONST dtypes.bool [] Invalid
12 Ops.WHERE dtypes.weakint [10, 7, 11]
13 Ops.INDEX dtypes.i32 [1, 12]
14 Ops.PARAM dtypes.bool [0] ParamArg(slot=1, dtype=dtypes.bool)
15 Ops.CONST dtypes.weakint [] 16
16 Ops.WHERE dtypes.weakint [8, 7, 15]
17 Ops.INDEX dtypes.bool [14, 16]
18 Ops.CONST dtypes.weakint [] 2
19 Ops.CONST dtypes.weakint [] 0
20 Ops.WHERE dtypes.weakint [17, 18, 19]
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.STORE dtypes.void [13, 21]
23 Ops.END dtypes.void [22, 3, 6]
24 Ops.SINK dtypes.void [23] KernelInfo(name="E_4_8", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST))), optimized=True)

