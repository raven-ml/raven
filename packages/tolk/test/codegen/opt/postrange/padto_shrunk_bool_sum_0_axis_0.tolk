# tolk D502 golden=5fb623fb914b5b63153e41eaf0313794
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CMPLT dtypes.bool [3, 0]
5 Ops.CONST dtypes.bool [] True
6 Ops.AND dtypes.bool [4, 5]
7 Ops.CONST dtypes.bool [] Invalid
8 Ops.WHERE dtypes.weakint [6, 3, 7]
9 Ops.INDEX dtypes.i32 [1, 8]
10 Ops.CONST dtypes.weakint [] 324
11 Ops.PARAM dtypes.bool [10] ParamArg(slot=1, dtype=dtypes.bool)
12 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
13 Ops.CONST dtypes.weakint [] 18
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 16
16 Ops.WHERE dtypes.weakint [4, 3, 15]
17 Ops.ADD dtypes.weakint [14, 16]
18 Ops.INDEX dtypes.bool [11, 17]
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.REDUCE dtypes.i32 [19, 12] (Ops.ADD, 0)
21 Ops.STORE dtypes.void [9, 20]
22 Ops.END dtypes.void [21, 3]
23 Ops.SINK dtypes.void [22] KernelInfo(name="r_32_17", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32),), optimized=True)

