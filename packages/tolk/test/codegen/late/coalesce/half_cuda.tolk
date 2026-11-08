# tolk D187 D261 golden=1912216c9cf82781b0cc738db5112b9f
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.f16 [1] ParamArg(slot=0, dtype=dtypes.f16)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 0)
5 Ops.MUL dtypes.weakint [4, 3]
6 Ops.SHRINK dtypes.f16 [2, 5, 3]
7 Ops.PARAM dtypes.f16 [1] ParamArg(slot=1, dtype=dtypes.f16)
8 Ops.SHRINK dtypes.f16 [7, 5, 3]
9 Ops.LOAD dtypes.f16 [8]
10 Ops.CONST dtypes.weakint [] 0
11 Ops.INDEX dtypes.f16 [9, 10]
12 Ops.CONST dtypes.weakint [] 1
13 Ops.INDEX dtypes.f16 [9, 12]
14 Ops.CONST dtypes.weakint [] 2
15 Ops.INDEX dtypes.f16 [9, 14]
16 Ops.CONST dtypes.weakint [] 3
17 Ops.INDEX dtypes.f16 [9, 16]
18 Ops.STACK dtypes.f16 [11, 13, 15, 17]
19 Ops.PARAM dtypes.f16 [1] ParamArg(slot=2, dtype=dtypes.f16)
20 Ops.SHRINK dtypes.f16 [19, 5, 3]
21 Ops.LOAD dtypes.f16 [20]
22 Ops.INDEX dtypes.f16 [21, 10]
23 Ops.INDEX dtypes.f16 [21, 12]
24 Ops.INDEX dtypes.f16 [21, 14]
25 Ops.INDEX dtypes.f16 [21, 16]
26 Ops.STACK dtypes.f16 [22, 23, 24, 25]
27 Ops.ADD dtypes.f16 [18, 26]
28 Ops.INDEX dtypes.f16 [27, 10]
29 Ops.INDEX dtypes.f16 [27, 12]
30 Ops.INDEX dtypes.f16 [27, 14]
31 Ops.INDEX dtypes.f16 [27, 16]
32 Ops.STACK dtypes.f16 [28, 29, 30, 31]
33 Ops.STORE dtypes.void [6, 32]
34 Ops.AFTER dtypes.void [0, 33, 33, 33, 33]
35 Ops.END dtypes.void [34, 4]
36 Ops.SINK dtypes.void [35] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)

