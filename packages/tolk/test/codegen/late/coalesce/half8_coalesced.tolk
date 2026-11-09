# tolk D187 D261 golden=16b3880df8f8e8f5a50c1db9442290da
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 32
2 Ops.PARAM dtypes.f16 [1] ParamArg(slot=0, dtype=dtypes.f16)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 0)
5 Ops.CONST dtypes.weakint [] 8
6 Ops.MUL dtypes.weakint [4, 5]
7 Ops.SHRINK dtypes.f16 [2, 6, 5]
8 Ops.PARAM dtypes.f16 [1] ParamArg(slot=1, dtype=dtypes.f16)
9 Ops.SHRINK dtypes.f16 [8, 6, 5]
10 Ops.LOAD dtypes.f16 [9]
11 Ops.CONST dtypes.weakint [] 0
12 Ops.INDEX dtypes.f16 [10, 11]
13 Ops.PARAM dtypes.f16 [1] ParamArg(slot=2, dtype=dtypes.f16)
14 Ops.SHRINK dtypes.f16 [13, 6, 5]
15 Ops.LOAD dtypes.f16 [14]
16 Ops.INDEX dtypes.f16 [15, 11]
17 Ops.ADD dtypes.f16 [12, 16]
18 Ops.CONST dtypes.weakint [] 1
19 Ops.INDEX dtypes.f16 [10, 18]
20 Ops.INDEX dtypes.f16 [15, 18]
21 Ops.ADD dtypes.f16 [19, 20]
22 Ops.CONST dtypes.weakint [] 2
23 Ops.INDEX dtypes.f16 [10, 22]
24 Ops.INDEX dtypes.f16 [15, 22]
25 Ops.ADD dtypes.f16 [23, 24]
26 Ops.CONST dtypes.weakint [] 3
27 Ops.INDEX dtypes.f16 [10, 26]
28 Ops.INDEX dtypes.f16 [15, 26]
29 Ops.ADD dtypes.f16 [27, 28]
30 Ops.INDEX dtypes.f16 [10, 3]
31 Ops.INDEX dtypes.f16 [15, 3]
32 Ops.ADD dtypes.f16 [30, 31]
33 Ops.CONST dtypes.weakint [] 5
34 Ops.INDEX dtypes.f16 [10, 33]
35 Ops.INDEX dtypes.f16 [15, 33]
36 Ops.ADD dtypes.f16 [34, 35]
37 Ops.CONST dtypes.weakint [] 6
38 Ops.INDEX dtypes.f16 [10, 37]
39 Ops.INDEX dtypes.f16 [15, 37]
40 Ops.ADD dtypes.f16 [38, 39]
41 Ops.CONST dtypes.weakint [] 7
42 Ops.INDEX dtypes.f16 [10, 41]
43 Ops.INDEX dtypes.f16 [15, 41]
44 Ops.ADD dtypes.f16 [42, 43]
45 Ops.STACK dtypes.f16 [17, 21, 25, 29, 32, 36, 40, 44]
46 Ops.STORE dtypes.void [7, 45]
47 Ops.AFTER dtypes.void [0, 46, 46, 46, 46, 46, 46, 46, 46]
48 Ops.END dtypes.void [47, 4]
49 Ops.SINK dtypes.void [48] KernelInfo(name="E_4_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)),), optimized=True)

