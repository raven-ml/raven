# tolk D187 golden=4c438dba798f623d1697c6ce80d12a2e
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.f64 [1] ParamArg(slot=0, dtype=dtypes.f64)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 0)
5 Ops.MUL dtypes.weakint [4, 3]
6 Ops.INDEX dtypes.f64 [2, 5]
7 Ops.PARAM dtypes.f64 [1] ParamArg(slot=1, dtype=dtypes.f64)
8 Ops.CONST dtypes.weakint [] 2
9 Ops.SHRINK dtypes.f64 [7, 5, 8]
10 Ops.LOAD dtypes.f64 [9]
11 Ops.CONST dtypes.weakint [] 0
12 Ops.INDEX dtypes.f64 [10, 11]
13 Ops.PARAM dtypes.f64 [1] ParamArg(slot=2, dtype=dtypes.f64)
14 Ops.SHRINK dtypes.f64 [13, 5, 8]
15 Ops.LOAD dtypes.f64 [14]
16 Ops.INDEX dtypes.f64 [15, 11]
17 Ops.ADD dtypes.f64 [12, 16]
18 Ops.STORE dtypes.void [6, 17]
19 Ops.CONST dtypes.weakint [] 1
20 Ops.ADD dtypes.weakint [5, 19]
21 Ops.INDEX dtypes.f64 [2, 20]
22 Ops.INDEX dtypes.f64 [10, 19]
23 Ops.INDEX dtypes.f64 [15, 19]
24 Ops.ADD dtypes.f64 [22, 23]
25 Ops.STORE dtypes.void [21, 24]
26 Ops.ADD dtypes.weakint [5, 8]
27 Ops.INDEX dtypes.f64 [2, 26]
28 Ops.SHRINK dtypes.f64 [7, 26, 8]
29 Ops.LOAD dtypes.f64 [28]
30 Ops.INDEX dtypes.f64 [29, 11]
31 Ops.SHRINK dtypes.f64 [13, 26, 8]
32 Ops.LOAD dtypes.f64 [31]
33 Ops.INDEX dtypes.f64 [32, 11]
34 Ops.ADD dtypes.f64 [30, 33]
35 Ops.STORE dtypes.void [27, 34]
36 Ops.CONST dtypes.weakint [] 3
37 Ops.ADD dtypes.weakint [5, 36]
38 Ops.INDEX dtypes.f64 [2, 37]
39 Ops.INDEX dtypes.f64 [29, 19]
40 Ops.INDEX dtypes.f64 [32, 19]
41 Ops.ADD dtypes.f64 [39, 40]
42 Ops.STORE dtypes.void [38, 41]
43 Ops.AFTER dtypes.void [0, 18, 25, 35, 42]
44 Ops.END dtypes.void [43, 4]
45 Ops.SINK dtypes.void [44] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)

