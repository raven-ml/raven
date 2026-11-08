# tolk D141 D280 golden=aacdeedfc6c8a4faf7a797862d267f35
0 Ops.CONST dtypes.weakint [] 7168
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 7
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 512
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 1)
10 Ops.CONST dtypes.weakint [] 2
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 7, 13]
15 Ops.CONST dtypes.weakint [] 1024
16 Ops.MUL dtypes.weakint [7, 15]
17 Ops.ADD dtypes.weakint [16, 13]
18 Ops.INDEX dtypes.f32 [14, 17]
19 Ops.CONST dtypes.weakint [] 4
20 Ops.CMPLT dtypes.bool [7, 19]
21 Ops.MOD dtypes.weakint [7, 19]
22 Ops.CMPLT dtypes.bool [21, 10]
23 Ops.MOD dtypes.weakint [7, 10]
24 Ops.CMPLT dtypes.bool [23, 2]
25 Ops.PARAM dtypes.f32 [15] ParamArg(slot=1, dtype=dtypes.f32)
26 Ops.INDEX dtypes.f32 [25, 13]
27 Ops.PARAM dtypes.f32 [15] ParamArg(slot=2, dtype=dtypes.f32)
28 Ops.INDEX dtypes.f32 [27, 13]
29 Ops.WHERE dtypes.f32 [24, 26, 28]
30 Ops.PARAM dtypes.f32 [15] ParamArg(slot=3, dtype=dtypes.f32)
31 Ops.INDEX dtypes.f32 [30, 13]
32 Ops.PARAM dtypes.f32 [15] ParamArg(slot=4, dtype=dtypes.f32)
33 Ops.INDEX dtypes.f32 [32, 13]
34 Ops.WHERE dtypes.f32 [24, 31, 33]
35 Ops.WHERE dtypes.f32 [22, 29, 34]
36 Ops.PARAM dtypes.f32 [15] ParamArg(slot=5, dtype=dtypes.f32)
37 Ops.INDEX dtypes.f32 [36, 13]
38 Ops.PARAM dtypes.f32 [15] ParamArg(slot=6, dtype=dtypes.f32)
39 Ops.INDEX dtypes.f32 [38, 13]
40 Ops.WHERE dtypes.f32 [24, 37, 39]
41 Ops.PARAM dtypes.f32 [15] ParamArg(slot=7, dtype=dtypes.f32)
42 Ops.INDEX dtypes.f32 [41, 13]
43 Ops.WHERE dtypes.f32 [22, 40, 42]
44 Ops.WHERE dtypes.f32 [20, 35, 43]
45 Ops.STORE dtypes.void [18, 44]
46 Ops.END dtypes.void [45, 3, 6, 9, 12]
47 Ops.SINK dtypes.void [46] KernelInfo(name="E_512_7_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST))), optimized=True)

