# tolk D202 D183 golden=a71bbcdf33f25a248dab4190240ec200
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 4
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
7 Ops.INDEX dtypes.f32 [5, 6]
8 Ops.REDUCE dtypes.f32 [7, 6] (Ops.ADD, 0)
9 Ops.STORE dtypes.void [3, 8]
10 Ops.SINK dtypes.void [9] KernelInfo()
11 Ops.CONST dtypes.weakint [] 512
12 Ops.BUFFER dtypes.i8 [11] ParamArg(slot=13, dtype=dtypes.i8, device="CPU")
13 Ops.SHRINK dtypes.i8 [12, 2, 4]
14 Ops.BITCAST dtypes.f32 [13] dtypes.f32
15 Ops.BUFFER dtypes.f32 [4] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
16 Ops.CALL dtypes.void [10, 14, 15] CallInfo()
17 Ops.PARAM dtypes.f32 [4] ParamArg(slot=0, dtype=dtypes.f32)
18 Ops.INDEX dtypes.f32 [17, 6]
19 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.INDEX dtypes.f32 [19, 2]
21 Ops.CONST dtypes.weakfloat [] 0.5
22 Ops.MUL dtypes.f32 [20, 21]
23 Ops.STORE dtypes.void [18, 22]
24 Ops.END dtypes.void [23, 6]
25 Ops.SINK dtypes.void [24] KernelInfo()
26 Ops.CONST dtypes.weakint [] 256
27 Ops.CONST dtypes.weakint [] 16
28 Ops.SHRINK dtypes.i8 [12, 26, 27]
29 Ops.BITCAST dtypes.f32 [28] dtypes.f32
30 Ops.CALL dtypes.void [25, 29, 14] CallInfo()
31 Ops.CALL dtypes.void [10, 14, 29] CallInfo()
32 Ops.ADD dtypes.f32 [18, 7]
33 Ops.STORE dtypes.void [18, 32]
34 Ops.END dtypes.void [33, 6]
35 Ops.SINK dtypes.void [34] KernelInfo()
36 Ops.BUFFER dtypes.f32 [4] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
37 Ops.CALL dtypes.void [35, 36, 29] CallInfo()
38 Ops.CONST dtypes.weakfloat [] 2.0
39 Ops.MUL dtypes.f32 [7, 38]
40 Ops.STORE dtypes.void [18, 39]
41 Ops.END dtypes.void [40, 6]
42 Ops.SINK dtypes.void [41] KernelInfo()
43 Ops.BUFFER dtypes.f32 [4] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
44 Ops.CALL dtypes.void [42, 43, 36] CallInfo()
45 Ops.MUL dtypes.f32 [18, 38]
46 Ops.STORE dtypes.void [18, 45]
47 Ops.END dtypes.void [46, 6]
48 Ops.SINK dtypes.void [47] KernelInfo()
49 Ops.BUFFER dtypes.f32 [4] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
50 Ops.CALL dtypes.void [48, 49] CallInfo()
51 Ops.LINEAR dtypes.void [16, 30, 31, 30, 31, 30, 37, 44, 50]

