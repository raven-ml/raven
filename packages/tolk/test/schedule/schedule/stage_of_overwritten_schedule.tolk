# tolk D160 golden=5e1469943f80a4499d8326bcdd0c57ad
0 Ops.CONST dtypes.weakint [] 2
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 4
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.INDEX dtypes.f32 [5, 2]
7 Ops.STORE dtypes.void [3, 6]
8 Ops.END dtypes.void [7, 2]
9 Ops.SINK dtypes.void [8] KernelInfo()
10 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
11 Ops.PARAM dtypes.f32 [4] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
12 Ops.CALL dtypes.void [9, 10, 11] CallInfo()
13 Ops.PARAM dtypes.f32 [4] ParamArg(slot=0, dtype=dtypes.f32)
14 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
15 Ops.INDEX dtypes.f32 [13, 14]
16 Ops.CONST dtypes.weakfloat [] 2.0
17 Ops.MUL dtypes.f32 [15, 16]
18 Ops.CONST dtypes.weakfloat [] 7.0
19 Ops.ADD dtypes.f32 [17, 18]
20 Ops.STORE dtypes.void [15, 19]
21 Ops.END dtypes.void [20, 14]
22 Ops.SINK dtypes.void [21] KernelInfo()
23 Ops.CALL dtypes.void [22, 11] CallInfo()
24 Ops.INDEX dtypes.f32 [5, 14]
25 Ops.CONST dtypes.weakfloat [] 3.0
26 Ops.MUL dtypes.f32 [24, 25]
27 Ops.STORE dtypes.void [15, 26]
28 Ops.END dtypes.void [27, 14]
29 Ops.SINK dtypes.void [28] KernelInfo()
30 Ops.ALLOC dtypes.f32 [4] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
31 Ops.CALL dtypes.void [29, 30, 11] CallInfo()
32 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
33 Ops.INDEX dtypes.f32 [32, 2]
34 Ops.PARAM dtypes.f32 [4] ParamArg(slot=2, dtype=dtypes.f32)
35 Ops.INDEX dtypes.f32 [34, 2]
36 Ops.ADD dtypes.f32 [33, 35]
37 Ops.STORE dtypes.void [3, 36]
38 Ops.END dtypes.void [37, 2]
39 Ops.SINK dtypes.void [38] KernelInfo()
40 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
41 Ops.CALL dtypes.void [39, 40, 10, 30] CallInfo()
42 Ops.LINEAR dtypes.void [12, 23, 31, 41]

