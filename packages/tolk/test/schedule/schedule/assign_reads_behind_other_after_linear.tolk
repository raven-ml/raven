# tolk D160 golden=2992df7efc01e3b132d7d3b3571fd29a
0 Ops.CONST dtypes.weakint [] 2
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
5 Ops.INDEX dtypes.f32 [4, 2]
6 Ops.CONST dtypes.weakfloat [] 1.0
7 Ops.ADD dtypes.f32 [5, 6]
8 Ops.STORE dtypes.void [3, 7]
9 Ops.END dtypes.void [8, 2]
10 Ops.SINK dtypes.void [9] KernelInfo()
11 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
12 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
13 Ops.CALL dtypes.void [10, 11, 12] CallInfo()
14 Ops.CONST dtypes.weakint [] 1
15 Ops.PARAM dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32)
16 Ops.CONST dtypes.weakint [] 0
17 Ops.INDEX dtypes.f32 [15, 16]
18 Ops.REDUCE dtypes.f32 [5, 2] (Ops.ADD, 0)
19 Ops.STORE dtypes.void [17, 18]
20 Ops.SINK dtypes.void [19] KernelInfo()
21 Ops.CONST dtypes.weakint [] 256
22 Ops.BUFFER dtypes.i8 [21] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
23 Ops.CONST dtypes.weakint [] 4
24 Ops.SHRINK dtypes.i8 [22, 16, 23]
25 Ops.BITCAST dtypes.f32 [24] dtypes.f32
26 Ops.CALL dtypes.void [20, 25, 11] CallInfo()
27 Ops.PARAM dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32)
28 Ops.INDEX dtypes.f32 [27, 16]
29 Ops.STORE dtypes.void [3, 28]
30 Ops.END dtypes.void [29, 2]
31 Ops.SINK dtypes.void [30] KernelInfo()
32 Ops.CALL dtypes.void [31, 12, 25] CallInfo()
33 Ops.LINEAR dtypes.void [13, 26, 32]

