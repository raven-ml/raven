# tolk D473 golden=07256247332c2418bb166b4b00999d96
0 Ops.CONST dtypes.weakint [] 32
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
10 Ops.INDEX dtypes.f32 [9, 7]
11 Ops.CONST dtypes.weakfloat [] 2.0
12 Ops.MUL dtypes.f32 [10, 11]
13 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
14 Ops.INDEX dtypes.f32 [13, 7]
15 Ops.ADD dtypes.f32 [12, 14]
16 Ops.STORE dtypes.void [8, 15]
17 Ops.END dtypes.void [16, 3, 6]
18 Ops.SINK dtypes.void [17] KernelInfo()
19 Ops.CONST dtypes.weakint [] 256
20 Ops.BUFFER dtypes.i8 [19] ParamArg(slot=156, dtype=dtypes.i8, device="CPU")
21 Ops.CONST dtypes.weakint [] 0
22 Ops.CONST dtypes.weakint [] 128
23 Ops.SHRINK dtypes.i8 [20, 21, 22]
24 Ops.BITCAST dtypes.f32 [23] dtypes.f32
25 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
26 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
27 Ops.CALL dtypes.void [18, 24, 25, 26] CallInfo()
28 Ops.PARAM dtypes.f32 [2] ParamArg(slot=0, dtype=dtypes.f32)
29 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
30 Ops.INDEX dtypes.f32 [28, 29]
31 Ops.MUL dtypes.weakint [29, 4]
32 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.INDEX dtypes.f32 [9, 33]
35 Ops.REDUCE dtypes.f32 [34, 32] (Ops.ADD, 0)
36 Ops.STORE dtypes.void [30, 35]
37 Ops.END dtypes.void [36, 29]
38 Ops.SINK dtypes.void [37] KernelInfo()
39 Ops.BUFFER dtypes.f32 [2] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
40 Ops.CALL dtypes.void [38, 39, 24] CallInfo()
41 Ops.LINEAR dtypes.void [27, 40]

