# tolk D473 golden=e1bce18ecd7ec33673378f2dfdbcbfb5
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
5 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
6 Ops.INDEX dtypes.f32 [4, 5]
7 Ops.REDUCE dtypes.f32 [6, 5] (Ops.ADD, 0)
8 Ops.CONST dtypes.weakfloat [] 0.5
9 Ops.MUL dtypes.f32 [7, 8]
10 Ops.STORE dtypes.void [3, 9]
11 Ops.END dtypes.void [10, 2]
12 Ops.SINK dtypes.void [11] KernelInfo()
13 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
14 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
15 Ops.CALL dtypes.void [12, 13, 14] CallInfo()
16 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
17 Ops.CALL dtypes.void [12, 16, 13] CallInfo()
18 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
19 Ops.CALL dtypes.void [12, 18, 16] CallInfo()
20 Ops.INDEX dtypes.f32 [1, 5]
21 Ops.ADD dtypes.f32 [20, 6]
22 Ops.STORE dtypes.void [20, 21]
23 Ops.END dtypes.void [22, 5]
24 Ops.SINK dtypes.void [23] KernelInfo()
25 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
26 Ops.CALL dtypes.void [24, 25, 18] CallInfo()
27 Ops.CONST dtypes.weakfloat [] 2.0
28 Ops.MUL dtypes.f32 [20, 27]
29 Ops.STORE dtypes.void [20, 28]
30 Ops.END dtypes.void [29, 5]
31 Ops.SINK dtypes.void [30] KernelInfo()
32 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
33 Ops.CALL dtypes.void [31, 32] CallInfo()
34 Ops.MUL dtypes.f32 [6, 27]
35 Ops.STORE dtypes.void [20, 34]
36 Ops.END dtypes.void [35, 5]
37 Ops.SINK dtypes.void [36] KernelInfo()
38 Ops.PARAM dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
39 Ops.CALL dtypes.void [37, 38, 25] CallInfo()
40 Ops.LINEAR dtypes.void [15, 17, 19, 26, 33, 39]

