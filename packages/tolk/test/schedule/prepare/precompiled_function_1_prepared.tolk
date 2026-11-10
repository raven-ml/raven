# tolk D473 golden=7e0a9cecbebf33dbdcbbfffeff1a2e88
0 Ops.CONST dtypes.weakint [] 8
1 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=-1, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
4 Ops.INDEX dtypes.f32 [2, 3]
5 Ops.CONST dtypes.weakint [] 32
6 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32)
7 Ops.CONST dtypes.weakint [] 4
8 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 0)
9 Ops.MUL dtypes.weakint [8, 0]
10 Ops.ADD dtypes.weakint [9, 3]
11 Ops.INDEX dtypes.f32 [6, 10]
12 Ops.CONST dtypes.weakfloat [] 2.0
13 Ops.MUL dtypes.f32 [11, 12]
14 Ops.REDUCE dtypes.f32 [13, 8] (Ops.ADD, 0)
15 Ops.STORE dtypes.void [4, 14]
16 Ops.END dtypes.void [15, 3]
17 Ops.SINK dtypes.void [16] KernelInfo()
18 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
19 Ops.PARAM dtypes.f32 [5] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
20 Ops.CALL dtypes.void [17, 18, 19] CallInfo()
21 Ops.LINEAR dtypes.void [20]
22 Ops.ALLOC dtypes.f32 [5] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
23 Ops.STACK dtypes.weakint [7, 0]
24 Ops.RESHAPE dtypes.f32 [22, 23]
25 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
26 Ops.RESHAPE dtypes.f32 [25, 23]
27 Ops.CONST dtypes.weakfloat [] 1.0
28 Ops.ADD dtypes.f32 [26, 27]
29 Ops.STORE dtypes.void [24, 28]
30 Ops.AFTER dtypes.f32 [22, 29]
31 Ops.RESHAPE dtypes.f32 [30, 23]
32 Ops.CALL dtypes.void [21, 31, 1] CallInfo(name="precompiled_function.<locals>.f", precompile=True)
33 Ops.AFTER dtypes.f32 [1, 32]
34 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
35 Ops.ADD dtypes.f32 [33, 27]
36 Ops.STORE dtypes.void [34, 35]
37 Ops.AFTER dtypes.f32 [34, 36]
38 Ops.SINK dtypes.void [33, 37]

