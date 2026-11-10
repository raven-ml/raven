# tolk D183 golden=051a008bc557ad762e53f9bb55804529
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakfloat [] 2.0
5 Ops.MUL dtypes.f32 [3, 4]
6 Ops.CONST dtypes.weakfloat [] 7.0
7 Ops.ADD dtypes.f32 [5, 6]
8 Ops.STORE dtypes.void [3, 7]
9 Ops.END dtypes.void [8, 2]
10 Ops.SINK dtypes.void [9] KernelInfo()
11 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
12 Ops.CALL dtypes.void [10, 11] CallInfo()
13 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.INDEX dtypes.f32 [13, 2]
15 Ops.CONST dtypes.weakfloat [] 3.0
16 Ops.MUL dtypes.f32 [14, 15]
17 Ops.STORE dtypes.void [3, 16]
18 Ops.END dtypes.void [17, 2]
19 Ops.SINK dtypes.void [18] KernelInfo()
20 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
21 Ops.CALL dtypes.void [19, 20, 11] CallInfo()
22 Ops.CONST dtypes.weakint [] 2
23 Ops.PARAM dtypes.f32 [22] ParamArg(slot=0, dtype=dtypes.f32)
24 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0)
25 Ops.INDEX dtypes.f32 [23, 24]
26 Ops.INDEX dtypes.f32 [13, 24]
27 Ops.STORE dtypes.void [25, 26]
28 Ops.END dtypes.void [27, 24]
29 Ops.SINK dtypes.void [28] KernelInfo()
30 Ops.ALLOC dtypes.f32 [22] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
31 Ops.CALL dtypes.void [29, 30, 20] CallInfo()
32 Ops.PARAM dtypes.f32 [22] ParamArg(slot=1, dtype=dtypes.f32)
33 Ops.INDEX dtypes.f32 [32, 24]
34 Ops.STORE dtypes.void [25, 33]
35 Ops.END dtypes.void [34, 24]
36 Ops.SINK dtypes.void [35] KernelInfo()
37 Ops.ALLOC dtypes.f32 [22] ParamArg(slot=3, dtype=dtypes.f32, device="CPU:1")
38 Ops.PARAM dtypes.f32 [22] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
39 Ops.CALL dtypes.void [36, 37, 38] CallInfo()
40 Ops.ALLOC dtypes.f32 [22] ParamArg(slot=4, dtype=dtypes.f32, device="CPU:1")
41 Ops.CALL dtypes.void [36, 40, 30] CallInfo()
42 Ops.PARAM dtypes.f32 [22] ParamArg(slot=2, dtype=dtypes.f32)
43 Ops.INDEX dtypes.f32 [42, 24]
44 Ops.ADD dtypes.f32 [33, 43]
45 Ops.STORE dtypes.void [25, 44]
46 Ops.END dtypes.void [45, 24]
47 Ops.SINK dtypes.void [46] KernelInfo()
48 Ops.PARAM dtypes.f32 [22] ParamArg(slot=1, dtype=dtypes.f32, device="CPU:1")
49 Ops.CALL dtypes.void [47, 48, 37, 40] CallInfo()
50 Ops.LINEAR dtypes.void [12, 21, 31, 39, 41, 49]

