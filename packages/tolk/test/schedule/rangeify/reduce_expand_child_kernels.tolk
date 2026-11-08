# tolk D163 golden=46de2b03aa91311cccae59a9f35db973
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 0
4 Ops.INDEX dtypes.f32 [2, 3]
5 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] 2.0
8 Ops.ADD dtypes.f32 [6, 7]
9 Ops.STORE dtypes.void [4, 8]
10 Ops.SINK dtypes.void [9] KernelInfo()
11 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
12 Ops.CONST dtypes.weakint [] 32
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
15 Ops.INDEX dtypes.f32 [13, 14]
16 Ops.REDUCE dtypes.f32 [15, 14] (Ops.ADD, 0)
17 Ops.STORE dtypes.void [4, 16]
18 Ops.SINK dtypes.void [17] KernelInfo()
19 Ops.ALLOC dtypes.f32 [12] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
20 Ops.PARAM dtypes.f32 [12] ParamArg(slot=0, dtype=dtypes.f32)
21 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 2)
22 Ops.INDEX dtypes.f32 [20, 21]
23 Ops.CONST dtypes.weakint [] 32768
24 Ops.PARAM dtypes.f32 [23] ParamArg(slot=1, dtype=dtypes.f32)
25 Ops.MUL dtypes.weakint [14, 12]
26 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 1)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.CONST dtypes.weakint [] 1024
29 Ops.MUL dtypes.weakint [21, 28]
30 Ops.ADD dtypes.weakint [27, 29]
31 Ops.INDEX dtypes.f32 [24, 30]
32 Ops.REDUCE dtypes.f32 [31, 14, 26] (Ops.ADD, 0)
33 Ops.STORE dtypes.void [22, 32]
34 Ops.END dtypes.void [33, 21]
35 Ops.SINK dtypes.void [34] KernelInfo()
36 Ops.PARAM dtypes.f32 [23] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
37 Ops.CALL dtypes.void [35, 19, 36] CallInfo()
38 Ops.AFTER dtypes.f32 [19, 37]
39 Ops.CALL dtypes.void [18, 11, 38] CallInfo()
40 Ops.AFTER dtypes.f32 [11, 39]
41 Ops.CALL dtypes.void [10, 1, 40] CallInfo()
42 Ops.AFTER dtypes.f32 [1, 41]
43 Ops.CONST dtypes.weakint [] 16
44 Ops.PARAM dtypes.f32 [43] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
45 Ops.PARAM dtypes.f32 [43] ParamArg(slot=0, dtype=dtypes.f32)
46 Ops.RANGE dtypes.weakint [43] (AxisType.WEAK, 0)
47 Ops.INDEX dtypes.f32 [45, 46]
48 Ops.PARAM dtypes.f32 [43] ParamArg(slot=2, dtype=dtypes.f32)
49 Ops.INDEX dtypes.f32 [48, 46]
50 Ops.ADD dtypes.f32 [6, 49]
51 Ops.STORE dtypes.void [47, 50]
52 Ops.END dtypes.void [51, 46]
53 Ops.SINK dtypes.void [52] KernelInfo()
54 Ops.PARAM dtypes.f32 [43] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
55 Ops.CALL dtypes.void [53, 44, 40, 54] CallInfo()
56 Ops.AFTER dtypes.f32 [44, 55]
57 Ops.SINK dtypes.void [42, 56]

