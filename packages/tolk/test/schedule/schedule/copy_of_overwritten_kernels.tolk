# tolk D160 golden=ebc3f46fa117e070666078df6d1cf957
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
4 Ops.INDEX dtypes.f32 [2, 3]
5 Ops.CONST dtypes.weakfloat [] 2.0
6 Ops.MUL dtypes.f32 [4, 5]
7 Ops.CONST dtypes.weakfloat [] 7.0
8 Ops.ADD dtypes.f32 [6, 7]
9 Ops.STORE dtypes.void [4, 8]
10 Ops.END dtypes.void [9, 3]
11 Ops.SINK dtypes.void [10] KernelInfo()
12 Ops.CALL dtypes.void [11, 1] CallInfo()
13 Ops.AFTER dtypes.f32 [1, 12]
14 Ops.CONST dtypes.weakint [] 2
15 Ops.PARAM dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32, device="CPU:1")
16 Ops.PARAM dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32)
17 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 0)
18 Ops.INDEX dtypes.f32 [16, 17]
19 Ops.PARAM dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.INDEX dtypes.f32 [19, 17]
21 Ops.PARAM dtypes.f32 [14] ParamArg(slot=2, dtype=dtypes.f32)
22 Ops.INDEX dtypes.f32 [21, 17]
23 Ops.ADD dtypes.f32 [20, 22]
24 Ops.STORE dtypes.void [18, 23]
25 Ops.END dtypes.void [24, 17]
26 Ops.SINK dtypes.void [25] KernelInfo()
27 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=3, dtype=dtypes.f32, device="CPU:1")
28 Ops.STORE dtypes.void [18, 20]
29 Ops.END dtypes.void [28, 17]
30 Ops.SINK dtypes.void [29] KernelInfo()
31 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
32 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
33 Ops.INDEX dtypes.f32 [32, 17]
34 Ops.STORE dtypes.void [18, 33]
35 Ops.END dtypes.void [34, 17]
36 Ops.SINK dtypes.void [35] KernelInfo()
37 Ops.CALL dtypes.void [36, 31, 1] CallInfo()
38 Ops.AFTER dtypes.f32 [31, 37]
39 Ops.CALL dtypes.void [30, 27, 38] CallInfo()
40 Ops.AFTER dtypes.f32 [27, 39]
41 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=5, dtype=dtypes.f32, device="CPU:1")
42 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
43 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
44 Ops.INDEX dtypes.f32 [32, 3]
45 Ops.CONST dtypes.weakfloat [] 3.0
46 Ops.MUL dtypes.f32 [44, 45]
47 Ops.STORE dtypes.void [4, 46]
48 Ops.END dtypes.void [47, 3]
49 Ops.SINK dtypes.void [48] KernelInfo()
50 Ops.CALL dtypes.void [49, 43, 13] CallInfo()
51 Ops.AFTER dtypes.f32 [43, 50]
52 Ops.CALL dtypes.void [36, 42, 51] CallInfo()
53 Ops.AFTER dtypes.f32 [42, 52]
54 Ops.CALL dtypes.void [30, 41, 53] CallInfo()
55 Ops.AFTER dtypes.f32 [41, 54]
56 Ops.CALL dtypes.void [26, 15, 40, 55] CallInfo()
57 Ops.AFTER dtypes.f32 [15, 56]
58 Ops.SINK dtypes.void [13, 57]

