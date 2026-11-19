# tolk D106 golden=531b552122c44e41a84b8a444d8d99cd
0 Ops.CONST dtypes.weakint [] 256
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 8
4 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 1)
5 Ops.CONST dtypes.weakint [] 32
6 Ops.MUL dtypes.weakint [4, 5]
7 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 2)
8 Ops.ADD dtypes.weakint [6, 7]
9 Ops.INDEX dtypes.f32 [2, 8]
10 Ops.CONST dtypes.weakint [] 512
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 64
13 Ops.MUL dtypes.weakint [4, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 0)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.INDEX dtypes.f32 [11, 15]
17 Ops.CONST dtypes.weakint [] 2048
18 Ops.PARAM dtypes.f32 [17] ParamArg(slot=2, dtype=dtypes.f32)
19 Ops.MUL dtypes.weakint [14, 5]
20 Ops.ADD dtypes.weakint [19, 7]
21 Ops.INDEX dtypes.f32 [18, 20]
22 Ops.MUL dtypes.f32 [16, 21]
23 Ops.REDUCE dtypes.f32 [22, 14] (Ops.ADD, 0)
24 Ops.STORE dtypes.void [9, 23]
25 Ops.END dtypes.void [24, 4, 7]
26 Ops.SINK dtypes.void [25] KernelInfo()
27 Ops.ALLOC dtypes.f32 [10] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
28 Ops.PARAM dtypes.f32 [10] ParamArg(slot=0, dtype=dtypes.f32)
29 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 0)
30 Ops.MUL dtypes.weakint [29, 12]
31 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 1)
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.INDEX dtypes.f32 [28, 32]
34 Ops.INDEX dtypes.f32 [11, 32]
35 Ops.PARAM dtypes.f32 [3] ParamArg(slot=2, dtype=dtypes.f32)
36 Ops.INDEX dtypes.f32 [35, 29]
37 Ops.RECIP dtypes.f32 [36]
38 Ops.MUL dtypes.f32 [34, 37]
39 Ops.PARAM dtypes.f32 [12] ParamArg(slot=3, dtype=dtypes.f32)
40 Ops.INDEX dtypes.f32 [39, 31]
41 Ops.MUL dtypes.f32 [38, 40]
42 Ops.STORE dtypes.void [33, 41]
43 Ops.END dtypes.void [42, 29, 31]
44 Ops.SINK dtypes.void [43] KernelInfo()
45 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
46 Ops.PARAM dtypes.f32 [3] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
47 Ops.PARAM dtypes.f32 [12] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
48 Ops.CALL dtypes.void [44, 27, 45, 46, 47] CallInfo()
49 Ops.AFTER dtypes.f32 [27, 48]
50 Ops.PARAM dtypes.f32 [17] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
51 Ops.CALL dtypes.void [26, 1, 49, 50] CallInfo()
52 Ops.AFTER dtypes.f32 [1, 51]
53 Ops.SINK dtypes.void [52]

