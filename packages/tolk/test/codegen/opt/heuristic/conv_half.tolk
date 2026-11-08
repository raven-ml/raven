# tolk D280 golden=721bac8ac1ab79212831c9d8ebae9f4d
0 Ops.CONST dtypes.weakint [] 32768
1 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 4)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 5)
5 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
6 Ops.AFTER dtypes.f16 [1, 3, 4, 5]
7 Ops.MUL dtypes.weakint [3, 2]
8 Ops.ADD dtypes.weakint [7, 4]
9 Ops.CONST dtypes.weakint [] 1024
10 Ops.MUL dtypes.weakint [5, 9]
11 Ops.ADD dtypes.weakint [8, 10]
12 Ops.INDEX dtypes.f16 [6, 11]
13 Ops.CONST dtypes.weakint [] 3
14 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 1)
15 Ops.ADD dtypes.weakint [14, 3]
16 Ops.CONST dtypes.weakint [] 1
17 Ops.CMPLT dtypes.bool [15, 16]
18 Ops.CONST dtypes.bool [] False
19 Ops.CMPEQ dtypes.bool [17, 18]
20 Ops.CONST dtypes.weakint [] 33
21 Ops.CMPLT dtypes.bool [15, 20]
22 Ops.AND dtypes.bool [19, 21]
23 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 2)
24 Ops.ADD dtypes.weakint [23, 4]
25 Ops.CMPLT dtypes.bool [24, 16]
26 Ops.CMPEQ dtypes.bool [25, 18]
27 Ops.CMPLT dtypes.bool [24, 20]
28 Ops.AND dtypes.bool [26, 27]
29 Ops.AND dtypes.bool [22, 28]
30 Ops.CONST dtypes.weakint [] 16384
31 Ops.PARAM dtypes.f16 [30] ParamArg(slot=1, dtype=dtypes.f16)
32 Ops.AND dtypes.bool [28, 22]
33 Ops.MUL dtypes.weakint [14, 2]
34 Ops.ADD dtypes.weakint [33, 7]
35 Ops.ADD dtypes.weakint [34, 24]
36 Ops.CONST dtypes.weakint [] 16
37 Ops.RANGE dtypes.weakint [36] (AxisType.WEAK, 0)
38 Ops.MUL dtypes.weakint [37, 9]
39 Ops.ADD dtypes.weakint [35, 38]
40 Ops.CONST dtypes.weakint [] -33
41 Ops.ADD dtypes.weakint [39, 40]
42 Ops.CONST dtypes.bool [] Invalid
43 Ops.WHERE dtypes.weakint [32, 41, 42]
44 Ops.INDEX dtypes.f16 [31, 43]
45 Ops.CONST dtypes.weakfloat [] 0.0
46 Ops.WHERE dtypes.f16 [29, 44, 45]
47 Ops.CONST dtypes.weakint [] 4608
48 Ops.PARAM dtypes.f16 [47] ParamArg(slot=2, dtype=dtypes.f16)
49 Ops.MUL dtypes.weakint [14, 13]
50 Ops.ADD dtypes.weakint [49, 23]
51 Ops.CONST dtypes.weakint [] 9
52 Ops.MUL dtypes.weakint [37, 51]
53 Ops.ADD dtypes.weakint [50, 52]
54 Ops.CONST dtypes.weakint [] 144
55 Ops.MUL dtypes.weakint [5, 54]
56 Ops.ADD dtypes.weakint [53, 55]
57 Ops.INDEX dtypes.f16 [48, 56]
58 Ops.MUL dtypes.f16 [46, 57]
59 Ops.CAST dtypes.f32 [58] dtypes.f32
60 Ops.REDUCE dtypes.f32 [59, 37, 14, 23] (Ops.ADD, 0)
61 Ops.CAST dtypes.f16 [60] dtypes.f16
62 Ops.STORE dtypes.void [12, 61]
63 Ops.END dtypes.void [62, 5, 3, 4]
64 Ops.SINK dtypes.void [63] KernelInfo()

