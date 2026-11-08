# tolk D163 golden=bfba7a4fc5207a79f28f0cdb829cf7f1
0 Ops.CONST dtypes.weakint [] 448
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 7
4 Ops.RANGE dtypes.weakint [3] (AxisType.WEAK, 1)
5 Ops.CONST dtypes.weakint [] 64
6 Ops.MUL dtypes.weakint [4, 5]
7 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 2)
8 Ops.ADD dtypes.weakint [6, 7]
9 Ops.INDEX dtypes.f32 [2, 8]
10 Ops.CONST dtypes.weakint [] 224
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.CONST dtypes.weakint [] 32
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
34 Ops.CONST dtypes.weakfloat [] 7.0
35 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
36 Ops.MUL dtypes.weakint [29, 5]
37 Ops.CONST dtypes.weakint [] 2
38 Ops.MUL dtypes.weakint [31, 37]
39 Ops.ADD dtypes.weakint [36, 38]
40 Ops.INDEX dtypes.f32 [35, 39]
41 Ops.CMPLT dtypes.bool [34, 40]
42 Ops.WHERE dtypes.f32 [41, 34, 40]
43 Ops.CONST dtypes.weakfloat [] 1.0
44 Ops.CONST dtypes.weakfloat [] 1.702
45 Ops.MUL dtypes.f32 [40, 44]
46 Ops.CONST dtypes.weakfloat [] -1.4426950408889634
47 Ops.MUL dtypes.f32 [45, 46]
48 Ops.EXP2 dtypes.f32 [47]
49 Ops.ADD dtypes.f32 [43, 48]
50 Ops.RECIP dtypes.f32 [49]
51 Ops.MUL dtypes.f32 [42, 50]
52 Ops.CONST dtypes.weakint [] 1
53 Ops.ADD dtypes.weakint [39, 52]
54 Ops.INDEX dtypes.f32 [35, 53]
55 Ops.CONST dtypes.weakfloat [] -7.0
56 Ops.CMPLT dtypes.bool [54, 55]
57 Ops.WHERE dtypes.f32 [56, 55, 54]
58 Ops.CMPLT dtypes.bool [34, 57]
59 Ops.WHERE dtypes.f32 [58, 34, 57]
60 Ops.ADD dtypes.f32 [59, 43]
61 Ops.MUL dtypes.f32 [51, 60]
62 Ops.STORE dtypes.void [33, 61]
63 Ops.END dtypes.void [62, 29, 31]
64 Ops.SINK dtypes.void [63] KernelInfo()
65 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
66 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 2)
67 Ops.MUL dtypes.weakint [66, 37]
68 Ops.RANGE dtypes.weakint [37] (AxisType.WEAK, 3)
69 Ops.ADD dtypes.weakint [67, 68]
70 Ops.ADD dtypes.weakint [69, 6]
71 Ops.INDEX dtypes.f32 [2, 70]
72 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 0)
73 Ops.ADD dtypes.weakint [6, 72]
74 Ops.INDEX dtypes.f32 [35, 73]
75 Ops.CONST dtypes.weakint [] 4096
76 Ops.PARAM dtypes.f32 [75] ParamArg(slot=2, dtype=dtypes.f32)
77 Ops.MUL dtypes.weakint [72, 5]
78 Ops.ADD dtypes.weakint [69, 77]
79 Ops.INDEX dtypes.f32 [76, 78]
80 Ops.MUL dtypes.f32 [74, 79]
81 Ops.REDUCE dtypes.f32 [80, 72] (Ops.ADD, 0)
82 Ops.STORE dtypes.void [71, 81]
83 Ops.END dtypes.void [82, 4, 66, 68]
84 Ops.SINK dtypes.void [83] KernelInfo()
85 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
86 Ops.PARAM dtypes.f32 [75] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
87 Ops.CALL dtypes.void [84, 65, 85, 86] CallInfo()
88 Ops.AFTER dtypes.f32 [65, 87]
89 Ops.CALL dtypes.void [64, 27, 88] CallInfo()
90 Ops.AFTER dtypes.f32 [27, 89]
91 Ops.PARAM dtypes.f32 [17] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
92 Ops.CALL dtypes.void [26, 1, 90, 91] CallInfo()
93 Ops.AFTER dtypes.f32 [1, 92]
94 Ops.SINK dtypes.void [93]

