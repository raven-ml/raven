# tolk D56 D126 golden=a853586cffb5458cba489922663adaf9
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CMPEQ dtypes.bool [1, 1]
3 Ops.CONST dtypes.bool [] False
4 Ops.CMPEQ dtypes.bool [2, 3]
5 Ops.CONST dtypes.weakfloat [] nan
6 Ops.CAST dtypes.f32 [5] dtypes.f32
7 Ops.CONST dtypes.weakfloat [] -0.0
8 Ops.CMPLT dtypes.bool [1, 7]
9 Ops.RECIP dtypes.f32 [1]
10 Ops.CONST dtypes.weakfloat [] -inf
11 Ops.CMPEQ dtypes.bool [9, 10]
12 Ops.CMPEQ dtypes.bool [11, 3]
13 Ops.CONST dtypes.weakfloat [] 0.0
14 Ops.CMPEQ dtypes.bool [1, 13]
15 Ops.CMPEQ dtypes.bool [14, 3]
16 Ops.CONST dtypes.weakfloat [] inf
17 Ops.CMPEQ dtypes.bool [1, 16]
18 Ops.CMPEQ dtypes.bool [17, 3]
19 Ops.CONST dtypes.weakfloat [] 1.1754943508222875e-38
20 Ops.CMPLT dtypes.bool [1, 19]
21 Ops.CONST dtypes.weakfloat [] 1.8446744073709552e+19
22 Ops.MUL dtypes.f32 [1, 21]
23 Ops.WHERE dtypes.f32 [20, 22, 1]
24 Ops.BITCAST dtypes.i32 [23] dtypes.i32
25 Ops.CONST dtypes.weakfloat [] 1.4142135623730951
26 Ops.MUL dtypes.f32 [23, 25]
27 Ops.BITCAST dtypes.i32 [26] dtypes.i32
28 Ops.CONST dtypes.weakint [] 8388608
29 Ops.DIV dtypes.i32 [27, 28]
30 Ops.CONST dtypes.weakint [] 255
31 Ops.AND dtypes.i32 [29, 30]
32 Ops.CONST dtypes.weakint [] 127
33 Ops.CONST dtypes.weakint [] -1
34 Ops.MUL dtypes.weakint [32, 33]
35 Ops.ADD dtypes.i32 [31, 34]
36 Ops.CAST dtypes.f32 [35] dtypes.f32
37 Ops.CONST dtypes.weakfloat [] -1.0
38 Ops.MUL dtypes.f32 [36, 37]
39 Ops.CAST dtypes.i32 [38] dtypes.i32
40 Ops.MUL dtypes.i32 [39, 28]
41 Ops.ADD dtypes.i32 [24, 40]
42 Ops.BITCAST dtypes.f32 [41] dtypes.f32
43 Ops.CONST dtypes.weakfloat [] 1.0
44 Ops.MUL dtypes.weakfloat [43, 37]
45 Ops.ADD dtypes.f32 [42, 44]
46 Ops.CONST dtypes.weakfloat [] 0.5
47 Ops.MUL dtypes.f32 [46, 45]
48 Ops.MUL dtypes.f32 [47, 45]
49 Ops.MUL dtypes.f32 [48, 37]
50 Ops.ADD dtypes.f32 [45, 49]
51 Ops.BITCAST dtypes.u32 [50] dtypes.u32
52 Ops.CONST dtypes.weakint [] 4294963200
53 Ops.CAST dtypes.u32 [52] dtypes.u32
54 Ops.AND dtypes.u32 [51, 53]
55 Ops.BITCAST dtypes.f32 [54] dtypes.f32
56 Ops.MUL dtypes.f32 [55, 37]
57 Ops.ADD dtypes.f32 [45, 56]
58 Ops.ADD dtypes.f32 [57, 49]
59 Ops.CONST dtypes.weakfloat [] 2.0
60 Ops.ADD dtypes.f32 [59, 45]
61 Ops.RECIP dtypes.f32 [60]
62 Ops.MUL dtypes.f32 [45, 61]
63 Ops.MUL dtypes.f32 [62, 62]
64 Ops.CONST dtypes.weakfloat [] 0.24279078841209412
65 Ops.MUL dtypes.f32 [64, 63]
66 Ops.CONST dtypes.weakfloat [] 0.2849878668785095
67 Ops.ADD dtypes.f32 [65, 66]
68 Ops.MUL dtypes.f32 [67, 63]
69 Ops.CONST dtypes.weakfloat [] 0.40000972151756287
70 Ops.ADD dtypes.f32 [68, 69]
71 Ops.MUL dtypes.f32 [70, 63]
72 Ops.CONST dtypes.weakfloat [] 0.6666666269302368
73 Ops.ADD dtypes.f32 [71, 72]
74 Ops.MUL dtypes.f32 [63, 73]
75 Ops.ADD dtypes.f32 [48, 74]
76 Ops.MUL dtypes.f32 [62, 75]
77 Ops.ADD dtypes.f32 [58, 76]
78 Ops.ADD dtypes.f32 [77, 55]
79 Ops.CONST dtypes.weakfloat [] -0.00017605285393
80 Ops.MUL dtypes.f32 [78, 79]
81 Ops.CONST dtypes.weakfloat [] 1.4428710938
82 Ops.MUL dtypes.f32 [77, 81]
83 Ops.ADD dtypes.f32 [80, 82]
84 Ops.CONST dtypes.weakfloat [] 64.0
85 Ops.MUL dtypes.weakfloat [84, 37]
86 Ops.ADD dtypes.f32 [36, 85]
87 Ops.WHERE dtypes.f32 [20, 86, 36]
88 Ops.MUL dtypes.f32 [55, 81]
89 Ops.ADD dtypes.f32 [87, 88]
90 Ops.MUL dtypes.f32 [89, 37]
91 Ops.ADD dtypes.f32 [87, 90]
92 Ops.ADD dtypes.f32 [91, 88]
93 Ops.ADD dtypes.f32 [83, 92]
94 Ops.ADD dtypes.f32 [93, 89]
95 Ops.CAST dtypes.f32 [16] dtypes.f32
96 Ops.WHERE dtypes.f32 [18, 94, 95]
97 Ops.CAST dtypes.f32 [10] dtypes.f32
98 Ops.WHERE dtypes.f32 [15, 96, 97]
99 Ops.WHERE dtypes.f32 [12, 98, 97]
100 Ops.WHERE dtypes.f32 [8, 6, 99]
101 Ops.WHERE dtypes.f32 [4, 6, 100]
102 Ops.SINK dtypes.void [101]

