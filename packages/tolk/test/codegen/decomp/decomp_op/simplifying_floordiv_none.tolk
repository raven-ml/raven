# tolk D413 golden=6ab4f3d18907f8337656655384cf1f46
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(0, 100), name="p", addrspace=AddrSpace.ALU)
2 Ops.CONST dtypes.weakint [] 3
3 Ops.DIV dtypes.i32 [1, 2]
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-100, 0), name="n", addrspace=AddrSpace.ALU)
5 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-7, -1), name="q", addrspace=AddrSpace.ALU)
6 Ops.DIV dtypes.i32 [4, 5]
7 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-10, 10), name="m", addrspace=AddrSpace.ALU)
8 Ops.DIV dtypes.i32 [7, 2]
9 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(1, 7), name="d", addrspace=AddrSpace.ALU)
10 Ops.DIV dtypes.i32 [7, 9]
11 Ops.DIV dtypes.i32 [1, 9]
12 Ops.CONST dtypes.weakint [] 8
13 Ops.DIV dtypes.i32 [7, 12]
14 Ops.CONST dtypes.weakint [] 1
15 Ops.DIV dtypes.i32 [7, 14]
16 Ops.CONST dtypes.weakint [] -4
17 Ops.DIV dtypes.i32 [7, 16]
18 Ops.PARAM dtypes.u64 [0] ParamArg(slot=-1, dtype=dtypes.u64, vmin_vmax=(0, 18446744073709551615), name="w", addrspace=AddrSpace.ALU)
19 Ops.CONST dtypes.weakint [] 9223372036854775808
20 Ops.CAST dtypes.u64 [19] dtypes.u64
21 Ops.DIV dtypes.u64 [18, 20]
22 Ops.DIV dtypes.u64 [18, 19]
23 Ops.CONST dtypes.weakint [] 0
24 Ops.DIV dtypes.u64 [18, 23]
25 Ops.PARAM dtypes.weakint [0] ParamArg(slot=-1, dtype=dtypes.weakint, vmin_vmax=(-10, 10), name="k", addrspace=AddrSpace.ALU)
26 Ops.DIV dtypes.weakint [25, 12]
27 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(0, 3), name="a", addrspace=AddrSpace.ALU)
28 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(0, 3), name="b", addrspace=AddrSpace.ALU)
29 Ops.DIV dtypes.i32 [27, 28]
30 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-3, 0), name="c", addrspace=AddrSpace.ALU)
31 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-3, 0), name="e", addrspace=AddrSpace.ALU)
32 Ops.DIV dtypes.i32 [30, 31]
33 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-5, 5), name="x", addrspace=AddrSpace.ALU)
34 Ops.MAX dtypes.i32 [33, 23]
35 Ops.ADD dtypes.i32 [34, 14]
36 Ops.DIV dtypes.i32 [35, 2]
37 Ops.SINK dtypes.void [3, 6, 8, 10, 11, 13, 15, 17, 21, 22, 24, 26, 29, 32, 36]
38 Ops.TRUNCDIV dtypes.i32 [1, 2]
39 Ops.TRUNCDIV dtypes.i32 [4, 5]
40 Ops.TRUNCDIV dtypes.i32 [7, 2]
41 Ops.TRUNCMOD dtypes.i32 [7, 2]
42 Ops.CMPEQ dtypes.bool [41, 23]
43 Ops.CONST dtypes.bool [] False
44 Ops.CMPEQ dtypes.bool [42, 43]
45 Ops.CMPLT dtypes.bool [7, 23]
46 Ops.CMPLT dtypes.bool [2, 23]
47 Ops.CMPEQ dtypes.bool [45, 46]
48 Ops.CMPEQ dtypes.bool [47, 43]
49 Ops.AND dtypes.bool [44, 48]
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.CONST dtypes.weakint [] -1
52 Ops.MUL dtypes.i32 [50, 51]
53 Ops.ADD dtypes.i32 [40, 52]
54 Ops.TRUNCDIV dtypes.i32 [7, 9]
55 Ops.TRUNCMOD dtypes.i32 [7, 9]
56 Ops.CMPEQ dtypes.bool [55, 23]
57 Ops.CMPEQ dtypes.bool [56, 43]
58 Ops.CMPLT dtypes.bool [9, 23]
59 Ops.CMPEQ dtypes.bool [45, 58]
60 Ops.CMPEQ dtypes.bool [59, 43]
61 Ops.AND dtypes.bool [57, 60]
62 Ops.CAST dtypes.i32 [61] dtypes.i32
63 Ops.MUL dtypes.i32 [62, 51]
64 Ops.ADD dtypes.i32 [54, 63]
65 Ops.TRUNCDIV dtypes.i32 [1, 9]
66 Ops.TRUNCDIV dtypes.i32 [7, 12]
67 Ops.TRUNCMOD dtypes.i32 [7, 12]
68 Ops.CMPEQ dtypes.bool [67, 23]
69 Ops.CMPEQ dtypes.bool [68, 43]
70 Ops.CMPLT dtypes.bool [12, 23]
71 Ops.CMPEQ dtypes.bool [45, 70]
72 Ops.CMPEQ dtypes.bool [71, 43]
73 Ops.AND dtypes.bool [69, 72]
74 Ops.CAST dtypes.i32 [73] dtypes.i32
75 Ops.MUL dtypes.i32 [74, 51]
76 Ops.ADD dtypes.i32 [66, 75]
77 Ops.TRUNCDIV dtypes.i32 [7, 14]
78 Ops.TRUNCMOD dtypes.i32 [7, 14]
79 Ops.CMPEQ dtypes.bool [78, 23]
80 Ops.CMPEQ dtypes.bool [79, 43]
81 Ops.CMPLT dtypes.bool [14, 23]
82 Ops.CMPEQ dtypes.bool [45, 81]
83 Ops.CMPEQ dtypes.bool [82, 43]
84 Ops.AND dtypes.bool [80, 83]
85 Ops.CAST dtypes.i32 [84] dtypes.i32
86 Ops.MUL dtypes.i32 [85, 51]
87 Ops.ADD dtypes.i32 [77, 86]
88 Ops.TRUNCDIV dtypes.i32 [7, 16]
89 Ops.TRUNCMOD dtypes.i32 [7, 16]
90 Ops.CMPEQ dtypes.bool [89, 23]
91 Ops.CMPEQ dtypes.bool [90, 43]
92 Ops.CMPLT dtypes.bool [16, 23]
93 Ops.CMPEQ dtypes.bool [45, 92]
94 Ops.CMPEQ dtypes.bool [93, 43]
95 Ops.AND dtypes.bool [91, 94]
96 Ops.CAST dtypes.i32 [95] dtypes.i32
97 Ops.MUL dtypes.i32 [96, 51]
98 Ops.ADD dtypes.i32 [88, 97]
99 Ops.TRUNCDIV dtypes.u64 [18, 20]
100 Ops.TRUNCDIV dtypes.u64 [18, 19]
101 Ops.TRUNCDIV dtypes.u64 [18, 23]
102 Ops.TRUNCDIV dtypes.weakint [25, 12]
103 Ops.TRUNCMOD dtypes.weakint [25, 12]
104 Ops.CMPEQ dtypes.bool [103, 23]
105 Ops.CMPEQ dtypes.bool [104, 43]
106 Ops.CMPLT dtypes.bool [25, 23]
107 Ops.CMPEQ dtypes.bool [106, 70]
108 Ops.CMPEQ dtypes.bool [107, 43]
109 Ops.AND dtypes.bool [105, 108]
110 Ops.CAST dtypes.weakint [109] dtypes.weakint
111 Ops.MUL dtypes.weakint [110, 51]
112 Ops.ADD dtypes.weakint [102, 111]
113 Ops.TRUNCDIV dtypes.i32 [27, 28]
114 Ops.TRUNCDIV dtypes.i32 [30, 31]
115 Ops.TRUNCDIV dtypes.i32 [35, 2]
116 Ops.SINK dtypes.void [38, 39, 53, 64, 65, 76, 87, 98, 99, 100, 101, 112, 113, 114, 115]
117 Ops.SINK dtypes.void [37, 116]

