# tolk D413 golden=c309b729b256457055864d30ec36a1b9
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
66 Ops.SHR dtypes.i32 [7, 2]
67 Ops.TRUNCDIV dtypes.i32 [7, 14]
68 Ops.TRUNCMOD dtypes.i32 [7, 14]
69 Ops.CMPEQ dtypes.bool [68, 23]
70 Ops.CMPEQ dtypes.bool [69, 43]
71 Ops.CMPLT dtypes.bool [14, 23]
72 Ops.CMPEQ dtypes.bool [45, 71]
73 Ops.CMPEQ dtypes.bool [72, 43]
74 Ops.AND dtypes.bool [70, 73]
75 Ops.CAST dtypes.i32 [74] dtypes.i32
76 Ops.MUL dtypes.i32 [75, 51]
77 Ops.ADD dtypes.i32 [67, 76]
78 Ops.TRUNCDIV dtypes.i32 [7, 16]
79 Ops.TRUNCMOD dtypes.i32 [7, 16]
80 Ops.CMPEQ dtypes.bool [79, 23]
81 Ops.CMPEQ dtypes.bool [80, 43]
82 Ops.CMPLT dtypes.bool [16, 23]
83 Ops.CMPEQ dtypes.bool [45, 82]
84 Ops.CMPEQ dtypes.bool [83, 43]
85 Ops.AND dtypes.bool [81, 84]
86 Ops.CAST dtypes.i32 [85] dtypes.i32
87 Ops.MUL dtypes.i32 [86, 51]
88 Ops.ADD dtypes.i32 [78, 87]
89 Ops.TRUNCDIV dtypes.u64 [18, 20]
90 Ops.CONST dtypes.weakint [] 63
91 Ops.SHR dtypes.u64 [18, 90]
92 Ops.TRUNCDIV dtypes.u64 [18, 23]
93 Ops.TRUNCDIV dtypes.weakint [25, 12]
94 Ops.TRUNCMOD dtypes.weakint [25, 12]
95 Ops.CMPEQ dtypes.bool [94, 23]
96 Ops.CMPEQ dtypes.bool [95, 43]
97 Ops.CMPLT dtypes.bool [25, 23]
98 Ops.CMPLT dtypes.bool [12, 23]
99 Ops.CMPEQ dtypes.bool [97, 98]
100 Ops.CMPEQ dtypes.bool [99, 43]
101 Ops.AND dtypes.bool [96, 100]
102 Ops.CAST dtypes.weakint [101] dtypes.weakint
103 Ops.MUL dtypes.weakint [102, 51]
104 Ops.ADD dtypes.weakint [93, 103]
105 Ops.TRUNCDIV dtypes.i32 [27, 28]
106 Ops.TRUNCDIV dtypes.i32 [30, 31]
107 Ops.TRUNCDIV dtypes.i32 [35, 2]
108 Ops.SINK dtypes.void [38, 39, 53, 64, 65, 66, 77, 88, 89, 91, 92, 104, 105, 106, 107]
109 Ops.SINK dtypes.void [37, 108]

