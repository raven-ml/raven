# tolk D413 golden=a5960a61ee2c9fdf56e98d5ca930ab27
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(0, 100), name="p", addrspace=AddrSpace.ALU)
2 Ops.CONST dtypes.weakint [] 3
3 Ops.MOD dtypes.i32 [1, 2]
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-100, 0), name="n", addrspace=AddrSpace.ALU)
5 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-7, -1), name="q", addrspace=AddrSpace.ALU)
6 Ops.MOD dtypes.i32 [4, 5]
7 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-10, 10), name="m", addrspace=AddrSpace.ALU)
8 Ops.MOD dtypes.i32 [7, 2]
9 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(1, 7), name="d", addrspace=AddrSpace.ALU)
10 Ops.MOD dtypes.i32 [7, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.MOD dtypes.i32 [7, 11]
13 Ops.CONST dtypes.weakint [] 1
14 Ops.MOD dtypes.i32 [7, 13]
15 Ops.CONST dtypes.weakint [] -4
16 Ops.MOD dtypes.i32 [7, 15]
17 Ops.PARAM dtypes.u64 [0] ParamArg(slot=-1, dtype=dtypes.u64, vmin_vmax=(0, 18446744073709551615), name="w", addrspace=AddrSpace.ALU)
18 Ops.CONST dtypes.weakint [] 9223372036854775808
19 Ops.CAST dtypes.u64 [18] dtypes.u64
20 Ops.MOD dtypes.u64 [17, 19]
21 Ops.MOD dtypes.u64 [17, 18]
22 Ops.CONST dtypes.weakint [] 0
23 Ops.MOD dtypes.u64 [17, 22]
24 Ops.PARAM dtypes.weakint [0] ParamArg(slot=-1, dtype=dtypes.weakint, vmin_vmax=(-10, 10), name="k", addrspace=AddrSpace.ALU)
25 Ops.CONST dtypes.weakint [] 8
26 Ops.MOD dtypes.weakint [24, 25]
27 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(0, 3), name="a", addrspace=AddrSpace.ALU)
28 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(0, 3), name="b", addrspace=AddrSpace.ALU)
29 Ops.MOD dtypes.i32 [27, 28]
30 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-3, 0), name="c", addrspace=AddrSpace.ALU)
31 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-3, 0), name="e", addrspace=AddrSpace.ALU)
32 Ops.MOD dtypes.i32 [30, 31]
33 Ops.SINK dtypes.void [3, 6, 8, 10, 12, 14, 16, 20, 21, 23, 26, 29, 32]
34 Ops.TRUNCMOD dtypes.i32 [1, 2]
35 Ops.TRUNCMOD dtypes.i32 [4, 5]
36 Ops.TRUNCMOD dtypes.i32 [7, 2]
37 Ops.CMPEQ dtypes.bool [36, 22]
38 Ops.CONST dtypes.bool [] False
39 Ops.CMPEQ dtypes.bool [37, 38]
40 Ops.CMPLT dtypes.bool [7, 22]
41 Ops.CMPLT dtypes.bool [2, 22]
42 Ops.CMPEQ dtypes.bool [40, 41]
43 Ops.CMPEQ dtypes.bool [42, 38]
44 Ops.AND dtypes.bool [39, 43]
45 Ops.WHERE dtypes.weakint [44, 2, 22]
46 Ops.CAST dtypes.i32 [45] dtypes.i32
47 Ops.ADD dtypes.i32 [36, 46]
48 Ops.TRUNCMOD dtypes.i32 [7, 9]
49 Ops.CMPEQ dtypes.bool [48, 22]
50 Ops.CMPEQ dtypes.bool [49, 38]
51 Ops.CMPLT dtypes.bool [9, 22]
52 Ops.CMPEQ dtypes.bool [40, 51]
53 Ops.CMPEQ dtypes.bool [52, 38]
54 Ops.AND dtypes.bool [50, 53]
55 Ops.CAST dtypes.i32 [22] dtypes.i32
56 Ops.WHERE dtypes.i32 [54, 9, 55]
57 Ops.ADD dtypes.i32 [48, 56]
58 Ops.TRUNCMOD dtypes.i32 [7, 11]
59 Ops.CMPEQ dtypes.bool [58, 22]
60 Ops.CMPEQ dtypes.bool [59, 38]
61 Ops.CMPLT dtypes.bool [11, 22]
62 Ops.CMPEQ dtypes.bool [40, 61]
63 Ops.CMPEQ dtypes.bool [62, 38]
64 Ops.AND dtypes.bool [60, 63]
65 Ops.WHERE dtypes.weakint [64, 11, 22]
66 Ops.CAST dtypes.i32 [65] dtypes.i32
67 Ops.ADD dtypes.i32 [58, 66]
68 Ops.TRUNCMOD dtypes.i32 [7, 13]
69 Ops.CMPEQ dtypes.bool [68, 22]
70 Ops.CMPEQ dtypes.bool [69, 38]
71 Ops.CMPLT dtypes.bool [13, 22]
72 Ops.CMPEQ dtypes.bool [40, 71]
73 Ops.CMPEQ dtypes.bool [72, 38]
74 Ops.AND dtypes.bool [70, 73]
75 Ops.WHERE dtypes.weakint [74, 13, 22]
76 Ops.CAST dtypes.i32 [75] dtypes.i32
77 Ops.ADD dtypes.i32 [68, 76]
78 Ops.TRUNCMOD dtypes.i32 [7, 15]
79 Ops.CMPEQ dtypes.bool [78, 22]
80 Ops.CMPEQ dtypes.bool [79, 38]
81 Ops.CMPLT dtypes.bool [15, 22]
82 Ops.CMPEQ dtypes.bool [40, 81]
83 Ops.CMPEQ dtypes.bool [82, 38]
84 Ops.AND dtypes.bool [80, 83]
85 Ops.WHERE dtypes.weakint [84, 15, 22]
86 Ops.CAST dtypes.i32 [85] dtypes.i32
87 Ops.ADD dtypes.i32 [78, 86]
88 Ops.TRUNCMOD dtypes.u64 [17, 19]
89 Ops.TRUNCMOD dtypes.u64 [17, 18]
90 Ops.TRUNCMOD dtypes.u64 [17, 22]
91 Ops.TRUNCMOD dtypes.weakint [24, 25]
92 Ops.CMPEQ dtypes.bool [91, 22]
93 Ops.CMPEQ dtypes.bool [92, 38]
94 Ops.CMPLT dtypes.bool [24, 22]
95 Ops.CMPLT dtypes.bool [25, 22]
96 Ops.CMPEQ dtypes.bool [94, 95]
97 Ops.CMPEQ dtypes.bool [96, 38]
98 Ops.AND dtypes.bool [93, 97]
99 Ops.WHERE dtypes.weakint [98, 25, 22]
100 Ops.ADD dtypes.weakint [91, 99]
101 Ops.TRUNCMOD dtypes.i32 [27, 28]
102 Ops.TRUNCMOD dtypes.i32 [30, 31]
103 Ops.SINK dtypes.void [34, 35, 47, 57, 67, 77, 87, 88, 89, 90, 100, 101, 102]
104 Ops.SINK dtypes.void [33, 103]

