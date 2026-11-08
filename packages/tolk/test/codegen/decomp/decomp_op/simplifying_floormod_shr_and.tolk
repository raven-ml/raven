# tolk D413 golden=e8362646c9c26524e5a1b8fa9d2790d2
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
58 Ops.AND dtypes.i32 [7, 2]
59 Ops.AND dtypes.i32 [7, 22]
60 Ops.TRUNCMOD dtypes.i32 [7, 15]
61 Ops.CMPEQ dtypes.bool [60, 22]
62 Ops.CMPEQ dtypes.bool [61, 38]
63 Ops.CMPLT dtypes.bool [15, 22]
64 Ops.CMPEQ dtypes.bool [40, 63]
65 Ops.CMPEQ dtypes.bool [64, 38]
66 Ops.AND dtypes.bool [62, 65]
67 Ops.WHERE dtypes.weakint [66, 15, 22]
68 Ops.CAST dtypes.i32 [67] dtypes.i32
69 Ops.ADD dtypes.i32 [60, 68]
70 Ops.TRUNCMOD dtypes.u64 [17, 19]
71 Ops.CONST dtypes.weakint [] 9223372036854775807
72 Ops.AND dtypes.u64 [17, 71]
73 Ops.TRUNCMOD dtypes.u64 [17, 22]
74 Ops.TRUNCMOD dtypes.weakint [24, 25]
75 Ops.CMPEQ dtypes.bool [74, 22]
76 Ops.CMPEQ dtypes.bool [75, 38]
77 Ops.CMPLT dtypes.bool [24, 22]
78 Ops.CMPLT dtypes.bool [25, 22]
79 Ops.CMPEQ dtypes.bool [77, 78]
80 Ops.CMPEQ dtypes.bool [79, 38]
81 Ops.AND dtypes.bool [76, 80]
82 Ops.WHERE dtypes.weakint [81, 25, 22]
83 Ops.ADD dtypes.weakint [74, 82]
84 Ops.TRUNCMOD dtypes.i32 [27, 28]
85 Ops.TRUNCMOD dtypes.i32 [30, 31]
86 Ops.SINK dtypes.void [34, 35, 47, 57, 58, 59, 69, 70, 72, 73, 83, 84, 85]
87 Ops.SINK dtypes.void [33, 86]

