# tolk D80 D151 golden=ec859e3f8ff99fbc15e94e733013838b
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.u64 [1] ParamArg(slot=0, dtype=dtypes.u64)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 3
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.INDEX dtypes.u64 [2, 10]
12 Ops.PARAM dtypes.u64 [1] ParamArg(slot=1, dtype=dtypes.u64)
13 Ops.INDEX dtypes.u64 [12, 10]
14 Ops.LOAD dtypes.u64 [13]
15 Ops.CONST dtypes.weakint [] 1
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.ADD dtypes.i32 [10, 16]
18 Ops.INDEX dtypes.u64 [12, 17]
19 Ops.LOAD dtypes.u64 [18]
20 Ops.CONST dtypes.weakint [] 2
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.ADD dtypes.i32 [10, 21]
23 Ops.INDEX dtypes.u64 [12, 22]
24 Ops.LOAD dtypes.u64 [23]
25 Ops.ADD dtypes.i32 [10, 9]
26 Ops.INDEX dtypes.u64 [12, 25]
27 Ops.LOAD dtypes.u64 [26]
28 Ops.CONST dtypes.weakint [] 4
29 Ops.CAST dtypes.i32 [28] dtypes.i32
30 Ops.ADD dtypes.i32 [10, 29]
31 Ops.INDEX dtypes.u64 [12, 30]
32 Ops.LOAD dtypes.u64 [31]
33 Ops.CONST dtypes.weakint [] 5
34 Ops.CAST dtypes.i32 [33] dtypes.i32
35 Ops.ADD dtypes.i32 [10, 34]
36 Ops.INDEX dtypes.u64 [12, 35]
37 Ops.LOAD dtypes.u64 [36]
38 Ops.CONST dtypes.weakint [] 6
39 Ops.CAST dtypes.i32 [38] dtypes.i32
40 Ops.ADD dtypes.i32 [10, 39]
41 Ops.INDEX dtypes.u64 [12, 40]
42 Ops.LOAD dtypes.u64 [41]
43 Ops.CONST dtypes.weakint [] 7
44 Ops.CAST dtypes.i32 [43] dtypes.i32
45 Ops.ADD dtypes.i32 [10, 44]
46 Ops.INDEX dtypes.u64 [12, 45]
47 Ops.LOAD dtypes.u64 [46]
48 Ops.STACK dtypes.u64 [14, 19, 24, 27, 32, 37, 42, 47]
49 Ops.CAST dtypes.u64 [8] dtypes.u64
50 Ops.SHR dtypes.u64 [48, 49]
51 Ops.CONST dtypes.weakint [] 0
52 Ops.CAST dtypes.i32 [51] dtypes.i32
53 Ops.INDEX dtypes.u64 [50, 52]
54 Ops.STORE dtypes.void [11, 53]
55 Ops.INDEX dtypes.u64 [2, 17]
56 Ops.INDEX dtypes.u64 [50, 16]
57 Ops.STORE dtypes.void [55, 56]
58 Ops.INDEX dtypes.u64 [2, 22]
59 Ops.INDEX dtypes.u64 [50, 21]
60 Ops.STORE dtypes.void [58, 59]
61 Ops.INDEX dtypes.u64 [2, 25]
62 Ops.INDEX dtypes.u64 [50, 9]
63 Ops.STORE dtypes.void [61, 62]
64 Ops.INDEX dtypes.u64 [2, 30]
65 Ops.INDEX dtypes.u64 [50, 29]
66 Ops.STORE dtypes.void [64, 65]
67 Ops.INDEX dtypes.u64 [2, 35]
68 Ops.INDEX dtypes.u64 [50, 34]
69 Ops.STORE dtypes.void [67, 68]
70 Ops.INDEX dtypes.u64 [2, 40]
71 Ops.INDEX dtypes.u64 [50, 39]
72 Ops.STORE dtypes.void [70, 71]
73 Ops.INDEX dtypes.u64 [2, 45]
74 Ops.INDEX dtypes.u64 [50, 44]
75 Ops.STORE dtypes.void [73, 74]
76 Ops.AFTER dtypes.void [0, 54, 57, 60, 63, 66, 69, 72, 75]
77 Ops.END dtypes.void [76, 6]
78 Ops.SINK dtypes.void [77] KernelInfo(name="E_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)),), optimized=True, split=(2, 2, 3))
79 Ops.CONST dtypes.weakint [] 32
80 Ops.PARAM dtypes.u8 [79] ParamArg(slot=0, dtype=dtypes.u8, name="args")
81 Ops.BITCAST dtypes.u64 [80] dtypes.u64
82 Ops.INDEX dtypes.u64 [81, 52]
83 Ops.LOAD dtypes.u64 [82]
84 Ops.REFPARAM dtypes.u64 [1, 83] ParamArg(slot=0, dtype=dtypes.u64)
85 Ops.BITCAST dtypes.i32 [80] dtypes.i32
86 Ops.INDEX dtypes.i32 [85, 29]
87 Ops.LOAD dtypes.i32 [86]
88 Ops.INDEX dtypes.i32 [85, 34]
89 Ops.LOAD dtypes.i32 [88]
90 Ops.SUB dtypes.i32 [89, 87]
91 Ops.RANGE dtypes.i32 [90] (AxisType.WEAK, 0)
92 Ops.ADD dtypes.i32 [87, 91]
93 Ops.SHL dtypes.i32 [92, 9]
94 Ops.INDEX dtypes.u64 [84, 93]
95 Ops.INDEX dtypes.u64 [81, 16]
96 Ops.LOAD dtypes.u64 [95]
97 Ops.REFPARAM dtypes.u64 [1, 96] ParamArg(slot=1, dtype=dtypes.u64)
98 Ops.INDEX dtypes.u64 [97, 93]
99 Ops.LOAD dtypes.u64 [98]
100 Ops.ADD dtypes.i32 [93, 16]
101 Ops.INDEX dtypes.u64 [97, 100]
102 Ops.LOAD dtypes.u64 [101]
103 Ops.ADD dtypes.i32 [93, 21]
104 Ops.INDEX dtypes.u64 [97, 103]
105 Ops.LOAD dtypes.u64 [104]
106 Ops.ADD dtypes.i32 [93, 9]
107 Ops.INDEX dtypes.u64 [97, 106]
108 Ops.LOAD dtypes.u64 [107]
109 Ops.ADD dtypes.i32 [93, 29]
110 Ops.INDEX dtypes.u64 [97, 109]
111 Ops.LOAD dtypes.u64 [110]
112 Ops.ADD dtypes.i32 [93, 34]
113 Ops.INDEX dtypes.u64 [97, 112]
114 Ops.LOAD dtypes.u64 [113]
115 Ops.ADD dtypes.i32 [93, 39]
116 Ops.INDEX dtypes.u64 [97, 115]
117 Ops.LOAD dtypes.u64 [116]
118 Ops.ADD dtypes.i32 [93, 44]
119 Ops.INDEX dtypes.u64 [97, 118]
120 Ops.LOAD dtypes.u64 [119]
121 Ops.STACK dtypes.u64 [99, 102, 105, 108, 111, 114, 117, 120]
122 Ops.SHR dtypes.u64 [121, 49]
123 Ops.INDEX dtypes.u64 [122, 52]
124 Ops.STORE dtypes.void [94, 123]
125 Ops.INDEX dtypes.u64 [84, 100]
126 Ops.INDEX dtypes.u64 [122, 16]
127 Ops.STORE dtypes.void [125, 126]
128 Ops.INDEX dtypes.u64 [84, 103]
129 Ops.INDEX dtypes.u64 [122, 21]
130 Ops.STORE dtypes.void [128, 129]
131 Ops.INDEX dtypes.u64 [84, 106]
132 Ops.INDEX dtypes.u64 [122, 9]
133 Ops.STORE dtypes.void [131, 132]
134 Ops.INDEX dtypes.u64 [84, 109]
135 Ops.INDEX dtypes.u64 [122, 29]
136 Ops.STORE dtypes.void [134, 135]
137 Ops.INDEX dtypes.u64 [84, 112]
138 Ops.INDEX dtypes.u64 [122, 34]
139 Ops.STORE dtypes.void [137, 138]
140 Ops.INDEX dtypes.u64 [84, 115]
141 Ops.INDEX dtypes.u64 [122, 39]
142 Ops.STORE dtypes.void [140, 141]
143 Ops.INDEX dtypes.u64 [84, 118]
144 Ops.INDEX dtypes.u64 [122, 44]
145 Ops.STORE dtypes.void [143, 144]
146 Ops.AFTER dtypes.void [0, 124, 127, 130, 133, 136, 139, 142, 145]
147 Ops.END dtypes.void [146, 91]
148 Ops.SINK dtypes.void [147] KernelInfo(name="E_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)),), estimates=Estimates(ops=16, lds=256, mem=256), optimized=True, split=(2, 2, 3))
149 Ops.LINEAR dtypes.void [79, 80, 28, 29, 85, 86, 87, 33, 34, 88, 89, 51, 52, 81, 82, 83, 15, 16, 95, 96, 1, 84, 97, 0, 20, 21, 8, 9, 38, 39, 43, 44, 49, 90, 91, 92, 93, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 98, 99, 121, 125, 128, 131, 134, 137, 140, 143, 94, 122, 123, 126, 129, 132, 135, 138, 141, 144, 127, 130, 133, 136, 139, 142, 145, 124, 146, 147, 148]
150 Ops.SOURCE dtypes.void [] "typedef unsigned long unsigned_long8 __attribute__((aligned(8),ext_vector_type(8)));\n__attribute__((min_vector_width(512))) void E_2_8(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = ((val0+Lidx0)<<3);\n    int alu1 = (alu0+1);\n    unsigned long val4 = (*(((unsigned long*)(val3))+alu1));\n    int alu2 = (alu0+2);\n    unsigned long val5 = (*(((unsigned long*)(val3))+alu2));\n    int alu3 = (alu0+3);\n    unsigned long val6 = (*(((unsigned long*)(val3))+alu3));\n    int alu4 = (alu0+4);\n    unsigned long val7 = (*(((unsigned long*)(val3))+alu4));\n    int alu5 = (alu0+5);\n    unsigned long val8 = (*(((unsigned long*)(val3))+alu5));\n    int alu6 = (alu0+6);\n    unsigned long val9 = (*(((unsigned long*)(val3))+alu6));\n    int alu7 = (alu0+7);\n    unsigned long val10 = (*(((unsigned long*)(val3))+alu7));\n    unsigned long val11 = (*(((unsigned long*)(val3))+alu0));\n    unsigned_long8 alu8 = ((unsigned_long8){val11,val4,val5,val6,val7,val8,val9,val10}>>3ul);\n    *(((unsigned long*)(val2))+alu1) = alu8[1];\n    *(((unsigned long*)(val2))+alu2) = alu8[2];\n    *(((unsigned long*)(val2))+alu3) = alu8[3];\n    *(((unsigned long*)(val2))+alu4) = alu8[4];\n    *(((unsigned long*)(val2))+alu5) = alu8[5];\n    *(((unsigned long*)(val2))+alu6) = alu8[6];\n    *(((unsigned long*)(val2))+alu7) = alu8[7];\n    *(((unsigned long*)(val2))+alu0) = alu8[0];\n  }\n}\n"
151 Ops.PROGRAM dtypes.void [148, 149, 150] ProgramInfo(global_size=(2, 1, 1), params=(%2, %12, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
152 Ops.SINK dtypes.void [78, 151]

