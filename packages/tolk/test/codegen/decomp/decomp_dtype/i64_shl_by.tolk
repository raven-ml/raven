# tolk D342 golden=8b953f001560946de1f24cf6c98980dc
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 8
2 Ops.PARAM dtypes.i64 [1] ParamArg(slot=0, dtype=dtypes.i64)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.SHL dtypes.i32 [5, 3]
7 Ops.INDEX dtypes.i64 [2, 6]
8 Ops.PARAM dtypes.i64 [1] ParamArg(slot=1, dtype=dtypes.i64)
9 Ops.INDEX dtypes.i64 [8, 6]
10 Ops.LOAD dtypes.i64 [9]
11 Ops.PARAM dtypes.i64 [1] ParamArg(slot=2, dtype=dtypes.i64)
12 Ops.INDEX dtypes.i64 [11, 6]
13 Ops.LOAD dtypes.i64 [12]
14 Ops.SHL dtypes.i64 [10, 13]
15 Ops.STORE dtypes.void [7, 14]
16 Ops.CONST dtypes.weakint [] 1
17 Ops.ADD dtypes.i32 [6, 16]
18 Ops.INDEX dtypes.i64 [2, 17]
19 Ops.INDEX dtypes.i64 [8, 17]
20 Ops.LOAD dtypes.i64 [19]
21 Ops.INDEX dtypes.i64 [11, 17]
22 Ops.LOAD dtypes.i64 [21]
23 Ops.SHL dtypes.i64 [20, 22]
24 Ops.STORE dtypes.void [18, 23]
25 Ops.ADD dtypes.i32 [6, 3]
26 Ops.INDEX dtypes.i64 [2, 25]
27 Ops.INDEX dtypes.i64 [8, 25]
28 Ops.LOAD dtypes.i64 [27]
29 Ops.INDEX dtypes.i64 [11, 25]
30 Ops.LOAD dtypes.i64 [29]
31 Ops.SHL dtypes.i64 [28, 30]
32 Ops.STORE dtypes.void [26, 31]
33 Ops.CONST dtypes.weakint [] 3
34 Ops.ADD dtypes.i32 [6, 33]
35 Ops.INDEX dtypes.i64 [2, 34]
36 Ops.INDEX dtypes.i64 [8, 34]
37 Ops.LOAD dtypes.i64 [36]
38 Ops.INDEX dtypes.i64 [11, 34]
39 Ops.LOAD dtypes.i64 [38]
40 Ops.SHL dtypes.i64 [37, 39]
41 Ops.STORE dtypes.void [35, 40]
42 Ops.AFTER dtypes.void [0, 15, 24, 32, 41]
43 Ops.END dtypes.void [42, 5]
44 Ops.SINK dtypes.void [43] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
45 Ops.CONST dtypes.weakint [] 16
46 Ops.PARAM dtypes.i32 [45] ParamArg(slot=0, dtype=dtypes.i32)
47 Ops.MUL dtypes.i32 [6, 3]
48 Ops.CONST dtypes.weakint [] 0
49 Ops.ADD dtypes.i32 [47, 48]
50 Ops.INDEX dtypes.i32 [46, 49]
51 Ops.PARAM dtypes.i32 [45] ParamArg(slot=2, dtype=dtypes.i32)
52 Ops.ADD dtypes.i32 [47, 16]
53 Ops.INDEX dtypes.i32 [51, 52]
54 Ops.LOAD dtypes.i32 [53]
55 Ops.CAST dtypes.i32 [48] dtypes.i32
56 Ops.CMPLT dtypes.bool [54, 55]
57 Ops.CMPEQ dtypes.bool [54, 55]
58 Ops.INDEX dtypes.i32 [51, 49]
59 Ops.LOAD dtypes.i32 [58]
60 Ops.BITCAST dtypes.u32 [59] dtypes.u32
61 Ops.CONST dtypes.weakint [] 64
62 Ops.CAST dtypes.i32 [61] dtypes.i32
63 Ops.BITCAST dtypes.u32 [62] dtypes.u32
64 Ops.CMPLT dtypes.bool [60, 63]
65 Ops.AND dtypes.bool [57, 64]
66 Ops.OR dtypes.bool [56, 65]
67 Ops.CONST dtypes.weakint [] 32
68 Ops.CMPLT dtypes.bool [59, 67]
69 Ops.CONST dtypes.bool [] False
70 Ops.CMPEQ dtypes.bool [68, 69]
71 Ops.PARAM dtypes.i32 [45] ParamArg(slot=1, dtype=dtypes.i32)
72 Ops.INDEX dtypes.i32 [71, 49]
73 Ops.LOAD dtypes.i32 [72]
74 Ops.BITCAST dtypes.u32 [73] dtypes.u32
75 Ops.CONST dtypes.weakint [] 31
76 Ops.AND dtypes.i32 [59, 75]
77 Ops.CAST dtypes.u32 [76] dtypes.u32
78 Ops.SHL dtypes.u32 [74, 77]
79 Ops.BITCAST dtypes.i32 [78] dtypes.i32
80 Ops.WHERE dtypes.i32 [70, 55, 79]
81 Ops.WHERE dtypes.i32 [66, 80, 55]
82 Ops.STORE dtypes.void [50, 81]
83 Ops.INDEX dtypes.i32 [46, 52]
84 Ops.INDEX dtypes.i32 [71, 52]
85 Ops.LOAD dtypes.i32 [84]
86 Ops.BITCAST dtypes.u32 [85] dtypes.u32
87 Ops.SHL dtypes.u32 [86, 77]
88 Ops.SHR dtypes.u32 [74, 16]
89 Ops.CONST dtypes.weakint [] 4294967295
90 Ops.MUL dtypes.u32 [77, 89]
91 Ops.ADD dtypes.u32 [75, 90]
92 Ops.SHR dtypes.u32 [88, 91]
93 Ops.OR dtypes.u32 [87, 92]
94 Ops.BITCAST dtypes.i32 [93] dtypes.i32
95 Ops.WHERE dtypes.i32 [70, 79, 94]
96 Ops.WHERE dtypes.i32 [66, 95, 55]
97 Ops.STORE dtypes.void [83, 96]
98 Ops.AFTER dtypes.void [0, 82, 97]
99 Ops.MUL dtypes.i32 [17, 3]
100 Ops.ADD dtypes.i32 [99, 48]
101 Ops.INDEX dtypes.i32 [46, 100]
102 Ops.ADD dtypes.i32 [99, 16]
103 Ops.INDEX dtypes.i32 [51, 102]
104 Ops.LOAD dtypes.i32 [103]
105 Ops.CMPLT dtypes.bool [104, 55]
106 Ops.CMPEQ dtypes.bool [104, 55]
107 Ops.INDEX dtypes.i32 [51, 100]
108 Ops.LOAD dtypes.i32 [107]
109 Ops.BITCAST dtypes.u32 [108] dtypes.u32
110 Ops.CMPLT dtypes.bool [109, 63]
111 Ops.AND dtypes.bool [106, 110]
112 Ops.OR dtypes.bool [105, 111]
113 Ops.CMPLT dtypes.bool [108, 67]
114 Ops.CMPEQ dtypes.bool [113, 69]
115 Ops.INDEX dtypes.i32 [71, 100]
116 Ops.LOAD dtypes.i32 [115]
117 Ops.BITCAST dtypes.u32 [116] dtypes.u32
118 Ops.AND dtypes.i32 [108, 75]
119 Ops.CAST dtypes.u32 [118] dtypes.u32
120 Ops.SHL dtypes.u32 [117, 119]
121 Ops.BITCAST dtypes.i32 [120] dtypes.i32
122 Ops.WHERE dtypes.i32 [114, 55, 121]
123 Ops.WHERE dtypes.i32 [112, 122, 55]
124 Ops.STORE dtypes.void [101, 123]
125 Ops.INDEX dtypes.i32 [46, 102]
126 Ops.INDEX dtypes.i32 [71, 102]
127 Ops.LOAD dtypes.i32 [126]
128 Ops.BITCAST dtypes.u32 [127] dtypes.u32
129 Ops.SHL dtypes.u32 [128, 119]
130 Ops.SHR dtypes.u32 [117, 16]
131 Ops.MUL dtypes.u32 [119, 89]
132 Ops.ADD dtypes.u32 [75, 131]
133 Ops.SHR dtypes.u32 [130, 132]
134 Ops.OR dtypes.u32 [129, 133]
135 Ops.BITCAST dtypes.i32 [134] dtypes.i32
136 Ops.WHERE dtypes.i32 [114, 121, 135]
137 Ops.WHERE dtypes.i32 [112, 136, 55]
138 Ops.STORE dtypes.void [125, 137]
139 Ops.AFTER dtypes.void [0, 124, 138]
140 Ops.MUL dtypes.i32 [25, 3]
141 Ops.ADD dtypes.i32 [140, 48]
142 Ops.INDEX dtypes.i32 [46, 141]
143 Ops.ADD dtypes.i32 [140, 16]
144 Ops.INDEX dtypes.i32 [51, 143]
145 Ops.LOAD dtypes.i32 [144]
146 Ops.CMPLT dtypes.bool [145, 55]
147 Ops.CMPEQ dtypes.bool [145, 55]
148 Ops.INDEX dtypes.i32 [51, 141]
149 Ops.LOAD dtypes.i32 [148]
150 Ops.BITCAST dtypes.u32 [149] dtypes.u32
151 Ops.CMPLT dtypes.bool [150, 63]
152 Ops.AND dtypes.bool [147, 151]
153 Ops.OR dtypes.bool [146, 152]
154 Ops.CMPLT dtypes.bool [149, 67]
155 Ops.CMPEQ dtypes.bool [154, 69]
156 Ops.INDEX dtypes.i32 [71, 141]
157 Ops.LOAD dtypes.i32 [156]
158 Ops.BITCAST dtypes.u32 [157] dtypes.u32
159 Ops.AND dtypes.i32 [149, 75]
160 Ops.CAST dtypes.u32 [159] dtypes.u32
161 Ops.SHL dtypes.u32 [158, 160]
162 Ops.BITCAST dtypes.i32 [161] dtypes.i32
163 Ops.WHERE dtypes.i32 [155, 55, 162]
164 Ops.WHERE dtypes.i32 [153, 163, 55]
165 Ops.STORE dtypes.void [142, 164]
166 Ops.INDEX dtypes.i32 [46, 143]
167 Ops.INDEX dtypes.i32 [71, 143]
168 Ops.LOAD dtypes.i32 [167]
169 Ops.BITCAST dtypes.u32 [168] dtypes.u32
170 Ops.SHL dtypes.u32 [169, 160]
171 Ops.SHR dtypes.u32 [158, 16]
172 Ops.MUL dtypes.u32 [160, 89]
173 Ops.ADD dtypes.u32 [75, 172]
174 Ops.SHR dtypes.u32 [171, 173]
175 Ops.OR dtypes.u32 [170, 174]
176 Ops.BITCAST dtypes.i32 [175] dtypes.i32
177 Ops.WHERE dtypes.i32 [155, 162, 176]
178 Ops.WHERE dtypes.i32 [153, 177, 55]
179 Ops.STORE dtypes.void [166, 178]
180 Ops.AFTER dtypes.void [0, 165, 179]
181 Ops.MUL dtypes.i32 [34, 3]
182 Ops.ADD dtypes.i32 [181, 48]
183 Ops.INDEX dtypes.i32 [46, 182]
184 Ops.ADD dtypes.i32 [181, 16]
185 Ops.INDEX dtypes.i32 [51, 184]
186 Ops.LOAD dtypes.i32 [185]
187 Ops.CMPLT dtypes.bool [186, 55]
188 Ops.CMPEQ dtypes.bool [186, 55]
189 Ops.INDEX dtypes.i32 [51, 182]
190 Ops.LOAD dtypes.i32 [189]
191 Ops.BITCAST dtypes.u32 [190] dtypes.u32
192 Ops.CMPLT dtypes.bool [191, 63]
193 Ops.AND dtypes.bool [188, 192]
194 Ops.OR dtypes.bool [187, 193]
195 Ops.CMPLT dtypes.bool [190, 67]
196 Ops.CMPEQ dtypes.bool [195, 69]
197 Ops.INDEX dtypes.i32 [71, 182]
198 Ops.LOAD dtypes.i32 [197]
199 Ops.BITCAST dtypes.u32 [198] dtypes.u32
200 Ops.AND dtypes.i32 [190, 75]
201 Ops.CAST dtypes.u32 [200] dtypes.u32
202 Ops.SHL dtypes.u32 [199, 201]
203 Ops.BITCAST dtypes.i32 [202] dtypes.i32
204 Ops.WHERE dtypes.i32 [196, 55, 203]
205 Ops.WHERE dtypes.i32 [194, 204, 55]
206 Ops.STORE dtypes.void [183, 205]
207 Ops.INDEX dtypes.i32 [46, 184]
208 Ops.INDEX dtypes.i32 [71, 184]
209 Ops.LOAD dtypes.i32 [208]
210 Ops.BITCAST dtypes.u32 [209] dtypes.u32
211 Ops.SHL dtypes.u32 [210, 201]
212 Ops.SHR dtypes.u32 [199, 16]
213 Ops.MUL dtypes.u32 [201, 89]
214 Ops.ADD dtypes.u32 [75, 213]
215 Ops.SHR dtypes.u32 [212, 214]
216 Ops.OR dtypes.u32 [211, 215]
217 Ops.BITCAST dtypes.i32 [216] dtypes.i32
218 Ops.WHERE dtypes.i32 [196, 203, 217]
219 Ops.WHERE dtypes.i32 [194, 218, 55]
220 Ops.STORE dtypes.void [207, 219]
221 Ops.AFTER dtypes.void [0, 206, 220]
222 Ops.AFTER dtypes.void [0, 98, 139, 180, 221]
223 Ops.END dtypes.void [222, 5]
224 Ops.SINK dtypes.void [223] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
225 Ops.SINK dtypes.void [44, 224]

