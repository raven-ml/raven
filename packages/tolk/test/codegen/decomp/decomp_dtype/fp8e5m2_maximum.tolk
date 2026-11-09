# tolk D27 golden=108c075fe55cacc1f9d18c92d05ec2f3
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 8
2 Ops.PARAM dtypes.fp8e5m2 [1] ParamArg(slot=0, dtype=dtypes.fp8e5m2)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.SHL dtypes.i32 [5, 3]
7 Ops.CONST dtypes.weakint [] 4
8 Ops.CAST dtypes.i32 [7] dtypes.i32
9 Ops.SHRINK dtypes.fp8e5m2 [2, 6, 8]
10 Ops.PARAM dtypes.fp8e5m2 [1] ParamArg(slot=1, dtype=dtypes.fp8e5m2)
11 Ops.SHRINK dtypes.fp8e5m2 [10, 6, 8]
12 Ops.LOAD dtypes.fp8e5m2 [11]
13 Ops.CONST dtypes.weakint [] 0
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.INDEX dtypes.fp8e5m2 [12, 14]
16 Ops.PARAM dtypes.fp8e5m2 [1] ParamArg(slot=2, dtype=dtypes.fp8e5m2)
17 Ops.SHRINK dtypes.fp8e5m2 [16, 6, 8]
18 Ops.LOAD dtypes.fp8e5m2 [17]
19 Ops.INDEX dtypes.fp8e5m2 [18, 14]
20 Ops.CMPLT dtypes.bool [15, 19]
21 Ops.WHERE dtypes.fp8e5m2 [20, 19, 15]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.fp8e5m2 [12, 23]
25 Ops.INDEX dtypes.fp8e5m2 [18, 23]
26 Ops.CMPLT dtypes.bool [24, 25]
27 Ops.WHERE dtypes.fp8e5m2 [26, 25, 24]
28 Ops.INDEX dtypes.fp8e5m2 [12, 4]
29 Ops.INDEX dtypes.fp8e5m2 [18, 4]
30 Ops.CMPLT dtypes.bool [28, 29]
31 Ops.WHERE dtypes.fp8e5m2 [30, 29, 28]
32 Ops.CONST dtypes.weakint [] 3
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.fp8e5m2 [12, 33]
35 Ops.INDEX dtypes.fp8e5m2 [18, 33]
36 Ops.CMPLT dtypes.bool [34, 35]
37 Ops.WHERE dtypes.fp8e5m2 [36, 35, 34]
38 Ops.STACK dtypes.fp8e5m2 [21, 27, 31, 37]
39 Ops.STORE dtypes.void [9, 38]
40 Ops.AFTER dtypes.void [0, 39]
41 Ops.END dtypes.void [40, 5]
42 Ops.SINK dtypes.void [41] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
43 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8)
44 Ops.SHRINK dtypes.u8 [43, 6, 8]
45 Ops.PARAM dtypes.u8 [1] ParamArg(slot=1, dtype=dtypes.u8)
46 Ops.ADD dtypes.i32 [6, 13]
47 Ops.INDEX dtypes.u8 [45, 46]
48 Ops.LOAD dtypes.u8 [47]
49 Ops.CONST dtypes.weakint [] 128
50 Ops.AND dtypes.u8 [48, 49]
51 Ops.CAST dtypes.u32 [50] dtypes.u32
52 Ops.CONST dtypes.weakint [] 16777216
53 Ops.MUL dtypes.u32 [51, 52]
54 Ops.CONST dtypes.weakint [] 127
55 Ops.AND dtypes.u8 [48, 54]
56 Ops.CAST dtypes.u32 [55] dtypes.u32
57 Ops.DIV dtypes.u32 [56, 7]
58 Ops.CONST dtypes.weakint [] 31
59 Ops.CMPEQ dtypes.bool [57, 58]
60 Ops.CONST dtypes.weakint [] 2097152
61 Ops.MUL dtypes.u32 [56, 60]
62 Ops.CONST dtypes.weakint [] 2139095040
63 Ops.OR dtypes.u32 [61, 62]
64 Ops.CMPEQ dtypes.bool [57, 13]
65 Ops.CMPEQ dtypes.bool [56, 13]
66 Ops.CAST dtypes.f32 [56] dtypes.f32
67 Ops.BITCAST dtypes.u32 [66] dtypes.u32
68 Ops.CONST dtypes.weakint [] 134217728
69 Ops.CAST dtypes.i64 [68] dtypes.i64
70 Ops.CONST dtypes.weakint [] -1
71 Ops.CAST dtypes.i64 [70] dtypes.i64
72 Ops.MUL dtypes.i64 [69, 71]
73 Ops.CAST dtypes.u32 [72] dtypes.u32
74 Ops.ADD dtypes.u32 [67, 73]
75 Ops.WHERE dtypes.u32 [65, 13, 74]
76 Ops.CONST dtypes.weakint [] 939524096
77 Ops.ADD dtypes.u32 [61, 76]
78 Ops.WHERE dtypes.u32 [64, 75, 77]
79 Ops.WHERE dtypes.u32 [59, 63, 78]
80 Ops.OR dtypes.u32 [53, 79]
81 Ops.BITCAST dtypes.f32 [80] dtypes.f32
82 Ops.ADD dtypes.i32 [6, 22]
83 Ops.INDEX dtypes.u8 [45, 82]
84 Ops.LOAD dtypes.u8 [83]
85 Ops.AND dtypes.u8 [84, 49]
86 Ops.CAST dtypes.u32 [85] dtypes.u32
87 Ops.MUL dtypes.u32 [86, 52]
88 Ops.AND dtypes.u8 [84, 54]
89 Ops.CAST dtypes.u32 [88] dtypes.u32
90 Ops.DIV dtypes.u32 [89, 7]
91 Ops.CMPEQ dtypes.bool [90, 58]
92 Ops.MUL dtypes.u32 [89, 60]
93 Ops.OR dtypes.u32 [92, 62]
94 Ops.CMPEQ dtypes.bool [90, 13]
95 Ops.CMPEQ dtypes.bool [89, 13]
96 Ops.CAST dtypes.f32 [89] dtypes.f32
97 Ops.BITCAST dtypes.u32 [96] dtypes.u32
98 Ops.ADD dtypes.u32 [97, 73]
99 Ops.WHERE dtypes.u32 [95, 13, 98]
100 Ops.ADD dtypes.u32 [92, 76]
101 Ops.WHERE dtypes.u32 [94, 99, 100]
102 Ops.WHERE dtypes.u32 [91, 93, 101]
103 Ops.OR dtypes.u32 [87, 102]
104 Ops.BITCAST dtypes.f32 [103] dtypes.f32
105 Ops.ADD dtypes.i32 [6, 3]
106 Ops.INDEX dtypes.u8 [45, 105]
107 Ops.LOAD dtypes.u8 [106]
108 Ops.AND dtypes.u8 [107, 49]
109 Ops.CAST dtypes.u32 [108] dtypes.u32
110 Ops.MUL dtypes.u32 [109, 52]
111 Ops.AND dtypes.u8 [107, 54]
112 Ops.CAST dtypes.u32 [111] dtypes.u32
113 Ops.DIV dtypes.u32 [112, 7]
114 Ops.CMPEQ dtypes.bool [113, 58]
115 Ops.MUL dtypes.u32 [112, 60]
116 Ops.OR dtypes.u32 [115, 62]
117 Ops.CMPEQ dtypes.bool [113, 13]
118 Ops.CMPEQ dtypes.bool [112, 13]
119 Ops.CAST dtypes.f32 [112] dtypes.f32
120 Ops.BITCAST dtypes.u32 [119] dtypes.u32
121 Ops.ADD dtypes.u32 [120, 73]
122 Ops.WHERE dtypes.u32 [118, 13, 121]
123 Ops.ADD dtypes.u32 [115, 76]
124 Ops.WHERE dtypes.u32 [117, 122, 123]
125 Ops.WHERE dtypes.u32 [114, 116, 124]
126 Ops.OR dtypes.u32 [110, 125]
127 Ops.BITCAST dtypes.f32 [126] dtypes.f32
128 Ops.ADD dtypes.i32 [6, 32]
129 Ops.INDEX dtypes.u8 [45, 128]
130 Ops.LOAD dtypes.u8 [129]
131 Ops.AND dtypes.u8 [130, 49]
132 Ops.CAST dtypes.u32 [131] dtypes.u32
133 Ops.MUL dtypes.u32 [132, 52]
134 Ops.AND dtypes.u8 [130, 54]
135 Ops.CAST dtypes.u32 [134] dtypes.u32
136 Ops.DIV dtypes.u32 [135, 7]
137 Ops.CMPEQ dtypes.bool [136, 58]
138 Ops.MUL dtypes.u32 [135, 60]
139 Ops.OR dtypes.u32 [138, 62]
140 Ops.CMPEQ dtypes.bool [136, 13]
141 Ops.CMPEQ dtypes.bool [135, 13]
142 Ops.CAST dtypes.f32 [135] dtypes.f32
143 Ops.BITCAST dtypes.u32 [142] dtypes.u32
144 Ops.ADD dtypes.u32 [143, 73]
145 Ops.WHERE dtypes.u32 [141, 13, 144]
146 Ops.ADD dtypes.u32 [138, 76]
147 Ops.WHERE dtypes.u32 [140, 145, 146]
148 Ops.WHERE dtypes.u32 [137, 139, 147]
149 Ops.OR dtypes.u32 [133, 148]
150 Ops.BITCAST dtypes.f32 [149] dtypes.f32
151 Ops.STACK dtypes.f32 [81, 104, 127, 150]
152 Ops.CAST dtypes.f32 [151] dtypes.f32
153 Ops.INDEX dtypes.f32 [152, 14]
154 Ops.PARAM dtypes.u8 [1] ParamArg(slot=2, dtype=dtypes.u8)
155 Ops.INDEX dtypes.u8 [154, 46]
156 Ops.LOAD dtypes.u8 [155]
157 Ops.AND dtypes.u8 [156, 49]
158 Ops.CAST dtypes.u32 [157] dtypes.u32
159 Ops.MUL dtypes.u32 [158, 52]
160 Ops.AND dtypes.u8 [156, 54]
161 Ops.CAST dtypes.u32 [160] dtypes.u32
162 Ops.DIV dtypes.u32 [161, 7]
163 Ops.CMPEQ dtypes.bool [162, 58]
164 Ops.MUL dtypes.u32 [161, 60]
165 Ops.OR dtypes.u32 [164, 62]
166 Ops.CMPEQ dtypes.bool [162, 13]
167 Ops.CMPEQ dtypes.bool [161, 13]
168 Ops.CAST dtypes.f32 [161] dtypes.f32
169 Ops.BITCAST dtypes.u32 [168] dtypes.u32
170 Ops.ADD dtypes.u32 [169, 73]
171 Ops.WHERE dtypes.u32 [167, 13, 170]
172 Ops.ADD dtypes.u32 [164, 76]
173 Ops.WHERE dtypes.u32 [166, 171, 172]
174 Ops.WHERE dtypes.u32 [163, 165, 173]
175 Ops.OR dtypes.u32 [159, 174]
176 Ops.BITCAST dtypes.f32 [175] dtypes.f32
177 Ops.INDEX dtypes.u8 [154, 82]
178 Ops.LOAD dtypes.u8 [177]
179 Ops.AND dtypes.u8 [178, 49]
180 Ops.CAST dtypes.u32 [179] dtypes.u32
181 Ops.MUL dtypes.u32 [180, 52]
182 Ops.AND dtypes.u8 [178, 54]
183 Ops.CAST dtypes.u32 [182] dtypes.u32
184 Ops.DIV dtypes.u32 [183, 7]
185 Ops.CMPEQ dtypes.bool [184, 58]
186 Ops.MUL dtypes.u32 [183, 60]
187 Ops.OR dtypes.u32 [186, 62]
188 Ops.CMPEQ dtypes.bool [184, 13]
189 Ops.CMPEQ dtypes.bool [183, 13]
190 Ops.CAST dtypes.f32 [183] dtypes.f32
191 Ops.BITCAST dtypes.u32 [190] dtypes.u32
192 Ops.ADD dtypes.u32 [191, 73]
193 Ops.WHERE dtypes.u32 [189, 13, 192]
194 Ops.ADD dtypes.u32 [186, 76]
195 Ops.WHERE dtypes.u32 [188, 193, 194]
196 Ops.WHERE dtypes.u32 [185, 187, 195]
197 Ops.OR dtypes.u32 [181, 196]
198 Ops.BITCAST dtypes.f32 [197] dtypes.f32
199 Ops.INDEX dtypes.u8 [154, 105]
200 Ops.LOAD dtypes.u8 [199]
201 Ops.AND dtypes.u8 [200, 49]
202 Ops.CAST dtypes.u32 [201] dtypes.u32
203 Ops.MUL dtypes.u32 [202, 52]
204 Ops.AND dtypes.u8 [200, 54]
205 Ops.CAST dtypes.u32 [204] dtypes.u32
206 Ops.DIV dtypes.u32 [205, 7]
207 Ops.CMPEQ dtypes.bool [206, 58]
208 Ops.MUL dtypes.u32 [205, 60]
209 Ops.OR dtypes.u32 [208, 62]
210 Ops.CMPEQ dtypes.bool [206, 13]
211 Ops.CMPEQ dtypes.bool [205, 13]
212 Ops.CAST dtypes.f32 [205] dtypes.f32
213 Ops.BITCAST dtypes.u32 [212] dtypes.u32
214 Ops.ADD dtypes.u32 [213, 73]
215 Ops.WHERE dtypes.u32 [211, 13, 214]
216 Ops.ADD dtypes.u32 [208, 76]
217 Ops.WHERE dtypes.u32 [210, 215, 216]
218 Ops.WHERE dtypes.u32 [207, 209, 217]
219 Ops.OR dtypes.u32 [203, 218]
220 Ops.BITCAST dtypes.f32 [219] dtypes.f32
221 Ops.INDEX dtypes.u8 [154, 128]
222 Ops.LOAD dtypes.u8 [221]
223 Ops.AND dtypes.u8 [222, 49]
224 Ops.CAST dtypes.u32 [223] dtypes.u32
225 Ops.MUL dtypes.u32 [224, 52]
226 Ops.AND dtypes.u8 [222, 54]
227 Ops.CAST dtypes.u32 [226] dtypes.u32
228 Ops.DIV dtypes.u32 [227, 7]
229 Ops.CMPEQ dtypes.bool [228, 58]
230 Ops.MUL dtypes.u32 [227, 60]
231 Ops.OR dtypes.u32 [230, 62]
232 Ops.CMPEQ dtypes.bool [228, 13]
233 Ops.CMPEQ dtypes.bool [227, 13]
234 Ops.CAST dtypes.f32 [227] dtypes.f32
235 Ops.BITCAST dtypes.u32 [234] dtypes.u32
236 Ops.ADD dtypes.u32 [235, 73]
237 Ops.WHERE dtypes.u32 [233, 13, 236]
238 Ops.ADD dtypes.u32 [230, 76]
239 Ops.WHERE dtypes.u32 [232, 237, 238]
240 Ops.WHERE dtypes.u32 [229, 231, 239]
241 Ops.OR dtypes.u32 [225, 240]
242 Ops.BITCAST dtypes.f32 [241] dtypes.f32
243 Ops.STACK dtypes.f32 [176, 198, 220, 242]
244 Ops.CAST dtypes.f32 [243] dtypes.f32
245 Ops.INDEX dtypes.f32 [244, 14]
246 Ops.CMPLT dtypes.bool [153, 245]
247 Ops.SHRINK dtypes.u8 [154, 6, 8]
248 Ops.LOAD dtypes.u8 [247]
249 Ops.INDEX dtypes.u8 [248, 14]
250 Ops.SHRINK dtypes.u8 [45, 6, 8]
251 Ops.LOAD dtypes.u8 [250]
252 Ops.INDEX dtypes.u8 [251, 14]
253 Ops.WHERE dtypes.u8 [246, 249, 252]
254 Ops.INDEX dtypes.f32 [152, 23]
255 Ops.INDEX dtypes.f32 [244, 23]
256 Ops.CMPLT dtypes.bool [254, 255]
257 Ops.INDEX dtypes.u8 [248, 23]
258 Ops.INDEX dtypes.u8 [251, 23]
259 Ops.WHERE dtypes.u8 [256, 257, 258]
260 Ops.INDEX dtypes.f32 [152, 4]
261 Ops.INDEX dtypes.f32 [244, 4]
262 Ops.CMPLT dtypes.bool [260, 261]
263 Ops.INDEX dtypes.u8 [248, 4]
264 Ops.INDEX dtypes.u8 [251, 4]
265 Ops.WHERE dtypes.u8 [262, 263, 264]
266 Ops.INDEX dtypes.f32 [152, 33]
267 Ops.INDEX dtypes.f32 [244, 33]
268 Ops.CMPLT dtypes.bool [266, 267]
269 Ops.INDEX dtypes.u8 [248, 33]
270 Ops.INDEX dtypes.u8 [251, 33]
271 Ops.WHERE dtypes.u8 [268, 269, 270]
272 Ops.STACK dtypes.u8 [253, 259, 265, 271]
273 Ops.STORE dtypes.void [44, 272]
274 Ops.AFTER dtypes.void [0, 273]
275 Ops.END dtypes.void [274, 5]
276 Ops.SINK dtypes.void [275] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
277 Ops.SINK dtypes.void [42, 276]

