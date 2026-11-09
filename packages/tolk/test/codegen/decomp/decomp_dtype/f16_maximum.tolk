# tolk D27 golden=ca769eccc1aa096525bd4a39cfd53c8a
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 8
2 Ops.PARAM dtypes.f16 [1] ParamArg(slot=0, dtype=dtypes.f16)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.SHL dtypes.i32 [5, 3]
7 Ops.CONST dtypes.weakint [] 4
8 Ops.CAST dtypes.i32 [7] dtypes.i32
9 Ops.SHRINK dtypes.f16 [2, 6, 8]
10 Ops.PARAM dtypes.f16 [1] ParamArg(slot=1, dtype=dtypes.f16)
11 Ops.SHRINK dtypes.f16 [10, 6, 8]
12 Ops.LOAD dtypes.f16 [11]
13 Ops.CONST dtypes.weakint [] 0
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.INDEX dtypes.f16 [12, 14]
16 Ops.PARAM dtypes.f16 [1] ParamArg(slot=2, dtype=dtypes.f16)
17 Ops.SHRINK dtypes.f16 [16, 6, 8]
18 Ops.LOAD dtypes.f16 [17]
19 Ops.INDEX dtypes.f16 [18, 14]
20 Ops.CMPLT dtypes.bool [15, 19]
21 Ops.WHERE dtypes.f16 [20, 19, 15]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.f16 [12, 23]
25 Ops.INDEX dtypes.f16 [18, 23]
26 Ops.CMPLT dtypes.bool [24, 25]
27 Ops.WHERE dtypes.f16 [26, 25, 24]
28 Ops.INDEX dtypes.f16 [12, 4]
29 Ops.INDEX dtypes.f16 [18, 4]
30 Ops.CMPLT dtypes.bool [28, 29]
31 Ops.WHERE dtypes.f16 [30, 29, 28]
32 Ops.CONST dtypes.weakint [] 3
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.f16 [12, 33]
35 Ops.INDEX dtypes.f16 [18, 33]
36 Ops.CMPLT dtypes.bool [34, 35]
37 Ops.WHERE dtypes.f16 [36, 35, 34]
38 Ops.STACK dtypes.f16 [21, 27, 31, 37]
39 Ops.STORE dtypes.void [9, 38]
40 Ops.AFTER dtypes.void [0, 39]
41 Ops.END dtypes.void [40, 5]
42 Ops.SINK dtypes.void [41] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
43 Ops.PARAM dtypes.u16 [1] ParamArg(slot=0, dtype=dtypes.u16)
44 Ops.SHRINK dtypes.u16 [43, 6, 8]
45 Ops.PARAM dtypes.u16 [1] ParamArg(slot=1, dtype=dtypes.u16)
46 Ops.ADD dtypes.i32 [6, 13]
47 Ops.INDEX dtypes.u16 [45, 46]
48 Ops.LOAD dtypes.u16 [47]
49 Ops.CONST dtypes.weakint [] 32768
50 Ops.AND dtypes.u16 [48, 49]
51 Ops.CAST dtypes.u32 [50] dtypes.u32
52 Ops.CONST dtypes.weakint [] 65536
53 Ops.MUL dtypes.u32 [51, 52]
54 Ops.CONST dtypes.weakint [] 32767
55 Ops.AND dtypes.u16 [48, 54]
56 Ops.CAST dtypes.u32 [55] dtypes.u32
57 Ops.CONST dtypes.weakint [] 1024
58 Ops.DIV dtypes.u32 [56, 57]
59 Ops.CONST dtypes.weakint [] 31
60 Ops.CMPEQ dtypes.bool [58, 59]
61 Ops.CONST dtypes.weakint [] 8192
62 Ops.MUL dtypes.u32 [56, 61]
63 Ops.CONST dtypes.weakint [] 2139095040
64 Ops.OR dtypes.u32 [62, 63]
65 Ops.CMPEQ dtypes.bool [58, 13]
66 Ops.CMPEQ dtypes.bool [56, 13]
67 Ops.CAST dtypes.f32 [56] dtypes.f32
68 Ops.BITCAST dtypes.u32 [67] dtypes.u32
69 Ops.CONST dtypes.weakint [] 201326592
70 Ops.CAST dtypes.i64 [69] dtypes.i64
71 Ops.CONST dtypes.weakint [] -1
72 Ops.CAST dtypes.i64 [71] dtypes.i64
73 Ops.MUL dtypes.i64 [70, 72]
74 Ops.CAST dtypes.u32 [73] dtypes.u32
75 Ops.ADD dtypes.u32 [68, 74]
76 Ops.WHERE dtypes.u32 [66, 13, 75]
77 Ops.CONST dtypes.weakint [] 939524096
78 Ops.ADD dtypes.u32 [62, 77]
79 Ops.WHERE dtypes.u32 [65, 76, 78]
80 Ops.WHERE dtypes.u32 [60, 64, 79]
81 Ops.OR dtypes.u32 [53, 80]
82 Ops.BITCAST dtypes.f32 [81] dtypes.f32
83 Ops.ADD dtypes.i32 [6, 22]
84 Ops.INDEX dtypes.u16 [45, 83]
85 Ops.LOAD dtypes.u16 [84]
86 Ops.AND dtypes.u16 [85, 49]
87 Ops.CAST dtypes.u32 [86] dtypes.u32
88 Ops.MUL dtypes.u32 [87, 52]
89 Ops.AND dtypes.u16 [85, 54]
90 Ops.CAST dtypes.u32 [89] dtypes.u32
91 Ops.DIV dtypes.u32 [90, 57]
92 Ops.CMPEQ dtypes.bool [91, 59]
93 Ops.MUL dtypes.u32 [90, 61]
94 Ops.OR dtypes.u32 [93, 63]
95 Ops.CMPEQ dtypes.bool [91, 13]
96 Ops.CMPEQ dtypes.bool [90, 13]
97 Ops.CAST dtypes.f32 [90] dtypes.f32
98 Ops.BITCAST dtypes.u32 [97] dtypes.u32
99 Ops.ADD dtypes.u32 [98, 74]
100 Ops.WHERE dtypes.u32 [96, 13, 99]
101 Ops.ADD dtypes.u32 [93, 77]
102 Ops.WHERE dtypes.u32 [95, 100, 101]
103 Ops.WHERE dtypes.u32 [92, 94, 102]
104 Ops.OR dtypes.u32 [88, 103]
105 Ops.BITCAST dtypes.f32 [104] dtypes.f32
106 Ops.ADD dtypes.i32 [6, 3]
107 Ops.INDEX dtypes.u16 [45, 106]
108 Ops.LOAD dtypes.u16 [107]
109 Ops.AND dtypes.u16 [108, 49]
110 Ops.CAST dtypes.u32 [109] dtypes.u32
111 Ops.MUL dtypes.u32 [110, 52]
112 Ops.AND dtypes.u16 [108, 54]
113 Ops.CAST dtypes.u32 [112] dtypes.u32
114 Ops.DIV dtypes.u32 [113, 57]
115 Ops.CMPEQ dtypes.bool [114, 59]
116 Ops.MUL dtypes.u32 [113, 61]
117 Ops.OR dtypes.u32 [116, 63]
118 Ops.CMPEQ dtypes.bool [114, 13]
119 Ops.CMPEQ dtypes.bool [113, 13]
120 Ops.CAST dtypes.f32 [113] dtypes.f32
121 Ops.BITCAST dtypes.u32 [120] dtypes.u32
122 Ops.ADD dtypes.u32 [121, 74]
123 Ops.WHERE dtypes.u32 [119, 13, 122]
124 Ops.ADD dtypes.u32 [116, 77]
125 Ops.WHERE dtypes.u32 [118, 123, 124]
126 Ops.WHERE dtypes.u32 [115, 117, 125]
127 Ops.OR dtypes.u32 [111, 126]
128 Ops.BITCAST dtypes.f32 [127] dtypes.f32
129 Ops.ADD dtypes.i32 [6, 32]
130 Ops.INDEX dtypes.u16 [45, 129]
131 Ops.LOAD dtypes.u16 [130]
132 Ops.AND dtypes.u16 [131, 49]
133 Ops.CAST dtypes.u32 [132] dtypes.u32
134 Ops.MUL dtypes.u32 [133, 52]
135 Ops.AND dtypes.u16 [131, 54]
136 Ops.CAST dtypes.u32 [135] dtypes.u32
137 Ops.DIV dtypes.u32 [136, 57]
138 Ops.CMPEQ dtypes.bool [137, 59]
139 Ops.MUL dtypes.u32 [136, 61]
140 Ops.OR dtypes.u32 [139, 63]
141 Ops.CMPEQ dtypes.bool [137, 13]
142 Ops.CMPEQ dtypes.bool [136, 13]
143 Ops.CAST dtypes.f32 [136] dtypes.f32
144 Ops.BITCAST dtypes.u32 [143] dtypes.u32
145 Ops.ADD dtypes.u32 [144, 74]
146 Ops.WHERE dtypes.u32 [142, 13, 145]
147 Ops.ADD dtypes.u32 [139, 77]
148 Ops.WHERE dtypes.u32 [141, 146, 147]
149 Ops.WHERE dtypes.u32 [138, 140, 148]
150 Ops.OR dtypes.u32 [134, 149]
151 Ops.BITCAST dtypes.f32 [150] dtypes.f32
152 Ops.STACK dtypes.f32 [82, 105, 128, 151]
153 Ops.CAST dtypes.f32 [152] dtypes.f32
154 Ops.INDEX dtypes.f32 [153, 14]
155 Ops.PARAM dtypes.u16 [1] ParamArg(slot=2, dtype=dtypes.u16)
156 Ops.INDEX dtypes.u16 [155, 46]
157 Ops.LOAD dtypes.u16 [156]
158 Ops.AND dtypes.u16 [157, 49]
159 Ops.CAST dtypes.u32 [158] dtypes.u32
160 Ops.MUL dtypes.u32 [159, 52]
161 Ops.AND dtypes.u16 [157, 54]
162 Ops.CAST dtypes.u32 [161] dtypes.u32
163 Ops.DIV dtypes.u32 [162, 57]
164 Ops.CMPEQ dtypes.bool [163, 59]
165 Ops.MUL dtypes.u32 [162, 61]
166 Ops.OR dtypes.u32 [165, 63]
167 Ops.CMPEQ dtypes.bool [163, 13]
168 Ops.CMPEQ dtypes.bool [162, 13]
169 Ops.CAST dtypes.f32 [162] dtypes.f32
170 Ops.BITCAST dtypes.u32 [169] dtypes.u32
171 Ops.ADD dtypes.u32 [170, 74]
172 Ops.WHERE dtypes.u32 [168, 13, 171]
173 Ops.ADD dtypes.u32 [165, 77]
174 Ops.WHERE dtypes.u32 [167, 172, 173]
175 Ops.WHERE dtypes.u32 [164, 166, 174]
176 Ops.OR dtypes.u32 [160, 175]
177 Ops.BITCAST dtypes.f32 [176] dtypes.f32
178 Ops.INDEX dtypes.u16 [155, 83]
179 Ops.LOAD dtypes.u16 [178]
180 Ops.AND dtypes.u16 [179, 49]
181 Ops.CAST dtypes.u32 [180] dtypes.u32
182 Ops.MUL dtypes.u32 [181, 52]
183 Ops.AND dtypes.u16 [179, 54]
184 Ops.CAST dtypes.u32 [183] dtypes.u32
185 Ops.DIV dtypes.u32 [184, 57]
186 Ops.CMPEQ dtypes.bool [185, 59]
187 Ops.MUL dtypes.u32 [184, 61]
188 Ops.OR dtypes.u32 [187, 63]
189 Ops.CMPEQ dtypes.bool [185, 13]
190 Ops.CMPEQ dtypes.bool [184, 13]
191 Ops.CAST dtypes.f32 [184] dtypes.f32
192 Ops.BITCAST dtypes.u32 [191] dtypes.u32
193 Ops.ADD dtypes.u32 [192, 74]
194 Ops.WHERE dtypes.u32 [190, 13, 193]
195 Ops.ADD dtypes.u32 [187, 77]
196 Ops.WHERE dtypes.u32 [189, 194, 195]
197 Ops.WHERE dtypes.u32 [186, 188, 196]
198 Ops.OR dtypes.u32 [182, 197]
199 Ops.BITCAST dtypes.f32 [198] dtypes.f32
200 Ops.INDEX dtypes.u16 [155, 106]
201 Ops.LOAD dtypes.u16 [200]
202 Ops.AND dtypes.u16 [201, 49]
203 Ops.CAST dtypes.u32 [202] dtypes.u32
204 Ops.MUL dtypes.u32 [203, 52]
205 Ops.AND dtypes.u16 [201, 54]
206 Ops.CAST dtypes.u32 [205] dtypes.u32
207 Ops.DIV dtypes.u32 [206, 57]
208 Ops.CMPEQ dtypes.bool [207, 59]
209 Ops.MUL dtypes.u32 [206, 61]
210 Ops.OR dtypes.u32 [209, 63]
211 Ops.CMPEQ dtypes.bool [207, 13]
212 Ops.CMPEQ dtypes.bool [206, 13]
213 Ops.CAST dtypes.f32 [206] dtypes.f32
214 Ops.BITCAST dtypes.u32 [213] dtypes.u32
215 Ops.ADD dtypes.u32 [214, 74]
216 Ops.WHERE dtypes.u32 [212, 13, 215]
217 Ops.ADD dtypes.u32 [209, 77]
218 Ops.WHERE dtypes.u32 [211, 216, 217]
219 Ops.WHERE dtypes.u32 [208, 210, 218]
220 Ops.OR dtypes.u32 [204, 219]
221 Ops.BITCAST dtypes.f32 [220] dtypes.f32
222 Ops.INDEX dtypes.u16 [155, 129]
223 Ops.LOAD dtypes.u16 [222]
224 Ops.AND dtypes.u16 [223, 49]
225 Ops.CAST dtypes.u32 [224] dtypes.u32
226 Ops.MUL dtypes.u32 [225, 52]
227 Ops.AND dtypes.u16 [223, 54]
228 Ops.CAST dtypes.u32 [227] dtypes.u32
229 Ops.DIV dtypes.u32 [228, 57]
230 Ops.CMPEQ dtypes.bool [229, 59]
231 Ops.MUL dtypes.u32 [228, 61]
232 Ops.OR dtypes.u32 [231, 63]
233 Ops.CMPEQ dtypes.bool [229, 13]
234 Ops.CMPEQ dtypes.bool [228, 13]
235 Ops.CAST dtypes.f32 [228] dtypes.f32
236 Ops.BITCAST dtypes.u32 [235] dtypes.u32
237 Ops.ADD dtypes.u32 [236, 74]
238 Ops.WHERE dtypes.u32 [234, 13, 237]
239 Ops.ADD dtypes.u32 [231, 77]
240 Ops.WHERE dtypes.u32 [233, 238, 239]
241 Ops.WHERE dtypes.u32 [230, 232, 240]
242 Ops.OR dtypes.u32 [226, 241]
243 Ops.BITCAST dtypes.f32 [242] dtypes.f32
244 Ops.STACK dtypes.f32 [177, 199, 221, 243]
245 Ops.CAST dtypes.f32 [244] dtypes.f32
246 Ops.INDEX dtypes.f32 [245, 14]
247 Ops.CMPLT dtypes.bool [154, 246]
248 Ops.SHRINK dtypes.u16 [155, 6, 8]
249 Ops.LOAD dtypes.u16 [248]
250 Ops.INDEX dtypes.u16 [249, 14]
251 Ops.SHRINK dtypes.u16 [45, 6, 8]
252 Ops.LOAD dtypes.u16 [251]
253 Ops.INDEX dtypes.u16 [252, 14]
254 Ops.WHERE dtypes.u16 [247, 250, 253]
255 Ops.INDEX dtypes.f32 [153, 23]
256 Ops.INDEX dtypes.f32 [245, 23]
257 Ops.CMPLT dtypes.bool [255, 256]
258 Ops.INDEX dtypes.u16 [249, 23]
259 Ops.INDEX dtypes.u16 [252, 23]
260 Ops.WHERE dtypes.u16 [257, 258, 259]
261 Ops.INDEX dtypes.f32 [153, 4]
262 Ops.INDEX dtypes.f32 [245, 4]
263 Ops.CMPLT dtypes.bool [261, 262]
264 Ops.INDEX dtypes.u16 [249, 4]
265 Ops.INDEX dtypes.u16 [252, 4]
266 Ops.WHERE dtypes.u16 [263, 264, 265]
267 Ops.INDEX dtypes.f32 [153, 33]
268 Ops.INDEX dtypes.f32 [245, 33]
269 Ops.CMPLT dtypes.bool [267, 268]
270 Ops.INDEX dtypes.u16 [249, 33]
271 Ops.INDEX dtypes.u16 [252, 33]
272 Ops.WHERE dtypes.u16 [269, 270, 271]
273 Ops.STACK dtypes.u16 [254, 260, 266, 272]
274 Ops.STORE dtypes.void [44, 273]
275 Ops.AFTER dtypes.void [0, 274]
276 Ops.END dtypes.void [275, 5]
277 Ops.SINK dtypes.void [276] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
278 Ops.SINK dtypes.void [42, 277]

