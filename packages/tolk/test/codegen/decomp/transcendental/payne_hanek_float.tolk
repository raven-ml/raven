# tolk D31 D126 D413 golden=aeb3b6719592211a801ddb66e8b10b9e
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.BITCAST dtypes.u32 [1] dtypes.u32
3 Ops.CONST dtypes.weakint [] 2155872255
4 Ops.AND dtypes.u32 [2, 3]
5 Ops.CONST dtypes.weakint [] 1056964608
6 Ops.OR dtypes.u32 [4, 5]
7 Ops.BITCAST dtypes.f32 [6] dtypes.f32
8 Ops.CONST dtypes.weakfloat [] 16777216.0
9 Ops.MUL dtypes.f32 [7, 8]
10 Ops.CAST dtypes.u64 [9] dtypes.u64
11 Ops.CONST dtypes.weakint [] 4294967295
12 Ops.AND dtypes.u64 [10, 11]
13 Ops.CONST dtypes.weakint [] 8388608
14 Ops.DIV dtypes.u32 [2, 13]
15 Ops.CONST dtypes.weakint [] 255
16 Ops.AND dtypes.u32 [14, 15]
17 Ops.CONST dtypes.weakint [] 127
18 Ops.CONST dtypes.weakint [] -1
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.ADD dtypes.u32 [16, 19]
21 Ops.CONST dtypes.weakint [] 1
22 Ops.ADD dtypes.u32 [20, 21]
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.CONST dtypes.weakint [] 104
25 Ops.ADD dtypes.i32 [23, 24]
26 Ops.CONST dtypes.weakint [] 0
27 Ops.MAX dtypes.i32 [25, 26]
28 Ops.CONST dtypes.weakint [] 32
29 Ops.DIV dtypes.i32 [27, 28]
30 Ops.AND dtypes.i32 [29, 21]
31 Ops.CMPEQ dtypes.bool [30, 26]
32 Ops.CONST dtypes.bool [] False
33 Ops.CMPEQ dtypes.bool [31, 32]
34 Ops.CONST dtypes.weakint [] 2
35 Ops.AND dtypes.i32 [29, 34]
36 Ops.CMPEQ dtypes.bool [35, 26]
37 Ops.CMPEQ dtypes.bool [36, 32]
38 Ops.CONST dtypes.weakint [] 4
39 Ops.AND dtypes.i32 [29, 38]
40 Ops.CMPEQ dtypes.bool [39, 26]
41 Ops.CMPEQ dtypes.bool [40, 32]
42 Ops.CONST dtypes.weakint [] 2102212464
43 Ops.CAST dtypes.u64 [42] dtypes.u64
44 Ops.CAST dtypes.u64 [26] dtypes.u64
45 Ops.WHERE dtypes.u64 [41, 43, 44]
46 Ops.CONST dtypes.weakint [] 2475754826
47 Ops.CAST dtypes.u64 [46] dtypes.u64
48 Ops.WHERE dtypes.u64 [41, 47, 44]
49 Ops.WHERE dtypes.u64 [37, 45, 48]
50 Ops.CONST dtypes.weakint [] 2131351028
51 Ops.CAST dtypes.u64 [50] dtypes.u64
52 Ops.WHERE dtypes.u64 [41, 51, 44]
53 Ops.CONST dtypes.weakint [] 683565275
54 Ops.CAST dtypes.u64 [53] dtypes.u64
55 Ops.WHERE dtypes.u64 [41, 54, 44]
56 Ops.WHERE dtypes.u64 [37, 52, 55]
57 Ops.WHERE dtypes.u64 [33, 49, 56]
58 Ops.CONST dtypes.weakint [] 31
59 Ops.AND dtypes.i32 [27, 58]
60 Ops.CAST dtypes.u64 [59] dtypes.u64
61 Ops.SHL dtypes.u64 [57, 60]
62 Ops.CONST dtypes.weakint [] 920167782
63 Ops.CAST dtypes.u64 [62] dtypes.u64
64 Ops.WHERE dtypes.u64 [41, 63, 54]
65 Ops.WHERE dtypes.u64 [37, 64, 52]
66 Ops.WHERE dtypes.u64 [33, 65, 49]
67 Ops.CONST dtypes.weakint [] 18446744073709551615
68 Ops.MUL dtypes.u64 [60, 67]
69 Ops.ADD dtypes.u64 [28, 68]
70 Ops.SHR dtypes.u64 [66, 69]
71 Ops.OR dtypes.u64 [61, 70]
72 Ops.AND dtypes.u64 [71, 11]
73 Ops.MUL dtypes.u64 [12, 72]
74 Ops.AND dtypes.u64 [73, 11]
75 Ops.SHL dtypes.u64 [66, 60]
76 Ops.CONST dtypes.weakint [] 1326507024
77 Ops.CAST dtypes.u64 [76] dtypes.u64
78 Ops.WHERE dtypes.u64 [41, 77, 47]
79 Ops.WHERE dtypes.u64 [37, 78, 45]
80 Ops.WHERE dtypes.u64 [33, 79, 65]
81 Ops.SHR dtypes.u64 [80, 69]
82 Ops.OR dtypes.u64 [75, 81]
83 Ops.AND dtypes.u64 [82, 11]
84 Ops.MUL dtypes.u64 [12, 83]
85 Ops.CONST dtypes.weakint [] 4294967296
86 Ops.DIV dtypes.u64 [84, 85]
87 Ops.ADD dtypes.u64 [74, 86]
88 Ops.AND dtypes.u64 [84, 11]
89 Ops.SHL dtypes.u64 [80, 60]
90 Ops.CONST dtypes.weakint [] 2140428522
91 Ops.CAST dtypes.u64 [90] dtypes.u64
92 Ops.WHERE dtypes.u64 [41, 91, 51]
93 Ops.WHERE dtypes.u64 [37, 92, 64]
94 Ops.WHERE dtypes.u64 [33, 93, 79]
95 Ops.SHR dtypes.u64 [94, 69]
96 Ops.OR dtypes.u64 [89, 95]
97 Ops.AND dtypes.u64 [96, 11]
98 Ops.MUL dtypes.u64 [12, 97]
99 Ops.DIV dtypes.u64 [98, 85]
100 Ops.ADD dtypes.u64 [88, 99]
101 Ops.AND dtypes.u64 [98, 11]
102 Ops.SHL dtypes.u64 [94, 60]
103 Ops.CONST dtypes.weakint [] 4155437400
104 Ops.CAST dtypes.u64 [103] dtypes.u64
105 Ops.WHERE dtypes.u64 [41, 104, 43]
106 Ops.WHERE dtypes.u64 [37, 105, 78]
107 Ops.WHERE dtypes.u64 [33, 106, 93]
108 Ops.SHR dtypes.u64 [107, 69]
109 Ops.OR dtypes.u64 [102, 108]
110 Ops.AND dtypes.u64 [109, 11]
111 Ops.MUL dtypes.u64 [12, 110]
112 Ops.DIV dtypes.u64 [111, 85]
113 Ops.ADD dtypes.u64 [101, 112]
114 Ops.AND dtypes.u64 [111, 11]
115 Ops.SHL dtypes.u64 [107, 60]
116 Ops.CONST dtypes.weakint [] 1841896334
117 Ops.CAST dtypes.u64 [116] dtypes.u64
118 Ops.WHERE dtypes.u64 [41, 117, 63]
119 Ops.WHERE dtypes.u64 [37, 118, 92]
120 Ops.WHERE dtypes.u64 [33, 119, 106]
121 Ops.SHR dtypes.u64 [120, 69]
122 Ops.OR dtypes.u64 [115, 121]
123 Ops.AND dtypes.u64 [122, 11]
124 Ops.MUL dtypes.u64 [12, 123]
125 Ops.DIV dtypes.u64 [124, 85]
126 Ops.ADD dtypes.u64 [114, 125]
127 Ops.AND dtypes.u64 [124, 11]
128 Ops.SHL dtypes.u64 [120, 60]
129 Ops.CONST dtypes.weakint [] 2425582776
130 Ops.CAST dtypes.u64 [129] dtypes.u64
131 Ops.WHERE dtypes.u64 [41, 130, 77]
132 Ops.WHERE dtypes.u64 [37, 131, 105]
133 Ops.WHERE dtypes.u64 [33, 132, 119]
134 Ops.SHR dtypes.u64 [133, 69]
135 Ops.OR dtypes.u64 [128, 134]
136 Ops.AND dtypes.u64 [135, 11]
137 Ops.MUL dtypes.u64 [12, 136]
138 Ops.DIV dtypes.u64 [137, 85]
139 Ops.ADD dtypes.u64 [127, 138]
140 Ops.AND dtypes.u64 [137, 11]
141 Ops.SHL dtypes.u64 [133, 60]
142 Ops.CONST dtypes.weakint [] 26364858
143 Ops.CAST dtypes.u64 [142] dtypes.u64
144 Ops.WHERE dtypes.u64 [41, 143, 91]
145 Ops.WHERE dtypes.u64 [37, 144, 118]
146 Ops.WHERE dtypes.u64 [33, 145, 132]
147 Ops.SHR dtypes.u64 [146, 69]
148 Ops.OR dtypes.u64 [141, 147]
149 Ops.AND dtypes.u64 [148, 11]
150 Ops.MUL dtypes.u64 [12, 149]
151 Ops.DIV dtypes.u64 [150, 85]
152 Ops.ADD dtypes.u64 [140, 151]
153 Ops.AND dtypes.u64 [150, 11]
154 Ops.DIV dtypes.u64 [153, 85]
155 Ops.ADD dtypes.u64 [152, 154]
156 Ops.DIV dtypes.u64 [155, 85]
157 Ops.ADD dtypes.u64 [139, 156]
158 Ops.DIV dtypes.u64 [157, 85]
159 Ops.ADD dtypes.u64 [126, 158]
160 Ops.DIV dtypes.u64 [159, 85]
161 Ops.ADD dtypes.u64 [113, 160]
162 Ops.DIV dtypes.u64 [161, 85]
163 Ops.ADD dtypes.u64 [100, 162]
164 Ops.DIV dtypes.u64 [163, 85]
165 Ops.ADD dtypes.u64 [87, 164]
166 Ops.AND dtypes.u64 [165, 11]
167 Ops.MUL dtypes.u64 [166, 85]
168 Ops.AND dtypes.u64 [163, 11]
169 Ops.OR dtypes.u64 [167, 168]
170 Ops.CONST dtypes.weakint [] 4611686018427387903
171 Ops.AND dtypes.u64 [169, 170]
172 Ops.CAST dtypes.i64 [171] dtypes.i64
173 Ops.CONST dtypes.weakint [] 2305843009213693952
174 Ops.DIV dtypes.u64 [169, 173]
175 Ops.AND dtypes.u64 [174, 21]
176 Ops.CAST dtypes.i64 [175] dtypes.i64
177 Ops.CONST dtypes.weakint [] 4611686018427387904
178 Ops.MUL dtypes.i64 [176, 177]
179 Ops.MUL dtypes.i64 [178, 18]
180 Ops.ADD dtypes.i64 [172, 179]
181 Ops.CONST dtypes.weakint [] 4398046511104
182 Ops.DIV dtypes.i64 [180, 181]
183 Ops.CAST dtypes.f32 [182] dtypes.f32
184 Ops.CONST dtypes.weakfloat [] 4398046511104.0
185 Ops.MUL dtypes.f32 [183, 184]
186 Ops.CONST dtypes.weakint [] 2097152
187 Ops.DIV dtypes.i64 [180, 186]
188 Ops.CONST dtypes.weakint [] 2097151
189 Ops.AND dtypes.i64 [187, 188]
190 Ops.CAST dtypes.f32 [189] dtypes.f32
191 Ops.CONST dtypes.weakfloat [] 2097152.0
192 Ops.MUL dtypes.f32 [190, 191]
193 Ops.ADD dtypes.f32 [185, 192]
194 Ops.AND dtypes.i64 [180, 188]
195 Ops.CAST dtypes.f32 [194] dtypes.f32
196 Ops.CONST dtypes.weakfloat [] 1.0
197 Ops.MUL dtypes.f32 [195, 196]
198 Ops.ADD dtypes.f32 [193, 197]
199 Ops.AND dtypes.u64 [161, 11]
200 Ops.MUL dtypes.u64 [199, 85]
201 Ops.AND dtypes.u64 [159, 11]
202 Ops.OR dtypes.u64 [200, 201]
203 Ops.CONST dtypes.weakint [] 8796093022208
204 Ops.DIV dtypes.u64 [202, 203]
205 Ops.CAST dtypes.f32 [204] dtypes.f32
206 Ops.CONST dtypes.weakfloat [] 1.1368683772161603e-13
207 Ops.MUL dtypes.f32 [205, 206]
208 Ops.ADD dtypes.f32 [198, 207]
209 Ops.CONST dtypes.weakfloat [] 3.4061216748705226e-19
210 Ops.MUL dtypes.f32 [208, 209]
211 Ops.CONST dtypes.weakfloat [] 4097.0
212 Ops.MUL dtypes.f32 [208, 211]
213 Ops.CONST dtypes.weakfloat [] -1.0
214 Ops.MUL dtypes.f32 [208, 213]
215 Ops.ADD dtypes.f32 [212, 214]
216 Ops.MUL dtypes.f32 [215, 213]
217 Ops.ADD dtypes.f32 [212, 216]
218 Ops.CONST dtypes.weakfloat [] 3.406131239146355e-19
219 Ops.MUL dtypes.f32 [217, 218]
220 Ops.MUL dtypes.f32 [210, 213]
221 Ops.ADD dtypes.f32 [219, 220]
222 Ops.CONST dtypes.weakfloat [] -9.564275832644382e-25
223 Ops.MUL dtypes.f32 [217, 222]
224 Ops.ADD dtypes.f32 [221, 223]
225 Ops.MUL dtypes.f32 [217, 213]
226 Ops.ADD dtypes.f32 [208, 225]
227 Ops.MUL dtypes.f32 [226, 218]
228 Ops.ADD dtypes.f32 [224, 227]
229 Ops.MUL dtypes.f32 [226, 222]
230 Ops.ADD dtypes.f32 [228, 229]
231 Ops.CONST dtypes.weakfloat [] -9.47839680047608e-27
232 Ops.MUL dtypes.f32 [208, 231]
233 Ops.ADD dtypes.f32 [230, 232]
234 Ops.CONST dtypes.weakfloat [] 0.0
235 Ops.CAST dtypes.f32 [234] dtypes.f32
236 Ops.MUL dtypes.f32 [185, 213]
237 Ops.ADD dtypes.f32 [193, 236]
238 Ops.MUL dtypes.f32 [237, 213]
239 Ops.ADD dtypes.f32 [192, 238]
240 Ops.ADD dtypes.f32 [235, 239]
241 Ops.MUL dtypes.f32 [193, 213]
242 Ops.ADD dtypes.f32 [198, 241]
243 Ops.MUL dtypes.f32 [242, 213]
244 Ops.ADD dtypes.f32 [197, 243]
245 Ops.ADD dtypes.f32 [240, 244]
246 Ops.MUL dtypes.f32 [198, 213]
247 Ops.ADD dtypes.f32 [208, 246]
248 Ops.MUL dtypes.f32 [247, 213]
249 Ops.ADD dtypes.f32 [207, 248]
250 Ops.ADD dtypes.f32 [245, 249]
251 Ops.MUL dtypes.f32 [250, 209]
252 Ops.ADD dtypes.f32 [233, 251]
253 Ops.ADD dtypes.f32 [210, 252]
254 Ops.DIV dtypes.u64 [169, 177]
255 Ops.ADD dtypes.u64 [254, 175]
256 Ops.CAST dtypes.i32 [255] dtypes.i32
257 Ops.SINK dtypes.void [253, 256]

