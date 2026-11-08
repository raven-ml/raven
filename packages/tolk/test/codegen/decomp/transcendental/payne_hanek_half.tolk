# tolk D31 D126 D413 golden=befaa0bec579157bec7c0f34112f9eb9
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
2 Ops.BITCAST dtypes.u16 [1] dtypes.u16
3 Ops.CONST dtypes.weakint [] 33791
4 Ops.AND dtypes.u16 [2, 3]
5 Ops.CONST dtypes.weakint [] 14336
6 Ops.OR dtypes.u16 [4, 5]
7 Ops.BITCAST dtypes.f16 [6] dtypes.f16
8 Ops.CAST dtypes.f32 [7] dtypes.f32
9 Ops.CONST dtypes.weakfloat [] 16777216.0
10 Ops.MUL dtypes.f32 [8, 9]
11 Ops.CAST dtypes.u64 [10] dtypes.u64
12 Ops.CONST dtypes.weakint [] 4294967295
13 Ops.AND dtypes.u64 [11, 12]
14 Ops.CONST dtypes.weakint [] 1024
15 Ops.DIV dtypes.u16 [2, 14]
16 Ops.CONST dtypes.weakint [] 31
17 Ops.AND dtypes.u16 [15, 16]
18 Ops.CONST dtypes.weakint [] 15
19 Ops.CONST dtypes.weakint [] -1
20 Ops.MUL dtypes.weakint [18, 19]
21 Ops.ADD dtypes.u16 [17, 20]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.ADD dtypes.u16 [21, 22]
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.CONST dtypes.weakint [] 104
26 Ops.ADD dtypes.i32 [24, 25]
27 Ops.CONST dtypes.weakint [] 0
28 Ops.MAX dtypes.i32 [26, 27]
29 Ops.CONST dtypes.weakint [] 32
30 Ops.DIV dtypes.i32 [28, 29]
31 Ops.AND dtypes.i32 [30, 22]
32 Ops.CMPEQ dtypes.bool [31, 27]
33 Ops.CONST dtypes.bool [] False
34 Ops.CMPEQ dtypes.bool [32, 33]
35 Ops.CONST dtypes.weakint [] 2
36 Ops.AND dtypes.i32 [30, 35]
37 Ops.CMPEQ dtypes.bool [36, 27]
38 Ops.CMPEQ dtypes.bool [37, 33]
39 Ops.CONST dtypes.weakint [] 4
40 Ops.AND dtypes.i32 [30, 39]
41 Ops.CMPEQ dtypes.bool [40, 27]
42 Ops.CMPEQ dtypes.bool [41, 33]
43 Ops.CONST dtypes.weakint [] 2102212464
44 Ops.CAST dtypes.u64 [43] dtypes.u64
45 Ops.CAST dtypes.u64 [27] dtypes.u64
46 Ops.WHERE dtypes.u64 [42, 44, 45]
47 Ops.CONST dtypes.weakint [] 2475754826
48 Ops.CAST dtypes.u64 [47] dtypes.u64
49 Ops.WHERE dtypes.u64 [42, 48, 45]
50 Ops.WHERE dtypes.u64 [38, 46, 49]
51 Ops.CONST dtypes.weakint [] 2131351028
52 Ops.CAST dtypes.u64 [51] dtypes.u64
53 Ops.WHERE dtypes.u64 [42, 52, 45]
54 Ops.CONST dtypes.weakint [] 683565275
55 Ops.CAST dtypes.u64 [54] dtypes.u64
56 Ops.WHERE dtypes.u64 [42, 55, 45]
57 Ops.WHERE dtypes.u64 [38, 53, 56]
58 Ops.WHERE dtypes.u64 [34, 50, 57]
59 Ops.AND dtypes.i32 [28, 16]
60 Ops.CAST dtypes.u64 [59] dtypes.u64
61 Ops.SHL dtypes.u64 [58, 60]
62 Ops.CONST dtypes.weakint [] 920167782
63 Ops.CAST dtypes.u64 [62] dtypes.u64
64 Ops.WHERE dtypes.u64 [42, 63, 55]
65 Ops.WHERE dtypes.u64 [38, 64, 53]
66 Ops.WHERE dtypes.u64 [34, 65, 50]
67 Ops.CONST dtypes.weakint [] 18446744073709551615
68 Ops.MUL dtypes.u64 [60, 67]
69 Ops.ADD dtypes.u64 [29, 68]
70 Ops.SHR dtypes.u64 [66, 69]
71 Ops.OR dtypes.u64 [61, 70]
72 Ops.AND dtypes.u64 [71, 12]
73 Ops.MUL dtypes.u64 [13, 72]
74 Ops.AND dtypes.u64 [73, 12]
75 Ops.SHL dtypes.u64 [66, 60]
76 Ops.CONST dtypes.weakint [] 1326507024
77 Ops.CAST dtypes.u64 [76] dtypes.u64
78 Ops.WHERE dtypes.u64 [42, 77, 48]
79 Ops.WHERE dtypes.u64 [38, 78, 46]
80 Ops.WHERE dtypes.u64 [34, 79, 65]
81 Ops.SHR dtypes.u64 [80, 69]
82 Ops.OR dtypes.u64 [75, 81]
83 Ops.AND dtypes.u64 [82, 12]
84 Ops.MUL dtypes.u64 [13, 83]
85 Ops.CONST dtypes.weakint [] 4294967296
86 Ops.DIV dtypes.u64 [84, 85]
87 Ops.ADD dtypes.u64 [74, 86]
88 Ops.AND dtypes.u64 [84, 12]
89 Ops.SHL dtypes.u64 [80, 60]
90 Ops.CONST dtypes.weakint [] 2140428522
91 Ops.CAST dtypes.u64 [90] dtypes.u64
92 Ops.WHERE dtypes.u64 [42, 91, 52]
93 Ops.WHERE dtypes.u64 [38, 92, 64]
94 Ops.WHERE dtypes.u64 [34, 93, 79]
95 Ops.SHR dtypes.u64 [94, 69]
96 Ops.OR dtypes.u64 [89, 95]
97 Ops.AND dtypes.u64 [96, 12]
98 Ops.MUL dtypes.u64 [13, 97]
99 Ops.DIV dtypes.u64 [98, 85]
100 Ops.ADD dtypes.u64 [88, 99]
101 Ops.AND dtypes.u64 [98, 12]
102 Ops.SHL dtypes.u64 [94, 60]
103 Ops.CONST dtypes.weakint [] 4155437400
104 Ops.CAST dtypes.u64 [103] dtypes.u64
105 Ops.WHERE dtypes.u64 [42, 104, 44]
106 Ops.WHERE dtypes.u64 [38, 105, 78]
107 Ops.WHERE dtypes.u64 [34, 106, 93]
108 Ops.SHR dtypes.u64 [107, 69]
109 Ops.OR dtypes.u64 [102, 108]
110 Ops.AND dtypes.u64 [109, 12]
111 Ops.MUL dtypes.u64 [13, 110]
112 Ops.DIV dtypes.u64 [111, 85]
113 Ops.ADD dtypes.u64 [101, 112]
114 Ops.AND dtypes.u64 [111, 12]
115 Ops.SHL dtypes.u64 [107, 60]
116 Ops.CONST dtypes.weakint [] 1841896334
117 Ops.CAST dtypes.u64 [116] dtypes.u64
118 Ops.WHERE dtypes.u64 [42, 117, 63]
119 Ops.WHERE dtypes.u64 [38, 118, 92]
120 Ops.WHERE dtypes.u64 [34, 119, 106]
121 Ops.SHR dtypes.u64 [120, 69]
122 Ops.OR dtypes.u64 [115, 121]
123 Ops.AND dtypes.u64 [122, 12]
124 Ops.MUL dtypes.u64 [13, 123]
125 Ops.DIV dtypes.u64 [124, 85]
126 Ops.ADD dtypes.u64 [114, 125]
127 Ops.AND dtypes.u64 [124, 12]
128 Ops.SHL dtypes.u64 [120, 60]
129 Ops.CONST dtypes.weakint [] 2425582776
130 Ops.CAST dtypes.u64 [129] dtypes.u64
131 Ops.WHERE dtypes.u64 [42, 130, 77]
132 Ops.WHERE dtypes.u64 [38, 131, 105]
133 Ops.WHERE dtypes.u64 [34, 132, 119]
134 Ops.SHR dtypes.u64 [133, 69]
135 Ops.OR dtypes.u64 [128, 134]
136 Ops.AND dtypes.u64 [135, 12]
137 Ops.MUL dtypes.u64 [13, 136]
138 Ops.DIV dtypes.u64 [137, 85]
139 Ops.ADD dtypes.u64 [127, 138]
140 Ops.AND dtypes.u64 [137, 12]
141 Ops.SHL dtypes.u64 [133, 60]
142 Ops.CONST dtypes.weakint [] 26364858
143 Ops.CAST dtypes.u64 [142] dtypes.u64
144 Ops.WHERE dtypes.u64 [42, 143, 91]
145 Ops.WHERE dtypes.u64 [38, 144, 118]
146 Ops.WHERE dtypes.u64 [34, 145, 132]
147 Ops.SHR dtypes.u64 [146, 69]
148 Ops.OR dtypes.u64 [141, 147]
149 Ops.AND dtypes.u64 [148, 12]
150 Ops.MUL dtypes.u64 [13, 149]
151 Ops.DIV dtypes.u64 [150, 85]
152 Ops.ADD dtypes.u64 [140, 151]
153 Ops.AND dtypes.u64 [150, 12]
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
166 Ops.AND dtypes.u64 [165, 12]
167 Ops.MUL dtypes.u64 [166, 85]
168 Ops.AND dtypes.u64 [163, 12]
169 Ops.OR dtypes.u64 [167, 168]
170 Ops.CONST dtypes.weakint [] 4611686018427387903
171 Ops.AND dtypes.u64 [169, 170]
172 Ops.CAST dtypes.i64 [171] dtypes.i64
173 Ops.CONST dtypes.weakint [] 2305843009213693952
174 Ops.DIV dtypes.u64 [169, 173]
175 Ops.AND dtypes.u64 [174, 22]
176 Ops.CAST dtypes.i64 [175] dtypes.i64
177 Ops.CONST dtypes.weakint [] 4611686018427387904
178 Ops.MUL dtypes.i64 [176, 177]
179 Ops.MUL dtypes.i64 [178, 19]
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
199 Ops.AND dtypes.u64 [161, 12]
200 Ops.MUL dtypes.u64 [199, 85]
201 Ops.AND dtypes.u64 [159, 12]
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
254 Ops.CAST dtypes.f16 [253] dtypes.f16
255 Ops.DIV dtypes.u64 [169, 177]
256 Ops.ADD dtypes.u64 [255, 175]
257 Ops.CAST dtypes.i32 [256] dtypes.i32
258 Ops.SINK dtypes.void [254, 257]

