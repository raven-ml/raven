(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled quantised products (RFC 0004).

   [lower] is the compiled form of [Nx_quant]'s effect, written as Nx operations
   that jit traces into its graph under its own handler. The form is chosen from
   static shapes. The kernel, for a matrix that meets at most a row bound of
   rows, and the grouped form, for experts that many routes share, do not exist
   yet: the row bound is 0 and the grouping cost infinite, so every product
   decodes, then multiplies. *)

(* Decoding. Values are computed at float32, where every product of a code and a
   scale is exact, and cast once. *)

(* nx has no ldexp: the 256 powers of two are a table. *)
let scale_values =
  Array.init 256 (fun e ->
      if e = 255 then Float.nan else Float.ldexp 1.0 (e - 127))

let group_scales scales =
  let table = Nx.create Nx.float32 [| 256 |] scale_values in
  let indices = Nx.reshape [| -1 |] (Nx.cast Nx.int32 scales) in
  Nx.reshape (Nx.shape scales) (Nx.take ~indices table)

(* The magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 are m / 2 up to 4, m - 2 for 5
   and 6, then 6. Arithmetic, where a 16-entry table would be a gather: the
   scheduler materialises a gather's indices, four bytes for each weight. *)
let code_values codes =
  let m = Nx.cast Nx.float32 (Nx.bitwise_and codes (Nx.scalar Nx.uint8 7)) in
  let magnitude =
    Nx.where (Nx.less_s m 5.0) (Nx.mul_s m 0.5)
      (Nx.where (Nx.less_s m 7.0) (Nx.sub_s m 2.0) (Nx.full_like m 6.0))
  in
  let sign = Nx.cast Nx.float32 (Nx.rshift codes 3) in
  Nx.mul magnitude (Nx.rsub_s 1.0 (Nx.mul_s sign 2.0))

(* [decode dt codes scales] is the values of MXFP4 parts [[| ...; n; k / 2 |]]
   and [[| ...; n; k / 32 |]] at [dt], [[| ...; n; k |]]. *)
let decode dt codes scales =
  let shape = Nx.shape codes in
  let rank = Array.length shape in
  let blocks = Nx.reshape (Array.append (Nx.shape scales) [| 16 |]) codes in
  let low = Nx.bitwise_and blocks (Nx.scalar Nx.uint8 15) in
  let high = Nx.rshift blocks 4 in
  let values = code_values (Nx.stack ~axis:(-1) [ low; high ]) in
  let scale =
    Nx.reshape (Array.append (Nx.shape scales) [| 1; 1 |]) (group_scales scales)
  in
  let out = Array.copy shape in
  out.(rank - 1) <- 2 * shape.(rank - 1);
  Nx.reshape out (Nx.cast dt (Nx.mul values scale))

(* Shapes *)

let ones n = Array.make n 1
let count dims = Array.fold_left ( * ) 1 dims

let broadcast a b =
  let la = Array.length a and lb = Array.length b in
  let l = max la lb in
  Array.init l (fun i ->
      let da = if i < l - la then 1 else a.(i - l + la) in
      let db = if i < l - lb then 1 else b.(i - l + lb) in
      max da db)

(* [lane_index lanes] is, for a weight whose leading axes are [lanes], each
   lane's position among them, of shape [lanes]: a unit axis indexes 0 whatever
   the ids broadcast it against. *)
let lane_index lanes =
  let p = Array.length lanes in
  let index = ref (Nx.zeros Nx.int32 (ones p)) and stride = ref 1 in
  for a = p - 1 downto 0 do
    if lanes.(a) > 1 then begin
      let shape = ones p in
      shape.(a) <- lanes.(a);
      let along = Nx.arange Nx.int32 0 (lanes.(a) * !stride) !stride in
      index := Nx.add !index (Nx.reshape shape along)
    end;
    stride := !stride * lanes.(a)
  done;
  !index

(* Decode-then-matmul. Decoded matrices are materialised: a product of two
   buffers is what tolk's heuristics take for a matrix product, and with the
   decode inside the product gpt-oss's decode step was four times slower on
   Metal. *)

let decoded dt codes scales = Nx.contiguous (decode dt codes scales)

(* With fewer positions than experts, the selected experts' packed rows are
   gathered, then decoded. A position that selects no expert gathers zero bytes,
   reading nothing, and its product is set to zero after it: a zero row times an
   [x] that is not finite is NaN. *)
let gathered ~transpose ~p ~e ids codes scales x =
  let ws = Nx.shape codes and is = Nx.shape ids in
  let valid =
    Nx.logical_and (Nx.greater_equal_s ids 0l) (Nx.less_s ids (Int32.of_int e))
  in
  (* An index into the weight's experts, lanes flattened: [-1] for none. *)
  let index =
    if p = 0 then ids
    else
      let q = Array.length is - p in
      let lane = lane_index (Array.sub ws 0 p) in
      let lane = Nx.reshape (Array.append (Nx.shape lane) (ones q)) lane in
      Nx.where valid
        (Nx.add (Nx.mul_s lane (Int32.of_int e)) ids)
        (Nx.full Nx.int32 [||] (-1l))
  in
  let take part =
    let s = Nx.shape part in
    let r = Array.length s in
    let inner = Array.sub s (r - 2) 2 in
    let flat =
      if p = 0 then part else Nx.reshape (Array.append [| -1 |] inner) part
    in
    let rows = Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] index) flat in
    Nx.reshape (Array.append (Nx.shape index) inner) rows
  in
  let w = decoded (Nx.dtype x) (take codes) (take scales) in
  let y = Nx.matmul x (if transpose then w else Nx.matrix_transpose w) in
  let valid = Nx.broadcast_to (Nx.shape index) valid in
  let tail = if Nx.ndim x = 1 then ones 1 else ones 2 in
  let valid = Nx.reshape (Array.append (Nx.shape valid) tail) valid in
  Nx.where valid y (Nx.zeros_like y)

(* Otherwise every matrix is decoded once, and every row of [x] is multiplied by
   every expert of its lane, each position then keeping its own expert's
   product: an index outside the experts gathers zeros. This multiplies rows by
   experts that did not select them, as the grouped form's stop outcome does; it
   stands in for the block kernel until the grouped form exists. The products
   are materialised: fused into the gather, the product loses its matmul
   kernel. *)
let every ~transpose ~p ~e ids codes scales x =
  let vector = Nx.ndim x = 1 in
  let x =
    if vector then Nx.reshape (Array.append [| 1 |] (Nx.shape x)) x else x
  in
  let xs = Nx.shape x and is = Nx.shape ids in
  let xr = Array.length xs in
  let xb = Array.sub xs 0 (xr - 2) and mk = Array.sub xs (xr - 2) 2 in
  let q = Array.length is - p in
  let rank = max (Array.length xb) (p + q) in
  let pre = rank - p - q in
  (* x's batch as [pre; lanes; experts; positions]. *)
  let xb = Array.append (ones (rank - Array.length xb)) xb in
  let x =
    Nx.reshape
      (Array.concat
         [ Array.sub xb 0 (pre + p); [| 1 |]; Array.sub xb (pre + p) q; mk ])
      x
  in
  let w = decoded (Nx.dtype x) codes scales in
  let ds = Nx.shape w in
  let w =
    Nx.reshape
      (Array.concat
         [ ones pre; Array.sub ds 0 (p + 1); ones q; Array.sub ds (p + 1) 2 ])
      w
  in
  let z =
    Nx.contiguous (Nx.matmul x (if transpose then w else Nx.matrix_transpose w))
  in
  let zs = Nx.shape z in
  let lanes = broadcast (Array.sub zs pre p) (Array.sub is 0 p)
  and positions = broadcast (Array.sub zs (pre + p + 1) q) (Array.sub is p q) in
  let rows = Array.sub zs (pre + p + 1 + q) 2 in
  let batch = Array.concat [ Array.sub zs 0 pre; lanes ] in
  let z =
    Nx.broadcast_to (Array.concat [ batch; [| e |]; positions; rows ]) z
  in
  let index =
    Nx.broadcast_to
      (Array.concat [ batch; [| 1 |]; positions; rows ])
      (Nx.reshape
         (Array.concat
            [ ones pre; Array.sub is 0 p; [| 1 |]; Array.sub is p q; ones 2 ])
         ids)
  in
  let y = Nx.take_along_axis ~axis:(pre + p) ~indices:index z in
  let rows = if vector then [| rows.(1) |] else rows in
  Nx.reshape (Array.concat [ batch; positions; rows ]) y

let product ~transpose ?ids (Nx_quant.Mxfp4 { codes; scales }) x =
  match ids with
  | None ->
      let w = decoded (Nx.dtype x) codes scales in
      Nx.matmul x (if transpose then w else Nx.matrix_transpose w)
  | Some ids ->
      let ws = Nx.shape codes and is = Nx.shape ids in
      let p = Array.length ws - 3 in
      let e = ws.(p) in
      let positions = count (Array.sub is p (Array.length is - p)) in
      if positions < e then gathered ~transpose ~p ~e ids codes scales x
      else every ~transpose ~p ~e ids codes scales x

(* Decoding happens at [x]'s dtype when it has float32's exponent range, and
   otherwise at float32. *)
let apply (type b) ~transpose ?ids w (x : (float, b) Nx.t) : (float, b) Nx.t =
  match Nx.dtype x with
  | Nx.Float32 | Nx.BFloat16 -> product ~transpose ?ids w x
  | dt -> Nx.cast dt (product ~transpose ?ids w (Nx.cast Nx.float32 x))

let lower : type a b. Nx_quant.t -> (a, b) Nx_quant.Effect.op -> (a, b) Nx.t =
 fun (Nx_quant.Mxfp4 { codes; scales } as w) -> function
  | Apply { ids; x; transpose } -> apply ~transpose ?ids w x
  | Dequant dt -> decode dt codes scales
