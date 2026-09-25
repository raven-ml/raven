(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled quantised products (RFC 0004).

   [lower] is the compiled form of [Nx_quant]'s effect, written as Nx operations
   that jit traces into its graph under its own handler. The form is chosen from
   static shapes. Routes that share experts are grouped by expert. A matrix that
   meets at most the device's row bound of rows takes tolk's kernel, which
   decodes in registers; every other product decodes, then multiplies, with
   tolk's block kernel where each block of rows has its own matrix. *)

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

(* Under [RUNE_JIT_DEBUG=1], each product's form is logged. *)
let report form =
  if Jit_cache.debug () >= 1 then
    Printf.eprintf "rune.jit: quantised product: %s\n%!" form

let broadcast a b =
  let la = Array.length a and lb = Array.length b in
  let l = max la lb in
  Array.init l (fun i ->
      let da = if i < l - la then 1 else a.(i - l + la) in
      let db = if i < l - lb then 1 else b.(i - l + lb) in
      if da = 1 then db else da)

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

(* What a single-device trace gives the lowering: its device, whose measured
   options fix the kernel's row bound and the grouping cost, [quant_matmul ?ids
   x ~codes ~scales], tolk's [Op.quant_matmul] over traced values: [x] [[| ix;
   m; k |]], matrices [[| e; n; k / 2 |]] and ids [[| i |]], each of the [i]
   instances taking block [t / (i / ix)] of [x]; and [block_matmul ~transpose x
   w ~ids], tolk's [Op.block_matmul]. *)
type kernels = {
  device : Tolk.Device.t;
  quant_matmul :
    'b.
    ?ids:Nx.int32_t ->
    (float, 'b) Nx.t ->
    codes:(int, Nx.uint8_elt) Nx.t ->
    scales:(int, Nx.uint8_elt) Nx.t ->
    (float, 'b) Nx.t;
  block_matmul :
    'b.
    transpose:bool ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t ->
    ids:Nx.int32_t ->
    (float, 'b) Nx.t;
}

(* [x]'s dtype in tolk, for the kernels' measured options. *)
let tolk_dtype (type b) (x : (float, b) Nx.t) =
  match Nx.dtype x with
  | Nx.Float32 -> Some Tolk_uop.Dtype.float32
  | Nx.BFloat16 -> Some Tolk_uop.Dtype.bfloat16
  | Nx.Float16 -> Some Tolk_uop.Dtype.float16
  | _ -> None

let renderer kernels = Tolk.Device.renderer kernels.device

(* The most rows one [n] by [k] matrix may meet in the kernel, for [x]'s
   dtype. *)
let row_bound kernels x ~n ~k =
  match tolk_dtype x with
  | Some dtype ->
      Tolk_frontend.Op.quant_row_bound (renderer kernels) dtype ~n ~k
  | None -> 0

(* [x] as the kernel's [[| ix; m; k |]] for a product whose batch is [batch]:
   its batch axes must be a prefix of [batch] followed by units, or it is
   broadcast to [batch] first. *)
let kernel_x ~batch x =
  let xs = Nx.shape x in
  let r = Array.length xs in
  let mk = Array.sub xs (r - 2) 2 in
  let b = Array.length batch in
  let xb = Array.append (ones (b - (r - 2))) (Array.sub xs 0 (r - 2)) in
  let rec prefix j =
    if j = b || xb.(j) <> batch.(j) then j else prefix (j + 1)
  in
  let j = prefix 0 in
  if Array.for_all (( = ) 1) (Array.sub xb j (b - j)) then
    Nx.reshape
      (Array.append [| count (Array.sub xb 0 j) |] mk)
      (Nx.contiguous x)
  else
    Nx.reshape
      (Array.append [| count batch |] mk)
      (Nx.contiguous (Nx.broadcast_to (Array.append batch mk) x))

(* Rule 1: without ids, a matrix meeting [r] rows takes the kernel while [r <=
   row_bound]; with ids and ungrouped, each instance takes it with [r = m].
   [None] where the product takes another form. *)
let by_kernel kernels ?ids (Nx_quant.Mxfp4 { codes; scales }) x =
  let vector = Nx.ndim x = 1 in
  let x =
    if vector then Nx.reshape (Array.append [| 1 |] (Nx.shape x)) x else x
  in
  let xs = Nx.shape x and cs = Nx.shape codes in
  let xr = Array.length xs and cr = Array.length cs in
  let m = xs.(xr - 2) and k = xs.(xr - 1) and n = cs.(cr - 2) in
  let xb = Array.sub xs 0 (xr - 2) in
  let finish batch y =
    let rows = if vector then [| n |] else [| m; n |] in
    Some (Nx.reshape (Array.append batch rows) y)
  in
  let rows_fit r = r > 0 && n > 0 && r <= row_bound kernels x ~n ~k in
  let parts e =
    (Nx.reshape [| e; n; k / 2 |] codes, Nx.reshape [| e; n; k / 32 |] scales)
  in
  match ids with
  | None ->
      let wb = Array.sub cs 0 (cr - 2) in
      if Array.for_all (( = ) 1) wb then
        let r = count xb * m in
        if not (rows_fit r) then None
        else
          let codes, scales = parts 1 in
          let x = Nx.reshape [| 1; r; k |] (Nx.contiguous x) in
          finish (broadcast wb xb) (kernels.quant_matmul x ~codes ~scales)
      else
        let batch = broadcast wb xb in
        let wb =
          Array.append (ones (Array.length batch - Array.length wb)) wb
        in
        if wb <> batch || not (rows_fit m) then None
        else
          let codes, scales = parts (count batch) in
          finish batch (kernels.quant_matmul (kernel_x ~batch x) ~codes ~scales)
  | Some ids ->
      let p = cr - 3 in
      let e = cs.(p) and is = Nx.shape ids in
      if not (rows_fit m) then None
      else
        (* Lanes flattened: lane [l]'s expert [id] is matrix [l * e + id], and
           an id outside the experts stays outside every lane's. *)
        let index =
          if p = 0 then ids
          else
            let valid =
              Nx.logical_and
                (Nx.greater_equal_s ids 0l)
                (Nx.less_s ids (Int32.of_int e))
            in
            let lane = lane_index (Array.sub cs 0 p) in
            let lane =
              Nx.reshape
                (Array.append (Nx.shape lane) (ones (Array.length is - p)))
                lane
            in
            Nx.where valid
              (Nx.add (Nx.mul_s lane (Int32.of_int e)) ids)
              (Nx.full Nx.int32 [||] (-1l))
        in
        let batch = broadcast xb (Nx.shape index) in
        if count batch = 0 then None
        else
          let ids =
            Nx.reshape
              [| count batch |]
              (Nx.contiguous (Nx.broadcast_to batch index))
          in
          let codes, scales = parts (count (Array.sub cs 0 (p + 1))) in
          finish batch
            (kernels.quant_matmul ~ids (kernel_x ~batch x) ~codes ~scales)

(* With as many positions as experts or more, every matrix is decoded once, and
   each block of rows is multiplied by the decoded matrix its id addresses, read
   in place by the block kernel. A route is a position of the product's batch: a
   row block of [x] and an id. Lanes are flattened into the experts, an id [i]
   of lane [l] addressing expert [l * e + i], so one ranking serves every
   lane. *)

(* Without the block kernel, in a program over several devices, the dense form
   (rule 3's stop outcome): every matrix is decoded once, and every row of [x]
   is multiplied by every expert of its lane, each position then keeping its own
   expert's product; an index outside the experts gathers zeros. It holds [e]
   products per position where a copy of one decoded matrix per position would
   hold [n * k] values each (68 GB for gpt-oss's [gate_up] at a 512-token
   prefill). Routes are not grouped there: their ranking would run along a
   sharded axis. The products are materialised: fused into the gather, the
   product loses its matmul kernel. *)
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

(* The product's batch: [x]'s batch [xb], [ids]'s shape [is] and the weight's
   lanes, broadcast. *)
let batch_shape ~lanes xb is =
  let q = Array.length is - Array.length lanes in
  broadcast (broadcast xb is) (Array.append lanes (ones q))

(* [routes ~e ~lanes ids xb] is the product's batch [ob] and, over it flattened,
   each route's expert among every lane's (-1 for none) and its row block of
   [x]. *)
let routes ~e ~lanes ids xb =
  let p = Array.length lanes in
  let is = Nx.shape ids in
  let ob = batch_shape ~lanes xb is in
  let rank = Array.length ob in
  let valid =
    Nx.logical_and (Nx.greater_equal_s ids 0l) (Nx.less_s ids (Int32.of_int e))
  in
  let expert =
    if p = 0 then Nx.where valid ids (Nx.full Nx.int32 [||] (-1l))
    else
      let q = Array.length is - p in
      let lane = lane_index lanes in
      let lane = Nx.reshape (Array.append (Nx.shape lane) (ones q)) lane in
      Nx.where valid
        (Nx.add (Nx.mul_s lane (Int32.of_int e)) ids)
        (Nx.full Nx.int32 [||] (-1l))
  in
  let flat t =
    let s = Nx.shape t in
    let t = Nx.reshape (Array.append (ones (rank - Array.length s)) s) t in
    Nx.reshape [| -1 |] (Nx.contiguous (Nx.broadcast_to ob t))
  in
  let row = Nx.reshape xb (Nx.arange Nx.int32 0 (count xb) 1) in
  (ob, flat expert, flat row)

(* [x] as [[| rows of its batch; m; cols |]], a vector lifted to one row. *)
let batch_rows x =
  let x =
    if Nx.ndim x = 1 then Nx.reshape (Array.append [| 1 |] (Nx.shape x)) x
    else x
  in
  let xs = Nx.shape x in
  let r = Array.length xs in
  let xb = Array.sub xs 0 (r - 2) in
  (xb, Nx.reshape [| count xb; xs.(r - 2); xs.(r - 1) |] (Nx.contiguous x))

(* The lanes of a weight whose parts may be mapped apart, and its matrices
   decoded with the lanes flattened into the experts. *)
let weight_lanes ~p codes scales =
  broadcast (Array.sub (Nx.shape codes) 0 p) (Array.sub (Nx.shape scales) 0 p)

let flat_parts ~lanes codes scales =
  let p = Array.length lanes in
  let flat t =
    let s = Nx.shape t in
    let tail = Array.sub s p 3 in
    let t =
      if Array.sub s 0 p = lanes then t
      else Nx.contiguous (Nx.broadcast_to (Array.append lanes tail) t)
    in
    Nx.reshape (Array.append [| -1 |] (Array.sub tail 1 2)) t
  in
  (flat codes, flat scales)

let flat_experts ~lanes codes scales x =
  let codes, scales = flat_parts ~lanes codes scales in
  decoded (Nx.dtype x) codes scales

let result_shape ~vector ob rows cols =
  if vector then Array.append ob [| cols |]
  else Array.concat [ ob; [| rows; cols |] ]

(* Each route is its own block of [m] rows. *)
let instances block_matmul ~transpose ~p ~e ids codes scales x =
  let vector = Nx.ndim x = 1 in
  let xb, rows = batch_rows x in
  let lanes = weight_lanes ~p codes scales in
  let ob, expert, row = routes ~e ~lanes ids xb in
  let w = flat_experts ~lanes codes scales x in
  let y =
    block_matmul ~transpose:(not transpose)
      (Nx.take ~axis:0 ~indices:row rows)
      w ~ids:expert
  in
  let ys = Nx.shape y in
  Nx.reshape (result_shape ~vector ob ys.(1) ys.(2)) y

(* The rows of a block that decode-then-matmul multiplies: the largest of the
   block kernel's row tiles at most the rows an expert meets on average, or the
   smallest. Where no options are pinned no size fills a tile, and a block is as
   many rows as an expert meets on average, rounded up to a power of two, at
   most 64. *)
let block_rows kernels dtype ~n ~k ~per_expert =
  match Tolk_frontend.Op.block_row_tiles (renderer kernels) dtype ~n ~k with
  | [] ->
      let rec up b = if b >= per_expert || b >= 64 then b else up (2 * b) in
      up 1
  | tiles -> (
      match List.find_opt (fun b -> b <= per_expert) tiles with
      | Some b -> b
      | None -> List.nth tiles (List.length tiles - 1))

(* The rows of a block that the kernel multiplies: the rows an expert meets on
   average, rounded up to a power of two, at most the kernel's row tile. *)
let kernel_rows kernels ~per_expert =
  let tile = Tolk_frontend.Op.quant_row_tile (renderer kernels) in
  let rec up b = if b >= per_expert || 2 * b > tile then b else up (2 * b) in
  up 1

(* Decode-then-matmul over blocks, at float32's exponent range. *)
let decoded_blocks (type b) kernels ~transpose ~lanes codes scales ~ids
    (x : (float, b) Nx.t) : (float, b) Nx.t =
  let run : type c. (float, c) Nx.t -> (float, c) Nx.t =
   fun x ->
    kernels.block_matmul ~transpose:(not transpose) x
      (flat_experts ~lanes codes scales x)
      ~ids
  in
  match Nx.dtype x with
  | Nx.Float32 | Nx.BFloat16 -> run x
  | dt -> Nx.cast dt (run (Nx.cast Nx.float32 x))

(* Grouped, for one row per route: routes ranked among their expert's, each
   expert's routes filling consecutive blocks of [b] rows, experts in order, so
   each block reads its expert once. The rank is a running sum of a one-hot of
   the experts; [nb] bounds the blocks for every assignment of routes. A slot no
   route fills reads nothing, and a block no route fills has an id past the
   experts. While the rows an expert meets on average are within the kernel's
   row bound, the blocks take the kernel (rule 3), each as many rows as an
   expert meets on average, rounded up to a power of two and at most the
   kernel's row tile: a row the kernel carries costs a share of a matrix read,
   and gpt-oss's decode step at 128 routes took 281 ms in tiles of 8 against 213
   ms in blocks of 4. Otherwise every expert is decoded and the block kernel
   multiplies each block by its expert. The transposed product, the reverse
   rule's, never takes the kernel. *)
let grouped kernels ~transpose ~p ~e ids codes scales x =
  let vector = Nx.ndim x = 1 in
  let xb, rows = batch_rows x in
  let cols = Nx.dim 2 rows in
  let lanes = weight_lanes ~p codes scales in
  let d = count lanes * e in
  let ob, expert, row = routes ~e ~lanes ids xb in
  let r = count ob in
  let n = Nx.dim (-2) codes in
  let per_expert = (r + d - 1) / d in
  let bound = if transpose then 0 else row_bound kernels x ~n ~k:cols in
  let on_kernel = bound > 0 && per_expert <= bound in
  let b =
    if on_kernel then kernel_rows kernels ~per_expert
    else
      (* The block kernel's own outputs and dtype: the weight's inputs for the
         transposed product, and float32 for a dtype decoded at float32. *)
      let n = if transpose then 2 * Nx.dim (-1) codes else n in
      let dtype =
        match tolk_dtype x with
        | Some dt when Tolk_uop.Dtype.equal dt Tolk_uop.Dtype.bfloat16 -> dt
        | _ -> Tolk_uop.Dtype.float32
      in
      block_rows kernels dtype ~n ~k:cols ~per_expert:(r / d)
  in
  report
    (Printf.sprintf
       "grouped, %d routes over %d experts, blocks of %d rows on %s" r d b
       (if on_kernel then "the kernel" else "the block kernel"));
  let nb = ((r + b - 1) / b) + d in
  let one_hot =
    Nx.cast Nx.int32
      (Nx.equal
         (Nx.reshape [| r; 1 |] expert)
         (Nx.reshape [| 1; d |] (Nx.arange Nx.int32 0 d 1)))
  in
  let rank =
    Nx.sum ~axes:[ 1 ]
      (Nx.mul one_hot (Nx.sub_s (Nx.cumsum ~axis:0 one_hot) 1l))
  in
  let blocks =
    Nx.div_s
      (Nx.add_s (Nx.sum ~axes:[ 0 ] one_hot) (Int32.of_int (b - 1)))
      (Int32.of_int b)
  in
  let last = Nx.cumsum blocks in
  let first = Nx.sub last blocks in
  let valid = Nx.greater_equal_s expert 0l in
  let slot =
    Nx.where valid
      (Nx.add (Nx.mul_s (Nx.take ~indices:expert first) (Int32.of_int b)) rank)
      (Nx.full Nx.int32 [||] (-1l))
  in
  let table =
    Nx.scatter ~unique_indices:true ~axis:0 ~indices:slot ~values:row
      (Nx.full Nx.int32 [| nb * b |] (-1l))
  in
  let xblocks =
    Nx.reshape [| nb; b; cols |]
      (Nx.take ~axis:0 ~indices:table (Nx.reshape [| -1; cols |] rows))
  in
  let block_ids =
    Nx.sum ~axes:[ 1 ]
      (Nx.cast Nx.int32
         (Nx.less_equal
            (Nx.reshape [| 1; d |] last)
            (Nx.reshape [| nb; 1 |] (Nx.arange Nx.int32 0 nb 1))))
  in
  let y =
    if on_kernel then
      let codes, scales = flat_parts ~lanes codes scales in
      kernels.quant_matmul ~ids:block_ids xblocks ~codes ~scales
    else
      decoded_blocks kernels ~transpose ~lanes codes scales ~ids:block_ids
        xblocks
  in
  let out = Nx.dim 2 y in
  let y = Nx.take ~axis:0 ~indices:slot (Nx.reshape [| nb * b; out |] y) in
  let y = Nx.where (Nx.reshape [| r; 1 |] valid) y (Nx.zeros_like y) in
  Nx.reshape (result_shape ~vector ob 1 out) y

(* Rule 2: routes of one row are grouped when the reads of an expert that a
   route shares with an earlier route, [r (r - 1) / 2e] under uniform routing of
   a lane's [r] routes, exceed the grouping's fixed cost [tau], counted in reads
   of one matrix and measured per device; a device not measured never groups.
   They are grouped only when a lane's routes exceed its experts: below that the
   kernel's blocks are one row, and a block of one row shares no read. On Metal,
   at gpt-oss-20b's decode step with bfloat16 activations, 32 routes over 32
   experts lost for that reason (111 against 101 ms per step) and 64 won (135
   against 150 ms; 213 against 249 ms at 128): [tau] is 16 there, its crossover
   measured no closer, as nothing between 33 and 63 routes was run. On the CPU,
   in gpt-oss-20b's MoE block with the block kernel's CPU options, grouping lost
   at 128 routes at float32 (0.98 against 0.76 s) and at bfloat16 up to 256
   (3.24 against 3.15 s; 2.88 against 2.39 s at 192), and won at both from 384
   (4.08 against 4.81 s at bfloat16; at 512, 1.53 against 3.04 s at float32 and
   4.91 against 6.36 s at bfloat16): [tau] is 1024 there, which over 32 experts
   groups from 257 routes, forgoing float32's wins at 192 and 256; nothing
   between 257 and 383 routes was run. *)
let tau device =
  match Tolk.Renderer.device (Tolk.Device.renderer device) with
  | "METAL" -> 16.0
  | "CPU" -> 1024.0
  | _ -> Float.infinity

let groups kernels ~ids (Nx_quant.Mxfp4 { codes; scales }) x =
  let ws = Nx.shape codes and xs = Nx.shape x in
  let p = Array.length ws - 3 in
  let e = ws.(p) in
  let rank = Array.length xs in
  let m = if rank = 1 then 1 else xs.(rank - 2) in
  let xb = Array.sub xs 0 (max 0 (rank - 2)) in
  let lanes = weight_lanes ~p codes scales in
  let routes =
    count (batch_shape ~lanes xb (Nx.shape ids)) / max 1 (count lanes)
  in
  let r = float_of_int routes in
  m = 1 && routes > e
  && r *. (r -. 1.0) /. (2.0 *. float_of_int e) > tau kernels.device

let product kernels ~transpose ?ids (Nx_quant.Mxfp4 { codes; scales }) x =
  match ids with
  | None ->
      report "decoded";
      let w = decoded (Nx.dtype x) codes scales in
      Nx.matmul x (if transpose then w else Nx.matrix_transpose w)
  | Some ids -> (
      let ws = Nx.shape codes and is = Nx.shape ids in
      let p = Array.length ws - 3 in
      let e = ws.(p) in
      let positions = count (Array.sub is p (Array.length is - p)) in
      if positions < e then begin
        report "gathered";
        gathered ~transpose ~p ~e ids codes scales x
      end
      else
        match kernels with
        | Some kernels ->
            report "one block per position";
            instances kernels.block_matmul ~transpose ~p ~e ids codes scales x
        | None ->
            report "dense";
            every ~transpose ~p ~e ids codes scales x)

(* Rule 2 first, then rule 1. The kernel takes float32, bfloat16 and float16 as
   they are, and the transposed product, the reverse rule's, never takes it.
   Decoding happens at [x]'s dtype when it has float32's exponent range, and
   otherwise at float32. *)
let apply (type b) kernels ~transpose ?ids w (x : (float, b) Nx.t) :
    (float, b) Nx.t =
  let decoded_form : type c. (float, c) Nx.t -> (float, c) Nx.t =
   fun x ->
    match Nx.dtype x with
    | Nx.Float32 | Nx.BFloat16 -> product kernels ~transpose ?ids w x
    | dt ->
        Nx.cast dt (product kernels ~transpose ?ids w (Nx.cast Nx.float32 x))
  in
  let form : type c. (float, c) Nx.t -> (float, c) Nx.t =
   fun x ->
    match (kernels, ids) with
    | Some kernels, Some ids when groups kernels ~ids w x ->
        let (Nx_quant.Mxfp4 { codes; scales }) = w in
        let p = Nx.ndim codes - 3 in
        grouped kernels ~transpose ~p ~e:(Nx.dim p codes) ids codes scales x
    | Some kernels, _ when not transpose -> (
        match by_kernel kernels ?ids w x with
        | Some y ->
            report "kernel";
            y
        | None -> decoded_form x)
    | _ -> decoded_form x
  in
  match Nx.dtype x with
  | Nx.Float32 | Nx.BFloat16 | Nx.Float16 -> form x
  | dt -> Nx.cast dt (form (Nx.cast Nx.float32 x))

(* [lower kernels w op] is [op]'s compiled form; [kernels] is [None] in a
   program over several devices. *)
let lower : type a b.
    kernels option -> Nx_quant.t -> (a, b) Nx_quant.Effect.op -> (a, b) Nx.t =
 fun kernels (Nx_quant.Mxfp4 { codes; scales } as w) -> function
  | Apply { ids; x; transpose } -> apply kernels ~transpose ?ids w x
  | Dequant dt -> decode dt codes scales
