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

(* What a trace gives the lowering: its first device, whose measured options fix
   the kernel's row bound and the grouping cost, [quant_matmul ?ids x ~codes
   ~scales], tolk's [Op.quant_matmul] over traced values: [x] [[| ix; m; k |]],
   matrices [[| e; n; k / 2 |]] and ids [[| i |]], each of the [i] instances
   taking block [t / (i / ix)] of [x]; and [block_matmul ~transpose x w ~ids],
   tolk's [Op.block_matmul]. *)
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

let renderer kernels = Tolk.Device.renderer kernels.device

(* The most rows a matrix may meet in the kernel, rules 1 and 3 alike. *)
let row_bound kernels = Tolk_frontend.Op.quant_row_bound (renderer kernels)

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
  let rows_fit r = r > 0 && n > 0 && r <= row_bound kernels in
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

(* Each route is its own block of [m] rows, padded with zero rows up to a
   multiple of the smallest of the block kernel's row tiles where it has any:
   rows that no tile divides run the kernel without its pinned options, on Metal
   2.3 times slower at 100 rows of gpt-oss's gate_up. [form] names the form in
   the report. *)
let instances kernels ~form ~transpose ~p ~e ids codes scales x =
  let vector = Nx.ndim x = 1 in
  let xb, rows = batch_rows x in
  let lanes = weight_lanes ~p codes scales in
  let ob, expert, row = routes ~e ~lanes ids xb in
  let w = flat_experts ~lanes codes scales x in
  let m = Nx.dim 1 rows and k = Nx.dim 2 rows in
  let n = if transpose then Nx.dim 2 w else Nx.dim 1 w in
  let tile =
    match
      List.rev (Tolk_frontend.Op.block_row_tiles (renderer kernels) ~n ~k)
    with
    | smallest :: _ -> smallest
    | [] -> 1
  in
  let padded = (m + tile - 1) / tile * tile in
  report
    (Printf.sprintf "%s, blocks of %d rows on the block kernel" form padded);
  let blocks = Nx.take ~axis:0 ~indices:row rows in
  let blocks =
    if padded = m then blocks
    else Nx.pad [| (0, 0); (0, padded - m); (0, 0) |] 0.0 blocks
  in
  let y =
    kernels.block_matmul ~transpose:(not transpose) blocks w ~ids:expert
  in
  let y =
    if padded = m then y else Nx.shrink [| (0, Nx.dim 0 y); (0, m); (0, n) |] y
  in
  Nx.reshape (result_shape ~vector ob m n) y

(* The rows of a block that decode-then-matmul multiplies: the largest of the
   block kernel's row tiles at most the rows an expert meets on average, or the
   smallest. Where no options are pinned no size fills a tile, and a block is as
   many rows as an expert meets on average, rounded up to a power of two, at
   most 64. *)
let block_rows kernels ~n ~k ~per_expert =
  match Tolk_frontend.Op.block_row_tiles (renderer kernels) ~n ~k with
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
  let bound = if transpose then 0 else row_bound kernels in
  let on_kernel = bound > 0 && per_expert <= bound in
  let b =
    if on_kernel then kernel_rows kernels ~per_expert
    else
      (* The block kernel's own outputs: the weight's inputs for the transposed
         product. *)
      let n = if transpose then 2 * Nx.dim (-1) codes else n in
      block_rows kernels ~n ~k:cols ~per_expert:(r / d)
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
   in gpt-oss-20b's MoE block, grouped / ungrouped time was, from 48 to 160
   routes, 1.33, 1.13, 0.83, 0.78, 0.77, 0.75, 0.70, 0.72, 0.67 and 0.64 at
   float32 (at 48, 64, 96, 100, 104, 112, 120, 124, 128 and 160 routes) and
   1.85, 1.52, 1.14, 1.12, 1.13, 1.07, 1.07, 1.07, 1.01 and 0.68 at bfloat16.
   [tau] is chosen as the row bound is: the threshold whose largest loss over
   both dtypes, a loss from grouping and a win forgone alike, is smallest.
   Grouping from 96 routes costs at most bfloat16's 1.14 there; from 100 it
   would forgo float32's 1.20 at 96, from 128 its 1.43 at 120, and from 64 it
   would cost bfloat16 1.52. [tau] is 140, which over 32 experts groups from 96
   routes. *)
let tau device =
  match Tolk.Renderer.device (Tolk.Device.renderer device) with
  | "METAL" -> 16.0
  | "CPU" -> 140.0
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

(* Every decoded form multiplies with the block kernel. Without ids, the
   weight's stack is its experts and each position of the product's batch
   addresses its own matrix. With fewer positions than experts, the selected
   experts' packed rows are gathered, then decoded, and each position addresses
   its gathered matrix; a position that selects no expert gathers zeros and
   addresses none. *)
let plain_blocks kernels ~transpose codes scales x =
  let cs = Nx.shape codes in
  let r = Array.length cs in
  let wb = Array.sub cs 0 (r - 2) in
  let flat t =
    let s = Nx.shape t in
    Nx.reshape (Array.append [| count wb |] (Array.sub s (r - 2) 2)) t
  in
  instances kernels ~form:"decoded" ~transpose ~p:0 ~e:(count wb)
    (lane_index wb) (flat codes) (flat scales) x

let gathered_blocks kernels ~transpose ~p ~e ids codes scales x =
  let lanes = weight_lanes ~p codes scales in
  let selected, expert, _ = routes ~e ~lanes ids [||] in
  let g = count selected in
  let codes, scales = flat_parts ~lanes codes scales in
  if g = 0 then
    let xb, rows = batch_rows x in
    let ob = batch_shape ~lanes xb (Nx.shape ids) in
    Nx.zeros (Nx.dtype x)
      (result_shape ~vector:(Nx.ndim x = 1) ob (Nx.dim 1 rows) (Nx.dim 1 codes))
  else
    let take part = Nx.take ~axis:0 ~indices:expert part in
    let own =
      Nx.where
        (Nx.greater_equal_s expert 0l)
        (Nx.arange Nx.int32 0 g 1)
        (Nx.full Nx.int32 [||] (-1l))
    in
    instances kernels ~form:"gathered" ~transpose ~p:0 ~e:g
      (Nx.reshape selected own) (take codes) (take scales) x

let product kernels ~transpose ?ids (Nx_quant.Mxfp4 { codes; scales }) x =
  match ids with
  | None -> plain_blocks kernels ~transpose codes scales x
  | Some ids ->
      let ws = Nx.shape codes and is = Nx.shape ids in
      let p = Array.length ws - 3 in
      let e = ws.(p) in
      let positions = count (Array.sub is p (Array.length is - p)) in
      if positions < e then
        gathered_blocks kernels ~transpose ~p ~e ids codes scales x
      else
        instances kernels ~form:"one block per position" ~transpose ~p ~e ids
          codes scales x

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
    match ids with
    | Some ids when groups kernels ~ids w x ->
        let (Nx_quant.Mxfp4 { codes; scales }) = w in
        let p = Nx.ndim codes - 3 in
        grouped kernels ~transpose ~p ~e:(Nx.dim p codes) ids codes scales x
    | _ when not transpose -> (
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

(* [lower kernels w op] is [op]'s compiled form. *)
let lower : type a b.
    kernels -> Nx_quant.t -> (a, b) Nx_quant.Effect.op -> (a, b) Nx.t =
 fun kernels (Nx_quant.Mxfp4 { codes; scales } as w) -> function
  | Apply { ids; x; transpose } -> apply kernels ~transpose ?ids w x
  | Dequant dt -> decode dt codes scales
