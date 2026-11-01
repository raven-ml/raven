(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'd t =
  | Mxfp4 of { codes : 'd Nx.uint8_t; scales : 'd Nx.uint8_t }
  | Mxfp4_blocks of { blocks : 'd Nx.uint8_t }
  | Q8_0 of { blocks : 'd Nx.uint8_t }
  | Q4_K of { blocks : 'd Nx.uint8_t }
  | Q6_K of { blocks : 'd Nx.uint8_t }

let strf = Printf.sprintf

let pp_shape s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* Block layouts. MXFP4 stores a group of 32 values as 16 bytes of codes and a
   scale byte: apart in a safetensors checkpoint, the scale first in a GGUF
   block. The GGUF formats store a block's scales with its quants, as
   ggml-common.h's block structs lay them out. *)

let mxfp4_group = 32
let mxfp4_bytes = mxfp4_group / 2

(* block_mxfp4: e, qs[QK_MXFP4 / 2]. *)
let mxfp4_block_bytes = 1 + mxfp4_bytes

(* QK8_0, QK_K and K_SCALE_SIZE, and sizeof (ggml_half). *)
let qk8_0 = 32
let qk_k = 256
let k_scale_size = 12
let half_bytes = 2

(* block_q8_0: d, qs[QK8_0]. *)
let q8_0_bytes = half_bytes + qk8_0

(* block_q4_K: d, dmin, scales[K_SCALE_SIZE], qs[QK_K / 2]. *)
let q4_k_bytes = (2 * half_bytes) + k_scale_size + (qk_k / 2)

(* block_q6_K: ql[QK_K / 2], qh[QK_K / 4], scales[QK_K / 16], d. *)
let q6_k_bytes = (qk_k / 2) + (qk_k / 4) + (qk_k / 16) + half_bytes

(* Construction. Checks read shapes, never bytes. *)

(* [check_blocks fn format bytes blocks] checks that [blocks] is rows of whole
   blocks of [bytes] bytes. *)
let check_blocks fn format bytes blocks =
  let s = Nx.shape blocks in
  let r = Array.length s in
  if r < 2 || s.(r - 1) mod bytes <> 0 then
    invalid_arg
      (strf
         "%s: blocks must have shape [...; n; b * %d], rows of whole %s blocks \
          of %d bytes, got %s"
         fn bytes format bytes (pp_shape s))

let check_mxfp4 fn codes scales =
  let s = Nx.shape codes in
  let r = Array.length s in
  if r < 2 || s.(r - 1) mod mxfp4_bytes <> 0 then
    invalid_arg
      (strf
         "%s: codes must have shape [...; n; k / 2] with k a multiple of 32, \
          got %s"
         fn (pp_shape s));
  let groups =
    Array.append (Array.sub s 0 (r - 1)) [| s.(r - 1) / mxfp4_bytes |]
  in
  if Nx.shape scales <> groups then
    invalid_arg
      (strf "%s: scales must have shape %s, one per 32 values, got %s" fn
         (pp_shape groups)
         (pp_shape (Nx.shape scales)))

let mxfp4 ~scales codes =
  check_mxfp4 "Nx_quant.mxfp4" codes scales;
  Mxfp4 { codes; scales }

let mxfp4_blocks blocks =
  check_blocks "Nx_quant.mxfp4_blocks" "MXFP4" mxfp4_block_bytes blocks;
  Mxfp4_blocks { blocks }

let q8_0 blocks =
  check_blocks "Nx_quant.q8_0" "Q8_0" q8_0_bytes blocks;
  Q8_0 { blocks }

let q4_k blocks =
  check_blocks "Nx_quant.q4_k" "Q4_K" q4_k_bytes blocks;
  Q4_K { blocks }

let q6_k blocks =
  check_blocks "Nx_quant.q6_k" "Q6_K" q6_k_bytes blocks;
  Q6_K { blocks }

(* A weight's last axis counts the blocks of one of its parts: MXFP4's scales,
   one per group, or a GGUF format's bytes. *)
let shape w =
  let part, per, values =
    match w with
    | Mxfp4 { scales; _ } -> (scales, 1, mxfp4_group)
    | Mxfp4_blocks { blocks } -> (blocks, mxfp4_block_bytes, mxfp4_group)
    | Q8_0 { blocks } -> (blocks, q8_0_bytes, qk8_0)
    | Q4_K { blocks } -> (blocks, q4_k_bytes, qk_k)
    | Q6_K { blocks } -> (blocks, q6_k_bytes, qk_k)
  in
  let s = Array.copy (Nx.shape part) in
  let r = Array.length s in
  s.(r - 1) <- s.(r - 1) / per * values;
  s

(* Decoding. Every format decodes at float32, from bytes split into their blocks
   first. A split keeps a cut of [k] whose tiles divide the blocks, and any
   other raises before a value is decoded: the merge of the decoded blocks into
   rows then finds each block where it was decoded. *)

(* [split ~bytes t] is the rows [t], [[| …; n; b * bytes |]], as their blocks,
   [[| …; n; b; bytes |]]. *)
let split ~bytes t =
  let s = Nx.shape t in
  let r = Array.length s in
  Nx.reshape
    (Array.append (Array.sub s 0 (r - 1)) [| s.(r - 1) / bytes; bytes |])
    t

(* [groups b] is the shape of the blocks [b], [[| …; n; b; bytes |]], without
   their bytes. *)
let groups b =
  let s = Nx.shape b in
  Array.sub s 0 (Array.length s - 1)

(* [rows g ~values v] is the values [v] of the blocks [g], [[| …; n; b |]],
   [values] to a block, as rows [[| …; n; b * values |]]. *)
let rows g ~values v =
  let s = Array.copy g in
  let r = Array.length s in
  s.(r - 1) <- s.(r - 1) * values;
  Nx.reshape s v

(* [bytes b lo hi] is the bytes \[[lo];[hi]) of each block of [b]. *)
let bytes b lo hi =
  Nx.slice (List.init (Nx.ndim b - 1) (fun _ -> Nx.A) @ [ Nx.R (lo, hi) ]) b

(* [per g n t] is [t] as [n] values per block of [g], each to be broadcast along
   a last axis. *)
let per g n t = Nx.reshape (Array.append g [| n; 1 |]) t

(* [pick n f] is [f i], [[| …; m |]], for each [i] below [n], along a new axis
   before the last: [[| …; n; m |]]. Each element takes its field by its index
   along that axis, so each device computes its own blocks' fields, where
   [Nx.stack] would assemble the fields whole on each device. *)
let pick n f =
  let field i = Nx.unsqueeze ~axes:[ -2 ] (f i) in
  let first = field 0 in
  let at = Nx.beside first in
  let index = Nx.reshape [| n; 1 |] (Nx.arange_at at Nx.int32 0 n 1) in
  let rec go i acc =
    if i = n then acc
    else go (i + 1) (Nx.where (Nx.equal_s index (Int32.of_int i)) (field i) acc)
  in
  go 1 first

(* MXFP4. A code is an e2m1 float, a nibble that float4_e2m1fn reads as it is: a
   sign bit, two exponent bits and a mantissa bit. A scale byte is an e8m0
   exponent, 2^(s - 127), with 255 a NaN; its float32 bits are built with
   integer operations, 2^-127, a subnormal, at 0. Every value, scaled, is exact
   at float32 barring overflow. *)

(* [e2m1 codes] is the codes of the bytes [codes], two to a byte, along a new
   last axis, low nibble first, at float32. *)
let e2m1 codes = Nx.cast Nx.float32 (Nx.bitcast Nx.float4_e2m1fn codes)

(* [e8m0 s] is the scales of the bytes [s] at float32. *)
let e8m0 s =
  let s = Nx.cast Nx.uint32 s in
  let k = Nx.scalar_like s in
  Nx.bitcast Nx.float32
    (Nx.where (Nx.equal_s s 0l) (k 0x00400000l)
       (Nx.where (Nx.equal_s s 255l) (k 0x7FC00000l) (Nx.lshift s 23)))

(* [follow scales] is [scales], [[| …; n; b |]], read where a split of the codes
   into groups leaves them. The split keeps a cut of [k] whose tiles divide the
   groups, and a single group's cut along its bytes, but scales, one byte per
   group, have no bytes to split. Read as two nibbles each, a merge of their
   groups into rows and back keeps their cut where the tiles divide the groups
   and refuses any other, naming the scales' place; it is a view where they
   are contiguous. Read where they lie, scales cut where the tiles do not
   divide the groups would cut the decoded product, and the merge into rows
   would raise naming it in place of the scales.

   It exists because a checkpoint keeps the scales apart from the codes. Kept
   with their codes, as GGUF's blocks keep them, the split would carry them;
   packing them so costs a copy of the checkpoint's bytes at construction, where
   the constructor reads none. *)
let follow scales =
  let g = Nx.shape scales in
  let r = Array.length g in
  let row = Array.append (Array.sub g 0 (r - 1)) [| 2 * g.(r - 1) |] in
  let nibbles = Nx.reshape row (Nx.bitcast Nx.uint4 scales) in
  Nx.bitcast Nx.uint8 (Nx.reshape (Array.append g [| 2 |]) nibbles)

(* A checkpoint's byte [i] of a group holds values [2 i] and [2 i + 1], in
   order, so its nibbles are the group's codes. *)
let mxfp4_values codes scales =
  let g = Nx.shape scales in
  let codes = split ~bytes:mxfp4_bytes codes in
  let v = Nx.mul (e2m1 codes) (per g 1 (e8m0 (follow scales))) in
  rows g ~values:mxfp4_group v

(* A GGUF block's byte [j] holds values [j] and [j + 16], so its nibbles are the
   group's codes with their last two axes swapped. *)
let mxfp4_blocks_values blocks =
  let b = split ~bytes:mxfp4_block_bytes blocks in
  let g = groups b in
  let codes = Nx.swapaxes (-1) (-2) (e2m1 (bytes b 1 mxfp4_block_bytes)) in
  let v = Nx.mul codes (per g 1 (e8m0 (bytes b 0 1))) in
  rows g ~values:mxfp4_group v

(* The GGUF formats, decoded as ggml's dequantize_row_q8_0, _q4_K and _q6_K
   decode them: the scales at float32, each product left to right. *)

(* [half b at] is the little-endian float16 at byte [at] of each block of [b] at
   float32, [[| …; b; 1 |]]. *)
let half b at =
  let byte i = Nx.cast Nx.uint16 (bytes b i (i + 1)) in
  Nx.cast Nx.float32
    (Nx.bitcast Nx.float16
       (Nx.bitwise_or (byte at) (Nx.lshift (byte (at + 1)) 8)))

(* [int8 t] is the bytes [t] read as int8, at float32. *)
let int8 t = Nx.cast Nx.float32 (Nx.bitcast Nx.int8 t)

(* Q8_0: a float16 scale [d] and 32 int8 quants [q], each value [d * q]. *)
let q8_0_values blocks =
  let b = split ~bytes:q8_0_bytes blocks in
  rows (groups b) ~values:qk8_0
    (Nx.mul (int8 (bytes b half_bytes q8_0_bytes)) (half b 0))

(* Q4_K: float16 [d] and [dmin], then 8 sub-blocks of 32 values, each with a
   6-bit scale [sc] and min [m] packed in 12 bytes, of 4-bit quants [q]: a value
   is [d * sc * q - dmin * m]. Sub-blocks 0 to 3 take the low 6 bits of bytes 0
   to 3 and 4 to 7; sub-blocks 4 to 7 the nibbles of bytes 8 to 11 under the top
   2 bits of bytes 0 to 3 and 4 to 7. Each 32 bytes of quants hold two
   sub-blocks, the low nibbles first. *)
let q4_k_values blocks =
  let b = split ~bytes:q4_k_bytes blocks in
  let g = groups b in
  let k = Nx.scalar_like b in
  let low6 t = Nx.bitwise_and t (k 63) in
  let top2 t = Nx.lshift (Nx.rshift t 6) 4 in
  let scales i =
    bytes b ((2 * half_bytes) + (4 * i)) ((2 * half_bytes) + (4 * i) + 4)
  in
  let s0 = scales 0 and s1 = scales 1 and s2 = scales 2 in
  let join first last =
    Nx.cast Nx.float32
      (Nx.reshape (Array.append g [| 8 |])
         (pick 2 (function 0 -> first | _ -> last)))
  in
  let sc =
    join (low6 s0) (Nx.bitwise_or (Nx.bitwise_and s2 (k 15)) (top2 s0))
  in
  let m = join (low6 s1) (Nx.bitwise_or (Nx.rshift s2 4) (top2 s1)) in
  let quants = (2 * half_bytes) + k_scale_size in
  let qs =
    Nx.reshape (Array.append g [| 4; 32 |]) (bytes b quants q4_k_bytes)
  in
  let q =
    Nx.reshape
      (Array.append g [| 8; 32 |])
      (Nx.swapaxes (-1) (-2) (Nx.bitcast Nx.uint4 qs))
  in
  rows g ~values:qk_k
    (Nx.sub
       (Nx.mul (per g 8 (Nx.mul (half b 0) sc)) (Nx.cast Nx.float32 q))
       (per g 8 (Nx.mul (half b half_bytes) m)))

(* Q6_K: 16 sub-blocks of 16 values, each with an int8 scale [sc], of 6-bit
   quants [q] offset by 32, and a float16 [d]: a value is [d * sc * (q - 32)].
   Each half of a block, 128 values, takes 64 bytes of low nibbles and 32 bytes
   of high bit pairs. Its quarters 0 and 1 take the low nibbles of the first and
   last 32 of the 64 bytes, quarters 2 and 3 their high nibbles, and quarter [j]
   the bit pair [j] of each of the 32 bytes. *)
let q6_k_values blocks =
  let b = split ~bytes:q6_k_bytes blocks in
  let g = groups b in
  let ql_end = qk_k / 2 in
  let qh_end = ql_end + (qk_k / 4) in
  let sc_end = qh_end + (qk_k / 16) in
  let ql = Nx.reshape (Array.append g [| 2; 2; 32 |]) (bytes b 0 ql_end) in
  let qh = Nx.reshape (Array.append g [| 2; 32 |]) (bytes b ql_end qh_end) in
  let low =
    Nx.reshape
      (Array.append g [| 2; 4; 32 |])
      (Nx.moveaxis (-1) (-3) (Nx.cast Nx.uint8 (Nx.bitcast Nx.uint4 ql)))
  in
  let k = Nx.scalar_like qh in
  let high = pick 4 (fun j -> Nx.bitwise_and (Nx.rshift qh (2 * j)) (k 3)) in
  let q =
    Nx.sub_s (Nx.cast Nx.float32 (Nx.bitwise_or low (Nx.lshift high 4))) 32.
  in
  let d = Nx.mul (half b sc_end) (int8 (bytes b qh_end sc_end)) in
  rows g ~values:qk_k
    (Nx.mul (per g 16 d) (Nx.reshape (Array.append g [| 16; 16 |]) q))

(* [values w] is the weight [w] at float32. *)
let values = function
  | Mxfp4 { codes; scales } -> mxfp4_values codes scales
  | Mxfp4_blocks { blocks } -> mxfp4_blocks_values blocks
  | Q8_0 { blocks } -> q8_0_values blocks
  | Q4_K { blocks } -> q4_k_values blocks
  | Q6_K { blocks } -> q6_k_values blocks

(* A cut along [k] *)

(* The window of each device of [t]'s placement along [t]'s last axis. *)
let last_windows t =
  let s = Nx.shape t in
  let r = Array.length s in
  Array.map
    (fun (_, (w : Nx_array.Move.range array)) -> w.(r - 1))
    (Nx.Prim.pieces (Nx.placement t) s)

(* Whether each device's window of [t], whose last axis holds blocks of
   [bytes] elements, starts and ends at blocks' boundaries. *)
let at_blocks bytes t =
  Array.for_all
    (fun (x : Nx_array.Move.range) ->
      x.count = 0
      || (x.start mod bytes = 0 && (x.start + x.count) mod bytes = 0))
    (last_windows t)

(* Whether each device holds all of [t]'s last axis. *)
let whole_along t =
  let k = (Nx.shape t).(Nx.ndim t - 1) in
  Array.for_all (fun (x : Nx_array.Move.range) -> x.count = k) (last_windows t)

let some t = Array.fold_left ( * ) 1 (Nx.shape t) > 0

(* Whether each device that holds MXFP4 codes holds the scales of the
   blocks its codes' window reaches. *)
let scales_cover codes scales =
  let r = Nx.ndim codes in
  let held = Nx.Prim.pieces (Nx.placement scales) (Nx.shape scales) in
  Array.for_all
    (fun (here, (w : Nx_array.Move.range array)) ->
      let c = w.(r - 1) in
      c.count = 0
      || Array.exists
           (fun (there, (v : Nx_array.Move.range array)) ->
             let g = v.(r - 1) in
             Nx.Placement.equal here there
             && g.start <= c.start / mxfp4_bytes
             && (c.start + c.count + mxfp4_bytes - 1) / mxfp4_bytes
                <= g.start + g.count)
           held)
    (Nx.Prim.pieces (Nx.placement codes) (Nx.shape codes))

(* Raises naming [fn] where a cut of [w] along [k] splits a block, or puts
   MXFP4 codes on a device without their blocks' scales, which each device
   would read whole: before any block is decoded, at the call that reads
   them. An MXFP4 weight of one block whose scales lie whole decodes its
   codes where they lie. *)
let check_k fn w =
  let splits =
    match w with
    | Mxfp4 { codes; scales } ->
        let one_block = (Nx.shape scales).(Nx.ndim scales - 1) = 1 in
        some codes
        && ((not (scales_cover codes scales))
           || (not (at_blocks mxfp4_bytes codes))
              && not (one_block && whole_along scales))
    | Mxfp4_blocks { blocks } ->
        some blocks && not (at_blocks mxfp4_block_bytes blocks)
    | Q8_0 { blocks } -> some blocks && not (at_blocks q8_0_bytes blocks)
    | Q4_K { blocks } -> some blocks && not (at_blocks q4_k_bytes blocks)
    | Q6_K { blocks } -> some blocks && not (at_blocks q6_k_bytes blocks)
  in
  if splits then
    invalid_arg
      (strf
         "%s: the weight lies cut along k so that a device lacks part of a \
          block or its scale; Nx.place (Placement.on s) w"
         fn)

(* Products *)

(* [w] at [dt], decoded, its cut along [k] checked by the caller. *)
let decoded dt w = Nx.cast dt (values w)

let dequant dt w =
  check_k "Nx_quant.dequant" w;
  decoded dt w

(* Batch axes, aligned on the right and broadcast as Nx.matmul's. *)
let broadcast fn a b =
  let la = Array.length a and lb = Array.length b in
  let l = max la lb in
  for i = 0 to l - 1 do
    let da = if i < l - la then 1 else a.(i - l + la) in
    let db = if i < l - lb then 1 else b.(i - l + lb) in
    if not (da = db || da = 1 || db = 1) then
      invalid_arg
        (strf "%s: batch axes %s and %s do not broadcast" fn (pp_shape a)
           (pp_shape b))
  done

(* [check_input ws xs] checks the shapes of a product of the weight [ws] and the
   input [xs]. *)
let check_input ws xs =
  let fn = "Nx_quant.apply" in
  let wr = Array.length ws and xr = Array.length xs in
  let k = ws.(wr - 1) in
  if xr = 0 then invalid_arg (strf "%s: x must have at least one axis" fn);
  if xs.(xr - 1) <> k then
    invalid_arg
      (strf "%s: x's last axis is %d, the weight's inputs are %d" fn
         xs.(xr - 1)
         k);
  broadcast fn
    (if xr = 1 then [||] else Array.sub xs 0 (xr - 2))
    (Array.sub ws 0 (wr - 2))

(* Whether every value of [w]'s format is exact at [dt]. *)
let exact (type b) (dt : (float, b) Nx.dtype) w =
  match (dt, w) with
  | (Nx.Dtype.Float32 | Nx.Dtype.Float64), _ -> true
  | Nx.Dtype.Bfloat16, (Mxfp4 _ | Mxfp4_blocks _) -> true
  | _ -> false

(* Whether [x]'s dtype sums at float32 with float32 operands. *)
let at_float32 (type b d) (x : (float, b, d) Nx.t) =
  match Nx.dtype x with
  | Nx.Dtype.Float32 | Nx.Dtype.Bfloat16 | Nx.Dtype.Float16 -> true
  | _ -> false

(* A weight read where it lies: its elements, of [[| …; n |]] then [inner]
   once reshaped, [k] as one axis or as its blocks and their values where a
   block's scale lies between its values, and their blocks' scales, of
   [[| …; n |]] then [blocks]. *)
type 'd stored =
  | Stored : {
      elements : ('v, 's, 'd) Nx.t;
      inner : int list;
      scales : 'd Nx.Prim.scale;
      blocks : int list;
    }
      -> 'd stored

(* [stored ws w] is [w], of shape [ws], read where it lies, if its format
   is: MXFP4's codes as float4 values and its scales as their groups'
   exponents, Q8_0's quants as int8 and its float16 scales as values. Each
   element times its scale is the value [dequant] gives, exact at
   float32. *)
let stored ws w =
  let k = ws.(Array.length ws - 1) in
  match w with
  | Mxfp4 { codes; scales } ->
      let elements = Nx.bitcast Nx.float4_e2m1fn codes in
      Some
        (Stored
           {
             elements;
             inner = [ k ];
             scales = Exponent scales;
             blocks = [ k / mxfp4_group ];
           })
  | Q8_0 { blocks } ->
      let b = split ~bytes:q8_0_bytes blocks and nb = k / qk8_0 in
      let elements = Nx.bitcast Nx.int8 (bytes b half_bytes q8_0_bytes) in
      let d = Nx.bitcast Nx.float16 (bytes b 0 half_bytes) in
      Some
        (Stored
           {
             elements;
             inner = [ nb; qk8_0 ];
             scales = Value d;
             blocks = [ nb; 1 ];
           })
  | Mxfp4_blocks _ | Q4_K _ | Q6_K _ -> None

(* [scaled ~by x ws s] is the product of [x], [[| …; m; k |]] or [[| k |]],
   with a weight of shape [ws], [[| …; n; k |]], stored as [s], in one
   contraction of [x]'s [k] with the elements' [inner] axes, each element
   of the weight entering the sum as its value. Its axes are planned as
   [Nx.matmul] plans them: a leading axis is a batch axis where both
   operands hold the result's extent, else a free axis of the operand that
   holds it, from which a unit axis of the other drops. It sums in the
   order, and gives the bits, of [x] by the decoded weight transposed:
   nx.cpu numbers the products of several contracted axes as of one. *)
let scaled ~by x ws (Stored { elements; inner; scales; blocks }) =
  let xs = Nx.shape x in
  let wr = Array.length ws and xr = Array.length xs in
  let p = List.length inner in
  let n = ws.(wr - 2) in
  let lead_w = Array.sub ws 0 (wr - 2) in
  let lead_x = if xr <= 2 then [||] else Array.sub xs 0 (xr - 2) in
  let lead = Nx.broadcast_shapes [ lead_x; lead_w ] in
  let l = Array.length lead in
  let holds s i =
    let j = i - (l - Array.length s) in
    j >= 0 && s.(j) = lead.(i)
  in
  let leads p = List.filter p (List.init l Fun.id) in
  let x_lead = leads (holds lead_x) in
  let w_lead = leads (fun i -> holds lead_w i || not (holds lead_x i)) in
  let both = List.filter (fun i -> List.mem i w_lead) x_lead in
  let only ax = List.filter (fun i -> not (List.mem i both)) ax in
  let index ax i = Option.get (List.find_index (Int.equal i) ax) in
  let extents ax last = Array.of_list (List.map (Array.get lead) ax @ last) in
  let m = if xr = 1 then [] else [ xs.(xr - 2) ] in
  let a = Nx.reshape (extents x_lead (m @ inner)) x in
  let b = Nx.reshape (extents w_lead (n :: inner)) elements in
  let s = extents w_lead (n :: blocks) in
  let b_scale : _ Nx.Prim.scale =
    match scales with
    | Exponent e -> Exponent (Nx.reshape s e)
    | Value v -> Value (Nx.reshape s v)
  in
  let batch =
    Iarray.of_list (List.map (fun i -> (index x_lead i, index w_lead i)) both)
  in
  let ra = Nx.ndim a and rb = Nx.ndim b in
  let contracting =
    Iarray.of_list (List.init p (fun i -> (ra - p + i, rb - p + i)))
  in
  let y =
    Nx.Prim.eval ~by
      (Contract
         {
           batch;
           contracting;
           acc = Nx.Dtype.Any Nx.float32;
           out = Nx.dtype x;
           a;
           b;
           init = None;
           a_scale = None;
           b_scale = Some b_scale;
           segments = None;
         })
  in
  (* The kernel's axes are the batch axes, x's free axes and row, then w's
     free axes and column, the result's the leading axes in order, the row and
     the column: the leading axes are named by their number, the row [l] and
     the column [l + 1]. *)
  let row = if xr = 1 then [] else [ l ] in
  let kernel = both @ only x_lead @ row @ only w_lead @ [ l + 1 ] in
  let result = List.init l Fun.id @ row @ [ l + 1 ] in
  if kernel = result then y
  else Nx.transpose ~axes:(List.map (index kernel) result) y

let apply w x =
  let ws = shape w in
  check_input ws (Nx.shape x);
  let dt = Nx.dtype x in
  let by = "Nx_quant.apply" in
  (* An MXFP4 weight read where it lies names its scales as the contraction
     reads them. *)
  (match w with
  | Mxfp4 _ when at_float32 x -> ()
  | _ -> check_k by w);
  match if at_float32 x then stored ws w else None with
  | Some s ->
      if exact dt w then scaled ~by x ws s
      else Nx.cast dt (scaled ~by (Nx.cast Nx.float32 x) ws s)
  | None ->
      if exact dt w then Nx.matmul x (Nx.matrix_transpose (decoded dt w))
      else
        Nx.cast dt
          (Nx.matmul (Nx.cast Nx.float32 x) (Nx.matrix_transpose (values w)))

(* Quantising. Q8_0 as ggml's quantize_row_q8_0_ref stores a block of 32
   float32 values [v]: [d = amax / 127] for the block's greatest magnitude
   [amax], stored at float16, and quants [round (v * id)], halves away from
   zero, for [id = 1 / d] at float32, [0] where [d] is. A product that is
   not finite, where [id] is infinite or [v] or [d] is not finite, has the
   quant [0], the low byte x86's conversion gives it in ggml. Nx.max keeps
   a NaN, so a NaN makes its block's scale NaN. *)
let quantize_q8_0 x =
  let s = Nx.shape x in
  let r = Array.length s in
  if r < 2 || s.(r - 1) mod qk8_0 <> 0 then
    invalid_arg
      (strf
         "Nx_quant.quantize_q8_0: x must have shape [...; n; k] with k a \
          multiple of 32, got %s"
         (pp_shape s));
  let g = Array.copy s in
  g.(r - 1) <- s.(r - 1) / qk8_0;
  let v = Nx.reshape (Array.append g [| qk8_0 |]) (Nx.cast Nx.float32 x) in
  let d = Nx.div_s (Nx.max ~axes:[ -1 ] ~keepdims:true (Nx.abs v)) 127. in
  let id = Nx.where (Nx.equal_s d 0.) d (Nx.recip d) in
  let q = Nx.round (Nx.mul v id) in
  let q = Nx.where (Nx.isfinite q) q (Nx.scalar_like q 0.) in
  let q = Nx.bitcast Nx.uint8 (Nx.cast Nx.int8 q) in
  let d =
    Nx.reshape
      (Array.append g [| half_bytes |])
      (Nx.bitcast Nx.uint8 (Nx.cast Nx.float16 d))
  in
  let blocks = Nx.concatenate ~axis:(-1) [ d; q ] in
  Q8_0 { blocks = rows g ~values:q8_0_bytes blocks }

(* [map f w] is [w] with [f] of each part. *)
let map f = function
  | Mxfp4 { codes; scales } -> Mxfp4 { codes = f codes; scales = f scales }
  | Mxfp4_blocks { blocks } -> Mxfp4_blocks { blocks = f blocks }
  | Q8_0 { blocks } -> Q8_0 { blocks = f blocks }
  | Q4_K { blocks } -> Q4_K { blocks = f blocks }
  | Q6_K { blocks } -> Q6_K { blocks = f blocks }

let take ~axis p w =
  let r = Array.length (shape w) in
  let a = if axis < 0 then axis + r else axis in
  if a < 0 || a >= r then
    invalid_arg
      (strf "Nx_quant.take: axis %d out of bounds for a weight of %d axes" axis
         r);
  if a = r - 1 then
    invalid_arg
      (strf
         "Nx_quant.take: axis %d is the weight's last axis, which a gather \
          would cut across blocks"
         axis);
  map (Nx.take ~axis:a p) w

(* Experts *)

(* [check_experts ws ids xs] checks the shapes of the experts [ws], the ids
   [ids] and the input [xs] of an expert product. *)
let check_experts ws ids xs =
  let fn = "Nx_quant.apply_experts" in
  let wr = Array.length ws and l = Array.length ids in
  if wr <> 3 then
    invalid_arg
      (strf "%s: the weight must have shape [e; n; k], got %s" fn (pp_shape ws));
  if Array.length xs <> l + 1 then
    invalid_arg
      (strf "%s: x must have shape [...; k] with one axis per id's, got %s" fn
         (pp_shape xs));
  if xs.(l) <> ws.(2) then
    invalid_arg
      (strf "%s: x's last axis is %d, the weight's inputs are %d" fn xs.(l)
         ws.(2));
  Array.iteri
    (fun i d ->
      if d <> 1 && d <> ids.(i) then
        invalid_arg
          (strf "%s: x's leading axes %s do not broadcast to the ids' %s" fn
             (pp_shape (Array.sub xs 0 l))
             (pp_shape ids)))
    (Array.sub xs 0 l)

(* An expert product is one contraction whose rows are segmented by expert:
   the ids, flattened, are sorted stably, [x]'s rows gathered in their order,
   and each row multiplies the matrix of [b] its group names, so that an id
   out of range falls in no group and its row has no products, [+0]. The
   result's rows are gathered back to the ids' order. A weight split along
   [n], the result's free columns, is computed by each device for its columns
   over every row, in the whole's order. A weight cut along its experts is
   computed by each device for the rows of its own experts, [+0] in the
   others, and one cut along [k] by each device for every row over its
   window of [k]: the devices' sums close by partials. *)

(* The positions sorted by id, the permutation that sorts them, their count
   and the ids' shape. *)
type 'd routed = {
  sorted : 'd Nx.int64_t;
  order : 'd Nx.int64_t;
  p : int;
  lead : int array;
}

let route ids =
  let lead = Nx.shape ids in
  let p = Array.fold_left ( * ) 1 lead in
  let sorted, order = Nx.sort (Nx.reshape [| p |] ids) in
  { sorted; order; p; lead }

(* Group [g] of the experts [[| e; n; k |]] is expert [g]'s rows: the ids
   below [g] counted. *)
let by_expert r e =
  let at = Nx.beside r.sorted in
  Nx.searchsorted ~side:`Left r.sorted (Nx.arange_at at Nx.int64 0 (e + 1) 1)

(* The experts the sorted ids name, in range, each once: [min p e] slots, slot
   [j] the [j]-th distinct id, and [e], which gathers zero bytes, where fewer
   ids are distinct. Group [j] is slot [j]'s rows: offset [j] is the position
   of the [j]-th distinct id's first row, or that past the ids below [e]
   where there is none, so that the ids out of range fall in no group. The
   slots' count, from the shapes alone, and an id's slot, its rank among the
   distinct ids, do not depend on the other ids. *)
let slots r e =
  let s = min r.p e in
  let sorted = r.sorted in
  let previous =
    Nx.slice [ Nx.R (0, r.p) ] (Nx.pad [| (1, 0) |] (-1L) sorted)
  in
  let first =
    Nx.logical_and
      (Nx.not_equal sorted previous)
      (Nx.logical_and (Nx.greater_equal_s sorted 0L)
         (Nx.less_s sorted (Int64.of_int e)))
  in
  let rank = Nx.cumsum (Nx.cast Nx.int64 first) in
  let here = Nx.beside rank in
  let at =
    Nx.searchsorted ~side:`Left rank (Nx.arange_at here Nx.int64 1 (s + 2) 1)
  in
  let past = Nx.scalar_like sorted (Int64.of_int e) in
  let below = Nx.searchsorted ~side:`Left sorted past in
  let offsets = Nx.minimum at below in
  let starts = Nx.slice [ Nx.R (0, s) ] at in
  let ids =
    Nx.where
      (Nx.less_s starts (Int64.of_int r.p))
      (Nx.take starts sorted) past
  in
  (ids, offsets)

(* [segmented r x b scale offsets] is the product of [x]'s rows with the
   matrices [b], [[| g; n; k |]], scaled by [scale], cut by [offsets], summed
   at [x]'s dtype or at float32 where that dtype does not accumulate. *)
let segmented (type b) r (x : (float, b, _) Nx.t) b scale offsets =
  let ws = Nx.shape b in
  let n = ws.(1) and k = ws.(2) in
  (* Each position's row of [x], whose leading axes broadcast to the ids'. *)
  let xs = Array.sub (Nx.shape x) 0 (Array.length r.lead) in
  let m = Array.fold_left ( * ) 1 xs in
  let here = Nx.beside x in
  let positions = Nx.reshape xs (Nx.arange_at here Nx.int64 0 m 1) in
  let source = Nx.reshape [| r.p |] (Nx.broadcast_to r.lead positions) in
  let rows =
    Nx.take ~axis:0 (Nx.take r.order source) (Nx.reshape [| m; k |] x)
  in
  let acc : Nx.Dtype.any =
    match Nx.dtype x with
    | Nx.Dtype.Float64 -> Any Nx.float64
    | _ -> Any Nx.float32
  in
  let batch = Iarray.of_list [] and contracting = Iarray.of_list [ (1, 2) ] in
  let y =
    Nx.Prim.eval ~by:"Nx_quant.apply_experts"
      (Contract
         {
           batch;
           contracting;
           acc;
           out = Nx.dtype x;
           a = rows;
           b;
           init = None;
           a_scale = None;
           b_scale = scale;
           segments = Some (Rows { offsets; a_axis = 0; b_axis = 0 });
         })
  in
  let back = Nx.argsort r.order in
  Nx.reshape (Array.append r.lead [| n |]) (Nx.take ~axis:0 back y)

(* Whether a part of [w] is cut along its first axis, a stack's experts: its
   placement is not that axis made whole. *)
let experts_cut w =
  let cut t =
    let p = Nx.placement t in
    let whole = Nx.Placement.(with_leading_axis (without_leading_axis p)) in
    not (Nx.Placement.equal p whole)
  in
  match w with
  | Mxfp4 { codes; scales } -> cut codes || cut scales
  | Mxfp4_blocks { blocks } | Q8_0 { blocks } | Q4_K { blocks }
  | Q6_K { blocks } ->
      cut blocks

(* An MXFP4 weight summed at float32 enters as its codes read as float4 values
   and its scales as their blocks' exponents, each element exact at float32,
   read where it lies, a group per expert. Any other weight, and one summed at
   float64, where a value past float32's range would be finite, enters as its
   values at [x]'s dtype: cut along its experts, all of them, each device
   decoding its own, a group per expert; else the experts the ids name alone,
   a group per slot. A segmented contraction's association does not depend
   on its offsets (Nx_kernel.S.contract), so a row's bits depend on the
   shapes, its row and its expert. *)
let experts ~ids w x =
  let r = route ids and e = (shape w).(0) in
  match w with
  | Mxfp4 { codes; scales } when at_float32 x ->
      let b = Nx.reshape (shape w) (Nx.bitcast Nx.float4_e2m1fn codes) in
      segmented r x b (Some (Nx.Prim.Exponent scales)) (by_expert r e)
  | _ when experts_cut w ->
      segmented r x (decoded (Nx.dtype x) w) None (by_expert r e)
  | _ ->
      let named, offsets = slots r e in
      segmented r x
        (decoded (Nx.dtype x) (take ~axis:0 named w))
        None offsets

let apply_experts ~ids w x =
  check_experts (shape w) (Nx.shape ids) (Nx.shape x);
  let dt = Nx.dtype x in
  (* An MXFP4 weight summed at float32 is read where it lies and names its
     scales as the contraction reads them; any other decodes the blocks the
     positions name, none for no position. *)
  let in_place =
    match w with
    | Mxfp4 _ -> not (Nx.Dtype.equal dt Nx.float64)
    | Mxfp4_blocks _ | Q8_0 _ | Q4_K _ | Q6_K _ -> false
  in
  if (not in_place) && some ids then check_k "Nx_quant.apply_experts" w;
  if exact dt w then experts ~ids w x
  else Nx.cast dt (experts ~ids w (Nx.cast Nx.float32 x))

(* Structure *)

let walk c w =
  let open Nx.Ptree.Walk in
  let fn = "Nx_quant.walk" in
  let blocks format bytes v =
    let blocks = field c "blocks" tensor v in
    check_blocks fn format bytes blocks;
    blocks
  in
  match w with
  | Mxfp4 { codes; scales } ->
      case c "mxfp4";
      let codes = field c "codes" tensor codes in
      let scales = field c "scales" tensor scales in
      check_mxfp4 fn codes scales;
      Mxfp4 { codes; scales }
  | Mxfp4_blocks { blocks = b } ->
      case c "mxfp4_blocks";
      Mxfp4_blocks { blocks = blocks "MXFP4" mxfp4_block_bytes b }
  | Q8_0 { blocks = b } ->
      case c "q8_0";
      Q8_0 { blocks = blocks "Q8_0" q8_0_bytes b }
  | Q4_K { blocks = b } ->
      case c "q4_k";
      Q4_K { blocks = blocks "Q4_K" q4_k_bytes b }
  | Q6_K { blocks = b } ->
      case c "q6_k";
      Q6_K { blocks = blocks "Q6_K" q6_k_bytes b }

type 'd weight = 'd t

module Structure = struct
  type ('a, 'd) t = 'd weight

  let walk = walk
end

let place p w = Nx.Ptree.place (module Structure) p w
