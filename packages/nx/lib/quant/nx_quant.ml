(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Mxfp4 of { codes : Nx.uint4_t; scales : (int, Nx.uint8_elt) Nx.t }
  | Q8_0 of { blocks : (int, Nx.uint8_elt) Nx.t }
  | Q4_K of { blocks : (int, Nx.uint8_elt) Nx.t }
  | Q6_K of { blocks : (int, Nx.uint8_elt) Nx.t }

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

(* Construction. Checks read shapes and placement, never bytes. *)

(* [cut fn ~axis ~values ~blocks pieces] raises: splitting a weight along its
   last axis, [axis], in [pieces] cuts its blocks of [values] values. *)
let cut fn ~axis ~values ~blocks pieces =
  invalid_arg
    (strf
       "%s: splitting the weight along axis %d in %d cuts a block of %d values \
        (%d blocks)"
       fn axis pieces values blocks)

(* [check_split fn ~bytes ~values part] raises unless every device's window of
   the rows of blocks [part] starts and stops at a block of [bytes] bytes and
   [values] values. *)
let check_split fn ~bytes ~values part =
  let c = Nx.shape part in
  let r = Array.length c in
  let p = Nx.placement part in
  List.iter
    (fun d ->
      let lo, hi = (Nx.Placement.window p c d).(r - 1) in
      if lo mod bytes <> 0 || hi mod bytes <> 0 then
        cut fn ~axis:(r - 1) ~values
          ~blocks:(c.(r - 1) / bytes)
          (c.(r - 1) / (hi - lo)))
    (Nx.Placement.devices p)

(* [check_groups fn codes] raises unless every device holds whole groups of the
   MXFP4 [codes], [[| ...; n; k / 32; 2; 16 |]]. *)
let check_groups fn codes =
  let c = Nx.shape codes in
  let r = Array.length c in
  let p = Nx.placement codes in
  List.iter
    (fun d ->
      let w = Nx.Placement.window p c d in
      let held a = snd w.(a) - fst w.(a) in
      let group = held (r - 2) * held (r - 1) in
      if group <> mxfp4_group then
        cut fn ~axis:(r - 3) ~values:mxfp4_group
          ~blocks:c.(r - 3)
          (mxfp4_group / group))
    (Nx.Placement.devices p)

let check_scales fn scales expected =
  if Nx.shape scales <> expected then
    invalid_arg
      (strf "%s: scales must have shape %s, one per 32 values, got %s" fn
         (pp_shape expected)
         (pp_shape (Nx.shape scales)))

let check_mxfp4 fn codes scales =
  let c = Nx.shape codes in
  let r = Array.length c in
  if r < 4 || c.(r - 2) <> 2 || c.(r - 1) <> 16 then
    invalid_arg
      (strf "%s: codes must have shape [...; n; k / 32; 2; 16], got %s" fn
         (pp_shape c));
  check_scales fn scales (Array.sub c 0 (r - 2));
  check_groups fn codes

(* [check_blocks fn format bytes values blocks] checks that [blocks] is rows of
   whole blocks of [bytes] bytes, each of [values] values. *)
let check_blocks fn format bytes values blocks =
  let s = Nx.shape blocks in
  let r = Array.length s in
  if r < 2 || s.(r - 1) mod bytes <> 0 then
    invalid_arg
      (strf
         "%s: blocks must have shape [...; n; b * %d], rows of whole %s blocks \
          of %d bytes, got %s"
         fn bytes format bytes (pp_shape s));
  check_split fn ~bytes ~values blocks

(* [bytes b lo hi] is the bytes \[[lo];[hi]) of each block of [b]. *)
let bytes b lo hi =
  let r = Nx.ndim b in
  Nx.shrink
    (Array.mapi
       (fun i d -> if i = r - 1 then (lo, hi) else (0, d))
       (Nx.shape b))
    b

(* Both files hold a group's codes as 16 bytes, two to a byte, the low nibble
   first. A checkpoint's byte [i] holds values [2 i] and [2 i + 1], in order, so
   its nibbles are the codes reshaped. A GGUF block's byte [j] holds values [j]
   and [j + 16], so its nibbles are the codes with their last two axes
   swapped. *)

let mxfp4 ~scales b =
  let fn = "Nx_quant.mxfp4" in
  let s = Nx.shape b in
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
  check_scales fn scales groups;
  check_split fn ~bytes:mxfp4_bytes ~values:mxfp4_group b;
  let codes =
    Nx.reshape (Array.append groups [| 2; 16 |]) (Nx.bitcast Nx.uint4 b)
  in
  Mxfp4 { codes; scales }

let mxfp4_blocks b =
  let fn = "Nx_quant.mxfp4_blocks" in
  check_blocks fn "MXFP4" mxfp4_block_bytes mxfp4_group b;
  let s = Nx.shape b in
  let r = Array.length s in
  let groups =
    Array.append (Array.sub s 0 (r - 1)) [| s.(r - 1) / mxfp4_block_bytes |]
  in
  let b = Nx.reshape (Array.append groups [| mxfp4_block_bytes |]) b in
  let scales = Nx.reshape groups (bytes b 0 1) in
  let codes =
    Nx.swapaxes (-1) (-2) (Nx.bitcast Nx.uint4 (bytes b 1 mxfp4_block_bytes))
  in
  Mxfp4 { codes; scales }

let q8_0 blocks =
  check_blocks "Nx_quant.q8_0" "Q8_0" q8_0_bytes qk8_0 blocks;
  Q8_0 { blocks }

let q4_k blocks =
  check_blocks "Nx_quant.q4_k" "Q4_K" q4_k_bytes qk_k blocks;
  Q4_K { blocks }

let q6_k blocks =
  check_blocks "Nx_quant.q6_k" "Q6_K" q6_k_bytes qk_k blocks;
  Q6_K { blocks }

(* A weight's last axis counts the blocks of one of its parts: MXFP4's scales,
   one per group, or a GGUF format's bytes. *)
let shape w =
  let part, per, values =
    match w with
    | Mxfp4 { scales; _ } -> (scales, 1, mxfp4_group)
    | Q8_0 { blocks } -> (blocks, q8_0_bytes, qk8_0)
    | Q4_K { blocks } -> (blocks, q4_k_bytes, qk_k)
    | Q6_K { blocks } -> (blocks, q6_k_bytes, qk_k)
  in
  let s = Array.copy (Nx.shape part) in
  let r = Array.length s in
  s.(r - 1) <- s.(r - 1) / per * values;
  s

(* Structure *)

let walk c w =
  let open Nx.Ptree.Walk in
  let blocks format bytes values v =
    case c format;
    let blocks = field c "blocks" tensor v in
    check_blocks "Nx_quant.ptree"
      (String.uppercase_ascii format)
      bytes values blocks;
    blocks
  in
  match w with
  | Mxfp4 { codes; scales } ->
      case c "mxfp4";
      let codes = field c "codes" tensor codes in
      let scales = field c "scales" tensor scales in
      check_mxfp4 "Nx_quant.ptree" codes scales;
      Mxfp4 { codes; scales }
  | Q8_0 { blocks = b } -> Q8_0 { blocks = blocks "q8_0" q8_0_bytes qk8_0 b }
  | Q4_K { blocks = b } -> Q4_K { blocks = blocks "q4_k" q4_k_bytes qk_k b }
  | Q6_K { blocks = b } -> Q6_K { blocks = blocks "q6_k" q6_k_bytes qk_k b }

type weight = t

module Structure = struct
  type _ t = weight

  let walk = walk
end

let ptree = Nx.Ptree.instantiate (module Structure)

(* Decoding. An e2m1 code is a sign bit, two exponent bits and a mantissa bit,
   the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6. A scale byte is an e8m0
   exponent, 2^(s - 127), with 255 a NaN. Every value, scaled, is exact at
   float32 barring overflow, and the product is rounded once.

   The values are assembled as float32 bits with integer operations, so a
   compiled product reads the code bytes themselves: a table lookup's int64
   index would be stored between gathering the experts and multiplying. *)

(* [code_bits q] is the float32 bits of the e2m1 codes [q], at most 15. An
   exponent of 0 is 0 or 0.5; another, [e], is 2^(e - 1) (1 + m / 2). *)
let code_bits q =
  let k = Nx.scalar_like q in
  let e = Nx.bitwise_and (Nx.rshift q 1) (k 3l)
  and m = Nx.bitwise_and q (k 1l) in
  let sign = Nx.lshift (Nx.bitwise_and q (k 8l)) 28 in
  let magnitude =
    Nx.where (Nx.equal_s e 0l)
      (Nx.mul_s m (Int32.shift_left 126l 23))
      (Nx.bitwise_or (Nx.lshift (Nx.add_s e 126l) 23) (Nx.lshift m 22))
  in
  Nx.bitwise_or sign magnitude

(* [scale_bits s] is the float32 bits of the e8m0 scales [s]: 2^-127, a
   subnormal, at 0, and NaN at 255. *)
let scale_bits s =
  let k = Nx.scalar_like s in
  Nx.where (Nx.equal_s s 0l) (k 0x00400000l)
    (Nx.where (Nx.equal_s s 255l) (k 0x7FC00000l) (Nx.lshift s 23))

(* [mxfp4_values codes scales] is the weight of [codes] and [scales] at float32,
   from their bits. Codes are decoded where they lie, so a view of a file's
   bytes is read in place. *)
let mxfp4_values codes scales =
  let v = Nx.bitcast Nx.float32 (code_bits (Nx.cast Nx.uint32 codes)) in
  let scale = Nx.bitcast Nx.float32 (scale_bits (Nx.cast Nx.uint32 scales)) in
  let groups = Nx.shape scales in
  let r = Array.length groups in
  let rows = Array.copy groups in
  rows.(r - 1) <- groups.(r - 1) * mxfp4_group;
  Nx.reshape rows (Nx.mul v (Nx.reshape (Array.append groups [| 1; 1 |]) scale))

(* [bfloat16 v] is the MXFP4 values [v] at bfloat16. A value has at most two
   significant bits, so its float32 bits end in 16 zeros, subnormal values
   included, and their high half is its bfloat16. *)
let bfloat16 v =
  Nx.bitcast Nx.bfloat16
    (Nx.cast Nx.uint16 (Nx.rshift (Nx.bitcast Nx.uint32 v) 16))

(* The GGUF formats, decoded as ggml's dequantize_row_q8_0, _q4_K and _q6_K
   decode them: the scales at float32, each product left to right. Each block's
   fields are views of its bytes, so a compiled product reads the bytes
   themselves. *)

(* [split ~bytes t] is the rows [t] [[| ...; n; b * bytes |]] as their blocks,
   [[| ...; n; b; bytes |]]. *)
let split ~bytes t =
  let s = Nx.shape t in
  let r = Array.length s in
  Nx.reshape
    (Array.append (Array.sub s 0 (r - 1)) [| s.(r - 1) / bytes; bytes |])
    (Nx.contiguous t)

(* [half b at] is the little-endian float16 at byte [at] of each block of [b] at
   float32, [[| ...; b; 1 |]]. *)
let half b at =
  let byte i = Nx.cast Nx.uint16 (bytes b i (i + 1)) in
  Nx.cast Nx.float32
    (Nx.bitcast Nx.float16
       (Nx.bitwise_or (byte at) (Nx.lshift (byte (at + 1)) 8)))

(* [unpack ~bits ~axis n t] is the [n] fields of [bits] bits of each byte of
   [t], lowest first, along a new axis at [axis] from the end. Each element
   picks its field by its index along that axis, so a compiled product unpacks
   each byte where it reads it. *)
let unpack ~bits ~axis n t =
  let s = Nx.shape t in
  let at = Array.length s + 1 + axis in
  let around mid =
    Array.concat [ Array.sub s 0 at; mid; Array.sub s at (-axis - 1) ]
  in
  let t = Nx.broadcast_to (around [| n |]) (Nx.reshape (around [| 1 |]) t) in
  let index =
    Nx.reshape
      (Array.init (-axis) (fun i -> if i = 0 then n else 1))
      (Nx.arange Nx.int32 0 n 1)
  in
  let mask = Nx.scalar_like t ((1 lsl bits) - 1) in
  let field i = Nx.bitwise_and (Nx.rshift t (bits * i)) mask in
  let rec pick i acc =
    if i = n then acc
    else
      pick (i + 1) (Nx.where (Nx.equal_s index (Int32.of_int i)) (field i) acc)
  in
  pick 1 (field 0)

(* [int8 t] is the bytes [t] read as int8, at float32. *)
let int8 t = Nx.cast Nx.float32 (Nx.bitcast Nx.int8 t)

(* [rows ~bytes ~values blocks v] is the values [v] of the blocks of [blocks],
   of [bytes] bytes and [values] values each, as rows: [[| ...; n; b; ... |]]
   becomes [[| ...; n; b * values |]]. *)
let rows ~bytes ~values blocks v =
  let s = Array.copy (Nx.shape blocks) in
  let r = Array.length s in
  s.(r - 1) <- s.(r - 1) / bytes * values;
  Nx.reshape s v

(* Q8_0: a float16 scale [d] and 32 int8 quants [q], each value [d * q]. *)
let q8_0_values blocks =
  let b = split ~bytes:q8_0_bytes blocks in
  rows ~bytes:q8_0_bytes ~values:qk8_0 blocks
    (Nx.mul (int8 (bytes b half_bytes q8_0_bytes)) (half b 0))

(* Q4_K: float16 [d] and [dmin], then 8 sub-blocks of 32 values, each with a
   6-bit scale [sc] and min [m] packed in 12 bytes, of 4-bit quants [q]: a value
   is [d * sc * q - dmin * m]. Sub-blocks 0 to 3 take the low 6 bits of bytes 0
   to 3 and 4 to 7; sub-blocks 4 to 7 the nibbles of bytes 8 to 11 under the top
   2 bits of bytes 0 to 3 and 4 to 7. Each 32 bytes of quants hold two
   sub-blocks, the low nibbles first. *)
let q4_k_values blocks =
  let b = split ~bytes:q4_k_bytes blocks in
  let s = Nx.shape b in
  let lead = Array.sub s 0 (Array.length s - 1) in
  let quants = (2 * half_bytes) + k_scale_size in
  let k = Nx.scalar_like b in
  let low6 t = Nx.bitwise_and t (k 63) in
  let top2 t = Nx.lshift (Nx.rshift t 6) 4 in
  let at i =
    bytes b ((2 * half_bytes) + (4 * i)) ((2 * half_bytes) + (4 * i) + 4)
  in
  let s0 = at 0 and s1 = at 1 and s2 = at 2 in
  let join first last =
    Nx.cast Nx.float32 (Nx.concatenate ~axis:(-1) [ first; last ])
  in
  let sc =
    join (low6 s0) (Nx.bitwise_or (Nx.bitwise_and s2 (k 15)) (top2 s0))
  in
  let m = join (low6 s1) (Nx.bitwise_or (Nx.rshift s2 4) (top2 s1)) in
  let per t = Nx.reshape (Array.append lead [| 8; 1 |]) t in
  let qs =
    Nx.reshape (Array.append lead [| 4; 32 |]) (bytes b quants q4_k_bytes)
  in
  let q =
    Nx.reshape (Array.append lead [| 8; 32 |]) (unpack ~bits:4 ~axis:(-2) 2 qs)
  in
  rows ~bytes:q4_k_bytes ~values:qk_k blocks
    (Nx.sub
       (Nx.mul (per (Nx.mul (half b 0) sc)) (Nx.cast Nx.float32 q))
       (per (Nx.mul (half b half_bytes) m)))

(* Q6_K: 16 sub-blocks of 16 values, each with an int8 scale [sc], of 6-bit
   quants [q] offset by 32, and a float16 [d]: a value is [d * sc * (q - 32)].
   Each half of a block, 128 values, takes 64 bytes of low nibbles and 32 bytes
   of high bit pairs. Its quarters 0 and 1 take the low nibbles of the first and
   last 32 of the 64 bytes, quarters 2 and 3 their high nibbles, and quarter [j]
   the bit pair [j] of each of the 32 bytes. *)
let q6_k_values blocks =
  let b = split ~bytes:q6_k_bytes blocks in
  let s = Nx.shape b in
  let lead = Array.sub s 0 (Array.length s - 1) in
  let ql_end = qk_k / 2 in
  let qh_end = ql_end + (qk_k / 4) in
  let sc_end = qh_end + (qk_k / 16) in
  let ql = Nx.reshape (Array.append lead [| 2; 2; 32 |]) (bytes b 0 ql_end) in
  let qh = Nx.reshape (Array.append lead [| 2; 32 |]) (bytes b ql_end qh_end) in
  let low = unpack ~bits:4 ~axis:(-3) 2 ql in
  let high = unpack ~bits:2 ~axis:(-2) 4 qh in
  let q =
    Nx.sub_s
      (Nx.cast Nx.float32
         (Nx.bitwise_or
            (Nx.reshape (Array.append lead [| 2; 4; 32 |]) low)
            (Nx.lshift high 4)))
      32.
  in
  let d = Nx.mul (half b sc_end) (int8 (bytes b qh_end sc_end)) in
  rows ~bytes:q6_k_bytes ~values:qk_k blocks
    (Nx.mul
       (Nx.reshape (Array.append lead [| 16; 1 |]) d)
       (Nx.reshape (Array.append lead [| 16; 16 |]) q))

(* [values w] is the weight [w] at float32. *)
let values = function
  | Mxfp4 { codes; scales } -> mxfp4_values codes scales
  | Q8_0 { blocks } -> q8_0_values blocks
  | Q4_K { blocks } -> q4_k_values blocks
  | Q6_K { blocks } -> q6_k_values blocks

(* Products *)

let dequant (type b) (dt : (float, b) Nx.dtype) w : (float, b) Nx.t =
  match (dt, w) with
  | Nx.BFloat16, Mxfp4 _ -> bfloat16 (values w)
  | _ -> Nx.cast dt (values w)

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

(* With bfloat16 rows, an MXFP4 weight is its bfloat16 values widened, so the
   product multiplies bfloat16 operands, which a GPU's tensor cores take; other
   rows keep the float32 values, which take fewer operations to decode. Float64
   rows accumulate at float64. *)
let apply (type b) w (x : (float, b) Nx.t) : (float, b) Nx.t =
  check_input (shape w) (Nx.shape x);
  match Nx.dtype x with
  | Nx.Float64 -> Nx.matmul x (Nx.matrix_transpose (dequant Nx.float64 w))
  | dt ->
      let v =
        match (dt, w) with
        | Nx.BFloat16, Mxfp4 _ -> Nx.cast Nx.float32 (bfloat16 (values w))
        | _ -> values w
      in
      Nx.cast dt (Nx.matmul (Nx.cast Nx.float32 x) (Nx.matrix_transpose v))

let take ~axis ~indices w =
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
  Nx.Ptree.map ptree (fun _ t -> Nx.take ~axis:a ~indices t) w
