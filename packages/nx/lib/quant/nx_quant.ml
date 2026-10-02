(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Mxfp4 of {
      codes : (int, Nx.uint8_elt) Nx.t;
      scales : (int, Nx.uint8_elt) Nx.t;
    }
  | Q8_0 of { blocks : (int, Nx.uint8_elt) Nx.t }
  | Q4_K of { blocks : (int, Nx.uint8_elt) Nx.t }
  | Q6_K of { blocks : (int, Nx.uint8_elt) Nx.t }

let strf = Printf.sprintf

let pp_shape s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

(* Block layouts. MXFP4 stores a group of 32 values as 16 bytes of codes and a
   scale byte apart. The GGUF formats store a block's scales with its quants, as
   ggml-common.h's block structs lay them out. *)

let mxfp4_group = 32

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

(* [layout w] is the part of [w] that holds its quants, the bytes of one block
   of it along its last axis, and the values of that block. *)
let layout = function
  | Mxfp4 { codes; _ } -> (codes, mxfp4_group / 2, mxfp4_group)
  | Q8_0 { blocks } -> (blocks, q8_0_bytes, qk8_0)
  | Q4_K { blocks } -> (blocks, q4_k_bytes, qk_k)
  | Q6_K { blocks } -> (blocks, q6_k_bytes, qk_k)

(* [map f w] is [w] with [f] applied to each of its parts. *)
let map f = function
  | Mxfp4 { codes; scales } -> Mxfp4 { codes = f codes; scales = f scales }
  | Q8_0 { blocks } -> Q8_0 { blocks = f blocks }
  | Q4_K { blocks } -> Q4_K { blocks = f blocks }
  | Q6_K { blocks } -> Q6_K { blocks = f blocks }

(* Construction. Checks read shapes only, never bytes. *)

let check_mxfp4 fn codes scales =
  let c = Nx.shape codes in
  let r = Array.length c in
  if r < 2 || c.(r - 1) mod 16 <> 0 then
    invalid_arg
      (strf
         "%s: codes must have shape [...; n; k / 2] with k a multiple of 32, \
          got %s"
         fn (pp_shape c));
  let expected = Array.copy c in
  expected.(r - 1) <- c.(r - 1) / 16;
  if Nx.shape scales <> expected then
    invalid_arg
      (strf "%s: scales must have shape %s, one per 32 values, got %s" fn
         (pp_shape expected)
         (pp_shape (Nx.shape scales)))

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

let mxfp4 ~scales codes =
  check_mxfp4 "Nx_quant.mxfp4" codes scales;
  Mxfp4 { codes; scales }

let q8_0 blocks =
  check_blocks "Nx_quant.q8_0" "Q8_0" q8_0_bytes blocks;
  Q8_0 { blocks }

let q4_k blocks =
  check_blocks "Nx_quant.q4_k" "Q4_K" q4_k_bytes blocks;
  Q4_K { blocks }

let q6_k blocks =
  check_blocks "Nx_quant.q6_k" "Q6_K" q6_k_bytes blocks;
  Q6_K { blocks }

let place p w =
  (* Every window must start and stop at a block of the quants. *)
  let part, bytes, values = layout w in
  let c = Nx.shape part in
  let r = Array.length c in
  List.iter
    (fun d ->
      let lo, hi = (Nx.Placement.window p c d).(r - 1) in
      if lo mod bytes <> 0 || hi mod bytes <> 0 then
        invalid_arg
          (strf
             "Nx_quant.place: splitting the weight along axis %d in %d cuts a \
              block of %d values (%d blocks)"
             (r - 1)
             (c.(r - 1) / (hi - lo))
             values
             (c.(r - 1) / bytes)))
    (Nx.Placement.devices p);
  map (Nx.place p) w

let shape w =
  let part, bytes, values = layout w in
  let s = Array.copy (Nx.shape part) in
  let r = Array.length s in
  s.(r - 1) <- s.(r - 1) / bytes * values;
  s

(* Structure *)

let walk c w =
  let open Nx.Ptree.Walk in
  let blocks format bytes v =
    case c format;
    let blocks = field c "blocks" tensor v in
    check_blocks "Nx_quant.walk" (String.uppercase_ascii format) bytes blocks;
    blocks
  in
  match w with
  | Mxfp4 { codes; scales } ->
      case c "mxfp4";
      let codes = field c "codes" tensor codes in
      let scales = field c "scales" tensor scales in
      check_mxfp4 "Nx_quant.walk" codes scales;
      Mxfp4 { codes; scales }
  | Q8_0 { blocks = b } -> Q8_0 { blocks = blocks "q8_0" q8_0_bytes b }
  | Q4_K { blocks = b } -> Q4_K { blocks = blocks "q4_k" q4_k_bytes b }
  | Q6_K { blocks = b } -> Q6_K { blocks = blocks "q6_k" q6_k_bytes b }

type weight = t

module Structure = struct
  type _ t = weight

  let walk = walk
end

let ptree = Nx.Ptree.instantiate (module Structure)

(* Decoding. A byte holds two e2m1 codes, the low nibble first: a sign bit, two
   exponent bits and a mantissa bit, the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and
   6. A scale byte is an e8m0 exponent, 2^(s - 127), with 255 a NaN. Every
   value, scaled, is exact at float32 barring overflow, and the product is
   rounded once.

   The values are assembled as float32 bits with integer operations, so a
   compiled product reads the code bytes themselves: a table lookup's int64
   index would be stored between gathering the experts and multiplying. *)

(* [scaled codes v scale] is the weight of [codes] [[| ...; n; k / 2 |]] from
   their values [v], two per byte, and their groups' scales: each value times
   its group's scale, [[| ...; n; k |]]. *)
let scaled codes v scale =
  let s = Nx.shape codes in
  let r = Array.length s in
  let lead = Array.sub s 0 (r - 1) and k = 2 * s.(r - 1) in
  let groups = Array.append lead [| k / 32 |] in
  Nx.reshape
    (Array.append lead [| k |])
    (Nx.mul
       (Nx.reshape (Array.append groups [| 32 |]) v)
       (Nx.reshape (Array.append groups [| 1 |]) scale))

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

(* [mxfp4_values codes scales] is the weight of contiguous [codes] and [scales]
   at float32, from their bits. *)
let mxfp4_values codes scales =
  let bytes = Nx.cast Nx.uint32 codes in
  let nibbles =
    Nx.stack ~axis:(-1)
      [ Nx.bitwise_and bytes (Nx.scalar_like bytes 15l); Nx.rshift bytes 4 ]
  in
  scaled codes
    (Nx.bitcast Nx.float32 (code_bits nibbles))
    (Nx.bitcast Nx.float32 (scale_bits (Nx.cast Nx.uint32 scales)))

(* The GGUF formats, decoded as ggml's dequantize_row_q8_0, _q4_K and _q6_K
   decode them: the scales at float32, each product left to right. Each block's
   fields are views of its bytes, so a compiled product reads the bytes
   themselves. *)

(* [split ~bytes t] is the contiguous rows [t] [[| ...; n; b * bytes |]] as
   their blocks, [[| ...; n; b; bytes |]]. *)
let split ~bytes t =
  let s = Nx.shape t in
  let r = Array.length s in
  Nx.reshape
    (Array.append (Array.sub s 0 (r - 1)) [| s.(r - 1) / bytes; bytes |])
    t

(* [bytes b lo hi] is the bytes \[[lo];[hi]) of each block of [b]. *)
let bytes b lo hi =
  let r = Nx.ndim b in
  Nx.shrink
    (Array.mapi
       (fun i d -> if i = r - 1 then (lo, hi) else (0, d))
       (Nx.shape b))
    b

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

(* [values w] is the weight [w] of contiguous parts at float32. *)
let values = function
  | Mxfp4 { codes; scales } -> mxfp4_values codes scales
  | Q8_0 { blocks } -> q8_0_values blocks
  | Q4_K { blocks } -> q4_k_values blocks
  | Q6_K { blocks } -> q6_k_values blocks

(* Batch axes, aligned on the right and broadcast as Nx.matmul's. *)

let broadcast fn a b =
  let la = Array.length a and lb = Array.length b in
  let l = max la lb in
  Array.init l (fun i ->
      let da = if i < l - la then 1 else a.(i - l + la) in
      let db = if i < l - lb then 1 else b.(i - l + lb) in
      if da = db || db = 1 then da
      else if da = 1 then db
      else
        invalid_arg
          (strf "%s: batch axes %s and %s do not broadcast" fn (pp_shape a)
             (pp_shape b)))

(* [batch ?ids ws xs] is the batch axes of [w'], the matrices a product meets,
   and of its result, after checking the shapes of the weight [ws], the ids
   [ids] and the input [xs]. *)
let batch ?ids ws xs =
  let fn = "Nx_quant.apply" in
  let wr = Array.length ws in
  let k = ws.(wr - 1) in
  let xr = Array.length xs in
  if xr = 0 then invalid_arg (strf "%s: x must have at least one axis" fn);
  if xs.(xr - 1) <> k then
    invalid_arg
      (strf "%s: x's last axis is %d, the weight's inputs are %d" fn
         xs.(xr - 1)
         k);
  let xb = if xr = 1 then [||] else Array.sub xs 0 (xr - 2) in
  let wb =
    match ids with
    | None -> Array.sub ws 0 (wr - 2)
    | Some is ->
        let p = wr - 3 in
        if p < 0 then
          invalid_arg
            (strf "%s: ids need a weight with an expert axis, got shape %s" fn
               (pp_shape ws));
        if Array.length is < p then
          invalid_arg
            (strf "%s: ids of shape %s lack the weight's %d leading axes" fn
               (pp_shape is) p);
        Array.append
          (broadcast fn (Array.sub ws 0 p) (Array.sub is 0 p))
          (Array.sub is p (Array.length is - p))
  in
  (wb, broadcast fn xb wb)

(* Routes. [routes ~wb ~lanes ~e ids] is, at each position of [w']'s batch axes
   [wb], the index of the matrix its id names among all [lanes]' [e] experts, an
   id outside them clamped among them, and whether it names one. *)
let routes ~wb ~lanes ~e ids =
  let ids = Nx.broadcast_to wb ids in
  (* A position's lane's row of experts, row-major over the lanes. *)
  let lane = ref (Nx.zeros Nx.int64 wb) and stride = ref 1 in
  for a = Array.length lanes - 1 downto 0 do
    if lanes.(a) > 1 then begin
      let shape = Array.mapi (fun b n -> if b = a then n else 1) wb in
      let iota = Nx.reshape shape (Nx.arange Nx.int64 0 lanes.(a) 1) in
      lane := Nx.add !lane (Nx.mul_s iota (Int64.of_int !stride))
    end;
    stride := !stride * lanes.(a)
  done;
  let named =
    Nx.logical_and (Nx.greater_equal_s ids 0L) (Nx.less_s ids (Int64.of_int e))
  in
  let at =
    Nx.add
      (Nx.mul_s !lane (Int64.of_int e))
      (Nx.clamp ~min:0L ~max:(Int64.of_int (e - 1)) ids)
  in
  (at, named)

(* [matrices t g] is the part [t] as its [g] matrices, [[| g; ...; ... |]]. *)
let matrices t g =
  let s = Nx.shape t in
  let r = Array.length s in
  Nx.reshape [| g; s.(r - 2); s.(r - 1) |] (Nx.contiguous t)

(* [product x w] is [x] times each matrix of the weight [w] of contiguous parts,
   transposed, at float32. *)
let product x w = Nx.matmul x (Nx.matrix_transpose (values w))

(* Grouped products. When a product's instances outnumber the matrices they
   meet, each matrix is multiplied once by the rows of many of its instances,
   rather than once per instance: the instances are sorted by matrix, each
   matrix's run is padded to whole blocks of [block] instances, and every block
   is one product with its matrix. A block is the unit of a matrix's reuse; the
   padding costs at most [block - 1] instances per matrix.

   Instances split over devices are grouped on each device: a sort cannot run
   along a split axis, and a device's instances are its own rows. *)

(* A block holds 4 instances, the size measured fastest on every device:
   gpt-oss-20b's gate and up product of 512 tokens takes 21.9 ms in blocks of 4
   and 25.3 ms in blocks of 2 on an RTX 5000 Ada, and 162 ms and 167 ms on an
   M1 Max's Metal; of 64 tokens on the host, 203 ms and 281 ms. *)
let block = 4

(* [shards t] is the number of devices' windows that split [t]'s first axis. *)
let shards t =
  let p = Nx.placement t and s = Nx.shape t in
  List.length
    (List.sort_uniq compare
       (List.map
          (fun d -> (Nx.Placement.window p s d).(0))
          (Nx.Placement.devices p)))

(* [grouped ~g ~r at named x w] is the products of the [x] rows [[| i; m; k |]]
   of [i] instances over [r] devices' shards, the instance [j] with the matrix
   [at.(j)] among [g] if [named.(j)], [[| i; m; n |]]. The product of an
   instance that names no matrix is left to the caller's mask. *)
let grouped ~g ~r at named x w =
  let i = Nx.dim 0 x and m = Nx.dim 1 x and k = Nx.dim 2 x in
  let j = i / r in
  let int64 = Int64.of_int in
  (* Instances naming no matrix sort last, as matrix [g], and take no slot. *)
  let key =
    Nx.reshape [| r; j |] (Nx.where named at (Nx.full_like at (int64 g)))
  in
  let order = Nx.argsort ~axis:1 key in
  (* Each matrix's run in sorted order, [[| r; g |]]: its length, first
     position, and slots padded to whole blocks. *)
  let count =
    Nx.scatter ~mode:`Add ~axis:1 ~indices:key ~values:(Nx.ones_like key)
      (Nx.zeros Nx.int64 [| r; g |])
  in
  let first = Nx.sub (Nx.cumsum ~axis:1 count) count in
  let padded =
    Nx.mul_s
      (Nx.div_s (Nx.add_s count (int64 (block - 1))) (int64 block))
      (int64 block)
  in
  let ends = Nx.cumsum ~axis:1 padded in
  let starts = Nx.sub ends padded in
  (* Slots, in blocks: a bound of the padded runs, whatever the ids. *)
  let blocks = (j + (min g j * (block - 1)) + block - 1) / block in
  let slots = blocks * block in
  (* Each block's matrix: the runs that end at or before its first slot. *)
  let owner =
    Nx.clamp
      ~max:(int64 (g - 1))
      (Nx.cast Nx.int64
         (Nx.sum ~axes:[ 2 ]
            (Nx.cast Nx.int32
               (Nx.less_equal
                  (Nx.reshape [| r; 1; g |] ends)
                  (Nx.reshape [| 1; blocks; 1 |]
                     (Nx.arange Nx.int64 0 slots block))))))
  in
  (* Each slot's instance: its rank in its block's run, in sorted order. A slot
     past its run reads index -1, a row of zeros whose gradient's scatter is
     dropped. *)
  let slot_owner =
    Nx.reshape [| r; slots |]
      (Nx.broadcast_to [| r; blocks; block |]
         (Nx.reshape [| r; blocks; 1 |] owner))
  in
  let along t indices = Nx.take_along_axis ~axis:1 ~indices t in
  let offset =
    Nx.sub
      (Nx.reshape [| 1; slots |] (Nx.arange Nx.int64 0 slots 1))
      (along starts slot_owner)
  in
  let in_run = Nx.less offset (along count slot_owner) in
  let instance = along order (Nx.add (along first slot_owner) offset) in
  let instance = Nx.where in_run instance (Nx.scalar_like instance (-1L)) in
  let rows =
    Nx.reshape
      [| r; blocks; block * m; k |]
      (along
         (Nx.reshape [| r; j; m; k |] x)
         (Nx.broadcast_to [| r; slots; m; k |]
            (Nx.reshape [| r; slots; 1; 1 |] instance)))
  in
  let weights t =
    let w =
      Nx.take ~axis:0
        ~indices:(Nx.reshape [| r * blocks |] owner)
        (matrices t g)
    in
    Nx.reshape (Array.append [| r; blocks |] (Array.sub (Nx.shape w) 1 2)) w
  in
  let y = product rows (map weights w) in
  let n = Nx.dim (-1) y in
  (* Each instance's slot: its run's first slot and its rank in the run. *)
  let rank =
    Nx.scatter ~unique_indices:true ~axis:1 ~indices:order
      ~values:(Nx.broadcast_to [| r; j |] (Nx.arange Nx.int64 0 j 1))
      (Nx.zeros Nx.int64 [| r; j |])
  in
  let slot = Nx.add (along starts key) (Nx.sub rank (along first key)) in
  Nx.reshape [| i; m; n |]
    (along
       (Nx.reshape [| r; slots; m; n |] y)
       (Nx.broadcast_to [| r; j; m; n |] (Nx.reshape [| r; j; 1; 1 |] slot)))

(* Products *)

let dequant dt w = Nx.cast dt (values (map Nx.contiguous w))

let apply (type b) ?ids w (x : (float, b) Nx.t) : (float, b) Nx.t =
  let ws = shape w in
  let wb, rb = batch ?ids:(Option.map Nx.shape ids) ws (Nx.shape x) in
  let x32 = Nx.cast Nx.float32 x in
  match ids with
  | None -> Nx.cast (Nx.dtype x) (product x32 (map Nx.contiguous w))
  | Some ids ->
      let wr = Array.length ws in
      let lanes = Array.sub ws 0 (wr - 3) and e = ws.(wr - 3) in
      let g = Array.fold_left ( * ) 1 lanes * e in
      let at, named = routes ~wb ~lanes ~e ids in
      let n = ws.(wr - 2) and k = ws.(wr - 1) in
      let vector = Nx.ndim x = 1 in
      let m = if vector then 1 else Nx.dim (-2) x in
      let i = Array.fold_left ( * ) 1 rb in
      let units = if vector then [| 1 |] else [| 1; 1 |] in
      let flat t = Nx.reshape [| i |] (Nx.broadcast_to rb t) in
      let rows =
        Nx.reshape [| i; m; k |]
          (Nx.broadcast_to
             (Array.append rb [| m; k |])
             (if vector then Nx.reshape [| 1; k |] x32 else x32))
      in
      (* The routes and rows join where one of them is split. *)
      let r = if i = 0 then 1 else max (shards (flat at)) (shards rows) in
      (* Grouped, the padding, at most a slot per matrix, stays below half the
         instances. *)
      let y =
        if i / r >= block * g && m * n * k > 0 then
          Nx.reshape
            (Array.append rb (if vector then [| n |] else [| m; n |]))
            (grouped ~g ~r (flat at) (flat named) rows w)
        else
          let take t =
            let matrix = Array.sub (Nx.shape t) (wr - 2) 2 in
            Nx.reshape (Array.append wb matrix)
              (Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] at) (matrices t g))
          in
          product x32 (map take w)
      in
      Nx.cast (Nx.dtype x)
        (Nx.where
           (Nx.reshape (Array.append wb units) named)
           y (Nx.zeros_like y))
