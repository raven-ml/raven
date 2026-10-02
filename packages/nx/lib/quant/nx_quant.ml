(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Mxfp4 of {
      codes : (int, Nx.uint8_elt) Nx.t;
      scales : (int, Nx.uint8_elt) Nx.t;
    }

let strf = Printf.sprintf

let pp_shape s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

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

let mxfp4 ~scales codes =
  check_mxfp4 "Nx_quant.mxfp4" codes scales;
  Mxfp4 { codes; scales }

let place p (Mxfp4 { codes; scales }) =
  (* A group is 16 bytes of codes: every window must start and stop at one. *)
  let c = Nx.shape codes in
  let r = Array.length c in
  List.iter
    (fun d ->
      let lo, hi = (Nx.Placement.window p c d).(r - 1) in
      if lo mod 16 <> 0 || hi mod 16 <> 0 then
        invalid_arg
          (strf
             "Nx_quant.place: splitting codes and scales along axis %d in %d \
              cuts a 32-value group (%d groups)"
             (r - 1)
             (c.(r - 1) / (hi - lo))
             (Nx.shape scales).(r - 1)))
    (Nx.Placement.devices p);
  Mxfp4 { codes = Nx.place p codes; scales = Nx.place p scales }

let shape (Mxfp4 { codes; _ }) =
  let s = Array.copy (Nx.shape codes) in
  let r = Array.length s in
  s.(r - 1) <- 2 * s.(r - 1);
  s

(* Structure *)

let walk c (Mxfp4 { codes; scales }) =
  let open Nx.Ptree.Walk in
  case c "mxfp4";
  let codes = field c "codes" tensor codes in
  let scales = field c "scales" tensor scales in
  check_mxfp4 "Nx_quant.walk" codes scales;
  Mxfp4 { codes; scales }

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

(* [values codes scales] is the weight of contiguous [codes] and [scales] at
   float32, from their bits. *)
let values codes scales =
  let bytes = Nx.cast Nx.uint32 codes in
  let nibbles =
    Nx.stack ~axis:(-1)
      [ Nx.bitwise_and bytes (Nx.scalar_like bytes 15l); Nx.rshift bytes 4 ]
  in
  scaled codes
    (Nx.bitcast Nx.float32 (code_bits nibbles))
    (Nx.bitcast Nx.float32 (scale_bits (Nx.cast Nx.uint32 scales)))

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
  let k = 2 * ws.(wr - 1) in
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
               (pp_shape (Array.append (Array.sub ws 0 (wr - 1)) [| k |])));
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

(* [product x codes scales] is [x] times each matrix of the weight of [codes]
   and [scales], transposed, at float32. *)
let product x codes scales =
  Nx.matmul x (Nx.matrix_transpose (values codes scales))

(* Grouped products. When a product's instances outnumber the matrices they
   meet, each matrix is multiplied once by the rows of many of its instances,
   rather than once per instance: the instances are sorted by matrix, each
   matrix's run is padded to whole blocks of [block] instances, and every block
   is one product with its matrix. A block is the unit of a matrix's reuse; the
   padding costs at most [block - 1] instances per matrix.

   Instances split over devices are grouped on each device: a sort cannot run
   along a split axis, and a device's instances are its own rows. *)

(* A block holds 4 instances, the size measured fastest end to end: on an
   RTX 5000 Ada, gpt-oss-20b's 512-token prefill takes 1.04 s in blocks of 4,
   1.11 s in blocks of 2 and 1.08 s in blocks of 8, and a host product of 64
   tokens 203 ms, against 281 ms and 275 ms. One expert's product of 512 tokens
   alone favours blocks of 2 on that GPU (28.6 against 58.2 ms); a model's
   routes give each expert far fewer. *)
let block = 4

(* [shards t] is the number of devices' windows that split [t]'s first axis. *)
let shards t =
  let p = Nx.placement t and s = Nx.shape t in
  List.length
    (List.sort_uniq compare
       (List.map
          (fun d -> (Nx.Placement.window p s d).(0))
          (Nx.Placement.devices p)))

(* [grouped ~g ~r at named x codes scales] is the products of the [x] rows [[|
   i; m; k |]] of [i] instances over [r] devices' shards, the instance [j] with
   the matrix [at.(j)] among [g] if [named.(j)], [[| i; m; n |]]. The product of
   an instance that names no matrix is left to the caller's mask. *)
let grouped ~g ~r at named x codes scales =
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
  let y = product rows (weights codes) (weights scales) in
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

let dequant dt (Mxfp4 { codes; scales }) =
  Nx.cast dt (values (Nx.contiguous codes) (Nx.contiguous scales))

let apply (type b) ?ids (Mxfp4 { codes; scales }) (x : (float, b) Nx.t) :
    (float, b) Nx.t =
  let ws = Nx.shape codes in
  let wb, rb = batch ?ids:(Option.map Nx.shape ids) ws (Nx.shape x) in
  let x32 = Nx.cast Nx.float32 x in
  match ids with
  | None ->
      Nx.cast (Nx.dtype x)
        (product x32 (Nx.contiguous codes) (Nx.contiguous scales))
  | Some ids ->
      let wr = Array.length ws in
      let lanes = Array.sub ws 0 (wr - 3) and e = ws.(wr - 3) in
      let g = Array.fold_left ( * ) 1 lanes * e in
      let at, named = routes ~wb ~lanes ~e ids in
      let n = ws.(wr - 2) and k = 2 * ws.(wr - 1) in
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
            (grouped ~g ~r (flat at) (flat named) rows codes scales)
        else
          let take t =
            let matrix = Array.sub (Nx.shape t) (wr - 2) 2 in
            Nx.reshape (Array.append wb matrix)
              (Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] at) (matrices t g))
          in
          product x32 (take codes) (take scales)
      in
      Nx.cast (Nx.dtype x)
        (Nx.where
           (Nx.reshape (Array.append wb units) named)
           y (Nx.zeros_like y))
