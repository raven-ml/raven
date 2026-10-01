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
   after checking the shapes of the weight [ws], the ids [ids] and the input
   [xs]. *)
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
  ignore (broadcast fn xb wb);
  wb

(* [gather ~wb ~lanes ~e ids codes scales] is the parts of the matrices of [w'],
   of batch axes [wb]: at each position, the expert its id names in its lane, an
   id outside the [e] experts clamped among them, and whether it names one. *)
let gather ~wb ~lanes ~e ids codes scales =
  let p = Array.length lanes in
  let ids = Nx.broadcast_to wb ids in
  (* A position's matrix among all lanes' experts: its lane's row of experts,
     then its id among them. *)
  let lane =
    let at = ref (Nx.zeros Nx.int64 wb) and stride = ref 1 in
    for a = p - 1 downto 0 do
      if lanes.(a) > 1 then begin
        let shape = Array.mapi (fun b n -> if b = a then n else 1) wb in
        let iota = Nx.reshape shape (Nx.arange Nx.int64 0 lanes.(a) 1) in
        at := Nx.add !at (Nx.mul_s iota (Int64.of_int !stride))
      end;
      stride := !stride * lanes.(a)
    done;
    !at
  in
  let named =
    Nx.logical_and (Nx.greater_equal_s ids 0L) (Nx.less_s ids (Int64.of_int e))
  in
  let at =
    Nx.reshape [| -1 |]
      (Nx.add
         (Nx.mul_s lane (Int64.of_int e))
         (Nx.clamp ~min:0L ~max:(Int64.of_int (e - 1)) ids))
  in
  let take t =
    let s = Nx.shape t in
    let matrix = Array.sub s (p + 1) 2 in
    Nx.reshape (Array.append wb matrix)
      (Nx.take ~axis:0 ~indices:at
         (Nx.reshape
            (Array.append
               [| Array.fold_left ( * ) 1 (Array.sub s 0 (p + 1)) |]
               matrix)
            (Nx.contiguous t)))
  in
  (take codes, take scales, named)

(* Products *)

let dequant dt (Mxfp4 { codes; scales }) =
  Nx.cast dt (values (Nx.contiguous codes) (Nx.contiguous scales))

let apply (type b) ?ids (Mxfp4 { codes; scales }) (x : (float, b) Nx.t) :
    (float, b) Nx.t =
  let ws = Nx.shape codes in
  let wb = batch ?ids:(Option.map Nx.shape ids) ws (Nx.shape x) in
  let x32 = Nx.cast Nx.float32 x in
  let product codes scales =
    Nx.matmul x32 (Nx.matrix_transpose (values codes scales))
  in
  match ids with
  | None ->
      Nx.cast (Nx.dtype x)
        (product (Nx.contiguous codes) (Nx.contiguous scales))
  | Some ids ->
      let wr = Array.length ws in
      let lanes = Array.sub ws 0 (wr - 3) and e = ws.(wr - 3) in
      let codes, scales, named = gather ~wb ~lanes ~e ids codes scales in
      let y = product codes scales in
      let units = if Nx.ndim x = 1 then [| 1 |] else [| 1; 1 |] in
      Nx.cast (Nx.dtype x)
        (Nx.where
           (Nx.reshape (Array.append wb units) named)
           y (Nx.zeros_like y))
