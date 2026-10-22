(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantised weights against their formats' definitions: dequant gives each
   value at its dtype, as the format's reference decodes it from its bytes,
   apply is the product with the dequantised weight within the error of a
   float32 sum, take is the gather of the dequantised weight, and a weight
   placed anywhere over memory devices gives the host's bits, lying where the
   interface says. *)

open Windtrap
module D = Nx_array.Dtype

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* The formats *)

type format = Mxfp4 | Mxfp4_blocks | Q8_0 | Q4_K | Q6_K

let formats = [ Mxfp4; Mxfp4_blocks; Q8_0; Q4_K; Q6_K ]

let format_name = function
  | Mxfp4 -> "mxfp4"
  | Mxfp4_blocks -> "mxfp4_blocks"
  | Q8_0 -> "q8_0"
  | Q4_K -> "q4_k"
  | Q6_K -> "q6_k"

let pp_format ppf f = Format.pp_print_string ppf (format_name f)

(* The values and the bytes of a block, MXFP4's codes alone where its scales lie
   apart, and the offsets of its float16 scales. *)
let block_values = function
  | Mxfp4 | Mxfp4_blocks | Q8_0 -> 32
  | Q4_K | Q6_K -> 256

let block_bytes = function
  | Mxfp4 -> 16
  | Mxfp4_blocks -> 17
  | Q8_0 -> 34
  | Q4_K -> 144
  | Q6_K -> 210

let halves = function
  | Mxfp4 | Mxfp4_blocks -> []
  | Q8_0 -> [ 0 ]
  | Q4_K -> [ 0; 2 ]
  | Q6_K -> [ 208 ]

let format_of (w : _ Nx_quant.t) =
  match w with
  | Mxfp4 _ -> Mxfp4
  | Mxfp4_blocks _ -> Mxfp4_blocks
  | Q8_0 _ -> Q8_0
  | Q4_K _ -> Q4_K
  | Q6_K _ -> Q6_K

let u8 shape b = Nx.create Nx.uint8 shape b

(* [weight_of format shape b] is the weight of [format] whose blocks, or MXFP4
   codes, of [shape], are [b], with an MXFP4 weight's [scales]. *)
let weight_of ?scales format shape b =
  let blocks = u8 shape b in
  match format with
  | Mxfp4 -> Nx_quant.mxfp4 ~scales:(Option.get scales) blocks
  | Mxfp4_blocks -> Nx_quant.mxfp4_blocks blocks
  | Q8_0 -> Nx_quant.q8_0 blocks
  | Q4_K -> Nx_quant.q4_k blocks
  | Q6_K -> Nx_quant.q6_k blocks

(* The references *)

(* MXFP4: a code's magnitude, signed, times [2 ^ (s - 127)] for its group's
   scale byte [s], NaN for [255], rounded once to float32. *)

let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |]

let mxfp4_value code s =
  if s = 255 then Float.nan
  else
    let m = to_f32 (e2m1.(code land 7) *. Float.ldexp 1. (s - 127)) in
    if code < 8 then m else -.m

(* The two files' MXFP4 layouts of [codes], one per value, and [scales], one per
   32 values. A checkpoint's byte [i] holds values [2 i] and [2 i + 1], the low
   nibble first, and its scales lie apart. A GGUF block is a group's scale byte,
   then 16 bytes, byte [j] holding value [j] in its low nibble and value [j +
   16] in its high one. *)

let checkpoint_bytes codes =
  Array.init
    (Array.length codes / 2)
    (fun i -> codes.(2 * i) lor (codes.((2 * i) + 1) lsl 4))

let gguf_blocks codes scales =
  Array.init
    (17 * Array.length scales)
    (fun i ->
      let g = i / 17 and j = (i mod 17) - 1 in
      if j < 0 then scales.(g)
      else codes.((32 * g) + j) lor (codes.((32 * g) + j + 16) lsl 4))

(* The GGUF formats, as ggml's dequantize_row_q8_0, _q4_K and _q6_K compute them
   at float32, over a block's bytes [b]. Each product of a float16 scale and
   integers is exact, so one rounding to float32 is ggml's. *)

(* The binary16 of bits [h]. *)
let binary16 h =
  let sign = if h land 0x8000 <> 0 then -1. else 1. in
  let e = (h lsr 10) land 31 and m = h land 1023 in
  if e = 31 then if m = 0 then sign *. Float.infinity else Float.nan
  else if e = 0 then sign *. Float.ldexp (Float.of_int m) (-24)
  else sign *. Float.ldexp (Float.of_int (m + 1024)) (e - 25)

let f16 b at = binary16 (b at lor (b (at + 1) lsl 8))
let int8 v = if v >= 128 then v - 256 else v

(* block_mxfp4: e, qs[16]. *)
let mxfp4_block b =
  Array.init 32 (fun j ->
      mxfp4_value ((b (1 + (j mod 16)) lsr (4 * (j / 16))) land 15) (b 0))

(* block_q8_0: d, qs[32]. *)
let q8_0_block b =
  let d = f16 b 0 in
  Array.init 32 (fun j -> to_f32 (d *. Float.of_int (int8 (b (2 + j)))))

(* block_q4_K: d, dmin, scales[12], qs[128]; get_scale_min_k4 unpacks a
   sub-block's 6-bit scale and min. *)
let q4_k_block b =
  let d = f16 b 0 and dmin = f16 b 2 in
  let q j = b (4 + j) in
  let scale_min j =
    if j < 4 then (q j land 63, q (j + 4) land 63)
    else
      ( q (j + 4) land 0xF lor ((q (j - 4) lsr 6) lsl 4),
        (q (j + 4) lsr 4) lor ((q j lsr 6) lsl 4) )
  in
  let y = Array.make 256 0. in
  for c = 0 to 3 do
    let sc1, m1 = scale_min (2 * c) and sc2, m2 = scale_min ((2 * c) + 1) in
    for l = 0 to 31 do
      let qs = b (16 + (32 * c) + l) in
      y.((64 * c) + l) <-
        to_f32
          ((d *. Float.of_int sc1 *. Float.of_int (qs land 15))
          -. (dmin *. Float.of_int m1));
      y.((64 * c) + 32 + l) <-
        to_f32
          ((d *. Float.of_int sc2 *. Float.of_int (qs lsr 4))
          -. (dmin *. Float.of_int m2))
    done
  done;
  y

(* block_q6_K: ql[128], qh[64], scales[16] (int8), d. *)
let q6_k_block b =
  let d = f16 b 208 in
  let y = Array.make 256 0. in
  for h = 0 to 1 do
    let ql i = b ((64 * h) + i) and qh i = b (128 + (32 * h) + i) in
    let v s q =
      to_f32 (d *. Float.of_int (int8 (b (192 + (8 * h) + s))) *. Float.of_int q)
    in
    for l = 0 to 31 do
      let is = l / 16 in
      let high k = ((qh l lsr (2 * k)) land 3) lsl 4 in
      y.((128 * h) + l) <- v is ((ql l land 0xF lor high 0) - 32);
      y.((128 * h) + l + 32) <-
        v (is + 2) ((ql (l + 32) land 0xF lor high 1) - 32);
      y.((128 * h) + l + 64) <- v (is + 4) (((ql l lsr 4) lor high 2) - 32);
      y.((128 * h) + l + 96) <-
        v (is + 6) (((ql (l + 32) lsr 4) lor high 3) - 32)
    done
  done;
  y

let gguf_values block bytes blocks =
  let a = Nx.to_array blocks in
  Array.concat
    (List.init
       (Array.length a / bytes)
       (fun i -> block (fun j -> a.((i * bytes) + j))))

(* The values of [w] in C order, at float32. *)
let values (w : _ Nx_quant.t) =
  match w with
  | Mxfp4 { codes; scales } ->
      let c = Nx.to_array codes and s = Nx.to_array scales in
      Array.init
        (2 * Array.length c)
        (fun i ->
          mxfp4_value ((c.(i / 2) lsr (4 * (i mod 2))) land 15) s.(i / 32))
  | Mxfp4_blocks { blocks } -> gguf_values mxfp4_block 17 blocks
  | Q8_0 { blocks } -> gguf_values q8_0_block 34 blocks
  | Q4_K { blocks } -> gguf_values q4_k_block 144 blocks
  | Q6_K { blocks } -> gguf_values q6_k_block 210 blocks

(* A float dtype and the rounding of a float32 value to it. *)
type fdt = F : { dtype : (float, 'b) Nx.dtype; round : float -> float } -> fdt

let fdts =
  let at dtype = F { dtype; round = D.of_float dtype } in
  [ at Nx.float32; at Nx.bfloat16; at Nx.float16; at Nx.float64 ]

let fdt = Gen.of_list ~pp:(fun ppf (F d) -> D.pp ppf d.dtype) fdts

(* [bits t] is each element's bits, so -0 and +0 differ and a NaN is its
   payload. *)
let bits (type b d) (t : (float, b, d) Nx.t) =
  match Nx.dtype t with
  | D.Float64 -> Nx.to_array (Nx.bitcast Nx.int64 t)
  | D.Float32 -> Array.map Int64.of_int32 (Nx.to_array (Nx.bitcast Nx.int32 t))
  | D.Float16 | D.Bfloat16 ->
      Array.map Int64.of_int (Nx.to_array (Nx.bitcast Nx.uint16 t))
  | dt -> failf "no bits for %a" D.pp dt

(* [bits t], every NaN alike: a product leaves a NaN's sign and payload
   unspecified ({!Nx_kernel.S.contract}). *)
let product_bits t =
  let nan = Array.map Float.is_nan (Nx.to_array (Nx.cast Nx.float64 t)) in
  Array.map2 (fun n b -> if n then Int64.min_int else b) nan (bits t)

(* Weights *)

(* A view of every part that keeps them a weight, where its logical shape and
   the values of its blocks allow it. Each part counts its blocks along its last
   axis and its rows along the axis before. *)
type view = {
  view : string;
  fits : int array -> int -> bool;
  apply : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t;
}

let views =
  let r = Array.length in
  let all t a = List.init a (fun _ -> Nx.A) @ [ t ] in
  [
    { view = "contiguous"; fits = (fun _ _ -> true); apply = Fun.id };
    {
      view = "every other row";
      fits = (fun s _ -> s.(r s - 2) > 0);
      apply =
        (fun t -> Nx.slice (all (Nx.Rs (0, max_int, 2)) (Nx.ndim t - 2)) t);
    };
    {
      view = "its leading axes swapped";
      fits = (fun s _ -> r s >= 4);
      apply = (fun t -> Nx.swapaxes 0 1 t);
    };
    {
      view = "the first half of its inputs";
      fits = (fun s values -> s.(r s - 1) / values mod 2 = 0);
      apply =
        (fun t ->
          let a = Nx.ndim t - 1 in
          Nx.slice (all (Nx.R (0, Nx.dim a t / 2)) a) t);
    };
  ]

(* A weight's parts, at a set. *)
let parts (w : 'd Nx_quant.t) : 'd Nx.uint8_t list =
  match w with
  | Mxfp4 { codes; scales } -> [ codes; scales ]
  | Mxfp4_blocks { blocks }
  | Q8_0 { blocks }
  | Q4_K { blocks }
  | Q6_K { blocks } ->
      [ blocks ]

let description = Nx.Ptree.v Nx_quant.walk

let pp_weight ppf (view, w) =
  Format.fprintf ppf "@[<v>%a %a, %s" pp_format (format_of w) pp_shape
    (Nx_quant.shape w) view;
  List.iter (fun t -> Format.fprintf ppf "@,%a" Nx.pp t) (parts w);
  Format.fprintf ppf "@]"

(* MXFP4 scale bytes at the overflow, subnormal and NaN corners too. *)
let e8m0 =
  Gen.frequency
    [
      (4, Gen.int_range 118 136);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 127; 253; 254; 255 ]);
    ]

(* Float16 scales: moderate ones of either sign, and zero, the largest finite,
   subnormal, infinite and NaN ones. *)
let half =
  Gen.frequency
    [
      ( 8,
        Gen.map
          (fun (s, e, m) -> (s lsl 15) lor (e lsl 10) lor m)
          (Gen.triple (Gen.int_range 0 1) (Gen.int_range 11 17)
             (Gen.int_range 0 1023)) );
      ( 1,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "0x%04x")
          [ 0x0000; 0x8000; 0x7BFF; 0xFBFF; 0x0001; 0x83FF; 0x7C00; 0x7E00 ] );
    ]

(* [gguf_bytes format count] is [count] blocks of [format]'s bytes, random but
   for their float16 scales, drawn from [half]. *)
let gguf_bytes format count =
  let open Gen in
  let bytes = block_bytes format and fields = halves format in
  let+ b = array ~size:(constant (count * bytes)) (int_range 0 255)
  and+ h = array ~size:(constant (count * List.length fields)) half in
  List.iteri
    (fun f at ->
      for i = 0 to count - 1 do
        let v = h.((i * List.length fields) + f) in
        b.((i * bytes) + at) <- v land 255;
        b.((i * bytes) + at + 1) <- v lsr 8
      done)
    fields;
  b

let format = Gen.of_list ~pp:pp_format formats

(* An extent from [0] to [hi], [0] seldom. *)
let extent hi = Gen.frequency [ (1, Gen.constant 0); (5, Gen.int_range 1 hi) ]

(* A weight of shape [[| lead…; n; k |]] under a view. *)
let weight ?(lead = Gen.list ~size:(Gen.int_range 0 2) (extent 3))
    ?(n = extent 4) ?(count = None) ?(format = format) () =
  let open Gen in
  with_pp pp_weight
    (let* f = format in
     let blocks =
       match count with
       | Some c -> c
       | None -> extent (if block_values f = 32 then 4 else 2)
     in
     let* lead, n, count = triple lead n blocks in
     let m = List.fold_left ( * ) 1 lead * n in
     let shape last = Array.of_list (lead @ [ n; last ]) in
     let+ w =
       match f with
       | Mxfp4 | Mxfp4_blocks ->
           let+ codes = array ~size:(constant (m * count * 32)) (int_range 0 15)
           and+ scales = array ~size:(constant (m * count)) e8m0 in
           if f = Mxfp4 then
             Nx_quant.mxfp4
               ~scales:(u8 (shape count) scales)
               (u8 (shape (count * 16)) (checkpoint_bytes codes))
           else
             Nx_quant.mxfp4_blocks
               (u8 (shape (count * 17)) (gguf_blocks codes scales))
       | f ->
           let+ b = gguf_bytes f (m * count) in
           weight_of f (shape (count * block_bytes f)) b
     and+ v = of_list views in
     if v.fits (Nx_quant.shape w) (block_values f) then
       (v.view, Nx.Ptree.map description (fun _ t -> v.apply t) w)
     else ("contiguous", w))

let dims w =
  let s = Nx_quant.shape w in
  let r = Array.length s in
  (Array.sub s 0 (r - 2), s.(r - 2), s.(r - 1))

(* A weight of fixed shape with finite values, for large weights. *)
let random_weight ?(format = Mxfp4) shape =
  let rng = Random.State.make [| 4 |] and r = Array.length shape in
  let lead = Array.sub shape 0 (r - 1) and k = shape.(r - 1) in
  let byte _ = Random.State.int rng 256 in
  let part last f = Nx.init Nx.uint8 (Array.append lead [| last |]) f in
  match format with
  | Mxfp4 ->
      Nx_quant.mxfp4
        ~scales:(part (k / 32) (fun _ -> 118 + Random.State.int rng 19))
        (part (k / 2) byte)
  | f ->
      let bytes = block_bytes f in
      let count = Array.fold_left ( * ) 1 lead * k / block_values f in
      let b = Array.init (count * bytes) byte in
      for i = 0 to count - 1 do
        if f = Mxfp4_blocks then b.(i * bytes) <- 118 + Random.State.int rng 19;
        List.iter
          (fun at ->
            (* A float16 in [-1, 1). *)
            let h =
              (Random.State.int rng 2 lsl 15)
              lor ((10 + Random.State.int rng 5) lsl 10)
              lor Random.State.int rng 1024
            in
            b.((i * bytes) + at) <- h land 255;
            b.((i * bytes) + at + 1) <- h lsr 8)
          (halves f)
      done;
      weight_of f (Array.append lead [| k / block_values f * bytes |]) b

let random_floats shape =
  let rng = Random.State.make [| 5 |] in
  Nx.init Nx.float32 shape (fun _ -> Random.State.float rng 2. -. 1.)

(* Products *)

(* [agrees ~at ~k expected bound actual] holds when [actual] is [expected]
   rounded to [at] within twice the error of a float32 sum of [k] terms whose
   magnitudes sum to [bound], subnormal terms included, and the error of the
   rounding; non-finite values must be exact. *)
let agrees ~at ~k expected bound actual =
  let (F d) = at in
  equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape actual);
  let b = Nx.to_array bound and a = Nx.to_array (Nx.cast Nx.float64 actual) in
  let format = D.float_format d.dtype in
  let rel = format.epsilon /. 2.
  and tiny = format.min_normal *. format.epsilon in
  let worst = ref 0. in
  Array.iteri
    (fun i e ->
      if not (Float.is_finite (d.round e)) then
        equal float_exact (d.round e) a.(i)
      else
        let tol =
          2. *. float_of_int k
          *. ((Float.ldexp 1. (-24) *. b.(i)) +. Float.ldexp 1. (-149))
          +. Float.max (rel *. Float.abs e) tiny
        in
        let err = Float.abs (a.(i) -. e) in
        let ratio = if err = 0. then 0. else err /. tol in
        worst :=
          Float.max !worst
            (if Float.is_nan ratio then Float.infinity else ratio))
    (Nx.to_array expected);
  at_most ~msg:"worst error over its bound" float_exact ~than:1. !worst

(* The product of [x] with each matrix of [w'], transposed, at float32, and the
   same product of magnitudes. A float32 sum may overflow where a wider one
   would not, so the reference sums at float32 too. *)
let product x w' =
  let x = Nx.cast Nx.float32 x in
  let t = Nx.matrix_transpose in
  (Nx.matmul x (t w'), Nx.matmul (Nx.abs x) (t (Nx.abs w')))

(* An input at one of the float dtypes. *)
type input = X : { at : fdt; x : (float, 'b, Nx.host) Nx.t } -> input

let usually =
  Gen.frequency
    [ (3, Gen.constant ~pp:Format.pp_print_bool true); (1, Gen.bool) ]

(* An input whose last axis is [inputs] and whose batch axes broadcast against
   [batch]: a vector, or rows behind [batch]'s last axes, each whole or of one,
   or behind all of them and an axis of its own in front. *)
let input ~batch ~inputs =
  let open Gen in
  let* shape =
    let+ vector = frequency [ (1, constant true); (4, constant false) ]
    and+ front = list ~size:(int_range 0 1) (int_range 1 2)
    and+ whole = list ~size:(constant (Array.length batch)) usually
    and+ drop = int_range 0 (Array.length batch)
    and+ m = int_range 0 3 in
    let own = List.mapi (fun i w -> if w then batch.(i) else 1) whole in
    if vector then [| inputs |]
    else
      Array.of_list
        ((if drop = 0 then front else [])
        @ List.filteri (fun i _ -> i >= drop) own
        @ [ m; inputs ])
  in
  let entry =
    frequency
      [
        (30, float_range (-1.) 1.);
        ( 1,
          of_list ~pp:Format.pp_print_float
            [ Float.nan; Float.infinity; Float.neg_infinity; -0. ] );
      ]
  in
  let+ at = fdt
  and+ xs = array ~size:(constant (Array.fold_left ( * ) 1 shape)) entry
  and+ view =
    of_list ~pp:Format.pp_print_string [ "contiguous"; "broadcast"; "flipped" ]
  in
  let (F d) = at in
  let r = Array.length shape in
  let x = Nx.cast d.dtype (Nx.create Nx.float64 shape xs) in
  (* Views whose batch axes do not merge. *)
  let x =
    match view with
    | "broadcast" when r >= 3 && shape.(r - 3) > 0 ->
        Nx.broadcast_to shape
          (Nx.slice (List.init (r - 3) (fun _ -> Nx.A) @ [ Nx.R (0, 1) ]) x)
    | "flipped" -> Nx.flip x
    | _ -> x
  in
  X { at; x }

let products =
  let open Gen in
  let* view, w = weight () in
  let lead, _, k = dims w in
  let+ x = input ~batch:lead ~inputs:k in
  (view, w, x)

let pp_product ppf (view, w, X { x; _ }) =
  Format.fprintf ppf "@[<v>%a@,x %a@]" pp_weight (view, w) Nx.pp x

let product_law (_, w, X { at; x }) =
  let _, _, k = dims w in
  let expected, bound = product x (Nx_quant.dequant Nx.float32 w) in
  List.iter
    (fun f -> cover ("a " ^ format_name f ^ " weight") (format_of w = f))
    formats;
  cover "one block per row" (k = block_values (format_of w));
  cover "an empty result" (Nx.numel expected = 0);
  cover "no input" (k = 0);
  let y = Nx_quant.apply w x in
  equal ~msg:"dtype" string (D.name (Nx.dtype x)) (D.name (Nx.dtype y));
  agrees ~at ~k expected bound y

(* [apply]'s stated equation, at [x]'s dtype where the format's values are exact
   there, else at float32. *)
let equation (type b) w (x : (float, b, Nx.host) Nx.t) =
  let dt = Nx.dtype x in
  let exact =
    match (dt, format_of w) with
    | (D.Float32 | D.Float64), _ -> true
    | D.Bfloat16, (Mxfp4 | Mxfp4_blocks) -> true
    | _ -> false
  in
  if exact then Nx.matmul x (Nx.matrix_transpose (Nx_quant.dequant dt w))
  else
    Nx.cast dt
      (Nx.matmul (Nx.cast Nx.float32 x)
         (Nx.matrix_transpose (Nx_quant.dequant Nx.float32 w)))

(* Known blocks. A block's bytes are written from the format's layout and its
   values from the format's formula. *)

let le16 h = [ h land 255; h lsr 8 ]
let repeat n l = List.concat (List.init n (fun _ -> l))

(* [block format b] is the weight of one row of one block of bytes [b]. *)
let block format b =
  let b = Array.of_list b in
  equal ~msg:"the block's size" int (block_bytes format) (Array.length b);
  weight_of format [| 1; Array.length b |] b

let q8_0_of d qs = block Q8_0 (le16 d @ List.map (fun q -> q land 255) qs)

(* A Q4_K block of float16 [d] and [dmin] whose sub-blocks 0 to 3 have scale 1
   and min 2, and 4 to 7 scale 58 and min 37, under quants of low nibble 15 and
   high nibble 9. *)
let q4_k_of d dmin =
  block Q4_K
    (le16 d @ le16 dmin @ repeat 4 [ 0xC1 ] @ repeat 4 [ 0x82 ]
   @ repeat 4 [ 0x5A ] @ repeat 128 [ 0x9F ])

let q4_k_values d dmin =
  let sub sc m q = List.init 32 (fun _ -> (d *. sc *. q) -. (dmin *. m)) in
  repeat 2 (sub 1. 2. 15. @ sub 1. 2. 9.)
  @ repeat 2 (sub 58. 37. 15. @ sub 58. 37. 9.)

(* A Q6_K block of float16 [d] and scales [-8] to [7], whose quarters of each
   half hold the quants 1, 19, 34 and 52: low nibbles 1 and 3 from the first and
   second 32 bytes, high nibbles 2 and 4, and high pairs 0 to 3. *)
let q6_k_quarters =
  block Q6_K
    (repeat 2 (repeat 32 [ 0x21 ] @ repeat 32 [ 0x43 ])
    @ repeat 64 [ 0xE4 ]
    @ List.init 16 (fun i -> (i - 8) land 255)
    @ le16 0xBC00)

(* An MXFP4 GGUF block of scale byte 128, 2: the codes 0 to 15 in bytes 0 to 7's
   low nibbles and 15 down to 0 in their high ones. *)
let mxfp4_block_of =
  block Mxfp4_blocks
    (128 :: List.init 16 (fun j -> j land 15 lor ((15 - j) lsl 4)))

let known =
  let zeros n = List.init n (fun _ -> 0.) in
  let code c = (if c < 8 then 1. else -1.) *. e2m1.(c land 7) in
  cases "a known block's values"
    ~name:(fun (n, _, _) -> n)
    [
      ("a zero Q8_0 block", q8_0_of 0 (repeat 32 [ 0 ]), zeros 32);
      ("a zero Q4_K block", block Q4_K (repeat 144 [ 0 ]), zeros 256);
      (* A zero quant is -32, and 0 * 0 * -32 is -0. *)
      ( "a zero Q6_K block",
        block Q6_K (repeat 210 [ 0 ]),
        List.init 256 (fun _ -> -0.) );
      ( "an MXFP4 GGUF block of every code, both ways",
        mxfp4_block_of,
        List.init 16 (fun j -> 2. *. code j)
        @ List.init 16 (fun j -> 2. *. code (15 - j)) );
      ( "Q8_0 at the largest scale, the int8 extremes",
        q8_0_of 0x7BFF (repeat 8 [ 127; -128; 0; 1 ]),
        repeat 8 [ 65504. *. 127.; 65504. *. -128.; 0.; 65504. ] );
      ( "Q8_0 at a negative scale",
        q8_0_of 0xB800 (repeat 16 [ -2; 3 ]),
        repeat 16 [ 1.; -1.5 ] );
      ( "Q4_K's packed scales and mins",
        q4_k_of 0x3C00 0x3800,
        q4_k_values 1. 0.5 );
      ( "Q4_K at a negative scale and min",
        q4_k_of 0xC000 0xBC00,
        q4_k_values (-2.) (-1.) );
      ( "Q4_K at the largest scales",
        q4_k_of 0x7BFF 0x7BFF,
        List.map to_f32 (q4_k_values 65504. 65504.) );
      ( "Q6_K at the largest scale, the extreme scale and quant",
        block Q6_K (repeat 192 [ 0 ] @ repeat 16 [ 0x80 ] @ le16 0x7BFF),
        repeat 256 [ 65504. *. -128. *. -32. ] );
      ( "Q6_K's quarters, at negative and positive scales",
        q6_k_quarters,
        List.concat
          (List.init 16 (fun s ->
               let q = [| -31.; -13.; 2.; 20. |].(s mod 8 / 2) in
               List.init 16 (fun _ -> -1. *. Float.of_int (s - 8) *. q))) );
    ]
    (fun (_, w, expected) ->
      equal (array float_exact) (Array.of_list expected)
        (Nx.to_array (Nx_quant.dequant Nx.float32 w)))

(* ggml's own values. golden/<format>.blocks holds a tensor of each format of 4
   rows of 512 values, and golden/<format>.values the float32 values gguf-py,
   llama.cpp's Python package, dequantises from it; gen/generate.py writes
   them. *)
let ggml =
  let read name = In_channel.with_open_bin name In_channel.input_all in
  cases "values are ggml's" ~name:format_name [ Mxfp4_blocks; Q8_0; Q4_K; Q6_K ]
    (fun f ->
      let name = match f with Mxfp4_blocks -> "mxfp4" | f -> format_name f in
      let b = read ("golden/" ^ name ^ ".blocks") in
      let v = read ("golden/" ^ name ^ ".values") in
      let w =
        weight_of f
          [| 4; 512 / block_values f * block_bytes f |]
          (Array.init (String.length b) (fun i -> Char.code b.[i]))
      in
      let expected =
        Array.init
          (String.length v / 4)
          (fun i -> Int32.float_of_bits (String.get_int32_le v (4 * i)))
      in
      (* ggml reads the MXFP4 code of sign 1 and magnitude 0 as 0., which the
         format's e2m1 reads as -0.: adding 0. makes -0. into 0. and keeps every
         other value. *)
      let got = Nx_quant.dequant Nx.float32 w in
      let got = if f = Mxfp4_blocks then Nx.add_s got 0. else got in
      equal (array float_exact) expected (Nx.to_array got))

(* Takes *)

(* A weight, an axis other than its last, counted from either end, and positions
   in range along it. *)
let takes =
  let open Gen in
  let pp ppf ((view, w), axis, p) =
    Format.fprintf ppf "@[<v>%a@,axis %d, positions [%s]@]" pp_weight (view, w)
      axis
      (String.concat "; " (List.map string_of_int p))
  in
  with_pp pp
    (let* ((_, w) as vw) =
       weight
         ~lead:(list ~size:(int_range 0 2) (int_range 1 3))
         ~n:(int_range 1 4) ()
     in
     let s = Nx_quant.shape w in
     let r = Array.length s in
     let* a, negative = pair (int_range 0 (r - 2)) bool in
     let+ p = list ~size:(int_range 0 4) (int_range 0 (s.(a) - 1)) in
     (vw, (if negative then a - r else a), p))

let positions p =
  Nx.create Nx.int64
    [| List.length p |]
    (Array.of_list (List.map Int64.of_int p))

let take_law ((_, w), axis, p) =
  cover "an axis counted from the end" (axis < 0);
  cover "no position" (p = []);
  cover "a repeated position"
    (List.length (List.sort_uniq compare p) < List.length p);
  let p = positions p in
  let taken = Nx_quant.take ~axis p w in
  List.iter
    (fun (F d) ->
      equal ~msg:(D.name d.dtype) (array int64)
        (bits (Nx.take ~axis p (Nx_quant.dequant d.dtype w)))
        (bits (Nx_quant.dequant d.dtype taken)))
    fdts

let values_and_products =
  group "values and products"
    [
      prop ~count:300
        "dequant gives each value of the format, computed at float32, at its \
         dtype"
        (Gen.pair (weight ()) fdt)
        (fun ((_, w), F d) ->
          let exact = values w in
          List.iter
            (fun f ->
              cover ("a " ^ format_name f ^ " weight") (format_of w = f))
            formats;
          cover "no value" (Array.length exact = 0);
          cover "an infinite value"
            (Array.exists (fun v -> Float.abs v = Float.infinity) exact);
          cover "a NaN value" (Array.exists Float.is_nan exact);
          cover "a -0. value"
            (Array.exists (fun v -> 1. /. v = Float.neg_infinity) exact);
          let got = Nx_quant.dequant d.dtype w in
          equal (array int) (Nx_quant.shape w) (Nx.shape got);
          equal (array float_exact) (Array.map d.round exact)
            (Nx.to_array (Nx.cast Nx.float64 got)));
      prop ~count:300 "apply is the product with the dequantised weight"
        (Gen.with_pp pp_product products)
        product_law;
      prop "apply is its equation, bit for bit"
        (Gen.with_pp pp_product products) (fun (_, w, X { x; _ }) ->
          equal (array int64) (bits (equation w x)) (bits (Nx_quant.apply w x)));
      test "a float64 input accumulates at float64" (fun () ->
          (* Every value of this weight is 1: each output sums the input. *)
          let w =
            Nx_quant.q8_0
              (u8 [| 1; 34 |]
                 (Array.init 34 (fun i ->
                      match i with 0 -> 0x00 | 1 -> 0x3C | _ -> 1)))
          in
          let tiny = Float.ldexp 1. (-40) in
          let x =
            Nx.create Nx.float64 [| 32 |]
              (Array.init 32 (fun i -> if i = 0 then 1. else tiny))
          in
          equal (array float_exact)
            [| 1. +. (31. *. tiny) |]
            (Nx.to_array (Nx_quant.apply w x)));
      prop "take is the gather of the dequantised weight at positions in range"
        takes take_law;
      cases ~name:format_name
        "a position out of range gathers zero bytes, which decode to zeros"
        formats (fun format ->
          let w = random_weight ~format [| 3; 4; 256 |] in
          let taken = Nx_quant.take ~axis:0 (positions [ -1; 1; 3 ]) w in
          let zero = if format = Q6_K then -0. else 0. in
          let row i t = Nx.to_array (Nx.slice [ Nx.I i ] t) in
          List.iter
            (fun t ->
              let none = Array.map (fun _ -> 0) (row 0 t) in
              equal ~msg:"part"
                (pair (array int) (array int))
                (none, none)
                (row 0 t, row 2 t))
            (parts taken);
          let got = Nx_quant.dequant Nx.float32 taken in
          equal (array float_exact) (Array.make (4 * 256) zero) (row 0 got);
          equal (array float_exact) (Array.make (4 * 256) zero) (row 2 got));
    ]

(* File layouts *)

(* Codes and scales of [rows] rows of [groups] groups. *)
let codes_and_scales =
  let open Gen in
  with_pp
    (fun ppf (rows, groups, _, _) ->
      Format.fprintf ppf "%d rows of %d groups" rows groups)
    (let* rows, groups = pair (int_range 0 3) (int_range 0 3) in
     let+ codes = array ~size:(constant (rows * groups * 32)) (int_range 0 15)
     and+ scales = array ~size:(constant (rows * groups)) (int_range 0 255) in
     (rows, groups, codes, scales))

let file_layouts =
  group "file layouts"
    [
      prop "both MXFP4 layouts of one weight decode alike, bit for bit"
        codes_and_scales (fun (rows, groups, codes, scales) ->
          let w =
            Nx_quant.mxfp4
              ~scales:(u8 [| rows; groups |] scales)
              (u8 [| rows; groups * 16 |] (checkpoint_bytes codes))
          in
          let g =
            Nx_quant.mxfp4_blocks
              (u8 [| rows; groups * 17 |] (gguf_blocks codes scales))
          in
          equal (array int64)
            (bits (Nx_quant.dequant Nx.float32 w))
            (bits (Nx_quant.dequant Nx.float32 g)));
      test "a constructor holds its bytes and allocates nothing" (fun () ->
          let rows = 64 and k = 4096 in
          let codes = Nx.copy (Nx.zeros Nx.uint8 [| rows; k / 2 |]) in
          let scales = Nx.copy (Nx.zeros Nx.uint8 [| rows; k / 32 |]) in
          let blocks = Nx.copy (Nx.zeros Nx.uint8 [| rows; k / 32 * 34 |]) in
          let before = Rig.allocated Rig.host in
          let w = Nx_quant.mxfp4 ~scales codes and q = Nx_quant.q8_0 blocks in
          equal ~msg:"allocated" int 0 (Rig.allocated Rig.host - before);
          equal ~msg:"its parts are the bytes" (pair bool bool) (true, true)
            ( (match w with
              | Mxfp4 r -> r.codes == codes && r.scales == scales
              | _ -> false),
              match q with Q8_0 r -> r.blocks == blocks | _ -> false ));
    ]

(* Construction *)

let errors =
  let w = random_weight [| 2; 5; 64 |] in
  let x = random_floats in
  let mxfp4 scales codes () =
    ignore
      (Nx_quant.mxfp4 ~scales:(Nx.zeros Nx.uint8 scales)
         (Nx.zeros Nx.uint8 codes))
  in
  let blocks f shape () = ignore (f (Nx.zeros Nx.uint8 shape)) in
  let halve t =
    if Nx.dim (-1) t = 2 then Nx.slice [ Nx.A; Nx.A; Nx.R (0, 1) ] t else t
  in
  let apply w x () = ignore (Nx_quant.apply w x) in
  let take axis () = ignore (Nx_quant.take ~axis (positions [ 0 ]) w) in
  cases "refuse, naming what is wrong"
    ~name:(fun (n, _, _) -> n)
    [
      ("mxfp4 codes of one axis", "codes", mxfp4 [| 2 |] [| 32 |]);
      ("mxfp4 k not a multiple of 32", "codes", mxfp4 [| 4; 1 |] [| 4; 8 |]);
      ("mxfp4 a scale per 16 values", "scales", mxfp4 [| 4; 4 |] [| 4; 32 |]);
      ( "mxfp4 scales without a leading axis",
        "scales",
        mxfp4 [| 4; 1 |] [| 2; 4; 16 |] );
      ("q8_0 blocks of one axis", "blocks", blocks Nx_quant.q8_0 [| 34 |]);
      ( "mxfp4_blocks blocks of one axis",
        "blocks",
        blocks Nx_quant.mxfp4_blocks [| 17 |] );
      ( "mxfp4_blocks a row of a part of a block",
        "blocks",
        blocks Nx_quant.mxfp4_blocks [| 2; 18 |] );
      ( "q8_0 a row of a part of a block",
        "blocks",
        blocks Nx_quant.q8_0 [| 2; 68 + 32 |] );
      ( "q4_k a row of a part of a block",
        "blocks",
        blocks Nx_quant.q4_k [| 2; 145 |] );
      ( "q6_k a row of Q4_K's blocks",
        "blocks",
        blocks Nx_quant.q6_k [| 2; 144 |] );
      ( "a map that changes a part's shape",
        "Nx_quant.walk",
        fun () -> ignore (Nx.Ptree.map description (fun _ t -> halve t) w) );
      ( "apply to a 0-d value",
        "at least one axis",
        apply w (Nx.scalar Nx.float32 1.) );
      ( "apply to a last axis of another size",
        "last axis",
        apply w (x [| 3; 32 |]) );
      ( "apply over batch axes that do not broadcast",
        "broadcast",
        apply w (x [| 3; 1; 64 |]) );
      ("take along the last axis", "last axis", take 2);
      ("take along the last axis, counted from the end", "last axis", take (-1));
      ("take along an axis past the weight's", "out of bounds", take 3);
    ]
    (fun (_, part, f) -> raises_match (Exn.invalid_arg ~substring:part) f)

(* Placement *)

(* Four memory devices, computed by nx.cpu. *)
module Mem = (val Nx.devices (List.init 4 Nx_support.memory))

let mesh = Nx.Mesh.v Mem.v [ ("a", 2); ("b", 2) ]

(* A placement of a weight's parts: a cut of one axis over four devices or over
   two, with its number of tiles, or whole on each. *)
type cut = Whole | Cut of { axis : int; tiles : int }

let at = function
  | Whole -> Mem.on
  | Cut { axis; tiles = 4 } -> Mem.split ~axis
  | Cut { axis; _ } -> Nx.Placement.mesh mesh [ (axis, [ "b" ]) ]

let pp_cut ppf = function
  | Whole -> Format.pp_print_string ppf "whole"
  | Cut { axis; tiles } -> Format.fprintf ppf "axis %d in %d" axis tiles

let placed =
  let open Gen in
  with_pp
    (fun ppf ((view, w), c) ->
      Format.fprintf ppf "@[<v>%a@,at %a@]" pp_weight (view, w) pp_cut c)
    (let* ((_, w) as vw) =
       weight
         ~lead:(list ~size:(int_range 0 1) (extent 4))
         ~count:(Some (extent 6))
         ()
     in
     let r = Array.length (Nx_quant.shape w) in
     let+ axis = frequency [ (2, constant (r - 1)); (1, int_range 0 (r - 1)) ]
     and+ tiles = of_list [ 2; 4 ]
     and+ whole = frequency [ (1, constant true); (8, constant false) ] in
     (vw, if whole then Whole else Cut { axis; tiles }))

let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

(* The placement the interface gives dequant's result, where it gives one: a cut
   along [n] or a leading axis kept; one along [k] of a weight with values kept
   where its tiles divide the blocks, and the result whole on each device where
   they do not and the blocks are two or more. *)
let decoded w = function
  | Whole -> Some Whole
  | Cut { axis; tiles } as c ->
      let s = Nx_quant.shape w in
      let r = Array.length s in
      let blocks = s.(r - 1) / block_values (format_of w) in
      let values = Array.fold_left ( * ) 1 s > 0 in
      if axis < r - 1 then Some c
      else if values && blocks mod tiles = 0 then Some c
      else if values && blocks >= 2 then Some Whole
      else None

let placement_law ((_, w), c) =
  let r = Array.length (Nx_quant.shape w) in
  let blocks = (Nx_quant.shape w).(r - 1) / block_values (format_of w) in
  List.iter
    (fun f -> cover ("a " ^ format_name f ^ " weight") (format_of w = f))
    formats;
  let along_k t =
    match c with
    | Cut { axis; tiles } -> axis = r - 1 && t tiles
    | Whole -> false
  in
  cover "a cut of k whose tiles divide the blocks"
    (along_k (fun t -> blocks > 0 && blocks mod t = 0));
  cover "a cut of k whose tiles do not divide two blocks or more"
    (along_k (fun t -> blocks >= 2 && blocks mod t <> 0));
  cover "a cut of one block along k" (along_k (fun _ -> blocks = 1));
  cover "an MXFP4 checkpoint weight cut across its groups"
    (format_of w = Mxfp4 && along_k (fun t -> blocks >= 2 && blocks mod t <> 0));
  cover "a weight with no values"
    (Array.fold_left ( * ) 1 (Nx_quant.shape w) = 0);
  cover "a cut of n or a leading axis"
    (match c with Cut { axis; _ } -> axis < r - 1 | Whole -> false);
  let wp = Nx_quant.place (at c) w in
  let got = Nx_quant.dequant Nx.float32 wp in
  equal ~msg:"its values" (array int64)
    (bits (Nx_quant.dequant Nx.float32 w))
    (bits got);
  Option.iter
    (fun d ->
      equal ~msg:"where dequant's result lies" placement (at d)
        (Option.get (Nx.placement got)))
    (decoded w c);
  let _, _, k = dims w in
  let x = random_floats [| 3; k |] in
  equal ~msg:"apply" (array int64)
    (product_bits (Nx_quant.apply w x))
    (product_bits (Nx_quant.apply wp (Nx.place Mem.on x)))

(* Work. Kernels that count the positions their maps compute, over four memory
   devices and over one, show where a weight decodes: each device all of [k], or
   its own blocks alone. *)

module C = Nx_support.Counting

module Counted =
  (val Nx.devices ~kernels:(module C) (List.init 4 Nx_support.memory))

module Counted_one =
  (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

module Counted_two =
  (val Nx.devices
         ~kernels:(module C)
         [ Nx_support.memory 0; Nx_support.memory 1 ])

(* The positions [dequant] computes over the weight [w] at [p]. *)
let decoding p w =
  let w = Nx_quant.place p w in
  C.reset ();
  ignore (Nx.to_array (Nx_quant.dequant Nx.float32 w));
  C.positions ()

let work =
  let open Gen in
  with_pp
    (fun ppf (f, n, b, t) ->
      Format.fprintf ppf "%a, %d rows of %d blocks, over %d devices" pp_format f
        n b t)
    (quad format (int_range 1 3) (int_range 2 6) (of_list [ 2; 4 ]))

(* The positions per row of a weight of [n] rows: the difference between [2 n]
   rows and [n] leaves out the constants each device computes. *)
let per_row f n b p =
  let row n = random_weight ~format:f [| n; b * block_values f |] in
  decoding p (row (2 * n)) - decoding p (row n)

let work_law (f, n, b, t) =
  List.iter
    (fun f' -> cover ("a " ^ format_name f' ^ " weight") (f = f'))
    formats;
  cover "tiles that divide the blocks" (b mod t = 0);
  cover "tiles that do not divide the blocks" (b mod t <> 0);
  cover "an MXFP4 weight's tiles that do not divide its groups"
    (f = Mxfp4 && b mod t <> 0);
  let one = per_row f n b Counted_one.on in
  let cut =
    if t = 4 then per_row f n b (Counted.split ~axis:1)
    else per_row f n b (Counted_two.split ~axis:1)
  in
  equal ~msg:"the positions per row over one device's" (pair int int)
    ((if b mod t = 0 then 1 else t), 0)
    (cut / one, cut mod one)

(* [routed w ids x] routes each row of [x] [[| p; k |]] through the expert of
   [w] [[| e; n; k |]] its id names. *)
let routed w ids x =
  let e = (Nx_quant.shape w).(0) in
  Nx.map_segments ~segments:e ids
    (fun owners rows -> Nx_quant.apply (Nx_quant.take ~axis:0 owners w) rows)
    x

let placements =
  group "placement"
    [
      prop ~count:300
        "a weight placed anywhere gives the host's bits, where the interface \
         says"
        placed placement_law;
      prop
        "a cut along k decodes each block once where its tiles divide the \
         blocks, and all of k on each device otherwise"
        work work_law;
      cases ~name:format_name
        "experts split over four devices give the host's routed product" formats
        (fun format ->
          let w = random_weight ~format [| 4; 3; 256 |] in
          let ids =
            Nx.create Nx.int64 [| 12 |]
              (Array.init 12 (fun i -> Int64.of_int ((i * 5 mod 6) - 1)))
          in
          let x = random_floats [| 12; 256 |] in
          let on t = Nx.place Mem.on t in
          equal (array int64)
            (bits (routed w ids x))
            (bits
               (routed (Nx_quant.place (Mem.split ~axis:0) w) (on ids) (on x))));
    ]

(* Structure *)

let structure =
  group "structure"
    [
      test "visits the case, then codes before scales" (fun () ->
          equal (list string)
            [ "the root: case \"mxfp4\""; "codes: a leaf"; "scales: a leaf" ]
            (List.map
               (Format.asprintf "%a" Nx.Ptree.pp_visit)
               (Nx.Ptree.visits description (random_weight [| 3; 4; 64 |]))));
      cases ~name:format_name "visits a block format's case, then its blocks"
        [ Mxfp4_blocks; Q8_0; Q4_K; Q6_K ] (fun format ->
          equal (list string)
            [
              Printf.sprintf "the root: case %S" (format_name format);
              "blocks: a leaf";
            ]
            (List.map
               (Format.asprintf "%a" Nx.Ptree.pp_visit)
               (Nx.Ptree.visits description
                  (random_weight ~format [| 3; 256 |]))));
      test "a walk that keeps each part rebuilds the same parts" (fun () ->
          let w = random_weight [| 3; 4; 64 |] in
          let w' = Nx.Ptree.map description (fun _ t -> t) w in
          equal bool true (List.for_all2 ( == ) (parts w) (parts w')));
    ]

let () =
  exit
    (run "nx quant"
       [
         known;
         ggml;
         values_and_products;
         file_layouts;
         errors;
         placements;
         structure;
       ])
