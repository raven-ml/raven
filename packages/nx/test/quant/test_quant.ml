(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantised weights against their formats' definitions: dequant gives each
   value at its dtype, as the format's reference decodes it from its bytes,
   apply is the product with the dequantised weight within the error of a
   float32 sum, and take is the gather of the dequantised weight. *)

open Windtrap
open Nx_test
module S = Nx_dtype.Scalar

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* The formats *)

type format = Mxfp4 | Q8_0 | Q4_K | Q6_K

let formats = [ Mxfp4; Q8_0; Q4_K; Q6_K ]

let format_name = function
  | Mxfp4 -> "mxfp4"
  | Q8_0 -> "q8_0"
  | Q4_K -> "q4_k"
  | Q6_K -> "q6_k"

let pp_format ppf f = Format.pp_print_string ppf (format_name f)

(* The values and the bytes of a block, and the offsets of its float16
   scales. *)
let block_values = function Mxfp4 | Q8_0 -> 32 | Q4_K | Q6_K -> 256

let block_bytes = function
  | Mxfp4 -> 16
  | Q8_0 -> 34
  | Q4_K -> 144
  | Q6_K -> 210

let halves = function
  | Mxfp4 -> []
  | Q8_0 -> [ 0 ]
  | Q4_K -> [ 0; 2 ]
  | Q6_K -> [ 208 ]

let format_of = function
  | Nx_quant.Mxfp4 _ -> Mxfp4
  | Nx_quant.Q8_0 _ -> Q8_0
  | Nx_quant.Q4_K _ -> Q4_K
  | Nx_quant.Q6_K _ -> Q6_K

(* [weight_of format shape bytes] is the weight of [format] whose quants,
   [shape] up to its last axis, are [bytes], with an MXFP4 weight's scales
   [scales]. *)
let weight_of ?scales format shape bytes =
  let blocks = Nx.create Nx.uint8 shape bytes in
  match format with
  | Mxfp4 -> Nx_quant.mxfp4 ~scales:(Option.get scales) blocks
  | Q8_0 -> Nx_quant.q8_0 blocks
  | Q4_K -> Nx_quant.q4_k blocks
  | Q6_K -> Nx_quant.q6_k blocks

(* MXFP4: a code's magnitude, signed, times [2 ^ (s - 127)] for its group's
   scale byte [s], NaN for [255]. The codes hold one value each, in order. *)

let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |]

let mxfp4_values codes scales =
  let scales = Nx.to_array scales in
  Array.mapi
    (fun i code ->
      let s = scales.(i / 32) in
      let m = e2m1.(code land 7) *. Float.ldexp 1. (s - 127) in
      if s = 255 then Float.nan else if code < 8 then m else -.m)
    (Nx.to_array codes)

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

let f16 b at = S.decode S.Float16 (b at lor (b (at + 1) lsl 8))
let int8 v = if v >= 128 then v - 256 else v

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

(* The values of [w] in row-major order, at float32. *)
let values = function
  | Nx_quant.Mxfp4 { codes; scales } -> mxfp4_values codes scales
  | Nx_quant.Q8_0 { blocks } -> gguf_values q8_0_block 34 blocks
  | Nx_quant.Q4_K { blocks } -> gguf_values q4_k_block 144 blocks
  | Nx_quant.Q6_K { blocks } -> gguf_values q6_k_block 210 blocks

(* A float dtype, the rounding of a value computed at float32 to it, and the
   error of that rounding: [rel] of the value, or [tiny] below its least
   normal. *)
type fdt =
  | F : {
      dtype : (float, 'b) Nx.dtype;
      round : float -> float;
      rel : float;
      tiny : float;
    }
      -> fdt

let narrow dtype s rel tiny =
  let round x =
    if Float.is_nan x then x else S.decode s (S.encode s (to_f32 x))
  in
  F { dtype; round; rel = Float.ldexp 1. rel; tiny = Float.ldexp 1. tiny }

let fdts =
  [
    F { dtype = Nx.float32; round = to_f32; rel = 0.; tiny = 0. };
    narrow Nx.bfloat16 S.BFloat16 (-8) (-134);
    narrow Nx.float16 S.Float16 (-11) (-25);
    F { dtype = Nx.float64; round = to_f32; rel = 0.; tiny = 0. };
  ]

let fdt =
  Gen.of_list
    ~pp:(fun ppf (F d) ->
      Format.pp_print_string ppf (Nx_dtype.to_string d.dtype))
    fdts

(* Weights *)

(* A view of every part that keeps them a weight, where its logical shape and
   the values of its blocks allow it. A part counts its blocks along its last
   axis, but MXFP4's codes, which hold a group in their last two. *)
type view = {
  view : string;
  fits : int array -> int -> bool;
  apply : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t;
}

let blocks_axis (type a b) (t : (a, b) Nx.t) =
  match Nx.dtype t with Nx.UInt4 -> Nx.ndim t - 3 | _ -> Nx.ndim t - 1

let views =
  let r = Array.length in
  [
    { view = "contiguous"; fits = (fun _ _ -> true); apply = Fun.id };
    {
      view = "every other row";
      fits = (fun s _ -> s.(r s - 2) > 0);
      apply =
        (fun t ->
          Nx.squeeze ~axes:[ -1 ]
            (Nx.sliding_window ~axis:(blocks_axis t - 1) ~window:1 ~step:2 t));
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
          let a = blocks_axis t in
          Nx.shrink
            (Array.mapi
               (fun i d -> if i = a then (0, d / 2) else (0, d))
               (Nx.shape t))
            t);
    };
  ]

let pp_weight ppf (view, w) =
  Format.fprintf ppf "@[<v>%a %a, %s" pp_format (format_of w) pp_shape
    (Nx_quant.shape w) view;
  Nx.Ptree.fold Nx_quant.ptree
    (fun path t () ->
      Format.fprintf ppf "@,%a %a" Nx.Ptree.Path.pp path Nx.pp t)
    w ();
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
let binary16 =
  Gen.frequency
    [
      (8, Gen.map (S.encode S.Float16) (Gen.float_range (-4.) 4.));
      ( 1,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "0x%04x")
          [ 0x0000; 0x8000; 0x7BFF; 0xFBFF; 0x0001; 0x83FF; 0x7C00; 0x7E00 ] );
    ]

(* [gguf_bytes format ~scale count] is [count] blocks of [format]'s bytes,
   random but for their float16 scales, drawn from [scale]. *)
let gguf_bytes format ~scale count =
  let open Gen in
  let bytes = block_bytes format and fields = halves format in
  let+ b = array ~size:(constant (count * bytes)) (int_range 0 255)
  and+ h = array ~size:(constant (count * List.length fields)) scale in
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

(* A weight of shape [[| lead...; n; k |]] under a view, an MXFP4 one read from
   either file's layout. *)
let weight ?(lead = Gen.list ~size:(Gen.int_range 0 2) (Gen.int_range 0 3))
    ?(n = Gen.int_range 0 4) ?(format = format) () =
  let open Gen in
  with_pp pp_weight
    (let* f = format in
     let* lead, n, count =
       triple lead n (int_range 1 (if block_values f = 32 then 4 else 2))
     in
     let m = List.fold_left ( * ) 1 lead * n in
     let shape last = Array.of_list (lead @ [ n; last ]) in
     let+ file, w =
       match f with
       | Mxfp4 ->
           let+ codes = array ~size:(constant (m * count * 32)) (int_range 0 15)
           and+ scales = array ~size:(constant (m * count)) e8m0
           and+ gguf = bool in
           if gguf then
             ( ", from GGUF blocks",
               Nx_quant.mxfp4_blocks
                 (Nx.create Nx.uint8
                    (shape (count * 17))
                    (gguf_blocks codes scales)) )
           else
             ( "",
               Nx_quant.mxfp4
                 ~scales:(Nx.create Nx.uint8 (shape count) scales)
                 (Nx.create Nx.uint8
                    (shape (count * 16))
                    (checkpoint_bytes codes)) )
       | f ->
           let+ b = gguf_bytes f ~scale:binary16 (m * count) in
           ("", weight_of f (shape (count * block_bytes f)) b)
     and+ v = of_list views in
     if v.fits (Nx_quant.shape w) (block_values f) then
       (v.view ^ file, Nx.Ptree.map Nx_quant.ptree (fun _ t -> v.apply t) w)
     else ("contiguous" ^ file, w))

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
        List.iter
          (fun at ->
            let h = S.encode S.Float16 (Random.State.float rng 2. -. 1.) in
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
let agrees ?(at = List.hd fdts) ~k expected bound actual =
  let (F d) = at in
  equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape actual);
  let b = Nx.to_array bound and a = Nx.to_array (Nx.cast Nx.float64 actual) in
  let worst = ref 0. in
  Array.iteri
    (fun i e ->
      if not (Float.is_finite (d.round e)) then
        equal float_exact (d.round e) a.(i)
      else
        let tol =
          2. *. float_of_int k
          *. ((Float.ldexp 1. (-24) *. b.(i)) +. Float.ldexp 1. (-149))
          +. Float.max (d.rel *. Float.abs e) d.tiny
        in
        let err = Float.abs (a.(i) -. e) in
        let ratio = if err = 0. then 0. else err /. tol in
        worst :=
          Float.max !worst
            (if Float.is_nan ratio then Float.infinity else ratio))
    (Nx.to_array expected);
  at_most ~msg:"worst error over its bound" float_exact ~than:1. !worst

(* The product of [x] with each matrix of [w'], transposed, at float32, and the
   same product of magnitudes. *)
let product x w' =
  let x = Nx.cast Nx.float32 x in
  let t = Nx.matrix_transpose in
  (Nx.matmul x (t w'), Nx.matmul (Nx.abs x) (t (Nx.abs w')))

(* An input at one of the float dtypes. *)
type input = X : { at : fdt; x : (float, 'b) Nx.t } -> input

let flag = Gen.of_list ~pp:Format.pp_print_bool [ false; true ]

let usually =
  Gen.frequency [ (3, Gen.constant ~pp:Format.pp_print_bool true); (1, flag) ]

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
            [ Float.nan; Float.infinity; Float.neg_infinity ] );
      ]
  in
  let+ at = fdt
  and+ xs = array ~size:(constant (Ref.numel shape)) entry
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
          (Nx.slice (List.init (r - 3) (fun _ -> Nx.A) @ [ R (0, 1) ]) x)
    | "flipped" -> Nx.flip x
    | _ -> x
  in
  X { at; x }

(* A weight and an input. *)
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
  let y = Nx_quant.apply w x in
  equal ~msg:"dtype" string
    (Nx_dtype.to_string (Nx.dtype x))
    (Nx_dtype.to_string (Nx.dtype y));
  agrees ~at ~k expected bound y

(* Known blocks. A block's bytes are written from the format's layout and its
   values from the format's formula. *)

let le16 h = [ h land 255; h lsr 8 ]
let repeat n l = List.concat (List.init n (fun _ -> l))

(* [block format bytes] is the weight of one row of one block of [bytes]. *)
let block format bytes =
  let b = Array.of_list bytes in
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

let known =
  let zeros n = List.init n (fun _ -> 0.) in
  cases "a known block's values"
    ~name:(fun (n, _, _) -> n)
    [
      ("a zero Q8_0 block", q8_0_of 0 (repeat 32 [ 0 ]), zeros 32);
      ("a zero Q4_K block", block Q4_K (repeat 144 [ 0 ]), zeros 256);
      (* A zero quant is -32, and 0 * 0 * -32 is -0. *)
      ( "a zero Q6_K block",
        block Q6_K (repeat 210 [ 0 ]),
        List.init 256 (fun _ -> -0.) );
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

(* ggml's own values. golden/ggml.gguf holds a tensor of each format and the
   values gguf-py, llama.cpp's Python package, dequantises from it; gen/
   generate.py writes it. *)
let ggml =
  cases "values are ggml's" ~name:format_name [ Q8_0; Q4_K; Q6_K ] (fun f ->
      let g = Nx_io.load_gguf "golden/ggml.gguf" in
      let name = format_name f in
      let info = Nx_io.Gguf.info name g in
      let stored =
        Nx_io.Archive.tensor
          ~shape:
            [|
              info.shape.(0); info.shape.(1) / block_values f * block_bytes f;
            |]
          Nx.uint8 name (Nx_io.Gguf.tensors g)
      in
      let w =
        match info.dtype with
        | Q8_0 -> Nx_quant.q8_0 stored
        | Q4_K -> Nx_quant.q4_k stored
        | Q6_K -> Nx_quant.q6_k stored
        | _ -> failf "%s: stored as another type" name
      in
      equal (tensor float_exact)
        (Nx_io.Archive.tensor ~shape:info.shape Nx.float32 (name ^ ".values")
           (Nx_io.Gguf.tensors g))
        (Nx_quant.dequant Nx.float32 w))

(* Takes *)

(* The parts of [w] at [index] along their first axis, as integers. *)
let parts_at index w =
  Nx.Ptree.fold Nx_quant.ptree
    (fun _ t acc ->
      Nx.to_array (Nx.cast Nx.uint8 (Nx.slice [ I index ] t)) :: acc)
    w []

(* [bits t] is each element's bits, so -0 and +0 differ and a NaN is its
   payload. *)
let bits (type b) (t : (float, b) Nx.t) =
  match Nx.dtype t with
  | Nx.Float64 -> Nx.to_array (Nx.bitcast Nx.int64 t)
  | Nx.Float32 -> Array.map Int64.of_int32 (Nx.to_array (Nx.bitcast Nx.int32 t))
  | Nx.Float16 | Nx.BFloat16 ->
      Array.map Int64.of_int (Nx.to_array (Nx.bitcast Nx.uint16 t))
  | dt -> failf "no bits for %s" (Nx_dtype.to_string dt)

(* A weight, an axis other than its last, counted from either end, and indices
   in range along it. *)
let takes =
  let open Gen in
  let pp ppf ((view, w), axis, indices) =
    Format.fprintf ppf "@[<v>%a@,axis %d, indices [%s]@]" pp_weight (view, w)
      axis
      (String.concat "; " (List.map string_of_int indices))
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
     let+ indices = list ~size:(int_range 0 4) (int_range 0 (s.(a) - 1)) in
     (vw, (if negative then a - r else a), indices))

let take_law ((_, w), axis, indices) =
  let indices =
    Nx.create Nx.int64
      [| List.length indices |]
      (Array.of_list (List.map Int64.of_int indices))
  in
  cover "an axis counted from the end" (axis < 0);
  cover "no index" (Nx.numel indices = 0);
  cover "a repeated index"
    (List.length (List.sort_uniq compare (Array.to_list (Nx.to_array indices)))
    < Nx.numel indices);
  let taken = Nx_quant.take ~axis ~indices w in
  List.iter
    (fun (F d) ->
      equal
        ~msg:(Nx_dtype.to_string d.dtype)
        (array int64)
        (bits (Nx.take ~axis ~indices (Nx_quant.dequant d.dtype w)))
        (bits (Nx_quant.dequant d.dtype taken)))
    fdts

let values_and_products =
  group "values and products"
    [
      prop
        "dequant gives each value of the format, computed at float32, at its \
         dtype"
        (Gen.pair (weight ()) fdt)
        (fun ((_, w), F d) ->
          let exact = values w in
          List.iter
            (fun f ->
              cover ("a " ^ format_name f ^ " weight") (format_of w = f))
            formats;
          cover "an infinite value at float32"
            (Array.exists (fun v -> Float.abs v > 0x1.fffffep127) exact);
          cover "a NaN value" (Array.exists Float.is_nan exact);
          let got = Nx_quant.dequant d.dtype w in
          equal
            (pair (array int) (array float_exact))
            (Nx_quant.shape w, Array.map d.round exact)
            (Nx.shape got, Nx.to_array (Nx.cast Nx.float64 got)));
      prop "apply is the product with the dequantised weight"
        (Gen.with_pp pp_product products)
        product_law;
      test "a float64 input accumulates at float64" (fun () ->
          (* Every value of this weight is 1: each output sums the input. *)
          let w =
            Nx_quant.q8_0
              (Nx.create Nx.uint8 [| 1; 34 |]
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
      prop "take is the gather of the dequantised weight at indices in range"
        takes take_law;
      cases ~name:format_name
        "an index out of range gathers zero bytes, which decode to zeros"
        formats (fun format ->
          let w = random_weight ~format [| 3; 4; 256 |] in
          let indices = Nx.create Nx.int64 [| 3 |] [| -1L; 1L; 3L |] in
          let zero = if format = Q6_K then -0. else 0. in
          let taken = Nx_quant.take ~axis:0 ~indices w in
          let got = Nx_quant.dequant Nx.float32 taken in
          let zeros = List.map (Array.map (fun _ -> 0)) (parts_at 0 w) in
          equal ~msg:"parts" (list (array int)) zeros (parts_at 0 taken);
          equal ~msg:"parts" (list (array int)) zeros (parts_at 2 taken);
          equal (array float_exact)
            (Array.make (4 * 256) zero)
            (Nx.to_array (Nx.slice [ I 0 ] got));
          equal (array float_exact)
            (Array.make (4 * 256) zero)
            (Nx.to_array (Nx.slice [ I 2 ] got)));
    ]

(* Routing and bfloat16 *)

(* [routed w ids x] routes each row of [x] [[| p; k |]] through the expert of
   [w] [[| e; n; k |]] its id names. *)
let routed w ids x =
  let e = (Nx_quant.shape w).(0) in
  Nx.map_segments ~segments:e ids
    (fun owners rows ->
      Nx_quant.apply (Nx_quant.take ~axis:0 ~indices:owners w) rows)
    x

let routing =
  group "routing and bfloat16"
    [
      cases ~name:format_name
        "experts split over two devices give one device's routed product, eager"
        formats (fun format ->
          let w = random_weight ~format [| 4; 3; 256 |] in
          let ids =
            Nx.create Nx.int64 [| 12 |]
              (Array.init 12 (fun i -> Int64.of_int ((i * 5 mod 6) - 1)))
          in
          let x = random_floats [| 12; 256 |] in
          let split = Nx.Placement.sharded ~axis:0 [ Devices.d1; Devices.d2 ] in
          let placed = Nx.Ptree.place Nx_quant.ptree split w in
          equal (array float_exact)
            (Nx.to_array (routed w ids x))
            (Nx.to_array (Nx.place Nx.Placement.host (routed placed ids x))));
      cases ~name:string_of_int
        "at bfloat16, an MXFP4 product is the bfloat16 product of its bfloat16 \
         values, bit for bit"
        [ 32; 256; 2048 ] (fun k ->
          let w = random_weight [| 16; k |] in
          let x = Nx.cast Nx.bfloat16 (random_floats [| 8; k |]) in
          equal (array int64)
            (bits
               (Nx.matmul x
                  (Nx.matrix_transpose (Nx_quant.dequant Nx.bfloat16 w))))
            (bits (Nx_quant.apply w x)));
    ]

(* File layouts. Each constructor reads its file's bytes as the codes, one per
   value, and the scales. *)

(* Codes and scales of [rows] rows of [groups] groups. *)
let codes_and_scales =
  let open Gen in
  let* rows, groups = pair (int_range 0 3) (int_range 0 3) in
  let+ codes = array ~size:(constant (rows * groups * 32)) (int_range 0 15)
  and+ scales = array ~size:(constant (rows * groups)) (int_range 0 255) in
  (rows, groups, codes, scales)

let pp_codes ppf (rows, groups, _, _) =
  Format.fprintf ppf "%d rows of %d groups" rows groups

let parts = function
  | Nx_quant.Mxfp4 { codes; scales } -> (Nx.to_array codes, Nx.to_array scales)
  | w -> failf "a %s weight" (format_name (format_of w))

let file_layouts =
  group "file layouts"
    [
      prop "mxfp4 reads a checkpoint's byte i as values 2 i and 2 i + 1"
        (Gen.with_pp pp_codes codes_and_scales)
        (fun (rows, groups, codes, scales) ->
          let w =
            Nx_quant.mxfp4
              ~scales:(Nx.create Nx.uint8 [| rows; groups |] scales)
              (Nx.create Nx.uint8
                 [| rows; groups * 16 |]
                 (checkpoint_bytes codes))
          in
          equal (pair (array int) (array int)) (codes, scales) (parts w));
      prop
        "mxfp4_blocks reads a GGUF block as its scale, then values j and j + \
         16 in byte j" (Gen.with_pp pp_codes codes_and_scales)
        (fun (rows, groups, codes, scales) ->
          let w =
            Nx_quant.mxfp4_blocks
              (Nx.create Nx.uint8
                 [| rows; groups * 17 |]
                 (gguf_blocks codes scales))
          in
          equal (pair (array int) (array int)) (codes, scales) (parts w));
      test "both layouts give codes of shape [...; n; k / 32; 2; 16]" (fun () ->
          let shape = function
            | Nx_quant.Mxfp4 { codes; _ } -> Nx.shape codes
            | w -> failf "a %s weight" (format_name (format_of w))
          in
          equal (array int) [| 2; 3; 4; 2; 16 |]
            (shape (random_weight [| 2; 3; 128 |]));
          equal (array int) [| 2; 3; 4; 2; 16 |]
            (shape (Nx_quant.mxfp4_blocks (Nx.zeros Nx.uint8 [| 2; 3; 68 |]))));
      test
        "mxfp4_blocks of a GGUF file's MXFP4 tensor views the file on the disk \
         and gives ggml's values" (fun () ->
          let g = Nx_io.load_gguf "golden/ggml.gguf" in
          let info = Nx_io.Gguf.info "mxfp4" g in
          let tensors = Nx_io.Gguf.tensors g in
          let stored =
            Nx_io.Archive.tensor
              ~shape:[| info.shape.(0); info.shape.(1) / 32 * 17 |]
              Nx.uint8 "mxfp4" tensors
          in
          let w = Nx_quant.mxfp4_blocks stored in
          let codes, scales =
            match w with
            | Nx_quant.Mxfp4 { codes; scales } -> (codes, scales)
            | w -> failf "a %s weight" (format_name (format_of w))
          in
          let where t = Format.asprintf "%a" Nx.Placement.pp (Nx.placement t) in
          equal ~msg:"placements" (pair string string)
            (where stored, where stored)
            (where codes, where scales);
          equal ~msg:"one storage" (pair bool bool) (true, true)
            ( Devices.storage_of codes == Devices.storage_of stored,
              Devices.storage_of scales == Devices.storage_of stored );
          (* ggml reads the code of sign 1 and magnitude 0 as 0., which the
             format's e2m1 reads as -0.: adding 0. makes -0. into 0. and keeps
             every other value. *)
          equal (tensor float_exact)
            (Nx_io.Archive.tensor ~shape:info.shape Nx.float32 "mxfp4.values"
               tensors)
            (Nx.add_s (Nx_quant.dequant Nx.float32 w) 0.));
      test "mxfp4 and mxfp4_blocks view their bytes, allocating nothing"
        (fun () ->
          let host = Nx.Device.memory Nx.Device.host in
          let rows = 64 and k = 4096 in
          (* Bytes of storage of their own, as a file's are. *)
          let b = Nx.copy (Nx.zeros Nx.uint8 [| rows; k / 2 |]) in
          let scales = Nx.copy (Nx.zeros Nx.uint8 [| rows; k / 32 |]) in
          let blocks = Nx.copy (Nx.zeros Nx.uint8 [| rows; k / 32 * 17 |]) in
          let made f =
            let before = Nx_device.stats host in
            let w = f () in
            let s = Nx_device.Stats.diff before (Nx_device.stats host) in
            (w, Nx_device.Stats.allocated s)
          in
          let shares t u = share_memory (storage t) (storage u) in
          let views b = function
            | Nx_quant.Mxfp4 { codes; scales } ->
                (shares codes b, shares scales b)
            | w -> failf "a %s weight" (format_name (format_of w))
          in
          let w, allocated = made (fun () -> Nx_quant.mxfp4 ~scales b) in
          equal ~msg:"mxfp4 allocates" int 0 allocated;
          equal ~msg:"mxfp4's codes share b" bool true (fst (views b w));
          let w, allocated = made (fun () -> Nx_quant.mxfp4_blocks blocks) in
          equal ~msg:"mxfp4_blocks allocates" int 0 allocated;
          equal ~msg:"mxfp4_blocks's codes and scales share its blocks"
            (pair bool bool) (true, true) (views blocks w));
    ]

(* Construction and placement *)

open Devices

let errors =
  let w = random_weight [| 2; 5; 64 |] in
  let x = random_floats in
  let mxfp4 scales codes () =
    ignore
      (Nx_quant.mxfp4 ~scales:(Nx.zeros Nx.uint8 scales)
         (Nx.zeros Nx.uint8 codes))
  in
  let blocks f ?(devices = false) shape () =
    let b = Nx.zeros Nx.uint8 shape in
    let b =
      if devices then Nx.place (Nx.Placement.sharded ~axis:1 [ d1; d2 ]) b
      else b
    in
    ignore (f b)
  in
  let halve (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    if Nx.dim (-1) t = 2 then Nx.slice [ A; A; R (0, 1) ] t else t
  in
  let merge (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    if Nx.ndim t = 5 then Nx.reshape [| 2; 5; 2; 32 |] t else t
  in
  (* Each device holds half of each group's codes. *)
  let halves (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    match Nx.dtype t with
    | Nx.UInt4 -> Nx.place (Nx.Placement.sharded ~axis:4 [ d1; d2 ]) t
    | _ -> Nx.place (Nx.Placement.replicated [ d1; d2 ]) t
  in
  let apply w x () = ignore (Nx_quant.apply w x) in
  let take axis () =
    ignore (Nx_quant.take ~axis ~indices:(Nx.zeros Nx.int64 [| 1 |]) w)
  in
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
        "Nx_quant.ptree",
        fun () -> ignore (Nx.Ptree.map Nx_quant.ptree (fun _ t -> halve t) w) );
      ( "a map that merges a group's codes into one axis",
        "codes",
        fun () -> ignore (Nx.Ptree.map Nx_quant.ptree (fun _ t -> merge t) w) );
      ( "apply to a scalar",
        "at least one axis",
        apply w (Nx.scalar Nx.float32 1.) );
      ( "apply to a last axis of another size",
        "last axis",
        apply w (x [| 3; 32 |]) );
      ( "apply over batch axes that do not broadcast",
        "broadcast",
        apply w (x [| 3; 1; 64 |]) );
      ( "q8_0 blocks split over two devices inside a block",
        "axis 1",
        blocks Nx_quant.q8_0 ~devices:true [| 2; 34 |] );
      ( "mxfp4 codes split over two devices inside a group",
        "axis 1",
        fun () ->
          ignore
            (Nx_quant.mxfp4
               ~scales:(Nx.zeros Nx.uint8 [| 2; 1 |])
               (Nx.place
                  (Nx.Placement.sharded ~axis:1 [ d1; d2 ])
                  (Nx.zeros Nx.uint8 [| 2; 16 |]))) );
      ( "mxfp4 codes placed by a map inside a group",
        "axis 2",
        fun () -> ignore (Nx.Ptree.map Nx_quant.ptree (fun _ t -> halves t) w)
      );
      ("take along the last axis", "last axis", take 2);
      ("take along the last axis, counted from the end", "last axis", take (-1));
      ("take along an axis past the weight's", "out of bounds", take 3);
    ]
    (fun (_, part, f) -> raises_match (Exn.invalid_arg ~substring:part) f)

let placements =
  let drawn =
    let open Gen in
    let* ((_, weight) as w) =
      weight
        ~lead:(list ~size:(int_range 0 1) (int_range 1 4))
        ~n:(int_range 1 4) ()
    in
    let k = Array.length (Nx_quant.shape weight) - 1 in
    let+ devices =
      of_list
        ~pp:(fun ppf l -> Format.fprintf ppf "over %d devices" (List.length l))
        [ [ d1; d2 ]; [ d1; d2; d3; d4 ] ]
    and+ axis =
      frequency [ (1, constant (Some k)); (2, option (int_range 0 3)) ]
    in
    (w, devices, axis)
  in
  prop "a weight is placed wherever its parts split, and never inside a block"
    drawn (fun ((_, w), devices, axis) ->
      let s = Nx_quant.shape w in
      let r = Array.length s in
      let m = List.length devices in
      let p =
        match axis with
        | None -> Nx.Placement.replicated devices
        | Some axis -> Nx.Placement.sharded ~axis devices
      in
      let dims a =
        Nx.Ptree.fold Nx_quant.ptree (fun _ t l -> Nx.dim a t :: l) w []
      in
      let uneven =
        match axis with
        | None -> false
        | Some a -> a >= r || List.exists (fun d -> d mod m <> 0) (dims a)
      in
      let cuts =
        axis = Some (r - 1) && s.(r - 1) / block_values (format_of w) mod m <> 0
      in
      cover "a split of k between blocks"
        (axis = Some (r - 1) && (not uneven) && not cuts);
      cover "a split of k inside a block" ((not uneven) && cuts);
      let place () = Nx.Ptree.place Nx_quant.ptree p w in
      if uneven then raises_invalid_arg place
      else if cuts then
        raises_match
          (Exn.invalid_arg ~substring:(Printf.sprintf "axis %d" (r - 1)))
          (fun () -> ignore (place ()))
      else
        equal (array float_exact)
          (Nx.to_array (Nx_quant.dequant Nx.float32 w))
          (Nx.to_array (Nx_quant.dequant Nx.float32 (place ()))))

let others =
  group "weights"
    [
      test "construction, maps and visits read no byte" (fun () ->
          let place s = Nx.place (Nx.Placement.on d1) (Nx.zeros Nx.uint8 s) in
          let codes = place [| 4; 6; 32 |]
          and scales = place [| 4; 6; 2 |]
          and bad = place [| 4; 6; 3 |] in
          let sent = bytes_out () in
          let w =
            Nx.Ptree.map Nx_quant.ptree
              (fun _ t -> t)
              (Nx_quant.mxfp4 ~scales codes)
          in
          ignore (Nx.Ptree.map2 Nx_quant.ptree (fun _ a _ -> a) w w);
          raises_invalid_arg (fun () -> Nx_quant.mxfp4 ~scales:bad codes);
          let q =
            Nx.Ptree.map Nx_quant.ptree
              (fun _ t -> t)
              (Nx_quant.q6_k (place [| 4; 6; 420 |]))
          in
          raises_invalid_arg (fun () -> Nx_quant.q6_k (place [| 4; 6; 144 |]));
          let g = Nx_quant.mxfp4_blocks (place [| 4; 6; 34 |]) in
          equal
            (list (array int))
            [ [| 4; 6; 64 |]; [| 4; 6; 512 |]; [| 4; 6; 64 |] ]
            [ Nx_quant.shape w; Nx_quant.shape q; Nx_quant.shape g ];
          equal int 0 (bytes_out () - sent));
      cases ~name:format_name "dequant and apply read no value's elements"
        formats (fun format ->
          let w = random_weight ~format [| 3; 4; 256 |] in
          let reads f =
            let seen = ref [] in
            let run : type r. r Nx.Op.t -> r =
             fun op ->
              (match op with Read { by; _ } -> seen := by :: !seen | _ -> ());
              Nx.Op.eval op
            in
            let claims : type r. r Nx.Op.t -> bool = function
              | Read _ -> true
              | _ -> false
            in
            Nx.Op.intercept { run; claims } (fun () -> ignore (f ()));
            List.sort_uniq String.compare !seen
          in
          equal (list string) []
            (reads (fun () -> Nx_quant.dequant Nx.float32 w));
          equal (list string) []
            (reads (fun () -> Nx_quant.apply w (random_floats [| 3; 1; 256 |]))));
      test "products live where their operands are" (fun () ->
          let p = Nx.Placement.on d1 in
          let w =
            Nx.Ptree.place Nx_quant.ptree p (random_weight [| 3; 4; 64 |])
          in
          let x = Nx.place p (random_floats [| 3; 2; 64 |]) in
          equal (pair placement placement) (p, p)
            ( Nx.placement (Nx_quant.dequant Nx.float32 w),
              Nx.placement (Nx_quant.apply w x) ));
      test
        "visits the case, then codes before scales; rebuild and place keep the \
         parts" (fun () ->
          let w = random_weight [| 3; 4; 64 |] in
          let parts = function
            | Nx_quant.Mxfp4 { codes; scales } -> (codes, scales)
            | w -> failf "a %s weight" (format_name (format_of w))
          in
          let codes, scales = parts w in
          let same w =
            let c, s = parts w in
            c == codes && s == scales
          in
          equal (list string)
            [ "the root: case \"mxfp4\""; "codes: a leaf"; "scales: a leaf" ]
            (List.map
               (Format.asprintf "%a" Nx.Ptree.pp_visit)
               (Nx.Ptree.visits Nx_quant.ptree w));
          is_true ~msg:"rebuild keeps the parts"
            (same
               (Nx.Ptree.rebuild Nx_quant.ptree ~like:w
                  (fst (Nx.Ptree.flatten Nx_quant.ptree w))));
          is_true ~msg:"place keeps parts already placed"
            (same (Nx.Ptree.place Nx_quant.ptree Nx.Placement.host w)));
      test "visits a GGUF format's case, then its blocks" (fun () ->
          equal (list string)
            [ "the root: case \"q8_0\""; "blocks: a leaf" ]
            (List.map
               (Format.asprintf "%a" Nx.Ptree.pp_visit)
               (Nx.Ptree.visits Nx_quant.ptree
                  (random_weight ~format:Q8_0 [| 3; 64 |]))));
    ]

let () =
  exit
    (run "nx quant"
       [
         known;
         ggml;
         values_and_products;
         routing;
         file_layouts;
         errors;
         placements;
         others;
       ])
