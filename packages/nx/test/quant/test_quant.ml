(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantised weights against their formats' definitions: dequant gives each
   value at its dtype, as the format's reference decodes it from its bytes, and
   apply is the product with the dequantised weight within the error of a
   float32 sum. *)

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
   scale byte [s], NaN for [255]. *)

let e2m1 = [| 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. |]

let mxfp4_values codes scales =
  let codes = Nx.to_array codes and scales = Nx.to_array scales in
  Array.init
    (2 * Array.length codes)
    (fun i ->
      let code = (codes.(i / 2) lsr (4 * (i mod 2))) land 15 in
      let s = scales.(i / 32) in
      let m = e2m1.(code land 7) *. Float.ldexp 1. (s - 127) in
      if s = 255 then Float.nan else if code < 8 then m else -.m)

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
   the values of its blocks allow it. *)
type view = {
  view : string;
  fits : int array -> int -> bool;
  apply : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t;
}

let views =
  let r = Array.length in
  let last t = List.init (Nx.ndim t - 1) (fun _ -> Nx.A) in
  [
    { view = "contiguous"; fits = (fun _ _ -> true); apply = Fun.id };
    {
      view = "every other row";
      fits = (fun s _ -> s.(r s - 2) > 0);
      apply =
        (fun t ->
          Nx.squeeze ~axes:[ -1 ]
            (Nx.sliding_window ~axis:(-2) ~window:1 ~step:2 t));
    };
    {
      view = "its leading axes swapped";
      fits = (fun s _ -> r s >= 4);
      apply = (fun t -> Nx.swapaxes 0 1 t);
    };
    {
      view = "the first half of its inputs";
      fits = (fun s values -> s.(r s - 1) / values mod 2 = 0);
      apply = (fun t -> Nx.slice (last t @ [ R (0, Nx.dim (-1) t / 2) ]) t);
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

(* [blocks format ~scale count] is [count] blocks of [format]'s bytes, random
   but for their float16 scales, drawn from [scale]. *)
let gguf_blocks format ~scale count =
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

(* A weight of shape [[| lead...; n; k |]] under a view. *)
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
     let+ w =
       match f with
       | Mxfp4 ->
           let+ codes =
             array ~size:(constant (m * count * 16)) (int_range 0 255)
           and+ scales = array ~size:(constant (m * count)) e8m0 in
           Nx_quant.mxfp4
             ~scales:(Nx.create Nx.uint8 (shape count) scales)
             (Nx.create Nx.uint8 (shape (count * 16)) codes)
       | f ->
           let+ b = gguf_blocks f ~scale:binary16 (m * count) in
           weight_of f (shape (count * block_bytes f)) b
     and+ v = of_list views in
     if v.fits (Nx_quant.shape w) (block_values f) then
       (v.view, Nx.Ptree.map Nx_quant.ptree (fun _ t -> v.apply t) w)
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

let broadcast a b =
  let n = Int.max (Array.length a) (Array.length b) in
  let dim s i =
    if i < n - Array.length s then 1 else s.(i - n + Array.length s)
  in
  Array.init n (fun i -> if dim a i = 1 then dim b i else dim a i)

(* [w'] of [apply ~ids] for the dequantised weight [dq] with [lanes] leading
   axes: each id's expert in its lane, ids clamped into the experts. *)
let gathered dq ~lanes ids =
  let ds = Nx.shape dq and is = Nx.shape ids in
  let wb =
    Array.append
      (broadcast (Array.sub ds 0 lanes) (Array.sub is 0 lanes))
      (Array.sub is lanes (Array.length is - lanes))
  in
  let ids = Nx.broadcast_to wb ids in
  let matrix i =
    let pos = unravel wb i in
    let id = Int64.to_int (Nx.item (Array.to_list pos) ids) in
    let lane = List.init lanes (fun a -> if ds.(a) = 1 then 0 else pos.(a)) in
    Nx.slice
      (List.map
         (fun i -> Nx.I i)
         (lane @ [ Int.max 0 (Int.min (ds.(lanes) - 1) id) ]))
      dq
  in
  let count = Ref.numel wb in
  Nx.reshape
    (Array.append wb [| ds.(lanes + 1); ds.(lanes + 2) |])
    (if count = 0 then Nx.zeros Nx.float32 [| 0 |]
     else Nx.stack ~axis:0 (List.init count matrix))

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

(* A weight, ids selecting its experts or none, and an input. *)
let products =
  let open Gen in
  let id e =
    frequency
      [
        (6, map Int64.of_int (int_range (-2) (e + 1)));
        ( 1,
          of_list
            ~pp:(fun ppf -> Format.fprintf ppf "%Ld")
            [ Int64.min_int; Int64.max_int; 0x1_0000_0000L ] );
      ]
  in
  let* experts, lanes =
    pair flag (list ~size:(int_range 0 1) (int_range 1 2))
  in
  let* view, w =
    if experts then
      weight ~lead:(map (fun e -> lanes @ [ e ]) (int_range 1 3)) ()
    else weight ()
  in
  let lead, _, k = dims w in
  let p = Array.length lead - 1 in
  let* ids, batch =
    if not experts then constant (None, lead)
    else
      let* whole, tokens =
        pair
          (list ~size:(constant p) usually)
          (list ~size:(int_range 0 2) (int_range 0 3))
      in
      let is =
        Array.append
          (Array.of_list
             (List.mapi (fun a w -> if w then lead.(a) else 1) whole))
          (Array.of_list tokens)
      in
      let+ ids = array ~size:(constant (Ref.numel is)) (id lead.(p)) in
      ( Some (Nx.create Nx.int64 is ids),
        Array.append
          (broadcast (Array.sub lead 0 p) (Array.sub is 0 p))
          (Array.of_list tokens) )
  in
  let+ x = input ~batch ~inputs:k in
  (view, w, ids, x)

let pp_product ppf (view, w, ids, X { x; _ }) =
  Format.fprintf ppf "@[<v>%a@,ids %a@,x %a@]" pp_weight (view, w)
    (Format.pp_print_option Nx.pp)
    ids Nx.pp x

(* [label] is [cover] in a property. *)
let product_law (_, w, ids, X { at; x }) =
  let lead, _, k = dims w in
  let dq = Nx_quant.dequant Nx.float32 w in
  let w' =
    match ids with
    | None -> dq
    | Some ids -> gathered dq ~lanes:(Array.length lead - 1) ids
  in
  let expected, bound = product x w' in
  let valid =
    match ids with
    | None -> Nx.full Nx.bool (Nx.shape expected) true
    | Some ids ->
        let e = Int64.of_int lead.(Array.length lead - 1) in
        let v = Nx.logical_and (Nx.greater_equal_s ids 0L) (Nx.less_s ids e) in
        let units = if Nx.ndim x = 1 then [| 1 |] else [| 1; 1 |] in
        Nx.broadcast_to (Nx.shape expected)
          (Nx.reshape (Array.append (Nx.shape v) units) v)
  in
  List.iter
    (fun f -> cover ("a " ^ format_name f ^ " weight") (format_of w = f))
    formats;
  cover "one block per row" (k = block_values (format_of w));
  cover "an id that selects no expert" (Array.mem false (Nx.to_array valid));
  cover "an empty result" (Nx.numel expected = 0);
  (let s = Nx.shape expected in
   let batch = Array.sub s 0 (Array.length s - min (Nx.ndim x) 2) in
   cover "twice as many instances as matrices or more"
     (ids <> None && Ref.numel batch >= 2 * Ref.numel lead));
  let y = Nx_quant.apply ?ids w x in
  equal ~msg:"dtype" string
    (Nx_dtype.to_string (Nx.dtype x))
    (Nx_dtype.to_string (Nx.dtype y));
  agrees ~at ~k (Nx.where valid expected (Nx.zeros_like expected)) bound y;
  let y = Nx.to_array (Nx.cast Nx.float64 y) in
  Array.iteri
    (fun i v -> if not v then equal ~msg:"no expert" float_exact 0. y.(i))
    (Nx.to_array valid)

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
      prop
        "apply is the product with the dequantised weight, or with each id's \
         expert and exactly zero where an id selects none"
        (Gen.with_pp pp_product products)
        product_law;
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
  let blocks f shape () = ignore (f (Nx.zeros Nx.uint8 shape)) in
  let halve (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t =
    if Nx.dim (-1) t = 2 then Nx.slice [ A; A; R (0, 1) ] t else t
  in
  let apply ?ids w x () = ignore (Nx_quant.apply ?ids w x) in
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
        fun () -> ignore (Nx.Ptree.map Nx_quant.ptree (fun _ t -> halve t) w) );
      ( "apply to a scalar",
        "at least one axis",
        apply w (Nx.scalar Nx.float32 1.) );
      ( "apply to a last axis of another size",
        "last axis",
        apply w (x [| 3; 32 |]) );
      ( "apply over batch axes that do not broadcast",
        "broadcast",
        apply w (x [| 3; 1; 64 |]) );
      ( "ids without an expert axis",
        "expert axis",
        apply
          ~ids:(Nx.zeros Nx.int64 [| 3 |])
          (random_weight [| 5; 64 |])
          (x [| 64 |]) );
      ( "ids without the weight's lanes",
        "leading axes",
        apply ~ids:(Nx.scalar Nx.int64 0L)
          (random_weight [| 2; 3; 5; 64 |])
          (x [| 64 |]) );
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
  prop "place splits any axis Nx.place can, and k only between blocks" drawn
    (fun ((_, w), devices, axis) ->
      let s = Nx_quant.shape w in
      let r = Array.length s in
      let p =
        match axis with
        | None -> Nx.Placement.replicated devices
        | Some axis -> Nx.Placement.sharded ~axis devices
      in
      let refused =
        match axis with
        | None -> false
        | Some a ->
            a >= r
            || (if a = r - 1 then s.(a) / block_values (format_of w) else s.(a))
               mod List.length devices
               <> 0
      in
      cover "a split of k" (axis = Some (r - 1) && not refused);
      cover "k refused" (axis = Some (r - 1) && refused);
      match Nx_quant.place p w with
      | exception Invalid_argument _ ->
          is_true ~msg:"refused only where a split cuts a group or an axis"
            refused
      | placed ->
          is_false ~msg:"placed where a split cuts a block or an axis" refused;
          let parts w =
            Nx.Ptree.fold Nx_quant.ptree (fun _ t l -> Nx.placement t :: l) w []
          in
          equal
            (pair (list placement) (array float_exact))
            ( List.map (fun _ -> p) (parts w),
              Nx.to_array (Nx_quant.dequant Nx.float32 w) )
            (parts placed, Nx.to_array (Nx_quant.dequant Nx.float32 placed)))

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
          equal
            (pair (array int) (array int))
            ([| 4; 6; 64 |], [| 4; 6; 512 |])
            (Nx_quant.shape w, Nx_quant.shape q);
          equal int 0 (bytes_out () - sent));
      cases ~name:format_name
        "ids that select no expert give zeros over NaN rows" formats
        (fun format ->
          let w = random_weight ~format [| 4; 6; 256 |] in
          let ids =
            Nx.create Nx.int64 [| 3; 2 |] [| -1L; 4L; 9L; -1L; -3L; 4L |]
          in
          equal (tensor float_exact)
            (Nx.zeros Nx.float32 [| 3; 2; 1; 6 |])
            (Nx_quant.apply ~ids w
               (Nx.full Nx.float32 [| 3; 1; 1; 256 |] Float.nan)));
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
            (reads (fun () ->
                 Nx_quant.apply
                   ~ids:(Nx.create Nx.int64 [| 2 |] [| 0L; 2L |])
                   w
                   (random_floats [| 2; 1; 256 |]))));
      test "products live where their operands are" (fun () ->
          let p = Nx.Placement.on d1 in
          let w = Nx_quant.place p (random_weight [| 3; 4; 64 |]) in
          let x = Nx.place p (random_floats [| 2; 1; 64 |])
          and ids = Nx.place p (Nx.create Nx.int64 [| 2 |] [| 0L; 2L |]) in
          equal (pair placement placement) (p, p)
            ( Nx.placement (Nx_quant.dequant Nx.float32 w),
              Nx.placement (Nx_quant.apply ~ids w x) ));
      test "routes split over two devices give the host's product" (fun () ->
          let w = random_weight [| 4; 8; 64 |] in
          let ids =
            Nx.create Nx.int64 [| 8; 2 |]
              (Array.init 16 (fun i -> Int64.of_int ((i * 5 mod 6) - 1)))
          and x = random_floats [| 8; 2; 1; 64 |] in
          let split t = Nx.place (Nx.Placement.sharded ~axis:0 [ d1; d2 ]) t in
          equal (tensor float_exact) (Nx_quant.apply ~ids w x)
            (Nx.place Nx.Placement.host
               (Nx_quant.apply ~ids:(split ids) w (split x))));
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
            (same (Nx_quant.place Nx.Placement.host w)));
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
       [ known; ggml; values_and_products; errors; placements; others ])
