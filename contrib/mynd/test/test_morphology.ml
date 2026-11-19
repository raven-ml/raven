(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* Footprints *)

type fp = { kh : int; kw : int; taps : bool array }

let footprint f = Nx.create Nx.bool [| f.kh; f.kw |] f.taps
let selected f p q = f.taps.((p * f.kw) + q)

(* Symmetric about the origin tap: each selected tap's reflection about it is a
   selected tap. *)
let symmetric f =
  let ok = ref true in
  for p = 0 to f.kh - 1 do
    for q = 0 to f.kw - 1 do
      let p' = (2 * (f.kh / 2)) - p and q' = (2 * (f.kw / 2)) - q in
      if
        selected f p q
        && not (p' >= 0 && p' < f.kh && q' >= 0 && q' < f.kw && selected f p' q')
      then ok := false
    done
  done;
  !ok

(* [reflect_pad f] is [f] flipped, a row or column of [false] before each even
   axis: dilation by it reads the taps erosion by [f] reads. *)
let reflect_pad f =
  let kh = f.kh + ((f.kh + 1) mod 2) and kw = f.kw + ((f.kw + 1) mod 2) in
  let oh = kh - f.kh and ow = kw - f.kw in
  {
    kh;
    kw;
    taps =
      Array.init (kh * kw) (fun k ->
          let p = (k / kw) - oh and q = (k mod kw) - ow in
          p >= 0 && q >= 0 && selected f (f.kh - 1 - p) (f.kw - 1 - q));
  }

(* [pad_front f] puts a row or column of [false] before each even axis. *)
let pad_front f =
  let kh = f.kh + ((f.kh + 1) mod 2) and kw = f.kw + ((f.kw + 1) mod 2) in
  let oh = kh - f.kh and ow = kw - f.kw in
  {
    kh;
    kw;
    taps =
      Array.init (kh * kw) (fun k ->
          let p = (k / kw) - oh and q = (k mod kw) - ow in
          p >= 0 && q >= 0 && selected f p q);
  }

(* Empty and full footprints, and any taps, at extents of one to five. *)
let gen_fp =
  Gen.(
    let* kh, kw = pair (int_range 1 5) (int_range 1 5) in
    let* kind = int_range 0 7 in
    let+ bits = array ~size:(constant (kh * kw)) bool in
    let taps =
      match kind with
      | 0 -> Array.make (kh * kw) false
      | 1 -> Array.make (kh * kw) true
      | _ -> bits
    in
    { kh; kw; taps })

let cover_fp f ~h ~w =
  cover "odd extents" (f.kh mod 2 = 1 && f.kw mod 2 = 1);
  cover "an even extent" (f.kh mod 2 = 0 || f.kw mod 2 = 0);
  cover "without the pixel" (not (selected f (f.kh / 2) (f.kw / 2)));
  cover "empty" (not (Array.exists Fun.id f.taps));
  cover "asymmetric" (not (symmetric f));
  cover "larger than the image" (f.kh > h || f.kw > w)

(* The definition *)

(* [taps ~least f h w i j] is the pixels inside an [h] by [w] image that erosion
   ([least]) or dilation reads at [(i, j)]. *)
let taps ~least f h w i j =
  List.concat
    (List.init f.kh (fun p ->
         List.filter_map
           (fun q ->
             let r, s =
               if least then (i - (f.kh / 2) + p, j - (f.kw / 2) + q)
               else (i + (f.kh / 2) - p, j + (f.kw / 2) - q)
             in
             if selected f p q && r >= 0 && r < h && s >= 0 && s < w then
               Some (r, s)
             else None)
           (List.init f.kw Fun.id)))

(* A dtype, values to draw, its order and its NaN test, and how to print a
   value. *)
type d =
  | D :
      string
      * ('v, 'b) Nx.dtype
      * 'v list
      * ('v -> 'v -> int)
      * ('v -> bool)
      * ('v -> string)
      -> d

let float_order a b =
  if a < b then -1
  else if a > b then 1
  else compare (Float.sign_bit b) (Float.sign_bit a)

let show_float v = if Float.is_nan v then "nan" else Printf.sprintf "%h" v
let specials = [ Float.nan; -0.; 0.; Float.infinity; Float.neg_infinity ]
let ties = [ -2.; -1.; 0.; 1.; 1.; 2.; 3.5 ]

let float_d n dt extra =
  D (n, dt, ties @ extra, float_order, Float.is_nan, show_float)

let dtypes =
  [
    D
      ( "bool",
        Nx.bool,
        [ false; true ],
        compare,
        (fun _ -> false),
        string_of_bool );
    D
      ( "uint8",
        Nx.uint8,
        [ 0; 1; 2; 7; 128; 255 ],
        compare,
        (fun _ -> false),
        string_of_int );
    D
      ( "int32",
        Nx.int32,
        [ Int32.min_int; -1l; 0l; 3l; Int32.max_int ],
        compare,
        (fun _ -> false),
        Int32.to_string );
    D
      ( "int64",
        Nx.int64,
        [ Int64.min_int; -1L; 0L; 3L; Int64.max_int ],
        compare,
        (fun _ -> false),
        Int64.to_string );
    D
      ( "uint32",
        Nx.uint32,
        [ 0l; 1l; 7l; Int32.min_int; -1l ],
        Int32.unsigned_compare,
        (fun _ -> false),
        Printf.sprintf "%lu" );
    D
      ( "uint64",
        Nx.uint64,
        [ 0L; 1L; Int64.min_int; -1L ],
        Int64.unsigned_compare,
        (fun _ -> false),
        Printf.sprintf "%Lu" );
    D
      ( "int8",
        Nx.int8,
        [ -128; -1; 0; 5; 127 ],
        compare,
        (fun _ -> false),
        string_of_int );
    D
      ( "int16",
        Nx.int16,
        [ -32768; 0; 3; 32767 ],
        compare,
        (fun _ -> false),
        string_of_int );
    D
      ( "uint16",
        Nx.uint16,
        [ 0; 1; 300; 65535 ],
        compare,
        (fun _ -> false),
        string_of_int );
    float_d "float64" Nx.float64 specials;
    float_d "float32" Nx.float32 specials;
    float_d "float16" Nx.float16 (65504. :: specials);
    float_d "bfloat16" Nx.bfloat16 specials;
    float_d "float8_e4m3fn" Nx.float8_e4m3fn [ Float.nan; -0.; 0.; 448.; -448. ];
  ]

let pick l = List.nth l (Random.int (List.length l))

let definition =
  prop "erode and dilate take the least and greatest of the taps inside"
    ~count:500
    Gen.(
      quad (of_list dtypes) gen_fp
        (pair (int_range 0 6) (int_range 0 6))
        (pair bool (int_range 1 2)))
    (fun (D (_, dt, pool, order, is_nan, show), f, (h, w), (least, c)) ->
      cover_fp f ~h ~w;
      cover "no pixel" (h * w = 0);
      let v = Array.init (h * w * c) (fun _ -> pick pool) in
      let x = Nx.create dt [| h; w; c |] v in
      (* The values as stored, after the dtype's rounding. *)
      let v = Nx.to_array x in
      let id = if least then Nx.Dtype.max_value dt else Nx.Dtype.min_value dt in
      let want =
        Array.init
          (h * w * c)
          (fun o ->
            let i = o / (w * c) and j = o / c mod w and l = o mod c in
            let read =
              List.map
                (fun (r, s) -> v.((((r * w) + s) * c) + l))
                (taps ~least f h w i j)
            in
            match List.find_opt is_nan read with
            | Some n -> show n
            | None ->
                let better a b =
                  let k = order a b in
                  if least then if k <= 0 then a else b
                  else if k >= 0 then a
                  else b
                in
                show (List.fold_left better id read))
      in
      let y = (if least then V.erode else V.dilate) (footprint f) x in
      equal (list int) [ h; w; c ] (Array.to_list (Nx.shape y));
      equal (array string) want (Array.map show (Nx.to_array y)))

(* Laws *)

let gen_image =
  Gen.(
    let* h, w = pair (int_range 1 7) (int_range 1 7) in
    let+ v =
      array ~size:(constant (h * w)) (map Float.of_int (int_range (-3) 3))
    in
    Nx.create Nx.float64 [| h; w; 1 |] v)

let pointwise ~msg le a b =
  let a = floats a and b = floats b in
  Array.iteri
    (fun i x ->
      if not (le x b.(i)) then
        failf "%s: element %d, %g against %g" msg i x b.(i))
    a

let laws =
  prop "openings and closings are what the adjunction makes them" ~count:400
    Gen.(
      triple gen_fp gen_image (array ~size:(constant 49) (float_range 0. 2.)))
    (fun (f, x, bump) ->
      let h = Nx.dim 0 x and w = Nx.dim 1 x in
      cover_fp f ~h ~w;
      let fp = footprint f in
      let opening x = V.dilate fp (V.erode fp x) in
      let closing x = V.erode fp (V.dilate fp x) in
      pointwise ~msg:"opening <= x" ( <= ) (opening x) x;
      pointwise ~msg:"closing >= x" ( >= ) (closing x) x;
      equal ~msg:"opening twice" (array float_exact)
        (floats (opening x))
        (floats (opening (opening x)));
      equal ~msg:"closing twice" (array float_exact)
        (floats (closing x))
        (floats (closing (closing x)));
      let x' =
        Nx.add x (Nx.create Nx.float64 [| h; w; 1 |] (Array.sub bump 0 (h * w)))
      in
      pointwise ~msg:"opening increases" ( <= ) (opening x) (opening x');
      pointwise ~msg:"closing increases" ( <= ) (closing x) (closing x'))

(* [dilate fp y <= x] exactly when [y <= erode fp x], on masks: [x] is drawn
   about [dilate fp y] so that both answers occur. *)
let adjunction =
  prop "erode and dilate are adjoint" ~count:500
    Gen.(
      quad gen_fp
        (pair (int_range 1 6) (int_range 1 6))
        (array ~size:(constant 36) (float_range 0. 1.))
        (pair (int_range 0 2) (int_range 0 35)))
    (fun (f, (h, w), draw, (kind, at)) ->
      cover_fp f ~h ~w;
      let fp = footprint f in
      let y =
        Nx.create Nx.bool [| h; w; 1 |]
          (Array.init (h * w) (fun k -> draw.(k) < 0.2))
      in
      let d = Nx.to_array (V.dilate fp y) in
      let x =
        Array.mapi
          (fun k b ->
            match kind with
            | 0 -> b
            | 1 -> if k = at mod (h * w) then not b else b
            | _ -> b || draw.(k) > 0.5)
          d
      in
      let x = Nx.create Nx.bool [| h; w; 1 |] x in
      let within a b =
        Array.for_all2 (fun a b -> (not a) || b) (Nx.to_array a) (Nx.to_array b)
      in
      let left = within (V.dilate fp y) x and right = within y (V.erode fp x) in
      cover "related" left;
      cover "unrelated" (not left);
      equal bool left right)

(* Dilating one true pixel draws the footprint, its origin tap on the pixel. *)
let stamp =
  prop "dilating a pixel draws the footprint" ~count:300
    Gen.(triple gen_fp (pair (int_range 1 7) (int_range 1 7)) (pair nat nat))
    (fun (f, (h, w), (a, b)) ->
      cover_fp f ~h ~w;
      let i0 = a mod h and j0 = b mod w in
      let m =
        Nx.init Nx.bool [| h; w; 1 |] (fun i -> i.(0) = i0 && i.(1) = j0)
      in
      let want =
        Array.init (h * w) (fun k ->
            let p = (k / w) - i0 + (f.kh / 2)
            and q = (k mod w) - j0 + (f.kw / 2) in
            p >= 0 && p < f.kh && q >= 0 && q < f.kw && selected f p q)
      in
      equal (array bool) want (Nx.to_array (V.dilate (footprint f) m)))

(* Erosion is dual to dilation by the reflected footprint, and the greatest of
   the taps erosion reads is dilation by it. *)
let reflection =
  prop "the reflected footprint gives the duals" ~count:300
    Gen.(pair gen_fp gen_image)
    (fun (f, x) ->
      let h = Nx.dim 0 x and w = Nx.dim 1 x in
      cover_fp f ~h ~w;
      let fp = footprint f and rp = footprint (reflect_pad f) in
      let m = Nx.greater_s x 0. in
      equal ~msg:"masks" (array bool)
        (Nx.to_array (V.erode fp m))
        (Nx.to_array (Nx.logical_not (V.dilate rp (Nx.logical_not m))));
      equal ~msg:"greatest of erosion's taps" (array float_exact)
        (floats (Nx.neg (V.erode fp (Nx.neg x))))
        (floats (V.dilate rp x)))

let batch =
  test "each image's result is the same in any batch" (fun () ->
      let x = Nx.init Nx.float32 [| 3; 5; 6; 2 |] (fun _ -> Random.float 1.) in
      let fp =
        Nx.create Nx.bool [| 2; 3 |] [| true; false; true; true; true; false |]
      in
      List.iter
        (fun f ->
          let whole = floats (f fp x) in
          let alone =
            Array.concat
              (List.init 3 (fun i -> floats (f fp (Nx.slice [ Nx.I i ] x))))
          in
          equal (array float_exact) alone whole)
        [ V.erode; V.dilate ])

(* Derivatives *)

(* [shares ~least f h w x r] is the gradient of [sum (r · y)], [y] the erosion
   or dilation of [x], by the .mli: each result's [r] shared equally among the
   taps read that equal it, none where it is NaN; and, per pixel, the sum of the
   magnitudes of its terms and their number. *)
let shares ~least f h w x y r =
  let g = Array.make (h * w) 0. in
  let size = Array.make (h * w) 0. and terms = Array.make (h * w) 0 in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      let o = (i * w) + j in
      let tied =
        List.filter
          (fun (a, b) -> x.((a * w) + b) = y.(o))
          (taps ~least f h w i j)
      in
      let n = Float.of_int (List.length tied) in
      List.iter
        (fun (a, b) ->
          let k = (a * w) + b in
          g.(k) <- g.(k) +. (r.(o) /. n);
          size.(k) <- size.(k) +. Float.abs (r.(o) /. n);
          terms.(k) <- terms.(k) + 1)
        tied
    done
  done;
  (g, size, terms)

type fd = G : string * (float, _) Nx.dtype * float * float -> fd

(* Each float format with the extremes it holds, which the fold's identity
   outside the image equals, and the epsilon it sums ties in. *)
let float_formats =
  let f32 = (Nx.Dtype.float_format Nx.float32).epsilon in
  [
    G ("float64", Nx.float64, Float.infinity, epsilon_float);
    G ("float32", Nx.float32, Float.infinity, f32);
    G ("float16", Nx.float16, Float.infinity, f32);
    G ("bfloat16", Nx.bfloat16, Float.infinity, f32);
    G ("float8_e4m3fn", Nx.float8_e4m3fn, 448., f32);
  ]

(* Integers tie often; a few values are NaN or the format's extremes. The
   gradient is the shares' sum rounded into the format once, its terms summed in
   [acc]: within [ε |g|], or the spacing [η] of the subnormals, and [(t + 1)
   acc] of their magnitudes. *)
let derivative =
  prop "ties share the derivative equally, none at NaN" ~count:400
    Gen.(
      quad
        (pair gen_fp (of_list float_formats))
        (pair (int_range 1 6) (int_range 1 6))
        (array ~size:(constant 36)
           (frequency
              [
                (12, map Float.of_int (int_range 0 2));
                (1, constant Float.nan);
                (1, constant 1000.);
                (1, constant (-1000.));
              ]))
        (pair bool (array ~size:(constant 36) (float_range (-1.) 1.))))
    (fun ((f, G (_, dt, big, acc)), (h, w), x, (least, r)) ->
      cover_fp f ~h ~w;
      (* [±1000] stands for the extremes. *)
      let x =
        Array.map
          (fun v -> if Float.abs v = 1000. then Float.copy_sign big v else v)
          (Array.sub x 0 (h * w))
      in
      cover "a NaN" (Array.exists Float.is_nan x);
      cover "an extreme" (Array.exists (fun v -> Float.abs v = big) x);
      let op = if least then V.erode else V.dilate in
      let tx = Nx.create dt [| h; w; 1 |] x in
      let tr =
        Nx.cast dt (Nx.create Nx.float64 [| h; w; 1 |] (Array.sub r 0 (h * w)))
      in
      let x = floats tx and r = floats tr in
      let y = op (footprint f) tx in
      let g, size, terms = shares ~least f h w x (floats y) r in
      let got =
        floats
          (Rune.grad' (fun x -> Nx.sum (Nx.mul (op (footprint f) x) tr)) tx)
      in
      let ff = Nx.Dtype.float_format dt in
      let e = ff.epsilon and eta = ff.min_normal *. ff.epsilon in
      Array.iteri
        (fun k want ->
          let bound =
            Float.max (e *. Float.abs want) eta
            +. (Float.of_int (terms.(k) + 1) *. acc *. size.(k))
          in
          if Float.abs (got.(k) -. want) > bound then
            failf "grad at pixel %d is %h, not %h within %h" k got.(k) want
              bound)
        g;
      (* Forward mode: the tangent of ones is one where a tap ties, else 0. *)
      let _, t = Rune.jvp' (op (footprint f)) tx (Nx.ones_like tx) in
      let yv = floats y in
      let ones =
        Array.init (h * w) (fun o ->
            let i = o / w and j = o mod w in
            if
              List.exists
                (fun (a, b) -> x.((a * w) + b) = yv.(o))
                (taps ~least f h w i j)
            then 1.
            else 0.)
      in
      equal ~msg:"jvp" (array float_exact) ones (floats t))

(* A constant image's centre, eroded by a full [k × k] footprint, shares its
   tangent among [k²] ties, more than float8_e4m3fn, bfloat16 and float16 count:
   each share is [1 / k²] within an ulp. *)
let many_ties =
  cases "many ties share equally in the narrow formats"
    ~id:(fun (G (n, _, _, _), k) -> Printf.sprintf "%s %d" n k)
    (List.concat_map
       (fun (G (n, _, _, _) as d) ->
         match n with
         | "float8_e4m3fn" -> [ (d, 5) ]
         | "bfloat16" -> [ (d, 17) ]
         | "float16" -> [ (d, 47) ]
         | _ -> [])
       float_formats)
    (fun (G (_, dt, _, _), k) ->
      let x = Nx.full dt [| k; k; 1 |] 1. in
      let fp = Nx.ones Nx.bool [| k; k |] in
      let centre =
        Nx.init dt [| k; k; 1 |] (fun i ->
            if i.(0) = k / 2 && i.(1) = k / 2 then 1. else 0.)
      in
      let g = Rune.grad' (fun x -> Nx.sum (Nx.mul (V.erode fp x) centre)) x in
      let share = 1. /. Float.of_int (k * k) in
      let e = (Nx.Dtype.float_format dt).epsilon in
      Array.iteri
        (fun i v ->
          if Float.abs (v -. share) > e *. share then
            failf "pixel %d takes %h, not %h within an ulp" i v share)
        (floats g))

let compiled =
  test "compiled is eager, its derivative too" (fun () ->
      let x =
        Nx.init Nx.float32 [| 2; 6; 7; 2 |] (fun _ ->
            Float.of_int (Random.int 3))
      in
      let fp =
        Nx.create Nx.bool [| 3; 2 |] [| true; true; false; true; true; true |]
      in
      List.iter
        (fun (n, f) ->
          equal ~msg:n (array float_exact)
            (floats (f fp x))
            (floats (Rune.jit' (f fp) x));
          let g x = Rune.grad' (fun x -> Nx.sum (Nx.mul (f fp x) x)) x in
          equal ~msg:n (array float_exact)
            (floats (g x))
            (floats (Rune.jit' g x)))
        [ ("erode", V.erode); ("dilate", V.dilate) ])

(* Against OpenCV, scikit-image and scipy, from golden/morphology.npz
   (contrib/mynd/test/gen/morphology.py) *)
let golden = lazy (Nx_io.load_npz "golden/morphology.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/morphology.npz has no %s" key

let fp_entry name =
  let t = entry Nx.bool ("fp." ^ name) in
  { kh = Nx.dim 0 t; kw = Nx.dim 1 t; taps = Nx.to_array t }

(* Each library takes its footprint at its own anchor and reading: OpenCV's
   dilation reads erosion's taps; scikit-image's anchor on an even axis is one
   tap earlier; scipy's dilation is mynd's. *)
let libraries =
  let names =
    [
      "rect3x3";
      "rect2x2";
      "rect4x4";
      "rect1x4";
      "cross5";
      "ellipse6";
      "asym3x3";
      "asym4x3";
      "nocentre3x5";
    ]
  in
  let rules =
    [
      ("opencv.erode", fun f -> (true, f));
      ("opencv.dilate", fun f -> (false, reflect_pad f));
      ("skimage.erosion", fun f -> (true, pad_front f));
      ("skimage.dilation", fun f -> (false, reflect_pad (pad_front f)));
    ]
  in
  cases "erode and dilate agree with OpenCV, scikit-image and scipy"
    ~id:(fun (r, n) -> Printf.sprintf "%s %s" r n)
    (List.concat_map (fun (r, _) -> List.map (fun n -> (r, n)) names) rules
    @ List.map (fun n -> ("scipy.dilation", n)) names)
    (fun (r, n) ->
      let f = fp_entry n in
      if r = "scipy.dilation" then
        equal (array float_exact)
          (floats (entry Nx.float32 (Printf.sprintf "%s.%s.f32" r n)))
          (floats (V.dilate (footprint f) (entry Nx.float32 "f32")))
      else
        let least, f = (List.assoc r rules) f in
        let fp = footprint f in
        let op fp x = if least then V.erode fp x else V.dilate fp x in
        equal ~msg:"float32" (array float_exact)
          (floats (entry Nx.float32 (Printf.sprintf "%s.%s.f32" r n)))
          (floats (op fp (entry Nx.float32 "f32")));
        equal ~msg:"uint8" (array float_exact)
          (floats (entry Nx.uint8 (Printf.sprintf "%s.%s.u8" r n)))
          (floats (op fp (entry Nx.uint8 "u8"))))

let refusals =
  let x = Nx.zeros Nx.float32 [| 4; 4; 1 |] in
  cases "misuse raises"
    ~id:(fun (n, _) -> n)
    [
      ( "a footprint of one axis",
        fun () -> ignore (V.erode (Nx.ones Nx.bool [| 3 |]) x) );
      ( "a footprint of three axes",
        fun () -> ignore (V.dilate (Nx.ones Nx.bool [| 3; 3; 1 |]) x) );
      ( "an image of two axes",
        fun () ->
          ignore
            (V.erode
               (Nx.ones Nx.bool [| 3; 3 |])
               (Nx.zeros Nx.float32 [| 4; 4 |])) );
      ( "complex values",
        fun () ->
          ignore
            (V.dilate
               (Nx.ones Nx.bool [| 3; 3 |])
               (Nx.zeros Nx.complex64 [| 4; 4; 1 |])) );
    ]
    (fun (_, f) -> raises_match (Exn.invalid_arg ~sub:"Mynd.") f)

let () =
  exit
    (run "mynd.morphology"
       [
         definition;
         laws;
         adjunction;
         stamp;
         reflection;
         batch;
         derivative;
         many_ties;
         compiled;
         libraries;
         refusals;
       ])
