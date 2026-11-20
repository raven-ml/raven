(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

let f32 = Bounds.f32
let row a = Nx.create Nx.float32 [| 1; Array.length a |] a
let values a = Nx.create Nx.float32 [| Array.length a |] a
let ints a = Nx.to_array (Nx.cast Nx.int64 a) |> Array.map Int64.to_int

let tokens a =
  Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

(* The token the module's preamble states, on edges in order: the number of
   [e(0) … e(n-2)] below [y], and [n - 1] for NaN. *)
let count e y =
  let n = Array.length e in
  if Float.is_nan y then n - 1
  else
    let k = ref 0 in
    for i = 0 to n - 2 do
      if e.(i) < y then incr k
    done;
    !k

(* The preamble's bisection, step by step. *)
let bisect e y =
  let lo = ref 0 and hi = ref (Array.length e - 1) in
  while !lo < !hi do
    let mid = !lo + ((!hi - !lo) / 2) in
    if e.(mid) >= y then hi := mid else lo := mid + 1
  done;
  !lo

(* Edges in order, on a grid of halves so that edges repeat and values tie with
   them. *)
let sorted =
  Gen.(
    let* n = int_range 1 12 in
    let+ xs = array ~size:(constant n) (int_range (-8) 8) in
    let e = Array.map (fun i -> float_of_int i /. 2.) xs in
    Array.sort compare e;
    e)

(* Values that tie with [e], fall between and beyond its edges, or are not
   finite. *)
let around e =
  Gen.(
    array ~size:(int_range 1 16)
      (frequency
         [
           (3, of_list (Array.to_list e));
           (3, map f32 (float_range (-5.) 5.));
           (1, of_list [ Float.nan; Float.infinity; Float.neg_infinity; -0. ]);
         ]))

let codebook_and_values edges =
  Gen.(bind edges (fun e -> pair (constant e) (around e)))

let mark e ys =
  let n = Array.length e in
  Array.iter
    (fun y ->
      cover "ties with an edge" (Array.exists (fun x -> x = y) e);
      cover "NaN" (Float.is_nan y);
      cover "above e(n-2)" (n >= 2 && y > e.(n - 2));
      cover "below e(0)" (y < e.(0)))
    ys

let counts =
  prop "encode counts the edges below, NaN last" (codebook_and_values sorted)
    (fun (e, ys) ->
      mark e ys;
      let t = Scalar.v Identity (row e) in
      equal (array int)
        (Array.map (count e) ys)
        (ints (Scalar.encode t (values ys))))

(* Float64 values at, just beside and between the edges. *)
let around64 e =
  Gen.(
    array ~size:(int_range 1 16)
      (frequency
         [
           (2, of_list (Array.to_list e));
           ( 3,
             map
               (fun (x, d) -> x +. d)
               (pair
                  (of_list (Array.to_list e))
                  (of_list [ -1e-9; 1e-9; -1e-12; 1e-12 ])) );
           (2, float_range (-5.) 5.);
           (1, of_list [ Float.nan; Float.infinity; -0. ]);
         ]))

let counts64 =
  prop "a float64 value is compared in float64"
    Gen.(bind sorted (fun e -> pair (constant e) (around64 e)))
    (fun (e, ys) ->
      Array.iter
        (fun y ->
          cover "rounds onto an edge in float32"
            (Array.exists (fun x -> x = f32 y) e
            && not (Array.exists (fun x -> x = y) e)))
        ys;
      let t = Scalar.v Identity (row e) in
      let v = Nx.create Nx.float64 [| Array.length ys |] ys in
      equal (array int) (Array.map (count e) ys) (ints (Scalar.encode t v)))

(* Upstream's tokens for float64 inputs, from torch 2.14.1's bucketize: it
   compares in float64, and its float64 log of the clamp e⁻³ is -3. *)
let float64 =
  cases "float64 inputs take upstream's tokens"
    ~id:(fun (n, _, _, _) -> n)
    [
      ( "above an edge by 1e-12",
        Scalar.Identity,
        [| 0.; 1.; 2.; 3. |],
        (1. +. 1e-12, 2) );
      ("on an edge", Identity, [| 0.; 1.; 2.; 3. |], (1., 1));
      ("NaN edges", Identity, [| 0.; 1.; 2.; 3.; 4.; nan; nan; nan |], (3.5, 7));
      ("NaN", Identity, [| 0.; 1.; 2.; 3.; 4.; nan; nan; nan |], (nan, 7));
      ("the clamp", Log (exp (-3.)), [| -3.; -3.; -3.; 0.; 1. |], (1e-9, 0));
    ]
    (fun (_, tr, e, (x, k)) ->
      let t = Scalar.v tr (row e) in
      let v = Nx.create Nx.float64 [| 1 |] [| x |] in
      equal (array int) [| k |] (ints (Scalar.encode t v)))

(* Edges in order, then NaN edges. *)
let nan_tailed =
  Gen.(
    let* e = sorted in
    let+ k = int_range 1 6 in
    Array.append e (Array.make k Float.nan))

let bisects =
  prop "encode runs the bisection on NaN edges" (codebook_and_values nan_tailed)
    (fun (e, ys) ->
      Array.iter
        (fun y ->
          cover "meets a NaN edge below a finite one"
            (bisect e y = Array.length e - 1 && count e y < Array.length e - 1))
        ys;
      let t = Scalar.v Identity (row e) in
      equal (array int)
        (Array.map (bisect e) ys)
        (ints (Scalar.encode t (values ys))))

let nan_edges =
  cases "NaN edges send a search right"
    ~id:(fun (y, _) -> Printf.sprintf "%g" y)
    [ (3.5, 7); (4.5, 7); (2.5, 3); (Float.nan, 7); (-1., 0) ]
    (fun (y, k) ->
      let e = [| 0.; 1.; 2.; 3.; 4.; Float.nan; Float.nan; Float.nan |] in
      let t = Scalar.v Identity (row e) in
      equal (array int) [| k |] (ints (Scalar.encode t (values [| y |]))))

let upper_edge =
  prop "a token decodes to its edge, the top of its bin"
    (codebook_and_values sorted) (fun (e, ys) ->
      let n = Array.length e in
      let t = Scalar.v Identity (row e) in
      let k = Scalar.encode t (values ys) in
      let d = Nx.to_array (Scalar.decode t k) in
      Array.iteri
        (fun i k ->
          equal float_exact e.(k) d.(i);
          let y = ys.(i) in
          if (not (Float.is_nan y)) && y <= e.(n - 1) then begin
            at_least float_exact ~than:y d.(i);
            if k > 0 then less float_exact ~than:y e.(k - 1)
          end)
        (ints k))

let out_of_range =
  cases "decode refuses a token outside the codebook" ~id:string_of_int
    [ -1; 3; 100 ] (fun k ->
      let t = Scalar.v Identity (row [| 0.; 1.; 2. |]) in
      raises_match (Exn.invalid_arg ~sub:"outside [0, 3)") (fun () ->
          Scalar.decode t (tokens [| 0; k |])))

let past_last =
  test "decode names the class past the last edge" (fun () ->
      (* Redshift's 1025th class, which the transformer predicts. *)
      let t = Scalar.v Identity (row (Array.init 1024 Float.of_int)) in
      raises_match
        (Exn.invalid_arg
           ~sub:"token 1024 at [1] outside [0, 1024), the class past")
        (fun () -> Scalar.decode t (tokens [| 0; 1024 |])))

(* Float32's neighbours of -3. *)
let below_3 = Int32.float_of_bits (Int32.succ (Int32.bits_of_float (-3.)))
let above_3 = Int32.float_of_bits (Int32.pred (Int32.bits_of_float (-3.)))

let floor_ =
  cases "Log clamps at float32 e⁻³, whose log is -3 exactly" ~id:string_of_float
    [ 0.; -0.; -1.; Float.neg_infinity; f32 (exp (-3.)); 1e-30 ]
    (fun v ->
      (* Edges at -3 and its two float32 neighbours: the token is 1 exactly when
         the transformed value is -3. *)
      let e = [| below_3; -3.; above_3; 0. |] in
      let t = Scalar.v (Log (exp (-3.))) (row e) in
      equal (array int) [| 1 |] (ints (Scalar.encode t (values [| v |]))))

let log_nan =
  test "Log keeps NaN, which takes the last token" (fun () ->
      let t = Scalar.v (Log (exp (-3.))) (row [| -3.; 0.; 1.; 2. |]) in
      equal (array int) [| 3 |]
        (ints (Scalar.encode t (values [| Float.nan |]))))

(* A token decodes to its edge's inverse transform, within nx.mli's float32
   bounds of the exact value: [exp] within 2 floats, [sinh] within 3 of each
   input it is given ({!Bounds}). *)
let decodes =
  let each name tr inverse =
    prop name sorted (fun e ->
        let t = Scalar.v tr (row e) in
        let k = tokens (Array.init (Array.length e) Fun.id) in
        Array.iteri
          (fun i d ->
            let bound = inverse e.(i) in
            if not (Bounds.inside bound d) then
              equal (pair float_exact float_exact) bound (d, d))
          (Nx.to_array (Scalar.decode t k)))
  in
  group "transforms"
    [
      each "Log decodes by exp" (Log 0.05) (fun y ->
          Bounds.around 2 (Float.exp y));
      each "Asinh_asinh decodes by sinh (sinh y)" Asinh_asinh (fun y ->
          Bounds.through 3 Float.sinh (Bounds.around 3 (Float.sinh y)));
    ]

let asinh =
  prop "Asinh_asinh tokenizes asinh (asinh v)" (codebook_and_values sorted)
    (fun (e, ys) ->
      let v = values ys in
      let y = Nx.asinh (Nx.asinh v) in
      equal (array int)
        (ints (Scalar.encode (Scalar.v Identity (row e)) y))
        (ints (Scalar.encode (Scalar.v Asinh_asinh (row e)) v)))

(* Two codebooks of one length, and rows of values for them. *)
let two =
  Gen.(
    let* n = int_range 1 8 in
    let book =
      let+ xs = array ~size:(constant n) (int_range (-8) 8) in
      let e = Array.map (fun i -> float_of_int i /. 2.) xs in
      Array.sort compare e;
      e
    in
    let* a = book in
    let* b = book in
    let* rows = int_range 0 6 in
    let+ vs = array ~size:(constant 12) (map f32 (float_range (-5.) 5.)) in
    (a, b, rows, vs))

let channels =
  prop "channel i is tokenized by codebook i" two (fun (a, b, rows, vs) ->
      let x = Nx.reshape [| rows; 2 |] (values (Array.sub vs 0 (2 * rows))) in
      let both = Scalar.v Identity (Nx.stack [ values a; values b ]) in
      let one e i =
        Scalar.encode (Scalar.v Identity (row e)) (Nx.slice [ A; I i ] x)
      in
      let k = Scalar.encode both x in
      equal (array int) (ints (Nx.stack ~axis:1 [ one a 0; one b 1 ])) (ints k);
      equal (array float_exact)
        (Nx.to_array
           (Nx.stack ~axis:1
              [ Nx.take (one a 0) (values a); Nx.take (one b 1) (values b) ]))
        (Nx.to_array (Scalar.decode both k)))

let shapes =
  group "shapes"
    [
      test "one channel takes any shape" (fun () ->
          let t = Scalar.v Identity (row [| 0.; 1.; 2. |]) in
          equal (array int) [||]
            (Nx.shape (Scalar.encode t (Nx.scalar Nx.float32 0.5)));
          equal (array int) [| 2; 3 |]
            (Nx.shape (Scalar.encode t (Nx.zeros Nx.float32 [| 2; 3 |]))));
      test "channels must match the last axis" (fun () ->
          let t = Scalar.v Identity (Nx.zeros Nx.float32 [| 2; 3 |]) in
          raises_match Exn.invalid_arg (fun () ->
              Scalar.encode t (Nx.zeros Nx.float32 [| 4; 3 |]));
          raises_match Exn.invalid_arg (fun () ->
              Scalar.decode t (Nx.zeros Nx.int64 [| 4 |])));
      cases "v refuses edges not of shape [c; n]"
        ~id:(fun s ->
          String.concat ";" (Array.to_list (Array.map string_of_int s)))
        [ [| 3 |]; [| 0; 3 |]; [| 2; 0 |]; [| 1; 2; 3 |] ]
        (fun s ->
          raises_match Exn.invalid_arg (fun () ->
              Scalar.v Identity (Nx.zeros Nx.float32 s)));
    ]

(* init *)

let transform =
  Testable.structural ~pp:(fun ppf -> function
    | Scalar.Identity -> Format.pp_print_string ppf "Identity"
    | Log m -> Format.fprintf ppf "Log %h" m
    | Asinh_asinh -> Format.pp_print_string ppf "Asinh_asinh")

let init =
  cases "init has the modality's transform and shapes, edges sorted"
    ~id:(fun s -> Modality.key (Scalar s))
    Modality.[ Flux_g; Z; Xp_bp ]
    (fun s ->
      let t = Scalar.init ~key:(Nx.Rng.key 3) s in
      equal transform (Scalar.transform_of s) (Scalar.transform t);
      equal (array int)
        [| Modality.tokens (Scalar s); 1024 |]
        (Nx.shape (Scalar.edges t));
      let e = Scalar.edges t in
      let sorted, _ = Nx.sort ~axis:1 e in
      equal (array float_exact) (Nx.to_array sorted) (Nx.to_array e))

(* The walk reports a Log's floor by its bits: floors of other bits, as [0.]
   and [-0.], key other programs, and equal floors the same, whatever the
   edges. *)
let walk =
  let visits tr e =
    List.map
      (Format.asprintf "%a" Nx.Ptree.pp_visit)
      (Nx.Ptree.visits
         (Nx.Ptree.instantiate (module Scalar))
         (Scalar.v tr (row e)))
  in
  group "walk"
    [
      test "floors of other bits give other visits" (fun () ->
          not_equal (list string)
            (visits (Log 0.) [| 0.; 1. |])
            (visits (Log (-0.)) [| 0.; 1. |]));
      test "equal floors give equal visits" (fun () ->
          equal (list string)
            (visits (Log (exp (-3.))) [| 0.; 1. |])
            (visits (Log (exp (-3.))) [| -2.; 5. |]));
    ]

(* Compiled with the codec as an argument, one function serves every codec:
   its walk reports the transform, a floor included, and its edges are an
   argument. The search is comparisons and gathers and the transforms are
   functions Rune.jit computes exactly on the host, so tokens and values agree
   bit for bit. *)
let signature () =
  Nx.Ptree.(instantiate (module Scalar) @-> tensor @-> returns tensor)

let encode32 = Rune.jit (signature ()) Scalar.encode
let encode64 = Rune.jit (signature ()) Scalar.encode
let decode_c = Rune.jit (signature ()) Scalar.decode

(* Codecs from init at three modalities' shapes, and at another floor. *)
let codec =
  Gen.(
    let* m = of_list Modality.[ Flux_g; Z; Xp_bp ] in
    let* seed = int_range 0 1000 in
    let+ floor = of_list [ None; Some 0.05 ] in
    let t = Scalar.init ~key:(Nx.Rng.key seed) m in
    match (floor, Scalar.transform t) with
    | Some f, Log _ -> Scalar.v (Log f) (Scalar.edges t)
    | _ -> t)

(* Four objects' values: edges, normal draws and values that are not finite. *)
let values_of t =
  let e = Nx.to_array (Scalar.edges t) in
  let c = Nx.dim 0 (Scalar.edges t) in
  Gen.(
    let+ v =
      array ~size:(constant (4 * c))
        (frequency
           [
             (2, of_list (Array.to_list e));
             (3, map Bounds.f32 (float_range (-4.) 4.));
             ( 1,
               of_list [ Float.nan; Float.infinity; Float.neg_infinity; -0. ]
             );
           ])
    in
    Nx.create Nx.float32 (if c = 1 then [| 4 |] else [| 4; c |]) v)

let compiled =
  prop "compiled equals eager, bit for bit"
    Gen.(bind codec (fun t -> pair (constant t) (values_of t)))
    (fun (t, v) ->
      cover "a floor other than AION-1's"
        (match Scalar.transform t with Log f -> f = 0.05 | _ -> false);
      cover "a codec of 55 channels" (Nx.dim 0 (Scalar.edges t) = 55);
      let k = Scalar.encode t v in
      equal (array int) (ints k) (ints (encode32 t v));
      let v64 = Nx.cast Nx.float64 v in
      equal (array int) (ints (Scalar.encode t v64)) (ints (encode64 t v64));
      let bits x = Array.map Int32.bits_of_float (Nx.to_array x) in
      equal (array int32) (bits (Scalar.decode t k)) (bits (decode_c t k)))

let compiled_refusal =
  test "a compiled codec refuses a token at the call on the host" (fun () ->
      let t = Scalar.v Identity (row [| 0.; 1.; 2. |]) in
      raises_match (Exn.invalid_arg ~sub:"token 3 at [1] outside [0, 3)")
        (fun () -> decode_c t (tokens [| 0; 3 |])))

(* of_archive's three layouts of codebooks. Row [i] holds [i + j / 10], so a
   transposed or reordered channel shows. *)
let book i n =
  Nx.create Nx.float32 [| n |]
    (Array.init n (fun j -> float_of_int i +. (float_of_int j /. 10.)))

let entries l = Nx_io.Archive.of_list (List.map (fun (k, v) -> (k, Nx.P v)) l)
let rows c n = Array.concat (List.init c (fun i -> Nx.to_array (book i n)))
let channel i = Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i

let layouts =
  cases "of_archive reads each layout of codebooks"
    ~id:(fun (n, _, _, _) -> n)
    [
      ( "one codebook beside its reservoir",
        Modality.Flux_g,
        [
          ("_quantizer._index_to_val", book 0 4);
          ("_quantizer._reservoir", book 7 9);
        ],
        1 );
      ("the redshift grid", Modality.Z, [ ("_quantizer.buckets", book 0 4) ], 1);
      ( "one codebook per channel, each beside its reservoir",
        Modality.Xp_bp,
        List.concat
          (List.init 55 (fun i ->
               [
                 (channel i, book i 4);
                 ( Printf.sprintf "_quantizer.quantizers.%d._reservoir" i,
                   book 0 9 );
               ])),
        55 );
    ]
    (fun (_, m, l, c) ->
      let t = Scalar.of_archive m (entries l) in
      equal (array int) [| c; 4 |] (Nx.shape (Scalar.edges t));
      equal (array float_exact) (rows c 4) (Nx.to_array (Scalar.edges t)))

let unreadable =
  cases "of_archive refuses a file it would read wrongly"
    ~id:(fun (n, _, _, _) -> n)
    [
      ( "no codebook",
        Modality.Flux_g,
        [ ("_quantizer._reservoir", book 0 4) ],
        channel 0 );
      ( "54 of 55 channels",
        Modality.Xp_rp,
        List.init 54 (fun i -> (channel i, book i 4)),
        channel 54 );
      ( "56 of 55 channels",
        Modality.Xp_rp,
        List.init 56 (fun i -> (channel i, book i 4)),
        "more than 55 codebooks" );
      ( "channels of two lengths",
        Modality.Xp_rp,
        List.init 55 (fun i -> (channel i, book i (if i = 9 then 3 else 4))),
        "codebooks of several lengths" );
      ( "an entry it does not read",
        Modality.Flux_g,
        [
          ("_quantizer._index_to_val", book 0 4);
          ("_quantizer.buckets", book 0 4);
        ],
        "_quantizer.buckets: an entry in the archive, never read" );
      ( "a channel's reservoir in a codec of one channel",
        Modality.Flux_g,
        [
          ("_quantizer._index_to_val", book 0 4);
          ("_quantizer.quantizers.0._reservoir", book 0 9);
        ],
        "_quantizer.quantizers.0._reservoir: an entry in the archive, never \
         read" );
      ( "the reservoir of a channel past the last",
        Modality.Xp_bp,
        List.init 55 (fun i -> (channel i, book i 4))
        @ [ ("_quantizer.quantizers.55._reservoir", book 0 9) ],
        "_quantizer.quantizers.55._reservoir: an entry in the archive, never \
         read" );
    ]
    (fun (_, m, l, sub) ->
      raises_match (Exn.failure ~sub) (fun () ->
          Scalar.of_archive m (entries l)))

let () =
  exit
    (run "aion.scalar"
       [
         group "encode"
           [
             counts;
             counts64;
             float64;
             bisects;
             nan_edges;
             floor_;
             log_nan;
             asinh;
             channels;
           ];
         group "decode" [ upper_edge; out_of_range; past_last; decodes ];
         shapes;
         group "of_archive" [ layouts; unreadable ];
         group "structure" [ init; walk ];
         group "compiled" [ compiled; compiled_refusal ];
       ])
