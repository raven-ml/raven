(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The scalar codecs' laws on synthetic codebooks and codecs of init, and the
   codecs against upstream's goldens (tests/test_data at db7fba6df53b):
   upstream/edges.safetensors holds AION-1's codebooks under their directories'
   names, upstream/goldens.safetensors upstream's inputs, tokens and decoded
   values. Of the Gaia XP codecs both keep the coefficients and rows where a
   token may flip or is the first or the last (the rule is in the generator the
   dune file names); every edge of every codebook is decoded, which covers every
   token a golden holds.

   Tolerances are nx.mli's float32 bounds, in floats from the exact value
   rounded ({!Bounds}): [log] 1, [exp] 2, [asinh] and [sinh] 3, each applied to
   the input it is given. *)

open Windtrap
open Aion

let f32 = Bounds.f32
let row a = Nx.create Nx.float32 [| 1; Array.length a |] a
let values a = Nx.create Nx.float32 [| Array.length a |] a
let ints a = Nx.to_array (Nx.cast Nx.int64 a) |> Array.map Int64.to_int

let tokens a =
  Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)

(* The token the preamble states, on edges in order: the number of [e(0) …
   e(n-2)] below [y], and [n - 1] for NaN. *)
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

(* Encoding *)

(* Edges in order, or followed by NaN edges, and float32 or float64 values. *)
let searches =
  prop ~count:200
    "encode counts the edges below, NaN last, and bisects on NaN edges"
    Gen.(
      let* e = sorted in
      let* nans = frequency [ (1, constant 0); (1, int_range 1 6) ] in
      let e = Array.append e (Array.make nans Float.nan) in
      let* dt64 = bool in
      let+ ys = if dt64 then around64 e else around e in
      (e, nans, dt64, ys))
    (fun (e, nans, dt64, ys) ->
      let n = Array.length e in
      let expected = if nans = 0 then count e else bisect e in
      Array.iter
        (fun y ->
          cover "ties with an edge" (Array.exists (fun x -> x = y) e);
          cover "NaN" (Float.is_nan y);
          cover "above e(n-2)" (nans = 0 && n >= 2 && y > e.(n - 2));
          cover "below e(0)" (y < e.(0));
          cover "meets a NaN edge below a finite one"
            (bisect e y = n - 1 && count e y < n - 1);
          cover "float64, rounding onto an edge in float32"
            (dt64
            && Array.exists (fun x -> x = f32 y) e
            && not (Array.exists (fun x -> x = y) e)))
        ys;
      let t = Scalar.v Identity (row e) in
      let k =
        if dt64 then
          Scalar.encode t (Nx.create Nx.float64 [| Array.length ys |] ys)
        else Scalar.encode t (values ys)
      in
      equal (array int) (Array.map expected ys) (ints k))

(* Float32's neighbours of -3. *)
let below_3 = Int32.float_of_bits (Int32.succ (Int32.bits_of_float (-3.)))
let above_3 = Int32.float_of_bits (Int32.pred (Int32.bits_of_float (-3.)))

(* Edges at -3 and its two float32 neighbours: the token is 1 exactly when the
   transformed value is -3, and 3 for NaN, which Log keeps. *)
let floor_ =
  cases "Log clamps at float32 e⁻³, whose log is -3 exactly, and keeps NaN"
    ~id:(fun (v, _) -> string_of_float v)
    [
      (0., 1);
      (-0., 1);
      (-1., 1);
      (Float.neg_infinity, 1);
      (f32 (exp (-3.)), 1);
      (1e-30, 1);
      (Float.nan, 3);
    ]
    (fun (v, k) ->
      let t =
        Scalar.v (Log (exp (-3.))) (row [| below_3; -3.; above_3; 0. |])
      in
      equal (array int) [| k |] (ints (Scalar.encode t (values [| v |]))))

(* At float64 the clamp is e⁻³ rounded to float64, whose log is -3 exactly, as
   torch's: on edges at -3, a log one float64 above it would take token 3. *)
let floor64 =
  test "Log clamps a float64 at float64 e⁻³, whose log is -3 exactly" (fun () ->
      let t = Scalar.v (Log (exp (-3.))) (row [| -3.; -3.; -3.; 0.; 1. |]) in
      let v = Nx.create Nx.float64 [| 1 |] [| 1e-9 |] in
      equal (array int) [| 0 |] (ints (Scalar.encode t v)))

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
  prop "channel i is tokenized and decoded by codebook i" two
    (fun (a, b, rows, vs) ->
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

(* Decoding *)

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

(* Codecs of init at three modalities' shapes, and at another floor. *)
let codec =
  Gen.(
    let* m = of_list Modality.[ Flux_g; Z; Xp_bp ] in
    let* seed = int_range 0 1000 in
    let+ floor = of_list [ None; Some 0.05 ] in
    let t = Scalar.init ~key:(Nx.Rng.key seed) m in
    match (floor, Scalar.transform t) with
    | Some f, Log _ -> Scalar.v (Log f) (Scalar.edges t)
    | _ -> t)

let bits x = Array.map Int32.bits_of_float (Nx.to_array x)

(* Token [j] of channel [i] at row [i], column [j]. *)
let all_values =
  prop "values is decode of every token, bit for bit" codec (fun t ->
      cover "Log" (match Scalar.transform t with Log _ -> true | _ -> false);
      cover "Identity" (Scalar.transform t = Identity);
      cover "Asinh_asinh" (Scalar.transform t = Asinh_asinh);
      let c = Nx.dim 0 (Scalar.edges t) and n = Nx.dim 1 (Scalar.edges t) in
      let k =
        Nx.broadcast_to [| n; c |]
          (Nx.reshape [| n; 1 |] (Nx.arange Nx.int64 0 n 1))
      in
      equal (array int32)
        (bits (Nx.transpose (Scalar.decode t k)))
        (bits (Scalar.values t)))

(* Compiled with the codec as an argument, one function serves every codec: its
   walk reports the transform, a floor included, and its edges are an argument.
   The search is comparisons and gathers and the transforms are functions
   Rune.jit computes exactly on the host: bit for bit. *)
let signature () = Nx.Ptree.(v Scalar.walk @-> tensor @-> returns tensor)
let encode32 = Rune.jit (signature ()) Scalar.encode
let encode64 = Rune.jit (signature ()) Scalar.encode
let decode_c = Rune.jit (signature ()) Scalar.decode

(* A codec's refusals of a token raise at the call on the host, eagerly and
   compiled; the token past the last edge, redshift's 1025th class, which the
   transformer predicts, is named. *)
let out_of_range =
  cases "decode refuses a token outside the codebook" ~id:string_of_int
    [ -1; 3; 100; 1024 ] (fun k ->
      let n = if k = 1024 then 1024 else 3 in
      let t = Scalar.v Identity (row (Array.init n Float.of_int)) in
      let sub =
        if k = n then
          Printf.sprintf "token %d at [1] outside [0, %d), the class past" k n
        else Printf.sprintf "token %d at [1] outside [0, %d)" k n
      in
      raises_match (Exn.invalid_arg ~sub) (fun () ->
          Scalar.decode t (tokens [| 0; k |]));
      raises_match (Exn.invalid_arg ~sub) (fun () ->
          decode_c t (tokens [| 0; k |])))

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

(* of_archive's three layouts of codebooks. Row [i] holds [i + j / 10], so a
   transposed or reordered channel shows. *)
let book i n =
  Nx.create Nx.float32 [| n |]
    (Array.init n (fun j -> float_of_int i +. (float_of_int j /. 10.)))

let entries l = Nx_io.Archive.of_list (List.map (fun (k, v) -> (k, Nx.P v)) l)
let channel i = Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i
let reservoir i = Printf.sprintf "_quantizer.quantizers.%d._reservoir" i

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
               [ (channel i, book i 4); (reservoir i, book 0 9) ])),
        55 );
    ]
    (fun (_, m, l, c) ->
      let t = Scalar.of_archive m (entries l) in
      equal (array int) [| c; 4 |] (Nx.shape (Scalar.edges t));
      equal (array float_exact)
        (Array.concat (List.init c (fun i -> Nx.to_array (book i 4))))
        (Nx.to_array (Scalar.edges t)))

let unreadable =
  let never k = k ^ ": an entry in the archive, never read" in
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
        never "_quantizer.buckets" );
      ( "a channel's reservoir in a codec of one channel",
        Modality.Flux_g,
        [ ("_quantizer._index_to_val", book 0 4); (reservoir 0, book 0 9) ],
        never (reservoir 0) );
      ( "the reservoir of a channel past the last",
        Modality.Xp_bp,
        List.init 55 (fun i -> (channel i, book i 4))
        @ [ (reservoir 55, book 0 9) ],
        never (reservoir 55) );
    ]
    (fun (_, m, l, sub) ->
      raises_match (Exn.failure ~sub) (fun () ->
          Scalar.of_archive m (entries l)))

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
      let e = Scalar.edges t in
      equal (array int) [| Modality.tokens (Scalar s); 1024 |] (Nx.shape e);
      let sorted, _ = Nx.sort ~axis:1 e in
      equal (array float_exact) (Nx.to_array sorted) (Nx.to_array e))

(* The walk reports a Log's floor by its bits: floors of other bits, as [0.] and
   [-0.], key other programs, and equal floors the same, whatever the edges. *)
let walk =
  test "the walk reports the floor by its bits, not the edges" (fun () ->
      let visits tr e =
        List.map
          (Format.asprintf "%a" Nx.Ptree.pp_visit)
          (Nx.Ptree.visits (Nx.Ptree.v Scalar.walk) (Scalar.v tr (row e)))
      in
      not_equal (list string)
        (visits (Log 0.) [| 0.; 1. |])
        (visits (Log (-0.)) [| 0.; 1. |]);
      equal (list string)
        (visits (Log (exp (-3.))) [| 0.; 1. |])
        (visits (Log (exp (-3.))) [| -2.; 5. |]))

(* Four objects' values: edges, normal draws and values that are not finite. *)
let values_of t =
  let e = Nx.to_array (Scalar.edges t) in
  let c = Nx.dim 0 (Scalar.edges t) in
  Gen.(
    let+ v =
      array
        ~size:(constant (4 * c))
        (frequency
           [
             (2, of_list (Array.to_list e));
             (3, map f32 (float_range (-4.) 4.));
             (1, of_list [ Float.nan; Float.infinity; Float.neg_infinity; -0. ]);
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
      equal (array int32) (bits (Scalar.decode t k)) (bits (decode_c t k)))

(* Upstream's goldens *)

let edges = Nx_io.load_safetensors "upstream/edges.safetensors"
let goldens = Nx_io.load_safetensors "upstream/goldens.safetensors"

let entry name a =
  match Nx_io.Archive.find name a with
  | Some p -> p
  | None -> failwith ("no entry " ^ name)

(* A codec's name in the fixtures: NAME of its Hub.file,
   codecs/NAME/model.safetensors, which the generator read. *)
let id m = Filename.basename (Filename.dirname (Hub.file (Scalar m)))

let aion m =
  Scalar.v (Scalar.transform_of m) (Nx.unpack Nx.float32 (entry (id m) edges))

let golden m part = entry (id m ^ "." ^ part) goldens

(* A token is upstream's or, under a transform that rounds, moves across one
   distinct edge value that lies where nx's transform may land. Each other
   difference is listed as (index, upstream's token, raven's). *)
let golden_tokens =
  cases "tokens are upstream's but where rounding crosses an edge" ~id
    Modality.scalars (fun m ->
      let t = aion m and tr = Scalar.transform_of m in
      let x = Nx.unpack Nx.float32 (golden m "input") in
      let xs = Nx.to_array x in
      let (Nx.P want) = golden m "encoded" in
      let got = ints (Scalar.encode t x) in
      let e = Scalar.edges t in
      let channel c = Nx.to_array (Nx.cast Nx.float64 (Nx.slice [ I c ] e)) in
      let channels = Array.init (Nx.dim 0 e) channel in
      let refuted = ref [] in
      Array.iteri
        (fun i w ->
          let c = channels.(i mod Array.length channels) in
          if not (Scalar_bounds.crosses tr c xs.(i) w got.(i)) then
            refuted := (i, w, got.(i)) :: !refuted)
        (ints want);
      equal (list (triple int int int)) [] (List.rev !refuted))

(* Token [k] decodes to the inverse transform of its edge within nx's bounds,
   bit for bit under the identity. Each other value is listed as (channel,
   token, value). *)
let every_edge =
  cases "every token decodes within nx's bound of its edge's inverse" ~id
    Modality.scalars (fun m ->
      let t = aion m and tr = Scalar.transform_of m in
      let e = Scalar.edges t in
      let c = Nx.dim 0 e and n = Nx.dim 1 e in
      let k = Nx.init Nx.int64 [| n; c |] (fun i -> Int64.of_int i.(0)) in
      let d = Nx.to_array (Scalar.decode t k) in
      let refuted = ref [] in
      for k = 0 to n - 1 do
        for ch = 0 to c - 1 do
          let v = Nx.item [ ch; k ] e and got = d.((k * c) + ch) in
          let ok =
            if Float.is_nan v then Float.is_nan got
            else if tr = Scalar.Identity then Float.equal v got
            else Bounds.inside (Scalar_bounds.inverse tr v) got
          in
          if not ok then refuted := (ch, k, got) :: !refuted
        done
      done;
      equal (list (triple int int float_exact)) [] (List.rev !refuted))

(* The log codebooks whose training sample reached the clamp: their lowest edge
   is the clamp's logarithm. *)
let at_clamp =
  Modality.
    [ Flux_g; Flux_r; Flux_i; Flux_w1; Flux_w2; Flux_w3; Flux_w4; Shape_r ]

let logs =
  List.filter
    (fun m -> match Scalar.transform_of m with Log _ -> true | _ -> false)
    Modality.scalars

(* A clamped input takes the bin its codebook built for the clamp: every edge
   within nx's bound of the clamp's exact logarithm is nx's float32 logarithm of
   the clamp, bit for bit, and so is the lowest edge of the codebooks whose
   sample reached the clamp, where a clamped input takes token 0. *)
let clamp =
  cases "edges at the log clamp are nx's log of it" ~id logs (fun m ->
      let t = aion m in
      let floor = match Scalar.transform_of m with Log f -> f | _ -> nan in
      let c =
        Nx.item [] (Nx.log (Nx.clamp ~min:floor (Nx.scalar Nx.float32 0.)))
      in
      let near = Scalar_bounds.forward (Log floor) 0. in
      let e = Nx.to_array (Scalar.edges t) in
      Array.iter (fun v -> if Bounds.inside near v then equal float_exact c v) e;
      if List.mem m at_clamp then begin
        equal float_exact c e.(0);
        equal (array int) [| 0 |] (ints (Scalar.encode t (values [| 0. |])))
      end)

let () =
  exit
    (run "aion.scalar"
       [
         group "encode" [ searches; floor_; floor64; asinh; channels ];
         group "decode" [ upper_edge; all_values; out_of_range ];
         shapes;
         group "of_archive" [ layouts; unreadable ];
         group "structure" [ init; walk ];
         compiled;
         group "upstream" [ golden_tokens; every_edge; clamp ];
       ])
