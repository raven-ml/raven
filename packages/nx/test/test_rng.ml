(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Random number generation. Keys, their derivations, the scope and each
   sampler's support are laws over drawn keys and parameters; each distribution
   is held to its closed-form moments or pmf at a fixed key, with a tolerance of
   about five standard errors. *)

open Windtrap
open Nx_test
module Rng = Nx.Rng

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let pp_int32 ppf x = Format.fprintf ppf "%ldl" x
let words (k : Rng.t) = Nx.to_array (k :> Nx.int32_t)
let keys = Testable.contramap words (array int32)
let of_words w = Rng.of_tensor (Nx.create Nx.int32 [| 2 |] w)

(* A draw of any dtype as its values in float64, NaN equal to NaN. *)
let drawn t = Ref.of_nx (Nx.cast Nx.float64 t)
let exactly = Ref.witness (close ~rel:0. ())

(* Generators *)

let word =
  Gen.frequency
    [
      (4, Gen.map Int32.of_int (Gen.int_range (-3) 3));
      (4, Gen.int32);
      ( 1,
        Gen.of_list ~pp:pp_int32
          [ Int32.min_int; Int32.succ Int32.min_int; Int32.max_int ] );
    ]

let key = Gen.map (fun (a, b) -> of_words [| a; b |]) (Gen.pair word word)

(* Words of shape [[...; 2]]: keys, or counters, along the last axis. *)
let word_tensor =
  let open Gen in
  let* lead = array ~size:(int_range 0 2) (int_range 0 3) in
  let s = Array.append lead [| 2 |] in
  let+ w = array ~size:(constant (Ref.numel s)) word in
  Nx.create Nx.int32 s w

(* A count, a seed or a counter: small values, the whole [int] range, its
   extremes and the edges of the two 32-bit words a counter is packed into. *)
let integer =
  Gen.frequency
    [
      (4, Gen.int_range (-3) 3);
      (2, Gen.int);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [
            min_int;
            min_int + 1;
            max_int - 1;
            max_int;
            0xFFFF_FFFF;
            0x1_0000_0000;
            0x1_0000_0001;
            -0x1_0000_0000;
          ] );
    ]

let shape =
  Gen.with_pp pp_shape (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 5))

type floating =
  | F : { name : string; dtype : (float, 'b) Nx.dtype; bits : int } -> floating

let floatings =
  [
    F { name = "float8_e5m2"; dtype = Nx.float8_e5m2; bits = 3 };
    F { name = "float8_e4m3"; dtype = Nx.float8_e4m3; bits = 4 };
    F { name = "bfloat16"; dtype = Nx.bfloat16; bits = 8 };
    F { name = "float16"; dtype = Nx.float16; bits = 11 };
    F { name = "float32"; dtype = Nx.float32; bits = 24 };
    F { name = "float64"; dtype = Nx.float64; bits = 53 };
  ]

let pp_floating ppf (F f) = Format.pp_print_string ppf f.name
let floating = Gen.of_list ~pp:pp_floating floatings

(* The dtypes a distribution's parameters come in. *)
let parameter_dtype =
  Gen.of_list ~pp:pp_floating
    (List.filter
       (fun (F f) -> List.mem f.name [ "float16"; "float32"; "float64" ])
       floatings)

let chosen l = Gen.of_list ~pp:pp_float l

let probability =
  Gen.frequency
    [
      (4, Gen.float_range 0. 1.);
      ( 2,
        chosen [ Float.nan; neg_infinity; -1.; -0.; 0.; 1.; 1.5; infinity; 0.5 ]
      );
    ]

let bound =
  Gen.frequency
    [
      (4, Gen.float_range (-3.) 3.);
      (2, chosen [ neg_infinity; infinity; -9.; -6.; -0.; 0.; 5.5; 7.; 9. ]);
    ]

(* Positive, over many orders of magnitude. *)
let concentration =
  Gen.frequency
    [
      (4, Gen.map (fun e -> 10. ** e) (Gen.float_range (-3.) 3.));
      (1, chosen [ 1e-3; 0.03; 1.; 1e3 ]);
    ]

let rate =
  Gen.frequency
    [
      (3, Gen.float_range 0. 60.);
      ( 2,
        chosen
          [ Float.nan; -1.; -0.; 0.; 1e-3; 9.99; 10.; 10.01; 1e3; 1e4; 9e4 ] );
    ]

let logit =
  Gen.frequency
    [ (4, Gen.float_range (-5.) 5.); (1, chosen [ neg_infinity; 0.; 30. ]) ]

(* Every function of this module that takes a key, at fixed arguments. *)

let p3 = Nx.create Nx.float32 [| 3 |] [| 0.2; 0.5; 0.9 |]

let keyed : (string * (Rng.t -> float Ref.t)) list =
  [
    ("bits", fun k -> drawn (Rng.bits k [| 3 |]));
    ("uniform", fun k -> drawn (Rng.uniform k Nx.float32 [| 3 |]));
    ("uniform at float64", fun k -> drawn (Rng.uniform k Nx.float64 [| 3 |]));
    ("normal", fun k -> drawn (Rng.normal k Nx.float32 [| 3 |]));
    ("normal at float64", fun k -> drawn (Rng.normal k Nx.float64 [| 3 |]));
    ("randint", fun k -> drawn (Rng.randint k ~low:(-3) ~high:7 [| 4 |]));
    ("bernoulli", fun k -> drawn (Rng.bernoulli k p3));
    ("truncated_normal", fun k -> drawn (Rng.truncated_normal k (Nx.neg p3) p3));
    ("gumbel", fun k -> drawn (Rng.gumbel k Nx.float32 [| 3 |]));
    ("exponential", fun k -> drawn (Rng.exponential k Nx.float32 [| 3 |]));
    ("gamma", fun k -> drawn (Rng.gamma k p3));
    ("beta", fun k -> drawn (Rng.beta k p3 (Nx.flip p3)));
    ("dirichlet", fun k -> drawn (Rng.dirichlet k p3));
    ( "poisson",
      fun k ->
        drawn (Rng.poisson k (Nx.create Nx.float32 [| 2 |] [| 3.; 30. |])) );
    ( "categorical",
      fun k -> drawn (Rng.categorical k (Nx.broadcast_to [| 4; 3 |] p3)) );
    ("permutation", fun k -> drawn (Rng.permutation k 5));
    ("shuffle", fun k -> drawn (Rng.shuffle k (Nx.arange Nx.float64 0 5 1)));
    ( "split",
      fun k ->
        drawn
          (Nx.stack
             (List.map
                (fun (k : Rng.t) -> (k :> Nx.int32_t))
                (Array.to_list (Rng.split k)))) );
    ("split_batch", fun k -> drawn (Rng.split_batch ~n:3 k :> Nx.int32_t));
    ("fold_in", fun k -> drawn (Rng.fold_in k 7 :> Nx.int32_t));
    ( "fold_in_tensor",
      fun k ->
        drawn (Rng.fold_in_tensor k (Nx.scalar Nx.int32 7l) :> Nx.int32_t) );
    ("fold_in_axis", fun k -> drawn (Rng.fold_in_axis k :> Nx.int32_t));
  ]

(* Keys *)

(* The same two words, held in a view of each kind. *)
let key_layouts =
  let create shape w = Nx.create Nx.int32 shape w in
  [
    ("contiguous", fun w -> create [| 2 |] w);
    ("reversed", fun w -> Nx.flip (create [| 2 |] [| w.(1); w.(0) |]));
    ( "every other word",
      fun w ->
        Nx.squeeze ~axes:[ -1 ]
          (Nx.sliding_window ~window:1 ~step:2
             (create [| 4 |] [| w.(0); 0l; w.(1); 0l |])) );
    ( "a row of a transposed matrix",
      fun w ->
        Nx.slice [ I 0 ]
          (Nx.transpose (create [| 2; 2 |] [| w.(0); 5l; w.(1); 6l |])) );
    ( "a row past an offset",
      fun w -> Nx.slice [ I 1 ] (create [| 2; 2 |] [| 5l; 6l; w.(0); w.(1) |])
    );
    ( "a row of a broadcast",
      fun w ->
        Nx.slice [ I 2 ] (Nx.broadcast_to [| 3; 2 |] (create [| 1; 2 |] w)) );
  ]

let held_key =
  Gen.pair (Gen.pair word word)
    (Gen.of_list
       ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
       key_layouts)

let refuses_batches =
  let batch = Rng.split_batch ~n:4 (Rng.key 0) in
  cases ~name:fst "a batch of keys is refused by" keyed (fun (_, draw) ->
      raises_invalid_arg (fun () -> draw batch))

let key_tests =
  group "keys"
    [
      prop
        "equal seeds give equal keys, and distinct seeds distinct keys (nx.mli \
         is silent on the second)"
        (Gen.pair integer integer)
        ~examples:[ (5, 5) ]
        (fun (a, b) ->
          equal bool (a = b) (words (Rng.key a) = words (Rng.key b)));
      prop "of_tensor inverts the coercion to words" key
        (Law.round_trip keys (tensor int32)
           (fun k -> (k :> Nx.int32_t))
           Rng.of_tensor);
      prop "the coercion inverts of_tensor on a batch of words" word_tensor
        (Law.round_trip (tensor int32) (tensor int32)
           (fun t -> (Rng.of_tensor t :> Nx.int32_t))
           Fun.id);
      cases
        ~name:(fun s -> Format.asprintf "of_tensor refuses shape %a" pp_shape s)
        "of_tensor"
        [ [||]; [| 3 |]; [| 0 |]; [| 2; 3 |]; [| 2; 1 |] ]
        (fun s ->
          raises_invalid_arg (fun () -> Rng.of_tensor (Nx.zeros Nx.int32 s)));
      refuses_batches;
      test "ptree rebuilds a key and checks it as of_tensor does" (fun () ->
          let k = Rng.key 3 in
          equal keys k (Nx.Ptree.map Rng.ptree (fun _ t -> t) k);
          raises_invalid_arg (fun () ->
              Nx.Ptree.map Rng.ptree (fun _ t -> Nx.sum t) k));
    ]

let splits =
  group "split"
    [
      test "split makes two subkeys by default" (fun () ->
          equal int 2 (Array.length (Rng.split (Rng.key 1))));
      prop "split's subkeys differ from each other and from their parent"
        (Gen.pair key (Gen.int_range 1 16))
        (fun (k, n) ->
          let subkeys = Array.to_list (Rng.split ~n k) in
          equal int n (List.length subkeys);
          let all = List.map words (k :: subkeys) in
          equal int (n + 1) (List.length (List.sort_uniq compare all)));
      prop "split_batch holds split's subkeys as its rows"
        (Gen.pair key (Gen.int_range 1 8))
        (fun (k, n) ->
          let batch = (Rng.split_batch ~n k :> Nx.int32_t) in
          equal (array int) [| n; 2 |] (Nx.shape batch);
          Array.iteri
            (fun i s ->
              equal
                ~msg:(Printf.sprintf "row %d" i)
                (array int32) (words s)
                (Nx.to_array (Nx.slice [ I i ] batch)))
            (Rng.split ~n k));
      cases
        ~name:(fun (name, _) -> name ^ " refuses n below one")
        "count"
        [
          ("split", fun n -> ignore (Rng.split ~n (Rng.key 0)));
          ("split_batch", fun n -> ignore (Rng.split_batch ~n (Rng.key 0)));
        ]
        (fun (_, f) ->
          raises_invalid_arg (fun () -> f 0);
          raises_invalid_arg (fun () -> f (-1)));
    ]

let fold_ins =
  group "fold_in"
    [
      prop "fold_in gives distinct counters distinct keys"
        (Gen.triple key integer integer)
        ~examples:
          [ (Rng.key 0, 1, 0x1_0000_0001); (Rng.key 0, -1, 0xFFFF_FFFF) ]
        (fun (k, a, b) ->
          assume (a <> b);
          not_equal keys (Rng.fold_in k a) (Rng.fold_in k b));
      prop "fold_in_tensor agrees with fold_in on every int32"
        (Gen.pair key word)
        ~examples:[ (Rng.key 0, -1l) ]
        (fun (k, i) ->
          equal keys
            (Rng.fold_in k (Int32.to_int i))
            (Rng.fold_in_tensor k (Nx.scalar Nx.int32 i)));
      prop "fold_in_axis outside a map is fold_in 0" key (fun k ->
          equal keys (Rng.fold_in k 0) (Rng.fold_in_axis k));
    ]

(* The block function. nx.mli names the generator Threefry and no more; the
   backend contract fixes it as Threefry-2x32 of 20 rounds, bit for bit, and
   [Nx_effect.threefry] is where the backend of [Nx] exposes it. *)

(* Threefry-2x32-20 on one block, in Int32 arithmetic, which wraps as uint32
   does. *)
let threefry2x32 (k0, k1) (c0, c1) =
  let rotl x r =
    Int32.logor (Int32.shift_left x r) (Int32.shift_right_logical x (32 - r))
  in
  let ks = [| k0; k1; Int32.logxor (Int32.logxor 0x1BD11BDAl k0) k1 |] in
  let rots = [| 13; 15; 26; 6; 17; 29; 16; 24 |] in
  let x0 = ref (Int32.add c0 k0) and x1 = ref (Int32.add c1 k1) in
  for r = 0 to 19 do
    x0 := Int32.add !x0 !x1;
    x1 := Int32.logxor (rotl !x1 rots.(r mod 8)) !x0;
    if (r + 1) mod 4 = 0 then (
      let s = (r + 1) / 4 in
      x0 := Int32.add !x0 ks.(s mod 3);
      x1 := Int32.add (Int32.add !x1 ks.((s + 1) mod 3)) (Int32.of_int s))
  done;
  (!x0, !x1)

(* Keys and counters of one shape. *)
let blocks =
  let open Gen in
  let* k = word_tensor in
  let+ c = array ~size:(constant (Nx.numel k)) word in
  (k, Nx.create Nx.int32 (Nx.shape k) c)

let threefry =
  group "threefry block"
    [
      cases
        ~name:(fun (name, _, _, _) ->
          "threefry gives the known answer for " ^ name)
        "known answers"
        [
          ("zeros", (0l, 0l), (0l, 0l), (1797259609l, -1715843330l));
          ("ones", (-1l, -1l), (-1l, -1l), (481924860l, -1157616665l));
          ( "pi",
            (0x13198a2el, 0x03707344l),
            (0x243f6a88l, 0x85a308d3l),
            (-997049700l, 1212020640l) );
        ]
        (fun (_, (k0, k1), (c0, c1), (e0, e1)) ->
          let mk a b = Nx.create Nx.int32 [| 2 |] [| a; b |] in
          equal (array int32) [| e0; e1 |]
            (Nx.to_array (Nx_effect.threefry (mk k0 k1) (mk c0 c1))));
      prop "threefry applies Threefry-2x32-20 to each block" blocks
        (fun (k, c) ->
          let kw = Ref.of_nx k and cw = Ref.of_nx c in
          let expected = Array.copy kw.data in
          for i = 0 to (Array.length expected / 2) - 1 do
            let a, b =
              threefry2x32
                (kw.data.(2 * i), kw.data.((2 * i) + 1))
                (cw.data.(2 * i), cw.data.((2 * i) + 1))
            in
            expected.(2 * i) <- a;
            expected.((2 * i) + 1) <- b
          done;
          equal (Ref.witness int32)
            { kw with data = expected }
            (Ref.of_nx (Nx_effect.threefry k c)));
    ]

(* Purity. Every function of a key gives the same values from the same words
   however they are held, and every parameter gives the same draw in any layout.
   Comparing a view against its copy also compares two calls. *)

let outcome f =
  match f () with r -> Ok r | exception Invalid_argument _ -> Error ()

let same_outcome copy view =
  equal (result exactly unit) (outcome copy) (outcome view)

let parametric :
    (string
    * float Gen.t
    * (Rng.t -> (float, Nx.float32_elt) Nx.t -> float Ref.t))
    list =
  [
    ("bernoulli's p", probability, fun k p -> drawn (Rng.bernoulli k p));
    ( "truncated_normal's lower bound",
      bound,
      fun k t -> drawn (Rng.truncated_normal k t (Nx.scalar Nx.float32 1.5)) );
    ( "truncated_normal's upper bound",
      bound,
      fun k t -> drawn (Rng.truncated_normal k (Nx.scalar Nx.float32 (-1.5)) t)
    );
    ("gamma's concentration", concentration, fun k c -> drawn (Rng.gamma k c));
    ( "beta's first concentration",
      concentration,
      fun k a -> drawn (Rng.beta k a (Nx.scalar Nx.float32 2.)) );
    ( "beta's second concentration",
      concentration,
      fun k b -> drawn (Rng.beta k (Nx.scalar Nx.float32 2.) b) );
    ( "dirichlet's concentration",
      concentration,
      fun k c -> drawn (Rng.dirichlet k c) );
    ("poisson's rate", rate, fun k r -> drawn (Rng.poisson k r));
    ("categorical's logits", logit, fun k l -> drawn (Rng.categorical k l));
    ("shuffle's tensor", logit, fun k t -> drawn (Rng.shuffle k t));
  ]

let purity =
  group "purity"
    (List.map
       (fun (name, draw) ->
         prop (name ^ " gives the same values from a key held in any layout")
           held_key (fun ((a, b), (_, view)) ->
             equal exactly
               (draw (of_words [| a; b |]))
               (draw (Rng.of_tensor (view [| a; b |])))))
       keyed
    @ List.map
        (fun (name, value, draw) ->
          prop
            (name ^ " gives the same draw in any layout")
            (Gen.pair key (viewed ~pp:pp_float Nx.float32 value))
            (fun (k, t) ->
              same_outcome (fun () -> draw k (Nx.copy t)) (fun () -> draw k t)))
        parametric)

(* Supports *)

let floats t = (drawn t).data

let uniforms =
  group "uniform and bits"
    [
      prop
        "uniform draws multiples of 2^-p in [0, 1 - 2^-p], p the significand \
         width" (Gen.triple key floating shape) (fun (k, F f, s) ->
          let u = drawn (Rng.uniform k f.dtype s) in
          equal (array int) s u.shape;
          let top = 1. -. Float.ldexp 1. (-f.bits) in
          Array.iter
            (fun v ->
              at_least float_exact ~than:0. v;
              at_most float_exact ~than:top v;
              is_true ~msg:"on the grid"
                (Float.is_integer (Float.ldexp v f.bits)))
            u.data);
      prop "uniform at float32 is the low 24 bits of bits, scaled by 2^-24"
        (Gen.pair key shape) (fun (k, s) ->
          let low24 w =
            Int32.to_float (Int32.logand w 0xFF_FFFFl) *. Float.ldexp 1. (-24)
          in
          let b = Ref.of_nx (Rng.bits k s) in
          equal (array int) s b.shape;
          equal exactly (Ref.map low24 b) (drawn (Rng.uniform k Nx.float32 s)));
      cases
        ~name:(fun (name, _) -> name ^ " refuses a negative dimension")
        "negative dimension"
        [
          ("bits", fun () -> ignore (Rng.bits (Rng.key 0) [| 2; -1 |]));
          ( "uniform",
            fun () -> ignore (Rng.uniform (Rng.key 0) Nx.float32 [| 2; -1 |]) );
          ( "normal",
            fun () -> ignore (Rng.normal (Rng.key 0) Nx.float32 [| 2; -1 |]) );
          ( "gumbel",
            fun () -> ignore (Rng.gumbel (Rng.key 0) Nx.float32 [| 2; -1 |]) );
          ( "exponential",
            fun () ->
              ignore (Rng.exponential (Rng.key 0) Nx.float32 [| 2; -1 |]) );
          ( "randint",
            fun () -> ignore (Rng.randint (Rng.key 0) ~high:3 [| 2; -1 |]) );
          ("rand", fun () -> ignore (Nx.rand Nx.float32 [| 2; -1 |]));
          ("randn", fun () -> ignore (Nx.randn Nx.float32 [| 2; -1 |]));
          ("keyless randint", fun () -> ignore (Nx.randint ~high:3 [| 2; -1 |]));
        ]
        (fun (_, f) -> raises_invalid_arg f);
    ]

let continuous =
  group "normal, gumbel and exponential"
    [
      prop
        "normal, gumbel and exponential draws are finite, of the draw's shape"
        (Gen.triple key floating shape) (fun (k, F f, s) ->
          List.iter
            (fun (name, t) ->
              equal ~msg:name (array int) s (Nx.shape t);
              Array.iter
                (fun v -> is_true ~msg:name (Float.is_finite v))
                (floats t))
            [
              ("normal", Rng.normal k f.dtype s);
              ("gumbel", Rng.gumbel k f.dtype s);
              ("exponential", Rng.exponential k f.dtype s);
            ]);
      prop "exponential draws are non-negative" (Gen.triple key floating shape)
        (fun (k, F f, s) ->
          Array.iter
            (fun v -> at_least float_exact ~than:0. v)
            (floats (Rng.exponential k f.dtype s)));
    ]

(* [low, high) across the edges of int32, where the offset from [low] exceeds
   what int32 holds. *)
let int32_bound =
  Gen.frequency
    [
      (4, Gen.int_range (-10) 10);
      (2, Gen.int_range (-0x8000_0000) 0x7FFF_FFFF);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ -0x8000_0000; -0x7FFF_FFFF; 0x7FFF_FFFE; 0x7FFF_FFFF; -1; 0 ] );
    ]

let randints =
  group "randint"
    [
      prop "randint draws lie in [low, high)"
        (Gen.triple key int32_bound int32_bound)
        ~examples:[ (Rng.key 1, -0x8000_0000, 0x7FFF_FFFF) ]
        (fun (k, a, b) ->
          assume (a <> b);
          let low = Int.min a b and high = Int.max a b in
          cover "a range wider than 2^31" (high - low > 0x8000_0000);
          Array.iter
            (fun v ->
              let v = Int32.to_int v in
              at_least int ~than:low v;
              less int ~than:high v)
            (Nx.to_array (Rng.randint k ~low ~high [| 64 |])));
      cases
        ~name:(fun (name, _, _) -> "randint refuses " ^ name)
        "range"
        [
          ("low = high", 3, 3);
          ("low > high", 4, 3);
          ("a low below int32", -0x8000_0001, 0);
          ("a high above int32", 0, 0x8000_0000);
        ]
        (fun (_, low, high) ->
          raises_invalid_arg (fun () ->
              Rng.randint (Rng.key 0) ~low ~high [| 2 |]);
          raises_invalid_arg (fun () -> Nx.randint ~low ~high [| 2 |]));
    ]

let bernoullis =
  group "bernoulli"
    [
      prop
        "bernoulli is true where p >= 1, false where p <= 0 or NaN, in p's \
         shape"
        (Gen.bind parameter_dtype (fun (F f) ->
             Gen.pair key (viewed ~pp:pp_float f.dtype probability)
             |> Gen.map (fun (k, p) -> (k, drawn p, drawn (Rng.bernoulli k p)))))
        (fun (_, p, b) ->
          equal (array int) p.shape b.shape;
          Array.iteri
            (fun i p ->
              if p >= 1. then equal ~msg:"p >= 1" float_exact 1. b.data.(i)
              else if not (p > 0.) then
                equal ~msg:"p <= 0 or NaN" float_exact 0. b.data.(i))
            p.data);
    ]

let truncated_normals =
  group "truncated_normal"
    [
      prop
        "truncated_normal lies between its bounds, given in either order, in \
         their broadcast shape"
        (Gen.bind parameter_dtype (fun (F f) ->
             Gen.triple key (viewed ~pp:pp_float f.dtype bound) bound
             |> Gen.map (fun (k, t, u) ->
                 let u' = Nx.scalar f.dtype u in
                 ( drawn t,
                   (drawn u').data.(0),
                   drawn (Rng.truncated_normal k t u'),
                   drawn (Rng.truncated_normal k u' t) ))))
        (fun (t, u, forward, backward) ->
          List.iter
            (fun (name, r) ->
              equal ~msg:name (array int) t.shape r.Ref.shape;
              Array.iteri
                (fun i v ->
                  let lo = Float.min t.data.(i) u
                  and hi = Float.max t.data.(i) u in
                  cover "a bound past 5.4" (Float.abs t.data.(i) > 5.4);
                  at_least ~msg:name float_exact ~than:lo v;
                  at_most ~msg:name float_exact ~than:hi v)
                r.data)
            [ ("lower first", forward); ("upper first", backward) ]);
      test "truncated_normal refuses bounds that do not broadcast" (fun () ->
          raises_invalid_arg (fun () ->
              Rng.truncated_normal (Rng.key 0)
                (Nx.zeros Nx.float32 [| 3 |])
                (Nx.ones Nx.float32 [| 4 |])));
    ]

let shape_with_components =
  Gen.with_pp pp_shape
    (Gen.map
       (fun (lead, c) -> Array.append lead [| c |])
       (Gen.pair
          (Gen.array ~size:(Gen.int_range 0 2) (Gen.int_range 0 3))
          (Gen.int_range 2 4)))

let no_layout = Gen.constant ~pp:pp_layout []

let gammas =
  group "gamma, beta and dirichlet"
    [
      prop "gamma draws are non-negative and finite"
        (Gen.bind parameter_dtype (fun (F f) ->
             Gen.pair key (viewed ~pp:pp_float f.dtype concentration)
             |> Gen.map (fun (k, c) -> floats (Rng.gamma k c))))
        (Array.iter (fun v ->
             at_least float_exact ~than:0. v;
             is_true (Float.is_finite v)));
      prop "beta draws lie in [0, 1]"
        (Gen.bind parameter_dtype (fun (F f) ->
             Gen.triple key
               (viewed ~pp:pp_float f.dtype concentration)
               concentration
             |> Gen.map (fun (k, a, b) ->
                 floats (Rng.beta k a (Nx.scalar f.dtype b)))))
        (Array.iter (fun v ->
             at_least float_exact ~than:0. v;
             at_most float_exact ~than:1. v));
      prop "every row of a dirichlet draw is non-negative and sums to one"
        (Gen.bind parameter_dtype (fun (F f) ->
             Gen.pair key
               (viewed ~shape:shape_with_components ~layout:no_layout
                  ~pp:pp_float f.dtype concentration)
             |> Gen.map (fun (k, c) ->
                 let tol = if f.bits = 53 then 1e-12 else 1e-2 in
                 (tol, drawn c, drawn (Rng.dirichlet k c)))))
        (fun (tol, c, d) ->
          equal (array int) c.shape d.shape;
          Array.iter (fun v -> at_least float_exact ~than:0. v) d.data;
          Array.iter
            (fun s -> equal (float tol) 1. s)
            (Ref.reduce ~axes:[ -1 ] ( +. ) 0. d).data);
      cases
        ~name:(fun (name, _) -> name)
        "parameter shapes"
        [
          ( "beta refuses concentrations that do not broadcast",
            fun () ->
              ignore
                (Rng.beta (Rng.key 0)
                   (Nx.ones Nx.float32 [| 3 |])
                   (Nx.ones Nx.float32 [| 4 |])) );
          ( "dirichlet refuses a single component",
            fun () ->
              ignore (Rng.dirichlet (Rng.key 0) (Nx.ones Nx.float32 [| 4; 1 |]))
          );
          ( "dirichlet refuses a scalar concentration",
            fun () ->
              ignore (Rng.dirichlet (Rng.key 0) (Nx.scalar Nx.float32 1.)) );
        ]
        (fun (_, f) -> raises_invalid_arg f);
    ]

let poissons =
  group "poisson"
    [
      prop
        "poisson counts are non-negative, and zero where the rate is not \
         positive"
        (Gen.bind parameter_dtype (fun (F f) ->
             Gen.pair key (viewed ~pp:pp_float f.dtype rate)
             |> Gen.map (fun (k, r) -> (drawn r, drawn (Rng.poisson k r)))))
        (fun (r, c) ->
          equal (array int) r.shape c.shape;
          Array.iteri
            (fun i v ->
              at_least float_exact ~than:0. v;
              if not (r.data.(i) > 0.) then equal float_exact 0. v)
            c.data);
    ]

(* Logits with an axis to draw along, which has at least one category. *)
let logits_along =
  Gen.bind parameter_dtype (fun (F f) ->
      let open Gen in
      let* t =
        viewed
          ~shape:(array ~size:(int_range 1 3) (int_range 0 4))
          ~pp:pp_float f.dtype logit
      in
      let+ k = key and+ axis = int_range (-Nx.ndim t) (Nx.ndim t - 1) in
      (drawn t, axis, fun () -> drawn (Rng.categorical k ~axis t)))

let categoricals =
  group "categorical"
    [
      prop
        "categorical gives one index per lane along axis, never on a -inf \
         logit beside a finite one"
        logits_along (fun (t, axis, draw) ->
          let a = Ref.axis t axis in
          let n = t.shape.(a) in
          assume (n > 0);
          let idx = draw () in
          (* Lane [l] of the logits with [axis] moved last is
             [idx.data.(l)]'s. *)
          let lanes = Ref.moveaxis a (-1) t in
          equal (array int) (Array.sub lanes.shape 0 (Ref.ndim t - 1)) idx.shape;
          Array.iteri
            (fun l i ->
              let i = int_of_float i in
              at_least int ~than:0 i;
              less int ~than:n i;
              let lane = Array.sub lanes.data (l * n) n in
              let finite = Array.exists Float.is_finite lane in
              cover "a lane with a -inf logit"
                (Array.exists (fun v -> v = neg_infinity) lane && finite);
              if finite then
                is_true ~msg:"drawn on a finite logit"
                  (Float.is_finite lane.(i)))
            idx.data);
      test "categorical's axis defaults to the last" (fun () ->
          let t =
            Nx.create Nx.float32 [| 2; 3 |] [| 0.; 1.; 2.; 2.; 0.5; -1. |]
          in
          equal exactly
            (drawn (Rng.categorical (Rng.key 42) ~axis:1 t))
            (drawn (Rng.categorical (Rng.key 42) t)));
      cases
        ~name:(fun (name, _) -> "categorical refuses " ^ name)
        "categorical"
        [
          ( "float8 logits",
            fun () ->
              ignore
                (Rng.categorical (Rng.key 0) (Nx.zeros Nx.float8_e4m3 [| 3 |]))
          );
          ( "an axis past the last",
            fun () ->
              ignore
                (Rng.categorical (Rng.key 0) ~axis:1
                   (Nx.zeros Nx.float32 [| 3 |])) );
          ( "an axis with no category",
            fun () ->
              ignore
                (Rng.categorical (Rng.key 0) (Nx.zeros Nx.float32 [| 3; 0 |]))
          );
          ( "an axis below the first",
            fun () ->
              ignore
                (Rng.categorical (Rng.key 0) ~axis:(-2)
                   (Nx.zeros Nx.float32 [| 3 |])) );
          ( "float8 logits, keyless",
            fun () -> ignore (Nx.categorical (Nx.zeros Nx.float8_e5m2 [| 3 |]))
          );
          ( "an axis out of bounds, keyless",
            fun () ->
              ignore (Nx.categorical ~axis:2 (Nx.zeros Nx.float32 [| 3; 2 |]))
          );
        ]
        (fun (_, f) -> raises_invalid_arg f);
    ]

(* The rows of [t]'s first axis, each element as its bits so that ordering the
   rows tells [-0.] from [0.]. *)
let rows_of t =
  let r = Ref.map Int64.bits_of_float (Ref.of_nx t) in
  if Ref.ndim r = 0 then [ r.data ]
  else
    let n = r.shape.(0) in
    let w = if n = 0 then 0 else Array.length r.data / n in
    List.init n (fun i -> Array.sub r.data (i * w) w)

let permutations =
  group "permutation and shuffle"
    [
      prop "permutation n is a permutation of [0, n)"
        (Gen.pair key
           (Gen.frequency
              [ (4, Gen.int_range 1 20); (1, Gen.int_range 1 5000) ]))
        ~examples:[ (Rng.key 0, 1) ]
        (fun (k, n) ->
          let p = Array.map Int32.to_int (Nx.to_array (Rng.permutation k n)) in
          Array.sort Int.compare p;
          equal (array int) (Array.init n Fun.id) p);
      prop "shuffle permutes the rows of its first axis"
        (Gen.pair key
           (viewed ~pp:pp_float Nx.float64 (Gen.float_range (-9.) 9.)))
        (fun (k, t) ->
          let s = Rng.shuffle k t in
          equal (array int) (Nx.shape t) (Nx.shape s);
          cover "an empty first axis" (Nx.ndim t > 0 && Nx.dim 0 t = 0);
          equal (slist (array int64) compare) (rows_of t) (rows_of s));
      test "shuffle returns a scalar unchanged" (fun () ->
          let t = Nx.scalar Nx.float32 4. in
          equal exactly (drawn t) (drawn (Rng.shuffle (Rng.key 0) t)));
      cases
        ~name:(fun (name, _) -> name ^ " refuses n <= 0")
        "permutation"
        [
          ("permutation", fun n -> ignore (Rng.permutation (Rng.key 0) n));
          ("keyless permutation", fun n -> ignore (Nx.permutation n));
        ]
        (fun (_, f) ->
          raises_invalid_arg (fun () -> f 0);
          raises_invalid_arg (fun () -> f (-1)));
    ]

(* The scope. A program is a list of keyless draws, each with its keyed twin:
   the keyed sampler of the same name, or the key itself for [next_key]. *)

let draws =
  let p = Nx.create Nx.float32 [| 4 |] [| 0.; 0.3; 0.7; 1. |] in
  let logits = Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; -1.; 0.5; 0. |] in
  let rows = Nx.reshape [| 3; 2 |] (Nx.arange Nx.float64 0 6 1) in
  let lower = Nx.scalar Nx.float32 (-1.) and upper = Nx.scalar Nx.float32 2. in
  let key_ref (k : Rng.t) = drawn (k :> Nx.int32_t) in
  [
    ( "rand float32 [|3|]",
      (fun () -> drawn (Nx.rand Nx.float32 [| 3 |])),
      fun k -> drawn (Rng.uniform k Nx.float32 [| 3 |]) );
    ( "rand float64 [|2|]",
      (fun () -> drawn (Nx.rand Nx.float64 [| 2 |])),
      fun k -> drawn (Rng.uniform k Nx.float64 [| 2 |]) );
    ( "randn float32 [|3|]",
      (fun () -> drawn (Nx.randn Nx.float32 [| 3 |])),
      fun k -> drawn (Rng.normal k Nx.float32 [| 3 |]) );
    ( "randn float16 [|2|]",
      (fun () -> drawn (Nx.randn Nx.float16 [| 2 |])),
      fun k -> drawn (Rng.normal k Nx.float16 [| 2 |]) );
    ( "randint ~low:(-4) ~high:9 [|5|]",
      (fun () -> drawn (Nx.randint ~low:(-4) ~high:9 [| 5 |])),
      fun k -> drawn (Rng.randint k ~low:(-4) ~high:9 [| 5 |]) );
    ( "bernoulli p",
      (fun () -> drawn (Nx.bernoulli p)),
      fun k -> drawn (Rng.bernoulli k p) );
    ( "truncated_normal (-1) 2",
      (fun () -> drawn (Nx.truncated_normal lower upper)),
      fun k -> drawn (Rng.truncated_normal k lower upper) );
    ( "categorical ~axis:0 logits",
      (fun () -> drawn (Nx.categorical ~axis:0 logits)),
      fun k -> drawn (Rng.categorical k ~axis:0 logits) );
    ( "permutation 6",
      (fun () -> drawn (Nx.permutation 6)),
      fun k -> drawn (Rng.permutation k 6) );
    ( "shuffle rows",
      (fun () -> drawn (Nx.shuffle rows)),
      fun k -> drawn (Rng.shuffle k rows) );
    ("next_key ()", (fun () -> key_ref (Rng.next_key ())), key_ref);
  ]

let program =
  Gen.list ~size:(Gen.int_range 0 6)
    (Gen.of_list
       ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
       draws)

let keyless p = List.map (fun (_, draw, _) -> draw ()) p
let twins p = List.map (fun (_, _, twin) -> twin (Rng.next_key ())) p
let values = list exactly

let scopes =
  group "scope"
    [
      prop "with_key replays a sequence of draws" (Gen.pair key program)
        (fun (k, p) ->
          equal values
            (Rng.with_key k (fun () -> keyless p))
            (Rng.with_key k (fun () -> keyless p)));
      prop "a keyless sampler is its keyed twin on next_key ()"
        (Gen.pair key program) (fun (k, p) ->
          equal values
            (Rng.with_key k (fun () -> twins p))
            (Rng.with_key k (fun () -> keyless p)));
      prop "next_key never repeats in a scope"
        (Gen.pair key (Gen.int_range 1 40))
        (fun (k, n) ->
          let drawn =
            Rng.with_key k (fun () ->
                List.init n (fun _ -> words (Rng.next_key ())))
          in
          equal int n (List.length (List.sort_uniq compare drawn)));
      test "next_key never repeats outside a scope" (fun () ->
          let drawn = List.init 20 (fun _ -> words (Rng.next_key ())) in
          equal int 20 (List.length (List.sort_uniq compare drawn)));
      prop
        "an inner scope replaces the outer one for its duration and leaves the \
         outer's sequence where it was"
        Gen.(pair (pair key key) (triple program program program))
        (fun ((k1, k2), (before, inner, after)) ->
          let outer, inside =
            Rng.with_key k1 (fun () ->
                let a = keyless before in
                let b = Rng.with_key k2 (fun () -> keyless inner) in
                (a @ keyless after, b))
          in
          equal ~msg:"inner" values
            (Rng.with_key k2 (fun () -> keyless inner))
            inside;
          equal ~msg:"outer" values
            (Rng.with_key k1 (fun () -> keyless (before @ after)))
            outer);
      test "a domain spawned inside a scope draws outside it" (fun () ->
          let spawned () =
            Rng.with_key (Rng.key 7) (fun () ->
                Domain.join (Domain.spawn (fun () -> words (Rng.next_key ()))))
          in
          not_equal (array int32) (spawned ()) (spawned ()));
    ]

(* Distributions, at fixed keys *)

let mean xs = Array.fold_left ( +. ) 0. xs /. float_of_int (Array.length xs)

let central p xs =
  let m = mean xs in
  Array.fold_left (fun acc x -> acc +. ((x -. m) ** p)) 0. xs
  /. float_of_int (Array.length xs)

let variance = central 2.

(* [k] standard errors of an estimate from [n] draws whose per-draw variance is
   [v]. *)
let se ?(k = 5.) ~n v = float (k *. Float.sqrt (v /. float_of_int n))

(* Mean and variance within five standard errors, given the distribution's
   variance [var] and fourth central moment [m4]. *)
let moments ~msg ~mean:m ~var ~m4 xs =
  let n = Array.length xs in
  equal ~msg:(msg ^ " mean") (se ~n var) m (mean xs);
  equal ~msg:(msg ^ " variance") (se ~n (m4 -. (var *. var))) var (variance xs)

let broadcast dtype n v = Nx.broadcast_to [| n |] (Nx.scalar dtype v)
let column i t = Nx.to_array (Nx.slice [ A; I i ] t)

let uniform_distribution =
  group "uniform distribution"
    [
      test "uniform reaches both ends of its grid at narrow dtypes" (fun () ->
          List.iter
            (fun (F f, n) ->
              let v = floats (Rng.uniform (Rng.key 4242) f.dtype [| n |]) in
              let seen = List.sort_uniq Float.compare (Array.to_list v) in
              equal ~msg:f.name int (1 lsl f.bits) (List.length seen))
            (List.filter_map
               (fun (F f as d) ->
                 List.assoc_opt f.name
                   [
                     ("float8_e5m2", 500);
                     ("float8_e4m3", 1_000);
                     ("bfloat16", 10_000);
                     ("float16", 50_000);
                   ]
                 |> Option.map (fun n -> (d, n)))
               floatings));
      test "uniform approaches both ends at float32 and float64" (fun () ->
          List.iter
            (fun (name, v) ->
              let hi = Array.fold_left Float.max 0. v
              and lo = Array.fold_left Float.min 1. v in
              greater ~msg:(name ^ " top") float_exact ~than:(1. -. 1e-3) hi;
              less ~msg:(name ^ " bottom") float_exact ~than:1e-3 lo)
            [
              ( "float32",
                floats (Rng.uniform (Rng.key 1) Nx.float32 [| 50_000 |]) );
              ( "float64",
                floats (Rng.uniform (Rng.key 1) Nx.float64 [| 50_000 |]) );
            ]);
      test "uniform at float64 carries more bits than float32's grid" (fun () ->
          let v = floats (Rng.uniform (Rng.key 3) Nx.float64 [| 50_000 |]) in
          equal int 0
            (Array.fold_left
               (fun c x ->
                 if Float.is_integer (Float.ldexp x 24) then c + 1 else c)
               0 v));
      test "uniform has mean 1/2 and variance 1/12" (fun () ->
          List.iter
            (fun (name, v) ->
              moments ~msg:name ~mean:0.5 ~var:(1. /. 12.) ~m4:(1. /. 80.) v)
            [
              ( "float32",
                floats (Rng.uniform (Rng.key 5) Nx.float32 [| 100_000 |]) );
              ( "float64",
                floats (Rng.uniform (Rng.key 5) Nx.float64 [| 100_000 |]) );
            ]);
    ]

let normal_distribution =
  group "normal distribution"
    [
      test "normal has mean 0 and variance 1" (fun () ->
          List.iter
            (fun (name, v) -> moments ~msg:name ~mean:0. ~var:1. ~m4:3. v)
            [
              ( "float32",
                floats (Rng.normal (Rng.key 5) Nx.float32 [| 100_000 |]) );
              ( "float64",
                floats (Rng.normal (Rng.key 5) Nx.float64 [| 100_000 |]) );
            ]);
      (* Box-Muller fills a draw two samples at a time: both halves of an odd
         length must be standard normal. *)
      test "both halves of an odd-length normal draw are standard normal"
        (fun () ->
          let v = floats (Rng.normal (Rng.key 5) Nx.float32 [| 20_001 |]) in
          moments ~msg:"first half" ~mean:0. ~var:1. ~m4:3.
            (Array.sub v 0 10_000);
          moments ~msg:"second half" ~mean:0. ~var:1. ~m4:3.
            (Array.sub v 10_000 10_001));
      test "gumbel has mean the Euler-Mascheroni constant and variance pi^2/6"
        (fun () ->
          let var = Float.pi *. Float.pi /. 6. in
          moments ~msg:"gumbel" ~mean:0.5772156649015329 ~var
            ~m4:(27. /. 5. *. var *. var)
            (floats (Rng.gumbel (Rng.key 1) Nx.float64 [| 100_000 |])));
      test "exponential has mean 1 and variance 1" (fun () ->
          moments ~msg:"exponential" ~mean:1. ~var:1. ~m4:9.
            (floats (Rng.exponential (Rng.key 2) Nx.float64 [| 100_000 |])));
    ]

let integer_distributions =
  group "integer distributions"
    [
      test "randint draws each value of [-5, 5) its share" (fun () ->
          let n = 100_000 in
          let counts = Array.make 10 0 in
          Array.iter
            (fun v ->
              let i = Int32.to_int v + 5 in
              counts.(i) <- counts.(i) + 1)
            (Nx.to_array (Rng.randint (Rng.key 91) ~low:(-5) ~high:5 [| n |]));
          Array.iteri
            (fun i c ->
              equal
                ~msg:(Printf.sprintf "value %d" (i - 5))
                (se ~n:1 (float_of_int n *. 0.1 *. 0.9))
                (float_of_int n *. 0.1)
                (float_of_int c))
            counts);
      test "randint over all of int32 draws a negative value half the time"
        (fun () ->
          let n = 10_000 in
          let v =
            Nx.to_array
              (Rng.randint (Rng.key 1) ~low:(-0x8000_0000) ~high:0x7FFF_FFFF
                 [| n |])
          in
          let negative =
            Array.fold_left (fun c x -> if x < 0l then c + 1 else c) 0 v
          in
          equal (se ~n 0.25) 0.5 (float_of_int negative /. float_of_int n));
      test "bernoulli is true with probability p, elementwise" (fun () ->
          let n = 20_000 in
          let p = Nx.create Nx.float32 [| 3 |] [| 0.1; 0.3; 0.9 |] in
          let t =
            Nx.cast Nx.float64
              (Rng.bernoulli (Rng.key 21) (Nx.broadcast_to [| n; 3 |] p))
          in
          List.iteri
            (fun i p ->
              equal
                ~msg:(Printf.sprintf "p = %g" p)
                (se ~n (p *. (1. -. p)))
                p
                (mean (column i t)))
            [ 0.1; 0.3; 0.9 ]);
      test "categorical draws each category with its softmax probability"
        (fun () ->
          let n = 20_000 in
          let logits = [| 0.; 1.; 2. |] in
          let z = Array.fold_left (fun s l -> s +. Float.exp l) 0. logits in
          let check name t =
            let counts = Array.make 3 0 in
            Array.iter
              (fun i ->
                let i = Int32.to_int i in
                counts.(i) <- counts.(i) + 1)
              (Nx.to_array t);
            Array.iteri
              (fun i l ->
                let p = Float.exp l /. z in
                equal
                  ~msg:(Printf.sprintf "%s category %d" name i)
                  (se ~n (p *. (1. -. p)))
                  p
                  (float_of_int counts.(i) /. float_of_int n))
              logits
          in
          let l = Nx.create Nx.float32 [| 3 |] logits in
          check "along the last axis"
            (Rng.categorical (Rng.key 123) (Nx.broadcast_to [| n; 3 |] l));
          check "along axis 0"
            (Rng.categorical (Rng.key 123) ~axis:0
               (Nx.broadcast_to [| 3; n |] (Nx.reshape [| 3; 1 |] l))));
      test "permutation puts each element at each position evenly" (fun () ->
          let n = 5 and trials = 4_000 in
          let counts = Array.make_matrix n n 0 in
          for t = 0 to trials - 1 do
            Array.iteri
              (fun pos v ->
                let v = Int32.to_int v in
                counts.(pos).(v) <- counts.(pos).(v) + 1)
              (Nx.to_array (Rng.permutation (Rng.key t) n))
          done;
          let p = 1. /. float_of_int n in
          Array.iteri
            (fun pos row ->
              Array.iteri
                (fun v c ->
                  equal
                    ~msg:(Printf.sprintf "element %d at position %d" v pos)
                    (se ~n:1 (float_of_int trials *. p *. (1. -. p)))
                    (float_of_int trials *. p)
                    (float_of_int c))
                row)
            counts);
    ]

(* Moments of the standard normal conditioned on [a, b], from the density [phi]
   and the upper tail [q], which keeps its digits where [a] and [b] are far
   out. *)
let truncated_moments a b =
  let phi x = Float.exp (-0.5 *. x *. x) /. Float.sqrt (2. *. Float.pi) in
  let q x = 0.5 *. Float.erfc (x /. Float.sqrt 2.) in
  let x_phi x = if Float.is_finite x then x *. phi x else 0. in
  let z = q a -. q b in
  let m = (phi a -. phi b) /. z in
  (m, 1. +. ((x_phi a -. x_phi b) /. z) -. (m *. m))

let truncated_distribution =
  let check (type b) ~msg (dtype : (float, b) Nx.dtype) lower upper =
    let n = 100_000 in
    let v =
      floats
        (Rng.truncated_normal (Rng.key 4242) (broadcast dtype n lower)
           (broadcast dtype n upper))
    in
    let m, var =
      truncated_moments (Float.min lower upper) (Float.max lower upper)
    in
    (* The variance's standard error needs the fourth central moment, which
       ranges from the normal's to the exponential's over these intervals: it is
       read from the draw. *)
    equal ~msg:(msg ^ " mean") (se ~n var) m (mean v);
    equal ~msg:(msg ^ " variance")
      (se ~n (central 4. v -. (variance v ** 2.)))
      var (variance v)
  in
  group "truncated normal distribution"
    [
      test "truncated_normal has the conditional moments" (fun () ->
          check ~msg:"[-0.75, 1.25] at float64" Nx.float64 (-0.75) 1.25;
          check ~msg:"[-0.75, 1.25] at float32" Nx.float32 (-0.75) 1.25;
          check ~msg:"bounds reversed" Nx.float64 1.25 (-0.75);
          check ~msg:"[-inf, 0]" Nx.float64 neg_infinity 0.;
          check ~msg:"[-inf, inf] at float32" Nx.float32 neg_infinity infinity);
      test "truncated_normal has the conditional moments in the tails"
        (fun () ->
          check ~msg:"[4, 5] at float32" Nx.float32 4. 5.;
          check ~msg:"[5, 6] at float64" Nx.float64 5. 6.;
          check ~msg:"[7, 8] at float64" Nx.float64 7. 8.;
          check ~msg:"[-8, -7] at float64" Nx.float64 (-8.) (-7.));
      test "truncated_normal spreads over a narrow interval" (fun () ->
          check ~msg:"[3, 3.01] at float64" Nx.float64 3. 3.01);
    ]

let gamma_distributions =
  group "gamma family distributions"
    [
      (* Gamma(a, 1) has mean a, variance a, fourth central moment 3a^2 + 6a and
         skewness 2 / sqrt a; the skewness is what catches an acceptance test
         that lets the tail through. Both sides of a = 1 are drawn, since below
         it the draw is shifted from a + 1. *)
      test "gamma has mean a, variance a and skewness 2 / sqrt a" (fun () ->
          List.iter
            (fun a ->
              let v =
                floats (Rng.gamma (Rng.key 31) (broadcast Nx.float64 100_000 a))
              in
              let msg = Printf.sprintf "gamma(%g)" a in
              moments ~msg ~mean:a ~var:a ~m4:((3. *. a *. a) +. (6. *. a)) v;
              equal ~msg:(msg ^ " skewness") (float 0.2)
                (2. /. Float.sqrt a)
                (central 3. v /. (variance v ** 1.5)))
            [ 0.4; 1.; 20. ]);
      test "gamma draws each concentration of a tensor at its own mean"
        (fun () ->
          let n = 50_000 in
          let t =
            Rng.gamma (Rng.key 33)
              (Nx.broadcast_to [| n; 2 |]
                 (Nx.create Nx.float64 [| 2 |] [| 0.5; 5. |]))
          in
          List.iteri
            (fun i a ->
              equal
                ~msg:(Printf.sprintf "concentration %g" a)
                (se ~n a) a
                (mean (column i t)))
            [ 0.5; 5. ]);
      (* Beta(a, b) has mean a / (a + b) and variance ab / ((a + b)^2 (a + b +
         1)). At concentrations of 0.005 most float32 gammas underflow to zero,
         and the draw must still be the ratio they stand for. *)
      test "beta has mean a / (a + b) and variance ab / ((a+b)^2 (a+b+1))"
        (fun () ->
          let check (type b) (dtype : (float, b) Nx.dtype) a b =
            let n = 20_000 in
            let v =
              floats
                (Rng.beta (Rng.key 5) (broadcast dtype n a)
                   (broadcast dtype n b))
            in
            let s = a +. b in
            let var = a *. b /. (s *. s *. (s +. 1.)) in
            let msg = Printf.sprintf "beta(%g, %g)" a b in
            equal ~msg:(msg ^ " mean") (se ~n var) (a /. s) (mean v);
            equal ~msg:(msg ^ " variance") (float 0.005) var (variance v)
          in
          check Nx.float64 2. 5.;
          check Nx.float64 0.5 0.5;
          check Nx.float64 8. 3.;
          check Nx.float32 0.005 0.005;
          check Nx.float32 0.02 0.05);
      test "dirichlet gives each component its share of the mass" (fun () ->
          let n = 50_000 in
          let c = [| 1.; 2.; 7. |] in
          let t =
            Rng.dirichlet (Rng.key 13)
              (Nx.broadcast_to [| n; 3 |] (Nx.create Nx.float64 [| 3 |] c))
          in
          let s = Array.fold_left ( +. ) 0. c in
          Array.iteri
            (fun i ci ->
              let p = ci /. s in
              equal
                ~msg:(Printf.sprintf "component %d" i)
                (se ~n (p *. (1. -. p) /. (s +. 1.)))
                p
                (mean (column i t)))
            c);
    ]

(* Poisson below 10 is exact, so its counts are held to the pmf; mean and
   variance both equal the rate, which a rejection sampler that stops short
   would break in the tail. *)
let poisson_distribution =
  let check (type b) (dtype : (float, b) Nx.dtype) ~n ~pmf rate =
    let v = floats (Rng.poisson (Rng.key 77) (broadcast dtype n rate)) in
    let msg =
      Printf.sprintf "poisson(%g) at %s" rate (Nx_dtype.to_string dtype)
    in
    moments ~msg ~mean:rate ~var:rate ~m4:(rate +. (3. *. rate *. rate)) v;
    if pmf then (
      let top = int_of_float (Float.ceil (rate +. (4. *. Float.sqrt rate))) in
      let counts = Array.make (top + 1) 0 in
      Array.iter
        (fun x ->
          let x = int_of_float x in
          if x <= top then counts.(x) <- counts.(x) + 1)
        v;
      let p = ref (Float.exp (-.rate)) in
      for c = 0 to top do
        let expected = !p *. float_of_int n in
        if expected > 50. then
          equal
            ~msg:(Printf.sprintf "%s count %d" msg c)
            (se ~n:1 expected) expected
            (float_of_int counts.(c));
        p := !p *. rate /. float_of_int (c + 1)
      done)
  in
  group "poisson distribution"
    [
      test "poisson counts follow the pmf across both regimes" (fun () ->
          List.iter (check Nx.float64 ~n:20_000 ~pmf:true) [ 0.7; 4.; 12.; 30. ];
          List.iter (check Nx.float32 ~n:20_000 ~pmf:true) [ 4.; 30. ]);
      test "poisson has mean and variance the rate at large rates" (fun () ->
          check Nx.float32 ~n:5_000 ~pmf:false 1e5;
          check Nx.float64 ~n:5_000 ~pmf:false 1e7);
      test "poisson draws each rate of a tensor at its own mean" (fun () ->
          let n = 20_000 in
          let rates = [| 0.5; 5.; 50.; 500. |] in
          let t =
            Rng.poisson (Rng.key 78)
              (Nx.broadcast_to [| n; 4 |] (Nx.create Nx.float32 [| 4 |] rates))
          in
          Array.iteri
            (fun i r ->
              equal
                ~msg:(Printf.sprintf "rate %g" r)
                (se ~n r) r
                (mean (Array.map Int32.to_float (column i t))))
            rates);
    ]

let () =
  exit
    (run "nx random"
       [
         key_tests;
         splits;
         fold_ins;
         threefry;
         purity;
         uniforms;
         continuous;
         randints;
         bernoullis;
         truncated_normals;
         gammas;
         poissons;
         categoricals;
         permutations;
         scopes;
         uniform_distribution;
         normal_distribution;
         integer_distributions;
         truncated_distribution;
         gamma_distributions;
         poisson_distribution;
       ])
