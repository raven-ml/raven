(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Random number generation. Keys, their derivations, the scope and each
   sampler's support are laws over drawn keys and parameters. Each distribution
   is a row of one table, whose draws one statistical check holds to the row's
   law. *)

open Windtrap
open Nx_test
module Rng = Nx.Rng

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let words (k : Rng.t) = Nx.to_array (k :> Nx.int32_t)
let keys = Testable.contramap words (array int32)
let of_words w = Rng.of_tensor (Nx.create Nx.int32 [| 2 |] w)
let k0 = Rng.key 0

(* A draw of any dtype as its values in float64, NaN equal to NaN. *)
let drawn t = Ref.of_nx (Nx.cast Nx.float64 t)
let floats t = (drawn t).data
let exactly = Ref.witness (close ~rel:0. ())
let chosen l = Gen.of_list ~pp:pp_float l

(* The significand width of a float dtype, its leading bit included. *)
let significand (type b) (dtype : (float, b) Nx.dtype) =
  match dtype with
  | Float8_e5m2 -> 3
  | Float8_e4m3 -> 4
  | BFloat16 -> 8
  | Float16 -> 11
  | Float32 -> 24
  | Float64 -> 53

(* Rows of [width] elements. *)
let rows width data =
  List.init
    (Array.length data / width)
    (fun i -> Array.sub data (i * width) width)

(* Generators *)

let word =
  Gen.frequency
    [
      (4, Gen.map Int32.of_int (Gen.int_range (-3) 3));
      (4, Gen.int32);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf -> Format.fprintf ppf "%ldl")
          [ Int32.min_int; -1l; Int32.max_int ] );
    ]

let key = Gen.map (fun (a, b) -> of_words [| a; b |]) (Gen.pair word word)

(* Words of shape [[...; 2]]: a key, a batch of keys, or counters. *)
let word_tensor =
  let open Gen in
  let* lead = array ~size:(int_range 0 2) (int_range 0 3) in
  let s = Array.append lead [| 2 |] in
  let+ w = array ~size:(constant (Ref.numel s)) word in
  Nx.create Nx.int32 s w

(* A seed or a counter: small values, the whole [int] range, its extremes and
   the edges of the two 32-bit words it is packed into. *)
let integer =
  Gen.frequency
    [
      (4, Gen.int_range (-3) 3);
      (2, Gen.int);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ min_int; max_int; 0xFFFF_FFFF; 0x1_0000_0000; -0x1_0000_0000 ] );
    ]

let shape =
  Gen.with_pp pp_shape (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 5))

type floating = F : (float, 'b) Nx.dtype -> floating

let floatings =
  Nx.[ F float8_e5m2; F float8_e4m3; F bfloat16; F float16; F float32 ]
  @ [ F Nx.float64 ]

let pp_floating ppf (F d) = Nx_dtype.pp ppf d
let floating = Gen.of_list ~pp:pp_floating floatings

(* A distribution's parameter: a tensor of a float dtype, in any layout. *)
type param = P : (float, 'b) Nx.t -> param

let param ?(dtypes = floatings) ?shape value =
  Gen.bind (Gen.of_list dtypes) (fun (F d) ->
      Gen.map
        (fun (_, t) -> P t)
        (Gen.pair
           (Gen.constant ~pp:pp_floating (F d))
           (viewed ?shape ~pp:pp_float d value)))

(* Values at the edges of each domain and past them. *)
let outside = [ Float.nan; neg_infinity; -1.; -0.; 0.; infinity ]
let inside x = Float.is_finite x && x > 0.

let probability =
  Gen.frequency [ (4, Gen.float_range 0. 1.); (2, chosen (1.5 :: outside)) ]

let bound =
  Gen.frequency
    [
      (4, Gen.float_range (-3.) 3.);
      ( 2,
        chosen [ Float.nan; neg_infinity; infinity; -9.; -6.; -0.; 5.5; 7.; 9. ]
      );
    ]

(* Positive over many orders of magnitude, where most float32 gammas underflow,
   and past the domain. *)
let concentration =
  Gen.frequency
    [
      (6, Gen.map (fun e -> 10. ** e) (Gen.float_range (-3.) 2.5));
      (1, chosen outside);
    ]

let rate =
  Gen.frequency
    [
      (3, Gen.float_range 0. 60.);
      (2, chosen ([ 1e-3; 9.99; 10.; 10.01; 1e3; 9e4 ] @ outside));
    ]

(* Trial counts: small, both regimes' worth, the int32 extremes and past
   them. *)
let trials =
  Gen.frequency
    [
      (4, Gen.int_range 0 60);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 1; 1000; 1_000_000; 0x7FFF_FFFF; -1; -0x8000_0000 ] );
    ]

(* Concentrations from zero over many orders of magnitude, and past the
   domain. *)
let circular =
  Gen.frequency
    [
      (6, Gen.map (fun e -> 10. ** e) (Gen.float_range (-3.) 6.));
      (1, chosen [ 0.; 1.; 1e30; Float.max_float ]);
      (1, chosen outside);
    ]

let logit =
  Gen.frequency
    [
      (4, Gen.float_range (-5.) 5.);
      (1, chosen [ neg_infinity; 0.; 30. ]);
      (1, chosen [ Float.nan; infinity ]);
    ]

(* The domains nx.mli states, NaN outside every one. *)
let in_unit p = p >= 0. && p <= 1.
let not_nan x = not (Float.is_nan x)
let in_rates r = Float.is_finite r && r >= 0.
let log_probability x = x < infinity

(* Every function of this module that takes a key, at fixed arguments. *)

let p3 = Nx.create Nx.float32 [| 3 |] [| 0.2; 0.5; 0.9 |]
let key_words (k : Rng.t) = drawn (k :> Nx.int32_t)

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
    ("poisson", fun k -> drawn (Rng.poisson k (Nx.mul_s p3 40.)));
    ( "binomial",
      fun k ->
        drawn (Rng.binomial k (Nx.create Nx.int32 [| 3 |] [| 5l; 90l; 0l |]) p3)
    );
    ("von_mises", fun k -> drawn (Rng.von_mises k (Nx.mul_s p3 10.)));
    ("categorical", fun k -> drawn (Rng.categorical k (Nx.tile [| 4; 1 |] p3)));
    ("permutation", fun k -> drawn (Rng.permutation k 5));
    ("shuffle", fun k -> drawn (Rng.shuffle k (Nx.arange Nx.float64 0 5 1)));
    ("split", fun k -> key_words (Rng.split k).(1));
    ("split_batch", fun k -> key_words (Rng.split_batch ~n:3 k));
    ("fold_in", fun k -> key_words (Rng.fold_in k 7));
    ( "fold_in_tensor",
      fun k -> key_words (Rng.fold_in_tensor k Nx.(scalar int32 7l)) );
  ]

(* The same two words, held with a negative stride, and with a stride of two
   past an offset. *)
let held_key =
  let layouts =
    [
      ( "reversed",
        fun w -> Nx.flip (Nx.create Nx.int32 [| 2 |] [| w.(1); w.(0) |]) );
      ( "as a column",
        fun w ->
          Nx.slice [ A; I 1 ]
            (Nx.create Nx.int32 [| 2; 2 |] [| 5l; w.(0); 6l; w.(1) |]) );
    ]
  in
  Gen.pair (Gen.pair word word)
    (Gen.of_list
       ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
       layouts)

(* Keys *)

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
      prop "the coercion to words inverts of_tensor, on keys and batches"
        word_tensor
        (Law.round_trip (tensor int32) (tensor int32)
           (fun t -> (Rng.of_tensor t :> Nx.int32_t))
           Fun.id);
      test "ptree rebuilds a key and checks it as of_tensor does" (fun () ->
          equal keys k0 (Nx.Ptree.map Rng.ptree (fun _ t -> t) k0);
          raises_invalid_arg (fun () ->
              Nx.Ptree.map Rng.ptree (fun _ t -> Nx.sum t) k0));
      cases ~name:fst "a batch of keys is refused by" keyed (fun (_, draw) ->
          raises_invalid_arg (fun () -> draw (Rng.split_batch ~n:4 k0)));
      group "the same values from the same key, held in any layout"
        (List.map
           (fun (name, draw) ->
             prop name held_key (fun ((a, b), (_, view)) ->
                 equal exactly
                   (draw (of_words [| a; b |]))
                   (draw (Rng.of_tensor (view [| a; b |])))))
           keyed);
      prop
        "split makes n subkeys, two by default, that differ from each other \
         and from their parent"
        (Gen.pair key (Gen.option (Gen.int_range 1 16)))
        ~examples:[ (k0, None) ]
        (fun (k, n) ->
          let all = List.map words (k :: Array.to_list (Rng.split ?n k)) in
          let n = Option.value n ~default:2 in
          equal int (n + 1) (List.length (List.sort_uniq compare all)));
      prop "split_batch holds split's subkeys as its rows"
        (Gen.pair key (Gen.int_range 1 8))
        (fun (k, n) ->
          let row (s : Rng.t) = (s :> Nx.int32_t) in
          equal (tensor int32)
            (Nx.stack (List.map row (Array.to_list (Rng.split ~n k))))
            (row (Rng.split_batch ~n k)));
      prop "fold_in gives distinct counters distinct keys"
        (Gen.triple key integer integer)
        ~examples:[ (k0, 1, 0x1_0000_0001); (k0, -1, 0xFFFF_FFFF) ]
        (fun (k, a, b) ->
          assume (a <> b);
          not_equal keys (Rng.fold_in k a) (Rng.fold_in k b));
      prop "fold_in_tensor agrees with fold_in on every int32, negatives too"
        (Gen.pair key word)
        ~examples:[ (k0, -1l); (k0, Int32.min_int) ]
        (fun (k, i) ->
          equal keys
            (Rng.fold_in k (Int32.to_int i))
            (Rng.fold_in_tensor k (Nx.scalar Nx.int32 i)));
    ]

(* Threefry-2x32-20 is the backend contract's block function, which nx.mli names
   only as Threefry, [Nx.Op.Threefry]. The known answers are Random123's. Every
   row of the distribution table draws from many blocks at once. *)
let threefry =
  test "threefry gives the known answers of Threefry-2x32-20" (fun () ->
      let answer k c =
        let block w = Nx.create Nx.int32 [| 2 |] w in
        Nx.to_array (Nx.Op.eval (Threefry (block k, block c)))
      in
      equal (array int32)
        [| 1797259609l; -1715843330l |]
        (answer [| 0l; 0l |] [| 0l; 0l |]);
      equal (array int32)
        [| 481924860l; -1157616665l |]
        (answer [| -1l; -1l |] [| -1l; -1l |]);
      equal (array int32)
        [| -997049700l; 1212020640l |]
        (answer [| 0x13198a2el; 0x03707344l |] [| 0x243f6a88l; 0x85a308d3l |]))

let counters =
  test "bits is Threefry of the key over the counters (2i, 2i + 1)" (fun () ->
      let k = Rng.key 42 and n = 37 in
      let counters =
        Nx.create Nx.int32 [| n; 2 |] (Array.init (2 * n) Int32.of_int)
      in
      let keys =
        Nx.broadcast_to [| n; 2 |] (Nx.reshape [| 1; 2 |] (k :> Nx.int32_t))
      in
      equal (array int32)
        (Nx.to_array (Nx.Op.eval (Threefry (keys, counters))))
        (Nx.to_array (Rng.bits k [| 2 * n |])))

(* Supports. Every sampler draws, at every dtype it takes, values of its support
   in the requested shape. A parameter with an element outside its domain
   raises, and the laws hold where the parameters are inside it. A parameter in
   any layout gives the draw its copy gives, which also compares two calls. *)

type sized = {
  draw : 'b. Rng.t -> (float, 'b) Nx.dtype -> int array -> (float, 'b) Nx.t;
}

let sized =
  let on_grid p v =
    v >= 0.
    && v <= 1. -. Float.ldexp 1. (-p)
    && Float.is_integer (Float.ldexp v p)
  in
  let finite _ = Float.is_finite in
  [
    ( "uniform draws are multiples of 2^-p in [0, 1 - 2^-p], p the significand \
       width,",
      { draw = Rng.uniform },
      on_grid );
    ("normal draws are finite", { draw = Rng.normal }, finite);
    ("gumbel draws are finite", { draw = Rng.gumbel }, finite);
    ( "exponential draws are finite and non-negative",
      { draw = Rng.exponential },
      fun _ v -> Float.is_finite v && v >= 0. );
  ]

(* Samplers of one parameter tensor, with the parameter's domain and what a draw
   holds given its parameter; a second parameter is a scalar inside its
   domain. *)
type elementwise = { sample : 'b. Rng.t -> (float, 'b) Nx.t -> float Ref.t }

let elementwise =
  let s t v = Nx.scalar (Nx.dtype t) v in
  let within a b v = v >= Float.min a b && v <= Float.max a b in
  [
    ( "bernoulli is true where p is 1 and false where p is 0",
      probability,
      in_unit,
      { sample = (fun k p -> drawn (Rng.bernoulli k p)) },
      fun p b -> (b = 0. || b = 1.) && (p < 1. || b = 1.) && (p > 0. || b = 0.)
    );
    ( "truncated_normal lies between a lower bound and 1.5, in either order",
      bound,
      not_nan,
      { sample = (fun k t -> drawn (Rng.truncated_normal k t (s t 1.5))) },
      within 1.5 );
    ( "truncated_normal lies between -1.5 and an upper bound, in either order",
      bound,
      not_nan,
      { sample = (fun k t -> drawn (Rng.truncated_normal k (s t (-1.5)) t)) },
      within (-1.5) );
    ( "gamma draws are finite and non-negative",
      concentration,
      inside,
      { sample = (fun k c -> drawn (Rng.gamma k c)) },
      fun _ g -> Float.is_finite g && g >= 0. );
    ( "beta draws lie in [0, 1], over their first concentration",
      concentration,
      inside,
      { sample = (fun k a -> drawn (Rng.beta k a (s a 2.))) },
      fun _ v -> within 0. 1. v );
    ( "beta draws lie in [0, 1], over their second concentration",
      concentration,
      inside,
      { sample = (fun k b -> drawn (Rng.beta k (s b 2.) b)) },
      fun _ v -> within 0. 1. v );
    ( "poisson counts are non-negative, and zero where the rate is zero",
      rate,
      in_rates,
      { sample = (fun k r -> drawn (Rng.poisson k r)) },
      fun r c -> c >= 0. && (r > 0. || c = 0.) );
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

let components =
  Gen.with_pp pp_shape
    (Gen.map
       (fun (lead, c) -> Array.append lead [| c |])
       (Gen.pair
          (Gen.array ~size:(Gen.int_range 0 2) (Gen.int_range 0 3))
          (Gen.int_range 2 4)))

(* Logits of a float dtype that categorical takes, every one but float8, and an
   axis to draw along. *)
let logits_along =
  Gen.with_pp
    (fun ppf (P t, axis) ->
      Format.fprintf ppf "axis %d of %a %a" axis Nx_dtype.pp (Nx.dtype t)
        (Ref.pp pp_float) (drawn t))
    (let open Gen in
     let shape = array ~size:(int_range 1 3) (int_range 1 4) in
     let wide = List.filter (fun (F d) -> significand d > 4) floatings in
     let* (P t) = param ~dtypes:wide ~shape logit in
     let+ axis = int_range (-Nx.ndim t) (Nx.ndim t - 1) in
     (P t, axis))

(* A probability and trial counts of its shape, or one count for all of it. *)
let binomial_args =
  Gen.with_pp
    (fun ppf (P p, n) ->
      Format.fprintf ppf "n %a, p %a %a"
        (Ref.pp Format.pp_print_int)
        (Ref.map Int32.to_int (Ref.of_nx n))
        Nx_dtype.pp (Nx.dtype p) (Ref.pp pp_float) (drawn p))
    (let open Gen in
     let* (P p) = param probability in
     let* one = bool in
     let+ ns = array ~size:(constant (if one then 1 else Nx.numel p)) trials in
     let ns = Array.map Int32.of_int ns in
     ( P p,
       if one then Nx.scalar Nx.int32 ns.(0)
       else Nx.create Nx.int32 (Nx.shape p) ns ))

(* [pi] rounded up to [p] bits: the largest angle a draw rounded to a dtype of
   [p] bits can reach. *)
let pi_up p = Float.ldexp (Float.ceil (Float.ldexp Float.pi (p - 2))) (2 - p)

let same_draw sampler t =
  equal exactly (drawn (sampler (Nx.copy t))) (drawn (sampler t))

let supports =
  group "supports"
    (List.map
       (fun (claim, s, holds) ->
         prop (claim ^ " at every float dtype") (Gen.triple key floating shape)
           (fun (k, F d, shape) ->
             let r = drawn (s.draw k d shape) in
             equal (array int) shape r.shape;
             Array.iter
               (fun v ->
                 is_true ~msg:(Printf.sprintf "%h" v) (holds (significand d) v))
               r.data))
       sized
    @ List.map
        (fun (claim, value, domain, s, holds) ->
          prop
            (claim
           ^ ", in its parameter's shape; a parameter outside its domain raises"
            )
            (Gen.pair key (param value))
            (fun (k, P t) ->
              let valid = Array.for_all domain (floats t) in
              cover "a parameter outside the domain" (not valid);
              cover "a non-empty parameter inside the domain"
                (valid && Nx.numel t > 0);
              if not valid then raises_invalid_arg (fun () -> s.sample k t)
              else
                let r = s.sample k t in
                equal exactly (s.sample k (Nx.copy t)) r;
                equal (array int) (Nx.shape t) r.shape;
                Array.iteri
                  (fun i p ->
                    let v = r.data.(i) in
                    is_true ~msg:(Printf.sprintf "%h from %h" v p) (holds p v))
                  (floats t)))
        elementwise
    @ [
        prop "uniform at float32 is the low 24 bits of bits, scaled by 2^-24"
          (Gen.pair key shape) (fun (k, s) ->
            let b = Ref.of_nx (Rng.bits k s) in
            equal (array int) s b.shape;
            let low24 w =
              Int32.to_float (Int32.logand w 0xFF_FFFFl) *. 0x1p-24
            in
            equal exactly (Ref.map low24 b) (drawn (Rng.uniform k Nx.float32 s)));
        prop
          "binomial counts lie in [0, n], are 0 where p or n is 0 and n where \
           p is 1, in the broadcast shape; an argument outside its domain \
           raises" (Gen.pair key binomial_args) (fun (k, (P p, n)) ->
            let ps = floats p and ns = Nx.to_array n in
            let valid =
              Array.for_all in_unit ps && Array.for_all (fun n -> n >= 0l) ns
            in
            cover "an argument outside the domain" (not valid);
            cover "a non-empty draw inside the domain" (valid && Nx.numel p > 0);
            if not valid then raises_invalid_arg (fun () -> Rng.binomial k n p)
            else (
              equal exactly
                (drawn (Rng.binomial k (Nx.copy n) (Nx.copy p)))
                (drawn (Rng.binomial k n p));
              let c = Ref.of_nx (Rng.binomial k n p) in
              equal (array int) (Nx.shape p) c.shape;
              let n = Ref.broadcast_to c.shape (Ref.of_nx n) in
              cover "a mean of 10 or more"
                (Array.exists2
                   (fun n p -> Int32.to_float n *. Float.min p (1. -. p) >= 10.)
                   n.data ps);
              Array.iteri
                (fun i c ->
                  let n = n.data.(i) and p = ps.(i) in
                  let msg = Printf.sprintf "%ld of %ld at %h" c n p in
                  at_least ~msg int32 ~than:0l c;
                  at_most ~msg int32 ~than:n c;
                  if p = 0. || n = 0l then equal ~msg int32 0l c;
                  if p = 1. then equal ~msg int32 n c)
                c.data));
        prop
          "von_mises draws are angles in [-pi, pi] at their dtype, in their \
           concentration's shape; a concentration outside [0, inf) raises"
          (Gen.pair key (param circular))
          (fun (k, P t) ->
            let valid = Array.for_all in_rates (floats t) in
            cover "a concentration outside the domain" (not valid);
            cover "a zero concentration" (Array.mem 0. (floats t));
            if not valid then raises_invalid_arg (fun () -> Rng.von_mises k t)
            else (
              same_draw (Rng.von_mises k) t;
              let r = drawn (Rng.von_mises k t) in
              equal (array int) (Nx.shape t) r.shape;
              let pi = pi_up (significand (Nx.dtype t)) in
              Array.iter
                (satisfies ~claim:"an angle in [-pi, pi], rounded to the dtype"
                   float_exact (fun v -> Float.abs v <= pi))
                r.data));
        cases
          ~name:(fun (n, p, c) ->
            Printf.sprintf "binomial of %ld trials at p = %g is %ld" n p c)
          "exact"
          [
            (0l, 0.3, 0l);
            (7l, 0., 0l);
            (7l, 1., 7l);
            (1000l, 0., 0l);
            (1000l, 1., 1000l);
            (0x7FFF_FFFFl, 0., 0l);
            (0x7FFF_FFFFl, 1., 0x7FFF_FFFFl);
          ]
          (fun (n, p, c) ->
            let n = Nx.full Nx.int32 [| 50 |] n in
            List.iter
              (fun (F d) ->
                Array.iter (equal int32 c)
                  (Nx.to_array (Rng.binomial k0 n (Nx.full d [| 50 |] p))))
              [ F Nx.float32; F Nx.float64 ]);
        prop "randint draws lie in [low, high)"
          (Gen.quad key int32_bound int32_bound shape)
          ~examples:[ (k0, -0x8000_0000, 0x7FFF_FFFF, [| 64 |]) ]
          (fun (k, a, b, s) ->
            assume (a <> b);
            let low = Int.min a b and high = Int.max a b in
            cover "a range wider than 2^31" (high - low > 0x8000_0000);
            let d = drawn (Rng.randint k ~low ~high s) in
            equal (array int) s d.shape;
            Array.iter
              (fun v ->
                at_least float_exact ~than:(float_of_int low) v;
                less float_exact ~than:(float_of_int high) v)
              d.data);
        test "randint over [0, 2^25) draws even values, from 24 random bits"
          (fun () ->
            Array.iter
              (fun v -> equal int32 0l (Int32.logand v 1l))
              (Nx.to_array (Rng.randint k0 ~high:(1 lsl 25) [| 1000 |])));
        cases
          ~name:(fun (F d, a, b, v) ->
            Format.asprintf
              "truncated_normal at %a over [%g, %g], past erf's reach, is %g"
              Nx_dtype.pp d a b v)
          "collapse"
          [
            (F Nx.float32, 6., 7., 6.);
            (F Nx.float16, -7., -6., -6.);
            (F Nx.float64, 9., 10., 9.);
          ]
          (fun (F d, a, b, v) ->
            let bound x = Nx.full d [| 5 |] x in
            Array.iter (equal float_exact v)
              (floats (Rng.truncated_normal k0 (bound a) (bound b))));
        (* Each component is rounded once to the dtype. A layout can leave a
           single component, which is refused. *)
        prop "every row of a dirichlet draw is non-negative and sums to one"
          (Gen.pair key (param ~shape:components concentration))
          (fun (k, P c) ->
            let n = Nx.dim (-1) c in
            let concentrations = Array.for_all inside (floats c) in
            cover "a concentration outside the domain" (not concentrations);
            if n < 2 || not concentrations then
              raises_invalid_arg (fun () -> Rng.dirichlet k c)
            else (
              same_draw (Rng.dirichlet k) c;
              let d = drawn (Rng.dirichlet k c) in
              equal (array int) (Nx.shape c) d.shape;
              let tol =
                float_of_int n *. Float.ldexp 1. (-significand (Nx.dtype c))
              in
              List.iter
                (fun row ->
                  Array.iter (at_least float_exact ~than:0.) row;
                  equal (float tol) 1. (Array.fold_left ( +. ) 0. row))
                (rows n d.data)));
        prop
          "categorical gives one index per lane along axis, by default the \
           last, never on a -inf logit beside a finite one, and raises on a \
           NaN or infinite logit" (Gen.pair key logits_along)
          (fun (k, (P t, axis)) ->
            let n = Nx.dim axis t in
            let logits = Array.for_all log_probability (floats t) in
            cover "a logit outside the domain" (not logits);
            if n = 0 || not logits then
              raises_invalid_arg (fun () -> Rng.categorical k ~axis t)
            else (
              same_draw (Rng.categorical k ~axis) t;
              let lanes = Ref.moveaxis axis (-1) (drawn t) in
              let idx = drawn (Rng.categorical k ~axis t) in
              equal (array int)
                (Array.sub lanes.shape 0 (Nx.ndim t - 1))
                idx.shape;
              List.iteri
                (fun l lane ->
                  let i = int_of_float idx.data.(l) in
                  at_least int ~than:0 i;
                  less int ~than:n i;
                  let finite = Array.exists Float.is_finite lane in
                  cover "a lane with a -inf logit"
                    (finite && Array.mem neg_infinity lane);
                  if finite then is_true (Float.is_finite lane.(i)))
                (rows n lanes.data);
              if axis = -1 then equal exactly idx (drawn (Rng.categorical k t))));
        prop "permutation n is a permutation of [0, n)"
          (Gen.pair key
             (Gen.frequency
                [ (4, Gen.int_range 1 20); (1, Gen.int_range 1 5000) ]))
          ~examples:[ (k0, 1) ]
          (fun (k, n) ->
            cover "n = 1" (n = 1);
            let p =
              Array.map Int64.to_int (Nx.to_array (Rng.permutation k n))
            in
            Array.sort Int.compare p;
            equal (array int) (Array.init n Fun.id) p);
        (* Each element as its bits, so that ordering tells [-0.] from [0.]. *)
        prop
          "shuffle permutes the rows of its first axis, an empty one too, and \
           returns a scalar unchanged"
          (Gen.pair key
             (viewed ~pp:pp_float Nx.float64 (Gen.float_range (-9.) 9.)))
          ~examples:
            [
              (k0, Nx.zeros Nx.float64 [| 0; 3 |]); (k0, Nx.scalar Nx.float64 4.);
            ]
          (fun (k, t) ->
            let s = Rng.shuffle k t in
            same_draw (Rng.shuffle k) t;
            equal (array int) (Nx.shape t) (Nx.shape s);
            cover "an empty first axis" (Nx.ndim t > 0 && Nx.dim 0 t = 0);
            cover "a scalar" (Nx.ndim t = 0);
            let bits t = Array.map Int64.bits_of_float (Nx.to_array t) in
            let rows_of t =
              if Nx.ndim t = 0 then [ bits t ]
              else List.init (Nx.dim 0 t) (fun i -> bits (Nx.slice [ I i ] t))
            in
            equal (slist (array int64) compare) (rows_of t) (rows_of s));
      ])

(* The uniform grid is the finest the dtype holds. At a narrow dtype every
   multiple of 2^-p is drawn: a draw of n = 2^p ln (2^p 1e7) misses one of them
   with probability below 2^p exp (-n 2^-p) = 1e-7. At float32 and float64 a
   draw of 100 holds an odd multiple of 2^-p but with probability 2^-100. *)
let uniform_grid =
  cases
    ~name:(fun (F d) ->
      Format.asprintf "uniform at %a uses every bit of its grid" Nx_dtype.pp d)
    "grid" floatings
    (fun (F d) ->
      let p = significand d and points = Float.ldexp 1. (significand d) in
      if p <= 11 then
        let n = int_of_float (points *. Float.log (points *. 1e7)) in
        let v = floats (Rng.uniform (Rng.key 7) d [| n |]) in
        equal int (1 lsl p)
          (List.length (List.sort_uniq compare (Array.to_list v)))
      else
        let odd x = not (Float.is_integer (Float.ldexp x (p - 1))) in
        is_true
          (Array.exists odd (floats (Rng.uniform (Rng.key 7) d [| 100 |]))))

let errors =
  let z s = Nx.zeros Nx.float32 s and w s = Nx.zeros Nx.int32 s in
  let negative = [| 2; -1 |] in
  cases ~name:fst "errors"
    [
      ( "of_tensor refuses a last axis other than 2",
        [
          (fun () -> ignore (Rng.of_tensor (w [||])));
          (fun () -> ignore (Rng.of_tensor (w [| 2; 3 |])));
          (fun () -> ignore (Rng.of_tensor (w [| 0 |])));
        ] );
      ( "split and split_batch refuse n < 1",
        [
          (fun () -> ignore (Rng.split ~n:0 k0));
          (fun () -> ignore (Rng.split_batch ~n:(-1) k0));
        ] );
      ( "every sampler of a shape refuses a negative dimension",
        [
          (fun () -> ignore (Rng.bits k0 negative));
          (fun () -> ignore (Rng.uniform k0 Nx.float32 negative));
          (fun () -> ignore (Rng.normal k0 Nx.float64 negative));
          (fun () -> ignore (Rng.gumbel k0 Nx.float32 negative));
          (fun () -> ignore (Rng.exponential k0 Nx.float32 negative));
          (fun () -> ignore (Rng.randint k0 ~high:3 negative));
          (fun () -> ignore (Nx.rand Nx.float32 negative));
          (fun () -> ignore (Nx.randn Nx.float32 negative));
        ] );
      ( "randint refuses an empty range and a bound outside int32",
        [
          (fun () -> ignore (Rng.randint k0 ~low:3 ~high:3 [| 2 |]));
          (fun () -> ignore (Nx.randint ~low:4 ~high:3 [| 2 |]));
          (fun () -> ignore (Rng.randint k0 ~low:(-0x8000_0001) ~high:0 [||]));
          (fun () -> ignore (Nx.randint ~high:0x8000_0000 [| 2 |]));
        ] );
      ( "truncated_normal, beta and binomial refuse parameters that do not \
         broadcast",
        [
          (fun () -> ignore (Rng.truncated_normal k0 (z [| 3 |]) (z [| 4 |])));
          (fun () -> ignore (Rng.beta k0 (z [| 3 |]) (z [| 4 |])));
          (fun () -> ignore (Rng.binomial k0 (w [| 3 |]) (z [| 4 |])));
        ] );
      ( "dirichlet refuses fewer than two components",
        [
          (fun () -> ignore (Rng.dirichlet k0 (z [| 4; 1 |])));
          (fun () -> ignore (Rng.dirichlet k0 (z [||])));
        ] );
      ( "categorical refuses float8 logits and an axis out of bounds or empty",
        [
          (fun () -> ignore (Rng.categorical k0 Nx.(zeros float8_e4m3 [| 3 |])));
          (fun () -> ignore (Nx.categorical Nx.(zeros float8_e5m2 [| 3 |])));
          (fun () -> ignore (Nx.categorical ~axis:1 (z [| 3 |])));
          (fun () -> ignore (Rng.categorical k0 ~axis:(-2) (z [| 3 |])));
          (fun () -> ignore (Nx.categorical (z [| 3; 0 |])));
        ] );
      ( "permutation refuses n <= 0",
        [
          (fun () -> ignore (Rng.permutation k0 0));
          (fun () -> ignore (Nx.permutation (-1)));
        ] );
    ]
    (fun (_, refused) -> List.iter raises_invalid_arg refused)

(* A parameter outside its domain is refused with a message naming the sampler,
   the parameter, the index of its first element outside the domain, its value
   and the domain. A broadcast parameter is indexed in its broadcast shape. *)
let refusals =
  let v d xs = Nx.create d [| Array.length xs |] xs in
  let m d r c xs = Nx.create d [| r; c |] xs in
  let f32 = Nx.float32 and f64 = Nx.float64 in
  let two = Nx.scalar f32 2. in
  cases ~name:fst "refusals"
    [
      ( "Nx.Rng.gamma: concentration at [2] is -0.5, not in (0, inf)",
        fun () -> ignore (Rng.gamma k0 (v f32 [| 1.; 2.; -0.5 |])) );
      ( "Nx.Rng.gamma: concentration at [1] is 0, not in (0, inf)",
        fun () -> ignore (Rng.gamma k0 (v f64 [| 1.; 0.; Float.nan |])) );
      ( "Nx.Rng.beta: a at [0; 1] is 0, not in (0, inf)",
        fun () -> ignore (Rng.beta k0 (m f32 1 2 [| 1.; 0. |]) two) );
      ( "Nx.Rng.beta: b at [1] is nan, not in (0, inf)",
        fun () ->
          ignore (Rng.beta k0 (v f32 [| 1.; 1. |]) (v f32 [| 1.; Float.nan |]))
      );
      ( "Nx.Rng.dirichlet: concentration at [1; 1] is inf, not in (0, inf)",
        fun () ->
          ignore (Rng.dirichlet k0 (m f64 2 2 [| 1.; 1.; 1.; infinity |])) );
      ( "Nx.Rng.poisson: rate is -1, not in [0, inf)",
        fun () -> ignore (Rng.poisson k0 (Nx.scalar f32 (-1.))) );
      ( "Nx.Rng.poisson: rate at [0] is inf, not in [0, inf)",
        fun () -> ignore (Rng.poisson k0 (v f64 [| infinity |])) );
      ( "Nx.Rng.binomial: n at [1] is -1, not in [0, inf)",
        fun () ->
          ignore
            (Rng.binomial k0
               (Nx.create Nx.int32 [| 2 |] [| 3l; -1l |])
               (v f32 [| 0.5; 1.5 |])) );
      ( "Nx.Rng.binomial: p at [1] is nan, not in [0, 1]",
        fun () ->
          ignore
            (Rng.binomial k0 (Nx.scalar Nx.int32 4l)
               (v f64 [| 0.5; Float.nan |])) );
      ( "Nx.Rng.von_mises: concentration at [2] is inf, not in [0, inf)",
        fun () -> ignore (Rng.von_mises k0 (v f32 [| 0.; 1.; infinity |])) );
      ( "Nx.Rng.bernoulli: p at [0; 2] is 1.5, not in [0, 1]",
        fun () ->
          ignore
            (Rng.bernoulli k0
               (Nx.broadcast_to [| 3; 4 |] (v f32 [| 0.5; 0.5; 1.5; 0.5 |]))) );
      ( "Nx.Rng.bernoulli: p at [1; 0] is -0.1, not in [0, 1]",
        fun () ->
          ignore
            (Nx.bernoulli
               (Nx.broadcast_to [| 3; 4 |] (m f32 3 1 [| 0.; -0.1; 1. |]))) );
      ( "Nx.Rng.truncated_normal: upper at [1] is nan, not in [-inf, inf]",
        fun () ->
          ignore
            (Rng.truncated_normal k0
               (v f32 [| 0.; 0. |])
               (v f32 [| 1.; Float.nan |])) );
      ( "Nx.Rng.truncated_normal: lower is nan, not in [-inf, inf]",
        fun () ->
          ignore
            (Nx.truncated_normal (Nx.scalar f64 Float.nan) (v f64 [| 1.; 2. |]))
      );
      ( "Nx.Rng.categorical: logits at [0; 1] is inf, not in [-inf, inf)",
        fun () ->
          ignore (Rng.categorical k0 (m f32 2 2 [| 0.; infinity; 0.; 0. |])) );
      ( "Nx.Rng.categorical: logits at [1; 0] is nan, not in [-inf, inf)",
        fun () ->
          ignore
            (Nx.categorical ~axis:0
               (m f32 2 2 [| 0.; neg_infinity; Float.nan; 0. |])) );
    ]
    (fun (message, draw) -> raises (Invalid_argument message) draw)

(* The scope. A program is a list of keyless draws, each with its keyed twin:
   the keyed sampler of the same name, or the key itself for [next_key]. *)

let draws =
  let p = Nx.create Nx.float32 [| 4 |] [| 0.; 0.3; 0.7; 1. |] in
  let logits = Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; 2.; -1.; 0.5; 0. |] in
  let rows = Nx.reshape [| 3; 2 |] (Nx.arange Nx.float64 0 6 1) in
  let lower = Nx.scalar Nx.float16 (-1.) and upper = Nx.scalar Nx.float16 2. in
  [
    ( "rand float64 [|2|]",
      (fun () -> drawn (Nx.rand Nx.float64 [| 2 |])),
      fun k -> drawn (Rng.uniform k Nx.float64 [| 2 |]) );
    ( "randn float16 [|3|]",
      (fun () -> drawn (Nx.randn Nx.float16 [| 3 |])),
      fun k -> drawn (Rng.normal k Nx.float16 [| 3 |]) );
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
    ("next_key ()", (fun () -> key_words (Rng.next_key ())), key_words);
  ]

let program =
  Gen.list ~size:(Gen.int_range 0 6)
    (Gen.of_list
       ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
       draws)

let keyless p = List.map (fun (_, draw, _) -> draw ()) p
let values = list exactly

let scopes =
  let fresh n = List.init n (fun _ -> words (Rng.next_key ())) in
  let distinct l = List.length (List.sort_uniq compare l) in
  group "scope"
    [
      prop
        "a keyless sampler is its keyed twin on next_key (), in a scope that \
         replays its draws"
        (Gen.pair key program) (fun (k, p) ->
          equal values
            (Rng.with_key k (fun () ->
                 List.map (fun (_, _, twin) -> twin (Rng.next_key ())) p))
            (Rng.with_key k (fun () -> keyless p)));
      prop "next_key never repeats in a scope"
        (Gen.pair key (Gen.int_range 1 40))
        (fun (k, n) ->
          equal int n (distinct (Rng.with_key k (fun () -> fresh n))));
      test "next_key never repeats outside a scope" (fun () ->
          equal int 20 (distinct (fresh 20)));
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
      prop
        "a scope rooted at its first draw that draws nothing takes no key and \
         runs no root"
        Gen.(pair key (pair program program))
        (fun (k, (before, after)) ->
          let ran = ref false in
          let drawn =
            Rng.with_key k (fun () ->
                let a = keyless before in
                Rng.with_root
                  (fun () ->
                    ran := true;
                    Rng.next_key ())
                  ignore;
                a @ keyless after)
          in
          equal ~msg:"draws" values
            (Rng.with_key k (fun () -> keyless (before @ after)))
            drawn;
          equal ~msg:"root ran" bool false !ran);
      prop
        "a scope rooted at its first draw that draws is the scope rooted at \
         its root, taken once from the scope around at that draw"
        Gen.(pair key (pair (triple program program program) small_int))
        (fun (k, ((before, inner, after), data)) ->
          cover "the scope draws" (inner <> []);
          let runs = ref 0 in
          let rooted =
            Rng.with_key k (fun () ->
                let a = keyless before in
                let b =
                  Rng.with_root
                    (fun () ->
                      incr runs;
                      Rng.fold_in (Rng.next_key ()) data)
                    (fun () -> keyless inner)
                in
                (a @ keyless after, b))
          in
          let expected =
            Rng.with_key k (fun () ->
                let a = keyless before in
                let b =
                  if inner = [] then []
                  else
                    Rng.with_key
                      (Rng.fold_in (Rng.next_key ()) data)
                      (fun () -> keyless inner)
                in
                (a @ keyless after, b))
          in
          equal ~msg:"outer" values (fst expected) (fst rooted);
          equal ~msg:"inner" values (snd expected) (snd rooted);
          equal ~msg:"root runs" int (if inner = [] then 0 else 1) !runs);
      test
        "a root that raises raises at the first draw, inside the scope, and \
         runs again at the next" (fun () ->
          let runs = ref 0 and cleaned = ref false in
          let root () =
            incr runs;
            if !runs = 1 then failwith "root" else Rng.key 3
          in
          let drawn =
            Rng.with_root root (fun () ->
                raises (Failure "root") (fun () ->
                    Fun.protect
                      ~finally:(fun () -> cleaned := true)
                      (fun () -> Rng.next_key ()));
                words (Rng.next_key ()))
          in
          equal ~msg:"the finaliser ran" bool true !cleaned;
          equal ~msg:"the root ran twice" int 2 !runs;
          equal ~msg:"the scope's first key" (array int32)
            (Rng.with_key (Rng.key 3) (fun () -> words (Rng.next_key ())))
            drawn);
      prop
        "peek is the key next_key returns next, and leaves the scope's \
         sequence as it was"
        Gen.(pair key (pair program program))
        (fun (k, (before, after)) ->
          let peeked, next, drawn =
            Rng.with_key k (fun () ->
                let a = keyless before in
                let peeked = words (Rng.peek ()) in
                let next = words (Rng.next_key ()) in
                (peeked, next, a @ keyless after))
          in
          equal ~msg:"the next key" (array int32) next peeked;
          equal ~msg:"the draws" values
            (Rng.with_key k (fun () ->
                 let a = keyless before in
                 ignore (Rng.next_key ());
                 a @ keyless after))
            drawn);
      test "peek runs a scope's root once, as a draw would" (fun () ->
          let runs = ref 0 in
          let peeked, next =
            Rng.with_root
              (fun () ->
                incr runs;
                Rng.key 5)
              (fun () ->
                let peeked = words (Rng.peek ()) in
                (peeked, words (Rng.next_key ())))
          in
          equal ~msg:"the next key" (array int32) next peeked;
          equal ~msg:"root runs" int 1 !runs);
      test "peek outside a scope is the next key" (fun () ->
          let peeked = words (Rng.peek ()) in
          equal (array int32) (words (Rng.next_key ())) peeked);
      test "a domain spawned inside a scope draws outside it" (fun () ->
          let spawned () =
            Rng.with_key (Rng.key 7) (fun () ->
                Domain.join (Domain.spawn (fun () -> fresh 1)))
          in
          not_equal (list (array int32)) (spawned ()) (spawned ()));
    ]

(* Distributions. Each row draws values from a fixed key and holds them to a
   law: a continuous cdf, a pmf on 0, 1, 2, ..., or the mean of a law on [0, 1].
   The statistic is the largest gap between the draws' cdf and the law's, or
   between their mean and the law's.

   A correct sampler fails a row with probability below 1e-7. For n independent
   draws of any law, the cdf gap exceeds e with probability at most 2 exp (-2 n
   e^2) (Dvoretzky-Kiefer-Wolfowitz, with Massart's constant), and so does the
   mean gap of draws on [0, 1] (Hoeffding); [tolerance] is the e where that is
   1e-7. Rounding each draw to its dtype moves its cdf by less than 1e-3 in
   every row, which [tolerance] adds.

   The same bound gives each row its power. At 100,000 draws [tolerance] is
   0.0102, and the draws' own gap exceeds 0.0092 with probability below 1e-7, so
   a sampler whose cdf is 0.02 or more from the law's at some point passes with
   probability below 1e-7. *)

type law = Cdf of (float -> float) | Pmf of (int -> float) | Mean of float

let tolerance n = Float.sqrt (Float.log 2e7 /. (2. *. float_of_int n)) +. 1e-3

let gap law xs =
  let xs = Array.copy xs in
  Array.sort Float.compare xs;
  let n = float_of_int (Array.length xs) in
  (* The law's cdf just below [x] and at [x]; a pmf's is summed up to [x] as [x]
     grows. *)
  let next = ref 0 and below = ref 0. in
  let cdf x =
    match law with
    | Cdf f -> (f x, f x)
    | Pmf p ->
        while float_of_int !next < x do
          below := !below +. p !next;
          incr next
        done;
        (!below, if x < 0. then 0. else !below +. p !next)
    | Mean _ -> assert false
  in
  match law with
  | Mean m -> Float.abs ((Array.fold_left ( +. ) 0. xs /. n) -. m)
  | Cdf _ | Pmf _ ->
      let d = ref 0. in
      Array.iteri
        (fun i x ->
          let below, at = cdf x in
          let i = float_of_int i in
          d :=
            Float.max !d
              (Float.max (below -. (i /. n)) (((i +. 1.) /. n) -. at)))
        xs;
      if Float.is_nan !d then infinity else !d

let clamp x = Float.min 1. (Float.max 0. x)
let phi x = 0.5 *. Float.erfc (-.x /. Float.sqrt 2.)
let pmf ps = Pmf (fun k -> if k < Array.length ps then ps.(k) else 0.)
let exp_neg x = Float.exp (-.x)

(* The standard normal conditioned on [a, b], through the tail that keeps its
   digits there. *)
let truncated a b =
  let a = Float.min a b and b = Float.max a b in
  let g x =
    if a +. b > 0. then -0.5 *. Float.erfc (x /. Float.sqrt 2.) else phi x
  in
  Cdf (fun x -> clamp ((g x -. g a) /. (g b -. g a)))

(* Gamma(a) for an integer a: fewer than a events of a unit-rate Poisson process
   by time x. Gamma(1/2) is the law of z^2 / 2 for a standard normal z. *)
let erlang a =
  Cdf
    (fun x ->
      let term = ref 1. and sum = ref 1. in
      for k = 1 to a - 1 do
        term := !term *. x /. float_of_int k;
        sum := !sum +. !term
      done;
      if x <= 0. then 0. else 1. -. (exp_neg x *. !sum))

let gamma_half = Cdf (fun x -> if x <= 0. then 0. else Float.erf (Float.sqrt x))

(* Beta(a, 1), Beta(1, b), and the arcsine law Beta(1/2, 1/2). *)
let beta_a a = Cdf (fun x -> clamp x ** a)
let beta_b b = Cdf (fun x -> 1. -. ((1. -. clamp x) ** b))
let arcsine = Cdf (fun x -> 2. /. Float.pi *. Float.asin (Float.sqrt (clamp x)))

(* The Poisson pmf, with log k! from Stirling's series from 20 up, where it
   holds to 1e-6. *)
let poisson rate =
  let log_factorial k =
    let x = float_of_int k in
    if k < 20 then
      Float.log
        (Seq.fold_left ( *. ) 1. (Seq.init k (fun i -> float_of_int (i + 1))))
    else
      ((x +. 0.5) *. Float.log x)
      -. x +. 0.9189385332046727
      +. (1. /. (12. *. x))
  in
  Pmf
    (fun k ->
      Float.exp ((float_of_int k *. Float.log rate) -. rate -. log_factorial k))

(* The binomial pmf, its factorials as [poisson]'s. *)
let binomial n p =
  let log_factorial k =
    let x = float_of_int k in
    if k < 20 then
      Float.log
        (Seq.fold_left ( *. ) 1. (Seq.init k (fun i -> float_of_int (i + 1))))
    else
      ((x +. 0.5) *. Float.log x)
      -. x +. 0.9189385332046727
      +. (1. /. (12. *. x))
  in
  Pmf
    (fun k ->
      if k > n then 0.
      else
        Float.exp
          (log_factorial n -. log_factorial k
          -. log_factorial (n - k)
          +. (float_of_int k *. Float.log p)
          +. (float_of_int (n - k) *. Float.log1p (-.p))))

(* The von Mises cdf on [-pi, pi], from its density [exp (kappa (cos t - 1))]
   summed by the trapezoid rule over 2^16 steps and normalised; a large
   concentration is the normal of variance [1 / kappa], within [1 / kappa] of
   it. *)
let von_mises kappa =
  if kappa > 1e4 then Cdf (fun x -> phi (x *. Float.sqrt kappa))
  else
    let steps = 1 lsl 16 in
    let h = 2. *. Float.pi /. float_of_int steps in
    let density i =
      Float.exp (kappa *. (Float.cos ((float_of_int i *. h) -. Float.pi) -. 1.))
    in
    let cumulative = Array.make (steps + 1) 0. in
    for i = 1 to steps do
      cumulative.(i) <-
        cumulative.(i - 1) +. (0.5 *. h *. (density (i - 1) +. density i))
    done;
    let total = cumulative.(steps) in
    Cdf
      (fun x ->
        let t = (x +. Float.pi) /. h in
        if t <= 0. then 0.
        else if t >= float_of_int steps then 1.
        else
          let i = int_of_float t in
          let f = t -. float_of_int i in
          ((cumulative.(i) *. (1. -. f)) +. (cumulative.(i + 1) *. f)) /. total)

(* The product of two independent uniforms, on which a dependence between them
   shows. *)
let product =
  Cdf
    (fun z ->
      if z <= 0. then 0. else if z >= 1. then 1. else z -. (z *. Float.log z))

(* Rows of the table: a sampler of a dtype, or of one or two parameters, at [n]
   draws, named after its arguments. *)
let n = 100_000
let at d v = Nx.broadcast_to [| n |] (Nx.scalar d v)
let pp_dtype = Nx_dtype.pp

let sized name s d law =
  ( Format.asprintf "%s at %a" name pp_dtype d,
    (fun k -> floats (s k d [| n |])),
    law )

let one name s d v law =
  ( Format.asprintf "%s(%g) at %a" name v pp_dtype d,
    (fun k -> floats (s k (at d v))),
    law )

let two name s d a b law =
  ( Format.asprintf "%s(%g, %g) at %a" name a b pp_dtype d,
    (fun k -> floats (s k (at d a) (at d b))),
    law )

(* [n] draws of binomial(trials, p), [p] at [d]. *)
let binomial_row d trials p =
  ( Format.asprintf "binomial(%d, %g) at %a" trials p pp_dtype d,
    (fun k ->
      floats
        (Rng.binomial k (Nx.scalar Nx.int32 (Int32.of_int trials)) (at d p))),
    binomial trials p )

let at2 v = Nx.broadcast_to [| n; 2 |] (Nx.scalar Nx.float64 v)

let trunc d a b =
  two "truncated_normal" Rng.truncated_normal d a b (truncated a b)

(* Column [i] of a draw from rows of parameters [v]. *)
let column name s v i law =
  let t =
    Nx.broadcast_to
      [| n; Array.length v |]
      (Nx.create Nx.float64 [| Array.length v |] v)
  in
  (name, (fun k -> floats (Nx.slice [ A; I i ] (s k t))), law)

let uniforms k = floats (Rng.uniform k Nx.float64 [| n |])

let products name a b =
  (name, (fun k -> Array.map2 ( *. ) (uniforms (a k)) (uniforms (b k))), product)

let distributions =
  let f32, f64 = Nx.(float32, float64) in
  let logits = Nx.create f32 [| 3 |] [| 0.; 1.; 2. |] in
  let z = 1. +. Float.exp 1. +. Float.exp 2. in
  let softmax = pmf (Array.map (fun l -> Float.exp l /. z) [| 0.; 1.; 2. |]) in
  (* 20_000 draws of one element of permutation 5, one key each. *)
  let element i k =
    Array.init 20_000 (fun j ->
        Int64.to_float (Nx.item [ i ] (Rng.permutation (Rng.fold_in k j) 5)))
  in
  let gumbel = Cdf (fun x -> exp_neg (exp_neg x)) in
  let subkey i k = (Rng.split k).(i) in
  [
    sized "uniform" Rng.uniform f32 (Cdf clamp);
    sized "uniform" Rng.uniform f64 (Cdf clamp);
    sized "normal" Rng.normal f32 (Cdf phi);
    sized "normal" Rng.normal f64 (Cdf phi);
    ( "the second half of an odd-length normal draw",
      (fun k ->
        Array.sub (floats (Rng.normal k f32 [| (2 * n) + 1 |])) n (n + 1)),
      Cdf phi );
    sized "gumbel" Rng.gumbel f64 gumbel;
    sized "exponential" Rng.exponential f64 (erlang 1);
    trunc f64 (-0.75) 1.25;
    trunc f64 1.25 (-0.75);
    trunc f32 neg_infinity infinity;
    trunc f32 3. 4.;
    trunc f64 5. 6.;
    trunc f64 7. 8.;
    trunc f64 (-8.) (-7.);
    trunc f64 3. 3.01;
    one "gamma" Rng.gamma f64 0.5 gamma_half;
    one "gamma" Rng.gamma f64 20. (erlang 20);
    column "gamma(5) beside gamma(1/2)" Rng.gamma [| 0.5; 5. |] 1 (erlang 5);
    two "beta" Rng.beta f64 4. 1. (beta_a 4.);
    two "beta" Rng.beta f64 1. 3. (beta_b 3.);
    two "beta" Rng.beta f64 0.5 0.5 arcsine;
    two "beta" Rng.beta f32 0.005 0.005 (Mean 0.5);
    two "beta" Rng.beta f32 0.02 0.05 (Mean (2. /. 7.));
    column "dirichlet(1, 2, 7)'s first component" Rng.dirichlet [| 1.; 2.; 7. |]
      0 (beta_b 9.);
    one "poisson" Rng.poisson f64 0.7 (poisson 0.7);
    one "poisson" Rng.poisson f64 12. (poisson 12.);
    one "poisson" Rng.poisson f64 30. (poisson 30.);
    one "poisson" Rng.poisson f32 4. (poisson 4.);
    one "poisson" Rng.poisson f32 1e5 (poisson 1e5);
    one "poisson" Rng.poisson f64 1e7 (poisson 1e7);
    column "poisson(500) beside poisson(0.5)" Rng.poisson [| 0.5; 500. |] 1
      (poisson 500.);
    binomial_row f32 20 0.5;
    binomial_row f64 30 0.2;
    binomial_row f64 5 0.9;
    binomial_row f32 100 0.97;
    binomial_row f32 1000 0.3;
    binomial_row f32 1_000_000 0.1;
    binomial_row f64 1_000_000 0.6;
    binomial_row f64 1_000_000_000 2e-9;
    binomial_row f32 1_000_000_000 9e-9;
    ( "binomial(40, 0.5) beside binomial(3, 0.5)",
      (fun k ->
        let n =
          Nx.broadcast_to [| n; 2 |] (Nx.create Nx.int32 [| 2 |] [| 3l; 40l |])
        in
        floats (Nx.slice [ A; I 1 ] (Rng.binomial k n (at2 0.5)))),
      binomial 40 0.5 );
    one "von_mises" Rng.von_mises f32 0.
      (Cdf (fun x -> clamp ((x +. Float.pi) /. (2. *. Float.pi))));
    one "von_mises" Rng.von_mises f64 0.5 (von_mises 0.5);
    one "von_mises" Rng.von_mises f32 0.9 (von_mises 0.9);
    one "von_mises" Rng.von_mises f64 1. (von_mises 1.);
    one "von_mises" Rng.von_mises f32 2. (von_mises 2.);
    one "von_mises" Rng.von_mises f64 30. (von_mises 30.);
    one "von_mises" Rng.von_mises f32 1e6 (von_mises 1e6);
    column "von_mises(4) beside von_mises(0)" Rng.von_mises [| 0.; 4. |] 1
      (von_mises 4.);
    column "bernoulli(0.9) beside bernoulli(0.1)" Rng.bernoulli [| 0.1; 0.9 |] 1
      (pmf [| 0.1; 0.9 |]);
    (* A comparison at the dtype's 3 bits would make it 6 in 16. *)
    one "bernoulli" Rng.bernoulli Nx.float8_e5m2 0.3125
      (pmf [| 11. /. 16.; 5. /. 16. |]);
    ( "categorical over logits (0, 1, 2) along the last axis",
      (fun k -> floats (Rng.categorical k (Nx.broadcast_to [| n; 3 |] logits))),
      softmax );
    ( "categorical over logits (0, 1, 2) along axis 0",
      (fun k ->
        let l = Nx.broadcast_to [| 3; n |] (Nx.reshape [| 3; 1 |] logits) in
        floats (Rng.categorical k ~axis:0 l)),
      softmax );
    ( "randint over [-5, 5), shifted by 5",
      (fun k -> floats (Nx.add_s (Rng.randint k ~low:(-5) ~high:5 [| n |]) 5l)),
      pmf (Array.make 10 0.1) );
    ( "randint over all of int32, scaled into [0, 1)",
      (fun k ->
        let v = Rng.randint k ~low:(-0x8000_0000) ~high:0x7FFF_FFFF [| n |] in
        Array.map (fun v -> Float.ldexp (v +. 0x1p31) (-32)) (floats v)),
      Cdf clamp );
    ("the first element of permutation 5", element 0, pmf (Array.make 5 0.2));
    ("the last element of permutation 5", element 4, pmf (Array.make 5 0.2));
    products "uniforms from split's two subkeys, multiplied" (subkey 0)
      (subkey 1);
    products "uniforms from a key and its first subkey, multiplied" Fun.id
      (subkey 0);
    products "uniforms from fold_in 0 and fold_in 1, multiplied"
      (fun k -> Rng.fold_in k 0)
      (fun k -> Rng.fold_in k 1);
    ( "neighbouring uniforms of one draw, multiplied",
      (fun k ->
        let v = floats (Rng.uniform k f32 [| 2 * n |]) in
        Array.init n (fun i -> v.(2 * i) *. v.((2 * i) + 1))),
      product );
  ]

let distribution_tests =
  cases ~tags:[ "slow" ]
    ~name:(fun (name, _, _) -> name ^ " follows its law")
    "distributions" distributions
    (fun (name, draw, law) ->
      let xs = draw (Rng.key (Hashtbl.hash name)) in
      at_most float_exact ~than:(tolerance (Array.length xs)) (gap law xs))

let () =
  exit
    (run "nx random"
       [
         key_tests;
         threefry;
         counters;
         supports;
         uniform_grid;
         errors;
         refusals;
         scopes;
         distribution_tests;
       ])
