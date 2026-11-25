(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtypes =
  [
    Dtype Nx.float64;
    Dtype Nx.float32;
    Dtype Nx.float16;
    Dtype Nx.bfloat16;
    Dtype Nx.float8_e4m3fn;
    Dtype Nx.float8_e5m2;
    Dtype Nx.float4_e2m1fn;
  ]

let values x = Nx.to_array (Nx.cast Nx.float64 x)
let f64 shape xs = Nx.create Nx.float64 shape xs
let f32 shape xs = Nx.create Nx.float32 shape xs
let count shape = Array.fold_left ( * ) 1 shape

let pp_array pp ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ") pp)
    (Array.to_list a)

let pp_shape = pp_array Format.pp_print_int

(* Where a mask comes from: the scope a seed's key roots, one key, or a batch of
   keys of a batch shape, split from a seed's key. *)
type keys = Scope of int | Key of int | Batch of int * int array

let pp_keys ppf = function
  | Scope seed -> Format.fprintf ppf "a scope at key %d" seed
  | Key seed -> Format.fprintf ppf "key %d" seed
  | Batch (seed, b) ->
      Format.fprintf ppf "keys %a split from key %d" pp_shape b seed

let batch_keys seed b =
  Nx.Rng.reshape b (Nx.Rng.split_batch ~n:(count b) (Nx.Rng.key seed))

(* [with_keys k f] is [f ?key] at the keys [k] names. Under [Scope], [f] draws
   first in a fresh scope, so two calls draw one mask. *)
let with_keys k f =
  match k with
  | Scope seed -> Nx.Rng.with_key (Nx.Rng.key seed) (fun () -> f None)
  | Key seed -> f (Some (Nx.Rng.key seed))
  | Batch (seed, b) -> f (Some (batch_keys seed b))

(* An input: [x]'s shape and elements in float64, cast to each dtype, a rate and
   keys. *)
type case = { shape : int array; xs : float array; rate : float; keys : keys }

let pp_case ppf c =
  Format.fprintf ppf "@[x %a %a,@ rate %h,@ %a@]" pp_shape c.shape
    (pp_array (fun ppf v -> Format.fprintf ppf "%h" v))
    c.xs c.rate pp_keys c.keys

(* A batch shape that broadcasts to [shape] aligned on the right: a suffix of
   it, each extent kept or 1, and 1 for an empty axis. *)
let batch_of shape =
  let open Gen in
  let r = Array.length shape in
  let* k = int_range 0 r in
  let+ keep = array ~size:(constant k) bool in
  Array.mapi
    (fun i keep ->
      let s = shape.(r - k + i) in
      if keep && s > 0 then s else 1)
    keep

let keys_of shape =
  let open Gen in
  let* seed = int_range 0 1_000_000 in
  frequency
    [
      (1, constant (Scope seed));
      (1, constant (Key seed));
      (2, map (fun b -> Batch (seed, b)) (batch_of shape));
    ]

let case ?(shape = Gen.(array ~size:(int_range 0 3) (int_range 0 3))) ~element
    ~rate () =
  let open Gen in
  with_pp pp_case
  @@ let* shape = shape in
     let* xs = array ~size:(constant (count shape)) element in
     let* rate = rate in
     let+ keys = keys_of shape in
     { shape; xs; rate; keys }

(* Elements that break a product: signed zeros, infinities and NaN, beside small
   values and floats of any magnitude. *)
let element =
  Gen.(
    frequency
      [
        (6, float_range (-4.) 4.);
        (1, of_list [ -0.; Float.infinity; Float.neg_infinity; Float.nan ]);
        (1, float);
      ])

let below_one = Float.pred 1.

let rate =
  Gen.(
    frequency
      [
        (1, constant 0.); (1, constant below_one); (6, float_range 0. below_one);
      ])

let inputs = case ~element ~rate ()

(* Inputs that reach each cover on every seed. *)
let examples =
  [
    { shape = [| 0; 3 |]; xs = [||]; rate = 0.5; keys = Key 1 };
    { shape = [||]; xs = [| -2.5 |]; rate = 0.25; keys = Scope 2 };
    {
      shape = [| 2; 3 |];
      xs = [| 1.; Float.nan; Float.infinity; Float.neg_infinity; -0.; 3. |];
      rate = 0.5;
      keys = Batch (3, [| 3 |]);
    };
    { shape = [| 4 |]; xs = [| 1.; -2.; 3.; -4. |]; rate = 0.; keys = Key 4 };
    {
      shape = [| 3 |];
      xs = [| 1.; Float.infinity; Float.nan |];
      rate = below_one;
      keys = Key 5;
    };
  ]

let cover_case c =
  let batch =
    match c.keys with Batch (_, b) -> Array.length b > 0 | _ -> false
  in
  cover "a zero-size x" (count c.shape = 0);
  cover "a one-element x" (count c.shape = 1);
  cover "a batch of keys" batch;
  cover "a scope's keys" (match c.keys with Scope _ -> true | _ -> false);
  cover "rate 0" (c.rate = 0.);
  cover "the rate below 1" (c.rate = below_one)

let each_dtype f =
  List.iter (function Dtype dt -> f (Nx.Dtype.name dt) (Dtype dt)) dtypes

(* Keys *)

(* The elements of an [x] of [shape] that the key at index [j] of the batch
   shape [b] reaches: [x] with each axis that [b] aligns with an extent above 1
   fixed at [j]'s index. *)
let reached shape b j =
  let r = Array.length shape and k = Array.length b in
  List.init r (fun a ->
      let i = a - (r - k) in
      if i >= 0 && b.(i) = shape.(a) && b.(i) > 1 then Nx.I j.(i) else Nx.A)

(* Each index of [b] with its place in C order. *)
let indices b =
  List.init (count b) (fun flat ->
      let j = Array.make (Array.length b) 0 in
      let rest = ref flat in
      for i = Array.length b - 1 downto 0 do
        j.(i) <- !rest mod b.(i);
        rest := !rest / b.(i)
      done;
      (flat, j))

let key_batches =
  let open Gen in
  with_pp (fun ppf (s, b, seed) ->
      Format.fprintf ppf "x %a, keys %a split from key %d" pp_shape s pp_shape b
        seed)
  @@ let* shape = array ~size:(int_range 1 3) (int_range 1 3) in
     let* b = batch_of shape in
     let+ seed = int_range 0 1_000_000 in
     (shape, b, seed)

(* The preamble's rows, [[| r; t; 1 |]] keys on [[| r; t; c |]], and one key per
   column. *)
let each_key_draws_alone =
  prop
    "each key of a batch draws the elements of x it reaches as it alone would"
    ~examples:[ ([| 2; 3; 5 |], [| 2; 3; 1 |], 3); ([| 3; 4 |], [| 4 |], 4) ]
    key_batches
    (fun (shape, b, seed) ->
      let last a = a.(Array.length a - 1) in
      cover "keys of batch shape [r; t; 1]"
        (Array.length b > 1 && last b = 1 && last shape > 1);
      cover "a key per column" (Array.length b = 1 && count b > 1);
      let x =
        f32 shape (Array.init (count shape) (fun i -> float_of_int (i + 1)))
      in
      let y =
        Dropout.apply ~rate:0.5 ~training:true ~key:(batch_keys seed b) x
      in
      let alone = Nx.Rng.split ~n:(count b) (Nx.Rng.key seed) in
      List.iter
        (fun (flat, j) ->
          let at = reached shape b j in
          equal
            ~msg:(Format.asprintf "key %a" pp_shape j)
            (array float_exact)
            (values
               (Dropout.apply ~rate:0.5 ~training:true ~key:alone.(flat)
                  (Nx.slice at x)))
            (values (Nx.slice at y)))
        (indices b))

(* Mask *)

let mask_is_bernoulli =
  prop
    "is Nx.Rng.bernoulli of 1 - rate at float32, whatever x's dtype and \
     elements"
    ~examples inputs (fun c ->
      cover_case c;
      let p = Nx.full Nx.float32 c.shape (1. -. c.rate) in
      let expected =
        with_keys c.keys (fun key -> Nx.to_array (Nx.Rng.bernoulli ?key p))
      in
      each_dtype (fun msg (Dtype dt) ->
          let x = Nx.cast dt (f64 c.shape c.xs) in
          let m =
            with_keys c.keys (fun key -> Dropout.mask ~rate:c.rate ?key x)
          in
          equal ~msg
            (pair (array int) (array bool))
            (c.shape, expected)
            (Nx.shape m, Nx.to_array m)))

let bad_rates = [ -0.1; 1.; 1.5; Float.nan; Float.infinity; Float.neg_infinity ]

let mask_refuses_rates =
  cases "raises if rate is outside [0, 1)" ~id:(Printf.sprintf "%h") bad_rates
    (fun rate ->
      raises_match (Exn.invalid_arg ~sub:"Dropout.mask: rate must be in [0, 1)")
        (fun () -> Dropout.mask ~rate (Nx.ones Nx.float32 [| 2 |])))

let mask_refuses_keys () =
  raises_match (Exn.invalid_arg ~sub:"Dropout.mask: keys of batch shape [3]")
    (fun () ->
      Dropout.mask ~rate:0.5 ~key:(batch_keys 1 [| 3 |])
        (Nx.ones Nx.float32 [| 2; 4 |]))

(* Apply *)

(* [(x * m) * s] in float64, rounded once to [dt]. [x * m] is [x] or [x * 0],
   and [s], rounded to [dt] first, has no more bits than [x], so below float64
   their product is exact before the rounding. *)
let dropout_of dt ~rate xs m =
  let s = Nx.Dtype.of_float dt (1. /. (1. -. rate)) in
  Array.map2
    (fun x keep -> Nx.Dtype.of_float dt (x *. (if keep then 1. else 0.) *. s))
    xs m

let training_is_scaled_product =
  prop
    "with training = true, the result is (x * m) * s, each product rounded to \
     x's dtype"
    ~examples inputs (fun c ->
      cover_case c;
      let kept =
        with_keys c.keys (fun key ->
            Nx.to_array (Dropout.mask ~rate:c.rate ?key (f64 c.shape c.xs)))
      in
      cover "a dropped infinity or NaN"
        (Array.exists2
           (fun x keep -> (not keep) && not (Float.is_finite x))
           c.xs kept);
      each_dtype (fun msg (Dtype dt) ->
          let x = Nx.cast dt (f64 c.shape c.xs) in
          let m =
            with_keys c.keys (fun key -> Dropout.mask ~rate:c.rate ?key x)
          in
          let y =
            with_keys c.keys (fun key ->
                Dropout.apply ~rate:c.rate ~training:true ?key x)
          in
          equal ~msg (array float_exact)
            (dropout_of dt ~rate:c.rate (values x) (Nx.to_array m))
            (values y)))

let counted = f32 [| 64 |] (Array.init 64 (fun i -> float_of_int (i + 1)))
let drop ?key x = Dropout.apply ~rate:0.5 ~training:true ?key x

(* The nth keyless draw in a scope at [k] is the draw at [fold_in k n]. *)
let fresh_mask_per_call () =
  let k = Nx.Rng.key 5 in
  let at n = values (drop ~key:(Nx.Rng.fold_in k n) counted) in
  let first, second =
    Nx.Rng.with_key k (fun () ->
        let first = values (drop counted) in
        (first, values (drop counted)))
  in
  equal
    (pair (array float_exact) (array float_exact))
    (at 0, at 1)
    (first, second)

(* Rates up to 0.9 and weights up to 2 keep [w * s] finite at every dtype. Two
   to four elements per axis at rates from 0.1 give most cases a dropped and a
   kept element. *)
let gradients_through_survivors =
  prop "gradients flow to x through the surviving elements only"
    (case
       ~shape:Gen.(array ~size:(int_range 1 3) (int_range 2 4))
       ~element:(Gen.float_range (-2.) 2.) ~rate:(Gen.float_range 0.1 0.9) ())
    (fun c ->
      let kept =
        with_keys c.keys (fun key ->
            Nx.to_array (Dropout.mask ~rate:c.rate ?key (f64 c.shape c.xs)))
      in
      cover "a dropped and a kept element"
        (Array.mem true kept && Array.mem false kept);
      each_dtype (fun msg (Dtype dt) ->
          let w = Nx.cast dt (f64 c.shape c.xs) in
          let x = Nx.ones dt c.shape in
          let m =
            with_keys c.keys (fun key -> Dropout.mask ~rate:c.rate ?key x)
          in
          let objective key x =
            Nx.sum (Nx.mul w (Dropout.apply ~rate:c.rate ~training:true ?key x))
          in
          let g =
            with_keys c.keys (fun key ->
                Rune.grad Nx.Ptree.tensor (objective key) x)
          in
          equal ~msg (array float_exact)
            (dropout_of dt ~rate:c.rate (values w) (Nx.to_array m))
            (values g)))

(* Past each dtype's range: 1 / (1 - rate) is 65536 at float16 and float8_e5m2,
   1024 at float8_e4m3fn (largest 448) and 8 at float4_e2m1fn (largest 6). Each
   x holds about 16 / (1 - rate) elements, so the row's key keeps about 16 of
   them. *)
let saturated_scale =
  let max_finite dt = (Nx.Dtype.float_format dt).max_finite in
  cases
    "s is 1 / (1 - rate) stored at x's dtype, infinite past its range or its \
     largest value where it has no infinity"
    ~id:(fun (Dtype dt, _, _) -> Nx.Dtype.name dt)
    [
      (Dtype Nx.float16, 1. -. (2. ** -16.), Float.infinity);
      (Dtype Nx.float8_e5m2, 1. -. (2. ** -16.), Float.infinity);
      (Dtype Nx.float8_e4m3fn, 1. -. (2. ** -10.), max_finite Nx.float8_e4m3fn);
      (Dtype Nx.float4_e2m1fn, 0.875, max_finite Nx.float4_e2m1fn);
    ]
    (fun (Dtype dt, rate, s) ->
      let n = int_of_float (16. /. (1. -. rate)) in
      let xs = Array.init n (fun i -> [| 0.5; -1.; 1.5; -0.5 |].(i mod 4)) in
      let x = Nx.cast dt (f64 [| n |] xs) in
      let key = Nx.Rng.key 1 in
      let kept = Nx.to_array (Dropout.mask ~rate ~key x) in
      let y = values (Dropout.apply ~rate ~training:true ~key x) in
      let at = Nx.Dtype.of_float dt in
      let wrong = ref [] in
      Array.iteri
        (fun i x ->
          let expected = at (x *. (if kept.(i) then 1. else 0.) *. s) in
          if
            Int64.bits_of_float expected <> Int64.bits_of_float y.(i)
            && not (Float.is_nan expected && Float.is_nan y.(i))
          then wrong := (i, y.(i)) :: !wrong)
        xs;
      at_least ~msg:"kept elements" int ~than:1
        (Array.fold_left (fun n k -> if k then n + 1 else n) 0 kept);
      equal ~msg:"elements other than (x * m) * s"
        (list (pair int float_exact))
        [] (List.rev !wrong))

let four =
  fixture (fun () ->
      Nx.devices
        (List.init 4 (fun k ->
             Result.get_ok (Rig.software (Printf.sprintf "kaun-dropout-%d" k)))))

(* A product over an axis cut across four devices, asked for as a sum: one mask
   scales each term. The terms are integers and the mask scales by two, so the
   sum's dropout and the closed value's agree exactly. *)
let keeps_a_sum () =
  let (module Four : Nx.Devices) = four () in
  let a =
    Nx.place (Four.split ~axis:1)
      (Nx.reshape [| 2; 8 |] (Nx.arange Nx.float32 0 16 1))
  and b = Nx.place (Four.split ~axis:0) (Nx.ones Nx.float32 [| 8; 3 |]) in
  let y = Nx.matmul ~sum:true a b in
  let key = Nx.Rng.place Four.on (Nx.Rng.key 5) in
  let d = drop ~key y in
  let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal in
  let sum = Nx.Placement.partial (Nx.Mesh.v Four.v [ ("x", 4) ]) [ "x" ] [] in
  equal ~msg:"the product" placement sum (Nx.placement y);
  equal ~msg:"its dropout" placement sum (Nx.placement d);
  let closed = Nx.place (Nx.Placement.closed (Nx.placement y)) y in
  equal ~msg:"dropout of the closed value" (array float_exact)
    (values (drop ~key closed))
    (values d)

let unchanged =
  prop "with training = false or rate = 0., the result is x unchanged"
    ~examples:(List.map (fun c -> (c, false)) examples)
    Gen.(pair inputs bool)
    (fun (c, training) ->
      cover "eval mode" (not training);
      cover "rate 0 in training mode" training;
      let rate = if training then 0. else c.rate in
      each_dtype (fun msg (Dtype dt) ->
          let x = Nx.cast dt (f64 c.shape c.xs) in
          let y =
            with_keys c.keys (fun key -> Dropout.apply ~rate ~training ?key x)
          in
          equal ~msg (array float_exact) (values x) (values y)))

let in_either_mode rows =
  List.concat_map (fun r -> [ (r, true); (r, false) ]) rows

let apply_refuses_rates =
  cases "raises if rate is outside [0, 1), in either mode"
    ~id:(fun (rate, training) -> Printf.sprintf "%h, training %b" rate training)
    (in_either_mode bad_rates)
    (fun (rate, training) ->
      raises_match
        (Exn.invalid_arg ~sub:"Dropout.apply: rate must be in [0, 1)")
        (fun () -> Dropout.apply ~rate ~training (Nx.ones Nx.float32 [| 2 |])))

let apply_refuses_keys =
  cases
    "raises if the keys' batch shape does not broadcast to x's shape, in \
     either mode"
    ~id:(fun (b, training) ->
      Format.asprintf "keys %a, training %b" pp_shape b training)
    (in_either_mode [ [| 2 |]; [| 3; 2; 1 |] ])
    (fun (b, training) ->
      raises_match (Exn.invalid_arg ~sub:"do not broadcast to x's shape [2; 4]")
        (fun () ->
          Dropout.apply ~rate:0.5 ~training ~key:(batch_keys 1 b)
            (Nx.ones Nx.float32 [| 2; 4 |])))

(* Under jit *)

module Keyed = struct
  type t = { x : Nx.host Nx.float32_t; key : Nx.host Nx.Rng.key }

  let walk c { x; key } =
    let open Nx.Ptree.Walk in
    let x = field c "x" tensor x in
    let key = field c "key" Nx.Rng.walk key in
    { x; key }
end

let keyless_raises () =
  let f = Rune.jit' (fun x -> drop x) in
  raises_match (Exn.invalid_arg ~sub:"Rune.jit") (fun () -> f counted)

(* One program, called at two keys. *)
let key_leaf_draws_fresh () =
  Compiled.here ();
  let f =
    Rune.jit
      Nx.Ptree.(v Keyed.walk @-> returns (pair tensor tensor))
      (fun { Keyed.x; key } ->
        (drop ~key x, drop ~key:(Nx.Rng.fold_in key 1) x))
  in
  let compiled seed =
    let a, b = f { Keyed.x = counted; key = Nx.Rng.key seed } in
    (values a, values b)
  in
  let eager seed =
    let key = Nx.Rng.key seed in
    ( values (drop ~key counted),
      values (drop ~key:(Nx.Rng.fold_in key 1) counted) )
  in
  let both = pair (array float_exact) (array float_exact) in
  equal ~msg:"key 21" both (eager 21) (compiled 21);
  equal ~msg:"key 22" both (eager 22) (compiled 22)

let closed_key_same_mask () =
  Compiled.here ();
  let key = Nx.Rng.key 21 in
  let f = Rune.jit' (fun x -> drop ~key x) in
  let other = Nx.mul_s counted (-0.5) in
  equal ~msg:"first call" (array float_exact)
    (values (drop ~key counted))
    (values (f counted));
  equal ~msg:"second call" (array float_exact)
    (values (drop ~key other))
    (values (f other))

(* Keyless dropout in a scope at the step's key draws at [fold_in key 0], the
   scope's first key. *)
let scope_at_step_key () =
  Compiled.here ();
  let f =
    Rune.jit
      Nx.Ptree.(v Keyed.walk @-> returns tensor)
      (fun { Keyed.x; key } -> Nx.Rng.with_key key (fun () -> drop x))
  in
  List.iter
    (fun seed ->
      let key = Nx.Rng.key seed in
      equal
        ~msg:(Printf.sprintf "key %d" seed)
        (array float_exact)
        (values (drop ~key:(Nx.Rng.fold_in key 0) counted))
        (values (f { Keyed.x = counted; key })))
    [ 23; 24 ]

let away =
  fixture (fun () -> Nx.devices [ Result.get_ok (Rig.software "kaun-dropout") ])

(* Off the host the sampler's parameter check is the compiled call's own, so the
   call compiles and nx evaluates nothing. *)
let off_the_host () =
  Compiled.here ();
  let (module Away : Nx.Devices) = away () in
  let x = Nx.place Away.on counted in
  let key = Nx.Rng.place Away.on (Nx.Rng.key 21) in
  let before = Rune.evaluations () in
  let y = Rune.jit' (fun x -> drop ~key x) x in
  equal ~msg:"evaluated calls" int 0 (Rune.evaluations () - before);
  equal ~msg:"eager's mask" (array float_exact)
    (values (drop ~key x))
    (values y)

(* Under vmap *)

let rows = f32 [| 3; 16 |] (Array.init 48 (fun i -> float_of_int (i + 1)))

(* Each lane's eager result, stacked. *)
let lanes f =
  values (Nx.stack ~axis:0 (List.init 3 (fun i -> f i (Nx.get [ i ] rows))))

let closed_key_every_lane () =
  let key = Nx.Rng.key 7 in
  equal (array float_exact)
    (lanes (fun _ row -> drop ~key row))
    (values (Rune.vmap' (fun row -> drop ~key row) rows))

let folded_lane_index () =
  let key = Nx.Rng.key 7 in
  let axis = Rune.axis () in
  let y =
    Rune.vmap' ~axis
      (fun row ->
        let i = Nx.cast Nx.int64 (Rune.lane_index axis row) in
        drop ~key:(Nx.Rng.fold_in_tensor key i) row)
      rows
  in
  equal (array float_exact)
    (lanes (fun i row -> drop ~key:(Nx.Rng.fold_in key i) row))
    (values y)

let batch_of_keys () =
  let key = Nx.Rng.key 7 in
  let keys = Nx.Rng.split ~n:3 key in
  let y =
    Rune.vmap
      Nx.Ptree.(pair Nx.Rng.ptree tensor @-> returns tensor)
      (fun (key, row) -> drop ~key row)
      (Nx.Rng.split_batch ~n:3 key, rows)
  in
  equal (array float_exact)
    (lanes (fun i row -> drop ~key:keys.(i) row))
    (values y)

let () =
  exit
    (run "kaun dropout"
       [
         group "keys" [ each_key_draws_alone ];
         group "mask"
           [
             mask_is_bernoulli;
             mask_refuses_rates;
             test
               "raises if the keys' batch shape does not broadcast to x's shape"
               mask_refuses_keys;
           ];
         group "apply"
           [
             training_is_scaled_product;
             test "without a key, each call draws a fresh mask from the scope"
               fresh_mask_per_call;
             saturated_scale;
             gradients_through_survivors;
             test "dropout of a sum is a sum" keeps_a_sum;
             unchanged;
             group "under jit"
               [
                 test
                   "a draw without a key outside a scope the function opens \
                    raises"
                   keyless_raises;
                 test
                   "a key leaf, or one fold_in-derived from it, draws eager's \
                    mask on each call"
                   key_leaf_draws_fresh;
                 test
                   "a key the function closes over draws the same mask on \
                    every call"
                   closed_key_same_mask;
                 test "a scope opened at the step's key draws its first key"
                   scope_at_step_key;
                 test "a key off the host compiles and draws eager's mask"
                   off_the_host;
               ];
             group "under vmap"
               [
                 test
                   "a key the function closes over draws the identical mask in \
                    every lane"
                   closed_key_every_lane;
                 test
                   "the lane index folded into the key gives lane i the mask \
                    of fold_in key i"
                   folded_lane_index;
                 test "a batch of keys gives lane i key i" batch_of_keys;
               ];
             apply_refuses_rates;
             apply_refuses_keys;
           ];
       ])
