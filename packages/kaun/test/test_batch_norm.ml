(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let wide = [ Dtype Nx.float64; Dtype Nx.float32 ]

let narrow =
  [
    Dtype Nx.float16;
    Dtype Nx.bfloat16;
    Dtype Nx.float8_e4m3fn;
    Dtype Nx.float8_e5m2;
    Dtype Nx.float4_e2m1fn;
  ]

let each_dtype dtypes f =
  List.iter (function Dtype dt -> f (Nx.Dtype.name dt) (Dtype dt)) dtypes

let values x = Nx.to_array (Nx.cast Nx.float64 x)
let vec dt xs = Nx.create dt [| Array.length xs |] xs
let f32 shape xs = Nx.create Nx.float32 shape xs
let count shape = Array.fold_left ( * ) 1 shape
let params () = Nx.Ptree.instantiate (module Batch_norm)
let stats () = Nx.Ptree.instantiate (module Batch_norm.Stats)

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

(* Each leaf's path and dtype, in walk order. *)
let dtypes_of s x =
  List.rev
    (Nx.Ptree.fold s
       (fun path t acc ->
         (Nx.Ptree.Path.to_string path, Nx.Dtype.name (Nx.dtype t)) :: acc)
       x [])

(* Each leaf's path and elements, in walk order. *)
let leaves s x =
  List.rev
    (Nx.Ptree.fold s
       (fun path t acc -> (Nx.Ptree.Path.to_string path, values t) :: acc)
       x [])

let leaves_t = list (pair string (array float_exact))

let pp_shape ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_array ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
       (fun ppf v -> Format.fprintf ppf "%h" v))
    (Array.to_list a)

(* An expected value and the error its derivation allows, against an actual
   value: equal when both are one float, NaN included, or within the
   allowance. *)
let within =
  Testable.make
    ~pp:(fun ppf (v, tol) ->
      if tol = 0. then Format.fprintf ppf "%h" v
      else Format.fprintf ppf "%h +- %.3g" v tol)
    ~equal:(fun (v, tol) (v', _) ->
      Float.equal v v' || Float.abs (v -. v') <= tol)

let exactly xs = Array.map (fun v -> (v, 0.)) xs
let u dt = (Nx.Dtype.float_format dt).epsilon /. 2.

(* The least positive subnormal: the most a rounding loses near zero. *)
let tiny dt =
  let f = Nx.Dtype.float_format dt in
  f.min_normal *. f.epsilon

let u64 = epsilon_float /. 2.

(* A layer and its input: [x]'s shape, features last, and elements, the
   parameters, the statistics and the options, all in float64 and cast to each
   dtype. *)
type case = {
  shape : int array;
  xs : float array;
  gamma : float array;
  beta : float array option;
  mean : float array;
  var : float array;
  momentum : float;
  eps : float;
}

let pp_case ppf c =
  Format.fprintf ppf
    "@[x %a %a,@ gamma %a,@ beta %a,@ mean %a,@ var %a,@ momentum %h,@ eps %h@]"
    pp_shape c.shape pp_array c.xs pp_array c.gamma
    (Format.pp_print_option
       ~none:(fun ppf () -> Format.fprintf ppf "None")
       pp_array)
    c.beta pp_array c.mean pp_array c.var c.momentum c.eps

let features c = c.shape.(Array.length c.shape - 1)
let rows c = count c.shape / features c

let layer dt c =
  ( { Batch_norm.gamma = vec dt c.gamma; beta = Option.map (vec dt) c.beta },
    { Batch_norm.Stats.mean = vec dt c.mean; var = vec dt c.var } )

let input dt c = Nx.cast dt (Nx.create Nx.float64 c.shape c.xs)

(* Elements that break a normalization: 1000.1, whose float32 mean is inexact
   (777f1e4f61), a signed zero, an infinity and NaN. *)
let element =
  Gen.(
    frequency
      [
        (5, float_range (-4.) 4.);
        (1, of_list [ 1000.1; -0.; Float.infinity; Float.nan ]);
      ])

let momentum =
  Gen.(frequency [ (1, constant 0.); (1, constant 1.); (6, float_range 0. 1.) ])

let case ~element =
  let open Gen in
  with_pp pp_case
  @@ let* lead = array ~size:(int_range 1 3) (int_range 1 3) in
     let* f = int_range 1 4 in
     let shape = Array.append lead [| f |] in
     let* xs = array ~size:(constant (count shape)) element in
     let* gamma = array ~size:(constant f) (float_range (-2.) 2.) in
     let* beta = option (array ~size:(constant f) (float_range (-2.) 2.)) in
     let* mean = array ~size:(constant f) (float_range (-4.) 4.) in
     let* var = array ~size:(constant f) (float_range 0. 4.) in
     let* momentum = momentum in
     let+ eps = float_range 1e-6 0.1 in
     { shape; xs; gamma; beta; mean; var; momentum; eps }

(* A batch of one row, whose variance is zero, with and without a shift, at
   momentum 0 and 1. *)
let one_row =
  {
    shape = [| 1; 3 |];
    xs = [| 2.5; -1.; 0.5 |];
    gamma = [| 1.5; -0.5; 2. |];
    beta = Some [| 0.25; 1.; -1. |];
    mean = [| 0.5; 0.; -1. |];
    var = [| 1.; 2.; 0.5 |];
    momentum = 0.;
    eps = 1e-5;
  }

let examples = [ one_row; { one_row with beta = None; momentum = 1. } ]

let cover_case c =
  cover "a one-row batch" (rows c = 1);
  cover "x of rank 4" (Array.length c.shape = 4);
  cover "a layer without a shift" (c.beta = None)

(* Parameters *)

let walk_visits =
  cases "visits gamma, then beta, reporting whether beta is present"
    ~id:(fun (shift, _) -> if shift then "with a shift" else "without a shift")
    [
      (true, [ "gamma: a leaf"; "beta: Some"; "beta: a leaf" ]);
      (false, [ "gamma: a leaf"; "beta: None" ]);
    ]
    (fun (shift, expected) ->
      let p, _ = Batch_norm.init ~shift ~features:2 Nx.Host.on in
      equal (list string) expected (visits (params ()) p))

(* [Nx_io.Archive.to_value] fails naming the entry one layer has and the other
   lacks. *)
let checkpoint_refuses_shift () =
  let shifted, _ = Batch_norm.init ~features:2 Nx.Host.on in
  let plain, _ = Batch_norm.init ~shift:false ~features:2 Nx.Host.on in
  let save x = Nx_io.Archive.of_value ~prefix:"n" (params ()) x in
  let load ~like a = Nx_io.Archive.to_value ~prefix:"n" (params ()) ~like a in
  raises_match ~msg:"a shifted checkpoint into a layer without a shift"
    (Exn.failure ~sub:"n.beta") (fun () -> load ~like:plain (save shifted));
  raises_match ~msg:"a checkpoint without a shift into a shifted layer"
    (Exn.failure ~sub:"n.beta") (fun () -> load ~like:shifted (save plain))

let params_cast () =
  let p, _ = Batch_norm.init ~features:2 Nx.Host.on in
  equal
    (list (pair string string))
    [ ("gamma", "float16"); ("beta", "float16") ]
    (dtypes_of (params ()) (Nx.Ptree.cast (module Batch_norm) Nx.float16 p))

(* Running statistics *)

let stats_walk () =
  let _, s = Batch_norm.init ~features:2 Nx.Host.on in
  equal (list string) [ "mean: a leaf"; "var: a leaf" ] (visits (stats ()) s)

let stats_cast () =
  let _, s = Batch_norm.init ~features:2 Nx.Host.on in
  equal
    (list (pair string string))
    [ ("mean", "bfloat16"); ("var", "bfloat16") ]
    (dtypes_of (stats ())
       (Nx.Ptree.cast (module Batch_norm.Stats) Nx.bfloat16 s))

let stats_saved () =
  let p, _ = Batch_norm.init ~features:2 Nx.Host.on in
  let s =
    {
      Batch_norm.Stats.mean = vec Nx.float32 [| 1.; 2. |];
      var = vec Nx.float32 [| 3.; 4. |];
    }
  in
  let archive =
    Nx_io.Archive.union
      [
        Nx_io.Archive.of_value ~prefix:"model" (params ()) p;
        Nx_io.Archive.of_value ~prefix:"stats" (stats ()) s;
      ]
  in
  equal ~msg:"names" (list string)
    [ "model.beta"; "model.gamma"; "stats.mean"; "stats.var" ]
    (Nx_io.Archive.names archive);
  let _, like = Batch_norm.init ~features:2 Nx.Host.on in
  equal ~msg:"statistics read back" leaves_t
    (leaves (stats ()) s)
    (leaves (stats ())
       (Nx_io.Archive.to_value ~prefix:"stats" (stats ()) ~like archive))

(* Constructors *)

let two =
  fixture (fun () ->
      Nx.devices
        [
          Result.get_ok (Rig.software "kaun-batch-norm-0");
          Result.get_ok (Rig.software "kaun-batch-norm-1");
        ])

let init_values () =
  let (module Two : Nx.Devices) = two () in
  let at = Two.split ~axis:0 in
  let p, s = Batch_norm.init ~features:4 at in
  let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal in
  let leaf ~msg expected x =
    equal ~msg
      (pair placement (array float_exact))
      (at, Array.make 4 expected)
      (Nx.placement x, values x)
  in
  leaf ~msg:"gamma" 1. p.gamma;
  leaf ~msg:"beta" 0. (Option.get p.beta);
  leaf ~msg:"mean" 0. s.mean;
  leaf ~msg:"var" 1. s.var

let init_without_shift () =
  let p, _ = Batch_norm.init ~shift:false ~features:3 Nx.Host.on in
  is_none ~pp:(fun ppf _ -> Format.fprintf ppf "a beta") p.beta

let init_refuses_features =
  cases "raises if features <= 0" ~id:string_of_int [ 0; -1 ] (fun features ->
      raises_match
        (Exn.invalid_arg ~sub:"Batch_norm.init: features must be positive")
        (fun () -> Batch_norm.init ~features Nx.Host.on))

(* Applying *)

let other_axes c = List.init (Array.length c.shape - 1) Fun.id

(* A per-feature vector with [x]'s rank, features last. *)
let per_feature c v =
  let shape = Array.map (fun _ -> 1) c.shape in
  shape.(Array.length shape - 1) <- features c;
  Nx.reshape shape v

let training_standardizes =
  prop
    "with training = true, y is Nx.standardize over the other axes with gamma \
     and beta, whatever stats"
    ~examples (case ~element) (fun c ->
      cover_case c;
      each_dtype (wide @ narrow) (fun msg (Dtype dt) ->
          let p, s = layer dt c in
          let x = input dt c in
          let y, _ =
            Batch_norm.apply ~momentum:c.momentum ~eps:c.eps p s ~training:true
              x
          in
          let expected =
            Nx.standardize ~axes:(other_axes c) ~eps:c.eps
              ~scale:(per_feature c p.gamma)
              ?shift:(Option.map (per_feature c) p.beta)
              x
          in
          equal ~msg (array float_exact) (values expected) (values y)))

(* [momentum * s + (1 - momentum) * b], with [momentum] and [1 - momentum]
   stored at the dtype ([Nx.mul_s]) and each product and the sum rounded: three
   roundings of at most [u] bound the error by [(1 + u)^3 - 1] of [|momentum *
   s| + |(1 - momentum) * b|]. The float64 reference's own three roundings and
   [1 - momentum]'s add [4 u64], and near zero each rounding loses half a
   subnormal more. *)
let blend dt ~momentum s b =
  let bound = ((1. +. u dt) ** 3.) -. 1. +. (4. *. u64) in
  Array.map2
    (fun s b ->
      let m = momentum *. s and n = (1. -. momentum) *. b in
      (m +. n, (bound *. (Float.abs m +. Float.abs n)) +. (3. *. tiny dt)))
    s b

let training_stats =
  prop
    "with training = true, stats' is momentum * stats + (1 - momentum) * \
     batch, the batch's Nx.mean and Nx.var at x's dtype"
    ~examples
    (case ~element:(Gen.float_range (-4.) 4.))
    (fun c ->
      cover_case c;
      cover "momentum 0" (c.momentum = 0.);
      cover "momentum 1" (c.momentum = 1.);
      each_dtype (wide @ narrow) (fun msg (Dtype dt) ->
          let p, s = layer dt c in
          let x = input dt c in
          let _, s' =
            Batch_norm.apply ~momentum:c.momentum ~eps:c.eps p s ~training:true
              x
          in
          let axes = other_axes c in
          let expected =
            ( blend dt ~momentum:c.momentum (values s.mean)
                (values (Nx.mean ~axes x)),
              blend dt ~momentum:c.momentum (values s.var)
                (values (Nx.var ~axes x)) )
          in
          equal ~msg
            (pair (array within) (array within))
            expected
            (exactly (values s'.mean), exactly (values s'.var))))

(* 777f1e4f61: a float32 batch mean of 1000.1 is not exact, and the feature
   still maps to beta. *)
let constant_feature () =
  let p, s = Batch_norm.init ~features:2 Nx.Host.on in
  let p = { p with beta = Some (vec Nx.float32 [| 0.5; -2. |]) } in
  let x =
    f32 [| 768; 2 |]
      (Array.init 1536 (fun i -> if i mod 2 = 0 then 1000.1 else 3.))
  in
  let y, _ = Batch_norm.apply p s ~training:true x in
  equal (array float_exact)
    (Array.init 1536 (fun i -> if i mod 2 = 0 then 0.5 else -2.))
    (values y)

module Model = struct
  type ('a, 'd) t = { lin : ('a, 'd) Linear.t; bn : ('a, 'd) Batch_norm.t }

  let walk c { lin; bn } =
    let open Nx.Ptree.Walk in
    let lin = field c "lin" Linear.walk lin in
    let bn = field c "bn" Batch_norm.walk bn in
    { lin; bn }
end

let model = Nx.Ptree.instantiate (module Model)

let steps =
  Gen.(
    with_pp (fun ppf (seed, n, f, m) ->
        Format.fprintf ppf "seed %d, %d rows, %d features, momentum %h" seed n f
          m)
    @@ quad (int_range 0 1_000_000) (int_range 1 6) (int_range 1 4) momentum)

(* A linear layer before the norm makes the statistics depend on the
   differentiated parameters. *)
let step_returns_stats =
  prop "value_and_grad_aux gives back a training step's stats' undifferentiated"
    ~examples:[ (1, 1, 3, 0.9); (2, 4, 2, 0.); (3, 4, 2, 1.) ]
    steps
    (fun (seed, n, features, momentum) ->
      cover "a one-row batch" (n = 1);
      Nx.Rng.with_key (Nx.Rng.key seed) @@ fun () ->
      let x = Nx.Rng.normal Nx.float32 [| n; 2 |] in
      let target = Nx.Rng.normal Nx.float32 [| n; features |] in
      let p =
        {
          Model.lin =
            Linear.init ~key:(Nx.Rng.next_key ()) ~inputs:2 ~outputs:features;
          bn = fst (Batch_norm.init ~features Nx.Host.on);
        }
      in
      let s =
        {
          Batch_norm.Stats.mean = Nx.Rng.normal Nx.float32 [| features |];
          var = Nx.exp (Nx.Rng.normal Nx.float32 [| features |]);
        }
      in
      let forward p =
        Batch_norm.apply ~momentum p.Model.bn s ~training:true
          (Linear.apply p.Model.lin x)
      in
      let objective p =
        let y, s' = forward p in
        (Loss.mse y target, s')
      in
      let loss, grads, s' =
        Rune.value_and_grad_aux model (stats ()) objective p
      in
      let loss_alone, grads_alone =
        Rune.value_and_grad model (fun p -> fst (objective p)) p
      in
      equal
        (triple (array float_exact) leaves_t leaves_t)
        ( values loss_alone,
          leaves model grads_alone,
          leaves (stats ()) (snd (forward p)) )
        (values loss, leaves model grads, leaves (stats ()) s'))

(* At float64, with each element of the objective weighted apart. *)
let gradients =
  prop "gradients flow to gamma, beta and, through the batch's statistics, to x"
    ~examples
    (case ~element:(Gen.float_range (-4.) 4.))
    (fun c ->
      cover_case c;
      let p, s = layer Nx.float64 c in
      let x = input Nx.float64 c in
      let w =
        Nx.create Nx.float64 c.shape
          (Array.init (count c.shape) (fun i ->
               float_of_int ((i * 7 mod 11) - 5) /. 3.))
      in
      let objective (p, x) =
        Nx.sum
          (Nx.mul w (fst (Batch_norm.apply ~eps:c.eps p s ~training:true x)))
      in
      equal (result unit string) (Ok ())
        (Rune.check_grads Nx.Ptree.(pair (params ()) tensor) objective (p, x)))

let same_stats () =
  Testable.make
    ~pp:(fun ppf (s : _ Batch_norm.Stats.t) ->
      Format.fprintf ppf "{ mean = %a; var = %a }" pp_array (values s.mean)
        pp_array (values s.var))
    ~equal:( == )

(* [gamma * (x - mean) / sqrt (var + eps) + beta] at the dtype. The difference
   rounds once ([u]); [eps] stored at the dtype and the sum under the root round
   once each ([2u]), which the root halves ([u]); a reciprocal root ([Nx.rsqrt])
   is within 2 ulps ([4u]); the two products round once each ([2u]). So the term
   before [beta] is within [8u] of its value to first order, [9u] with the
   second-order terms, and the sum adds [u] of [y]. The float64 reference,
   multiplying by [gamma] last, adds [6 u64] and [u64]. Near zero the two
   products and the sum each lose up to half a subnormal more, on both sides. *)
let eval_reference dt c ~xs ~gamma ~beta ~mean ~var =
  let f = features c and u = u dt in
  let near_zero = 2. *. (tiny dt +. tiny Nx.float64) in
  Array.mapi
    (fun i x ->
      let j = i mod f in
      let t = (x -. mean.(j)) /. sqrt (var.(j) +. c.eps) *. gamma.(j) in
      let y = match beta with None -> t | Some b -> t +. b.(j) in
      ( y,
        (((9. *. u) +. (6. *. u64)) *. Float.abs t)
        +. ((u +. u64) *. Float.abs y)
        +. near_zero ))
    xs

let eval_affine =
  prop
    "with training = false, y is gamma * (x - mean) / sqrt (var + eps) + beta \
     from stats, and stats' == stats"
    ~examples
    (case ~element:(Gen.float_range (-4.) 4.))
    (fun c ->
      cover_case c;
      each_dtype wide (fun msg (Dtype dt) ->
          let p, s = layer dt c in
          let x = input dt c in
          let y, s' = Batch_norm.apply ~eps:c.eps p s ~training:false x in
          let expected =
            eval_reference dt c ~xs:(values x) ~gamma:(values p.gamma)
              ~beta:(Option.map values p.beta) ~mean:(values s.mean)
              ~var:(values s.var)
          in
          equal ~msg:(msg ^ ": y") (array within) expected (exactly (values y));
          satisfies ~msg:(msg ^ ": stats'") ~claim:"stats itself"
            (same_stats ())
            (fun s' -> s' == s)
            s'))

let up x = Nx.cast Nx.float32 x

let eval_island =
  prop
    "with training = false, a narrow x is normalized in a float32 island and \
     rounded once"
    ~examples (case ~element) (fun c ->
      cover_case c;
      each_dtype narrow (fun msg (Dtype dt) ->
          let p, s = layer dt c in
          let x = input dt c in
          let y, _ = Batch_norm.apply ~eps:c.eps p s ~training:false x in
          let y32, _ =
            Batch_norm.apply ~eps:c.eps
              { gamma = up p.gamma; beta = Option.map up p.beta }
              { mean = up s.mean; var = up s.var }
              ~training:false (up x)
          in
          equal ~msg (array float_exact) (values (Nx.cast dt y32)) (values y)))

let eval_no_row () =
  let p, s = Batch_norm.init ~features:3 Nx.Host.on in
  let y, s' =
    Batch_norm.apply p s ~training:false (Nx.zeros Nx.float32 [| 0; 3 |])
  in
  equal ~msg:"y"
    (pair (array int) (array float_exact))
    ([| 0; 3 |], [||])
    (Nx.shape y, values y);
  satisfies ~msg:"stats'" ~claim:"stats itself" (same_stats ())
    (fun s' -> s' == s)
    s'

let defaults () =
  let p =
    {
      Batch_norm.gamma = vec Nx.float32 [| 1.5; -0.5; 2. |];
      beta = Some (vec Nx.float32 [| 0.25; 1.; -1. |]);
    }
  and s =
    {
      Batch_norm.Stats.mean = vec Nx.float32 [| 0.5; 0.; -1. |];
      var = vec Nx.float32 [| 1e-4; 2.; 0.5 |];
    }
  and x = f32 [| 2; 3 |] [| 1.; -2.; 3.; 0.5; 4.; -1. |] in
  let run ?momentum ?eps training =
    let y, s' = Batch_norm.apply ?momentum ?eps p s ~training x in
    (values y, leaves (stats ()) s')
  in
  let t = pair (array float_exact) leaves_t in
  equal ~msg:"training" t (run ~momentum:0.99 ~eps:1e-5 true) (run true);
  equal ~msg:"eval" t (run ~eps:1e-5 false) (run false)

(* Each row changes one argument of a valid call: [x] of shape [[| 2; 3 |]] for
   3 features, momentum 0.9, eps 1e-5. *)
let refusals =
  [
    ("x of rank 1", [| 3 |], 0.9, 1e-5, "input must have at least 2 axes");
    ("x of rank 0", [||], 0.9, 1e-5, "input must have at least 2 axes");
    ("a last axis of 4", [| 2; 4 |], 0.9, 1e-5, "the last axis has size 4");
    ("momentum -0.1", [| 2; 3 |], -0.1, 1e-5, "momentum must be in [0, 1]");
    ("momentum 1.1", [| 2; 3 |], 1.1, 1e-5, "momentum must be in [0, 1]");
    ( "momentum inf",
      [| 2; 3 |],
      Float.infinity,
      1e-5,
      "momentum must be in [0, 1]" );
    ("momentum nan", [| 2; 3 |], Float.nan, 1e-5, "momentum must be in [0, 1]");
    ("eps nan", [| 2; 3 |], 0.9, Float.nan, "eps must be positive");
    ("eps 0", [| 2; 3 |], 0.9, 0., "eps must be positive");
    ("eps -0.", [| 2; 3 |], 0.9, -0., "eps must be positive");
    ("eps -1", [| 2; 3 |], 0.9, -1., "eps must be positive");
    ( "a training batch of no row",
      [| 0; 3 |],
      0.9,
      1e-5,
      "a training batch has no example" );
    ( "a training batch of no element",
      [| 2; 0; 3 |],
      0.9,
      1e-5,
      "a training batch has no example" );
  ]

let apply_refuses =
  cases
    "raises if x has fewer than 2 axes, its last axis is not features, \
     momentum is outside [0, 1], eps is not positive, or a training x has no \
     element"
    ~id:(fun (id, _, _, _, _) -> id)
    refusals
    (fun (_, shape, momentum, eps, sub) ->
      let p, s = Batch_norm.init ~features:3 Nx.Host.on in
      let x = Nx.zeros Nx.float32 shape in
      raises_match
        (Exn.invalid_arg ~sub:("Batch_norm.apply: " ^ sub))
        (fun () -> Batch_norm.apply ~momentum ~eps p s ~training:true x))

let () =
  exit
    (run "kaun batch norm"
       [
         group "walk"
           [
             walk_visits;
             test "a checkpoint of one shift does not load into the other"
               checkpoint_refuses_shift;
             test "Nx.Ptree.cast converts the parameters' precision" params_cast;
           ];
         group "Stats"
           [
             test "walk visits mean, then var" stats_walk;
             test "Nx.Ptree.cast converts the statistics' precision" stats_cast;
             test
               "statistics are saved beside the parameters, under their own \
                field"
               stats_saved;
           ];
         group "init"
           [
             test
               "gamma is ones, beta zeros, mean zeros and var ones, at the \
                placement"
               init_values;
             test "~shift:false sets beta to None" init_without_shift;
             init_refuses_features;
           ];
         group "apply"
           [
             training_standardizes;
             training_stats;
             test "a constant feature maps to beta" constant_feature;
             step_returns_stats;
             gradients;
             eval_affine;
             eval_island;
             test
               "with training = false, an x of no row gives an empty y and \
                stats' == stats"
               eval_no_row;
             test "momentum defaults to 0.99 and eps to 1e-5" defaults;
             apply_refuses;
           ];
       ])
