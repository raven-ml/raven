(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let narrow =
  Nx.
    [
      Dtype float16;
      Dtype bfloat16;
      Dtype float8_e4m3fn;
      Dtype float8_e5m2;
      Dtype float4_e2m1fn;
    ]

let dtypes = Dtype Nx.float64 :: Dtype Nx.float32 :: narrow
let dtype_name (Dtype dt) = Nx.Dtype.name dt
let floats t = Nx.to_array (Nx.cast Nx.float64 t)
let of_floats dt shape xs = Nx.cast dt (Nx.create Nx.float64 shape xs)
let vec xs = Nx.create Nx.float32 [| Array.length xs |] xs

(* A tensor as its shape and its elements' bits. *)
let value = pair (array int) (array float_exact)
let value_of t = (Nx.shape t, floats t)
let rms_norm64 = Nx.Ptree.instantiate (module Rms_norm)
let grads_ok = function Ok () -> () | Error m -> fail m

(* Two software devices, minted when a test first needs them. *)
let two =
  lazy
    (Nx.devices
       (List.map
          (fun n -> Result.get_ok (Rig.software n))
          [ "kaun-rms-norm-0"; "kaun-rms-norm-1" ]))

(* A witness for placements of one set: each set has its own brand. *)
let placement () =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)

(* A layer and an input at one dtype: [x] of [batch] vectors of [dim] features,
   a quarter of them constant, holding NaN, infinities, signed zeros and 300.,
   whose square overflows float16. *)

type case = {
  dtype : dtype;
  batch : int array;
  dim : int;
  x : float array;
  gamma : float array;
  eps : float option;
}

let pp_case ppf c =
  let pp_floats =
    Format.(pp_print_array ~pp_sep:pp_print_space pp_print_float)
  in
  Format.fprintf ppf "@[<v>%s, batch [%a], dim %d, eps %a@,x [%a]@,gamma [%a]@]"
    (dtype_name c.dtype)
    Format.(pp_print_array ~pp_sep:pp_print_space pp_print_int)
    c.batch c.dim
    (Format.pp_print_option
       ~none:(fun ppf () -> Format.pp_print_string ppf "None")
       Format.pp_print_float)
    c.eps pp_floats c.x pp_floats c.gamma

let element =
  Gen.(
    frequency
      [
        (8, float_range (-8.) 8.);
        (1, of_list [ nan; infinity; neg_infinity; -0.; 0.; 300. ]);
      ])

let row dim =
  Gen.(
    frequency
      [
        (3, array ~size:(constant dim) element);
        (1, map (Array.make dim) element);
      ])

let case dtypes =
  let open Gen in
  with_pp pp_case
  @@ let* dtype = of_list dtypes
     and+ batch = array ~size:(int_range 0 2) (int_range 0 3)
     and+ dim = int_range 1 9
     and+ eps =
       frequency
         [
           (2, constant None);
           (1, constant (Some 0.));
           (2, map Option.some (float_range 0. 1.));
         ]
     in
     let rows = Array.fold_left ( * ) 1 batch in
     let+ x = map Array.concat (list ~size:(constant rows) (row dim))
     and+ gamma = array ~size:(constant dim) (float_range (-4.) 4.) in
     { dtype; batch; dim; x; gamma; eps }

let layer_of dt c = { Rms_norm.gamma = of_floats dt [| c.dim |] c.gamma }
let input_of dt c = of_floats dt (Array.append c.batch [| c.dim |]) c.x

let zero_vector c =
  let rows = Array.length c.x / c.dim in
  List.exists
    (fun r -> Array.for_all (fun v -> v = 0.) (Array.sub c.x (r * c.dim) c.dim))
    (List.init rows Fun.id)

let rms_norm_examples =
  [
    (* A mean square of 1e-6, where eps 1e-6 and 1e-5 give other results. *)
    {
      dtype = Dtype Nx.float32;
      batch = [| 1 |];
      dim = 2;
      x = [| 0.001; -0.001 |];
      gamma = [| 1.; 2. |];
      eps = None;
    };
    (* Zero vectors, of both signs, at eps 0 and at the default. *)
    {
      dtype = Dtype Nx.float32;
      batch = [| 2 |];
      dim = 2;
      x = [| 0.; -0.; -0.; -0. |];
      gamma = [| 1.; -1. |];
      eps = Some 0.;
    };
    {
      dtype = Dtype Nx.bfloat16;
      batch = [| 2 |];
      dim = 2;
      x = [| 0.; -0.; nan; 1. |];
      gamma = [| 1.; -1. |];
      eps = None;
    };
  ]

let law_rms_norm c =
  let (Dtype dt) = c.dtype in
  let p = layer_of dt c and x = input_of dt c in
  let eps = Option.value c.eps ~default:1e-6 in
  List.iter (fun d -> cover ("at " ^ dtype_name d) (d == c.dtype)) dtypes;
  cover "one feature" (c.dim = 1);
  cover "a zero vector" (zero_vector c);
  cover "a NaN" (Array.exists Float.is_nan c.x);
  cover "an infinity" (Array.exists (fun v -> Float.abs v = infinity) c.x);
  cover "a -0." (Array.exists (fun v -> 1. /. v = neg_infinity) c.x);
  cover "eps 0" (c.eps = Some 0.);
  cover "the default eps" (c.eps = None);
  equal value
    (value_of (Nx.rms_norm ~eps ~scale:p.gamma x))
    (value_of (Rms_norm.apply ?eps:c.eps p x))

(* Squares that overflow float16 and starve bfloat16: the mean square is taken
   at float32. *)
let island_examples =
  List.map
    (fun dtype ->
      {
        dtype;
        batch = [| 2 |];
        dim = 4;
        x = [| 300.0; -200.0; 100.0; 250.0; 0.5; -0.25; 1.0; 0.75 |];
        gamma = [| 1.0; 2.0; 0.5; 1.0 |];
        eps = None;
      })
    Nx.[ Dtype float16; Dtype bfloat16 ]

let law_float32_island c =
  let (Dtype dt) = c.dtype in
  let p = layer_of dt c and x = input_of dt c in
  let p32 = Nx.Ptree.cast (module Rms_norm) Nx.float32 p in
  List.iter (fun d -> cover ("at " ^ dtype_name d) (d == c.dtype)) narrow;
  equal value
    (value_of
       (Nx.cast dt (Rms_norm.apply ?eps:c.eps p32 (Nx.cast Nx.float32 x))))
    (value_of (Rms_norm.apply ?eps:c.eps p x))

(* make *)

let test_ones () =
  equal value
    ([| 3 |], [| 1.; 1.; 1. |])
    (value_of (Rms_norm.make ~dim:3 Nx.float64 Nx.Host.on).gamma)

let test_placement () =
  let (module Two) = Lazy.force two in
  let cut = Two.split ~axis:0 in
  equal (placement ()) cut (Nx.placement (Rms_norm.init ~dim:4 cut).gamma)

let test_make_refuses =
  cases ~id:string_of_int "raises if dim is not positive" [ 0; -1 ] (fun dim ->
      raises_match (Exn.invalid_arg ~sub:"Rms_norm.make") (fun () ->
          Rms_norm.make ~dim Nx.float32 Nx.Host.on))

(* init *)

let test_init () =
  equal value
    (value_of (Rms_norm.make ~dim:3 Nx.float32 Nx.Host.on).gamma)
    (value_of (Rms_norm.init ~dim:3 Nx.Host.on).gamma)

(* apply *)

(* Each row's mean square plus eps is 4 or 16, so every step is exact. *)
let test_formula =
  cases
    ~id:(fun (id, _, _, _) -> id)
    "is x / sqrt (mean (x * x) + eps) * gamma along the last axis"
    [
      ("mean square 16", 0., [| 4.; -4. |], [| 2.; -0.5 |]);
      ("mean square 2, eps 2", 2., [| 0.; 2. |], [| 0.; 0.5 |]);
      ("a constant vector is not centered", 0., [| -4.; -4. |], [| -2.; -0.5 |]);
    ]
    (fun (_, eps, x, expected) ->
      let p = { Rms_norm.gamma = vec [| 2.; 0.5 |] } in
      equal value
        ([| 1; 2 |], expected)
        (value_of (Rms_norm.apply ~eps p (Nx.create Nx.float32 [| 1; 2 |] x))))

let test_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 13) @@ fun () ->
  let gamma = Nx.add_s (Nx.mul_s (Nx.Rng.normal Nx.float64 [| 5 |]) 0.3) 1. in
  let p = { Rms_norm.gamma } in
  let x = Nx.Rng.normal Nx.float64 [| 3; 5 |] in
  let w = Nx.Rng.normal Nx.float64 [| 3; 5 |] in
  let loss x p = Nx.sum (Nx.mul w (Rms_norm.apply p x)) in
  grads_ok (Rune.check_grads rms_norm64 (loss x) p);
  grads_ok (Rune.check_grads Nx.Ptree.tensor (fun x -> loss x p) x)

let test_apply_refuses =
  cases
    ~id:(fun (id, _, _) -> id)
    "raises if x is a scalar, its last axis is not dim, or eps is negative or \
     NaN"
    [
      ("a scalar", [||], 1e-6);
      ("last axis 3", [| 2; 3 |], 1e-6);
      ("eps -1", [| 2; 4 |], -1.);
      ("eps NaN", [| 2; 4 |], nan);
    ]
    (fun (_, shape, eps) ->
      let p = Rms_norm.init ~dim:4 Nx.Host.on in
      raises_match (Exn.invalid_arg ~sub:"Rms_norm.apply") (fun () ->
          Rms_norm.apply ~eps p (Nx.zeros Nx.float32 shape)))

(* walk *)

let test_visits () =
  equal (list string) [ "gamma: a leaf" ]
    (List.map
       (Format.asprintf "%a" Nx.Ptree.pp_visit)
       (Nx.Ptree.visits rms_norm64 (Rms_norm.make ~dim:2 Nx.float64 Nx.Host.on)))

let test_cast () =
  let p = Rms_norm.init ~dim:2 Nx.Host.on in
  equal string "bfloat16"
    (Nx.Dtype.name
       (Nx.dtype (Nx.Ptree.cast (module Rms_norm) Nx.bfloat16 p).gamma))

let () =
  exit
    (run "kaun rms norm"
       [
         group "make"
           [
             test "gamma is [dim] ones" test_ones;
             test "gamma lies at the placement" test_placement;
             test_make_refuses;
           ];
         group "init" [ test "is make at float32" test_init ];
         group "apply"
           [
             prop ~count:200 ~examples:rms_norm_examples
               "is Nx.rms_norm ~eps ~scale:gamma x, eps defaulting to 1e-6"
               (case dtypes) law_rms_norm;
             prop ~count:200 ~examples:island_examples
               "a narrow x is normalized and rescaled at float32 and rounded \
                once"
               (case narrow) law_float32_island;
             test_formula;
             test "gradients in gamma and x agree with central differences"
               test_gradients;
             test_apply_refuses;
           ];
         group "walk"
           [
             test "visits gamma" test_visits;
             test "Nx.Ptree.cast converts gamma" test_cast;
           ];
       ])
