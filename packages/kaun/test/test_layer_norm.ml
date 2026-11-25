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
let layer_norm64 = Nx.Ptree.instantiate (module Layer_norm)
let grads_ok = function Ok () -> () | Error m -> fail m

let visits s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

(* Two software devices, minted when a test first needs them. *)
let two =
  lazy
    (Nx.devices
       (List.map
          (fun n -> Result.get_ok (Rig.software n))
          [ "kaun-layer-norm-0"; "kaun-layer-norm-1" ]))

(* A witness for placements of one set: each set has its own brand. *)
let placement () =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)

(* A layer and an input at one dtype: [x] of [batch] vectors of [dim] features,
   a quarter of them constant, holding NaN, infinities, signed zeros and 1000.1,
   whose float32 mean is inexact. *)

type case = {
  dtype : dtype;
  batch : int array;
  dim : int;
  x : float array;
  gamma : float array;
  beta : float array option;
  eps : float option;
}

let pp_case ppf c =
  let pp_floats =
    Format.(pp_print_array ~pp_sep:pp_print_space pp_print_float)
  in
  let pp_option pp =
    Format.pp_print_option
      ~none:(fun ppf () -> Format.pp_print_string ppf "None")
      pp
  in
  Format.fprintf ppf
    "@[<v>%s, batch [%a], dim %d, eps %a@,x [%a]@,gamma [%a]@,beta %a@]"
    (dtype_name c.dtype)
    Format.(pp_print_array ~pp_sep:pp_print_space pp_print_int)
    c.batch c.dim
    (pp_option Format.pp_print_float)
    c.eps pp_floats c.x pp_floats c.gamma
    (pp_option (fun ppf b -> Format.fprintf ppf "[%a]" pp_floats b))
    c.beta

let element =
  Gen.(
    frequency
      [
        (8, float_range (-8.) 8.);
        (1, of_list [ nan; infinity; neg_infinity; -0.; 0.; 1000.1 ]);
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
     and+ shift = bool in
     let rows = Array.fold_left ( * ) 1 batch in
     let+ x = map Array.concat (list ~size:(constant rows) (row dim))
     and+ gamma = array ~size:(constant dim) (float_range (-4.) 4.)
     and+ beta = array ~size:(constant dim) (float_range (-4.) 4.) in
     {
       dtype;
       batch;
       dim;
       x;
       gamma;
       beta = (if shift then Some beta else None);
       eps;
     }

let layer_of dt c =
  {
    Layer_norm.gamma = of_floats dt [| c.dim |] c.gamma;
    beta = Option.map (of_floats dt [| c.dim |]) c.beta;
  }

let input_of dt c = of_floats dt (Array.append c.batch [| c.dim |]) c.x

let constant_row c =
  let rows = Array.length c.x / c.dim in
  c.dim >= 2
  && List.exists
       (fun r ->
         Array.for_all
           (Float.equal c.x.(r * c.dim))
           (Array.sub c.x (r * c.dim) c.dim))
       (List.init rows Fun.id)

(* Without a shift, [apply] adds no zero that would turn a [-0.] into [+0.].
   [gamma] alternates sign, so a constant lane gives both zeros. *)
let shift_free dtype ~dim xs =
  {
    dtype;
    batch = [| Array.length xs / dim |];
    dim;
    x = xs;
    gamma = Array.init dim (fun i -> if i mod 2 = 0 then -1.5 else 0.75);
    beta = None;
    eps = None;
  }

let standardize_examples =
  let wide = (1 lsl 17) + 1 in
  List.concat_map
    (fun dtype ->
      [
        shift_free dtype ~dim:1 [| 3.; -2. |];
        shift_free dtype ~dim:2 [| 0.; 2.; -1.; 7. |];
        shift_free dtype ~dim:4 (Array.make 8 1000.1);
        shift_free dtype ~dim:3 [| 1.; nan; 2.; 4.; 5.; 6. |];
        shift_free dtype ~dim:2 [| -0.; -0.; 0.; -0. |];
        shift_free dtype ~dim:wide
          (Array.init wide (fun i -> Float.of_int (i mod 13)));
      ])
    Nx.[ Dtype float16; Dtype bfloat16; Dtype float32; Dtype float64 ]
  @ [
      {
        dtype = Dtype Nx.float32;
        batch = [| 2 |];
        dim = 3;
        x = [| 1.; infinity; 2.; neg_infinity; 0.; 1. |];
        gamma = [| 1.; 2.; 3. |];
        beta = Some [| 0.; 1.; 2. |];
        eps = Some 0.;
      };
      (* A variance of 1e-6, where eps 1e-5 and 1e-6 give other results. *)
      {
        dtype = Dtype Nx.float32;
        batch = [| 1 |];
        dim = 2;
        x = [| 0.; 0.002 |];
        gamma = [| 1.; 1. |];
        beta = Some [| 0.5; -0.5 |];
        eps = None;
      };
    ]

let law_standardize c =
  let (Dtype dt) = c.dtype in
  let p = layer_of dt c and x = input_of dt c in
  let eps = Option.value c.eps ~default:1e-5 in
  List.iter (fun d -> cover ("at " ^ dtype_name d) (d == c.dtype)) dtypes;
  cover "one feature" (c.dim = 1);
  cover "a constant vector" (constant_row c);
  cover "a NaN" (Array.exists Float.is_nan c.x);
  cover "an infinity" (Array.exists (fun v -> Float.abs v = infinity) c.x);
  cover "a -0." (Array.exists (fun v -> 1. /. v = neg_infinity) c.x);
  cover "no shift" (c.beta = None);
  cover "eps 0" (c.eps = Some 0.);
  cover "the default eps" (c.eps = None);
  equal value
    (value_of (Nx.standardize ~eps ~scale:p.gamma ?shift:p.beta x))
    (value_of (Layer_norm.apply ?eps:c.eps p x))

(* A large common offset: centred at float32, the offsets survive half
   precision. bfloat16 spaces by 0.5 near 100, so they round on input. *)
let island_examples =
  List.map
    (fun dtype ->
      {
        dtype;
        batch = [| 2 |];
        dim = 4;
        x = [| 100.0; 100.25; 99.75; 100.5; -50.0; -50.25; -49.75; -50.5 |];
        gamma = [| 1.0; 2.0; 0.5; 1.0 |];
        beta = Some [| 0.25; 0.0; -0.25; 0.0 |];
        eps = None;
      })
    Nx.[ Dtype float16; Dtype bfloat16 ]

let law_float32_island c =
  let (Dtype dt) = c.dtype in
  let p = layer_of dt c and x = input_of dt c in
  let p32 = Nx.Ptree.cast (module Layer_norm) Nx.float32 p in
  List.iter (fun d -> cover ("at " ^ dtype_name d) (d == c.dtype)) narrow;
  equal value
    (value_of
       (Nx.cast dt (Layer_norm.apply ?eps:c.eps p32 (Nx.cast Nx.float32 x))))
    (value_of (Layer_norm.apply ?eps:c.eps p x))

(* t and make *)

let test_identity () =
  let p = Layer_norm.make ~dim:3 Nx.float64 Nx.Host.on in
  equal
    (pair value (option value))
    (([| 3 |], [| 1.; 1.; 1. |]), Some ([| 3 |], [| 0.; 0.; 0. |]))
    (value_of p.gamma, Option.map value_of p.beta)

let test_no_shift () =
  is_none (Layer_norm.make ~shift:false ~dim:3 Nx.float32 Nx.Host.on).beta

let test_placement () =
  let (module Two) = Lazy.force two in
  let cut = Two.split ~axis:0 in
  let p = Layer_norm.init ~dim:4 cut in
  equal
    (pair (placement ()) (option (placement ())))
    (cut, Some cut)
    (Nx.placement p.gamma, Option.map Nx.placement p.beta)

let test_make_refuses =
  cases ~id:string_of_int "raises if dim is not positive" [ 0; -1 ] (fun dim ->
      raises_match (Exn.invalid_arg ~sub:"Layer_norm.make") (fun () ->
          Layer_norm.make ~dim Nx.float32 Nx.Host.on))

(* init *)

let test_init =
  cases ~id:string_of_bool "is make at float32" [ true; false ] (fun shift ->
      let p = Layer_norm.init ~shift ~dim:3 Nx.Host.on in
      let q = Layer_norm.make ~shift ~dim:3 Nx.float32 Nx.Host.on in
      equal
        (pair value (option value))
        (value_of q.gamma, Option.map value_of q.beta)
        (value_of p.gamma, Option.map value_of p.beta))

(* apply *)

(* Each row has variance 1 or 4 and eps 0 or 3, so every step is exact. *)
let test_formula =
  cases
    ~id:(fun (id, _, _, _) -> id)
    "is (x - mean) / sqrt (var + eps) * gamma + beta along the last axis"
    [
      ("mean 1, variance 1", 0., [| 0.; 2. |], [| -1.; 2. |]);
      ("mean 12, variance 4", 0., [| 10.; 14. |], [| -1.; 2. |]);
      ("variance 1, eps 3", 3., [| 0.; 2. |], [| 0.; 0.5 |]);
    ]
    (fun (_, eps, x, expected) ->
      let p =
        { Layer_norm.gamma = vec [| 2.; 3. |]; beta = Some (vec [| 1.; -1. |]) }
      in
      equal value
        ([| 1; 2 |], expected)
        (value_of (Layer_norm.apply ~eps p (Nx.create Nx.float32 [| 1; 2 |] x))))

(* 777f1e4f61: 1000.1 has no exact float32 mean over 768 terms. *)
let constant_vectors =
  [
    ("768 features of 5", 768, 5.);
    ("768 features of 1000.1", 768, 1000.1);
    ("768 features of -0.3", 768, -0.3);
    ("2^17 features of 1000.1", 1 lsl 17, 1000.1);
  ]

let test_constant_to_beta =
  cases
    ~id:(fun (id, _, _) -> id)
    "a constant vector maps to beta" constant_vectors
    (fun (_, dim, v) ->
      let beta = Array.init dim (fun i -> Float.of_int (i mod 7) -. 3.) in
      let p =
        {
          Layer_norm.gamma = Nx.full Nx.float32 [| dim |] 1.5;
          beta = Some (vec beta);
        }
      in
      equal value
        ([| 2; dim |], Array.append beta beta)
        (value_of (Layer_norm.apply p (Nx.full Nx.float32 [| 2; dim |] v))))

let test_constant_to_zero =
  cases
    ~id:(fun (id, _, _) -> id)
    "a constant vector maps to zero without a shift" constant_vectors
    (fun (_, dim, v) ->
      let p =
        { Layer_norm.gamma = Nx.full Nx.float32 [| dim |] 1.5; beta = None }
      in
      equal value
        ([| 2; dim |], Array.make (2 * dim) 0.)
        (value_of (Layer_norm.apply p (Nx.full Nx.float32 [| 2; dim |] v))))

let test_gradients () =
  Nx.Rng.with_key (Nx.Rng.key 11) @@ fun () ->
  let gamma = Nx.Rng.normal Nx.float64 [| 4 |] in
  let p =
    { Layer_norm.gamma; beta = Some (Nx.Rng.normal Nx.float64 [| 4 |]) }
  in
  let no_shift = { p with beta = None } in
  let x = Nx.Rng.normal Nx.float64 [| 3; 4 |] in
  let loss x p =
    let y = Layer_norm.apply p x in
    Nx.sum (Nx.mul y y)
  in
  grads_ok (Rune.check_grads layer_norm64 (loss x) p);
  grads_ok (Rune.check_grads layer_norm64 (loss x) no_shift);
  grads_ok (Rune.check_grads Nx.Ptree.tensor (fun x -> loss x p) x)

let test_gradient_no_shift () =
  let p =
    {
      Layer_norm.gamma = Nx.create Nx.float64 [| 2 |] [| 2.; -1. |];
      beta = None;
    }
  in
  let x = Nx.create Nx.float64 [| 1; 2 |] [| 0.; 3. |] in
  is_none
    (Rune.grad layer_norm64 (fun p -> Nx.sum (Layer_norm.apply p x)) p).beta

let test_apply_refuses =
  cases
    ~id:(fun (id, _, _) -> id)
    "raises if x is a scalar, its last axis is not dim, or eps is negative or \
     NaN"
    [
      ("a scalar", [||], 1e-5);
      ("last axis 3", [| 2; 3 |], 1e-5);
      ("eps -1", [| 2; 4 |], -1.);
      ("eps NaN", [| 2; 4 |], nan);
    ]
    (fun (_, shape, eps) ->
      let p = Layer_norm.init ~dim:4 Nx.Host.on in
      raises_match (Exn.invalid_arg ~sub:"Layer_norm.apply") (fun () ->
          Layer_norm.apply ~eps p (Nx.zeros Nx.float32 shape)))

(* walk *)

let test_visits =
  cases ~id:fst "visits gamma, then beta, reporting whether beta is present"
    [
      ("with a shift", (true, [ "gamma: a leaf"; "beta: Some"; "beta: a leaf" ]));
      ("without a shift", (false, [ "gamma: a leaf"; "beta: None" ]));
    ]
    (fun (_, (shift, expected)) ->
      equal (list string) expected
        (visits layer_norm64
           (Layer_norm.make ~shift ~dim:2 Nx.float64 Nx.Host.on)))

(* [Nx_io.Archive.to_value] raises naming an entry [like] owns that no array
   names, or one [like] names that the file lacks. *)
let test_checkpoints () =
  let s = Nx.Ptree.instantiate (module Layer_norm) in
  let save x = Nx_io.Archive.of_value ~prefix:"n" s x in
  let load ~like a = Nx_io.Archive.to_value ~prefix:"n" s ~like a in
  let shifted = Layer_norm.make ~dim:2 Nx.float32 Nx.Host.on in
  let plain = { Layer_norm.gamma = vec [| 0.5; -3. |]; beta = None } in
  let like = { plain with gamma = Nx.zeros Nx.float32 [| 2 |] } in
  equal ~msg:"a shift-free file into a shift-free layer" value
    (value_of plain.gamma)
    (value_of (load ~like (save plain)).gamma);
  raises_match ~msg:"a shifted file into a shift-free layer"
    (Exn.failure ~sub:"n.beta") (fun () -> load ~like:plain (save shifted));
  raises_match ~msg:"a shift-free file into a shifted layer"
    (Exn.failure ~sub:"n.beta") (fun () -> load ~like:shifted (save plain))

let test_cast () =
  let p = Layer_norm.init ~dim:2 Nx.Host.on in
  equal (list string) [ "bfloat16"; "bfloat16" ]
    (Nx.Ptree.fold
       (Nx.Ptree.instantiate (module Layer_norm))
       (fun _ x acc -> Nx.Dtype.name (Nx.dtype x) :: acc)
       (Nx.Ptree.cast (module Layer_norm) Nx.bfloat16 p)
       [])

let () =
  exit
    (run "kaun layer norm"
       [
         group "t"
           [
             test "beta is None for a layer made without a shift" test_no_shift;
           ];
         group "make"
           [
             test "is the identity normalization, gamma ones and beta zeros"
               test_identity;
             test "the parameters lie at the placement" test_placement;
             test_make_refuses;
           ];
         group "init" [ test_init ];
         group "apply"
           [
             prop ~count:200 ~examples:standardize_examples
               "is Nx.standardize ~eps ~scale:gamma ?shift:beta x, eps \
                defaulting to 1e-5"
               (case dtypes) law_standardize;
             prop ~count:200 ~examples:island_examples
               "a narrow x is normalized and rescaled at float32 and rounded \
                once"
               (case narrow) law_float32_island;
             test_formula;
             test_constant_to_beta;
             test_constant_to_zero;
             test
               "gradients in the parameters and x agree with central \
                differences"
               test_gradients;
             test "without a shift, the gradient has no shift"
               test_gradient_no_shift;
             test_apply_refuses;
           ];
         group "walk"
           [
             test_visits;
             test "a checkpoint of one shift does not load into the other"
               test_checkpoints;
             test "Nx.Ptree.cast converts every leaf" test_cast;
           ];
       ])
