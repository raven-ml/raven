(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Norn.Bij

let f64 = Nx.scalar Nx.float64
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let floats x = Nx.to_array (Nx.cast Nx.float64 x)

let pp_floats ppf xs =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%.17g") xs)))

(* A bijector under test: its coordinates' length, whether a value lies in its
   open support at the precision norn.mli states for the dtype of epsilon [eps],
   and its free components, the ones that determine the rest. *)
type 'f bijector = {
  name : string;
  b : ('f, Nx.host) B.t;
  dim : int;
  inside : eps:float -> float array -> bool;
  free :
    (float, Nx.float64_elt, Nx.host) Nx.t ->
    (float, Nx.float64_elt, Nx.host) Nx.t;
}

let low = 1.5
and high = 4.25

let all p xs = Array.for_all p xs
let finite ~eps:_ xs = all Float.is_finite xs
let positive ~eps:_ xs = all (fun x -> x > 0. && Float.is_finite x) xs

let increasing ~eps:_ xs =
  all Float.is_finite xs
  && Array.for_all Fun.id
       (Array.init (Array.length xs - 1) (fun i -> xs.(i + 1) > xs.(i)))

(* A simplex of [K] components sums to one within [4 K ε]. *)
let on_simplex ~eps xs =
  let k = float_of_int (Array.length xs) in
  all (fun x -> x > 0. && x <= 1.) xs
  && Float.abs (Array.fold_left ( +. ) 0. xs -. 1.) <= 4. *. k *. eps

(* A sum-to-zero vector of [K] components sums to zero within [4 K ε] times its
   largest magnitude. *)
let sums_to_zero ~eps xs =
  let k = float_of_int (Array.length xs) in
  let m = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. xs in
  all Float.is_finite xs
  && Float.abs (Array.fold_left ( +. ) 0. xs) <= 4. *. k *. eps *. m

(* A lower-triangular [n × n] matrix of positive diagonal whose rows' squared
   norms are within [4 n ε] of one. *)
let unit_rows n ~eps xs =
  let row i =
    let s = ref 0. and upper = ref false in
    for j = 0 to n - 1 do
      let v = xs.((i * n) + j) in
      if j > i && v <> 0. then upper := true;
      s := !s +. (v *. v)
    done;
    (not !upper)
    && xs.((i * n) + i) > 0.
    && Float.abs (!s -. 1.) <= 4. *. float_of_int n *. eps
  in
  List.for_all row (List.init n Fun.id)

(* The strict lower triangle of a 4 × 4 factor, row by row. *)
let lower x =
  Nx.take
    (Nx.create Nx.int64 [| 6 |] [| 4L; 8L; 9L; 12L; 13L; 14L |])
    (Nx.reshape [| 16 |] x)

let first k x = Nx.slice [ Nx.R (0, k) ] x

let bijectors (type f) (dt : (float, f) Nx.dtype) : f bijector list =
  let c = Nx.scalar dt and v xs = Nx.create dt [| Array.length xs |] xs in
  [
    {
      name = "identity";
      b = B.identity;
      dim = 3;
      inside = finite;
      free = Fun.id;
    };
    { name = "exp"; b = B.exp; dim = 3; inside = positive; free = Fun.id };
    {
      name = "greater";
      b = B.greater ~low:(c low);
      dim = 3;
      inside = (fun ~eps:_ -> all (fun x -> x > low && Float.is_finite x));
      free = Fun.id;
    };
    {
      name = "less";
      b = B.less ~high:(c high);
      dim = 3;
      inside = (fun ~eps:_ -> all (fun x -> x < high && Float.is_finite x));
      free = Fun.id;
    };
    {
      name = "interval";
      b = B.interval ~low:(c low) ~high:(c high);
      dim = 3;
      inside = (fun ~eps:_ -> all (fun x -> x > low && x < high));
      free = Fun.id;
    };
    {
      name = "affine";
      b = B.affine ~loc:(v [| 1.; -2.; 0.5 |]) ~scale:(v [| 2.; 0.25; -3. |]);
      dim = 3;
      inside = finite;
      free = Fun.id;
    };
    {
      name = "affine_tril";
      b =
        B.affine_tril
          ~loc:(v [| 1.; -2.; 0.5 |])
          ~scale_tril:
            (Nx.create dt [| 3; 3 |]
               [| 2.; 9.; 9.; 0.5; 1.5; 9.; -1.; 0.25; 0.75 |]);
      dim = 3;
      inside = finite;
      free = Fun.id;
    };
    {
      name = "simplex";
      b = B.simplex;
      dim = 3;
      inside = on_simplex;
      free = first 3;
    };
    {
      name = "ordered";
      b = B.ordered;
      dim = 4;
      inside = increasing;
      free = Fun.id;
    };
    {
      name = "sum_to_zero";
      b = B.sum_to_zero;
      dim = 3;
      inside = sums_to_zero;
      free = first 3;
    };
    {
      name = "cholesky_corr";
      b = B.cholesky_corr;
      dim = 6;
      inside = unit_rows 4;
      free = lower;
    };
    {
      name = "compose cholesky_corr affine";
      b = B.compose B.cholesky_corr (B.affine ~loc:(c 0.25) ~scale:(c 0.5));
      dim = 6;
      inside = unit_rows 4;
      free = lower;
    };
    {
      name = "compose exp affine";
      b = B.compose B.exp (B.affine ~loc:(c 0.5) ~scale:(c 2.));
      dim = 3;
      inside = positive;
      free = Fun.id;
    };
  ]

(* The bijectors, built once at each dtype the laws draw, the same index naming
   the same bijector at both. *)
let bijectors64 = Array.of_list (bijectors Nx.float64)
let bijectors32 = Array.of_list (bijectors Nx.float32)
let count = Array.length bijectors64

let named name =
  Option.get (Array.find_index (fun c -> c.name = name) bijectors64)

let cover_each i = Array.iteri (fun j c -> cover c.name (i = j)) bijectors64

(* A bijector's index and coordinates moderate enough that none saturates. *)
let moderate =
  Gen.(
    let* i = int_range 0 (count - 1) in
    let+ u =
      array ~size:(constant bijectors64.(i).dim) (float_range (-3.) 3.)
    in
    (i, u))

(* Every finite coordinate, the dtype's extremes and the saturation points among
   them. *)
let extreme =
  Gen.frequency
    [
      (4, Gen.float_range (-40.) 40.);
      (2, Gen.float_range (-1e6) 1e6);
      ( 2,
        Gen.of_list ~pp:Format.pp_print_float
          [
            0.;
            -0.;
            709.;
            710.;
            -708.;
            -746.;
            1e300;
            -1e300;
            Float.max_float;
            -.Float.max_float;
            19.;
            37.;
            -37.;
            1e-300;
          ] );
    ]

(* Forward *)

(* Each bijector at each dtype, with one example per pair that saturates: a
   coordinate past every map's overflow. *)
let totality =
  let float32 = Float.of_string "0x1.fffffep127" in
  prop "every finite coordinate lands in the open support" ~count:(26 * 300)
    ~examples:
      (List.concat_map
         (fun i ->
           let u = Array.make bijectors64.(i).dim 0. in
           u.(0) <- Float.max_float;
           [ (i, false, u); (i, true, u) ])
         (List.init count Fun.id))
    Gen.(
      let* i = int_range 0 (count - 1) in
      let* narrow = bool in
      let+ u =
        with_pp pp_floats (array ~size:(constant bijectors64.(i).dim) extreme)
      in
      (i, narrow, u))
    (fun (i, narrow, u) ->
      let at (type f) (dt : (float, f) Nx.dtype) (c : f bijector) huge =
        let eps = (Nx.Dtype.float_format dt).epsilon in
        let clip x = Float.min huge (Float.max (-.huge) x) in
        let x, ld =
          B.forward c.b (Nx.create dt [| c.dim |] (Array.map clip u))
        in
        let ld = floats ld in
        Array.iteri
          (fun j b ->
            if b.name <> "identity" then
              cover (b.name ^ " saturates")
                (i = j && Array.exists (( = ) Float.neg_infinity) ld))
          bijectors64;
        cover (Nx.Dtype.name dt) true;
        satisfies ~msg:c.name ~claim:"in the open support" (array float_exact)
          (c.inside ~eps) (floats x);
        satisfies ~msg:c.name ~claim:"finite or -inf" (array float_exact)
          (Array.for_all (fun l -> Float.is_finite l || l = Float.neg_infinity))
          ld
      in
      cover_each i;
      if narrow then at Nx.float32 bijectors32.(i) float32
      else at Nx.float64 bijectors64.(i) Float.max_float)

(* The Jacobian of the map onto the free components, by forward-mode
   derivatives, and its log-determinant by LU: within [n κ ε] of the exact one,
   [κ] the Jacobian's condition number, below [10^3] for coordinates in [-3, 3]:
   [2^-32] holds [6 · 10^3 ε] with room. *)
let log_det =
  prop "the log-determinant is the Jacobian's over the free components"
    ~count:(13 * 100) moderate (fun (i, u) ->
      let c = bijectors64.(i) in
      cover_each i;
      let u = vec u in
      let _, ld = B.forward c.b u in
      let jac = Rune.jacfwd' (fun u -> c.free (fst (B.forward c.b u))) u in
      equal ~msg:c.name (float 0x1p-32)
        (Nx.item [] (snd (Nx.slogdet Nx.float64 jac)))
        (Nx.item [] (Nx.sum ld)))

let forward =
  group "forward"
    [
      totality;
      log_det;
      test "a matrix of simplex coordinates maps row by row" (fun () ->
          let u = Nx.create Nx.float64 [| 2; 2 |] [| 0.3; -1.; 2.; 0.5 |] in
          let x, ld = B.forward B.simplex u in
          equal (array int) [| 2; 3 |] (Nx.shape x);
          equal (array int) [| 2 |] (Nx.shape ld);
          let x1, ld1 = B.forward B.simplex (Nx.slice [ Nx.I 1 ] u) in
          (* a batch may sum a row in another order: a few roundings *)
          let near = float (8. *. epsilon_float) in
          equal (array near) (floats x1) (floats (Nx.slice [ Nx.I 1 ] x));
          equal near (Nx.item [] ld1) (Nx.item [ 1 ] ld));
      test "a log-determinant has one element per unit" (fun () ->
          let _, ld =
            B.forward B.cholesky_corr (Nx.zeros Nx.float64 [| 5; 3 |])
          in
          equal (array int) [| 5 |] (Nx.shape ld));
      test "a vector bound maps each element by its own" (fun () ->
          let low = vec [| 1.; 2.; 3. |] in
          let u =
            Nx.create Nx.float64 [| 2; 3 |] [| 0.; 1.; -1.; 0.5; 0.; 2. |]
          in
          let x, _ = B.forward (B.greater ~low) u in
          equal (array float_exact) (floats (Nx.add low (Nx.exp u))) (floats x));
    ]

let constructors =
  group "constructors"
    [
      test "exp of 0 is 1 with log-determinant 0" (fun () ->
          let x, ld = B.forward B.exp (f64 0.) in
          equal
            (pair float_exact float_exact)
            (1., 0.)
            (Nx.item [] x, Nx.item [] ld));
      test "exp past its overflow is the largest float, log-determinant -inf"
        (fun () ->
          let x, ld = B.forward B.exp (f64 800.) in
          equal
            (pair float_exact float_exact)
            (Float.max_float, Float.neg_infinity)
            (Nx.item [] x, Nx.item [] ld));
      test "exp past its underflow is the least normal, log-determinant -inf"
        (fun () ->
          let x, ld = B.forward B.exp (f64 (-800.)) in
          equal
            (pair float_exact float_exact)
            (Float.min_float, Float.neg_infinity)
            (Nx.item [] x, Nx.item [] ld));
      test "interval never returns its bound" (fun () ->
          let x, ld =
            B.forward (B.interval ~low:(f64 0.) ~high:(f64 1.)) (f64 50.)
          in
          less float_exact ~than:1. (Nx.item [] x);
          equal float_exact Float.neg_infinity (Nx.item [] ld));
      test "affine_tril reads scale_tril's lower triangle" (fun () ->
          let b upper =
            B.affine_tril
              ~loc:(vec [| 1.; -2. |])
              ~scale_tril:
                (Nx.create Nx.float64 [| 2; 2 |] [| 2.; upper; 0.5; 1.5 |])
          in
          let u = vec [| 0.3; -0.7 |] in
          equal (array float_exact)
            (floats (fst (B.forward (b 0.) u)))
            (floats (fst (B.forward (b 9.) u))));
      test "the simplex of zero coordinates is uniform" (fun () ->
          let x, _ = B.forward B.simplex (Nx.zeros Nx.float64 [| 3 |]) in
          equal
            (array (float epsilon_float))
            [| 0.25; 0.25; 0.25; 0.25 |]
            (floats x));
      (* An orthonormal basis keeps lengths: [‖clr x‖ = ‖u‖] for the simplex,
         [clr] the centred logarithms, and [‖x‖ = ‖u‖] for sum_to_zero, within a
         few roundings per component. *)
      prop "simplex and sum_to_zero keep the length of the coordinates"
        Gen.(pair bool (array ~size:(int_range 1 5) (float_range (-3.) 3.)))
        (fun (simplex, u) ->
          cover "simplex" simplex;
          cover "sum_to_zero" (not simplex);
          cover "one coordinate" (Array.length u = 1);
          let norm xs =
            Float.sqrt (Array.fold_left (fun a x -> a +. (x *. x)) 0. xs)
          in
          let x =
            floats
              (fst
                 (B.forward
                    (if simplex then B.simplex else B.sum_to_zero)
                    (vec u)))
          in
          let v =
            if not simplex then x
            else
              let l = Array.map Float.log x in
              let m =
                Array.fold_left ( +. ) 0. l /. float_of_int (Array.length l)
              in
              Array.map (fun l -> l -. m) l
          in
          let n = float_of_int (Array.length u + 1) in
          equal
            (float (16. *. n *. epsilon_float *. (1. +. norm u)))
            (norm u) (norm v));
      (* [x0 = u0] and [xk = x(k-1) + exp uk], each step a few roundings. *)
      prop "ordered is the first coordinate then sums of exponentials"
        Gen.(array ~size:(int_range 1 5) (float_range (-3.) 3.))
        (fun u ->
          cover "one coordinate" (Array.length u = 1);
          let x = Array.copy u in
          for k = 1 to Array.length u - 1 do
            x.(k) <- x.(k - 1) +. Float.exp u.(k)
          done;
          let scale = Array.fold_left (fun a v -> a +. Float.abs v) 1. x in
          equal
            (array (float (4. *. epsilon_float *. scale)))
            x
            (floats (fst (B.forward B.ordered (vec u)))));
      test "cholesky_corr of zero coordinates is the identity" (fun () ->
          let x, ld = B.forward B.cholesky_corr (Nx.zeros Nx.float64 [| 3 |]) in
          equal (array float_exact) (floats (Nx.eye Nx.float64 3)) (floats x);
          equal float_exact 0. (Nx.item [] ld));
      (* The one partial correlation of a 2 × 2 factor is its correlation, and
         its diagonal [√(1 - z²)] is [1 / cosh u]. The map reads the diagonal
         from [z = tanh u] rounded, whose error of [ε / 2] moves [1 - z²] by [ε]
         relatively over [1 - z² = sech² u]: the diagonal, [sech u], is within
         [ε cosh u] absolutely. *)
      prop "cholesky_corr of one coordinate is [[1, 0], [tanh u, sech u]]"
        ~examples:[ 3.839634256037935 ]
        Gen.(float_range (-5.) 5.)
        (fun u ->
          equal
            (array (float (epsilon_float *. (4. +. Float.cosh u))))
            [| 1.; 0.; Float.tanh u; 1. /. Float.cosh u |]
            (floats (fst (B.forward B.cholesky_corr (vec [| u |])))));
      (* Row by row, coordinate 3 of a 4 × 4 factor is element (3, 0); by
         columns it would be (2, 1). *)
      test "cholesky_corr reads the strict lower triangle row by row" (fun () ->
          let x =
            fst (B.forward B.cholesky_corr (vec [| 0.; 0.; 0.; 0.5; 0.; 0. |]))
          in
          equal
            (pair (float (4. *. epsilon_float)) float_exact)
            (Float.tanh 0.5, 0.)
            (Nx.item [ 3; 0 ] x, Nx.item [ 2; 1 ] x));
      test "compose's log-determinant is on the coarser units" (fun () ->
          let u =
            Nx.create Nx.float64 [| 2; 3 |] [| 0.3; -1.; 2.; 0.5; 0.; -0.2 |]
          in
          let _, ld = B.forward (B.compose B.simplex B.exp) u in
          let _, simplex = B.forward B.simplex (Nx.exp u) in
          equal
            (array (float (8. *. epsilon_float)))
            (floats (Nx.add (Nx.sum ~axes:[ 1 ] u) simplex))
            (floats ld));
    ]

(* Inverse *)

(* At coordinates in [-3, 3] the worst-conditioned inverse is cholesky_corr's:
   it divides a row's element by [√(1 - Σ l²)] of the ones before it, the
   product of [1 - z²] over their partial correlations, each [1 - tanh² 3 ≈
   0.0099] at worst. Three in the last row scale [ε] by [0.0099^-3 ≈ 10^6],
   which [2^-31] holds. *)
let inverse_of_forward =
  prop "undoes forward" ~count:(13 * 100) moderate (fun (i, u) ->
      let c = bijectors64.(i) in
      cover_each i;
      Law.round_trip
        (array (float 0x1p-31))
        pass
        (fun u -> fst (B.forward c.b (vec u)))
        (fun x -> floats (B.inverse c.b x))
        u)

(* A value of the support, as forward makes it from coordinates up to [30], near
   the bounds where interval's inverse reads [high - x]: forward of the inverse
   is the value within [8 ε] of its magnitude and of the bounds [low], [high],
   [loc] and the affine maps' scales, all below [10]. The examples are the
   floats next to interval's bounds. Cholesky factors are left to the law above:
   near unit partial correlations a row's [1 - Σ l²] loses the digits this bound
   would need. *)
let forward_of_inverse =
  let interval = named "interval" in
  let drawn =
    List.filter (fun i -> bijectors64.(i).dim <> 6) (List.init count Fun.id)
  in
  let near_low = Float.succ low and near_high = Float.pred high in
  prop "forward of the inverse is the value" ~count:(13 * 100)
    ~examples:
      [
        (interval, [| near_low; near_high; 2. |]);
        (interval, [| near_high; near_low; near_high |]);
      ]
    Gen.(
      let* i = of_list ~pp:Format.pp_print_int drawn in
      let c = bijectors64.(i) in
      let+ u = array ~size:(constant c.dim) (float_range (-30.) 30.) in
      (i, floats (fst (B.forward c.b (vec u)))))
    (fun (i, x) ->
      let c = bijectors64.(i) in
      List.iter (fun j -> cover bijectors64.(j).name (i = j)) drawn;
      Law.round_trip
        (array
           (float_rel ~rel:(8. *. epsilon_float) ~abs:(80. *. epsilon_float)))
        pass
        (fun x -> B.inverse c.b (vec x))
        (fun u -> floats (fst (B.forward c.b u)))
        x)

let shapes =
  group "shape"
    [
      cases "is the coordinates' shape of a value's"
        ~id:(fun (name, _, _, _) -> name)
        [
          ("exp keeps it", B.exp, [| 2; 3 |], [| 2; 3 |]);
          ( "affine_tril keeps it",
            B.affine_tril
              ~loc:(vec [| 0.; 0. |])
              ~scale_tril:(Nx.eye Nx.float64 2),
            [| 5; 2 |],
            [| 5; 2 |] );
          ("ordered keeps it", B.ordered, [| 4 |], [| 4 |]);
          ( "the simplex has one coordinate fewer per vector",
            B.simplex,
            [| 5; 4 |],
            [| 5; 3 |] );
          ( "sum_to_zero has one coordinate fewer",
            B.sum_to_zero,
            [| 4 |],
            [| 3 |] );
          ( "a Cholesky factor of n × n has n (n - 1) / 2",
            B.cholesky_corr,
            [| 2; 4; 4 |],
            [| 2; 6 |] );
          (* cholesky_corr's [6] of a 4 × 4 factor, then simplex's [5]; the
             other order would read [4; 3] as a matrix and refuse it *)
          ( "compose reads its outer bijector's first",
            B.compose B.cholesky_corr B.simplex,
            [| 4; 4 |],
            [| 5 |] );
        ]
        (fun (_, b, s, expected) -> equal (array int) expected (B.shape b s));
      cases "refuses a shape without a unit"
        ~id:(fun (name, _, _) -> name)
        [
          ("a scalar has no vector", B.simplex, [||]);
          ("an empty vector has no simplex", B.simplex, [| 0 |]);
          ("an empty vector has no order", B.ordered, [| 0 |]);
          ("a rectangle is no Cholesky factor", B.cholesky_corr, [| 3; 4 |]);
        ]
        (fun (_, b, s) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Bij.shape:") (fun () ->
              B.shape b s));
    ]

let pp =
  cases "formats the bijector's name" ~id:fst
    [
      ("exp", B.exp);
      ( "compose(exp, affine)",
        B.compose B.exp (B.affine ~loc:(f64 0.) ~scale:(f64 1.)) );
    ]
    (fun (name, b) -> equal string name (Format.asprintf "%a" B.pp b))

let () =
  exit
    (run "Norn.Bij"
       [
         constructors;
         forward;
         group "inverse" [ inverse_of_forward; forward_of_inverse ];
         shapes;
         group "pp" [ pp ];
       ])
