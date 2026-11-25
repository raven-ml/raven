(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module D = Norn.Dist

let f64 = Nx.scalar Nx.float64
let i32 n = Nx.scalar Nx.int32 (Int32.of_int n)
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let item x = Nx.item [] x
let wide x = Nx.item [] (Nx.cast Nx.float64 x)

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let std = D.normal ~loc:(f64 0.) ~scale:(f64 1.)

(* Two normal components, at 0 and 1. *)
let two = D.normal ~loc:(vec [| 0.; 1. |]) ~scale:(f64 1.)

(* An hmm of [n] steps over [two]'s states, every logit 0. *)
let hmm2 n =
  D.hmm
    ~initial:(vec [| 0.; 0. |])
    ~transition:(Nx.zeros Nx.float64 [| 2; 2 |])
    n two

let trials_3_7 = Nx.create Nx.int32 [| 2 |] [| 3l; 7l |]

let softplus v =
  if v > 0. then v +. Float.log1p (Float.exp (-.v))
  else Float.log1p (Float.exp v)

let sigmoid l = Float.exp (-.softplus (-.l))
let log_sigmoid l = -.softplus (-.l)

let logsumexp xs =
  let m = Array.fold_left Float.max Float.neg_infinity xs in
  if m = Float.neg_infinity then m
  else
    m +. Float.log (Array.fold_left (fun a x -> a +. Float.exp (x -. m)) 0. xs)

let half_log_2pi = 0.5 *. Float.log (2. *. Float.pi)

(* norn.mli's bound on a truncation's log density [v]: [64 ε (1 + |log f| + |log
   m|)], [log f] the base's at the point and [log m = log f - v] the log
   mass. *)
let truncation_bound eps ~base ~v =
  64. *. eps *. (1. +. Float.abs base +. Float.abs (base -. v))

(* norn.mli's bound on a quantile [x] read from the tail [m], [f] the density at
   [x]: [64 ε (|x| + (1 + |log m|) m / f)]. *)
let quantile_bound ~x ~log_m ~log_f =
  64. *. epsilon_float
  *. (Float.abs x +. (Float.exp (log_m -. log_f) *. (1. +. Float.abs log_m)))

let support_w = Testable.make ~pp:Norn.Support.pp ~equal:Norn.Support.equal

(* gen/dist.py's goldens *)

(* The golden's lines, each split into its words, read once per worker. *)
let golden =
  fixture (fun () ->
      In_channel.with_open_text "golden/dist.golden" In_channel.input_lines
      |> List.filter (fun l -> l <> "" && l.[0] <> '#')
      |> List.map (String.split_on_char ' '))

let lines kind = List.filter (fun l -> List.hd l = kind) (golden ())
let floats ws = List.map float_of_string ws

(* [families kind] is [kind]'s lines grouped by their family, the second word,
   in the order the families first appear, each line's words after the
   family. *)
let families kind =
  let rows = Hashtbl.create 16 and order = ref [] in
  List.iter
    (function
      | _ :: family :: rest ->
          if not (Hashtbl.mem rows family) then order := family :: !order;
          Hashtbl.replace rows family
            (rest :: Option.value ~default:[] (Hashtbl.find_opt rows family))
      | _ -> failwith "dist.golden: a line without a family")
    (lines kind);
  List.rev_map (fun f -> (f, List.rev (Hashtbl.find rows f))) !order

(* gen/dist.py's truncations: each base, and its low and high bounds. *)
let truncations =
  [
    ("truncated_normal", ("normal", Some (-1.), Some 3.));
    ("truncated_normal_far", ("normal", Some 60.5, None));
    ("truncated_normal_below", ("normal", None, Some (-79.5)));
    ("truncated_normal_window", ("normal", Some 20.5, Some 21.5));
    ("truncated_gamma", ("gamma", Some 50., None));
    ("truncated_gamma_deep", ("gamma", None, Some 1e-130));
    ("truncated_laplace_far", ("laplace", Some 1000., None));
    ("truncated_gamma_both", ("gamma_small", Some 0.1, Some 2.));
    ("truncated_exponential", ("exponential", None, Some 0.01));
    ("truncated_cauchy", ("cauchy", Some 1e6, None));
    ("truncated_laplace", ("laplace", Some (-50.), Some (-40.)));
    ("truncated_inverse_gamma", ("inverse_gamma", None, Some 0.05));
    ("truncated_uniform", ("uniform", Some 0., Some 1.));
    ("truncated_half_normal", ("half_normal", None, Some 1e-3));
    ("truncated_beta", ("beta", Some 0.2, Some 0.6));
  ]

(* The distributions gen/dist.py names, at its parameters. *)
let rec continuous_case name =
  match name with
  | "normal" -> D.normal ~loc:(f64 0.5) ~scale:(f64 2.)
  | "half_normal" -> D.half_normal ~scale:(f64 1.5)
  | "lognormal" -> D.lognormal ~loc:(f64 0.3) ~scale:(f64 0.8)
  | "student_t" -> D.student_t ~df:(f64 3.5) ~loc:(f64 (-1.)) ~scale:(f64 0.7)
  | "cauchy" -> D.cauchy ~loc:(f64 1.) ~scale:(f64 0.5)
  | "half_cauchy" -> D.half_cauchy ~scale:(f64 2.)
  | "laplace" -> D.laplace ~loc:(f64 0.) ~scale:(f64 1.3)
  | "logistic" -> D.logistic ~loc:(f64 2.) ~scale:(f64 0.4)
  | "exponential" -> D.exponential ~rate:(f64 1.7)
  | "gamma" -> D.gamma ~concentration:(f64 2.5) ~rate:(f64 1.5)
  | "gamma_small" -> D.gamma ~concentration:(f64 0.4) ~rate:(f64 0.8)
  | "inverse_gamma" -> D.inverse_gamma ~concentration:(f64 3.) ~scale:(f64 2.)
  | "beta" -> D.beta ~a:(f64 2.) ~b:(f64 5.)
  | "beta_u" -> D.beta ~a:(f64 0.5) ~b:(f64 0.5)
  | "uniform" -> D.uniform ~low:(f64 (-1.)) ~high:(f64 3.)
  | "von_mises" -> D.von_mises ~loc:(f64 0.3) ~concentration:(f64 2.)
  | "von_mises_flat" -> D.von_mises ~loc:(f64 (-1.)) ~concentration:(f64 0.)
  | name when List.mem_assoc name truncations ->
      let base, low, high = List.assoc name truncations in
      D.truncated ?low:(Option.map f64 low) ?high:(Option.map f64 high)
        (continuous_case base)
  | _ -> failwith ("dist.golden: no continuous case " ^ name)

let count_case name =
  match name with
  | "poisson" -> D.poisson ~rate:(f64 3.2)
  | "neg_binomial" -> D.neg_binomial ~mean:(f64 4.) ~dispersion:(f64 2.5)
  | "binomial" -> D.binomial ~trials:(i32 12) ~logits:(f64 0.4)
  | "binomial_rare" -> D.binomial ~trials:(i32 40) ~logits:(f64 (-2.5))
  | "binomial_many" -> D.binomial ~trials:(i32 200) ~logits:(f64 (-1.))
  | _ -> failwith ("dist.golden: no count case " ^ name)

let draws = 20000

let chi_square name values =
  let bins =
    List.filter_map
      (function
        | [ _; n; lo; hi; p ] when n = name ->
            Some (float_of_string lo, float_of_string hi, float_of_string p)
        | _ -> None)
      (lines "bin")
  in
  List.fold_left
    (fun acc (lo, hi, p) ->
      let o =
        Array.fold_left
          (fun n v -> if v >= lo && v < hi then n + 1 else n)
          0 values
      in
      let e = p *. float_of_int draws in
      acc +. (((float_of_int o -. e) ** 2.) /. e))
    0. bins

let threshold name =
  match List.find (fun l -> List.nth l 1 = name) (lines "threshold") with
  | [ _; _; v ] -> float_of_string v
  | _ -> assert false

(* Normalisation in coordinates *)

(* [integrate f] is the integral of [f] over the reals, split at [0] where a
   density may have a kink: on each half-line the exp-sinh rule, [u = ±exp (π/2
   sinh t)] on a grid of [t], exact to rounding for densities with algebraic or
   exponential tails. *)
let integrate f =
  let h = 1. /. 64. and t_max = 4. in
  let n = int_of_float (2. *. t_max /. h) + 1 in
  let t = Array.init n (fun i -> -.t_max +. (float_of_int i *. h)) in
  let r = Array.map (fun t -> Float.exp (Float.pi /. 2. *. Float.sinh t)) t in
  let w =
    Array.mapi (fun i t -> h *. Float.pi /. 2. *. Float.cosh t *. r.(i)) t
  in
  let u = Array.append r (Array.map Float.neg r) in
  let values = Nx.to_array (f (vec u)) in
  let s = ref 0. in
  Array.iteri (fun i v -> s := !s +. (w.(i mod n) *. v)) values;
  !s

let pullback d u =
  let x, ld = Norn.Bij.forward (D.coords d) u in
  Nx.exp (Nx.add (D.factors d x) ld)

(* A pullback through coords holds the target's mass between the bijector's
   saturation points (norn.mli, Bij's totality), and the rule is exact to
   rounding. Two cases leave most beyond them: beta_u's density [(1 - x)^(-1/2)
   / π] past interval's saturation at [1 - x ≈ ε] holds [2 √ε / π ≈ 9.5e-9], and
   lkj at [η = 1/2], whose density in [u] is [sech u / π], past cholesky_corr's
   where [tanh u] rounds to 1, [|u| ≈ 18.4], holds [4 e^-18.4 / π ≈ 1.3e-8]. *)
let saturated_mass = 2e-8

(* Validation on use *)

let open_bounds =
  let normal = D.normal ~loc:(f64 0.5) ~scale:(f64 2.) in
  Windtrap.cases "a value on a support's bound has density -inf"
    ~id:(fun (name, _, _) -> name)
    [
      ("exponential at 0", D.exponential ~rate:(f64 1.7), 0.);
      ("exponential at -0", D.exponential ~rate:(f64 1.7), -0.);
      ("half_normal at 0", D.half_normal ~scale:(f64 1.5), 0.);
      ("half_cauchy at 0", D.half_cauchy ~scale:(f64 2.), 0.);
      ( "gamma of concentration 1 at 0",
        D.gamma ~concentration:(f64 1.) ~rate:(f64 1.7),
        0. );
      ("gamma below 0", continuous_case "gamma", -1.);
      ("inverse_gamma below 0", continuous_case "inverse_gamma", -1.);
      ("beta at 0", continuous_case "beta", 0.);
      ("beta at 1", continuous_case "beta", 1.);
      ("beta past 1", continuous_case "beta", 1.5);
      ("uniform at low", D.uniform ~low:(f64 (-1.)) ~high:(f64 3.), -1.);
      ("uniform at high", D.uniform ~low:(f64 (-1.)) ~high:(f64 3.), 3.);
      ( "truncated at low",
        D.truncated ~low:(f64 (-1.)) ~high:(f64 3.) normal,
        -1. );
      ( "truncated at high",
        D.truncated ~low:(f64 (-1.)) ~high:(f64 3.) normal,
        3. );
    ]
    (fun (_, d, x) ->
      equal float_exact Float.neg_infinity (item (D.factors d (f64 x))))

let validation =
  let zero = f64 0. and one = f64 1. and inf = f64 Float.infinity in
  group "validation on use"
    [
      test "a parameter outside its domain raises naming its index and value"
        (fun () ->
          let d = D.normal ~loc:(f64 0.) ~scale:(vec [| 1.; 2.; 0.; -1. |]) in
          raises
            (Invalid_argument
               "Norn.Dist.log_density: normal: scale at [2] is 0, not in (0, \
                inf)") (fun () -> D.log_density d (f64 0.)));
      test "a scalar parameter's message names the parameter" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.sample: poisson: rate")
            (fun () -> D.sample (Nx.Rng.key 0) (D.poisson ~rate:(f64 (-1.)))));
      test "NaN is outside every domain" (fun () ->
          raises
            (Invalid_argument
               "Norn.Dist.factors: bernoulli: logits is nan, not in [-inf, inf]")
            (fun () ->
              D.factors
                (D.bernoulli ~logits:(f64 Float.nan))
                (Nx.scalar Nx.bool true)));
      (* One claim over the families: each raises at its first eliminator,
         naming it, the family and the parameter. *)
      Windtrap.cases "each family checks its parameters' domains"
        ~id:(fun (name, _, _) -> name)
        [
          ( "normal of infinite loc",
            "Norn.Dist.log_density: normal: loc is inf",
            fun () -> D.log_density (D.normal ~loc:inf ~scale:one) zero );
          ( "von_mises of negative concentration",
            "Norn.Dist.log_density: von_mises: concentration is -1",
            fun () ->
              D.log_density
                (D.von_mises ~loc:zero ~concentration:(f64 (-1.)))
                zero );
          ( "lkj_cholesky of concentration 0",
            "Norn.Dist.log_density: lkj_cholesky: concentration is 0",
            fun () ->
              D.log_density
                (D.lkj_cholesky 2 ~concentration:zero)
                (Nx.eye Nx.float64 2) );
          ( "poisson of infinite rate",
            "Norn.Dist.log_density: poisson: rate is inf",
            fun () -> D.log_density (D.poisson ~rate:inf) (i32 0) );
          ( "neg_binomial of negative mean",
            "Norn.Dist.log_density: neg_binomial: mean is -1",
            fun () ->
              D.log_density
                (D.neg_binomial ~mean:(f64 (-1.)) ~dispersion:one)
                (i32 0) );
          ( "binomial of negative trials",
            "Norn.Dist.log_density: binomial: trials at [1] is -2",
            fun () ->
              D.log_density
                (D.binomial
                   ~trials:(Nx.create Nx.int32 [| 2 |] [| 3l; -2l |])
                   ~logits:zero)
                (i32 0) );
          ( "mixture of an infinite logit",
            "Norn.Dist.log_density: mixture: logits at [1] is inf",
            fun () ->
              D.log_density
                (D.mixture
                   ~logits:(vec [| 0.; Float.infinity |])
                   (D.normal ~loc:(vec [| 0.; 1. |]) ~scale:one))
                zero );
          ( "uniform of no width",
            "Norn.Dist.log_density: uniform: high - low is 0",
            fun () -> D.log_density (D.uniform ~low:one ~high:one) one );
        ]
        (fun (_, msg, f) ->
          raises_match (Exn.invalid_arg ~sub:msg) (fun () -> ignore (f ())));
      test "parameters that do not broadcast raise naming the constructor"
        (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.normal:") (fun () ->
              D.normal ~loc:(vec [| 0.; 1. |]) ~scale:(vec [| 1.; 1.; 1. |])));
      open_bounds;
      test "a compiled eliminator checks the parameters" (fun () ->
          Compiled.here ();
          raises_match
            (Exn.invalid_arg ~sub:"Norn.Dist.log_density: normal: scale")
            (fun () ->
              Rune.jit'
                (fun scale -> D.log_density (D.normal ~loc:zero ~scale) zero)
                (f64 (-1.))));
      (* norn.mli's example of a total density. *)
      test "valid and check give a total density" (fun () ->
          let total scale =
            let d = D.normal ~loc:zero ~scale:(f64 scale) in
            let ok = D.valid d in
            let d = D.check ~unless:(Nx.scalar Nx.bool true) "" d in
            item (Nx.where ok (D.log_density d one) (f64 Float.neg_infinity))
          in
          equal
            (pair float_exact float_exact)
            ( Float.neg_infinity,
              item (D.log_density (D.normal ~loc:zero ~scale:one) one) )
            (total (-1.), total 1.));
    ]

(* Continuous families *)

(* The bound norn.mli states on a log density [f]: [ε (16 |f| + 48 Λ + 12 M)],
   for [ε] the dtype's epsilon and the family's [Λ] and [M] at the exact
   parameters and point. *)
let density_bound eps f ~logs ~off =
  eps *. ((16. *. Float.abs f) +. (48. *. logs) +. (12. *. off))

let alog v = Float.abs (Float.log v)

(* [Λ] and [M] of a dirichlet of concentrations [alpha] at [x], read as [x /
   Σx]. *)
let simplex_terms alpha x =
  let total = Array.fold_left ( +. ) 0. alpha in
  let s = Array.fold_left ( +. ) 0. x in
  let x = Array.map (fun v -> v /. s) x in
  let logs = ref (float_of_int (Array.length alpha) +. alog total) in
  let off = ref 0. in
  Array.iteri
    (fun k a ->
      logs := !logs +. alog a +. alog x.(k);
      off := !off +. Float.abs (a -. (total *. x.(k))))
    alpha;
  (!logs, !off)

(* [Λ] and [M] of a golden line's family at its arguments. *)
let density_terms family args =
  match (family, args) with
  | "gamma", [ a; rate; x ] ->
      (1. +. alog a +. alog x, Float.abs ((rate *. x) -. a))
  | "inverse_gamma", [ a; scale; x ] ->
      (1. +. alog a +. alog x, Float.abs ((scale /. x) -. a))
  | "beta", [ a; b; x ] -> simplex_terms [| a; b |] [| x; 1. -. x |]
  | "student_t", [ df; scale; _ ] -> (1. +. alog df +. alog scale, 0.)
  | "dirichlet", [ a1; a2; a3; x1; x2; x3 ] ->
      simplex_terms [| a1; a2; a3 |] [| x1; x2; x3 |]
  | _ -> failwith ("dist.golden: a malformed density line for " ^ family)

(* [log1p (z²)], without overflow. *)
let log1p_square z =
  let z = Float.abs z in
  if z > 1. then (2. *. Float.log z) +. Float.log1p ((1. /. z) ** 2.)
  else Float.log1p (z *. z)

(* The terms [t] of a closed-form family's log density as norn.mli writes it, at
   a golden line's arguments: its bound is [8 ε Σ |t|]. *)
let closed_terms family args =
  match (family, args) with
  | "normal", [ loc; s; x ] ->
      let z = (x -. loc) /. s in
      [ z *. z /. 2.; Float.log s; half_log_2pi ]
  | "half_normal", [ s; x ] ->
      let z = x /. s in
      [ z *. z /. 2.; Float.log s; half_log_2pi; Float.log 2. ]
  | "lognormal", [ loc; s; x ] ->
      let lx = Float.log x in
      let z = (lx -. loc) /. s in
      [ z *. z /. 2.; Float.log s; half_log_2pi; lx; z *. lx /. s ]
  | "cauchy", [ loc; s; x ] ->
      [ log1p_square ((x -. loc) /. s); Float.log s; Float.log Float.pi ]
  | "half_cauchy", [ s; x ] ->
      [ log1p_square (x /. s); Float.log s; Float.log Float.pi; Float.log 2. ]
  | "laplace", [ loc; s; x ] -> [ (x -. loc) /. s; Float.log s; Float.log 2. ]
  | "logistic", [ loc; s; x ] ->
      let z = (x -. loc) /. s in
      [ z; 2. *. softplus (-.z); Float.log s ]
  | "exponential", [ rate; x ] -> [ Float.log rate; rate *. x ]
  | "uniform", [ low; high; _ ] -> [ Float.log (high -. low) ]
  | "von_mises", [ loc; k; x ] ->
      let d = x -. loc in
      [
        2. *. k *. (Float.sin (d /. 2.) ** 2.);
        Float.log (2. *. Float.pi *. item (Nx.i0e (f64 k)));
        k *. d *. Float.sin d;
      ]
  | "mvn", [ _; _; a; _; c; d; x0; x1 ] ->
      (* [z = L⁻¹ (x - loc)], and Skeel's [κ = ‖ |L⁻¹| |L| ‖∞] of the lower
         triangle [[a, 0], [c, d]]. *)
      let z0 = (x0 -. 1.) /. a in
      let z1 = (x1 +. 1. -. (c *. z0)) /. d in
      let zz = (z0 *. z0) +. (z1 *. z1) in
      let row1 =
        (Float.abs (c /. (a *. d)) *. Float.abs a)
        +. (Float.abs c /. Float.abs d)
        +. 1.
      in
      let kappa = Float.max 1. row1 in
      [
        zz /. 2.;
        Float.log (Float.abs a);
        Float.log (Float.abs d);
        2. *. half_log_2pi;
        4. *. kappa *. zz;
      ]
  | "categorical", [ l0; l1; l2; k ] ->
      let logits = [| l0; l1; l2 |] in
      [ logits.(int_of_float k); logsumexp logits; Float.log 3. ]
  | "bernoulli", [ l; x ] -> [ softplus (if x = 1. then -.l else l) ]
  | _ -> failwith ("dist.golden: a malformed density line for " ^ family)

(* A golden line's family at [dt], its log density at the line's point. *)
let density_at (type b) (dt : (float, b) Nx.dtype) family args =
  let s v = Nx.scalar dt v in
  let v xs = Nx.create dt [| Array.length xs |] xs in
  let at d x = wide (D.factors d x) in
  match (family, args) with
  | "gamma", [ a; rate; x ] ->
      at (D.gamma ~concentration:(s a) ~rate:(s rate)) (s x)
  | "inverse_gamma", [ a; scale; x ] ->
      at (D.inverse_gamma ~concentration:(s a) ~scale:(s scale)) (s x)
  | "beta", [ a; b; x ] -> at (D.beta ~a:(s a) ~b:(s b)) (s x)
  | "student_t", [ df; scale; x ] ->
      at (D.student_t ~df:(s df) ~loc:(s 0.) ~scale:(s scale)) (s x)
  | "dirichlet", [ a1; a2; a3; x1; x2; x3 ] ->
      (* Points of float32 components sum to one within float32's rounding: read
         at a wider dtype, they are first put back on the simplex. *)
      let x = v [| x1; x2; x3 |] in
      at (D.dirichlet ~concentration:(v [| a1; a2; a3 |])) (Nx.div x (Nx.sum x))
  | "normal", [ loc; scale; x ] ->
      at (D.normal ~loc:(s loc) ~scale:(s scale)) (s x)
  | "half_normal", [ scale; x ] -> at (D.half_normal ~scale:(s scale)) (s x)
  | "lognormal", [ loc; scale; x ] ->
      at (D.lognormal ~loc:(s loc) ~scale:(s scale)) (s x)
  | "cauchy", [ loc; scale; x ] ->
      at (D.cauchy ~loc:(s loc) ~scale:(s scale)) (s x)
  | "half_cauchy", [ scale; x ] -> at (D.half_cauchy ~scale:(s scale)) (s x)
  | "laplace", [ loc; scale; x ] ->
      at (D.laplace ~loc:(s loc) ~scale:(s scale)) (s x)
  | "logistic", [ loc; scale; x ] ->
      at (D.logistic ~loc:(s loc) ~scale:(s scale)) (s x)
  | "exponential", [ rate; x ] -> at (D.exponential ~rate:(s rate)) (s x)
  | "uniform", [ low; high; x ] ->
      at (D.uniform ~low:(s low) ~high:(s high)) (s x)
  | "von_mises", [ loc; k; x ] ->
      at (D.von_mises ~loc:(s loc) ~concentration:(s k)) (s x)
  | "mvn", [ l0; l1; a; b; c; d; x0; x1 ] ->
      at
        (D.mvn
           ~loc:(v [| l0; l1 |])
           ~scale_tril:(Nx.create dt [| 2; 2 |] [| a; b; c; d |]))
        (v [| x0; x1 |])
  | "categorical", [ l0; l1; l2; k ] ->
      wide
        (D.factors
           (D.categorical ~logits:(v [| l0; l1; l2 |]))
           (Nx.scalar Nx.int64 (Int64.of_float k)))
  | "bernoulli", [ l; x ] ->
      wide (D.factors (D.bernoulli ~logits:(s l)) (Nx.scalar Nx.bool (x = 1.)))
  | _ -> failwith ("dist.golden: a malformed density line for " ^ family)

(* [v] read at [dt] is [v]. *)
let holds (type b) (dt : (float, b) Nx.dtype) v =
  (not (Float.is_finite v))
  || Float.equal v (Nx.item [] (Nx.cast Nx.float64 (Nx.scalar dt v)))

let saddle_point =
  [ "gamma"; "inverse_gamma"; "beta"; "student_t"; "dirichlet" ]

(* [near_density dt family args v actual] checks [actual], [dt]'s log density at
   a golden line, against the line's [v], within the family's bound: the
   saddle-point families' at float32 for a narrower dtype, which computes at
   float32 and rounds once, half an ulp more; the closed forms' [8 ε Σ |t|] at
   [dt]'s own [ε], and the least subnormal where the log density underflows. *)
let near_density (type b) ?(msg = "") (dt : (float, b) Nx.dtype) family args v
    actual =
  let format = Nx.Dtype.float_format dt in
  let narrow = format.fraction_bits < 23 in
  let tol =
    if List.mem family saddle_point then
      let logs, off = density_terms family args in
      if narrow then
        density_bound (Nx.Dtype.float_format Nx.float32).epsilon v ~logs ~off
        +. (0.5 *. format.epsilon *. (Float.abs v +. format.min_normal))
      else density_bound format.epsilon v ~logs ~off
    else
      let t =
        List.fold_left
          (fun a t -> a +. Float.abs t)
          0. (closed_terms family args)
      in
      (8. *. format.epsilon *. t) +. (format.min_normal *. format.epsilon)
  in
  let msg = msg ^ Nx.Dtype.name dt in
  if Float.is_finite v then
    equal ~msg (float (Float.max tol Float.min_float)) v actual
  else equal ~msg float_exact v actual

let check_density dt family args v =
  near_density dt family args v (density_at dt family args)

let golden_densities =
  test "log densities are within their bound of mpmath's" (fun () ->
      List.iter
        (fun (family, rows) ->
          subtest family (fun () ->
              List.iter
                (fun row ->
                  let row = floats row in
                  let n = List.length row - 1 in
                  let args = List.filteri (fun i _ -> i < n) row in
                  let v = List.nth row n in
                  let at (type b) (dt : (float, b) Nx.dtype) =
                    let fits =
                      Float.abs v < (Nx.Dtype.float_format dt).max_finite
                      || not (Float.is_finite v)
                    in
                    if List.for_all (holds dt) args && fits then
                      check_density dt family args v
                  in
                  at Nx.float64;
                  at Nx.float32;
                  (* nx solves triangular systems at float32 and float64
                     alone. *)
                  if family <> "mvn" then begin
                    at Nx.float16;
                    at Nx.bfloat16
                  end)
                rows))
        (families "density"))

(* Against mpmath at float64 alone, where the mean count lies far from the
   concentration and the logarithm of their ratio changes form. *)
let golden_far_densities =
  test "log densities far from the mean count are within their bound" (fun () ->
      List.iter
        (fun (family, rows) ->
          subtest family (fun () ->
              List.iter
                (fun row ->
                  match floats row with
                  | [ a; s; x; v ] ->
                      check_density Nx.float64 family [ a; s; x ] v
                  | _ -> failwith "dist.golden: a malformed density64 line")
                rows))
        (families "density64"))

(* The log mass from log-gamma sums at float64, and a bound on its error: 16
   ulps on each log-gamma (nx's) and on each other term. Exact enough to judge
   the narrower dtypes, whose counts stay below [2^24]. *)
let lgamma v = item (Nx.lgamma (f64 v))

(* Scores of the continuous densities against their closed forms, from nx's
   digamma, within [64 ε T] for [T] one plus the magnitudes the closed form
   sums: it cancels where the score nears [0]. Concentrations reach from
   [1e-300], where the score is near [1 / α], to [1e30]; points lie near the
   mode and far into both tails.

   The bound, with [u = ε / 2]. The derivative Rune takes sums those of Loader's
   terms, at most eight: of [δ_h], within [30 u] absolutely and [4 u |log α|]
   (each step's derivative within [4 u] of its [1 / m], and below 1 the first
   step's [log α]); of [log α], [1 / α] within [u]; of [D], [log (α / m)] within
   [2 u] plus [u] of itself ([ratio_log]); the logs of the point and parameters
   within [4 u]; ratios of the parameters and the point within [u]. From [ψ(α) =
   log α - 1 / (2 α) - δ'(α)], [|log α|] and [1 / α] are within [2 |ψ(α)| + 2],
   and [|log (α / m)|] within [|log α| + |log (rate x)|], so the terms'
   magnitudes are within [3 T]. Their errors and the sum's rounding of [8 u] of
   [3 T] give [63 u T]; the closed form's digamma, within 16 ulps or [16 ε] near
   its root, its log and its sums give [18 ε T]: [50 ε T] in all. Where
   [ratio_log] takes the logs' difference, its error is [4 u (|log α| + |log
   m|)] instead, within [16 u T], [|log α|] and [|log m|] each within [2 T]: [58
   ε T]. *)
let digamma v = item (Nx.digamma (f64 v))

(* [log10] of a concentration: mostly moderate, often extreme. *)
let concentration =
  Gen.(
    map
      (fun e -> 10. ** e)
      (frequency
         [
           (4, float_range (-3.) 7.);
           (1, float_range (-300.) (-3.));
           (1, float_range 7. 30.);
         ]))

(* A unit gamma point of concentration [a]: [a] times [10^e], its mean count
   from [1e-3 a] to [1e6 a], where the logarithm of the count over its mean
   takes each of its forms. *)
let unit_point a = Gen.(map (fun e -> a *. (10. ** e)) (float_range (-3.) 6.))

let score_tol terms =
  let scale = List.fold_left (fun s t -> s +. Float.abs t) 1. terms in
  float (64. *. epsilon_float *. scale)

let grad2 f a b =
  let ga, gb =
    Rune.grad Nx.Ptree.(pair tensor tensor) (fun (a, b) -> f a b) (f64 a, f64 b)
  in
  (item ga, item gb)

let power lo hi = Gen.(map (fun e -> 10. ** e) (float_range lo hi))

let continuous_scores =
  group "scores"
    [
      prop "the gamma's scores are log (rate x) - ψ(α) and α / rate - x"
        ~examples:[ (1e-5, 1., 1e-5); (1e8, 1., 1e8) ]
        Gen.(
          let* a = concentration in
          let+ y = unit_point a and+ rate = power (-6.) 6. in
          (a, rate, y /. rate))
        (fun (a, rate, x) ->
          assume (Float.is_finite x && x > 0. && rate *. x > 0.);
          cover "concentration below 1e-3" (a < 1e-3);
          cover "concentration above 1e7" (a > 1e7);
          cover "a mean count above twice the concentration"
            (rate *. x > 2. *. a);
          let ga, gr =
            grad2
              (fun a rate -> D.factors (D.gamma ~concentration:a ~rate) (f64 x))
              a rate
          in
          let y = rate *. x in
          equal ~msg:"concentration"
            (score_tol [ Float.log y; digamma a ])
            (Float.log y -. digamma a)
            ga;
          equal ~msg:"rate" (score_tol [ a /. rate; x ]) ((a /. rate) -. x) gr);
      prop "the inverse gamma's scores are log (s / x) - ψ(α) and α / s - 1 / x"
        ~examples:[ (1e-5, 1., 1e5); (1e8, 1., 1e-8) ]
        Gen.(
          let* a = concentration in
          let+ y = unit_point a and+ s = power (-6.) 6. in
          (a, s, s /. y))
        (fun (a, s, x) ->
          assume (Float.is_finite x && x > 0. && s /. x > 0.);
          cover "concentration below 1e-3" (a < 1e-3);
          cover "concentration above 1e7" (a > 1e7);
          cover "a mean count above twice the concentration" (s /. x > 2. *. a);
          let ga, gs =
            grad2
              (fun a scale ->
                D.factors (D.inverse_gamma ~concentration:a ~scale) (f64 x))
              a s
          in
          let y = s /. x in
          equal ~msg:"concentration"
            (score_tol [ Float.log y; digamma a ])
            (Float.log y -. digamma a)
            ga;
          equal ~msg:"scale"
            (score_tol [ a /. s; 1. /. x ])
            ((a /. s) -. (1. /. x))
            gs);
      (* The point from its logit: [log (a / b)] and [z] of the logit's spread,
         about [sqrt (1 / a + 1 / b)], within [[-700, 36]], where [x] is a float
         of [(0, 1)]. The examples hold counts far below their means, where [a]
         is large and [x] near [1/2]. *)
      prop "the beta's scores are log x - ψ(a) + ψ(a + b) and its mirror"
        ~examples:
          [
            (1e-5, 1., 0.5);
            (1e8, 2., 1. -. 2e-8);
            (1e-20, 1e-25, 1e-304);
            (1e4, 0.1, 0.5);
            (1e8, 1., 0.5);
          ]
        Gen.(
          with_pp
            (fun ppf (a, b, x) ->
              Format.fprintf ppf "a = %h, b = %h, x = %h" a b x)
            (let+ a = concentration
             and+ b = concentration
             and+ z = float_range (-30.) 30. in
             let spread = Float.sqrt ((1. /. a) +. (1. /. b)) in
             let t = Float.log a -. Float.log b +. (z *. spread) in
             let t = Float.min 36. (Float.max (-700.) t) in
             (a, b, 1. /. (1. +. Float.exp (-.t)))))
        (fun (a, b, x) ->
          let total = a +. b in
          cover "a below 1e-3" (a < 1e-3);
          cover "a above 1e7" (a > 1e7);
          cover "a count below half its mean"
            (a < total *. x /. 2. || b < total *. (1. -. x) /. 2.);
          let ga, gb =
            grad2 (fun a b -> D.factors (D.beta ~a ~b) (f64 x)) a b
          in
          let n = digamma (a +. b) in
          equal ~msg:"a"
            (score_tol [ Float.log x; digamma a; n ])
            (Float.log x -. digamma a +. n)
            ga;
          let l1 = Float.log1p (-.x) in
          equal ~msg:"b"
            (score_tol [ l1; digamma b; n ])
            (l1 -. digamma b +. n)
            gb);
      (* Where the mean count [rate x] or [scale / x] is subnormal or [0], [dm /
         m] overflows; the scores stay the closed forms'. *)
      Windtrap.cases "the scores where the mean count underflows"
        ~id:(fun (name, _, _, _, _) -> name)
        [
          ("gamma, rate x of 0", false, 1., 1e-200, 1e-200);
          ("gamma, rate x subnormal", false, 1., 1., 1e-310);
          ("inverse gamma, scale / x subnormal", true, 2., 1e-10, 1e300);
          ("inverse gamma, scale / x of 0", true, 2., 1e-30, 1e300);
          ( "gamma of concentration 1e-20, rate x of 0",
            false,
            1e-20,
            1e-19,
            1e-305 );
        ]
        (fun (_, inverse, a, s, x) ->
          let f a s =
            D.factors
              (if inverse then D.inverse_gamma ~concentration:a ~scale:s
               else D.gamma ~concentration:a ~rate:s)
              (f64 x)
          in
          let ga, gs = grad2 f a s in
          (* The log of the mean count, from the logs: [log (rate x)] or [log
             (scale / x)]. *)
          let ly =
            if inverse then Float.log s -. Float.log x
            else Float.log s +. Float.log x
          in
          let gs_exact =
            if inverse then (a /. s) -. (1. /. x) else (a /. s) -. x
          in
          equal ~msg:"concentration"
            (score_tol [ ly; digamma a ])
            (ly -. digamma a)
            ga;
          equal ~msg:"scale"
            (float_rel ~rel:(64. *. epsilon_float) ~abs:0.)
            gs_exact gs);
      (* A mean count [n x] that rounds to 0 while [x / (n x)] is finite: a
         count below about [1e-16] at a point near the least normal. The closed
         forms are within the bound of the interface from their terms, each
         within a few ulps of itself. *)
      Windtrap.cases "a density whose mean count underflows is its closed form"
        ~id:(fun (family, _) -> family)
        [
          ("gamma", [ 1e-20; 1e-19; 1e-305 ]); ("beta", [ 1e-20; 1e-25; 1e-304 ]);
        ]
        (fun (family, args) ->
          let exact =
            match args with
            | [ a; s; x ] when family = "gamma" ->
                (a *. Float.log s)
                +. ((a -. 1.) *. Float.log x)
                -. (s *. x) -. lgamma a
            | [ a; b; x ] ->
                ((a -. 1.) *. Float.log x)
                +. ((b -. 1.) *. Float.log1p (-.x))
                -. (lgamma a +. lgamma b -. lgamma (a +. b))
            | _ -> assert false
          in
          let logs, off = density_terms family args in
          equal
            (float (density_bound epsilon_float exact ~logs ~off))
            exact
            (density_at Nx.float64 family args));
      prop "a dirichlet's scores are ψ(A) - ψ(α_k) + log x_k"
        Gen.(
          (* The point is the mean, or the mean's components each scaled by
             [10^e] and normalised. *)
          let off = map (fun e -> 10. ** e) (float_range (-3.) 6.) in
          pair
            (triple concentration concentration concentration)
            (frequency [ (1, constant (1., 1., 1.)); (2, triple off off off) ]))
        (fun ((a1, a2, a3), (w1, w2, w3)) ->
          let alpha = [| a1; a2; a3 |] in
          let total = a1 +. a2 +. a3 in
          let weight = [| a1 *. w1; a2 *. w2; a3 *. w3 |] in
          let whole = Array.fold_left ( +. ) 0. weight in
          let x = Array.map (fun w -> w /. whole) weight in
          assume (Float.is_finite whole && Array.for_all (fun v -> v > 0.) x);
          cover "a concentration above 1e7"
            (Array.exists (fun a -> a > 1e7) alpha);
          cover "the mean point" (w1 = 1. && w2 = 1. && w3 = 1.);
          cover "a mean count above twice its concentration"
            (Array.exists2 (fun a v -> total *. v > 2. *. a) alpha x);
          let g =
            Rune.grad'
              (fun a -> D.factors (D.dirichlet ~concentration:a) (vec x))
              (vec alpha)
          in
          Array.iteri
            (fun k a ->
              equal
                (score_tol [ digamma total; digamma a; Float.log x.(k) ])
                (digamma total -. digamma a +. Float.log x.(k))
                (Nx.item [ k ] g))
            alpha);
      (* [∂ν f = (ψ((ν + 1) / 2) - ψ(ν / 2) - 1/ν - log1p (z²/ν) + (ν + 1) z² /
         (ν (ν + z²))) / 2] and [∂σ f = ((ν + 1) z² / (ν + z²) - 1) / σ]. *)
      prop "the student t's scores in its degrees of freedom and scale"
        ~examples:[ (1., 1., 1.); (4., 0.5, 2.); (1e-5, 1., 1.); (1e8, 1., 1.) ]
        Gen.(triple concentration (power (-3.) 3.) (power (-3.) 4.))
        (fun (df, scale, z) ->
          cover "df below 1e-3" (df < 1e-3);
          cover "df above 1e7" (df > 1e7);
          let x = z *. scale in
          let gd, gs =
            grad2
              (fun df scale ->
                D.factors (D.student_t ~df ~loc:(f64 0.) ~scale) (f64 x))
              df scale
          in
          let z2 = z *. z in
          let w = (df +. 1.) *. z2 /. (df *. (df +. z2)) in
          let a = digamma ((df +. 1.) /. 2.) and b = digamma (df /. 2.) in
          let l = Float.log1p (z2 /. df) in
          equal ~msg:"df"
            (score_tol [ a; b; 1. /. df; l; w ])
            (0.5 *. (a -. b -. (1. /. df) -. l +. w))
            gd;
          let v = (df +. 1.) *. z2 /. (df +. z2) in
          equal ~msg:"scale"
            (score_tol [ v /. scale; 1. /. scale ])
            ((v -. 1.) /. scale)
            gs);
    ]

(* Compiled, the golden lines of each family in one program of their parameters:
   each density within its bound of mpmath's, as eager. *)
let compiled_densities =
  Windtrap.cases "a compiled density is within its bound"
    ~id:(fun family -> family)
    [ "gamma"; "inverse_gamma"; "beta"; "student_t"; "dirichlet"; "von_mises" ]
    (fun family ->
      Compiled.here ();
      let rows =
        List.filter_map
          (function
            | _ :: f :: rest when f = family -> Some (floats rest) | _ -> None)
          (lines "density" @ lines "density64")
      in
      let n = List.length rows in
      let k = List.length (List.hd rows) - 1 in
      let args = List.map (List.filteri (fun i _ -> i < k)) rows in
      let v = Array.of_list (List.map (fun r -> List.nth r k) rows) in
      let col i = vec (Array.of_list (List.map (fun r -> List.nth r i) args)) in
      let simplex first =
        Nx.create Nx.float64 [| n; 3 |]
          (Array.concat
             (List.map
                (fun r ->
                  Array.of_list
                    (List.filteri (fun i _ -> i >= first && i < first + 3) r))
                args))
      in
      let density p =
        match family with
        | "gamma" -> D.factors (D.gamma ~concentration:p ~rate:(col 1)) (col 2)
        | "inverse_gamma" ->
            D.factors (D.inverse_gamma ~concentration:p ~scale:(col 1)) (col 2)
        | "beta" -> D.factors (D.beta ~a:p ~b:(col 1)) (col 2)
        | "student_t" ->
            D.factors
              (D.student_t ~df:p
                 ~loc:(Nx.zeros Nx.float64 [| n |])
                 ~scale:(col 1))
              (col 2)
        | "von_mises" ->
            D.factors (D.von_mises ~loc:(col 0) ~concentration:p) (col 2)
        | _ ->
            (* Points back on the simplex, as [density_at] puts them. *)
            let x = simplex 3 in
            D.factors
              (D.dirichlet ~concentration:p)
              (Nx.div x (Nx.sum ~axes:[ 1 ] ~keepdims:true x))
      in
      let p =
        match family with
        | "dirichlet" -> simplex 0
        | "von_mises" -> col 1
        | _ -> col 0
      in
      let compiled = Nx.to_array (Rune.jit' density p) in
      List.iteri
        (fun i a ->
          let msg =
            String.concat " " (List.map (Printf.sprintf "%h") a) ^ " "
          in
          near_density ~msg Nx.float64 family a v.(i) compiled.(i))
        args)

(* At float16 [z²] overflows from [z = 256], where [-z² / 2] is still a float16
   until [z = 362]. The exact value at [z = 300] is [-45000 - log √(2π)], within
   [8 ε] of its magnitude at float16. *)
let float16_squares =
  let h = Nx.scalar Nx.float16 in
  let at d x = Nx.item [] (Nx.cast Nx.float64 (D.factors d (h x))) in
  let tol v = float (8. *. 0x1p-10 *. Float.abs v) in
  Windtrap.cases "a float16 log density holds where z² overflows"
    ~id:(fun (name, _, _) -> name)
    [
      ("normal", D.normal ~loc:(h 0.) ~scale:(h 1.), -45000. -. half_log_2pi);
      ( "half_normal",
        D.half_normal ~scale:(h 1.),
        Float.log 2. -. 45000. -. half_log_2pi );
    ]
    (fun (_, d, v) -> equal (tol v) v (at d 300.))

let normal = group "normal" [ float16_squares ]

let cauchy_tails =
  let exact ~half d s =
    let lead = if half then Float.log 2. else 0. in
    lead -. Float.log Float.pi -. Float.log s
    -. (2. *. (Float.log d -. Float.log s))
    -. Float.log1p ((s /. d) ** 2.)
  in
  let magnitude d s =
    2. +. (3. *. Float.abs (Float.log s)) +. (2. *. Float.abs (Float.log d))
  in
  Windtrap.cases "a Cauchy's log density far past where z² overflows"
    ~id:(fun (name, _, _, _) -> name)
    [
      ("cauchy, scale 1e-160", false, 1., 1e-160);
      ("cauchy, x 1e200", false, 1e200, 1.);
      ("cauchy, z past the floats", false, 1., 1e-310);
      ("half_cauchy, scale 1e-160", true, 1., 1e-160);
      ("half_cauchy, x 1e200", true, 1e200, 1.);
    ]
    (fun (_, half, d, s) ->
      let dist =
        if half then D.half_cauchy ~scale:(f64 s)
        else D.cauchy ~loc:(f64 0.) ~scale:(f64 s)
      in
      equal
        (float (8. *. epsilon_float *. magnitude d s))
        (exact ~half d s)
        (item (D.factors dist (f64 d))))

let cauchy = group "cauchy" [ cauchy_tails ]

(* Far in a Student t's tail, [z = (x - loc) / scale] or [z / √ν] overflows
   where its log density does not: [c - ((ν + 1) / 2) log1p (z² / ν)], [c] its
   normaliser less [log scale], is read with [log1p (z² / ν)] as [2 (log d - log
   s) + log1p ((s / d)²)], [s = scale √ν], [log s] read as [log scale + log √ν]
   where [s] itself underflows. The bound is norn.mli's, [ε (16 |f| + 48 Λ)], [Λ
   = 1 + |log ν| + |log scale|]. *)
let student_t_tails =
  Windtrap.cases "a Student t's log density far past where z overflows"
    ~id:(fun (df, d, s) -> Printf.sprintf "df %g, x - loc %g, scale %g" df d s)
    [
      (1., 1., 1e-310);
      (1e-300, 1e160, 1.);
      (3.5, 1e200, 1e-100);
      (1e-300, 1., 1e-200);
    ]
    (fun (df, d, s) ->
      let c =
        lgamma ((df +. 1.) /. 2.)
        -. lgamma (df /. 2.)
        -. (0.5 *. (Float.log df +. Float.log Float.pi))
        -. Float.log s
      in
      let log_s = Float.log s +. (0.5 *. Float.log df) in
      let l =
        (2. *. (Float.log d -. log_s))
        +. Float.log1p (Float.exp (2. *. (log_s -. Float.log d)))
      in
      let exact = c -. ((df +. 1.) /. 2. *. l) in
      let logs = 1. +. alog df +. alog s in
      equal
        (float (density_bound epsilon_float exact ~logs ~off:0.))
        exact
        (item
           (D.factors
              (D.student_t ~df:(f64 df) ~loc:(f64 0.) ~scale:(f64 s))
              (f64 d))))

(* At float16, components of a dirichlet of many small concentrations
   underflowed: a draw then had density [-inf]. *)
let narrow_dirichlet_draws =
  Windtrap.cases "a narrow dirichlet's draws have finite densities"
    ~id:(fun k -> Printf.sprintf "%d components" k)
    [ 100; 1000 ]
    (fun k ->
      let d = D.dirichlet ~concentration:(Nx.full Nx.float16 [| k |] 0.7) in
      let x = D.sample (Nx.Rng.key 41) (D.iid [| 2000 |] d) in
      let f =
        Nx.to_array (Nx.cast Nx.float64 (D.factors (D.iid [| 2000 |] d) x))
      in
      satisfies ~claim:"every density finite" (array float_exact)
        (Array.for_all Float.is_finite)
        f;
      let least = Float.ldexp 1. (-24) in
      satisfies ~claim:"every component at least float16's least positive value"
        (array float_exact)
        (Array.for_all (fun v -> v >= least))
        (Nx.to_array (Nx.cast Nx.float64 x)))

let dirichlet =
  group "dirichlet"
    [
      (* norn.mli: a vector whose sum is within [4 K ε] of one is read as [x / Σ
         x]. A point whose components are scaled by [1 + 12 r ε], [K = 3], has a
         sum [12 r ε] off one, and rounding to the dtype moves it by under [2
         ε]: with [|r|] below [0.8] it has the density of the point on its ray,
         above [1.2] density [-inf]. *)
      prop "a vector within 4Kε of the simplex has its ray's density"
        ~examples:[ (0, 0.8, (1., 1., 1.)); (1, -1.2, (0.5, 0.3, 0.2)) ]
        Gen.(
          triple (int_range 0 1)
            (one_of
               [
                 float_range (-0.8) 0.8;
                 float_range 1.2 1.5;
                 float_range (-1.5) (-1.2);
               ])
            (triple (float_range 0.01 1.) (float_range 0.01 1.)
               (float_range 0.01 1.)))
        (fun (i, r, (a, b, c)) ->
          cover "float64" (i = 0);
          cover "float32" (i = 1);
          cover "within" (Float.abs r <= 0.8);
          cover "beyond" (Float.abs r >= 1.2);
          let at (type b) (dt : (float, b) Nx.dtype) =
            let eps = (Nx.Dtype.float_format dt).epsilon in
            let s = a +. b +. c in
            let point =
              Array.map
                (fun v -> v /. s *. (1. +. (12. *. r *. eps)))
                [| a; b; c |]
            in
            let point = Array.map (fun v -> wide (Nx.scalar dt v)) point in
            let alpha = [| 1.5; 2.; 0.7 |] in
            let f =
              wide
                (D.factors
                   (D.dirichlet ~concentration:(Nx.create dt [| 3 |] alpha))
                   (Nx.create dt [| 3 |] point))
            in
            if Float.abs r >= 1.2 then
              equal ~msg:(Nx.Dtype.name dt) float_exact Float.neg_infinity f
            else begin
              let total = Array.fold_left ( +. ) 0. point in
              let ray = Array.map (fun v -> v /. total) point in
              let exact =
                wide
                  (D.factors (D.dirichlet ~concentration:(vec alpha)) (vec ray))
              in
              let logs, off = simplex_terms alpha ray in
              equal ~msg:(Nx.Dtype.name dt)
                (float (density_bound eps exact ~logs ~off))
                exact f
            end
          in
          if i = 0 then at Nx.float64 else at Nx.float32);
      (* The sums' tolerance at its neighbours, [4 K ε] for [K = 3]: a point
         just inside reads finite, one just outside [-inf]. bfloat16's [1/3]
         rounds to [0.333984375], a sum of [1.0019] within its [0.094]. *)
      Windtrap.cases "a sum past 4Kε of one has density -inf"
        ~id:(fun (name, _) -> name)
        [
          ("bfloat16 thirds", ([| 1. /. 3.; 1. /. 3.; 1. /. 3. |], true));
          ("bfloat16 sum 1.3", ([| 0.5; 0.5; 0.3 |], false));
        ]
        (fun (_, (x, inside)) ->
          let bf xs = Nx.create Nx.bfloat16 [| Array.length xs |] xs in
          let d = D.dirichlet ~concentration:(bf [| 1.; 1.; 1. |]) in
          let f = wide (D.factors d (bf x)) in
          if inside then satisfies ~claim:"finite" float_exact Float.is_finite f
          else equal float_exact Float.neg_infinity f);
      test "a matrix of concentrations is independent rows" (fun () ->
          let a =
            Nx.create Nx.float64 [| 2; 3 |] [| 1.5; 2.; 0.7; 3.; 1.; 0.5 |]
          in
          let x =
            Nx.create Nx.float64 [| 2; 3 |] [| 0.2; 0.3; 0.5; 0.6; 0.39; 0.01 |]
          in
          let row i = Nx.get [ i ] in
          let one i =
            item (D.factors (D.dirichlet ~concentration:(row i a)) (row i x))
          in
          let rows = [| one 0; one 1 |] in
          (* a batch may sum each row in another order: a rounding of each
             term *)
          equal
            (array
               (float
                  (16. *. epsilon_float
                  *. (1. +. Float.abs rows.(0) +. Float.abs rows.(1)))))
            rows
            (Nx.to_array (D.factors (D.dirichlet ~concentration:a) x)));
      narrow_dirichlet_draws;
      test "fewer than two components raise naming dirichlet" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.dirichlet:") (fun () ->
              D.dirichlet ~concentration:(vec [| 1. |])));
    ]

let mvn =
  let loc = vec [| 1.; -1. |] in
  group "mvn"
    [
      test "its density reads scale_tril's lower triangle" (fun () ->
          let tril u = Nx.create Nx.float64 [| 2; 2 |] [| 2.; u; 0.5; 1. |] in
          let at u =
            Nx.to_array
              (D.factors (D.mvn ~loc ~scale_tril:(tril u)) (vec [| 5.; -3. |]))
          in
          equal (array float_exact) (at 0.) (at 9.));
      test "a scale_tril of another length raises naming mvn" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.mvn:") (fun () ->
              D.mvn ~loc ~scale_tril:(Nx.eye Nx.float64 3)));
      test "a dtype other than float32 and float64 raises naming mvn" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.mvn:") (fun () ->
              D.mvn
                ~loc:(Nx.zeros Nx.float16 [| 2 |])
                ~scale_tril:(Nx.eye Nx.float16 2)));
      test "a scale_tril that is not square raises naming mvn" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.mvn:") (fun () ->
              D.mvn ~loc ~scale_tril:(Nx.zeros Nx.float64 [| 2; 3 |])));
    ]

(* Von Mises *)

(* The mean of [cos (x - loc)] is [I₁(κ) / I₀(κ)] and its variance [(1 + I₂ /
   I₀) / 2 - (I₁ / I₀)²]; at [z = 5.3] the check fails falsely with probability
   about 1.2e-7. *)
let von_mises =
  group "von_mises"
    [
      Windtrap.cases "the mean resultant of its draws is I₁(κ) / I₀(κ)"
        ~id:(fun k -> Printf.sprintf "κ = %g" k)
        [ 0.5; 2.; 40.; 1e4 ]
        (fun k ->
          let m = 20000 in
          let d =
            D.iid [| m |] (D.von_mises ~loc:(f64 2.9) ~concentration:(f64 k))
          in
          let xs = Nx.to_array (D.sample (Nx.Rng.key 11) d) in
          let c = Array.map (fun x -> Float.cos (x -. 2.9)) xs in
          let mean = Array.fold_left ( +. ) 0. c /. float_of_int m in
          let r = item (Nx.div (Nx.i1e (f64 k)) (Nx.i0e (f64 k))) in
          let i2 =
            (* I₂ = I₀ - 2 I₁ / κ *)
            1. -. (2. *. r /. k)
          in
          let var = ((1. +. i2) /. 2.) -. (r *. r) in
          equal (float (5.3 *. Float.sqrt (var /. float_of_int m))) r mean);
      test "its draws lie on [-π, π]" (fun () ->
          let d =
            D.iid [| 2000 |]
              (D.von_mises ~loc:(f64 3.) ~concentration:(f64 0.5))
          in
          satisfies ~claim:"within [-π, π]" (array float_exact)
            (Array.for_all (fun x -> x >= -.Float.pi && x <= Float.pi))
            (Nx.to_array (D.sample (Nx.Rng.key 11) d)));
      (* Two log densities each within [8 ε Σ |t|] of the exact one. *)
      prop "its density is periodic in loc"
        Gen.(
          triple (float_range (-10.) 10.) (float_range 0.1 100.)
            (float_range (-3.1) 3.1))
        (fun (loc, k, x) ->
          let at loc =
            item
              (D.factors
                 (D.von_mises ~loc:(f64 loc) ~concentration:(f64 k))
                 (f64 x))
          in
          let loc' = loc +. (2. *. Float.pi) in
          let t loc =
            List.fold_left
              (fun a t -> a +. Float.abs t)
              0.
              (closed_terms "von_mises" [ loc; k; x ])
          in
          equal
            (float (8. *. epsilon_float *. (t loc +. t loc')))
            (at loc) (at loc'));
      prop "its log density is finite at every finite concentration"
        ~examples:[ (300., Float.pi); (300., -.Float.pi); (-30., 0.) ]
        Gen.(pair (float_range (-30.) 300.) (float_range (-.Float.pi) Float.pi))
        (fun (e, x) ->
          let k = 10. ** e in
          satisfies ~claim:"finite" float_exact Float.is_finite
            (item
               (D.factors
                  (D.von_mises ~loc:(f64 0.) ~concentration:(f64 k))
                  (f64 x))));
      (* [∂κ f = cos (x - loc) - I₁(κ) / I₀(κ)]: Rune's derivative of [-2κ sin²
         ((x - loc) / 2) - log (2π i0e κ)] sums [2 sin²], at most [2], and
         [i0e'/i0e], at most [1], each within a few roundings, and nx's ratio
         [i1e / i0e] adds as many: [64 ε] holds them. *)
      prop "the score of its concentration is cos (x - loc) - I₁ / I₀"
        Gen.(
          pair
            (map (fun e -> 10. ** e) (float_range (-3.) 5.))
            (float_range (-3.1) 3.1))
        (fun (k, x) ->
          let g =
            item
              (Rune.grad'
                 (fun k ->
                   D.factors
                     (D.von_mises ~loc:(f64 0.2) ~concentration:k)
                     (f64 x))
                 (f64 k))
          in
          let r = item (Nx.div (Nx.i1e (f64 k)) (Nx.i0e (f64 k))) in
          equal (float (64. *. epsilon_float)) (Float.cos (x -. 0.2) -. r) g);
    ]

let matrix n xs = Nx.create Nx.float64 [| n; n |] xs

(* nx's digamma within 16 ulps: the normaliser's score sums [n - 1] of them, and
   the density's [n] logs of the diagonal. *)
(* norn.mli's bound on an lkj log density [v], [diag i] the diagonal of
   row [i] from 1: [16 ε (|v| + Σ_(i >= 2) |(n - i + 2η - 2) log L_ii|)]. *)
let lkj_bound eps n eta diag v =
  let t = ref (Float.abs v) in
  for i = 2 to n do
    let c = float_of_int (n - i) +. (2. *. eta) -. 2. in
    t := !t +. Float.abs (c *. Float.log (diag i))
  done;
  16. *. eps *. !t

let lkj =
  let densities =
    test "log densities are within their bound of mpmath's" (fun () ->
        List.iter
          (function
            | _ :: n :: eta :: rest ->
                let n = int_of_string n and eta = float_of_string eta in
                let rest = floats rest in
                let entries =
                  Array.of_list (List.filteri (fun i _ -> i < n * n) rest)
                in
                let v = List.nth rest (n * n) in
                (* The rows of float32 values have unit norms within float32's
                   rounding: read at float64, each row's off-diagonal entries
                   are rescaled to a unit norm, which keeps the diagonal the
                   density reads. *)
                let rescaled = Array.copy entries in
                for i = 1 to n - 1 do
                  let off = ref 0. in
                  for j = 0 to i - 1 do
                    off := !off +. (rescaled.((i * n) + j) ** 2.)
                  done;
                  let dii = rescaled.((i * n) + i) in
                  let s = Float.sqrt ((1. -. (dii *. dii)) /. !off) in
                  for j = 0 to i - 1 do
                    rescaled.((i * n) + j) <- rescaled.((i * n) + j) *. s
                  done
                done;
                let diag i = entries.(((i - 1) * n) + i - 1) in
                let check (type b) (dt : (float, b) Nx.dtype) entries =
                  let eps = (Nx.Dtype.float_format dt).epsilon in
                  let d = D.lkj_cholesky n ~concentration:(Nx.scalar dt eta) in
                  equal
                    ~msg:
                      (Printf.sprintf "%s, n = %d, η = %g" (Nx.Dtype.name dt) n
                         eta)
                    (float (lkj_bound eps n eta diag v))
                    v
                    (wide (D.factors d (Nx.create dt [| n; n |] entries)))
                in
                check Nx.float64 rescaled;
                check Nx.float32 entries
            | _ -> failwith "dist.golden: a malformed lkj line")
          (lines "lkj"))
  in
  let correlation n eta key =
    let d = D.iid [| draws |] (D.lkj_cholesky n ~concentration:(f64 eta)) in
    let l = D.sample (Nx.Rng.key key) d in
    let c = Nx.matmul l (Nx.matrix_transpose l) in
    (l, d, Nx.to_array (Nx.slice [ Nx.A; Nx.I (n - 1); Nx.I 0 ] c))
  in
  group "lkj_cholesky"
    [
      densities;
      Windtrap.cases "the pullback through coords integrates to one at n = 2"
        ~id:(fun eta -> Printf.sprintf "η = %g" eta)
        [ 0.5; 1.; 3.; 30. ]
        (fun eta ->
          let d = D.lkj_cholesky 2 ~concentration:(f64 eta) in
          let f u =
            let m = Nx.shape u in
            let u = Nx.reshape [| m.(0); 1 |] u in
            let l, ld = Norn.Bij.forward (D.coords d) u in
            Nx.exp (Nx.add (D.factors d l) ld)
          in
          equal (float saturated_mass) 1. (integrate f));
      Windtrap.cases "an off-diagonal correlation of its draws is a scaled beta"
        ~id:(fun (n, eta, _) -> Printf.sprintf "n = %d, η = %g" n eta)
        [ (3, 2., 31); (5, 0.8, 32) ]
        (fun (n, eta, key) ->
          let _, _, r = correlation n eta key in
          at_most float_exact
            ~than:(threshold (Printf.sprintf "lkj_%d" n))
            (chi_square (Printf.sprintf "lkj_%d" n) r));
      (* At a small [η] and many rows the last diagonals near [0], where [√(1 -
         y)] read [0] for [y] rounded to [1]; at float16 a row's normal over its
         norm underflowed below [2.4e-4], leaving a NaN diagonal or a row off
         unit norm. *)
      test "its draws have finite densities" (fun () ->
          let l, d, _ = correlation 5 0.8 32 in
          satisfies ~claim:"every draw has a finite density" (array float_exact)
            (Array.for_all Float.is_finite)
            (Nx.to_array (D.factors d l)));
      Windtrap.cases "its draws have finite densities at narrow dtypes"
        ~id:(fun (Dtype dt, n, eta, _) ->
          Printf.sprintf "%s, n = %d, η = %g" (Nx.Dtype.name dt) n eta)
        [
          (Dtype Nx.float32, 10, 0.5, 2000);
          (Dtype Nx.float16, 10, 0.5, 2000);
          (Dtype Nx.bfloat16, 10, 0.5, 2000);
          (Dtype Nx.float16, 4, 1.5, 4000);
        ]
        (fun (Dtype dt, n, eta, m) ->
          let d =
            D.iid [| m |] (D.lkj_cholesky n ~concentration:(Nx.scalar dt eta))
          in
          let l = D.sample (Nx.Rng.key 35) d in
          satisfies ~claim:"every density finite" (array float_exact)
            (Array.for_all Float.is_finite)
            (Nx.to_array (Nx.cast Nx.float64 (D.factors d l))));
      (* [∂η f = Σ_(i >= 2) 2 log L_ii - Σ_k (ψ(η + (n - 1 - k) / 2) - ψ(η + (n
         - 1) / 2))], its terms each within 16 ulps (nx's digamma) and its sum
         within a rounding of each: [64 ε] of their magnitudes. *)
      test "the score of its concentration" (fun () ->
          let n = 4 in
          let l, _, _ = correlation n 1.5 33 in
          let l = Nx.get [ 0 ] l in
          List.iter
            (fun eta ->
              let g =
                item
                  (Rune.grad'
                     (fun eta ->
                       D.factors (D.lkj_cholesky n ~concentration:eta) l)
                     (f64 eta))
              in
              let logs = ref 0. in
              for i = 1 to n - 1 do
                logs := !logs +. (2. *. Float.log (Nx.item [ i; i ] l))
              done;
              let norm = ref 0. and terms = ref (Float.abs !logs) in
              for k = 1 to n - 1 do
                let a = digamma (eta +. (float_of_int (n - 1 - k) /. 2.)) in
                let b = digamma (eta +. (float_of_int (n - 1) /. 2.)) in
                norm := !norm +. a -. b;
                terms := !terms +. Float.abs a +. Float.abs b
              done;
              let expected = !logs -. !norm in
              equal (float (64. *. epsilon_float *. !terms)) expected g)
            [ 0.3; 1.; 7.; 1e5 ]);
      (* gen/dist.py's factors of 2 × 2, one program over their concentrations,
         each within the bound. Rows of float32 values are put back on the unit
         circle, as the eager test reads them. *)
      test "a compiled density is within its bound" (fun () ->
          Compiled.here ();
          let rows =
            List.filter_map
              (function
                | [ _; "2"; eta; _; _; l21; l22; v ] ->
                    Some
                      ( float_of_string eta,
                        float_of_string l21,
                        float_of_string l22,
                        float_of_string v )
                | _ -> None)
              (lines "lkj")
          in
          let factor (_, l21, l22, _) =
            [|
              1.; 0.; Float.copy_sign (Float.sqrt (1. -. (l22 *. l22))) l21; l22;
            |]
          in
          let ls =
            Nx.create Nx.float64
              [| List.length rows; 2; 2 |]
              (Array.concat (List.map factor rows))
          in
          let etas =
            vec (Array.of_list (List.map (fun (e, _, _, _) -> e) rows))
          in
          let compiled =
            Nx.to_array
              (Rune.jit'
                 (fun eta -> D.factors (D.lkj_cholesky 2 ~concentration:eta) ls)
                 etas)
          in
          List.iteri
            (fun i (eta, _, l22, v) ->
              equal
                ~msg:(Printf.sprintf "η = %g" eta)
                (float (lkj_bound epsilon_float 2 eta (fun _ -> l22) v))
                v compiled.(i))
            rows);
      (* A row's squared norm within [4 n ε] of one counts as in the support:
         [0.0625] at bfloat16 for [n = 2]. Its values of [0.6], [0.83] and
         [0.85] are [0.6015625], [0.83203125] and [0.8515625], so the rows'
         squared norms are [1.054] inside and [1.087] outside. *)
      Windtrap.cases "a row's norm within 4nε of one is in the support"
        ~id:(fun (x, _) -> Printf.sprintf "[0.6; %g] at bfloat16" x)
        [ (0.8, true); (0.83, true); (0.85, false); (0.95, false) ]
        (fun (x, inside) ->
          let l = D.lkj_cholesky 2 ~concentration:(Nx.scalar Nx.bfloat16 2.) in
          let f =
            wide
              (D.factors l
                 (Nx.create Nx.bfloat16 [| 2; 2 |] [| 1.; 0.; 0.6; x |]))
          in
          if inside then satisfies ~claim:"finite" float_exact Float.is_finite f
          else equal float_exact Float.neg_infinity f);
      (* One matrix against concentrations [0.5; 3]: each factor the scalar
         concentration's, within a rounding of each term (a batch may sum in
         another order). *)
      Windtrap.cases "a vector of concentrations is batch axes"
        ~id:(fun n -> Printf.sprintf "n = %d" n)
        [ 2; 3 ]
        (fun n ->
          let l =
            if n = 2 then matrix 2 [| 1.; 0.; 0.6; 0.8 |]
            else Nx.eye Nx.float64 3
          in
          let one eta =
            item (D.factors (D.lkj_cholesky n ~concentration:(f64 eta)) l)
          in
          let rows = [| one 0.5; one 3. |] in
          equal
            (array
               (float
                  (4. *. epsilon_float
                  *. (1. +. Float.abs rows.(0) +. Float.abs rows.(1)))))
            rows
            (Nx.to_array
               (D.factors
                  (D.lkj_cholesky n ~concentration:(vec [| 0.5; 3. |]))
                  l)));
      test "a matrix off the support has density -inf" (fun () ->
          let d = D.lkj_cholesky 2 ~concentration:(f64 2.) in
          List.iter
            (fun xs ->
              equal float_exact Float.neg_infinity
                (item (D.factors d (matrix 2 xs))))
            [
              [| 1.; 0.; 0.6; -0.8 |];
              [| 1.; 0.1; 0.6; 0.8 |];
              [| 1.; 0.; 0.6; 0.9 |];
            ]);
      test "fewer than two rows raise naming lkj_cholesky" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.lkj_cholesky:")
            (fun () -> D.lkj_cholesky 1 ~concentration:(f64 1.)));
    ]

(* Discrete families *)

(* [dt]'s epsilon and least subnormal. *)
let eps_of (type b) (dt : (float, b) Nx.dtype) =
  let format = Nx.Dtype.float_format dt in
  (format.epsilon, format.min_normal *. format.epsilon)

(* The largest count [dt] holds with every count below it. *)
let top_of (type b) (dt : (float, b) Nx.dtype) =
  Float.ldexp 1. ((Nx.Dtype.float_format dt).fraction_bits + 1)

type count_family = Poisson | Binomial | Neg_binomial

(* The bound norn.mli states on the log mass [f] of a count [x] of mean [m]: [ε
   (c_f |f| + c_m |x - m|)], for [ε] the dtype's epsilon, and half the least
   subnormal [s]; a whole [s] against a reference [f] itself rounded to a
   float. *)
let mass_bound family (eps, s) f x m =
  let c_f, c_m =
    match family with
    | Poisson -> (24., 1.)
    | Binomial -> (24., 9.)
    | Neg_binomial -> (64., 6.)
  in
  (eps *. ((c_f *. Float.abs f) +. (c_m *. Float.abs (x -. m)))) +. s

let count_family = function
  | "poisson" -> Poisson
  | "binomial" -> Binomial
  | "neg_binomial" -> Neg_binomial
  | f -> failwith ("dist.golden: no count family " ^ f)

(* The log mass at [x] of [family] at its parameters [a] and [b] as
   gen/dist.py's [mass] lines give them: rate; trials and logits; mean and
   dispersion. *)
let count_mass (type b) (dt : (float, b) Nx.dtype) family a b x =
  let s v = Nx.scalar dt v in
  let d =
    match family with
    | Poisson -> D.poisson ~rate:(s a)
    | Binomial -> D.binomial ~trials:(i32 (int_of_float a)) ~logits:(s b)
    | Neg_binomial -> D.neg_binomial ~mean:(s a) ~dispersion:(s b)
  in
  Nx.item [] (Nx.cast Nx.float64 (D.factors d (i32 (int_of_float x))))

let count_mean family a b =
  match family with Binomial -> a *. sigmoid b | Poisson | Neg_binomial -> a

(* [check_masses ~run family rows] checks [family]'s golden rows at float64, in
   one program of their parameters through [run], the program itself or its
   compilation: each mass within its bound of mpmath's, and [-inf] exactly
   outside the support. *)
let check_masses ~run family rows =
  let col i = Array.of_list (List.map (fun r -> List.nth r i) rows) in
  let a = col 0 and b = col 1 and x = col 2 and v = col 3 in
  let ints xs =
    Nx.create Nx.int32 [| Array.length xs |] (Array.map Int32.of_float xs)
  in
  let counts = ints x in
  let mass p =
    match family with
    | Poisson -> D.factors (D.poisson ~rate:p) counts
    | Binomial -> D.factors (D.binomial ~trials:(ints a) ~logits:p) counts
    | Neg_binomial ->
        D.factors (D.neg_binomial ~mean:p ~dispersion:(vec b)) counts
  in
  let p = vec (match family with Binomial -> b | _ -> a) in
  Array.iteri
    (fun i f ->
      let msg = Printf.sprintf "%h %h at %.0f" a.(i) b.(i) x.(i) in
      if Float.is_finite v.(i) then
        let m = count_mean family a.(i) b.(i) in
        equal ~msg
          (float (mass_bound family (eps_of Nx.float64) v.(i) x.(i) m))
          v.(i) f
      else equal ~msg float_exact v.(i) f)
    (Nx.to_array (run mass p))

(* Against mpmath at float64 in one program, and at float32 row by row, at the
   counts and trials it holds. *)
let golden_masses =
  test "log masses are within their bound of mpmath's" (fun () ->
      List.iter
        (fun (name, rows) ->
          subtest name (fun () ->
              let family = count_family name in
              let rows = List.map floats rows in
              check_masses ~run:(fun f p -> f p) family rows;
              let top = top_of Nx.float32 in
              List.iter
                (function
                  | [ a; b; x; v ]
                    when x <= top && (family <> Binomial || a <= top) ->
                      let msg = "float32" in
                      let actual = count_mass Nx.float32 family a b x in
                      if Float.is_finite v then
                        let m = count_mean family a b in
                        equal ~msg
                          (float (mass_bound family (eps_of Nx.float32) v x m))
                          v actual
                      else equal ~msg float_exact v actual
                  | _ -> ())
                rows))
        (families "mass"))

let classic family a b x =
  let terms =
    match family with
    | Poisson -> [ x *. Float.log a; -.a; -.lgamma (x +. 1.) ]
    | Binomial ->
        let n = a in
        [
          lgamma (n +. 1.);
          -.lgamma (x +. 1.);
          -.lgamma (n -. x +. 1.);
          (if x = 0. then 0. else x *. log_sigmoid b);
          (if x = n then 0. else (n -. x) *. log_sigmoid (-.b));
        ]
    | Neg_binomial ->
        let m = a and r = b in
        [
          lgamma (x +. r);
          -.lgamma r;
          -.lgamma (x +. 1.);
          -.(r *. Float.log1p (m /. r));
          (if x = 0. then 0. else x *. Float.log (m /. (r +. m)));
        ]
  in
  let sum = List.fold_left ( +. ) 0. in
  (sum terms, 16. *. epsilon_float *. sum (List.map Float.abs terms))

(* Where a count lies against its mean [m]. *)
type place = Near of float | Below of float | Above of float | Zero | Top

let place =
  Gen.(
    frequency
      [
        (3, map (fun z -> Near z) (float_range (-3.) 3.));
        (2, map (fun f -> Below f) (float_range 0.02 0.5));
        (2, map (fun f -> Above f) (float_range 2. 12.));
        (1, constant Zero);
        (1, constant Top);
      ])

let smaller = [| Dtype Nx.float32; Dtype Nx.float16; Dtype Nx.bfloat16 |]

(* A family's log mass at float32, float16 and bfloat16, against float64 at
   parameters rounded to the dtype: [u] places the scale parameter between
   [1e-3] and the largest the dtype holds with room, [w] the other one. The
   narrower dtypes compute at float32 and round once, half an ulp more, and a
   least subnormal's half where the mass is subnormal. *)
let count_families = [| Poisson; Binomial; Neg_binomial |]

let family_name = function
  | Poisson -> "poisson"
  | Binomial -> "binomial"
  | Neg_binomial -> "neg_binomial"

let narrow_masses =
  prop "a log mass at narrow dtypes is within its bound" ~count:300
    ~examples:
      (List.concat_map
         (fun k ->
           List.map
             (fun (w, at) -> (k, (0, 0.5, w, at)))
             [
               (0.2, Zero);
               (0.8, Top);
               (0.2, Below 0.3);
               (0.8, Above 5.);
               (0.5, Near 0.);
             ])
         [ 0; 1; 2 ])
    Gen.(
      pair (int_range 0 2)
        (quad (int_range 0 2) (float_range 0. 1.) (float_range 0. 1.) place))
    (fun (k, (i, u, w, at)) ->
      let family = count_families.(k) in
      let (Dtype dt) = smaller.(i) in
      let format = Nx.Dtype.float_format dt in
      let top = top_of Nx.float32 in
      let most = Float.min top (format.max_finite /. 64.) in
      let round v = Nx.item [] (Nx.cast Nx.float64 (Nx.scalar dt v)) in
      let span lo hi = lo *. ((hi /. lo) ** u) in
      let a, b =
        match family with
        | Poisson -> (round (span 1e-3 most), 0.)
        | Binomial -> (Float.round (span 1. top), round ((24. *. w) -. 12.))
        | Neg_binomial ->
            (round (span 1e-3 (most /. 16.)), round (1e-3 *. (1e7 ** w)))
      in
      let m = count_mean family a b in
      let var =
        match family with
        | Poisson -> m
        | Binomial -> m *. sigmoid (-.b)
        | Neg_binomial -> m +. (m *. m /. b)
      in
      let last = if family = Binomial then a else top in
      let x =
        match at with
        | Near z -> Float.round (m +. (z *. Float.sqrt var))
        | Below f | Above f -> Float.round (f *. m)
        | Zero -> 0.
        | Top -> last
      in
      let x = Float.min last (Float.max 0. x) in
      let f, err = classic family a b x in
      (* A mass past the dtype's range rounds to [-inf]. *)
      assume (Float.abs f < format.max_finite);
      Array.iteri (fun j (Dtype d) -> cover (Nx.Dtype.name d) (i = j)) smaller;
      Array.iteri (fun j f -> cover (family_name f) (k = j)) count_families;
      cover "at zero" (x = 0.);
      cover "near the mean" (x > m /. 2. && x < 2. *. m);
      cover "far below the mean" (x > 0. && x <= m /. 2.);
      cover "far above the mean" (x >= 2. *. m);
      cover "binomial at trials" (family = Binomial && x = a && x > 0.);
      cover "neg_binomial dispersion below 5.25"
        (family = Neg_binomial && b <= 5.25);
      cover "neg_binomial dispersion above 5.25"
        (family = Neg_binomial && b > 5.25);
      let rounding =
        if format.fraction_bits >= 23 then 0.
        else 0.5 *. format.epsilon *. (Float.abs f +. format.min_normal)
      in
      equal ~msg:(family_name family)
        (float (mass_bound family (eps_of Nx.float32) f x m +. rounding +. err))
        f
        (count_mass dt family a b x))

(* Where the means' rounding weighs most: a binomial count 4 to 6 standard
   deviations from its mean at up to [2^24] trials, at float32 against float64's
   log-gamma sums. *)
let binomial_tails =
  prop "a binomial's log mass in its tails at many trials is within its bound"
    Gen.(
      triple (float_range 0. 1.) (float_range (-6.) 6.)
        (pair bool (float_range 4. 6.)))
    (fun (u, l, (above, k)) ->
      let n = Float.round (1e4 *. ((0x1p24 /. 1e4) ** u)) in
      let l = Nx.item [] (Nx.cast Nx.float64 (Nx.scalar Nx.float32 l)) in
      let m = n *. sigmoid l in
      let sd = Float.sqrt (m *. sigmoid (-.l)) in
      let x = Float.round (if above then m +. (k *. sd) else m -. (k *. sd)) in
      assume (x >= 0. && x <= n);
      cover "above the mean" above;
      cover "below the mean" (not above);
      let f, err = classic Binomial n l x in
      equal
        (float (mass_bound Binomial (eps_of Nx.float32) f x m +. err))
        f
        (count_mass Nx.float32 Binomial n l x))

(* At means down to the dtype's least subnormal, the log masses' scores are
   finite wherever the exact ones are, at a count of [0] too, whose mass does
   not read the deviance. [lm] is the log of the mean, or the logit. *)
let small_means =
  let check (type b) (dt : (float, b) Nx.dtype) family lm x =
    let format = Nx.Dtype.float_format dt in
    let s v = Nx.scalar dt v in
    let p =
      Nx.item []
        (Nx.cast Nx.float64
           (s
              (match family with
              | Binomial -> lm
              | Poisson | Neg_binomial -> Float.exp lm)))
    in
    assume (p > 0. || family = Binomial);
    let n = 10. and r = 2. in
    let score, scale =
      match family with
      | Poisson -> ((x /. p) -. 1., (x /. p) +. 1.)
      | Binomial -> (x -. (n *. sigmoid p), x +. (n *. sigmoid p))
      | Neg_binomial ->
          (r *. (x -. p) /. (p *. (r +. p)), (x /. p) +. ((x +. r) /. (r +. p)))
    in
    let counts = i32 (int_of_float x) in
    let mass p =
      match family with
      | Poisson -> D.factors (D.poisson ~rate:p) counts
      | Binomial ->
          D.factors (D.binomial ~trials:(i32 (int_of_float n)) ~logits:p) counts
      | Neg_binomial ->
          D.factors (D.neg_binomial ~mean:p ~dispersion:(s r)) counts
    in
    let g = Nx.item [] (Nx.cast Nx.float64 (Rune.grad' mass (s p))) in
    if Float.abs score > format.max_finite then
      equal
        ~msg:(family_name family ^ " " ^ Nx.Dtype.name dt)
        float_exact
        (Float.copy_sign Float.infinity score)
        g
    else
      equal ~msg:(Nx.Dtype.name dt)
        (float ((64. *. format.epsilon *. scale) +. format.min_normal))
        score g
  in
  let lowest (type b) (dt : (float, b) Nx.dtype) =
    let format = Nx.Dtype.float_format dt in
    Float.log (format.min_normal *. format.epsilon)
  in
  prop "a score in the mean or logits is finite at small means" ~count:300
    Gen.(quad (int_range 0 2) bool (float_range 0. 1.) (int_range 0 3))
    (fun (k, float64, u, x) ->
      let family = count_families.(k) in
      let x = float_of_int x in
      Array.iteri (fun j f -> cover (family_name f) (k = j)) count_families;
      cover "at zero" (x = 0.);
      cover "above zero" (x > 0.);
      cover "float64" float64;
      cover "float32" (not float64);
      (* Logits past where their sigmoid underflows; log means from the least
         subnormal up to 0. *)
      let at (type b) (dt : (float, b) Nx.dtype) =
        let low = lowest dt in
        let lm =
          match family with
          | Binomial ->
              let edge = Float.log (Nx.Dtype.float_format dt).min_normal in
              (u -. 0.5) *. 2.4 *. Float.abs edge
          | Poisson | Neg_binomial -> low +. 0.5 -. (u *. (low +. 0.5))
        in
        check dt family lm x
      in
      if float64 then at Nx.float64 else at Nx.float32)

(* At dispersions down to the least subnormal, the negative binomial's score in
   its dispersion [r] is [Σ_(j < x) 1 / (r + j) + log (r / (r + mean)) + (mean -
   x) / (r + mean)], finite at [0] and overflowing above it where [1 / r] does;
   its score in the mean stays small. *)
let small_dispersions =
  prop "the negative binomial dispersion's score is finite at small dispersions"
    Gen.(quad bool (float_range 0. 1.) (int_range 0 2) (int_range 0 3))
    (fun (float64, u, k, x) ->
      let mean = [| 0.5; 3.; 1000. |].(k) and x = float_of_int x in
      cover "at zero" (x = 0.);
      cover "above zero" (x > 0.);
      cover "float64" float64;
      cover "float32" (not float64);
      let at (type b) (dt : (float, b) Nx.dtype) =
        let format = Nx.Dtype.float_format dt in
        let low = Float.log (format.min_normal *. format.epsilon) in
        let r = Float.exp (low +. 0.5 -. (u *. (low +. 0.5))) in
        let r = Nx.item [] (Nx.cast Nx.float64 (Nx.scalar dt r)) in
        assume (r > 0.);
        let terms =
          List.init (int_of_float x) (fun j -> 1. /. (r +. float_of_int j))
          @ [ Float.log r -. Float.log (r +. mean); (mean -. x) /. (r +. mean) ]
        in
        let score = List.fold_left ( +. ) 0. terms in
        let scale = List.fold_left (fun a t -> a +. Float.abs t) 0. terms in
        let mass r =
          D.factors
            (D.neg_binomial ~mean:(Nx.scalar dt mean) ~dispersion:r)
            (i32 (int_of_float x))
        in
        let g =
          Nx.item [] (Nx.cast Nx.float64 (Rune.grad' mass (Nx.scalar dt r)))
        in
        if Float.abs score > format.max_finite then
          equal ~msg:(Nx.Dtype.name dt) float_exact Float.infinity g
        else
          equal ~msg:(Nx.Dtype.name dt)
            (float ((64. *. format.epsilon *. scale) +. format.min_normal))
            score g;
        (* The mean's score, [r (x - mean) / (mean (r + mean))], is small. *)
        let by_mean mean =
          D.factors
            (D.neg_binomial ~mean ~dispersion:(Nx.scalar dt r))
            (i32 (int_of_float x))
        in
        let gm = Rune.grad' by_mean (Nx.scalar dt mean) in
        let gm = Nx.item [] (Nx.cast Nx.float64 gm) in
        let along = r *. (x -. mean) /. (mean *. (r +. mean)) in
        let scale = r *. (x +. mean) /. (mean *. (r +. mean)) in
        equal
          ~msg:(Nx.Dtype.name dt ^ " mean")
          (float ((64. *. format.epsilon *. scale) +. format.min_normal))
          along gm
      in
      if float64 then at Nx.float64 else at Nx.float32)

(* [rune]'s gradient of [f] at [p], against [score]. *)
let grad_of f p = item (Rune.grad' (fun p -> f p) (f64 p))

(* The masses' scores against their closed forms, within [16 ε] of the
   magnitudes [T] they sum. Rune's derivative of Loader's form sums the
   derivatives of [D(x, m)] and of the mean in the parameter, as [1 - x / m]
   times [m]'s: the ratios [x / m] and [1] and the mean's derivative, each
   within [2 u] of itself, [u = ε / 2], and the sum's rounding within [2 u] of
   their magnitudes, which are [T]: [4 ε T], and the closed form's own ratio and
   sums [2 ε T]. *)
let scores =
  let count = Gen.(pair (float_range 0. 1.) place) in
  let x_of m sd = function
    | Near z -> Float.max 0. (Float.round (m +. (z *. sd)))
    | Below f | Above f -> Float.round (f *. m)
    | Zero -> 0.
    | Top -> Float.round (20. *. m)
  in
  group "scores"
    [
      prop "the poisson rate's score is x / rate - 1"
        ~examples:[ (0.5, Near 0.) ]
        count
        (fun (u, at) ->
          let rate = 1e-3 *. (1e9 ** u) in
          let x = x_of rate (Float.sqrt rate) at in
          cover "near the mean" (x > rate /. 2. && x < 2. *. rate);
          cover "far from the mean" (x <= rate /. 2. || x >= 2. *. rate);
          let g =
            grad_of
              (fun rate -> D.factors (D.poisson ~rate) (i32 (int_of_float x)))
              rate
          in
          equal
            (float (16. *. epsilon_float *. ((x /. rate) +. 1.)))
            ((x /. rate) -. 1.)
            g);
      test "the bernoulli logit's score at 0 is x - 1/2" (fun () ->
          let score x =
            grad_of
              (fun logits ->
                D.factors (D.bernoulli ~logits) (Nx.scalar Nx.bool x))
              0.
          in
          equal
            (pair float_exact float_exact)
            (-0.5, 0.5)
            (score false, score true));
      prop "the binomial logits' score is x - trials sigmoid logits"
        ~examples:[ ((0., Zero), 0.) ]
        Gen.(pair count (float_range (-12.) 12.))
        (fun ((u, at), l) ->
          let n = Float.round (1e7 ** u) in
          let m = n *. sigmoid l in
          let x = Float.min n (x_of m (Float.sqrt (m *. sigmoid (-.l))) at) in
          cover "near the mean" (x > m /. 2. && x < 2. *. m);
          cover "far from the mean" (x <= m /. 2. || x >= 2. *. m);
          let trials = i32 (int_of_float n) in
          let g =
            grad_of
              (fun logits ->
                D.factors (D.binomial ~trials ~logits) (i32 (int_of_float x)))
              l
          in
          equal (float (16. *. epsilon_float *. (x +. m))) (x -. m) g);
      prop
        "the negative binomial mean's score is r (x - mean) / (mean (r + mean))"
        Gen.(pair count (float_range 0. 1.))
        (fun ((u, at), w) ->
          let mean = 1e-3 *. (1e9 ** u) and r = 1e-3 *. (1e10 ** w) in
          let x = x_of mean (Float.sqrt (mean +. (mean *. mean /. r))) at in
          cover "dispersion below 5.25" (r <= 5.25);
          cover "dispersion above 5.25" (r > 5.25);
          let dispersion = f64 r in
          let g =
            grad_of
              (fun mean ->
                D.factors
                  (D.neg_binomial ~mean ~dispersion)
                  (i32 (int_of_float x)))
              mean
          in
          let score = r *. (x -. mean) /. (mean *. (r +. mean)) in
          let scale = (x /. mean) +. ((x +. r) /. (r +. mean)) in
          equal (float (16. *. epsilon_float *. scale)) score g);
      (* Central differences at a step [h = r 1e-5]: truncation near [h² |f'''|
         / 6], about [3e-11 / r], and the masses' errors over [h], each within
         its bound. *)
      prop "the negative binomial dispersion's gradient is its slope"
        Gen.(pair count (float_range 0. 1.))
        (fun ((u, at), w) ->
          let mean = 1e-3 *. (1e9 ** u) and r = 1e-3 *. (1e10 ** w) in
          let x = x_of mean (Float.sqrt (mean +. (mean *. mean /. r))) at in
          cover "dispersion below 5.25" (r <= 5.25);
          cover "dispersion above 5.25" (r > 5.25);
          let f r =
            D.factors
              (D.neg_binomial ~mean:(f64 mean) ~dispersion:r)
              (i32 (int_of_float x))
          in
          let h = r *. 1e-5 in
          let at r = item (f (f64 r)) in
          let slope = (at (r +. h) -. at (r -. h)) /. (2. *. h) in
          let noise =
            mass_bound Neg_binomial (eps_of Nx.float64) (at r) x mean /. h
          in
          equal
            (float ((1e-6 *. Float.abs slope) +. noise +. (1e-10 /. r)))
            slope (grad_of f r));
    ]

let compiled_masses =
  test "a compiled log mass is within its bound" (fun () ->
      Compiled.here ();
      List.iter
        (fun (name, rows) ->
          subtest name (fun () ->
              check_masses ~run:Rune.jit' (count_family name)
                (List.map floats rows)))
        (families "mass"))

let masses =
  group "count masses"
    [
      golden_masses;
      narrow_masses;
      binomial_tails;
      small_means;
      small_dispersions;
      test "a negative binomial at a subnormal mean and dispersion" (fun () ->
          (* At [x = 1], [log (r Γ(r) / Γ(r)) + r log (1/2) + log (1/2)]. *)
          let r = 1e-310 in
          let d = D.neg_binomial ~mean:(f64 r) ~dispersion:(f64 r) in
          let exact = Float.log r +. ((1. +. r) *. Float.log 0.5) in
          equal
            (float (mass_bound Neg_binomial (eps_of Nx.float64) exact 1. r))
            exact
            (item (D.factors d (i32 1))));
      scores;
      Windtrap.cases "a zero mean puts all mass at zero"
        ~id:(fun (name, _) -> name)
        [
          ("poisson", D.poisson ~rate:(f64 0.));
          ("neg_binomial", D.neg_binomial ~mean:(f64 0.) ~dispersion:(f64 2.));
        ]
        (fun (_, d) ->
          let counts = Nx.create Nx.int32 [| 4 |] [| -1l; 0l; 1l; 100l |] in
          equal
            (array (float Float.min_float))
            [| Float.neg_infinity; 0.; Float.neg_infinity; Float.neg_infinity |]
            (Nx.to_array (D.factors d counts)));
      compiled_masses;
    ]

(* Mostly finite logits, often infinite ones. *)
let logit =
  Gen.frequency
    [
      (6, Gen.float_range (-30.) 30.);
      (1, Gen.constant Float.infinity);
      (1, Gen.constant Float.neg_infinity);
    ]

(* The examples reach the covers that a sixth of the draws reach. *)
let binomial_mass =
  prop "a binomial's mass sums to one over {0, ..., trials}"
    ~examples:[ (0, 0.3) ]
    Gen.(
      pair
        (frequency
           [ (1, constant 0); (1, int_range 1 3); (4, int_range 4 300) ])
        logit)
    (fun (n, l) ->
      cover "no trials" (n = 0);
      cover "an infinite logit" (Float.abs l = Float.infinity);
      cover "a finite logit" (Float.is_finite l);
      let d = D.binomial ~trials:(i32 n) ~logits:(f64 l) in
      let xs = Nx.arange Nx.int32 0 (n + 1) 1 in
      let f = D.factors d xs in
      (* Each mass within its bound, ε (24 |f| + 9 |x - m|) relatively, and the
         sum's rounding within ε per term. *)
      let m = float_of_int n *. sigmoid l in
      let tol =
        Array.fold_left ( +. )
          (float_of_int (n + 1) *. epsilon_float)
          (Array.mapi
             (fun x f ->
               if Float.is_finite f then
                 Float.exp f *. epsilon_float
                 *. ((24. *. Float.abs f)
                    +. (9. *. Float.abs (float_of_int x -. m)))
               else 0.)
             (Nx.to_array f))
      in
      equal (float tol) 1. (item (Nx.sum (Nx.exp f))))

let binomial_bernoulli =
  prop "a binomial of one trial is a bernoulli" logit (fun l ->
      let b = D.bernoulli ~logits:(f64 l) in
      let d = D.binomial ~trials:(i32 1) ~logits:(f64 l) in
      let flags = Nx.create Nx.bool [| 2 |] [| false; true |] in
      let counts = Nx.create Nx.int32 [| 2 |] [| 0l; 1l |] in
      equal
        (array (float_rel ~rel:(64. *. epsilon_float) ~abs:Float.min_float))
        (Nx.to_array (D.factors b flags))
        (Nx.to_array (D.factors d counts)))

(* The mean [n p] and variance [n p q] of [m] draws, each within [z] of its
   standard error: [sqrt (n p q / m)] for the mean and, for the variance, [sqrt
   ((mu4 - (n p q)^2) / m)] with [mu4 = n p q (1 + 3 p q (n - 2))]. At [z = 5.3]
   each check fails falsely with probability about 1.2e-7, the eight together
   below 1e-6. *)
let binomial_moments =
  let m = 20000 and z = 5.3 in
  Windtrap.cases "a binomial's draws have its mean and variance"
    ~id:(fun (n, l) -> Printf.sprintf "trials %d, logits %g" n l)
    [ (12, 0.4); (1000, -6.); (1_000_000, 3.); (50, 8.) ]
    (fun (n, l) ->
      let d = D.iid [| m |] (D.binomial ~trials:(i32 n) ~logits:(f64 l)) in
      let xs =
        Array.map Int32.to_float (Nx.to_array (D.sample (Nx.Rng.key n) d))
      in
      let fm = float_of_int m and fn = float_of_int n in
      let mean = Array.fold_left ( +. ) 0. xs /. fm in
      let var =
        Array.fold_left (fun a x -> a +. ((x -. mean) ** 2.)) 0. xs /. (fm -. 1.)
      in
      let p = sigmoid l and q = sigmoid (-.l) in
      let v = fn *. p *. q in
      let mu4 = v *. (1. +. (3. *. p *. q *. (fn -. 2.))) in
      equal (float (z *. Float.sqrt (v /. fm))) (fn *. p) mean;
      equal (float (z *. Float.sqrt ((mu4 -. (v *. v)) /. fm))) v var)

let binomial_edges =
  Windtrap.cases "a binomial's point masses"
    ~id:(fun (name, _, _, _) -> name)
    [
      ("p = 0", 5, Float.neg_infinity, 0);
      ("p = 1", 5, Float.infinity, 5);
      ("no trials", 0, 0.3, 0);
    ]
    (fun (_, n, l, at) ->
      let d = D.binomial ~trials:(i32 n) ~logits:(f64 l) in
      let xs = Array.init 8 (fun i -> i - 1) in
      let counts = Nx.create Nx.int32 [| 8 |] (Array.map Int32.of_int xs) in
      equal (array float_exact)
        (Array.map (fun x -> if x = at then 0. else Float.neg_infinity) xs)
        (Nx.to_array (D.factors d counts));
      equal (array int32)
        (Array.make 100 (Int32.of_int at))
        (Nx.to_array (D.sample (Nx.Rng.key 1) (D.iid [| 100 |] d))))

let binomial =
  group "binomial"
    [
      binomial_mass;
      binomial_bernoulli;
      binomial_moments;
      binomial_edges;
      test "a count above an element's trials has factor -inf" (fun () ->
          let d = D.binomial ~trials:trials_3_7 ~logits:(f64 0.) in
          let seven = D.binomial ~trials:(i32 7) ~logits:(f64 0.) in
          equal (array float_exact)
            [| Float.neg_infinity; item (D.factors seven (i32 5)) |]
            (Nx.to_array (D.factors d (i32 5))));
      (* A draw is [Nx.Rng.binomial] at [sigmoid (-|logits|)], mirrored by
         [where] and a subtraction. Rune's interface states exact on a faithful
         device, as the host is, the draws of [Nx.Rng] and [sigmoid], which nx
         composes from [exp], a division and exact operations, so compiled a
         draw gives eager's bits. *)
      test "a compiled draw is the eager one" (fun () ->
          Compiled.here ();
          let d logits = D.iid [| 50 |] (D.binomial ~trials:(i32 30) ~logits) in
          let k = Nx.Rng.key 4 in
          equal (array int32)
            (Nx.to_array (D.sample k (d (f64 0.5))))
            (Nx.to_array (Rune.jit' (fun l -> D.sample k (d l)) (f64 0.5))));
    ]

let categorical =
  group "categorical"
    [
      test "a -inf logit's category has factor -inf and is never drawn"
        (fun () ->
          let d =
            D.categorical ~logits:(vec [| 0.; Float.neg_infinity; 1. |])
          in
          equal float_exact Float.neg_infinity
            (item (D.factors d (Nx.scalar Nx.int64 1L)));
          let draws =
            Nx.to_array (D.sample (Nx.Rng.key 9) (D.iid [| 1000 |] d))
          in
          satisfies ~claim:"never category 1" (array int64)
            (Array.for_all (fun k -> k <> 1L))
            draws);
      test "scalar logits raise naming categorical" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.categorical:")
            (fun () -> D.categorical ~logits:(f64 0.)));
      test "logits of no category raise naming categorical" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.categorical:")
            (fun () -> D.categorical ~logits:(Nx.zeros Nx.float64 [| 0 |])));
    ]

(* Combinators *)

let iid =
  group "iid"
    [
      test "prepends its shape" (fun () ->
          equal (array int) [| 4; 2; 3 |]
            (D.shape
               (D.iid [| 4; 2 |]
                  (D.dirichlet ~concentration:(vec [| 1.5; 2.; 0.7 |])))));
      test "of no copy has log density 0" (fun () ->
          let d = D.iid [| 0 |] std in
          equal float_exact 0.
            (item (D.log_density d (Nx.zeros Nx.float64 [| 0 |]))));
      test "a negative length raises naming iid" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.iid:") (fun () ->
              D.iid [| 2; -1 |] std));
    ]

let sorted =
  group "sorted"
    [
      test "its density is n! times the product" (fun () ->
          let x = vec [| -1.; 0.25; 2. |] in
          let expected =
            Float.log 6. +. item (D.log_density (D.iid [| 3 |] std) x)
          in
          equal
            (float (8. *. epsilon_float *. Float.abs expected))
            expected
            (item (D.log_density (D.sorted 3 std) x)));
      Windtrap.cases "a decreasing pair has density -inf, a tie does not"
        ~id:(fun (x, _) ->
          Format.asprintf "%a" (Testable.pp (array float_exact)) x)
        [ ([| 1.; 0.; 2. |], false); ([| 1.; 1.; 2. |], true) ]
        (fun (x, inside) ->
          let f = item (D.log_density (D.sorted 3 std) (vec x)) in
          if inside then satisfies ~claim:"finite" float_exact Float.is_finite f
          else equal float_exact Float.neg_infinity f);
      test "its draws are sorted" (fun () ->
          let x =
            Nx.to_array
              (D.sample (Nx.Rng.key 12) (D.iid [| 200 |] (D.sorted 5 std)))
          in
          satisfies ~claim:"each draw nondecreasing" (array float_exact)
            (fun x ->
              let ok = ref true in
              for i = 0 to 199 do
                for j = 1 to 4 do
                  if x.((i * 5) + j) < x.((i * 5) + j - 1) then ok := false
                done
              done;
              !ok)
            x);
      test "a vector distribution raises naming sorted" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.sorted:") (fun () ->
              D.sorted 2 (D.iid [| 2 |] std)));
      test "no element raises naming sorted" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.sorted:") (fun () ->
              D.sorted 0 std));
    ]

let transform =
  group "transform"
    [
      (* The exp of a normal is the lognormal of its location and scale: two
         computations of one density, each within the closed forms' bound. *)
      prop "the exp of a normal is a lognormal"
        ~examples:[ 0.; -0.; -1.; Float.infinity; 1e-300 ]
        Gen.(float_range 1e-6 1e6)
        (fun x ->
          let t =
            D.transform Norn.Bij.exp (D.normal ~loc:(f64 0.3) ~scale:(f64 0.8))
          in
          let l = D.lognormal ~loc:(f64 0.3) ~scale:(f64 0.8) in
          let expected = item (D.factors l (f64 x)) in
          let t_x = item (D.factors t (f64 x)) in
          if Float.is_finite expected then
            let lx = Float.log x in
            let z = (lx -. 0.3) /. 0.8 in
            let terms =
              (z *. z /. 2.) +. Float.abs lx +. 1. +. Float.abs (z *. lx /. 0.8)
            in
            equal (float (16. *. epsilon_float *. terms)) expected t_x
          else equal float_exact expected t_x);
      test "a value outside its bijector's image has density -inf" (fun () ->
          let t = D.transform Norn.Bij.exp std in
          equal float_exact Float.neg_infinity (item (D.factors t (f64 (-1.)))));
    ]

(* Truncation *)

(* [check_quantiles ~run name rows] checks gen/dist.py's case [name] at its
   golden [p] in one call through [run], [D.quantile] itself or its compilation,
   within norn.mli's bound: [m] and [f] the tail and the density of the case's
   base at the quantile, which gen/dist.py gives as logarithms. *)
let check_quantiles ~run name rows =
  let ps = vec (Array.of_list (List.map List.hd rows)) in
  let xs = Nx.to_array (run (D.quantile (continuous_case name)) ps) in
  List.iteri
    (fun i row ->
      match row with
      | [ p; v; log_m; log_f ] ->
          equal
            ~msg:(Printf.sprintf "p = %h" p)
            (float (quantile_bound ~x:v ~log_m ~log_f))
            v xs.(i)
      | _ -> failwith "dist.golden: a malformed ppf line")
    rows

(* [check_truncation ~run name rows] checks gen/dist.py's truncation [name] at
   its golden points in one call through [run], [D.factors] itself or its
   compilation: each log density within norn.mli's bound, [-inf] exactly
   outside. *)
let check_truncation ~run name rows =
  let d = continuous_case name in
  let base, _, _ = List.assoc name truncations in
  let xs = vec (Array.of_list (List.map List.hd rows)) in
  let fs = Nx.to_array (run (D.factors d) xs) in
  let bs = Nx.to_array (D.factors (continuous_case base) xs) in
  List.iteri
    (fun i row ->
      match row with
      | [ x; v ] ->
          let msg = Printf.sprintf "x = %h" x in
          if Float.is_finite v then
            equal ~msg
              (float (truncation_bound epsilon_float ~base:bs.(i) ~v))
              v fs.(i)
          else equal ~msg float_exact v fs.(i)
      | _ -> failwith "dist.golden: a malformed logpdf line")
    rows

let truncation =
  let normal = D.normal ~loc:(f64 0.5) ~scale:(f64 2.) in
  group "truncated"
    [
      test "log densities are within their bound of mpmath's" (fun () ->
          List.iter
            (fun (name, rows) ->
              subtest name (fun () ->
                  check_truncation
                    ~run:(fun f x -> f x)
                    name (List.map floats rows)))
            (families "logpdf"));
      test "its draws lie within the bounds" (fun () ->
          let d = D.truncated ~low:(f64 60.5) ~high:(f64 61.) normal in
          let xs = Nx.to_array (D.sample (Nx.Rng.key 7) (D.iid [| 1000 |] d)) in
          satisfies ~claim:"within (60.5, 61)" (array float_exact)
            (Array.for_all (fun x -> x > 60.5 && x < 61.))
            xs);
      (* Newton's steps from the bound 1e-130 once crossed the gamma's 0. *)
      test "its draws lie in the support where a bound is deep in a tail"
        (fun () ->
          let d = continuous_case "truncated_gamma_deep" in
          satisfies ~claim:"within (0, 1e-130)" (array float_exact)
            (Array.for_all (fun v -> v > 0. && v < 1e-130))
            (Nx.to_array (D.sample (Nx.Rng.key 5) (D.iid [| 1000 |] d))));
      (* [∂loc f = (z + (φ(b) - φ(a)) / (Φ(b) - Φ(a))) / σ] for [z = (x - loc) /
         σ] and the standardised bounds [a] and [b]. *)
      prop "the score of a truncated normal's location"
        Gen.(
          triple (float_range (-3.) 3.) (float_range 0. 4.)
            (float_range 0.01 0.99))
        (fun (a, w, u) ->
          let b = a +. w +. 0.01 in
          let x = a +. (u *. (b -. a)) in
          let g =
            item
              (Rune.grad'
                 (fun loc ->
                   D.factors
                     (D.truncated ~low:(f64 a) ~high:(f64 b)
                        (D.normal ~loc ~scale:(f64 1.)))
                     (f64 x))
                 (f64 0.))
          in
          let phi t =
            Float.exp (-.(t *. t) /. 2.) /. Float.sqrt (2. *. Float.pi)
          in
          let cdf t = item (Nx.ndtr (f64 t)) in
          (* The mass from the smaller tails, which keep its digits, as their
             difference: it loses [(F₁ + F₂) / mass] of [ε], which the ratio
             carries; each term is within a few roundings, [64 ε]. *)
          let f1, f2 =
            if a > 0. then (cdf (-.a), cdf (-.b)) else (cdf b, cdf a)
          in
          let mass = f1 -. f2 in
          let ratio = (phi b -. phi a) /. mass in
          let expected = x +. ratio in
          let spread = 1. +. ((f1 +. f2) /. mass) in
          equal
            (float
               (64. *. epsilon_float
               *. (1. +. Float.abs x +. (Float.abs ratio *. spread))))
            expected g);
      (* gen/dist.py's truncated laplace on (-50, -40), whose tails are closed
         forms: a normal's, read at each of the quantile's Newton steps, compile
         to a kernel of 700 KB that took a loaded runner past 60 s. *)
      test "a compiled density and quantile are within their bounds" (fun () ->
          Compiled.here ();
          let rows kind =
            List.map floats (List.assoc "truncated_laplace" (families kind))
          in
          check_truncation ~run:Rune.jit' "truncated_laplace" (rows "logpdf");
          check_quantiles ~run:Rune.jit' "truncated_laplace" (rows "ppf"));
      test "a bound outside the support leaves no mass and raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"log mass between the bounds")
            (fun () ->
              D.log_density
                (D.truncated ~high:(f64 (-1.))
                   (D.gamma ~concentration:(f64 2.) ~rate:(f64 1.)))
                (f64 (-2.))));
      test "its quantile is the bound or the open side's edge at p = 0 and 1"
        (fun () ->
          let q d p = item (D.quantile d (f64 p)) in
          let lo = D.truncated ~low:(f64 60.5) normal in
          let hi = D.truncated ~high:(f64 (-79.5)) normal in
          let both = D.truncated ~low:(f64 (-1.)) ~high:(f64 3.) normal in
          equal (array float_exact)
            [| 60.5; Float.infinity; Float.neg_infinity; -79.5; -1.; 3. |]
            [| q lo 0.; q lo 1.; q hi 0.; q hi 1.; q both 0.; q both 1. |]);
      (* A window too narrow for the distribution function to resolve: its mass
         is [(erf (h / √2) - erf (l / √2)) / 2], exact from [0], or [φ] at the
         window's middle times its width elsewhere: at [30] the log density
         moves by [3e-5] across it, [4e-11] in the mass. *)
      test "a narrow window's mass is its density's integral" (fun () ->
          let erf v = item (Nx.erf (f64 v)) in
          let log_phi v =
            (-0.5 *. v *. v) -. (0.5 *. Float.log (2. *. Float.pi))
          in
          let exact l h x =
            log_phi x
            -. Float.log
                 ((erf (h /. Float.sqrt 2.) -. erf (l /. Float.sqrt 2.)) /. 2.)
          in
          let at (type b) (dt : (float, b) Nx.dtype) l h x =
            let c = Nx.scalar dt in
            let d =
              D.truncated ~low:(c l) ~high:(c h)
                (D.normal ~loc:(c 0.) ~scale:(c 1.))
            in
            Nx.item [] (Nx.cast Nx.float64 (D.factors d (c x)))
          in
          let f32 = (Nx.Dtype.float_format Nx.float32).epsilon in
          let bound eps x v =
            float (truncation_bound eps ~base:(log_phi x) ~v)
          in
          let v = exact 0. 1e-5 5e-6 in
          equal ~msg:"float32 [0, 1e-5]" (bound f32 5e-6 v) v
            (at Nx.float32 0. 1e-5 5e-6);
          let v = exact 0. 1e-8 5e-9 in
          equal ~msg:"float32 [0, 1e-8]" (bound f32 5e-9 v) v
            (at Nx.float32 0. 1e-8 5e-9);
          (* At float64 the erf difference cancels too: across the window the
             log density moves by [3e-10], so the mass is [φ] at its middle
             times its width, exact in floats (Sterbenz). *)
          let l = 0.3 and h = 0.3 +. 1e-9 in
          let mid = (l +. h) /. 2. in
          let v = log_phi mid -. (log_phi mid +. Float.log (h -. l)) in
          equal ~msg:"float64 [0.3, 0.3 + 1e-9]"
            (bound epsilon_float mid v)
            v (at Nx.float64 l h mid);
          let l = 30. and h = 30. +. 1e-6 in
          let mid = (l +. h) /. 2. in
          (* the reference's own error, [4e-11] in the mass, adds to the
             bound *)
          equal ~msg:"float64 [30, 30 + 1e-6]"
            (float
               (truncation_bound epsilon_float ~base:(log_phi mid)
                  ~v:(-.Float.log (h -. l))
               +. 4e-11))
            (-.Float.log (h -. l))
            (item
               (D.factors
                  (D.truncated ~low:(f64 l) ~high:(f64 h) std)
                  (f64 mid))));
      (* At bounds 10^6 standard deviations out the base's log density and the
         log mass are both about [-5e11]: float32 and float64 each within
         norn.mli's bound of the exact value. *)
      test "far bounds keep the base's absolute precision" (fun () ->
          let at (type b) (dt : (float, b) Nx.dtype) =
            let c = Nx.scalar dt in
            let d =
              D.truncated ~low:(c 1e6) (D.normal ~loc:(c 0.) ~scale:(c 1.))
            in
            Nx.item [] (Nx.cast Nx.float64 (D.factors d (c 1e6)))
          in
          let eps = (Nx.Dtype.float_format Nx.float32).epsilon in
          let v = at Nx.float64 and base = -5e11 in
          equal
            (float
               (truncation_bound eps ~base ~v
               +. truncation_bound epsilon_float ~base ~v))
            v (at Nx.float32));
      test "no bound raises naming truncated" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.truncated:") (fun () ->
              D.truncated normal));
      test "a family without a distribution function raises naming truncated"
        (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.truncated:") (fun () ->
              D.truncated ~low:(f64 0.)
                (D.student_t ~df:(f64 3.5) ~loc:(f64 0.) ~scale:(f64 1.))));
      test "a vector of bounds truncates each element at its own" (fun () ->
          let d =
            D.truncated ~low:(vec [| -1.; 0.5 |]) (D.iid [| 2 |] normal)
          in
          let f = Nx.to_array (D.factors d (vec [| 0.; 0. |])) in
          equal float_exact Float.neg_infinity f.(1);
          equal float_exact
            (item (D.factors (D.truncated ~low:(f64 (-1.)) normal) (f64 0.)))
            f.(0));
      test "a window of no mass is not valid" (fun () ->
          let d =
            D.truncated ~high:(f64 (-1.))
              (D.gamma ~concentration:(f64 2.) ~rate:(f64 1.))
          in
          equal (array bool) [| false |] [| Nx.item [] (D.valid d) |]);
    ]

let mixture =
  let two = D.normal ~loc:(vec [| -1.; 2. |]) ~scale:(vec [| 0.5; 1. |]) in
  group "mixture"
    [
      test "its density sums its weighted components" (fun () ->
          let logits = vec [| 0.3; -0.2 |] in
          let d = D.mixture ~logits two in
          let x = 0.7 in
          let w = Nx.to_array (Nx.softmax logits) in
          let p loc scale =
            Float.exp
              (item
                 (D.log_density
                    (D.normal ~loc:(f64 loc) ~scale:(f64 scale))
                    (f64 x)))
          in
          let expected =
            Float.log ((w.(0) *. p (-1.) 0.5) +. (w.(1) *. p 2. 1.))
          in
          (* two normal terms within [8 ε Σ |t|] of moderate terms, their
             weights and log-sum-exp a few roundings more *)
          equal
            (float (8. *. epsilon_float *. (1. +. Float.abs expected)))
            expected
            (item (D.log_density d (f64 x))));
      (* Components of three elements at locations -100 and 100: a draw takes
         one component whole, so its elements share a sign. *)
      test "a draw takes one component for the whole value" (fun () ->
          let comps =
            D.normal
              ~loc:
                (Nx.create Nx.float64 [| 2; 3 |]
                   [| -100.; -100.; -100.; 100.; 100.; 100. |])
              ~scale:(f64 1.)
          in
          let d =
            D.iid [| 200 |] (D.mixture ~logits:(vec [| 0.; 0. |]) comps)
          in
          let x = Nx.to_array (D.sample (Nx.Rng.key 13) d) in
          satisfies ~claim:"each draw's three elements share a sign"
            (array float_exact)
            (fun x ->
              let ok = ref true in
              for i = 0 to 199 do
                let s = Float.sign_bit x.(3 * i) in
                for j = 1 to 2 do
                  if Float.sign_bit x.((3 * i) + j) <> s then ok := false
                done
              done;
              !ok)
            x);
      test "a component per point is iid of a mixture" (fun () ->
          let d =
            D.iid [| 3 |]
              (D.mixture
                 ~logits:(vec [| 0.; 0. |])
                 (D.normal ~loc:(vec [| -1.; 1. |]) ~scale:(f64 1.)))
          in
          let x = vec [| -1.; 0.; 1. |] in
          equal (array int) [| 3 |] (Nx.shape (D.factors d x));
          let m = D.membership d x in
          equal (array int) [| 3; 2 |] (Nx.shape m);
          (* two equal joint terms: a softmax of one rounding each *)
          equal (float (2. *. epsilon_float)) 0.5 (Nx.item [ 1; 0 ] m));
      test "its draws have its shape" (fun () ->
          let d =
            D.mixture
              ~logits:(vec [| 0.; 0. |])
              (D.normal ~loc:(vec [| -1.; 1. |]) ~scale:(f64 1.))
          in
          equal (array int) [||] (Nx.shape (D.sample (Nx.Rng.key 0) d));
          equal (array int) [| 5 |]
            (Nx.shape (D.sample (Nx.Rng.key 0) (D.iid [| 5 |] d))));
      test "a mixture over trials has the density of its components" (fun () ->
          let trials = trials_3_7 in
          let d =
            D.mixture
              ~logits:(vec [| 0.; 0. |])
              (D.binomial ~trials ~logits:(f64 0.))
          in
          (* 5 is outside the first component's support. *)
          let seven = D.binomial ~trials:(i32 7) ~logits:(f64 0.) in
          let expected = Float.log 0.5 +. item (D.log_density seven (i32 5)) in
          equal
            (float (8. *. epsilon_float *. Float.abs expected))
            expected
            (item (D.log_density d (i32 5))));
      test "logits of another length raise naming mixture" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.mixture:") (fun () ->
              D.mixture ~logits:(Nx.zeros Nx.float64 [| 3 |]) two));
      test "logits that are not a vector raise naming mixture" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.mixture:") (fun () ->
              D.mixture ~logits:(f64 0.) two));
    ]

(* Hidden Markov models *)

(* [enumerate k n] is every sequence of [n] states out of [k]. *)
let rec enumerate k n =
  if n = 0 then [ [] ]
  else
    List.concat_map
      (fun s -> List.init k (fun j -> j :: s))
      (enumerate k (n - 1))

let log_softmax xs =
  let m = Array.fold_left Float.max Float.neg_infinity xs in
  let s = Array.fold_left (fun a x -> a +. Float.exp (x -. m)) 0. xs in
  Array.map (fun x -> x -. m -. Float.log s) xs

(* The joint log density of the states [z] and the observations [e] read as
   emission log densities [e.(t).(j)], from the chain's log probabilities. *)
let path_density init trans e z =
  let z = Array.of_list z in
  let li = log_softmax init and lt = Array.map log_softmax trans in
  let s = ref (li.(z.(0)) +. e.(0).(z.(0))) in
  for t = 1 to Array.length z - 1 do
    s := !s +. lt.(z.(t - 1)).(z.(t)) +. e.(t).(z.(t))
  done;
  !s

(* A row of logits with at least one finite. *)
let logits k =
  Gen.(
    map
      (fun xs ->
        if Array.for_all (fun x -> x = Float.neg_infinity) xs then xs.(0) <- 0.;
        xs)
      (array ~size:(constant k)
         (frequency
            [ (8, float_range (-3.) 3.); (1, constant Float.neg_infinity) ])))

let gen_hmm =
  Gen.(
    let* k = int_range 1 3 in
    let* n = int_range 1 5 in
    let+ init = logits k
    and+ trans = array ~size:(constant k) (logits k)
    and+ locs = array ~size:(constant k) (float_range (-3.) 3.)
    and+ ys = array ~size:(constant n) (float_range (-4.) 4.) in
    (k, n, init, trans, locs, ys))

let hmm_of k n init trans locs =
  D.hmm ~initial:(vec init)
    ~transition:
      (Nx.create Nx.float64 [| k; k |] (Array.concat (Array.to_list trans)))
    n
    (D.normal ~loc:(vec locs) ~scale:(f64 0.8))

let normal_lp y l =
  let z = (y -. l) /. 0.8 in
  (-0.5 *. z *. z) -. Float.log 0.8 -. (0.5 *. Float.log (2. *. Float.pi))

(* The log density summed over every state sequence. *)
let brute k n init trans locs ys =
  let e = Array.map (fun y -> Array.map (normal_lp y) locs) ys in
  logsumexp
    (Array.of_list (List.map (path_density init trans e) (enumerate k n)))

(* The forward algorithm and the enumeration each sum [n] steps of [k]-term
   log-sum-exps, each within a few roundings of the running magnitude, which [1
   + |v|] bounds: [32 n k ε (1 + |v|)] holds both. *)
let hmm_bound n k v =
  32. *. float_of_int (n * k) *. epsilon_float *. (1. +. Float.abs v)

let hmm_tests =
  group "hmm"
    [
      prop "its log density is the sum over its state sequences" gen_hmm
        (fun (k, n, init, trans, locs, ys) ->
          cover "one state" (k = 1);
          cover "one step" (n = 1);
          cover "a forbidden transition"
            (Array.exists
               (Array.exists (fun x -> x = Float.neg_infinity))
               trans);
          let d = hmm_of k n init trans locs in
          let v = brute k n init trans locs ys in
          equal (float (hmm_bound n k v)) v (item (D.log_density d (vec ys))));
      prop "its membership is each step's state's posterior" gen_hmm
        (fun (k, n, init, trans, locs, ys) ->
          let d = hmm_of k n init trans locs in
          let m = D.membership d (vec ys) in
          equal (array int) [| n; k |] (Nx.shape m);
          let e = Array.map (fun y -> Array.map (normal_lp y) locs) ys in
          let paths = enumerate k n in
          let total =
            logsumexp
              (Array.of_list (List.map (path_density init trans e) paths))
          in
          for t = 0 to n - 1 do
            for j = 0 to k - 1 do
              let p =
                List.fold_left
                  (fun a z ->
                    if List.nth z t = j then
                      a +. Float.exp (path_density init trans e z -. total)
                    else a)
                  0. paths
              in
              equal (float (hmm_bound n k total)) p (Nx.item [ t; j ] m)
            done
          done);
      (* Central differences of the enumeration at [h = 1e-5]: truncation near
         [h^2 / 6] times the third derivative, below [1e-10] for these bounded
         logits, and rounding near [1e-16 / h]. *)
      prop "its gradient in the transition logits is the enumeration's slope"
        Gen.(
          let* n = int_range 2 4 in
          let+ init = array ~size:(constant 2) (float_range (-2.) 2.)
          and+ trans = array ~size:(constant 4) (float_range (-2.) 2.)
          and+ ys = array ~size:(constant n) (float_range (-3.) 3.) in
          (n, init, trans, ys))
        (fun (n, init, trans, ys) ->
          let locs = [| -1.; 1.5 |] in
          let rows t = [| Array.sub t 0 2; Array.sub t 2 2 |] in
          let g =
            Rune.grad'
              (fun tr ->
                D.log_density
                  (D.hmm ~initial:(vec init) ~transition:tr n
                     (D.normal ~loc:(vec locs) ~scale:(f64 0.8)))
                  (vec ys))
              (Nx.create Nx.float64 [| 2; 2 |] trans)
          in
          let h = 1e-5 in
          Array.iteri
            (fun i _ ->
              let at s =
                let t = Array.copy trans in
                t.(i) <- t.(i) +. s;
                brute 2 n init (rows t) locs ys
              in
              let slope = (at h -. at (-.h)) /. (2. *. h) in
              equal (float 1e-8) slope (Nx.to_array g).(i))
            trans);
      test "one step is a mixture" (fun () ->
          let comps = D.normal ~loc:(vec [| -1.; 2. |]) ~scale:(f64 0.8) in
          let initial = vec [| 0.3; -0.2 |] in
          let d =
            D.hmm ~initial ~transition:(Nx.zeros Nx.float64 [| 2; 2 |]) 1 comps
          in
          let v =
            item (D.log_density (D.mixture ~logits:initial comps) (f64 0.7))
          in
          equal
            (float (hmm_bound 1 2 v))
            v
            (item (D.log_density d (vec [| 0.7 |]))));
      test "a chain that alternates is drawn and read as one" (fun () ->
          let ninf = Float.neg_infinity in
          let d =
            D.hmm
              ~initial:(vec [| 0.; ninf |])
              ~transition:
                (Nx.create Nx.float64 [| 2; 2 |] [| ninf; 0.; 0.; ninf |])
              6
              (D.normal ~loc:(vec [| -20.; 20. |]) ~scale:(f64 1.))
          in
          let x = D.sample (Nx.Rng.key 3) d in
          equal (array bool)
            [| false; true; false; true; false; true |]
            (Array.map (fun v -> v > 0.) (Nx.to_array x));
          equal (array float_exact)
            [| 1.; 0.; 0.; 1.; 1.; 0.; 0.; 1.; 1.; 0.; 0.; 1. |]
            (Nx.to_array (D.membership d x)));
      (* [m] sequences of [n] steps, the states read from emissions 40 standard
         deviations apart. The count of first states in state 0, and of the
         stays out of each state given the moves out of it, are binomial; at [z
         = 5.3] each of the three checks fails falsely with probability about
         1.2e-7. *)
      test "its draws follow the chain's probabilities" (fun () ->
          let m = 400 and n = 50 in
          let initial = vec [| 0.4; -0.3 |] in
          let trans = [| 1.2; -0.3; 0.5; 0. |] in
          let d =
            D.iid [| m |]
              (D.hmm ~initial
                 ~transition:(Nx.create Nx.float64 [| 2; 2 |] trans)
                 n
                 (D.normal ~loc:(vec [| -20.; 20. |]) ~scale:(f64 1.)))
          in
          let x = Nx.to_array (D.sample (Nx.Rng.key 21) d) in
          let state i t = if x.((i * n) + t) > 0. then 1 else 0 in
          let firsts = ref 0 in
          let from = Array.make 2 0 and stay = Array.make 2 0 in
          for i = 0 to m - 1 do
            if state i 0 = 0 then incr firsts;
            for t = 1 to n - 1 do
              let a = state i (t - 1) in
              from.(a) <- from.(a) + 1;
              if state i t = a then stay.(a) <- stay.(a) + 1
            done
          done;
          let within count total p =
            let tf = float_of_int total in
            equal
              (float (5.3 *. Float.sqrt (p *. (1. -. p) /. tf)))
              p
              (float_of_int count /. tf)
          in
          let soft a b = 1. /. (1. +. Float.exp (b -. a)) in
          within !firsts m (soft 0.4 (-0.3));
          within stay.(0) from.(0) (soft 1.2 (-0.3));
          within stay.(1) from.(1) (soft 0. 0.5));
      test "a compiled density and membership are the enumeration's" (fun () ->
          Compiled.here ();
          let ys = vec [| 0.3; -1.; 2.; 2.5; -0.7; 0.1; 1.9; -2. |] in
          let d tr =
            D.hmm
              ~initial:(vec [| 0.; 0.5; -1. |])
              ~transition:tr 8
              (D.normal ~loc:(vec [| -1.; 0.5; 2. |]) ~scale:(f64 0.8))
          in
          let tr =
            Nx.create Nx.float64 [| 3; 3 |]
              [| 1.; 0.; -1.; 0.2; 0.8; 0.; -2.; 0.; 1. |]
          in
          (* Against the enumeration of its 3^8 state sequences. *)
          let init = [| 0.; 0.5; -1. |] and locs = [| -1.; 0.5; 2. |] in
          let rows =
            [| [| 1.; 0.; -1. |]; [| 0.2; 0.8; 0. |]; [| -2.; 0.; 1. |] |]
          in
          let ys' = Nx.to_array ys in
          let v = brute 3 8 init rows locs ys' in
          equal
            (float (hmm_bound 8 3 v))
            v
            (item (Rune.jit' (fun tr -> D.log_density (d tr) ys) tr));
          let e = Array.map (fun y -> Array.map (normal_lp y) locs) ys' in
          let paths = enumerate 3 8 in
          let m = Rune.jit' (fun tr -> D.membership (d tr) ys) tr in
          for t = 0 to 7 do
            for j = 0 to 2 do
              let p =
                List.fold_left
                  (fun a z ->
                    if List.nth z t = j then
                      a +. Float.exp (path_density init rows e z -. v)
                    else a)
                  0. paths
              in
              equal (float (hmm_bound 8 3 v)) p (Nx.item [ t; j ] m)
            done
          done);
      test "iid copies are independent sequences" (fun () ->
          let d =
            D.hmm
              ~initial:(vec [| 0.; 0. |])
              ~transition:(Nx.zeros Nx.float64 [| 2; 2 |])
              3
              (D.poisson ~rate:(vec [| 1.; 10. |]))
          in
          let x = Nx.create Nx.int32 [| 2; 3 |] [| 0l; 1l; 12l; 9l; 8l; 1l |] in
          let f = D.factors (D.iid [| 2 |] d) x in
          let rows =
            [|
              item (D.log_density d (Nx.get [ 0 ] x));
              item (D.log_density d (Nx.get [ 1 ] x));
            |]
          in
          equal
            (array
               (float
                  (hmm_bound 3 2
                     (Float.max (Float.abs rows.(0)) (Float.abs rows.(1))))))
            rows (Nx.to_array f);
          equal (array int) [| 2; 3; 2 |]
            (Nx.shape (D.membership (D.iid [| 2 |] d) x)));
      test "a value's steps must be the hmm's" (fun () ->
          let d n =
            D.hmm
              ~initial:(vec [| 0.; 0. |])
              ~transition:(Nx.zeros Nx.float64 [| 2; 2 |])
              n two
          in
          let x = vec [| 0.1; 0.2; 0.3; 0.4; 0.5 |] in
          List.iter
            (fun n ->
              raises_match (Exn.invalid_arg ~sub:"a value of 5 steps")
                (fun () -> D.log_density (d n) x);
              raises_match (Exn.invalid_arg ~sub:"a value of 5 steps")
                (fun () -> D.membership (d n) x))
            [ 1; 3 ]);
      Windtrap.cases "parameters that do not match its states raise naming hmm"
        ~id:(fun (name, _, _, _) -> name)
        [
          ("a transition not square", 2, [| 2; 3 |], 4);
          ("an initial of another length", 3, [| 2; 2 |], 4);
          ("components of another number", 3, [| 3; 3 |], 4);
          ("no step", 2, [| 2; 2 |], 0);
        ]
        (fun (_, k, ts, n) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.hmm:") (fun () ->
              D.hmm
                ~initial:(Nx.zeros Nx.float64 [| k |])
                ~transition:(Nx.zeros Nx.float64 ts) n two));
    ]

(* Eliminators *)

let log_density =
  group "log_density"
    [
      prop "is the sum of the factors"
        Gen.(array ~size:(int_range 0 6) (float_range (-3.) 3.))
        (fun xs ->
          let d =
            D.iid
              [| Array.length xs |]
              (D.normal ~loc:(f64 0.2) ~scale:(f64 1.5))
          in
          let x = vec xs in
          let f = D.factors d x in
          (* the same sum, in another order at most *)
          let t =
            Array.fold_left (fun a v -> a +. Float.abs v) 0. (Nx.to_array f)
          in
          equal
            (float (Float.max (8. *. epsilon_float *. t) Float.min_float))
            (item (Nx.sum f))
            (item (D.log_density d x)));
    ]

let factors =
  group "factors"
    [
      test "iid's axes come in front of the units" (fun () ->
          let d =
            D.iid [| 4; 2 |]
              (D.dirichlet ~concentration:(vec [| 1.5; 2.; 0.7 |]))
          in
          equal (array int) [| 4; 2 |]
            (Nx.shape
               (D.factors d (Nx.full Nx.float64 [| 4; 2; 3 |] (1. /. 3.)))));
      test "the value broadcasts against the parameters" (fun () ->
          let d = D.normal ~loc:(vec [| 0.; 1.; 2. |]) ~scale:(f64 1.) in
          let f = D.factors d (f64 1.) in
          let one loc =
            item (D.factors (D.normal ~loc:(f64 loc) ~scale:(f64 1.)) (f64 1.))
          in
          equal (array float_exact) [| one 0.; one 1.; one 2. |] (Nx.to_array f));
    ]

(* A family of an open support at [dt], at gen/dist.py's parameters. *)
let open_family (type b) (dt : (float, b) Nx.dtype) name :
    ((float, b, Nx.host) Nx.t, b, Nx.host) D.t =
  let s = Nx.scalar dt in
  match name with
  | "uniform" -> D.uniform ~low:(s (-1.)) ~high:(s 3.)
  | "exponential" -> D.exponential ~rate:(s 1.7)
  | "half_normal" -> D.half_normal ~scale:(s 1.5)
  | "half_cauchy" -> D.half_cauchy ~scale:(s 2.)
  | "lognormal" -> D.lognormal ~loc:(s 0.) ~scale:(s 8.)
  | "gamma" -> D.gamma ~concentration:(s 0.0625) ~rate:(s 1.)
  | "beta" -> D.beta ~a:(s 0.0625) ~b:(s 0.0625)
  | "truncated" ->
      D.truncated ~low:(s 60.5) ~high:(s 61.)
        (D.normal ~loc:(s 0.5) ~scale:(s 2.))
  | _ -> failwith name

(* At a narrow dtype one draw in a few hundred rounds onto a bound, so 4000
   draws per row reach it. *)
let draws_inside =
  Windtrap.cases "a draw lies in the open support, at every dtype"
    ~id:(fun (name, Dtype dt) -> name ^ " at " ^ Nx.Dtype.name dt)
    (List.concat_map
       (fun name ->
         List.map
           (fun d -> (name, d))
           [
             Dtype Nx.float64;
             Dtype Nx.float32;
             Dtype Nx.float16;
             Dtype Nx.bfloat16;
           ])
       [
         "uniform";
         "exponential";
         "half_normal";
         "half_cauchy";
         "lognormal";
         "gamma";
         "beta";
         "truncated";
       ])
    (fun (name, Dtype dt) ->
      let d = D.iid [| 4000 |] (open_family dt name) in
      let f = D.factors d (D.sample (Nx.Rng.key 17) d) in
      satisfies ~claim:"every draw's factor finite" (array float_exact)
        (Array.for_all Float.is_finite)
        (Nx.to_array (Nx.cast Nx.float64 f)))

(* Goodness of fit: Pearson's chi-square over the bins of gen/dist.py, whose
   thresholds hold the family-wise false-alarm rate of these tests at 1%. Each
   case draws from its own key, a hash of its name. *)
let fit =
  let key name = Nx.Rng.key (Hashtbl.hash name) in
  let draws_of name =
    let k = key name in
    let sample d = D.sample k (D.iid [| draws |] d) in
    match name with
    | "poisson" | "neg_binomial" | "binomial" | "binomial_rare"
    | "binomial_many" ->
        Array.map Int32.to_float (Nx.to_array (sample (count_case name)))
    | "bernoulli" ->
        Array.map
          (fun b -> if b then 1. else 0.)
          (Nx.to_array (sample (D.bernoulli ~logits:(f64 0.4))))
    | "categorical" ->
        Array.map Int64.to_float
          (Nx.to_array
             (sample (D.categorical ~logits:(vec [| 0.1; -1.; 2. |]))))
    | _ -> Nx.to_array (sample (continuous_case name))
  in
  test "its draws fit the density" (fun () ->
      List.iter
        (fun (name, _) ->
          if not (String.starts_with ~prefix:"lkj" name) then
            subtest name (fun () ->
                at_most (float 1e-9) ~than:(threshold name)
                  (chi_square name (draws_of name))))
        (families "threshold"))

(* A draw [x] of a family of parameter [θ = 2.5], [u] a standard draw: [x = loc
   + θ u] has [∂x/∂θ = (x - loc) / θ], and so on. The rows' other parameters are
   [1]. *)
let reparameterised =
  let theta = 2.5 and one = f64 1. in
  let k = Nx.Rng.key 3 in
  Windtrap.cases "a draw's derivative in its parameter is the closed form's"
    ~id:(fun (name, _, _) -> name)
    [
      ("normal loc", (fun t -> D.normal ~loc:t ~scale:one), fun _ -> 1.);
      ( "normal scale",
        (fun t -> D.normal ~loc:one ~scale:t),
        fun x -> (x -. 1.) /. theta );
      ( "cauchy scale",
        (fun t -> D.cauchy ~loc:one ~scale:t),
        fun x -> (x -. 1.) /. theta );
      ( "laplace scale",
        (fun t -> D.laplace ~loc:one ~scale:t),
        fun x -> (x -. 1.) /. theta );
      ( "logistic scale",
        (fun t -> D.logistic ~loc:one ~scale:t),
        fun x -> (x -. 1.) /. theta );
      ( "student_t scale",
        (fun t -> D.student_t ~df:(f64 3.) ~loc:one ~scale:t),
        fun x -> (x -. 1.) /. theta );
      ( "half_normal scale",
        (fun t -> D.half_normal ~scale:t),
        fun x -> x /. theta );
      ( "half_cauchy scale",
        (fun t -> D.half_cauchy ~scale:t),
        fun x -> x /. theta );
      ( "exponential rate",
        (fun t -> D.exponential ~rate:t),
        fun x -> -.x /. theta );
      ("lognormal loc", (fun t -> D.lognormal ~loc:t ~scale:one), fun x -> x);
      ( "uniform high",
        (fun t -> D.uniform ~low:one ~high:t),
        fun x -> (x -. 1.) /. (theta -. 1.) );
    ]
    (fun (_, family, slope) ->
      (* Each draw is a product or sum of a few roundings: its derivative within
         [16 ε] of the closed form's magnitude. *)
      let draws t = D.sample k (D.iid [| 5 |] (family t)) in
      let x = Nx.to_array (draws (f64 theta)) in
      let g = item (Rune.grad' (fun t -> Nx.sum (draws t)) (f64 theta)) in
      let scale = Array.fold_left (fun a x -> a +. Float.abs (slope x)) 1. x in
      equal
        (float (16. *. epsilon_float *. scale))
        (Array.fold_left (fun a x -> a +. slope x) 0. x)
        g)

let sample = group "sample" [ reparameterised; fit; draws_inside ]

let quantile_golden =
  test "is mpmath's at gen/dist.py's p" (fun () ->
      List.iter
        (fun (name, rows) ->
          subtest name (fun () ->
              check_quantiles ~run:(fun f p -> f p) name (List.map floats rows)))
        (families "ppf"))

type gamma_family = Gamma | Inverse_gamma

(* A gamma of concentration [a] and rate [s], or an inverse gamma of
   concentration [a] and scale [s]. *)
type gamma_case = { family : gamma_family; a : float; s : float }

let gamma_case family a s = { family; a; s }

let gamma_dist c =
  match c.family with
  | Gamma -> D.gamma ~concentration:(f64 c.a) ~rate:(f64 c.s)
  | Inverse_gamma -> D.inverse_gamma ~concentration:(f64 c.a) ~scale:(f64 c.s)

(* [cdf c] and [sf c] are the CDF and its complement, from nx's incomplete gamma
   functions: the gamma's CDF is [P(a, s x)], the inverse gamma's [Q(a, s /
   x)]. *)
let cdf c x =
  let a = Nx.scalar_like x c.a in
  match c.family with
  | Inverse_gamma -> Nx.gammaincc a (Nx.div (Nx.scalar_like x c.s) x)
  | Gamma -> Nx.gammainc a (Nx.mul_s x c.s)

let sf c x =
  let a = Nx.scalar_like x c.a in
  match c.family with
  | Inverse_gamma -> Nx.gammainc a (Nx.div (Nx.scalar_like x c.s) x)
  | Gamma -> Nx.gammaincc a (Nx.mul_s x c.s)

let quantile_at c p = item (D.quantile (gamma_dist c) (f64 p))

(* [x f(x)], for [f] the density. The quantile's condition number at [x] is [m /
   (x f(x))], for [m] the tail it is read from. *)
let x_density c x =
  Float.exp (Float.log x +. item (D.factors (gamma_dist c) (f64 x)))

(* nx's bound on the incomplete gamma functions at a tail [t], in ulps, which
   [κ] scales in their inverses' bound: [32 (1 + |log t|) + 4]. nx states an
   inverse's bound at the smaller tail [m = min (p, 1 - p)], with [κ = m / (x
   f(x))]. *)
let tail_ulps t = (32. *. (1. +. Float.abs (Float.log t))) +. 4.
let ulps n = float_rel ~rel:(n *. epsilon_float) ~abs:0.

(* Concentrations in [2^-lowest, 2^10], rates and scales in [2^-10, 2^10]. *)
let gen_case ?(lowest = 4.) () =
  Gen.(
    let+ family =
      of_list
        ~pp:(fun ppf f ->
          Format.pp_print_string ppf
            (if f = Gamma then "Gamma" else "Inverse_gamma"))
        [ Gamma; Inverse_gamma ]
    and+ la = float_range (-.lowest) 10.
    and+ ls = float_range (-10.) 10. in
    gamma_case family (Float.pow 2. la) (Float.pow 2. ls))

(* A probability [p] with its smaller tail [m], exact: [2^e] in the lower tail
   and [1 - 2^e] in the upper, whose complement is exact by Sterbenz's lemma.
   The lower tail goes down to [2^-deepest], the upper to [2^-52]. In the tail
   where the quantile nears [0] or [inf], the gamma's lower and the inverse
   gamma's upper, the unit gamma variable [y] has mass about [y^a] below it, so
   that tail goes down to [2^(-reach a)] at most, keeping [y] above about
   [2^-reach]. *)
let gen_tail ~deepest ~reach c =
  Gen.(
    let edge = reach *. c.a in
    let lower, upper =
      match c.family with
      | Inverse_gamma -> (deepest, Float.min 52. edge)
      | Gamma -> (Float.min deepest edge, 52.)
    in
    one_of
      [
        map
          (fun e -> (Float.pow 2. e, Float.pow 2. e))
          (float_range (-.lower) (-1.));
        map
          (fun e ->
            let p = 1. -. Float.pow 2. e in
            (p, 1. -. p))
          (float_range (-.upper) (-1.));
      ])

let gen_case_tail ?lowest ~deepest ~reach () =
  Gen.(
    let* c = gen_case ?lowest () in
    let+ p, m = gen_tail ~deepest ~reach c in
    (c, p, m))

let cover_case c =
  cover "inverse gamma" (c.family = Inverse_gamma);
  cover "gamma" (c.family = Gamma);
  cover "concentration below 1" (c.a < 1.);
  cover "concentration above 100" (c.a > 100.)

let cover_tails p =
  cover "p below 2^-100" (p < 0x1p-100);
  cover "p above 1 - 2^-26" (p > 1. -. 0x1p-26)

(* The quantile [x] is within [4 + κ B(m)] ulps of the exact one, for [B] the
   bound of [tail_ulps]; the division by the rate or into the scale adds half an
   ulp, as does the product or quotient [cdf] and [sf] take. The tail at [x] is
   then within [5 / κ + 2 B(m)] ulps of [m]. *)
let cdf_of_quantile =
  prop "the CDF of the quantile at p is p"
    (gen_case_tail ~deepest:990. ~reach:800. ()) (fun (c, p, m) ->
      cover_case c;
      cover_tails p;
      let x = quantile_at c p in
      let tail = if p <= 0.5 then cdf c (f64 x) else sf c (f64 x) in
      let tol = (5. *. x_density c x /. m) +. (2. *. tail_ulps m) in
      equal (ulps tol) m (item tail))

(* An [x] whose CDF lies in either tail, from [y = s x] for the gamma and [y = s
   / x] for the inverse gamma: [log2 y] in [[log2 a - min (800, 200 / a), log2
   a]] for the lower tail of [P(a, y)] and in [[log2 a, log2 (2 a + 40)]] for
   its upper. Half the gamma's cases whose [a] allows it stay below [log2 a -
   100 / a - 3/2], where the Chernoff bound [(y / a)^a e^(a - y)] puts [P(a, y)]
   below [2^-100], so that every run reaches that tail. *)
let gen_point c =
  Gen.(
    let la = Float.log2 c.a in
    let deepest = la -. Float.min 800. (200. /. c.a) in
    let deep = la -. (100. /. c.a) -. 1.5 in
    let tail =
      if c.family = Inverse_gamma || deep <= deepest then []
      else [ (2, float_range deepest deep) ]
    in
    let+ ly =
      frequency
        (tail
        @ [
            (1, float_range deepest la);
            (1, float_range la (Float.log2 ((2. *. c.a) +. 40.)));
          ])
    in
    let y = Float.pow 2. ly in
    match c.family with Inverse_gamma -> c.s /. y | Gamma -> y /. c.s)

(* [p] is within [B(p)] ulps of the CDF at [x]. Read from its smaller tail [m],
   [p] is then within [p B(p) / m] ulps of [m]. The quantile scales that and its
   own [B(m)] by [κ = m / (x f(x))] and adds [5] ulps, as in [cdf_of_quantile]:
   it is within [5 + (m B(m) + p B(p)) / (x f(x))] ulps of [x]. *)
let quantile_of_cdf =
  prop "the quantile at the CDF of x is x"
    ~examples:[ (gamma_case Gamma 2. 1., 0x1p-60) ]
    Gen.(
      let* c = gen_case () in
      let+ x = gen_point c in
      (c, x))
    (fun (c, x) ->
      let p = item (cdf c (f64 x)) and q = item (sf c (f64 x)) in
      assume (Float.is_finite x && x > 0. && p > 0. && p < 1.);
      cover_case c;
      cover "p below 2^-100" (p < 0x1p-100);
      cover "p above 1/2" (p > 0.5);
      let m = Float.min p q in
      let tol =
        5. +. (((m *. tail_ulps m) +. (p *. tail_ulps p)) /. x_density c x)
      in
      equal (ulps tol) x (quantile_at c p))

let gamma_variable c x =
  match c.family with Inverse_gamma -> c.s /. x | Gamma -> c.s *. x

(* [scaled c d x] is [d f(x)], taken in logs: [d] may be beyond the floats'
   range where [d f(x)] is not. *)
let scaled c d x =
  Float.copy_sign
    (Float.exp
       (Float.log (Float.abs d) +. item (D.factors (gamma_dist c) (f64 x))))
    d

(* An upper bound on the magnitudes the log density and the derivative rule
   round, in units of their result: [16] times it bounds their error in ulps. *)
let magnitudes c x d =
  let y = gamma_variable c x in
  Float.abs (c.a *. Float.log c.s)
  +. Float.abs (Float.log c.s)
  +. Float.abs (item (Nx.lgamma (f64 c.a)))
  +. ((c.a +. 1.) *. (Float.abs (Float.log x) +. Float.abs (Float.log y)))
  +. y
  +. Float.abs (Float.log (Float.abs d))

(* The derivatives of the quantile [x] at [(a, s, p)]. The quantile is off by
   [δ = 5 + κ B(m)] ulps, as in [cdf_of_quantile], and a reference taken at
   it rather than at the exact one moves by its derivative in [log x] times
   that; [log f] moves by at most [(a + 1 + y) δ], for [y = s x] or [s / x].

   - In [p], [∂x/∂p f(x) = 1]. rune holds the inverse's derivative within
     [64 + 16 κ B(m)] ulps.
   - In [s], [∂x/∂s = ±x / s], within a few roundings.
   - In [a], [∂x/∂a f(x) = ∓∂ₐP(a, y)] by the implicit function theorem on
     [F(x) = p]. rune holds the inverse's derivative within [2^-40] and
     [∂ₐP] within [2^-40 (1 + |log m|)]. [∂ₐP] moves by
     [y |∂_y log |∂ₐP||] times [δ], which rune computes too.

   Draws reach concentrations of [2^-7], and quantiles up to [2^910] and
   down to [2^-910], whose derivatives are still finite; the examples are
   inverse gammas whose quantile is past [2^512], where [s / y²]
   overflows. *)
(* [quantile_of c] is the quantile as a function of [(a, (s, p))]. *)
let quantile_of c (a, (s, p)) =
  let dist =
    match c.family with
    | Inverse_gamma -> D.inverse_gamma ~concentration:a ~scale:s
    | Gamma -> D.gamma ~concentration:a ~rate:s
  in
  D.quantile dist p

let args = Nx.Ptree.(pair tensor (pair tensor tensor))

let derivatives =
  prop "the quantile's derivatives are the implicit function's"
    ~examples:
      (List.map
         (fun (c, p) -> (c, p, 1. -. p))
         [
           (gamma_case Inverse_gamma 0.01 1., 0.99);
           (gamma_case Inverse_gamma 0.1 1., 1. -. 0x1p-52);
         ])
    (gen_case_tail ~lowest:7. ~deepest:990. ~reach:900. ())
    (fun (c, p, m) ->
      cover_case c;
      cover_tails p;
      cover "concentration below 2^-4" (c.a < 0x1p-4);
      let a = f64 c.a in
      let x, (da, (ds, dp)) =
        Rune.value_and_grad args (quantile_of c) (a, (f64 c.s, f64 p))
      in
      let x = item x and da = item da and ds = item ds and dp = item dp in
      let y = gamma_variable c x in
      let kappa = m /. x_density c x in
      let delta = 5. +. (kappa *. tail_ulps m) in
      let shift = (c.a +. 1. +. y) *. delta in
      equal
        (ulps
           (64.
           +. (16. *. kappa *. tail_ulps m)
           +. shift
           +. (16. *. magnitudes c x dp)))
        1. (scaled c dp x);
      let sign = match c.family with Inverse_gamma -> 1. | Gamma -> -1. in
      equal (ulps 4.) (sign *. x /. c.s) ds;
      let dP_da a y = Rune.grad' (fun a -> Nx.gammainc a y) a in
      let reference = item (dP_da a (f64 y)) in
      let moves =
        y
        *. Float.abs
             (item (Rune.grad' (fun y -> Nx.log (Nx.abs (dP_da a y))) (f64 y)))
      in
      let tol =
        (0x1p-40 *. (2. +. Float.abs (Float.log m)))
        +. epsilon_float
           *. (((moves +. 2.) *. (delta +. 1.))
              +. shift
              +. (16. *. magnitudes c x da))
      in
      equal (float_rel ~rel:tol ~abs:0.) reference (sign *. scaled c da x))

(* Forward mode applies the derivatives reverse mode transposes: each is within
   twice its budget of [derivatives] of the other. The points include inverse
   gammas whose quantile is past [2^512], where [s / y²] overflows, and two
   whose derivative in [p] overflows, which must leave the derivatives in [a]
   and [s] finite. *)
let forward_mode =
  Windtrap.cases "forward mode agrees with reverse mode"
    ~id:(fun (n, _, _) -> n)
    [
      ("gamma", gamma_case Gamma 2.5 1.5, 0.3);
      ("gamma near 1", gamma_case Gamma 0.01 1., 1. -. 0x1p-40);
      ("inverse_gamma", gamma_case Inverse_gamma 3. 2., 0.3);
      ("inverse_gamma at 0.99", gamma_case Inverse_gamma 0.01 1., 0.99);
      ("inverse_gamma near 1", gamma_case Inverse_gamma 0.1 1., 1. -. 0x1p-52);
      ( "inverse_gamma at a subnormal p",
        gamma_case Inverse_gamma 2.5 1.,
        0x1p-1050 );
      ( "inverse_gamma of a large scale",
        gamma_case Inverse_gamma 2.5 0x1p100,
        1e-300 );
    ]
    (fun (_, c, p) ->
      let at = (f64 c.a, (f64 c.s, f64 p)) in
      let da, (ds, dp) = Rune.grad args (quantile_of c) at in
      let along t =
        item (snd (Rune.jvp args Nx.Ptree.tensor (quantile_of c) at t))
      in
      let zero = f64 0. and one = f64 1. in
      let x = quantile_at c p in
      let m = Float.min p (1. -. p) in
      let kappa = m /. x_density c x in
      equal
        (float_rel ~rel:0x1p-39 ~abs:0.)
        (item da)
        (along (one, (zero, zero)));
      equal (ulps 8.) (item ds) (along (zero, (one, zero)));
      equal
        (ulps (2. *. (64. +. (16. *. kappa *. tail_ulps m))))
        (item dp)
        (along (zero, (zero, one))))

(* Compiled, the quantile keeps its eager bound, [5 + κ B(m)] ulps of the exact
   one, so the two are within twice it. Each family is a test of its own: its
   quantile's program, nx's inverse incomplete gamma, is one kernel of 250 KB of
   C whatever the input, and macOS runners took 34 to 52 s to compile the two
   from a cold cache. *)
let compiled_quantile =
  let ps = [| 1e-200; 1e-10; 0.3; 0.5; 0.9; 1. -. 1e-10 |] in
  let compiled name c =
    Windtrap.test name (fun () ->
        Compiled.here ();
        let compiled =
          Nx.to_array (Rune.jit' (D.quantile (gamma_dist c)) (vec ps))
        in
        Array.iteri
          (fun i p ->
            let x = quantile_at c p in
            let m = Float.min p (1. -. p) in
            let tol = 2. *. (5. +. (m /. x_density c x *. tail_ulps m)) in
            equal (ulps tol) x compiled.(i))
          ps)
  in
  Windtrap.group "a compiled quantile is the eager one"
    [
      compiled "gamma" (gamma_case Gamma 2.5 1.5);
      compiled "inverse_gamma" (gamma_case Inverse_gamma 3. 2.);
    ]

let gamma_quantiles =
  group "quantile of the gamma and the inverse gamma"
    [
      cdf_of_quantile;
      quantile_of_cdf;
      derivatives;
      forward_mode;
      compiled_quantile;
      Windtrap.cases "the quantile is 0 at p = 0 and inf at p = 1"
        ~id:(fun (n, _) -> n)
        [
          ("gamma", gamma_case Gamma 2.5 1.5);
          ("gamma of small concentration", gamma_case Gamma 0.1 3.);
          ("inverse_gamma", gamma_case Inverse_gamma 3. 2.);
          ( "inverse_gamma of small concentration",
            gamma_case Inverse_gamma 0.1 0.5 );
        ]
        (fun (_, c) ->
          equal float_exact 0. (quantile_at c 0.);
          equal float_exact Float.infinity (quantile_at c 1.));
    ]

(* Quantiles of the beta *)

(* [beta_quantile_of (a, (b, p))] is the beta's quantile as a function of its
   shapes and [p]. *)
let beta_quantile_of (a, (b, p)) = D.quantile (D.beta ~a ~b) p

(* The exact derivatives are finite for shapes in [(0, inf)] and [p] in [(0,
   1)]: in [p] [1 / f(x)], in a shape [-∂I/∂a / f(x)], [f] the density. Draws
   reach shapes of [2^-60], where [log (a B(a, b))] is O(a), and quantiles that
   underflow, whose derivatives are finite too. *)
let beta_derivatives =
  (* Eight points per case through one elementwise call: 16 cases draw 128
     points, more than the 100 of one point each, and each cover is met by some
     point of a case with probability 0.77 or more. *)
  prop "the beta's quantile's first derivatives are finite" ~count:16
    ~examples:[ [| (2., 3., 1. -. 0x1p-40) |] ]
    Gen.(
      let shape = map (fun e -> Float.pow 2. e) (float_range (-60.) 20.) in
      let point =
        let+ a = shape
        and+ b = shape
        and+ p =
          one_of
            [
              float_range 0x1p-20 (1. -. 0x1p-20);
              map (fun e -> Float.pow 2. e) (float_range (-1000.) (-1.));
              map (fun e -> 1. -. Float.pow 2. e) (float_range (-52.) (-1.));
            ]
        in
        (a, b, p)
      in
      array ~size:(constant 8) point)
    (fun points ->
      let some f = Array.exists f points in
      cover "a shape below 2^-30"
        (some (fun (a, b, _) -> Float.min a b < 0x1p-30));
      cover "a shape above 2^10"
        (some (fun (a, b, _) -> Float.max a b > 0x1p10));
      cover "p below 2^-100" (some (fun (_, _, p) -> p < 0x1p-100));
      cover "p above 1 - 2^-26" (some (fun (_, _, p) -> p > 1. -. 0x1p-26));
      let col f = vec (Array.map f points) in
      let at =
        ( col (fun (a, _, _) -> a),
          (col (fun (_, b, _) -> b), col (fun (_, _, p) -> p)) )
      in
      let x = beta_quantile_of at in
      let da, (db, dp) =
        Rune.grad args (fun t -> Nx.sum (beta_quantile_of t)) at
      in
      satisfies ~claim:"the quantiles and their derivatives finite"
        (array float_exact)
        (Array.for_all Float.is_finite)
        (Array.concat (List.map Nx.to_array [ x; da; db; dp ])))

(* Forward mode applies the derivatives reverse mode transposes: both are one
   program's, so they agree to its roundings. *)
let beta_forward_mode =
  Windtrap.cases "the beta's forward mode agrees with reverse mode"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("central", 2.5, 4., 0.3);
      ("a of 1e-12", 1e-12, 1., 0.5);
      ("b of 1e-20 against 2^20", 0x1p20, 1e-20, 1e-300);
      ("a small beside b large", 0.5, 1e5, 1e-3);
      ("near 1", 1e5, 0.5, 0.999);
    ]
    (fun (_, a, b, p) ->
      let at = (f64 a, (f64 b, f64 p)) in
      let da, (db, dp) = Rune.grad args beta_quantile_of at in
      let along t =
        item (snd (Rune.jvp args Nx.Ptree.tensor beta_quantile_of at t))
      in
      let zero = f64 0. and one = f64 1. in
      let close = float_rel ~rel:0x1p-40 ~abs:0. in
      equal close (item da) (along (one, (zero, zero)));
      equal close (item db) (along (zero, (one, zero)));
      equal close (item dp) (along (zero, (zero, one))))

let beta_quantiles =
  group "quantile of the beta"
    [
      beta_derivatives;
      beta_forward_mode;
      Windtrap.cases "the quantile is 0 at p = 0 and 1 at p = 1"
        ~id:(fun (a, b) -> Printf.sprintf "a = %g, b = %g" a b)
        [ (2., 5.); (0.5, 0.5); (1e-3, 1e3) ]
        (fun (a, b) ->
          let q p = item (D.quantile (D.beta ~a:(f64 a) ~b:(f64 b)) (f64 p)) in
          equal (pair float_exact float_exact) (0., 1.) (q 0., q 1.));
    ]

let laplace_quantile =
  prop "the laplace quantile's derivative is 2 scale / (1 - |2p - 1|)"
    ~examples:[ 0.5 ]
    Gen.(float_range 0.01 0.99)
    (fun p ->
      cover "at the median" (p = 0.5);
      let d = D.laplace ~loc:(f64 0.3) ~scale:(f64 1.3) in
      (* the derivative of [loc + scale log (2p)] or its mirror: a quotient and
         a product, a few roundings *)
      equal
        (float_rel ~rel:(8. *. epsilon_float) ~abs:0.)
        (* [1 - |2p - 1|] is [2 min (p, 1 - p)], read without cancelling *)
        (1.3 /. Float.min p (1. -. p))
        (item (Rune.grad' (D.quantile d) (f64 p))))

(* The quantile's derivative in [p] is [1 / f(x)] at [x = quantile p]: their
   product is [1] within the inverse's conditioning, [κ = m / (x f(x))] for [m]
   the smaller tail, times a few hundred ulps, and within the change in [log f]
   that rounding [x] makes, [|x ∂ log f / ∂x| ε]: near a bound of the support,
   as [x] near [1] for a beta, that change is large. *)
let quantile_slope =
  (* [p] from [1e-6] to [1 - 1e-6], evenly in the logit. *)
  let ps =
    Array.init 41 (fun i ->
        let t = -13.8 +. (27.6 *. float_of_int i /. 40.) in
        1. /. (1. +. Float.exp (-.t)))
  in
  Windtrap.cases "its derivative in p is one over the density" ~id:Fun.id
    [
      "normal";
      "half_normal";
      "lognormal";
      "cauchy";
      "half_cauchy";
      "laplace";
      "logistic";
      "exponential";
      "gamma";
      "gamma_small";
      "inverse_gamma";
      "beta";
      "beta_u";
      "uniform";
      "truncated_normal";
      "truncated_gamma_both";
      "truncated_beta";
    ] (fun name ->
      let d = continuous_case name in
      let sum f v = Nx.sum (f v) in
      let dps = Rune.grad' (sum (D.quantile d)) (vec ps) in
      let xs = D.quantile d (vec ps) in
      let slopes = Rune.grad' (sum (D.factors d)) xs in
      let log_f = Nx.to_array (D.factors d xs) and xs = Nx.to_array xs in
      let dps = Nx.to_array dps and slopes = Nx.to_array slopes in
      Array.iteri
        (fun i p ->
          let f = Float.exp log_f.(i) in
          let m = Float.min p (1. -. p) in
          let kappa = Float.abs (m /. (xs.(i) *. f)) in
          let moved = Float.abs (xs.(i) *. slopes.(i)) in
          equal
            ~msg:(Printf.sprintf "p = %h" p)
            (float
               (epsilon_float *. ((512. *. (1. +. kappa)) +. (4. *. moved))))
            1.
            (dps.(i) *. f))
        ps)

let quantile =
  group "quantile"
    [
      quantile_golden;
      gamma_quantiles;
      beta_quantiles;
      laplace_quantile;
      quantile_slope;
      test "an iid's quantile is its distribution's, elementwise" (fun () ->
          let d = D.normal ~loc:(f64 0.5) ~scale:(f64 2.) in
          let ps = vec [| 0.1; 0.5; 0.9 |] in
          equal (array float_exact)
            (Nx.to_array (D.quantile d ps))
            (Nx.to_array (D.quantile (D.iid [| 3 |] d) ps)));
      Windtrap.cases "a family without a quantile raises naming quantile"
        ~id:(fun (name, _) -> name)
        [
          ("student_t", D.student_t ~df:(f64 1.) ~loc:(f64 0.) ~scale:(f64 1.));
          ("von_mises", D.von_mises ~loc:(f64 0.) ~concentration:(f64 1.));
          ("dirichlet", D.dirichlet ~concentration:(vec [| 1.; 1. |]));
          ( "mvn",
            D.mvn ~loc:(vec [| 0.; 0. |]) ~scale_tril:(Nx.eye Nx.float64 2) );
          ("lkj_cholesky", D.lkj_cholesky 2 ~concentration:(f64 1.));
          ("sorted", D.sorted 2 std);
          ("transform", D.transform Norn.Bij.exp std);
          ("mixture", D.mixture ~logits:(vec [| 0.; 0. |]) two);
          ("hmm", hmm2 2);
        ]
        (fun (_, d) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.quantile:") (fun () ->
              D.quantile d (f64 0.5)));
    ]

let bij_name b = Format.asprintf "%a" Norn.Bij.pp b

let coords =
  group "coords"
    [
      Windtrap.cases "is the bijector onto each family's support" ~id:fst
        [
          ("normal", ("identity", continuous_case "normal"));
          ("gamma", ("exp", continuous_case "gamma"));
          ("half_cauchy", ("exp", continuous_case "half_cauchy"));
          ("beta", ("interval", continuous_case "beta"));
          ("uniform", ("interval", continuous_case "uniform"));
          ("von_mises", ("interval", continuous_case "von_mises"));
          ( "dirichlet",
            ("simplex", D.dirichlet ~concentration:(vec [| 1.; 2. |])) );
          ( "lkj_cholesky",
            ("cholesky_corr", D.lkj_cholesky 3 ~concentration:(f64 1.)) );
          ( "truncated on both sides",
            ("interval", continuous_case "truncated_normal") );
          ( "truncated below",
            ("greater", continuous_case "truncated_normal_far") );
          ("truncated above", ("less", continuous_case "truncated_normal_below"));
          ( "sorted half_normal",
            ("compose(exp, ordered)", D.sorted 3 (D.half_normal ~scale:(f64 1.)))
          );
          ( "hmm, as a mixture",
            ( "interval",
              D.hmm
                ~initial:(vec [| 0.; 0. |])
                ~transition:(Nx.zeros Nx.float64 [| 2; 2 |])
                3
                (D.uniform ~low:(vec [| 0.; 2. |]) ~high:(vec [| 1.; 5. |])) )
          );
        ]
        (fun (_, (name, d)) -> equal string name (bij_name (D.coords d)));
      (* Every case gen/dist.py takes a quantile of, the Student t and the von
         Mises. *)
      test "the pullback through coords integrates to one" (fun () ->
          List.iter
            (fun name ->
              subtest name (fun () ->
                  equal (float saturated_mass) 1.
                    (integrate (pullback (continuous_case name)))))
            (List.map fst (families "ppf")
            @ [ "student_t"; "von_mises"; "von_mises_flat" ]));
      test "a sorted vector's coordinates keep its order and support" (fun () ->
          let d = D.sorted 3 (D.half_normal ~scale:(f64 1.)) in
          let x, _ = Norn.Bij.forward (D.coords d) (vec [| -2.; 0.; -1. |]) in
          satisfies ~claim:"positive and increasing" (array float_exact)
            (fun x -> x.(0) > 0. && x.(0) < x.(1) && x.(1) < x.(2))
            (Nx.to_array x));
      test "a mixture's coordinates cover its components' supports" (fun () ->
          let d =
            D.mixture
              ~logits:(vec [| 0.; 0. |])
              (D.uniform ~low:(vec [| 0.; 2. |]) ~high:(vec [| 1.; 5. |]))
          in
          let x, _ = Norn.Bij.forward (D.coords d) (f64 0.) in
          (* [0 + 5 sigmoid 0], exact *)
          equal float_exact 2.5 (item x));
      test "checks the parameters" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Norn.Dist.coords: uniform: high - low")
            (fun () -> D.coords (D.uniform ~low:(f64 1.) ~high:(f64 0.))));
    ]

let standardize =
  let s d = Norn.Bij.forward (D.standardize d) in
  let z = 0.75 in
  group "standardize"
    [
      Windtrap.cases "maps a standard draw to the family's"
        ~id:(fun (name, _, _) -> name)
        [
          ("normal", D.normal ~loc:(f64 2.) ~scale:(f64 3.), 2. +. (3. *. z));
          ( "student_t",
            D.student_t ~df:(f64 3.) ~loc:(f64 2.) ~scale:(f64 3.),
            2. +. (3. *. z) );
          ("cauchy", D.cauchy ~loc:(f64 2.) ~scale:(f64 3.), 2. +. (3. *. z));
          ("laplace", D.laplace ~loc:(f64 2.) ~scale:(f64 3.), 2. +. (3. *. z));
          ("logistic", D.logistic ~loc:(f64 2.) ~scale:(f64 3.), 2. +. (3. *. z));
          ( "lognormal",
            D.lognormal ~loc:(f64 0.5) ~scale:(f64 0.25),
            Float.exp (0.5 +. (0.25 *. z)) );
          ( "iid of normal",
            D.iid [| 1 |] (D.normal ~loc:(f64 2.) ~scale:(f64 3.)),
            2. +. (3. *. z) );
        ]
        (fun (_, d, x) ->
          let shape = D.shape d in
          let u = Nx.full Nx.float64 shape z in
          equal
            (float (4. *. epsilon_float *. Float.abs x))
            x
            (Nx.item
               (List.map (fun _ -> 0) (Array.to_list shape))
               (fst (s d u))));
      test "an mvn is standardised by its Cholesky factor" (fun () ->
          let d =
            D.mvn
              ~loc:(vec [| 1.; -1. |])
              ~scale_tril:
                (Nx.create Nx.float64 [| 2; 2 |] [| 2.; 0.; 0.5; 1. |])
          in
          equal (array float_exact) [| 3.; 0.5 |]
            (Nx.to_array (fst (s d (vec [| 1.; 1. |])))));
      test "a gamma has no standard form" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.standardize:")
            (fun () ->
              D.standardize (D.gamma ~concentration:(f64 1.) ~rate:(f64 1.))));
      test "checks the parameters" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Norn.Dist.standardize: normal: scale")
            (fun () ->
              D.standardize (D.normal ~loc:(f64 0.) ~scale:(f64 (-1.)))));
    ]

let membership =
  group "membership"
    [
      test "a normal has no latent variable" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Dist.membership:") (fun () ->
              D.membership std (f64 0.)));
    ]

(* Observers *)

let family =
  group "family"
    [
      Windtrap.cases "names the family or combinator" ~id:fst
        [
          ( "iid keeps its argument's",
            ( "dirichlet",
              D.iid [| 2 |] (D.dirichlet ~concentration:(vec [| 1.; 1. |])) ) );
          ("sorted", ("sorted", D.sorted 2 std));
          ("transform", ("transform", D.transform Norn.Bij.exp std));
          ("truncated", ("truncated", continuous_case "truncated_normal"));
          ("mixture", ("mixture", D.mixture ~logits:(vec [| 0.; 0. |]) two));
          ("hmm", ("hmm", hmm2 2));
        ]
        (fun (_, (name, d)) -> equal string name (D.family d));
    ]

let shape =
  group "shape"
    [
      Windtrap.cases "is the parameters' broadcast and the family's event axes"
        ~id:fst
        [
          ( "normal",
            ([| 3 |], D.normal ~loc:(vec [| 0.; 1.; 2. |]) ~scale:(f64 1.)) );
          ( "dirichlet",
            ([| 3 |], D.dirichlet ~concentration:(vec [| 1.; 1.; 1. |])) );
          ( "mvn",
            ( [| 2 |],
              D.mvn ~loc:(vec [| 0.; 0. |]) ~scale_tril:(Nx.eye Nx.float64 2) )
          );
          ( "lkj_cholesky",
            ([| 4; 4 |], D.lkj_cholesky 4 ~concentration:(f64 1.)) );
          ("sorted", ([| 5 |], D.sorted 5 std));
          ("hmm", ([| 6 |], hmm2 6));
          ( "transform simplex",
            ([| 3 |], D.transform Norn.Bij.simplex (D.iid [| 2 |] std)) );
        ]
        (fun (_, (s, d)) -> equal (array int) s (D.shape d));
    ]

(* Two software devices on the host: the parameters on the second, a key on the
   first. *)
let placement =
  group "placement"
    [
      test "a draw lies where the parameters do, whatever the key's placement"
        (fun () ->
          let module P =
            (val Nx.devices
                   (List.init 2 (fun k ->
                        Result.get_ok
                          (Rig.software (Printf.sprintf "norn-dist-%d" k)))))
          in
          let devices = Nx.Devices.to_list P.v in
          let on k = Nx.Placement.device P.v (List.nth devices k) in
          let pp p = Format.asprintf "%a" Nx.Placement.pp p in
          let at x = Nx.place (on 1) (f64 x) in
          let d = D.normal ~loc:(at 0.) ~scale:(at 1.) in
          let key = Nx.Rng.place (on 0) (Nx.Rng.key 2) in
          equal string (pp (on 1)) (pp (D.placement d));
          equal string (pp (on 1)) (pp (Nx.placement (D.sample key d)));
          equal string (pp (on 1)) (pp (D.placement (D.iid [| 3 |] d)));
          equal string
            (pp (on 1))
            (pp (D.placement (D.transform Norn.Bij.exp d))));
    ]

let transform_support =
  test "a transform's support is its bijector's image" (fun () ->
      (* The bound is the C library's exp at 1, which FreeBSD's rounds one ulp
         from glibc's. *)
      equal support_w
        (Norn.Support.Interval (1., Float.exp 1.))
        (D.support
           (D.transform Norn.Bij.exp (D.uniform ~low:(f64 0.) ~high:(f64 1.))));
      equal support_w (Norn.Support.Greater 0.)
        (D.support (D.transform Norn.Bij.exp std));
      equal support_w (Norn.Support.Simplex 3)
        (D.support (D.transform Norn.Bij.simplex (D.iid [| 2 |] std))))

let per_element_interval =
  test "a per-element interval's image holds every element's" (fun () ->
      (* Element i maps (0, 1) onto (low_i + w_i sigmoid 0, low_i + w_i sigmoid
         1), w_i = high_i - low_i; the support is their hull. *)
      let low = [| 0.; 0. |] and high = [| 1.; 10. |] in
      let at x i = low.(i) +. ((high.(i) -. low.(i)) *. sigmoid x) in
      let b = Norn.Bij.interval ~low:(vec low) ~high:(vec high) in
      let base = D.uniform ~low:(vec [| 0.; 0. |]) ~high:(vec [| 1.; 1. |]) in
      let interval = function
        | Norn.Support.Interval (a, b) -> Some (a, b)
        | _ -> None
      in
      (* each bound a few roundings of values below 10 *)
      let near = float (64. *. epsilon_float) in
      equal (pair near near)
        (Float.min (at 0. 0) (at 0. 1), Float.max (at 1. 0) (at 1. 1))
        (require_match interval (D.support (D.transform b base))))

let support =
  group "support"
    [
      Windtrap.cases "is each family's set" ~id:fst
        [
          ( "normal",
            (Norn.Support.Real, fun () -> D.support (continuous_case "normal"))
          );
          ( "half_normal",
            ( Norn.Support.Greater 0.,
              fun () -> D.support (continuous_case "half_normal") ) );
          ( "lognormal",
            ( Norn.Support.Greater 0.,
              fun () -> D.support (continuous_case "lognormal") ) );
          ( "exponential",
            ( Norn.Support.Greater 0.,
              fun () -> D.support (continuous_case "exponential") ) );
          ( "gamma",
            ( Norn.Support.Greater 0.,
              fun () -> D.support (continuous_case "gamma") ) );
          ( "beta",
            ( Norn.Support.Interval (0., 1.),
              fun () -> D.support (continuous_case "beta") ) );
          ( "uniform",
            ( Norn.Support.Interval (-1., 3.),
              fun () -> D.support (continuous_case "uniform") ) );
          ( "von_mises",
            ( Norn.Support.Interval (-.Float.pi, Float.pi),
              fun () -> D.support (continuous_case "von_mises") ) );
          ( "dirichlet",
            ( Norn.Support.Simplex 3,
              fun () ->
                D.support (D.dirichlet ~concentration:(vec [| 1.; 1.; 1. |])) )
          );
          ( "lkj_cholesky",
            ( Norn.Support.Correlation_cholesky 3,
              fun () -> D.support (D.lkj_cholesky 3 ~concentration:(f64 1.)) )
          );
          ( "sorted",
            (Norn.Support.Ordered, fun () -> D.support (D.sorted 2 std)) );
          ( "bernoulli",
            ( Norn.Support.Boolean,
              fun () -> D.support (D.bernoulli ~logits:(f64 0.)) ) );
          ( "poisson",
            ( Norn.Support.Integers_from 0,
              fun () -> D.support (D.poisson ~rate:(f64 1.)) ) );
          ( "categorical",
            ( Norn.Support.Integer_interval (0, 2),
              fun () -> D.support (D.categorical ~logits:(vec [| 0.; 0.; 0. |]))
            ) );
          ( "binomial of trials 3 and 7",
            ( Norn.Support.Integer_interval (0, 7),
              fun () ->
                D.support (D.binomial ~trials:trials_3_7 ~logits:(f64 0.)) ) );
          ( "truncated on both sides",
            ( Norn.Support.Interval (-1., 3.),
              fun () -> D.support (continuous_case "truncated_normal") ) );
          ( "truncated above",
            ( Norn.Support.Less 2.,
              fun () ->
                D.support
                  (D.truncated ~high:(f64 2.) (continuous_case "normal")) ) );
          ( "truncated gamma clips to its support",
            ( Norn.Support.Interval (0., 2.),
              fun () ->
                D.support (D.truncated ~high:(f64 2.) (continuous_case "gamma"))
            ) );
          ( "truncated exponential clips its low bound",
            ( Norn.Support.Interval (0., 1.),
              fun () ->
                D.support
                  (D.truncated ~low:(f64 (-5.)) ~high:(f64 1.)
                     (continuous_case "exponential")) ) );
          ( "mixture over trials",
            ( Norn.Support.Integer_interval (0, 7),
              fun () ->
                D.support
                  (D.mixture
                     ~logits:(vec [| 0.; 0. |])
                     (D.binomial ~trials:trials_3_7 ~logits:(f64 0.))) ) );
          ( "mixture of uniforms",
            ( Norn.Support.Interval (0., 5.),
              fun () ->
                D.support
                  (D.mixture
                     ~logits:(vec [| 0.; 0. |])
                     (D.uniform
                        ~low:(vec [| 0.; 2. |])
                        ~high:(vec [| 1.; 5. |]))) ) );
        ]
        (fun (_, (s, f)) -> equal support_w s (f ()));
      transform_support;
      per_element_interval;
      test "reading a parameter bound inside jit raises" (fun () ->
          Compiled.here ();
          raises_match (Exn.invalid_arg ~sub:"") (fun () ->
              Rune.jit'
                (fun low ->
                  ignore (D.support (D.uniform ~low ~high:(f64 3.)));
                  low)
                (f64 0.)));
      test "reading a bound that depends on vmap's argument raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"") (fun () ->
              Rune.vmap
                Nx.Ptree.(tensor @-> returns tensor)
                (fun low ->
                  ignore (D.support (D.uniform ~low ~high:(f64 3.)));
                  low)
                (vec [| 0.; 1. |])));
    ]

let bounds =
  let lo_hi d =
    let lo, hi = D.bounds d in
    (Nx.to_array lo, Nx.to_array hi)
  in
  let inf = Float.infinity in
  group "bounds"
    [
      Windtrap.cases "are each element's least and greatest value" ~id:fst
        [
          ( "normal",
            ( ([| -.inf |], [| inf |]),
              fun () -> lo_hi (D.iid [| 1 |] (continuous_case "normal")) ) );
          ( "gamma",
            ( ([| 0. |], [| inf |]),
              fun () -> lo_hi (D.iid [| 1 |] (continuous_case "gamma")) ) );
          ( "uniform per element",
            ( ([| 0.; 2. |], [| 1.; 5. |]),
              fun () ->
                lo_hi
                  (D.uniform ~low:(vec [| 0.; 2. |]) ~high:(vec [| 1.; 5. |]))
            ) );
          ( "dirichlet",
            ( ([| 0.; 0. |], [| 1.; 1. |]),
              fun () -> lo_hi (D.dirichlet ~concentration:(vec [| 1.; 1. |])) )
          );
          ( "truncated normal",
            ( ([| -1. |], [| 3. |]),
              fun () ->
                lo_hi (D.iid [| 1 |] (continuous_case "truncated_normal")) ) );
          ( "binomial per element",
            ( ([| 0.; 0. |], [| 3.; 7. |]),
              fun () -> lo_hi (D.binomial ~trials:trials_3_7 ~logits:(f64 0.))
            ) );
          ( "a mixture over trials",
            ( ([| 0. |], [| 7. |]),
              fun () ->
                lo_hi
                  (D.iid [| 1 |]
                     (D.mixture
                        ~logits:(vec [| 0.; 0. |])
                        (D.binomial ~trials:trials_3_7 ~logits:(f64 0.)))) ) );
          ( "a mixture of no component",
            ( ([| inf |], [| -.inf |]),
              fun () ->
                lo_hi
                  (D.iid [| 1 |]
                     (D.mixture
                        ~logits:(Nx.zeros Nx.float64 [| 0 |])
                        (D.binomial
                           ~trials:(Nx.zeros Nx.int32 [| 0 |])
                           ~logits:(f64 0.)))) ) );
        ]
        (fun (_, (expected, f)) ->
          equal (pair (array float_exact) (array float_exact)) expected (f ()));
      test "read inside jit, they are the eager ones" (fun () ->
          Compiled.here ();
          let hi low =
            snd (D.bounds (D.uniform ~low ~high:(vec [| 1.; 5. |])))
          in
          equal (array float_exact)
            (Nx.to_array (hi (vec [| 0.; 2. |])))
            (Nx.to_array (Rune.jit' hi (vec [| 0.; 2. |]))));
    ]

let pp =
  group "pp"
    [
      test "formats the family and its parameters' dtypes and shapes" (fun () ->
          equal string "normal(loc: float64 [], scale: float64 [8])"
            (Format.asprintf "%a" D.pp
               (D.normal ~loc:(f64 0.) ~scale:(Nx.ones Nx.float64 [| 8 |]))));
    ]

(* Checks *)

let check =
  group "check"
    [
      test "names its context" (fun () ->
          raises
            (Invalid_argument
               "site tau: half_normal: scale is 0, not in (0, inf)") (fun () ->
              D.check "site tau" (D.half_normal ~scale:(f64 0.))));
      test "leaves the eliminators no check to run" (fun () ->
          let d =
            D.check ~unless:(Nx.scalar Nx.bool true) "interpreter"
              (D.normal ~loc:(f64 0.) ~scale:(f64 (-1.)))
          in
          ignore (D.log_density d (f64 0.)));
      Windtrap.cases "unless broadcasts against the parameters"
        ~id:(fun (u, _) -> Format.asprintf "%a" (Testable.pp (array bool)) u)
        [
          ([| true; false |], Some "[1]");
          ([| false; true |], Some "[0]");
          ([| true; true |], None);
        ]
        (fun (u, at) ->
          let d = D.normal ~loc:(f64 0.) ~scale:(vec [| -1.; -1. |]) in
          let unless = Nx.create Nx.bool [| 2 |] u in
          match at with
          | Some i ->
              raises_match
                (Exn.invalid_arg ~sub:("scale at " ^ i))
                (fun () -> D.check ~unless "here" d)
          | None -> ignore (D.check ~unless "here" d));
    ]

(* Every positive parameter at [+inf], the others at a value inside. *)
let infinite_positive =
  let inf = f64 Float.infinity and one = f64 1. and zero = f64 0. in
  [
    ("normal scale", D.valid (D.normal ~loc:zero ~scale:inf));
    ("half_normal scale", D.valid (D.half_normal ~scale:inf));
    ("lognormal scale", D.valid (D.lognormal ~loc:zero ~scale:inf));
    ("student_t df", D.valid (D.student_t ~df:inf ~loc:zero ~scale:one));
    ("student_t scale", D.valid (D.student_t ~df:one ~loc:zero ~scale:inf));
    ("cauchy scale", D.valid (D.cauchy ~loc:zero ~scale:inf));
    ("half_cauchy scale", D.valid (D.half_cauchy ~scale:inf));
    ("laplace scale", D.valid (D.laplace ~loc:zero ~scale:inf));
    ("logistic scale", D.valid (D.logistic ~loc:zero ~scale:inf));
    ("exponential rate", D.valid (D.exponential ~rate:inf));
    ("gamma concentration", D.valid (D.gamma ~concentration:inf ~rate:one));
    ("gamma rate", D.valid (D.gamma ~concentration:one ~rate:inf));
    ( "inverse_gamma concentration",
      D.valid (D.inverse_gamma ~concentration:inf ~scale:one) );
    ( "inverse_gamma scale",
      D.valid (D.inverse_gamma ~concentration:one ~scale:inf) );
    ("beta a", D.valid (D.beta ~a:inf ~b:one));
    ("beta b", D.valid (D.beta ~a:one ~b:inf));
    ( "dirichlet concentration",
      D.valid (D.dirichlet ~concentration:(vec [| 1.; Float.infinity |])) );
    ( "mvn scale_tril's diagonal",
      D.valid
        (D.mvn
           ~loc:(vec [| 0.; 0. |])
           ~scale_tril:
             (Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.; 0.; Float.infinity |]))
    );
    ( "neg_binomial dispersion",
      D.valid (D.neg_binomial ~mean:one ~dispersion:inf) );
  ]

let valid =
  let reads v = [| Nx.item [] v |] in
  group "valid"
    [
      Windtrap.cases "a parameter of +inf is outside (0, inf)" ~id:fst
        infinite_positive (fun (_, v) ->
          equal (array bool) [| false |] (reads v));
      test "parameters inside their domains are valid" (fun () ->
          equal (array bool) [| true |] (reads (D.valid std)));
      Windtrap.cases "is false where a parameter is outside its domain" ~id:fst
        [
          ("NaN logits", D.valid (D.bernoulli ~logits:(f64 Float.nan)));
          ( "one bad element of a vector",
            D.valid (D.normal ~loc:(f64 0.) ~scale:(vec [| 1.; -1.; 2. |])) );
          ( "a mixture's bad component",
            D.valid
              (D.mixture
                 ~logits:(vec [| 0.; 0. |])
                 (D.normal ~loc:(vec [| 0.; 1. |]) ~scale:(vec [| 1.; -1. |])))
          );
          ( "a mixture's NaN logit",
            D.valid (D.mixture ~logits:(vec [| 0.; Float.nan |]) two) );
          ( "iid copies of a bad distribution",
            D.valid (D.iid [| 3 |] (D.half_normal ~scale:(f64 (-1.)))) );
          ( "a checked distribution",
            D.valid
              (D.check ~unless:(Nx.scalar Nx.bool true) "interpreter"
                 (D.normal ~loc:(f64 0.) ~scale:(f64 (-1.)))) );
          ( "a NaN below scale_tril's diagonal",
            D.valid
              (D.mvn
                 ~loc:(vec [| 0.; 0. |])
                 ~scale_tril:
                   (Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.; Float.nan; 1. |]))
          );
        ]
        (fun (_, v) -> equal (array bool) [| false |] (reads v));
      test "is a scalar over every element" (fun () ->
          equal (array int) [||]
            (Nx.shape
               (D.valid (D.normal ~loc:(f64 0.) ~scale:(vec [| 1.; -1.; 2. |])))));
      test "under a map it is one value per lane" (fun () ->
          let v =
            Rune.vmap
              Nx.Ptree.(tensor @-> returns tensor)
              (fun scale -> D.valid (D.normal ~loc:(f64 0.) ~scale))
              (vec [| 1.; -1.; 2. |])
          in
          equal (array bool) [| true; false; true |] (Nx.to_array v));
    ]

let () =
  exit
    (run "Norn.Dist"
       [
         validation;
         group "continuous families"
           [
             golden_densities;
             golden_far_densities;
             continuous_scores;
             compiled_densities;
             normal;
             cauchy;
             group "student_t" [ student_t_tails ];
             dirichlet;
             mvn;
             von_mises;
             lkj;
           ];
         group "discrete families" [ masses; binomial; categorical ];
         group "combinators"
           [ iid; sorted; transform; truncation; mixture; hmm_tests ];
         group "eliminators"
           [
             log_density;
             factors;
             sample;
             quantile;
             coords;
             standardize;
             membership;
           ];
         group "observers" [ family; shape; placement; support; bounds; pp ];
         group "checks" [ check; valid ];
       ])
