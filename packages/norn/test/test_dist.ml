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

(* The cases of gen/dist.py, at its parameters. *)
type case =
  | Real of (Nx.host Nx.float64_t, Nx.float64_elt, Nx.host) D.t
  | Count of (Nx.host Nx.int32_t, Nx.float64_elt, Nx.host) D.t
  | Flag of (Nx.host Nx.bool_t, Nx.float64_elt, Nx.host) D.t
  | Index of (Nx.host Nx.int64_t, Nx.float64_elt, Nx.host) D.t

let reals =
  [
    ("normal", D.normal ~loc:(f64 0.5) ~scale:(f64 2.));
    ("half_normal", D.half_normal ~scale:(f64 1.5));
    ("lognormal", D.lognormal ~loc:(f64 0.3) ~scale:(f64 0.8));
    ("student_t", D.student_t ~df:(f64 3.5) ~loc:(f64 (-1.)) ~scale:(f64 0.7));
    ("cauchy", D.cauchy ~loc:(f64 1.) ~scale:(f64 0.5));
    ("half_cauchy", D.half_cauchy ~scale:(f64 2.));
    ("laplace", D.laplace ~loc:(f64 0.) ~scale:(f64 1.3));
    ("logistic", D.logistic ~loc:(f64 2.) ~scale:(f64 0.4));
    ("exponential", D.exponential ~rate:(f64 1.7));
    ("gamma", D.gamma ~concentration:(f64 2.5) ~rate:(f64 1.5));
    ("gamma_small", D.gamma ~concentration:(f64 0.4) ~rate:(f64 0.8));
    ("inverse_gamma", D.inverse_gamma ~concentration:(f64 3.) ~scale:(f64 2.));
    ("beta", D.beta ~a:(f64 2.) ~b:(f64 5.));
    ("beta_u", D.beta ~a:(f64 0.5) ~b:(f64 0.5));
    ("uniform", D.uniform ~low:(f64 (-1.)) ~high:(f64 3.));
  ]

let cases =
  List.map (fun (n, d) -> (n, Real d)) reals
  @ [
      ("poisson", Count (D.poisson ~rate:(f64 3.2)));
      ( "neg_binomial",
        Count (D.neg_binomial ~mean:(f64 4.) ~dispersion:(f64 2.5)) );
      ("bernoulli", Flag (D.bernoulli ~logits:(f64 0.4)));
      ("binomial", Count (D.binomial ~trials:(i32 12) ~logits:(f64 0.4)));
      ( "binomial_rare",
        Count (D.binomial ~trials:(i32 40) ~logits:(f64 (-2.5))) );
      ( "binomial_many",
        Count (D.binomial ~trials:(i32 200) ~logits:(f64 (-1.))) );
      ("categorical", Index (D.categorical ~logits:(vec [| 0.1; -1.; 2. |])));
    ]

let dirichlet = D.dirichlet ~concentration:(vec [| 1.5; 2.; 0.7 |])

let mvn =
  D.mvn
    ~loc:(vec [| 1.; -1. |])
    ~scale_tril:(Nx.create Nx.float64 [| 2; 2 |] [| 2.; 0.; 0.5; 1. |])

(* Goldens *)

let golden =
  lazy
    (let ic = open_in "golden/dist.golden" in
     let rec go acc =
       match input_line ic with
       | l when String.length l > 0 && l.[0] = '#' -> go acc
       | l -> go (String.split_on_char ' ' l :: acc)
       | exception End_of_file ->
           close_in ic;
           List.rev acc
     in
     go [])

let lines kind = List.filter (fun l -> List.hd l = kind) (Lazy.force golden)
let floats ws = List.map float_of_string ws

(* Within the rounding of nx's special functions: a few hundred ulps of the
   terms, absolutely near zero. *)
let close = float_rel ~rel:1e-12 ~abs:1e-12

let exact_inf =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun a b ->
      if Float.is_finite a then Testable.equal close a b else Float.equal a b)

let factor name xs =
  match List.assoc name cases with
  | Real d -> item (D.factors d (f64 (List.hd xs)))
  | Count d ->
      item (D.factors d (Nx.scalar Nx.int32 (Int32.of_float (List.hd xs))))
  | Flag d -> item (D.factors d (Nx.scalar Nx.bool (List.hd xs = 1.)))
  | Index d ->
      item (D.factors d (Nx.scalar Nx.int64 (Int64.of_float (List.hd xs))))

let densities =
  group "log density against SciPy"
    (List.map
       (fun l ->
         match l with
         | [ _; name; x; v ] when name <> "dirichlet" && name <> "mvn" ->
             test
               (Printf.sprintf "%s at %h" name (float_of_string x))
               (fun () ->
                 equal exact_inf (float_of_string v)
                   (factor name [ float_of_string x ]))
         | _ :: name :: rest ->
             let rest = floats rest in
             let n = List.length rest - 1 in
             let x =
               vec (Array.of_list (List.filteri (fun i _ -> i < n) rest))
             in
             let v = List.nth rest n in
             test
               (Printf.sprintf "%s at [%s]" name
                  (String.concat "; "
                     (List.map (Printf.sprintf "%g")
                        (Array.to_list (Nx.to_array x)))))
               (fun () ->
                 let d = if name = "dirichlet" then dirichlet else mvn in
                 equal exact_inf v (item (D.factors d x)))
         | _ -> failwith "dist.golden: a malformed logpdf line")
       (lines "logpdf"))

let quantiles =
  group "quantile against mpmath"
    (List.map
       (fun l ->
         match l with
         | [ _; name; p; v ] ->
             test (Printf.sprintf "%s at %s" name p) (fun () ->
                 let d =
                   match List.assoc name cases with
                   | Real d -> d
                   | _ -> assert false
                 in
                 equal close (float_of_string v)
                   (item (D.quantile d (f64 (float_of_string p)))))
         | _ -> assert false)
       (lines "ppf"))

(* Quantiles of the gamma and the inverse gamma *)

(* A gamma of concentration [a] and rate [s], or an inverse gamma of
   concentration [a] and scale [s]. *)
type gamma_case = {
  inverse : bool;
  a : float;
  s : float;
  dist : (Nx.host Nx.float64_t, Nx.float64_elt, Nx.host) D.t;
}

let gamma_case inverse a s =
  let dist =
    if inverse then D.inverse_gamma ~concentration:(f64 a) ~scale:(f64 s)
    else D.gamma ~concentration:(f64 a) ~rate:(f64 s)
  in
  { inverse; a; s; dist }

(* [cdf c] and [sf c] are the CDF and its complement, from nx's incomplete
   gamma functions: the gamma's CDF is [P(a, s x)], the inverse gamma's
   [Q(a, s / x)]. *)
let cdf c x =
  let a = Nx.scalar_like x c.a in
  if c.inverse then Nx.gammaincc a (Nx.div (Nx.scalar_like x c.s) x)
  else Nx.gammainc a (Nx.mul_s x c.s)

let sf c x =
  let a = Nx.scalar_like x c.a in
  if c.inverse then Nx.gammainc a (Nx.div (Nx.scalar_like x c.s) x)
  else Nx.gammaincc a (Nx.mul_s x c.s)

let quantile_at c p = item (D.quantile c.dist (f64 p))

(* [x f(x)], for [f] the density. The quantile's condition number at [x] is
   [m / (x f(x))], for [m] the tail it is read from. *)
let x_density c x = Float.exp (Float.log x +. item (D.factors c.dist (f64 x)))

(* nx's bound on the incomplete gamma functions at a tail [t], in ulps, which
   [κ] scales in their inverses' bound: [32 (1 + |log t|) + 4]. nx states an
   inverse's bound at the smaller tail [m = min (p, 1 - p)], with
   [κ = m / (x f(x))]. *)
let tail_ulps t = (32. *. (1. +. Float.abs (Float.log t))) +. 4.
let ulps n = float_rel ~rel:(n *. epsilon_float) ~abs:0.

(* Concentrations in [2^-lowest, 2^10], rates and scales in [2^-10, 2^10]. *)
let gen_case ?(lowest = 4.) () =
  Gen.(
    let+ inverse = bool
    and+ la = float_range (-.lowest) 10.
    and+ ls = float_range (-10.) 10. in
    gamma_case inverse (Float.pow 2. la) (Float.pow 2. ls))

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
    let lower = if c.inverse then deepest else Float.min deepest edge in
    let upper = if c.inverse then Float.min 52. edge else 52. in
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
  cover "inverse gamma" c.inverse;
  cover "gamma" (not c.inverse);
  cover "concentration below 1" (c.a < 1.);
  cover "concentration above 100" (c.a > 100.)

let cover_tails p =
  cover "p below 2^-100" (p < 0x1p-100);
  cover "p above 1 - 2^-26" (p > 1. -. 0x1p-26)

(* The quantile [x] is within [4 + κ B(m)] ulps of the exact one, for [B] the
   bound of [tail_ulps]; the division by the rate or into the scale adds half
   an ulp, as does the product or quotient [cdf] and [sf] take. The tail at
   [x] is then within [5 / κ + 2 B(m)] ulps of [m]. *)
let cdf_of_quantile =
  prop "the CDF of the quantile at p is p"
    (gen_case_tail ~deepest:990. ~reach:800. ())
    (fun (c, p, m) ->
      cover_case c;
      cover_tails p;
      let x = quantile_at c p in
      let tail = if p <= 0.5 then cdf c (f64 x) else sf c (f64 x) in
      let tol = (5. *. x_density c x /. m) +. (2. *. tail_ulps m) in
      equal (ulps tol) m (item tail))

(* An [x] whose CDF lies in either tail, from [y = s x] for the gamma and
   [y = s / x] for the inverse gamma: [log2 y] in
   [[log2 a - min (800, 200 / a), log2 a]] for the lower tail of [P(a, y)]
   and in [[log2 a, log2 (2 a + 40)]] for its upper. *)
let gen_point c =
  Gen.(
    let la = Float.log2 c.a in
    let+ ly =
      one_of
        [
          float_range (la -. Float.min 800. (200. /. c.a)) la;
          float_range la (Float.log2 ((2. *. c.a) +. 40.));
        ]
    in
    let y = Float.pow 2. ly in
    if c.inverse then c.s /. y else y /. c.s)

(* [p] is within [B(p)] ulps of the CDF at [x]. Read from its smaller tail
   [m], [p] is then within [p B(p) / m] ulps of [m]. The quantile scales that
   and its own [B(m)] by [κ = m / (x f(x))] and adds [5] ulps, as in
   [cdf_of_quantile]: it is within [5 + (m B(m) + p B(p)) / (x f(x))] ulps of
   [x]. *)
let quantile_of_cdf =
  prop "the quantile at the CDF of x is x"
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

let gamma_variable c x = if c.inverse then c.s /. x else c.s *. x

(* [scaled c d x] is [d f(x)], taken in logs: [d] may be beyond the floats'
   range where [d f(x)] is not. *)
let scaled c d x =
  Float.copy_sign
    (Float.exp (Float.log (Float.abs d) +. item (D.factors c.dist (f64 x))))
    d

(* An upper bound on the magnitudes the log density and the derivative rule
   round, in units of their result: [16] times it bounds their error in
   ulps. *)
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
    if c.inverse then D.inverse_gamma ~concentration:a ~scale:s
    else D.gamma ~concentration:a ~rate:s
  in
  D.quantile dist p

let args = Nx.Ptree.(pair tensor (pair tensor tensor))

let derivatives =
  prop "the quantile's derivatives are the implicit function's"
    ~examples:
      (List.map
         (fun (c, p) -> (c, p, 1. -. p))
         [
           (gamma_case true 0.01 1., 0.99);
           (gamma_case true 0.1 1., 1. -. 0x1p-52);
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
           (64. +. (16. *. kappa *. tail_ulps m) +. shift
           +. (16. *. magnitudes c x dp)))
        1. (scaled c dp x);
      let sign = if c.inverse then 1. else -1. in
      equal (ulps 4.) (sign *. x /. c.s) ds;
      let dP_da a y = Rune.grad' (fun a -> Nx.gammainc a y) a in
      let reference = item (dP_da a (f64 y)) in
      let moves =
        y
        *. Float.abs
             (item
                (Rune.grad' (fun y -> Nx.log (Nx.abs (dP_da a y))) (f64 y)))
      in
      let tol =
        (0x1p-40 *. (2. +. Float.abs (Float.log m)))
        +. epsilon_float
           *. (((moves +. 2.) *. (delta +. 1.))
              +. shift
              +. (16. *. magnitudes c x da))
      in
      equal (float_rel ~rel:tol ~abs:0.) reference (sign *. scaled c da x))

(* Forward mode applies the derivatives reverse mode transposes: each is
   within twice its budget of [derivatives] of the other. The points include
   inverse gammas whose quantile is past [2^512], where [s / y²] overflows,
   and two whose derivative in [p] overflows, which must leave the
   derivatives in [a] and [s] finite. *)
let forward_mode =
  Windtrap.cases "forward mode agrees with reverse mode"
    ~id:(fun (n, _, _) -> n)
    [
      ("gamma", gamma_case false 2.5 1.5, 0.3);
      ("gamma near 1", gamma_case false 0.01 1., 1. -. 0x1p-40);
      ("inverse_gamma", gamma_case true 3. 2., 0.3);
      ("inverse_gamma at 0.99", gamma_case true 0.01 1., 0.99);
      ("inverse_gamma near 1", gamma_case true 0.1 1., 1. -. 0x1p-52);
      ("inverse_gamma at a subnormal p", gamma_case true 2.5 1., 0x1p-1050);
      ("inverse_gamma of a large scale", gamma_case true 2.5 0x1p100, 1e-300);
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
        (item dp) (along (zero, (zero, one))))

(* Compiled, the quantile keeps its eager bound, [5 + κ B(m)] ulps of the
   exact one, so the two are within twice it. *)
let compiled_quantile =
  let ps = [| 1e-200; 1e-10; 0.3; 0.5; 0.9; 1. -. 1e-10 |] in
  Windtrap.cases "a compiled quantile is the eager one"
    ~id:(fun (n, _) -> n)
    [
      ("gamma", gamma_case false 2.5 1.5);
      ("inverse_gamma", gamma_case true 3. 2.);
    ]
    (fun (_, c) ->
      let compiled = Nx.to_array (Rune.jit' (D.quantile c.dist) (vec ps)) in
      Array.iteri
        (fun i p ->
          let x = quantile_at c p in
          let m = Float.min p (1. -. p) in
          let tol = 2. *. (5. +. (m /. x_density c x *. tail_ulps m)) in
          equal (ulps tol) x compiled.(i))
        ps)

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
          ("gamma", gamma_case false 2.5 1.5);
          ("gamma of small concentration", gamma_case false 0.1 3.);
          ("inverse_gamma", gamma_case true 3. 2.);
          ("inverse_gamma of small concentration", gamma_case true 0.1 0.5);
        ]
        (fun (_, c) ->
          equal float_exact 0. (quantile_at c 0.);
          equal float_exact Float.infinity (quantile_at c 1.));
    ]

(* Goodness of fit: Pearson's chi-square over the bins of gen/dist.py, whose
   thresholds hold the family-wise false-alarm rate of these tests at 1%. *)

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

let fit =
  group "sample fits the density"
    (List.mapi
       (fun i (name, c) ->
         slow
         @@ test name (fun () ->
             let k = Nx.Rng.key (100 + i) in
             let values =
               match c with
               | Real d -> Nx.to_array (D.sample k (D.iid [| draws |] d))
               | Count d ->
                   Array.map Int32.to_float
                     (Nx.to_array (D.sample k (D.iid [| draws |] d)))
               | Flag d ->
                   Array.map
                     (fun b -> if b then 1. else 0.)
                     (Nx.to_array (D.sample k (D.iid [| draws |] d)))
               | Index d ->
                   Array.map Int64.to_float
                     (Nx.to_array (D.sample k (D.iid [| draws |] d)))
             in
             at_most (float 1e-9) ~than:(threshold name)
               (chi_square name values)))
       (List.filter (fun (n, _) -> n <> "gamma_small" && n <> "beta_u") cases))

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

let normalisation =
  group "the pullback through coords integrates to one"
    (List.map
       (fun (name, d) ->
         test name (fun () -> equal (float 1e-8) 1. (integrate (pullback d))))
       reals)

(* Laws and combinators *)

let sums =
  prop "log_density is the sum of the factors"
    Gen.(array ~size:(int_range 0 6) (float_range (-3.) 3.))
    (fun xs ->
      let d =
        D.iid [| Array.length xs |] (D.normal ~loc:(f64 0.2) ~scale:(f64 1.5))
      in
      let x = vec xs in
      equal close (item (Nx.sum (D.factors d x))) (item (D.log_density d x)))

let combinators =
  group "combinators"
    [
      test "iid prepends its shape" (fun () ->
          let d = D.iid [| 4; 2 |] dirichlet in
          equal (array int) [| 4; 2; 3 |] (D.shape d);
          equal (array int) [| 4; 2 |]
            (Nx.shape
               (D.factors d (Nx.full Nx.float64 [| 4; 2; 3 |] (1. /. 3.))));
          equal string "dirichlet" (D.family d));
      test "a sorted vector's density is n! times the product" (fun () ->
          let d = D.normal ~loc:(f64 0.) ~scale:(f64 1.) in
          let x = vec [| -1.; 0.25; 2. |] in
          let expected =
            Float.log 6. +. item (D.log_density (D.iid [| 3 |] d) x)
          in
          equal close expected (item (D.log_density (D.sorted 3 d) x)));
      test "an unsorted vector has density -inf" (fun () ->
          let d = D.sorted 3 (D.normal ~loc:(f64 0.) ~scale:(f64 1.)) in
          equal float_exact Float.neg_infinity
            (item (D.log_density d (vec [| 1.; 0.; 2. |]))));
      test "sorted refuses a vector distribution" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"not over scalars") (fun () ->
              D.sorted 2
                (D.iid [| 2 |] (D.normal ~loc:(f64 0.) ~scale:(f64 1.)))));
      test "the exp of a normal is a lognormal" (fun () ->
          let t =
            D.transform Norn.Bij.exp (D.normal ~loc:(f64 0.3) ~scale:(f64 0.8))
          in
          let l = D.lognormal ~loc:(f64 0.3) ~scale:(f64 0.8) in
          List.iter
            (fun x ->
              equal close
                (item (D.factors l (f64 x)))
                (item (D.factors t (f64 x))))
            [ 0.01; 0.5; 1.; 10. ];
          equal float_exact Float.neg_infinity (item (D.factors t (f64 (-1.)))));
      test "a mixture's density sums its weighted components" (fun () ->
          let logits = vec [| 0.3; -0.2 |] in
          let d =
            D.mixture ~logits
              (D.normal ~loc:(vec [| -1.; 2. |]) ~scale:(vec [| 0.5; 1. |]))
          in
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
          equal close expected (item (D.log_density d (f64 x))));
      test "a component per point is iid of a mixture" (fun () ->
          let d =
            D.iid [| 3 |]
              (D.mixture
                 ~logits:(vec [| 0.; 0. |])
                 (D.normal ~loc:(vec [| -1.; 1. |]) ~scale:(f64 1.)))
          in
          let x = vec [| -1.; 0.; 1. |] in
          equal (array int) [| 3 |] (Nx.shape (D.factors d x));
          let m = D.mixture_membership d x in
          equal (array int) [| 3; 2 |] (Nx.shape m);
          equal (array close) [| 1.; 1.; 1. |]
            (Nx.to_array (Nx.sum ~axes:[ 1 ] m));
          equal close 0.5 (Nx.item [ 1; 0 ] m));
      test "a mixture's coordinates cover its components' supports" (fun () ->
          let d =
            D.mixture
              ~logits:(vec [| 0.; 0. |])
              (D.uniform ~low:(vec [| 0.; 2. |]) ~high:(vec [| 1.; 5. |]))
          in
          equal string "(0, 5)"
            (Format.asprintf "%a" Norn.Support.pp (D.support d));
          let x, _ = Norn.Bij.forward (D.coords d) (f64 0.) in
          equal close 2.5 (item x));
      test "a mixture's draws have its shape" (fun () ->
          let d =
            D.mixture
              ~logits:(vec [| 0.; 0. |])
              (D.normal ~loc:(vec [| -1.; 1. |]) ~scale:(f64 1.))
          in
          equal (array int) [||] (Nx.shape (D.sample (Nx.Rng.key 0) d));
          equal (array int) [| 5 |]
            (Nx.shape (D.sample (Nx.Rng.key 0) (D.iid [| 5 |] d))));
    ]

let bijectors =
  group "coordinates"
    [
      test "a normal is standardised by its location and scale" (fun () ->
          let b = D.standardize (D.normal ~loc:(f64 2.) ~scale:(f64 3.)) in
          equal close 5. (item (fst (Norn.Bij.forward b (f64 1.)))));
      test "an mvn is standardised by its Cholesky factor" (fun () ->
          let x, _ = Norn.Bij.forward (D.standardize mvn) (vec [| 1.; 1. |]) in
          equal (array close) [| 3.; 0.5 |] (Nx.to_array x));
      test "a gamma has no standard form" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"gamma has no standard form")
            (fun () ->
              D.standardize (D.gamma ~concentration:(f64 1.) ~rate:(f64 1.))));
      test "a sorted vector's coordinates keep its order and support" (fun () ->
          let d = D.sorted 3 (D.half_normal ~scale:(f64 1.)) in
          let x, _ = Norn.Bij.forward (D.coords d) (vec [| -2.; 0.; -1. |]) in
          let x = Nx.to_array x in
          satisfies ~claim:"positive and increasing" (array float_exact)
            (fun x -> x.(0) > 0. && x.(0) < x.(1) && x.(1) < x.(2))
            x);
      test "a transform's support is its bijector's image" (fun () ->
          let pp d = Format.asprintf "%a" Norn.Support.pp (D.support d) in
          let t =
            D.transform Norn.Bij.exp (D.uniform ~low:(f64 0.) ~high:(f64 1.))
          in
          equal string (Printf.sprintf "(1, %g)" (Float.exp 1.)) (pp t);
          equal string "(0, inf)"
            (pp
               (D.transform Norn.Bij.exp
                  (D.normal ~loc:(f64 0.) ~scale:(f64 1.))));
          equal string "simplex of 3"
            (pp
               (D.transform Norn.Bij.simplex
                  (D.iid [| 2 |] (D.normal ~loc:(f64 0.) ~scale:(f64 1.))))));
      test "a per-element interval's image holds every element's" (fun () ->
          (* Element i maps (0, 1) onto (low_i + w_i sigmoid 0, low_i + w_i
             sigmoid 1), w_i = high_i - low_i; the support is their hull. *)
          let low = [| 0.; 0. |] and high = [| 1.; 10. |] in
          let sigmoid x = 1. /. (1. +. Float.exp (-.x)) in
          let at x i = low.(i) +. ((high.(i) -. low.(i)) *. sigmoid x) in
          let b = Norn.Bij.interval ~low:(vec low) ~high:(vec high) in
          let base =
            D.uniform ~low:(vec [| 0.; 0. |]) ~high:(vec [| 1.; 1. |])
          in
          let interval = function
            | Norn.Support.Interval (a, b) -> Some (a, b)
            | _ -> None
          in
          equal (pair close close)
            (Float.min (at 0. 0) (at 0. 1), Float.max (at 1. 0) (at 1. 1))
            (require_match interval (D.support (D.transform b base))));
      test "supports print as sets" (fun () ->
          equal string "(0, inf)"
            (Format.asprintf "%a" Norn.Support.pp
               (D.support (D.gamma ~concentration:(f64 1.) ~rate:(f64 1.))));
          equal string "(-1, 3)"
            (Format.asprintf "%a" Norn.Support.pp
               (D.support (D.uniform ~low:(f64 (-1.)) ~high:(f64 3.))));
          equal string "{0, 1, 2, ...}"
            (Format.asprintf "%a" Norn.Support.pp
               (D.support (D.poisson ~rate:(f64 1.)))));
    ]

(* Binomial *)

(* [sigmoid l], to its precision where [exp (-l)] overflows. *)
let sigmoid l =
  if l < 0. then
    let e = Float.exp l in
    e /. (1. +. e)
  else 1. /. (1. +. Float.exp (-.l))

(* Mostly finite logits, often infinite ones. *)
let logit =
  Gen.frequency
    [
      (6, Gen.float_range (-30.) 30.);
      (1, Gen.constant Float.infinity);
      (1, Gen.constant Float.neg_infinity);
    ]

let binomial_mass =
  prop "a binomial's mass sums to one over {0, ..., trials}"
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
      let mass = Nx.sum (Nx.exp (D.factors d xs)) in
      (* Each of at most 301 terms within a few hundred ulps. *)
      equal (float 1e-11) 1. (item mass))

let binomial_bernoulli =
  prop "a binomial of one trial is a bernoulli" logit (fun l ->
      let b = D.bernoulli ~logits:(f64 l) in
      let d = D.binomial ~trials:(i32 1) ~logits:(f64 l) in
      let flags = Nx.create Nx.bool [| 2 |] [| false; true |] in
      let counts = Nx.create Nx.int32 [| 2 |] [| 0l; 1l |] in
      equal (array exact_inf)
        (Nx.to_array (D.factors b flags))
        (Nx.to_array (D.factors d counts)))

(* The mean [n p] and variance [n p q] of [m] draws, each within [z] of its
   standard error: [sqrt (n p q / m)] for the mean and, for the variance,
   [sqrt ((mu4 - (n p q)^2) / m)] with [mu4 = n p q (1 + 3 p q (n - 2))]. At
   [z = 5.3] each check fails falsely with probability about 1.2e-7, the eight
   together below 1e-6. *)
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
        Array.fold_left (fun a x -> a +. ((x -. mean) ** 2.)) 0. xs
        /. (fm -. 1.)
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
      equal (array exact_inf)
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
      test "its support and bounds reach the most trials" (fun () ->
          let d =
            D.binomial
              ~trials:(Nx.create Nx.int32 [| 2 |] [| 3l; 7l |])
              ~logits:(f64 0.)
          in
          equal string "{0, 1, ..., 7}"
            (Format.asprintf "%a" Norn.Support.pp (D.support d));
          let lo, hi = D.bounds d in
          equal (array float_exact) [| 0.; 0. |] (Nx.to_array lo);
          equal (array float_exact) [| 3.; 7. |] (Nx.to_array hi);
          let f = Nx.to_array (D.factors d (i32 5)) in
          equal float_exact Float.neg_infinity f.(0);
          let seven = D.binomial ~trials:(i32 7) ~logits:(f64 0.) in
          equal close (item (D.factors seven (i32 5))) f.(1));
      test "a mixture over trials has the hull of their supports" (fun () ->
          let trials = Nx.create Nx.int32 [| 2 |] [| 3l; 7l |] in
          let d =
            D.mixture
              ~logits:(vec [| 0.; 0. |])
              (D.binomial ~trials ~logits:(f64 0.))
          in
          equal string "{0, 1, ..., 7}"
            (Format.asprintf "%a" Norn.Support.pp (D.support d));
          let lo, hi = D.bounds d in
          equal (pair float_exact float_exact) (0., 7.) (item lo, item hi);
          let none =
            D.mixture
              ~logits:(Nx.zeros Nx.float64 [| 0 |])
              (D.binomial ~trials:(Nx.zeros Nx.int32 [| 0 |]) ~logits:(f64 0.))
          in
          let lo, hi = D.bounds none in
          equal (pair float_exact float_exact)
            (Float.infinity, Float.neg_infinity)
            (item lo, item hi);
          (* 5 is outside the first component's support. *)
          let seven = D.binomial ~trials:(i32 7) ~logits:(f64 0.) in
          equal close
            (Float.log 0.5 +. item (D.log_density seven (i32 5)))
            (item (D.log_density d (i32 5))));
      test "the score of the logits is x - trials sigmoid logits" (fun () ->
          let x = Nx.create Nx.int32 [| 3 |] [| 0l; 3l; 12l |] in
          let g =
            Rune.grad'
              (fun logits ->
                D.log_density
                  (D.iid [| 3 |] (D.binomial ~trials:(i32 12) ~logits))
                  x)
              (f64 0.4)
          in
          equal close (15. -. (36. *. sigmoid 0.4)) (item g));
      test "a compiled density is the eager one" (fun () ->
          let d logits = D.iid [| 50 |] (D.binomial ~trials:(i32 30) ~logits) in
          let x =
            Nx.create Nx.int32 [| 50 |]
              (Array.init 50 (fun i -> Int32.of_int (i mod 31)))
          in
          equal
            (array (float 1e-12))
            (Nx.to_array (D.factors (d (f64 0.5)) x))
            (Nx.to_array (Rune.jit' (fun l -> D.factors (d l) x) (f64 0.5))));
      test "a compiled draw is the eager one" (fun () ->
          let d logits = D.iid [| 50 |] (D.binomial ~trials:(i32 30) ~logits) in
          let k = Nx.Rng.key 4 in
          equal (array int32)
            (Nx.to_array (D.sample k (d (f64 0.5))))
            (Nx.to_array (Rune.jit' (fun l -> D.sample k (d l)) (f64 0.5))));
    ]

(* Count masses *)

type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

(* [dt]'s epsilon and least subnormal. *)
let eps_of (type b) (dt : (float, b) Nx.dtype) =
  let format = Nx.Dtype.float_format dt in
  (format.epsilon, format.min_normal *. format.epsilon)

(* The largest count [dt] holds with every count below it. *)
let top_of (type b) (dt : (float, b) Nx.dtype) =
  Float.ldexp 1. ((Nx.Dtype.float_format dt).fraction_bits + 1)

type count_family = Poisson | Binomial | Neg_binomial

(* The bound norn.mli states on the log mass [f] of a count [x] of mean [m]:
   [ε (c_f |f| + c_m |x - m|)], for [ε] the dtype's epsilon, and half the
   least subnormal [s]; a whole [s] against a reference [f] itself rounded
   to a float. *)
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

(* Against mpmath at float64 and float32, at the counts and trials each
   holds. *)
let exact_masses =
  let check (type b) (dt : (float, b) Nx.dtype) family a b x v =
    if x > top_of dt || (family = Binomial && a > top_of dt) then ()
    else
      let m = count_mean family a b in
      equal ~msg:(Nx.Dtype.name dt)
        (float (mass_bound family (eps_of dt) v x m))
        v
        (count_mass dt family a b x)
  in
  group "log masses against mpmath"
    (List.map
       (fun l ->
         match l with
         | [ _; name; a; b; x; v ] ->
             let a = float_of_string a and b = float_of_string b in
             let x = float_of_string x and v = float_of_string v in
             let family = count_family name in
             test (Printf.sprintf "%s %h %h at %.0f" name a b x) (fun () ->
                 check Nx.float64 family a b x v;
                 check Nx.float32 family a b x v)
         | _ -> failwith "dist.golden: a malformed mass line")
       (lines "mass"))

(* The log mass from log-gamma sums at float64, and a bound on its error:
   16 ulps on each log-gamma (nx's) and on each other term. Exact enough to
   judge the narrower dtypes, whose counts stay below [2^24]. *)
let lgamma v = item (Nx.lgamma (f64 v))
let log_sigmoid l = -.Float.log1p (Float.exp (-.l))

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

(* [family]'s log mass at float32, float16 and bfloat16, against float64 at
   parameters rounded to the dtype: [u] places the scale parameter between
   [1e-3] and the largest the dtype holds with room, [w] the other one. The
   narrower dtypes compute at float32 and round once, half an ulp more, and a
   least subnormal's half where the mass is subnormal. *)
let narrow_masses family name =
  prop
    (Printf.sprintf "%s's log mass at narrow dtypes is within its bound" name)
    Gen.(quad (int_range 0 2) (float_range 0. 1.) (float_range 0. 1.) place)
    (fun (i, u, w, at) ->
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
      cover (Nx.Dtype.name dt) true;
      cover "at zero" (x = 0.);
      cover "near the mean" (x > m /. 2. && x < 2. *. m);
      cover "far below the mean" (x > 0. && x <= m /. 2.);
      cover "far above the mean" (x >= 2. *. m);
      if family = Binomial then cover "at trials" (x = a && x > 0.);
      if family = Neg_binomial then begin
        cover "dispersion below 5.25" (b <= 5.25);
        cover "dispersion above 5.25" (b > 5.25)
      end;
      let rounding =
        if format.fraction_bits >= 23 then 0.
        else
          0.5 *. format.epsilon
          *. (Float.abs f +. format.min_normal)
      in
      equal
        (float (mass_bound family (eps_of Nx.float32) f x m +. rounding +. err))
        f
        (count_mass dt family a b x))

(* Where the means' rounding weighs most: a binomial count 4 to 6 standard
   deviations from its mean at up to [2^24] trials, at float32 against
   float64's log-gamma sums. *)
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

(* At means down to the dtype's least subnormal, the log masses' scores
   are finite wherever the exact ones are, at a count of [0] too, whose
   mass does not read the deviance. [lm] is the log of the mean, or the
   logit. *)
let small_means =
  let check (type b) (dt : (float, b) Nx.dtype) family lm x =
    let format = Nx.Dtype.float_format dt in
    let s v = Nx.scalar dt v in
    let p = Nx.item [] (Nx.cast Nx.float64 (s (match family with
      | Binomial -> lm | Poisson | Neg_binomial -> Float.exp lm)))
    in
    assume (p > 0. || family = Binomial);
    let n = 10. and r = 2. in
    let score, scale =
      match family with
      | Poisson -> ((x /. p) -. 1., (x /. p) +. 1.)
      | Binomial -> (x -. (n *. sigmoid p), x +. (n *. sigmoid p))
      | Neg_binomial ->
          ( r *. (x -. p) /. (p *. (r +. p)),
            (x /. p) +. ((x +. r) /. (r +. p)) )
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
      equal ~msg:(Nx.Dtype.name dt) float_exact
        (Float.copy_sign Float.infinity score) g
    else
      equal ~msg:(Nx.Dtype.name dt)
        (float ((64. *. format.epsilon *. scale) +. format.min_normal))
        score g
  in
  let lowest (type b) (dt : (float, b) Nx.dtype) =
    let format = Nx.Dtype.float_format dt in
    Float.log (format.min_normal *. format.epsilon)
  in
  List.map
    (fun family ->
      let name =
        match family with
        | Poisson -> "the poisson rate"
        | Binomial -> "the binomial logits"
        | Neg_binomial -> "the negative binomial mean"
      in
      prop
        (Printf.sprintf "%s's score is finite at small means" name)
        Gen.(triple bool (float_range 0. 1.) (int_range 0 3))
        (fun (wide, u, x) ->
          let x = float_of_int x in
          cover "at zero" (x = 0.);
          cover "above zero" (x > 0.);
          cover "float64" wide;
          cover "float32" (not wide);
          (* Logits past where their sigmoid underflows; log means from the
             least subnormal up to 0. *)
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
          if wide then at Nx.float64 else at Nx.float32))
    [ Poisson; Binomial; Neg_binomial ]

(* At dispersions down to the least subnormal, the negative binomial's
   score in its dispersion [r] is
   [Σ_(j < x) 1 / (r + j) + log (r / (r + mean)) + (mean - x) / (r + mean)],
   finite at [0] and overflowing above it where [1 / r] does; its score in
   the mean stays small. *)
let small_dispersions =
  prop "the negative binomial dispersion's score is finite at small dispersions"
    Gen.(quad bool (float_range 0. 1.) (int_range 0 2) (int_range 0 3))
    (fun (wide, u, k, x) ->
      let mean = [| 0.5; 3.; 1000. |].(k) and x = float_of_int x in
      cover "at zero" (x = 0.);
      cover "above zero" (x > 0.);
      cover "float64" wide;
      cover "float32" (not wide);
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
        equal ~msg:(Nx.Dtype.name dt ^ " mean")
          (float ((64. *. format.epsilon *. scale) +. format.min_normal))
          along gm
      in
      if wide then at Nx.float64 else at Nx.float32)

(* [rune]'s gradient of [f] at [p], against [score]. *)
let grad_of f p = item (Rune.grad' (fun p -> f p) (f64 p))

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
      prop "the poisson rate's score is x / rate - 1" count (fun (u, at) ->
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
            ((x /. rate) -. 1.) g);
      test "the bernoulli logit's score at 0 is x - 1/2" (fun () ->
          let score x =
            grad_of
              (fun logits ->
                D.factors (D.bernoulli ~logits) (Nx.scalar Nx.bool x))
              0.
          in
          equal (pair float_exact float_exact) (-0.5, 0.5)
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
      prop "the negative binomial mean's score is r (x - mean) / (mean (r + \
            mean))"
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
      (* Central differences at a step [h = r 1e-5]: truncation near
         [h² |f'''| / 6], about [3e-11 / r], and the masses' errors
         over [h], each within its bound. *)
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

let count_masses =
  group "count masses"
    [
      exact_masses;
      narrow_masses Poisson "poisson";
      narrow_masses Binomial "binomial";
      narrow_masses Neg_binomial "neg_binomial";
      binomial_tails;
      group "small means" small_means;
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
          equal (array exact_inf)
            [| Float.neg_infinity; 0.; Float.neg_infinity; Float.neg_infinity |]
            (Nx.to_array (D.factors d counts)));
      Windtrap.cases "a compiled log mass is the eager one"
        ~id:(fun (name, _) -> name)
        [
          ("poisson", fun p -> D.poisson ~rate:(Nx.mul_s p 3e5));
          ("binomial", fun p -> D.binomial ~trials:(i32 1_000_000) ~logits:p);
          ( "neg_binomial",
            fun p -> D.neg_binomial ~mean:(Nx.mul_s p 3e5) ~dispersion:p );
        ]
        (fun (_, d) ->
          let xs = [| 0l; 1l; 3l; 100_000l; 268_000l; 300_000l; 1_000_000l |] in
          let counts = Nx.create Nx.int32 [| 7 |] xs in
          let p = f64 1.5 in
          equal
            (array (float_rel ~rel:1e-14 ~abs:1e-14))
            (Nx.to_array (D.factors (d p) counts))
            (Nx.to_array (Rune.jit' (fun p -> D.factors (d p) counts) p)));
    ]

(* Continuous densities *)

(* The bound norn.mli states on a log density [f]:
   [ε (16 |f| + 48 Λ + 12 M)], for [ε] the dtype's epsilon and the family's
   [Λ] and [M] at the exact parameters and point. *)
let density_bound eps f ~logs ~off =
  eps *. ((16. *. Float.abs f) +. (48. *. logs) +. (12. *. off))

let alog v = Float.abs (Float.log v)

(* [Λ] and [M] of a dirichlet of concentrations [alpha] at [x], read as
   [x / Σx]. *)
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

(* The log density of a golden line's family at [dt]. *)
let density_at (type b) (dt : (float, b) Nx.dtype) family args =
  let s v = Nx.scalar dt v in
  let v xs = Nx.create dt [| Array.length xs |] xs in
  let f =
    match (family, args) with
    | "gamma", [ a; rate; x ] ->
        D.factors (D.gamma ~concentration:(s a) ~rate:(s rate)) (s x)
    | "inverse_gamma", [ a; scale; x ] ->
        D.factors (D.inverse_gamma ~concentration:(s a) ~scale:(s scale)) (s x)
    | "beta", [ a; b; x ] -> D.factors (D.beta ~a:(s a) ~b:(s b)) (s x)
    | "student_t", [ df; scale; x ] ->
        D.factors (D.student_t ~df:(s df) ~loc:(s 0.) ~scale:(s scale)) (s x)
    | "dirichlet", [ a1; a2; a3; x1; x2; x3 ] ->
        (* Points of float32 components sum to one within float32's
           rounding: read at a wider dtype, they are first put back on the
           simplex. *)
        let x = v [| x1; x2; x3 |] in
        D.factors
          (D.dirichlet ~concentration:(v [| a1; a2; a3 |]))
          (Nx.div x (Nx.sum x))
    | _ -> failwith ("dist.golden: a malformed density line for " ^ family)
  in
  Nx.item [] (Nx.cast Nx.float64 f)

(* [v] read at [dt] is [v]. *)
let holds (type b) (dt : (float, b) Nx.dtype) v =
  Float.equal v (Nx.item [] (Nx.cast Nx.float64 (Nx.scalar dt v)))

(* Against mpmath at float64 and float32, and at float16 and bfloat16 where
   they hold the arguments: those compute at float32 and round once, half an
   ulp more. *)
let exact_densities =
  let check (type b) (dt : (float, b) Nx.dtype) family args v =
    let format = Nx.Dtype.float_format dt in
    let fits = Float.abs v < format.max_finite in
    if List.for_all (holds dt) args && fits then begin
      let logs, off = density_terms family args in
      let eps, rounding =
        if format.fraction_bits >= 23 then (format.epsilon, 0.)
        else
          ( (Nx.Dtype.float_format Nx.float32).epsilon,
            0.5 *. format.epsilon *. (Float.abs v +. format.min_normal) )
      in
      equal ~msg:(Nx.Dtype.name dt)
        (float (density_bound eps v ~logs ~off +. rounding))
        v (density_at dt family args)
    end
  in
  group "log densities against mpmath"
    (List.map
       (fun l ->
         match l with
         | _ :: family :: rest ->
             let rest = floats rest in
             let n = List.length rest - 1 in
             let args = List.filteri (fun i _ -> i < n) rest in
             let v = List.nth rest n in
             test
               (Printf.sprintf "%s at %s" family
                  (String.concat " " (List.map (Printf.sprintf "%h") args)))
               (fun () ->
                 check Nx.float64 family args v;
                 check Nx.float32 family args v;
                 check Nx.float16 family args v;
                 check Nx.bfloat16 family args v)
         | _ -> failwith "dist.golden: a malformed density line")
       (lines "density"))

(* Scores of the continuous densities against their closed forms, from
   nx's digamma, within [64 ε] of the magnitudes they sum: the closed form
   itself cancels where the score nears [0]. Concentrations reach from
   [1e-300], where the score is near [1 / α], to [1e30]; points lie near the
   mode and far into both tails. *)
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

(* A unit gamma point of concentration [a]: [a] times [10^e]. *)
let unit_point a =
  Gen.(map (fun e -> a *. (10. ** e)) (float_range (-3.) 1.))

let score_tol terms =
  let scale = List.fold_left (fun s t -> s +. Float.abs t) 1. terms in
  float (64. *. epsilon_float *. scale)

let grad2 f a b =
  let ga, gb =
    Rune.grad
      Nx.Ptree.(pair tensor tensor)
      (fun (a, b) -> f a b)
      (f64 a, f64 b)
  in
  (item ga, item gb)

let power lo hi = Gen.(map (fun e -> 10. ** e) (float_range lo hi))

(* Against mpmath at float64 alone, where the mean count lies far from the
   concentration and the logarithm of their ratio changes form. *)
let far_densities =
  group "log densities far from the mean count against mpmath"
    (List.map
       (fun l ->
         match l with
         | [ _; family; a; s; x; v ] ->
             let args = floats [ a; s; x ] and v = float_of_string v in
             test
               (Printf.sprintf "%s at %s" family
                  (String.concat " " (List.map (Printf.sprintf "%h") args)))
               (fun () ->
                 let logs, off = density_terms family args in
                 equal
                   (float (density_bound epsilon_float v ~logs ~off))
                   v
                   (density_at Nx.float64 family args))
         | _ -> failwith "dist.golden: a malformed density64 line")
       (lines "density64"))

let continuous_scores =
  group "scores"
    [
      prop "the gamma's scores are log (rate x) - ψ(α) and α / rate - x"
        Gen.(
          let* a = concentration in
          let+ y = unit_point a and+ rate = power (-6.) 6. in
          (a, rate, y /. rate))
        (fun (a, rate, x) ->
          assume (Float.is_finite x && x > 0. && rate *. x > 0.);
          cover "concentration below 1e-3" (a < 1e-3);
          cover "concentration above 1e7" (a > 1e7);
          let ga, gr =
            grad2
              (fun a rate ->
                D.factors (D.gamma ~concentration:a ~rate) (f64 x))
              a rate
          in
          let y = rate *. x in
          equal ~msg:"concentration"
            (score_tol [ Float.log y; digamma a ])
            (Float.log y -. digamma a) ga;
          equal ~msg:"rate"
            (score_tol [ a /. rate; x ])
            ((a /. rate) -. x) gr);
      prop
        "the inverse gamma's scores are log (s / x) - ψ(α) and α / s - 1 / x"
        Gen.(
          let* a = concentration in
          let+ y = unit_point a and+ s = power (-6.) 6. in
          (a, s, s /. y))
        (fun (a, s, x) ->
          assume (Float.is_finite x && x > 0. && s /. x > 0.);
          cover "concentration below 1e-3" (a < 1e-3);
          cover "concentration above 1e7" (a > 1e7);
          let ga, gs =
            grad2
              (fun a scale ->
                D.factors (D.inverse_gamma ~concentration:a ~scale) (f64 x))
              a s
          in
          let y = s /. x in
          equal ~msg:"concentration"
            (score_tol [ Float.log y; digamma a ])
            (Float.log y -. digamma a) ga;
          equal ~msg:"scale"
            (score_tol [ a /. s; 1. /. x ])
            ((a /. s) -. (1. /. x))
            gs);
      prop "the beta's scores are log x - ψ(a) + ψ(a + b) and its mirror"
        Gen.(
          let+ a = concentration
          and+ b = concentration
          and+ z = float_range (-6.) 6. in
          let n = a +. b in
          let m = a /. n in
          let sd = Float.sqrt (m *. (1. -. m) /. (n +. 1.)) in
          (a, b, m +. (z *. sd)))
        (fun (a, b, x) ->
          assume (x > 0. && x < 1.);
          cover "a below 1e-3" (a < 1e-3);
          cover "a above 1e7" (a > 1e7);
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
            (l1 -. digamma b +. n) gb);
      (* Where the mean count [rate x] or [scale / x] is subnormal or [0],
         [dm / m] overflows; the scores stay the closed forms'. *)
      Windtrap.cases "the scores where the mean count underflows"
        ~id:(fun (name, _, _, _, _) -> name)
        [
          ("gamma, rate x of 0", false, 1., 1e-200, 1e-200);
          ("gamma, rate x subnormal", false, 1., 1., 1e-310);
          ("inverse gamma, scale / x subnormal", true, 2., 1e-10, 1e300);
          ("inverse gamma, scale / x of 0", true, 2., 1e-30, 1e300);
        ]
        (fun (_, inverse, a, s, x) ->
          let f a s =
            D.factors
              (if inverse then D.inverse_gamma ~concentration:a ~scale:s
               else D.gamma ~concentration:a ~rate:s)
              (f64 x)
          in
          let ga, gs = grad2 f a s in
          (* The log of the mean count, from the logs: [log (rate x)] or
             [log (scale / x)]. *)
          let ly =
            if inverse then Float.log s -. Float.log x
            else Float.log s +. Float.log x
          in
          let gs_exact =
            if inverse then (a /. s) -. (1. /. x) else (a /. s) -. x
          in
          equal ~msg:"concentration"
            (score_tol [ ly; digamma a ])
            (ly -. digamma a) ga;
          equal ~msg:"scale"
            (float_rel ~rel:(64. *. epsilon_float) ~abs:0.)
            gs_exact gs);
      prop "a dirichlet's scores are ψ(A) - ψ(α_k) + log x_k"
        Gen.(triple concentration concentration concentration)
        (fun (a1, a2, a3) ->
          let alpha = [| a1; a2; a3 |] in
          let total = a1 +. a2 +. a3 in
          let x = Array.map (fun a -> a /. total) alpha in
          assume (Array.for_all (fun v -> v > 0.) x);
          cover "a concentration above 1e7"
            (Array.exists (fun a -> a > 1e7) alpha);
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
      (* [∂ν f = (ψ((ν + 1) / 2) - ψ(ν / 2) - 1/ν - log1p (z²/ν)
         + (ν + 1) z² / (ν (ν + z²))) / 2] and
         [∂σ f = ((ν + 1) z² / (ν + z²) - 1) / σ]. *)
      prop "the student t's scores in its degrees of freedom and scale"
        ~examples:[ (1., 1., 1.); (4., 0.5, 2.) ]
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
            ((v -. 1.) /. scale) gs);
    ]

(* Each family at large parameters, at points near its mode and in its
   tails. *)
let compiled_densities =
  let pos = vec [| 1e-7; 0.25; 0.9; 1.; 1.1; 3.; 1e6 |] in
  let unit = vec [| 1e-7; 0.1; 0.249; 0.25; 0.2501; 0.5; 0.999 |] in
  let real = vec [| -1e6; -3.; -0.5; 0.; 1e-3; 2.; 1e20 |] in
  Windtrap.cases "a compiled density is the eager one"
    ~id:(fun (name, _, _) -> name)
    [
      ("gamma", pos, fun p -> D.gamma ~concentration:(Nx.mul_s p 1e6) ~rate:p);
      ( "inverse_gamma",
        pos,
        fun p -> D.inverse_gamma ~concentration:(Nx.mul_s p 1e6) ~scale:p );
      ("beta", unit, fun p -> D.beta ~a:(Nx.mul_s p 1e6) ~b:(Nx.mul_s p 3e6));
      ( "student_t",
        real,
        fun p -> D.student_t ~df:(Nx.mul_s p 1e7) ~loc:(f64 0.) ~scale:p );
    ]
    (fun (_, xs, d) ->
      let p = f64 1.5 in
      let f p = D.factors (D.iid [| 7 |] (d p)) xs in
      (* Each within its bound, a few hundred ulps of [|f|] and the logs. *)
      equal
        (array (float_rel ~rel:1e-12 ~abs:1e-11))
        (Nx.to_array (f p))
        (Nx.to_array (Rune.jit' f p)))

let bfloat16_simplex =
  test "at bfloat16 a dirichlet's tolerance is the point's rounding"
    (fun () ->
      let bf xs = Nx.create Nx.bfloat16 [| Array.length xs |] xs in
      let d = D.dirichlet ~concentration:(bf [| 1.; 1.; 1. |]) in
      let f x = Nx.item [] (Nx.cast Nx.float64 (D.factors d (bf x))) in
      satisfies ~claim:"finite" float_exact Float.is_finite
        (f [| 1. /. 3.; 1. /. 3.; 1. /. 3. |]);
      equal float_exact Float.neg_infinity (f [| 0.5; 0.5; 0.3 |]))

(* At float16, components of a dirichlet of many small concentrations
   underflowed: a draw then had density [-inf]. *)
let narrow_dirichlet_draws =
  Windtrap.cases "a narrow dirichlet's draws have finite densities"
    ~id:(fun k -> Printf.sprintf "%d components" k)
    [ 100; 1000 ]
    (fun k ->
      let d =
        D.dirichlet ~concentration:(Nx.full Nx.float16 [| k |] 0.7)
      in
      let x = D.sample (Nx.Rng.key 41) (D.iid [| 2000 |] d) in
      let f = Nx.to_array (Nx.cast Nx.float64 (D.factors (D.iid [| 2000 |] d) x)) in
      satisfies ~claim:"every density finite" (array float_exact)
        (Array.for_all Float.is_finite) f)

let compiled_dirichlet =
  test "a compiled dirichlet density is the eager one" (fun () ->
      let x =
        Nx.create Nx.float64 [| 2; 3 |] [| 0.2; 0.3; 0.5; 0.6; 0.39; 0.01 |]
      in
      let f a = D.factors (D.iid [| 2 |] (D.dirichlet ~concentration:a)) x in
      let a = vec [| 1.5; 2e6; 0.7 |] in
      equal
        (array (float_rel ~rel:1e-14 ~abs:1e-14))
        (Nx.to_array (f a))
        (Nx.to_array (Rune.jit' f a)))

let laplace =
  group "laplace"
    [
      prop "the laplace quantile's derivative is 2 scale / (1 - |2p - 1|)"
        ~examples:[ 0.5 ]
        Gen.(float_range 0.01 0.99)
        (fun p ->
          cover "at the median" (p = 0.5);
          let d = D.laplace ~loc:(f64 0.3) ~scale:(f64 1.3) in
          equal
            (float_rel ~rel:1e-12 ~abs:0.)
            (2.6 /. (1. -. Float.abs ((2. *. p) -. 1.)))
            (item (Rune.grad' (D.quantile d) (f64 p))));
    ]

let validation =
  group "parameter checks"
    [
      test "a scale outside its domain raises on use" (fun () ->
          let d = D.normal ~loc:(f64 0.) ~scale:(vec [| 1.; 2.; 0.; -1. |]) in
          raises
            (Invalid_argument
               "Norn.Dist.log_density: normal: scale at [2] is 0, not in (0, \
                inf)") (fun () -> D.log_density d (f64 0.)));
      test "a scalar parameter's message has no index" (fun () ->
          raises
            (Invalid_argument
               "Norn.Dist.sample: poisson: rate is -1, not in [0, inf)")
            (fun () -> D.sample (Nx.Rng.key 0) (D.poisson ~rate:(f64 (-1.)))));
      test "NaN is outside every domain" (fun () ->
          raises
            (Invalid_argument
               "Norn.Dist.factors: bernoulli: logits is nan, not in [-inf, inf]")
            (fun () ->
              D.factors
                (D.bernoulli ~logits:(f64 Float.nan))
                (Nx.scalar Nx.bool true)));
      test "a checked distribution is not checked again" (fun () ->
          let d = D.normal ~loc:(f64 0.) ~scale:(f64 (-1.)) in
          let d = D.check ~unless:(Nx.scalar Nx.bool true) "interpreter" d in
          satisfies ~claim:"nan" float_exact Float.is_nan
            (item (D.log_density d (f64 0.))));
      test "a check names its context" (fun () ->
          raises
            (Invalid_argument
               "site tau: half_normal: scale is 0, not in (0, inf)") (fun () ->
              D.check "site tau" (D.half_normal ~scale:(f64 0.))));
      test "parameters must broadcast" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"do not broadcast") (fun () ->
              D.normal ~loc:(vec [| 0.; 1. |]) ~scale:(vec [| 1.; 1.; 1. |])));
      test "a dirichlet needs two components" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"fewer than two components")
            (fun () -> D.dirichlet ~concentration:(vec [| 1. |])));
      test "trials are counted from zero" (fun () ->
          raises
            (Invalid_argument
               "Norn.Dist.log_density: binomial: trials at [1] is -2, not in \
                [0, inf)") (fun () ->
              D.log_density
                (D.binomial
                   ~trials:(Nx.create Nx.int32 [| 2 |] [| 3l; -2l |])
                   ~logits:(f64 0.))
                (i32 0)));
      test "the uniform's bounds are ordered" (fun () ->
          raises
            (Invalid_argument
               "Norn.Dist.log_density: uniform: high - low is 0, not in (0, \
                inf)") (fun () ->
              D.log_density (D.uniform ~low:(f64 1.) ~high:(f64 1.)) (f64 1.)));
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
  group "valid"
    [
      Windtrap.cases "a parameter of +inf is outside (0, inf)" ~id:fst
        infinite_positive (fun (_, v) -> equal bool false (Nx.item [] v));
      test "parameters inside their domains are valid" (fun () ->
          equal bool true
            (Nx.item [] (D.valid (D.normal ~loc:(f64 0.) ~scale:(f64 1.)))));
      test "NaN logits are not valid" (fun () ->
          equal bool false
            (Nx.item [] (D.valid (D.bernoulli ~logits:(f64 Float.nan)))));
      test "one bad element of a vector makes the scalar false" (fun () ->
          let v =
            D.valid (D.normal ~loc:(f64 0.) ~scale:(vec [| 1.; -1.; 2. |]))
          in
          equal (array int) [||] (Nx.shape v);
          equal bool false (Nx.item [] v));
      test "under a map, valid is one value per lane" (fun () ->
          let v =
            Rune.vmap
              Nx.Ptree.(tensor @-> returns tensor)
              (fun scale -> D.valid (D.normal ~loc:(f64 0.) ~scale))
              (vec [| 1.; -1.; 2. |])
          in
          equal (array bool) [| true; false; true |] (Nx.to_array v));
      test "a mixture is not valid with a bad component or NaN logits"
        (fun () ->
          let components scale = D.normal ~loc:(vec [| 0.; 1. |]) ~scale in
          equal bool false
            (Nx.item []
               (D.valid
                  (D.mixture
                     ~logits:(vec [| 0.; 0. |])
                     (components (vec [| 1.; -1. |])))));
          equal bool false
            (Nx.item []
               (D.valid
                  (D.mixture
                     ~logits:(vec [| 0.; Float.nan |])
                     (components (vec [| 1.; 1. |]))))));
      test "iid copies are valid as their distribution is" (fun () ->
          equal bool false
            (Nx.item []
               (D.valid (D.iid [| 3 |] (D.half_normal ~scale:(f64 (-1.)))))));
      test "a checked distribution still reads as not valid" (fun () ->
          let d =
            D.check ~unless:(Nx.scalar Nx.bool true) "interpreter"
              (D.normal ~loc:(f64 0.) ~scale:(f64 (-1.)))
          in
          equal bool false (Nx.item [] (D.valid d)));
      test "a NaN below the diagonal of scale_tril is not valid" (fun () ->
          equal bool false
            (Nx.item []
               (D.valid
                  (D.mvn
                     ~loc:(vec [| 0.; 0. |])
                     ~scale_tril:
                       (Nx.create Nx.float64 [| 2; 2 |]
                          [| 1.; 0.; Float.nan; 1. |])))));
    ]

let transforms =
  group "transformations"
    [
      test "a compiled density is the eager one" (fun () ->
          let d = D.student_t ~df:(f64 3.) ~loc:(f64 0.) ~scale:(f64 1.) in
          let x = vec [| -2.; 0.; 5. |] in
          let compiled = Rune.jit' (fun x -> D.factors (D.iid [| 3 |] d) x) x in
          equal
            (array (float 1e-12))
            (Nx.to_array (D.factors (D.iid [| 3 |] d) x))
            (Nx.to_array compiled));
      test "a normal draw differentiates in its parameters" (fun () ->
          let g =
            Rune.grad'
              (fun loc ->
                Nx.sum
                  (D.sample (Nx.Rng.key 3)
                     (D.iid [| 5 |] (D.normal ~loc ~scale:(f64 2.)))))
              (f64 0.)
          in
          equal close 5. (item g));
      test "a float32 density agrees with float64" (fun () ->
          let d32 =
            D.gamma ~concentration:(Nx.scalar Nx.float32 2.5)
              ~rate:(Nx.scalar Nx.float32 1.5)
          in
          let d64 = D.gamma ~concentration:(f64 2.5) ~rate:(f64 1.5) in
          equal
            (float_rel ~rel:1e-5 ~abs:1e-6)
            (item (D.factors d64 (f64 0.7)))
            (Nx.item [] (D.factors d32 (Nx.scalar Nx.float32 0.7))));
    ]

let () =
  exit
    (run "Norn.Dist"
       [
         densities;
         quantiles;
         gamma_quantiles;
         fit;
         normalisation;
         group "laws" [ sums ];
         combinators;
         bijectors;
         validation;
         valid;
         transforms;
         binomial;
         count_masses;
         group "continuous densities"
           [
             exact_densities;
             far_densities;
             continuous_scores;
             compiled_densities;
             compiled_dirichlet;
             bfloat16_simplex;
             narrow_dirichlet_draws;
           ];
         laplace;
       ])
