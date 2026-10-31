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
  group "quantile against SciPy"
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

let sigmoid l = 1. /. (1. +. Float.exp (-.l))

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
       ])
