(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module D = Norn.Dist

let f64 = Nx.scalar Nx.float64
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let item x = Nx.item [] x

(* The cases of gen/dist.py, at its parameters. *)
type case =
  | Real of (Nx.float64_t, Nx.float64_elt) D.t
  | Count of (Nx.int32_t, Nx.float64_elt) D.t
  | Flag of (Nx.bool_t, Nx.float64_elt) D.t
  | Index of (Nx.int64_t, Nx.float64_elt) D.t

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
         slow name (fun () ->
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
          raises_match (Exn.invalid_arg ~substring:"not over scalars")
            (fun () ->
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
          raises_match (Exn.invalid_arg ~substring:"gamma has no standard form")
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
          raises_match (Exn.invalid_arg ~substring:"do not broadcast")
            (fun () ->
              D.normal ~loc:(vec [| 0.; 1. |]) ~scale:(vec [| 1.; 1.; 1. |])));
      test "a dirichlet needs two components" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"fewer than two components")
            (fun () -> D.dirichlet ~concentration:(vec [| 1. |])));
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
      Windtrap.cases ~name:fst "a parameter of +inf is outside (0, inf)"
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
         fit;
         normalisation;
         group "laws" [ sums ];
         combinators;
         bijectors;
         validation;
         valid;
         transforms;
       ])
