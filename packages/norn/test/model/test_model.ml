(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module M = Norn_model
module D = Norn.Dist

let f64 = Nx.scalar Nx.float64
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let item x = Nx.item [] x
let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* Eight schools *)

type 'a schools = { mu : 'a; tau : 'a; theta : 'a }

module Schools = struct
  type 'a t = 'a schools

  let walk c { mu; tau; theta } =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf mu in
    let tau = field c "tau" leaf tau in
    let theta = field c "theta" leaf theta in
    { mu; tau; theta }
end

let schools : Nx.float64_t schools Nx.Ptree.t =
  Nx.Ptree.instantiate (module Schools)

let y = vec [| 28.; 8.; -3.; 7.; -1.; 1.; 18.; 12. |]
let sigma = vec [| 15.; 10.; 16.; 11.; 9.; 11.; 10.; 18. |]

let eight_schools =
  M.v Nx.float64 schools Nx.Ptree.tensor @@ fun () ->
  let mu = M.sample (D.normal ~loc:(f64 0.) ~scale:(f64 5.)) in
  let tau = M.sample (D.half_cauchy ~scale:(f64 5.)) in
  let theta = M.sample (D.iid [| 8 |] (D.normal ~loc:mu ~scale:tau)) in
  let y = M.sample (D.normal ~loc:theta ~scale:sigma) in
  ({ mu; tau; theta }, y)

let noncentred = M.noncentre (fun p -> p.theta) eight_schools

(* Positions of [c] chains drawn at a fixed key, moderate in coordinates. *)
let positions c =
  let k = Nx.Rng.key 5 in
  {
    mu = Nx.Rng.normal (Nx.Rng.fold_in k 0) Nx.float64 [| c |];
    tau = Nx.Rng.normal (Nx.Rng.fold_in k 1) Nx.float64 [| c |];
    theta = Nx.Rng.normal (Nx.Rng.fold_in k 2) Nx.float64 [| c; 8 |];
  }

(* The density's arguments are coordinates; positions built here are taken as
   coordinates through the round trip of a model whose coordinates are its own
   structure. *)
let as_coords m (x : Nx.float64_t schools) : Nx.float64_t schools M.coords =
  Nx.Ptree.rebuild (M.coords m)
    ~like:(M.from_prior m ~n:1 (Nx.Rng.key 0))
    (fst (Nx.Ptree.flatten schools x))

let close = float_rel ~rel:1e-10 ~abs:1e-10

(* Law 1: one posterior *)

let one_posterior m name =
  test (name ^ ": the density is the prior plus the likelihood") (fun () ->
      let c = as_coords m (positions 4) in
      let d = M.log_density m y c
      and p = M.log_prior m c
      and l = M.log_likelihood m y c in
      equal (array close) (floats d) (floats (Nx.add p l)))

let pointwise_sums m name =
  test (name ^ ": the pointwise likelihood sums to the likelihood") (fun () ->
      let c = as_coords m (positions 1) in
      let l = Nx.item [ 0 ] (M.log_likelihood m y c) in
      let c1 = Nx.Ptree.map (M.coords m) (fun _ t -> Nx.slice [ Nx.I 0 ] t) c in
      let pw = M.pointwise m y (M.constrain m c1) in
      equal (array int) [| 8 |] (Nx.shape pw);
      equal close l (item (Nx.sum pw)))

let laws_1 =
  group "one posterior"
    [
      one_posterior eight_schools "centred";
      one_posterior noncentred "non-centred";
      pointwise_sums eight_schools "centred";
      pointwise_sums noncentred "non-centred";
    ]

(* Coordinates as one vector *)

let single m c = Nx.Ptree.map (M.coords m) (fun _ t -> Nx.slice [ Nx.I 0 ] t) c

(* [unflatten m like v] is the coordinates whose elements, in walk order, are
   the vector [v]. *)
let unflatten m like v =
  let offset = ref 0 in
  Nx.Ptree.map (M.coords m)
    (fun _ (type a b) (t : (a, b) Nx.t) : (a, b) Nx.t ->
      let n = Nx.numel t in
      let piece = Nx.shrink [| (!offset, !offset + n) |] v in
      offset := !offset + n;
      Nx.reshape (Nx.shape t) (Nx.cast (Nx.dtype t) piece))
    like

let flatten u x =
  Nx.concatenate ~axis:0
    (List.rev
       (Nx.Ptree.fold u
          (fun _ t acc -> Nx.reshape [| -1 |] (Nx.cast Nx.float64 t) :: acc)
          x []))

(* Law 6: a reparameterisation keeps the density of the values. *)
let change_of_variables m name =
  test (name ^ ": the density is the values' joint density times the Jacobian")
    (fun () ->
      let c = single m (as_coords m (positions 1)) in
      let v = flatten (M.coords m) c in
      let lhs = Nx.item [ 0 ] (M.log_density m y (as_coords m (positions 1))) in
      let values v = flatten schools (M.constrain m (unflatten m c v)) in
      let jac = Rune.jacfwd' values v in
      let _, logdet = Nx.slogdet jac in
      let rhs =
        item (M.log_joint eight_schools y (M.constrain m c)) +. item logdet
      in
      equal close rhs lhs)

let partially_centred =
  M.reparam
    (fun p -> p.theta)
    (fun _ -> Norn.Bij.affine ~loc:(f64 1.) ~scale:(f64 2.))
    eight_schools

let laws_6 =
  group "change of variables"
    [
      change_of_variables eight_schools "centred";
      change_of_variables noncentred "non-centred";
      change_of_variables partially_centred "reparameterised";
    ]

let round_trip =
  group "coordinates"
    [
      test "constrain undoes unconstrain" (fun () ->
          let p, _ = M.simulate eight_schools (Nx.Rng.key 2) in
          let p' = M.constrain eight_schools (M.unconstrain eight_schools p) in
          equal (array close) (floats p.theta) (floats p'.theta);
          equal close (item p.tau) (item p'.tau));
      test "non-centred coordinates are the standardised values" (fun () ->
          let p, _ = M.simulate eight_schools (Nx.Rng.key 2) in
          let c = (M.unconstrain noncentred p :> Nx.float64_t schools) in
          let z = Nx.div (Nx.sub p.theta p.mu) p.tau in
          equal (array close) (floats z) (floats c.theta));
      test "simulated values have a finite joint density" (fun () ->
          for i = 0 to 9 do
            let p, y = M.simulate eight_schools (Nx.Rng.key i) in
            satisfies ~claim:"finite" float_exact Float.is_finite
              (item (M.log_joint eight_schools y p))
          done);
    ]

(* Law 2: rows *)

let rows =
  test "each chain's density reads only its row" (fun () ->
      let c = as_coords eight_schools (positions 4) in
      let rev =
        Nx.Ptree.map (M.coords eight_schools)
          (fun _ t -> Nx.flip ~axes:[ 0 ] t)
          c
      in
      let d = floats (M.log_density eight_schools y c) in
      let r = floats (M.log_density eight_schools y rev) in
      equal (array close) (Array.of_list (List.rev (Array.to_list d))) r)

let compiled =
  test "a compiled density is the eager one" (fun () ->
      let c = as_coords eight_schools (positions 4) in
      let u = M.coords eight_schools in
      let eager = floats (M.log_density eight_schools y c) in
      let jit =
        Rune.jit
          Nx.Ptree.(u @-> returns tensor)
          (M.log_density eight_schools y)
          c
      in
      equal (array (float_rel ~rel:1e-9 ~abs:1e-9)) eager (floats jit))

let gradient =
  test "the density differentiates in its coordinates" (fun () ->
      let c = as_coords eight_schools (positions 2) in
      let u = M.coords eight_schools in
      let g =
        Rune.grad u (fun c -> Nx.sum (M.log_density eight_schools y c)) c
      in
      let g = (g :> Nx.float64_t schools) in
      Array.iter
        (fun x -> satisfies ~claim:"finite" float_exact Float.is_finite x)
        (floats g.theta))

(* The result rule *)

type 'a two = { a : 'a; b : 'a }

module Two = struct
  type 'a t = 'a two

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let two : Nx.float64_t two Nx.Ptree.t = Nx.Ptree.instantiate (module Two)
let std = D.normal ~loc:(f64 0.) ~scale:(f64 1.)
let model gen = M.v Nx.float64 two Nx.Ptree.tensor gen

(* Coordinates of [two], drawn from a model that has a prior everywhere. *)
let two_coords =
  lazy
    (M.from_prior
       (model (fun () ->
            let a = M.sample std and b = M.sample std in
            ({ a; b }, M.sample std)))
       ~n:1 (Nx.Rng.key 1))

let result_rule =
  group "the result rule"
    [
      test "a computed value in the result is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.v: the latent result at b is not a sampled value; a \
                model returns every sampled value unchanged") (fun () ->
              model (fun () ->
                  let a = M.sample std and b = M.sample std in
                  let y = M.sample std in
                  ({ a; b = Nx.exp b }, y))));
      test "a sample missing from the result is refused" (fun () ->
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "the 3rd sample, a normal of shape [], is not in the result")
            (fun () ->
              model (fun () ->
                  let a = M.sample std and b = M.sample std in
                  let _ = M.sample std in
                  ({ a; b }, M.sample std))));
      test "a value returned twice is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"appears twice") (fun () ->
              model (fun () ->
                  let a = M.sample std in
                  ({ a; b = a }, M.sample std))));
      test "a discrete latent site is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.v: site 1: poisson is over {0, 1, 2, ...}, and a \
                latent site needs a continuous support") (fun () ->
              M.v Nx.float64
                Nx.Ptree.(pair tensor tensor)
                Nx.Ptree.tensor
                (fun () ->
                  let a = M.sample std in
                  let b = M.sample (D.poisson ~rate:(f64 1.)) in
                  ((a, b), M.sample std))
              |> ignore));
      test "a run whose result holds another value is refused" (fun () ->
          let runs = ref 0 in
          let m =
            model (fun () ->
                incr runs;
                let a = M.sample std and b = M.sample std in
                ({ a; b = (if !runs > 1 then Nx.exp b else b) }, M.sample std))
          in
          raises
            (Invalid_argument
               "Norn_model.log_density: site b: the result holds another value \
                at b; a model returns every sampled value unchanged, outside \
                any loop or map") (fun () ->
              M.log_density m (f64 0.) (Lazy.force two_coords)));
      test "a fixed value of another shape is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.fix: site tau is float64 [], the value float64 [3]")
            (fun () ->
              M.fix (fun p -> p.tau) (Nx.zeros Nx.float64 [| 3 |]) eight_schools));
      test "a failure of the model names the sites learned" (fun () ->
          raises
            (Failure
               "Norn_model.v: the model raised after 1 site (a normal of shape \
                []): boom") (fun () ->
              model (fun () ->
                  ignore (M.sample std);
                  failwith "boom")));
      test "other exceptions of the model propagate" (fun () ->
          raises Not_found (fun () ->
              model (fun () ->
                  ignore (M.sample std);
                  raise Not_found)));
      test "a sample outside an interpreter is unhandled" (fun () ->
          raises_match
            (function Effect.Unhandled _ -> true | _ -> false)
            (fun () -> M.sample std));
      test "a model whose sites change raises naming the site" (fun () ->
          let n = ref 3 in
          let m =
            model (fun () ->
                let a = M.sample std and b = M.sample (D.iid [| !n |] std) in
                ({ a; b }, M.sample std))
          in
          let c = M.from_prior m ~n:2 (Nx.Rng.key 0) in
          n := 4;
          raises
            (Invalid_argument
               "Norn_model.log_density: site b: the model is not static: a \
                normal of shape [3] when it was built, a normal of shape [4] \
                now") (fun () ->
              M.log_density m (f64 0.)
                (Nx.Ptree.map (M.coords m) (fun _ t -> t) c)));
    ]

let observations =
  group "observations"
    [
      test "data of the wrong shape are refused at once" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.log_density: the observations do not match the \
                model: at y, data of shape [7], the site's shape [8]")
            (fun () ->
              M.log_density eight_schools (Nx.zeros Nx.float64 [| 7 |])));
      test
        "a parameter the model computes outside its domain raises naming the \
         site" (fun () ->
          let m =
            model (fun () ->
                let a = M.sample std in
                let b =
                  M.sample
                    (D.normal ~loc:(f64 0.) ~scale:(Nx.sub_s (Nx.abs a) 10.))
                in
                ({ a; b }, M.sample std))
          in
          let c = Lazy.force two_coords in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "Norn_model.log_density: site b, chain 0: normal: scale is -")
            (fun () -> M.log_density m (f64 0.) c));
      test "after a factor of -inf a bad parameter gives -inf" (fun () ->
          let m =
            model (fun () ->
                let a = M.sample std in
                M.factor (f64 Float.neg_infinity);
                let b =
                  M.sample
                    (D.normal ~loc:(f64 0.) ~scale:(Nx.sub_s (Nx.abs a) 10.))
                in
                ({ a; b }, M.sample std))
          in
          let c = Lazy.force two_coords in
          equal float_exact Float.neg_infinity
            (Nx.item [ 0 ] (M.log_density m (f64 0.) c)));
      test "a factor joins the likelihood as its points" (fun () ->
          let m =
            model (fun () ->
                let a = M.sample std and b = M.sample std in
                M.factor (vec [| -1.; -2. |]);
                ({ a; b }, M.sample std))
          in
          let p, _ = M.simulate m (Nx.Rng.key 0) in
          let pw = M.pointwise m (f64 0.5) p in
          equal (array int) [| 3 |] (Nx.shape pw);
          equal (array close) [| -1.; -2. |] (Array.sub (floats pw) 0 2));
    ]

(* One latent site of three elements. *)
let triple =
  M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
      let x = M.sample (D.iid [| 3 |] std) in
      (x, M.sample (D.normal ~loc:(Nx.sum x) ~scale:(f64 1.))))

(* Refusals to start, named by the data a check carries *)

let refusals =
  group "refusals to start"
    [
      test "an observation outside its support is named by its element"
        (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let rate =
                  M.sample (D.gamma ~concentration:(f64 2.) ~rate:(f64 1.))
                in
                (rate, M.sample (D.iid [| 2; 3 |] (D.poisson ~rate))))
          in
          let counts =
            Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 0l; 4l; -3l; 1l |]
          in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "no finite log density in 100 candidates; at the last, site y \
                  has log density -inf: element [1; 1] is outside the support \
                  of poisson, {0, 1, 2, ...}; the terms are p ") (fun () ->
              M.init m counts ~chains:1 (Nx.Rng.key 0)));
      test "a saturated coordinate is named as such" (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let x = M.sample (D.exponential ~rate:(f64 1.)) in
                (x, M.sample (D.normal ~loc:x ~scale:(f64 1.))))
          in
          let p = Nx.scalar Nx.float64 1e-320 in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "site p has log density -inf: its coordinates saturate: the \
                  log-determinant is -inf") (fun () ->
              M.init ~from:(M.Init.near p) m (f64 0.) ~chains:1 (Nx.Rng.key 0)));
      test "a compiled start names the support from its arguments" (fun () ->
          let m low high =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let x = M.sample std in
                (x, M.sample (D.uniform ~low ~high)))
          in
          let start =
            Rune.jit
              Nx.Ptree.(
                Nx.Rng.ptree @-> tensor @-> tensor @-> tensor @-> returns tensor)
              (fun k low high y ->
                (M.init (m low high) y ~chains:1 k :> Nx.float64_t))
          in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "site y has log density -inf: it is outside the support of \
                  uniform, (2, 3)") (fun () ->
              start (Nx.Rng.key 0) (f64 2.) (f64 3.) (f64 5.)));
    ]

let coordinate_shapes =
  group "coordinate shapes"
    [
      test "batched coordinates of the wrong shape are refused" (fun () ->
          let c = M.from_prior triple ~n:2 (Nx.Rng.key 0) in
          let c =
            Nx.Ptree.map (M.coords triple)
              (fun _ t -> Nx.shrink [| (0, 2); (0, 2) |] t)
              c
          in
          raises
            (Invalid_argument
               "Norn_model.log_density: the position does not match the model: \
                at p, coordinates of shape [2; 2], a position of 2 chains has \
                [2; 3]") (fun () -> M.log_density triple (f64 0.) c));
      test "one instance's coordinates of the wrong shape are refused"
        (fun () ->
          let c = M.from_prior triple ~n:1 (Nx.Rng.key 0) in
          let c =
            Nx.Ptree.map (M.coords triple)
              (fun _ t -> Nx.shrink [| (0, 1); (0, 3) |] t)
              c
          in
          raises
            (Invalid_argument
               "Norn_model.constrain: the coordinates do not match the model: \
                at p, coordinates of shape [1; 3], the site's [3]") (fun () ->
              M.constrain triple c));
    ]

(* A scale vector whose middle element is the latent [a]. *)
let scaled =
  M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
      let a = M.sample std in
      let scale =
        Nx.concatenate ~axis:0
          [ vec [| 1. |]; Nx.reshape [| 1 |] a; vec [| 1. |] ]
      in
      let b = M.sample (D.normal ~loc:(f64 0.) ~scale) in
      ({ a; b }, M.sample std))

(* Chain 0 at [a = 0.5], chain 1 at [a = -2]. *)
let scaled_coords =
  lazy
    (let twin =
       M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
           let a = M.sample std and b = M.sample (D.iid [| 3 |] std) in
           ({ a; b }, M.sample std))
     in
     Nx.Ptree.map (M.coords scaled)
       (fun p t ->
         if Nx.Ptree.Path.to_string p = "a" then
           Nx.cast (Nx.dtype t) (vec [| 0.5; -2. |])
         else t)
       (M.from_prior twin ~n:2 (Nx.Rng.key 0)))

let chains =
  group "chains in messages"
    [
      test "a vector parameter names its element and the chain" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.log_density: site b, chain 1: normal: scale at [1] \
                is -2, not in (0, inf)") (fun () ->
              M.log_density scaled (f64 0.) (Lazy.force scaled_coords)));
      test "a refusal under init names the chain, not the candidate" (fun () ->
          let p = { a = f64 (-0.05); b = vec [| 0.1; 0.2; 0.3 |] } in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "Norn_model.log_density: site b, chain 0: normal: scale at \
                  [1] is") (fun () ->
              M.init ~from:(M.Init.near p) scaled (f64 0.) ~chains:3
                (Nx.Rng.key 0)));
    ]

let inits =
  group "init"
    [
      test "every chain starts where the density is finite" (fun () ->
          let c = M.init eight_schools y ~chains:4 (Nx.Rng.key 0) in
          Array.iter
            (fun x -> satisfies ~claim:"finite" float_exact Float.is_finite x)
            (floats (M.log_density eight_schools y c)));
      test "a chain's start does not depend on the chain count" (fun () ->
          let c2 =
            (M.init eight_schools y ~chains:2 (Nx.Rng.key 0)
              :> Nx.float64_t schools)
          in
          let c4 =
            (M.init eight_schools y ~chains:4 (Nx.Rng.key 0)
              :> Nx.float64_t schools)
          in
          equal (array float_exact) (floats c2.theta)
            (floats (Nx.shrink [| (0, 2); (0, 8) |] c4.theta)));
      test "a prior start is finite" (fun () ->
          let c =
            M.init ~from:M.Init.prior eight_schools y ~chains:3 (Nx.Rng.key 4)
          in
          Array.iter
            (fun x -> satisfies ~claim:"finite" float_exact Float.is_finite x)
            (floats (M.log_density eight_schools y c)));
      test "a start near values is near them" (fun () ->
          let p, _ = M.simulate eight_schools (Nx.Rng.key 2) in
          let c =
            (M.init ~from:(M.Init.near p) eight_schools y ~chains:2
               (Nx.Rng.key 4)
              :> Nx.float64_t schools)
          in
          let u = (M.unconstrain eight_schools p :> Nx.float64_t schools) in
          less (float 1e-12) ~than:0.1
            (item (Nx.max (Nx.abs (Nx.sub c.mu u.mu)))));
      test "no finite candidate raises" (fun () ->
          let m =
            model (fun () ->
                let a = M.sample std and b = M.sample std in
                ({ a; b }, M.sample (D.uniform ~low:(f64 0.) ~high:(f64 1.))))
          in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "Norn_model.init: no finite log density in 100 candidates; at \
                  the last, site y has log density -inf: it is outside the \
                  support of uniform, (0, 1); the terms are a ") (fun () ->
              M.init m (f64 5.) ~chains:2 (Nx.Rng.key 0)));
      test "a prior draw does not depend on the draw count" (fun () ->
          let a =
            (M.from_prior eight_schools ~n:2 (Nx.Rng.key 9)
              :> Nx.float64_t schools)
          in
          let b =
            (M.from_prior eight_schools ~n:5 (Nx.Rng.key 9)
              :> Nx.float64_t schools)
          in
          equal (array float_exact) (floats a.mu)
            (floats (Nx.shrink [| (0, 2) |] b.mu)));
    ]

let fixed =
  test "a fixed site has no coordinates and keeps its value" (fun () ->
      let m = M.fix (fun p -> p.tau) (f64 2.) eight_schools in
      let c = M.from_prior m ~n:3 (Nx.Rng.key 0) in
      let c' = (c :> Nx.float64_t schools) in
      equal (array int) [| 3; 0 |] (Nx.shape c'.tau);
      let p = M.constrain m (single m c) in
      equal float_exact 2. (item p.tau);
      let with_tau = M.log_joint eight_schools y p in
      let tau_term =
        item (D.log_density (D.half_cauchy ~scale:(f64 5.)) (f64 2.))
      in
      equal close
        (item with_tau -. tau_term)
        (Nx.item [ 0 ] (M.log_density m y c)))

let kinds =
  group "values of every kind"
    [
      test "counts are simulated and scored" (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let rate =
                  M.sample (D.gamma ~concentration:(f64 2.) ~rate:(f64 1.))
                in
                let counts = M.sample (D.iid [| 5 |] (D.poisson ~rate)) in
                (rate, counts))
          in
          let rate, counts = M.simulate m (Nx.Rng.key 3) in
          equal string "int32" (Nx_dtype.to_string (Nx.dtype counts));
          satisfies ~claim:"finite" float_exact Float.is_finite
            (item (M.log_joint m counts rate));
          let c = M.init m counts ~chains:2 (Nx.Rng.key 0) in
          equal (array int) [| 2 |] (Nx.shape (M.log_density m counts c)));
      test "a simplex site has one coordinate fewer" (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let w =
                  M.sample (D.dirichlet ~concentration:(vec [| 1.; 2.; 3. |]))
                in
                (w, M.sample (D.iid [| 4 |] (D.categorical ~logits:(Nx.log w)))))
          in
          let c = M.from_prior m ~n:2 (Nx.Rng.key 0) in
          equal (array int) [| 2; 2 |] (Nx.shape (c :> Nx.float64_t));
          let w, _ = M.simulate m (Nx.Rng.key 1) in
          let w' = M.constrain m (M.unconstrain m w) in
          equal (array close) (floats w) (floats w'));
      test "terms are each site's part of the density" (fun () ->
          let c = as_coords eight_schools (positions 3) in
          let ts, factors = M.terms eight_schools y c in
          equal (list string) [ "mu"; "tau"; "theta"; "y" ] (List.map fst ts);
          let sum = List.fold_left (fun a (_, t) -> Nx.add a t) factors ts in
          equal (array close)
            (floats (M.log_density eight_schools y c))
            (floats sum));
    ]

let printing =
  test "the site table" (fun () ->
      expect (Format.asprintf "%a" M.pp noncentred)
      @@ __POS_OF__
           {|
        site   role      family       shape  points  support      coordinates
        mu     latent    normal       []             (-inf, inf)  identity
        tau    latent    half_cauchy  []             (0, inf)     exp
        theta  latent    normal       [8]            (-inf, inf)  affine
        y      observed  normal       [8]    8       (-inf, inf)
        |})

let () =
  exit
    (run "Norn_model"
       [
         laws_1;
         laws_6;
         round_trip;
         group "rows" [ rows; compiled; gradient ];
         result_rule;
         observations;
         inits;
         group "fix" [ fixed ];
         kinds;
         coordinate_shapes;
         refusals;
         chains;
         group "pp" [ printing ];
       ])
