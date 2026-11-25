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
let std = D.normal ~loc:(f64 0.) ~scale:(f64 1.)
let log_std x = item (D.log_density std (f64 x))

(* Structures *)

type 'a schools = { mu : 'a; tau : 'a; theta : 'a }

module Schools = struct
  type ('a, _) t = 'a schools

  let walk c { mu; tau; theta } =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf mu in
    let tau = field c "tau" leaf tau in
    let theta = field c "theta" leaf theta in
    { mu; tau; theta }
end

let schools : (Nx.host Nx.float64_t schools, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module Schools)

type 'a two = { a : 'a; b : 'a }

module Two = struct
  type ('a, _) t = 'a two

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let two : (Nx.host Nx.float64_t two, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module Two)

(* The models the laws hold on *)

(* Eight schools, centred: normal, half-Cauchy (exp coordinates) and iid normal
   latent sites, and a normal observed site. *)
let y_schools = vec [| 28.; 8.; -3.; 7.; -1.; 1.; 18.; 12. |]
let sigma = vec [| 15.; 10.; 16.; 11.; 9.; 11.; 10.; 18. |]

let centred =
  lazy
    ( M.v Nx.float64 schools Nx.Ptree.tensor @@ fun () ->
      let mu = M.sample (D.normal ~loc:(f64 0.) ~scale:(f64 5.)) in
      let tau = M.sample (D.half_cauchy ~scale:(f64 5.)) in
      let theta = M.sample (D.iid [| 8 |] (D.normal ~loc:mu ~scale:tau)) in
      let y = M.sample (D.normal ~loc:theta ~scale:sigma) in
      ({ mu; tau; theta }, y) )

let noncentred = lazy (M.noncentre (fun p -> p.theta) (Lazy.force centred))

let affine =
  lazy
    (M.reparam
       (fun p -> p.theta)
       (fun _ -> Norn.Bij.affine ~loc:(f64 1.) ~scale:(f64 2.))
       (Lazy.force centred))

let fixed = lazy (M.fix (fun p -> p.tau) (f64 2.) (Lazy.force centred))

(* A Dirichlet weight (simplex coordinates) and categorical draws. *)
let simplex =
  lazy
    ( M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor @@ fun () ->
      let w = M.sample (D.dirichlet ~concentration:(vec [| 1.; 2.; 3. |])) in
      (w, M.sample (D.iid [| 4 |] (D.categorical ~logits:(Nx.log w)))) )

let categories = Nx.create Nx.int64 [| 4 |] [| 0L; 2L; 1L; 2L |]

(* Sites performed out of the result's walk order ([b] before [a]), latent sites
   bounded above and on both sides, counts, and a factor between two observed
   sites. *)
let bounded =
  lazy
    ( M.v Nx.float64 two Nx.Ptree.(pair tensor tensor) @@ fun () ->
      let b = M.sample (D.truncated ~high:(f64 2.) std) in
      let a = M.sample (D.uniform ~low:(f64 (-1.)) ~high:(f64 1.)) in
      let counts = M.sample (D.iid [| 3 |] (D.poisson ~rate:(Nx.exp a))) in
      M.factor (Nx.stack [ Nx.neg (Nx.square b); Nx.neg (Nx.square a) ]);
      let z = M.sample (D.iid [| 2 |] (D.normal ~loc:b ~scale:(f64 1.))) in
      ({ a; b }, (counts, z)) )

let bounded_data =
  (Nx.create Nx.int32 [| 3 |] [| 1l; 0l; 3l |], vec [| 0.5; -0.3 |])

(* A model with its data. *)
type model =
  | Model : {
      name : string;
      model : ('p, 'y, Nx.float64_elt, Nx.host) M.t Lazy.t;
      latent : ('p, Nx.host) Nx.Ptree.t;
      observed : ('y, Nx.host) Nx.Ptree.t;
      data : 'y;
    }
      -> model

let schools_like name model =
  Model
    {
      name;
      model;
      latent = schools;
      observed = Nx.Ptree.tensor;
      data = y_schools;
    }

let laws =
  [
    schools_like "centred" centred;
    schools_like "non-centred" noncentred;
    schools_like "affine" affine;
    schools_like "fixed" fixed;
    Model
      {
        name = "simplex";
        model = simplex;
        latent = Nx.Ptree.tensor;
        observed = Nx.Ptree.tensor;
        data = categories;
      };
    Model
      {
        name = "bounded";
        model = bounded;
        latent = two;
        observed = Nx.Ptree.(pair tensor tensor);
        data = bounded_data;
      };
  ]

let name (Model l) = l.name

(* [covers models m] marks the case with the model it draws. *)
let covers models m =
  List.iter (fun l -> cover (name l) (name l = name m)) models

(* Coordinates as one vector *)

(* [vector u x] is [x]'s elements, in walk order, as one float64 vector. *)
let vector u x =
  Nx.concatenate ~axis:0
    (vec [||]
    :: List.rev
         (Nx.Ptree.fold u
            (fun _ t acc -> Nx.reshape [| -1 |] (Nx.cast Nx.float64 t) :: acc)
            x []))

(* [one m] is one instance's coordinates of [m]. *)
let one m = M.unconstrain m (fst (M.simulate m (Nx.Rng.key 0)))

let coordinate_count m =
  Nx.Ptree.fold (M.coords m) (fun _ t n -> n + Nx.numel t) (one m) 0

(* [at m ~chains v] is the position of [chains] chains whose elements, site by
   site in walk order and chain by chain within a site, are [v]. *)
let at m ~chains v =
  let offset = ref 0 in
  Nx.Ptree.map (M.coords m)
    (fun _ (type a b) (t : (a, b, _) Nx.t) : (a, b, _) Nx.t ->
      let n = chains * Nx.numel t in
      let piece = Nx.slice [ Nx.R (!offset, !offset + n) ] v in
      offset := !offset + n;
      Nx.reshape
        (Array.append [| chains |] (Nx.shape t))
        (Nx.cast (Nx.dtype t) piece))
    (one m)

(* [instance m c] is the coordinates of [c]'s first chain. *)
let instance m c =
  Nx.Ptree.map (M.coords m) (fun _ t -> Nx.slice [ Nx.I 0 ] t) c

(* [chain m c i] is the one-chain position holding [c]'s chain [i]. *)
let chain m c i =
  Nx.Ptree.map (M.coords m) (fun _ t -> Nx.slice [ Nx.R (i, i + 1) ] t) c

(* [scale m y c] is, per chain, the sum of the finite terms' magnitudes, the
   scale of a density's rounding error. *)
let scale m y c =
  let sites, factors = M.terms m y c in
  let finite t = Nx.where (Nx.isfinite t) (Nx.abs t) (Nx.zeros_like t) in
  floats
    (List.fold_left
       (fun acc (_, t) -> Nx.add acc (finite t))
       (finite factors) sites)

(* Two computations of one density that sum its terms in other orders differ by
   at most γ(k) Σ|terms| each, k the summands of a reduction: at most 64 here,
   so γ(k) < 64u = 7.1e-15. nx's transcendental functions are correctly rounded
   on the host, eagerly and compiled. 1e-12 (1 + Σ|terms|) covers both sides
   with room for cancellation inside a site's term. A chain at -inf must be -inf
   on both sides. *)
let agrees s = float_rel ~rel:0. ~abs:(1e-12 *. (1. +. s))

let densities_agree ~scale expected actual =
  Array.iteri
    (fun i e ->
      equal ~msg:(Printf.sprintf "chain %d" i) (agrees scale.(i)) e actual.(i))
    expected

(* [positions models ~chains ~box] draws one of [models] and a position of
   [chains] chains with coordinates in (-box, box). *)
let positions models ~chains ~box =
  let pp ppf (m, v) =
    Format.fprintf ppf "(%s, [|%a|])" (name m)
      Format.(
        pp_print_list ~pp_sep:(fun p () -> fprintf p ";@ ") pp_print_float)
      (Array.to_list v)
  in
  Gen.with_pp pp
    Gen.(
      let* (Model l as m) = of_list models in
      let n = chains * coordinate_count (Lazy.force l.model) in
      let+ v = array ~size:(constant n) (float_range (-.box) box) in
      (m, v))

(* One example per model, its position at the origin. *)
let at_origin models ~chains =
  List.map
    (fun (Model l as m) ->
      (m, Array.make (chains * coordinate_count (Lazy.force l.model)) 0.))
    models

(* One example per model: chains at +40, -40 and 0, where the first two saturate
   the bijectors. *)
let saturating models =
  List.map
    (fun (Model l as m) ->
      let model = Lazy.force l.model in
      let site _ t acc =
        let n = Nx.numel t in
        acc
        @ List.concat_map (fun x -> List.init n (fun _ -> x)) [ 40.; -40.; 0. ]
      in
      (m, Array.of_list (Nx.Ptree.fold (M.coords model) site (one model) [])))
    models

(* A law model and a key for [simulate]. *)
let simulations =
  Gen.(
    pair
      (of_list ~pp:(fun ppf m -> Format.pp_print_string ppf (name m)) laws)
      (int_range 0 999))

(* The result rule and static models: every interpreter *)

let interpreters =
  [
    "log_density";
    "log_prior";
    "log_likelihood";
    "terms";
    "from_prior";
    "init";
    "log_joint";
    "pointwise";
    "simulate";
    "predict";
    "constrain";
    "unconstrain";
  ]

(* [interpret name m ~c ~p] runs the interpreter [name] of [m], a model over
   [two] observing one scalar, at the position [c] or the values [p]. *)
let interpret name m ~c ~p =
  let y = f64 0. and k = Nx.Rng.key 1 in
  match name with
  | "log_density" -> ignore (M.log_density m y c)
  | "log_prior" -> ignore (M.log_prior m y c)
  | "log_likelihood" -> ignore (M.log_likelihood m y c)
  | "terms" -> ignore (M.terms m y c)
  | "from_prior" -> ignore (M.from_prior m y ~n:2 k)
  | "init" -> ignore (M.init m y ~chains:2 k)
  | "log_joint" -> ignore (M.log_joint m y p)
  | "pointwise" -> ignore (M.pointwise m y p)
  | "simulate" -> ignore (M.simulate m k)
  | "predict" -> ignore (M.predict m k p)
  | "constrain" -> ignore (M.constrain m (instance m c))
  | "unconstrain" -> ignore (M.unconstrain m p)
  | _ -> invalid_arg name

(* [changed make] is the model [make later] once [later] is set, after [v] and
   the draws of a position and of values from it. *)
let changed make =
  let later = ref false in
  let m = make later in
  let c = M.from_prior m (f64 0.) ~n:2 (Nx.Rng.key 0)
  and p = fst (M.simulate m (Nx.Rng.key 0)) in
  later := true;
  (m, c, p)

(* Returns [Nx.exp b] where it sampled [b], once [later] is set. *)
let computing later =
  M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
      let a = M.sample std and b = M.sample std in
      ({ a; b = (if !later then Nx.exp b else b) }, M.sample std))

(* Samples [b] with another shape once [later] is set. *)
let growing later =
  M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
      let a = M.sample std in
      let b = M.sample (D.iid [| (if !later then 4 else 3) |] std) in
      ({ a; b }, M.sample std))

let result_rule =
  group "the result rule"
    [
      test "a computed value in the result is refused" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Norn_model.v: the latent result at b")
            (fun () ->
              M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
                  let a = M.sample std and b = M.sample std in
                  ({ a; b = Nx.exp b }, M.sample std))));
      cases "every interpreter checks it on every run, naming the site"
        ~id:Fun.id interpreters (fun name ->
          let m, c, p = changed computing in
          raises_match (Exn.invalid_arg ~sub:": site b: ") (fun () ->
              interpret name m ~c ~p));
    ]

let static_models =
  group "static models"
    [
      cases "every interpreter checks each site, naming it" ~id:Fun.id
        interpreters (fun name ->
          let m, c, p = changed growing in
          raises_match (Exn.invalid_arg ~sub:"site b: the model is not static")
            (fun () -> interpret name m ~c ~p));
      test "a sample inside a map does not reach the result unchanged"
        (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Norn_model.v: the latent result at p ")
            (fun () ->
              M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                  (Nx.stack (List.init 3 (fun _ -> M.sample std)), M.sample std))));
    ]

(* Interpreters under rune's transformations *)

let compiled_is_eager (Model l) =
  Compiled.here ();
  let m = Lazy.force l.model in
  let lp = M.log_density m l.data in
  let compiled = Rune.jit Nx.Ptree.(M.coords m @-> returns tensor) lp in
  let n = coordinate_count m in
  let drawn =
    Nx.sub_s
      (Nx.mul_s (Nx.Rng.uniform ~key:(Nx.Rng.key 3) Nx.float64 [| 3 * n |]) 10.)
      5.
  in
  let _, saturated =
    List.find (fun (m', _) -> name m' = l.name) (saturating laws)
  in
  List.iter
    (fun v ->
      let c = at m ~chains:3 v in
      densities_agree ~scale:(scale m l.data c)
        (floats (lp c))
        (floats (compiled c)))
    [ vec saturated; drawn ]

(* Central differences D(h) = (f(x + h) - f(x - h)) / 2h against the gradient,
   along each coordinate, at h = 1e-4. Each density is computed in one order,
   within 128 u Σ|terms| of the exact value, so rounding moves D(h) by at most
   128 u Σ|terms| / h. Truncation is h² f‴ / 6 + O(h⁴), and D(2h) - D(h) = h² f‴
   / 2 + O(h⁴) is three times it (Richardson), so |D(2h) - D(h)| bounds it with
   room for the O(h⁴) terms. *)
let gradient_law ((Model l as model), v) =
  covers laws model;
  let m = Lazy.force l.model in
  let u = M.coords m in
  let density v =
    item (Nx.sum (M.log_density m l.data (at m ~chains:1 (vec v))))
  in
  let c = at m ~chains:1 (vec v) in
  let g =
    floats
      (vector u (Rune.grad u (fun c -> Nx.sum (M.log_density m l.data c)) c))
  in
  let h = 1e-4 and s = (scale m l.data c).(0) in
  let rounding = 128. *. (epsilon_float /. 2.) *. s /. h in
  let moved i d = Array.mapi (fun j x -> if i = j then x +. d else x) v in
  let difference i h =
    (density (moved i h) -. density (moved i (-.h))) /. (2. *. h)
  in
  Array.iteri
    (fun i g_i ->
      let truncation = Float.abs (difference i (2. *. h) -. difference i h) in
      equal
        ~msg:(Printf.sprintf "coordinate %d" i)
        (float (rounding +. truncation +. Float.min_float))
        (difference i h) g_i)
    g

(* An mvn whose correlation [r] is the latent value: at [r = 2] the matrix is
   not positive definite and its Cholesky factor holds NaN. *)
let correlated () =
  M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
      let r = M.sample std in
      let cov =
        Nx.add (Nx.eye Nx.float64 2)
          (Nx.mul r (Nx.create Nx.float64 [| 2; 2 |] [| 0.; 1.; 1.; 0. |]))
      in
      let x =
        M.sample
          (D.mvn
             ~loc:(Nx.zeros Nx.float64 [| 2 |])
             ~scale_tril:(Nx.cholesky cov))
      in
      (r, x))

(* [finite_then_neg_inf lp] asserts that chain 0 is finite and chain 1 -inf: a
   parameter outside its domain in one chain leaves the other alone. *)
let finite_then_neg_inf lp =
  satisfies ~claim:"finite" float_exact Float.is_finite (Nx.item [ 0 ] lp);
  equal float_exact Float.neg_infinity (Nx.item [ 1 ] lp)

(* [holding m xs] is the position of [m], one latent scalar, whose chains hold
   [xs]. *)
let holding m xs = at m ~chains:(Array.length xs) (vec xs)

let interpreters_under_rune =
  group "interpreters"
    [
      cases "a compiled density is the eager one" ~id:name laws
        compiled_is_eager;
      prop "a gradient is the central differences of the density" ~count:20
        ~examples:(at_origin laws ~chains:1)
        (positions laws ~chains:1 ~box:2.)
        gradient_law;
      test "a compiled density is -inf where a Cholesky factor holds NaN"
        (fun () ->
          Compiled.here ();
          let m = correlated () in
          let lp = M.log_density m (vec [| 0.5; -0.5 |]) in
          finite_then_neg_inf
            (Rune.jit
               Nx.Ptree.(M.coords m @-> returns tensor)
               lp
               (holding m [| 0.5; 2. |])));
    ]

(* v *)

(* A two-device set of software devices, which compute on the host. *)
let two_devices =
  fixture (fun () ->
      Nx.devices
        (List.init 2 (fun k ->
             Result.get_ok (Rig.software (Printf.sprintf "norn-m%d" k)))))

(* Samples a distribution on [on] first, once [later] is set. *)
let straying_site (type e) (on : e Nx.Placement.t) later =
  M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
      if !later then
        ignore
          (M.sample
             (D.normal
                ~loc:(Nx.place on (f64 0.))
                ~scale:(Nx.place on (f64 1.))));
      let a = M.sample std and b = M.sample std in
      ({ a; b }, M.sample std))

(* Adds a factor on [on] first, once [later] is set. *)
let straying_factor (type e) (on : e Nx.Placement.t) later =
  M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
      if !later then M.factor (Nx.place on (f64 0.));
      let a = M.sample std and b = M.sample std in
      ({ a; b }, M.sample std))

exception Boom of int

let boom () = raise (Boom __LINE__)

let v_tests =
  group "v"
    [
      test "each site's log density has the model's dtype" (fun () ->
          let m =
            M.v Nx.float32 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let x = M.sample std in
                (x, M.sample (D.normal ~loc:x ~scale:(f64 1.))))
          in
          let lp = M.log_density m (f64 0.) (holding m [| 0.5 |]) in
          equal string "float32" (Nx.Dtype.name (Nx.dtype lp)));
      test "answers each site with its bijector's image of zero coordinates"
        (fun () ->
          (* Bij: exp 0 = 1, interval (2, 3) at 0 is 2 + sigmoid 0 = 2.5, and
             identity 0 = 0. *)
          let seen = ref [] in
          let _ =
            M.v Nx.float64
              Nx.Ptree.(pair (pair tensor tensor) tensor)
              Nx.Ptree.tensor
              (fun () ->
                let t = M.sample (D.half_cauchy ~scale:(f64 5.)) in
                let u = M.sample (D.uniform ~low:(f64 2.) ~high:(f64 3.)) in
                let x = M.sample std in
                seen := [ item t; item u; item x ];
                (((t, u), x), M.sample std))
          in
          equal (list float_exact) [ 1.; 2.5; 0. ] !seen);
      test "checks no domain" (fun () ->
          (* At zero coordinates the scale is -10: v builds the model, and its
             density there is -inf. *)
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let a = M.sample std in
                (a, M.sample (D.normal ~loc:(f64 0.) ~scale:(Nx.sub_s a 10.))))
          in
          let c =
            Nx.Ptree.map (M.coords m)
              (fun _ t -> Nx.reshape [| 1 |] t)
              (M.unconstrain m (f64 0.))
          in
          equal float_exact Float.neg_infinity
            (Nx.item [ 0 ] (M.log_density m (f64 0.) c)));
      cases "a failure of the model is raised again naming the sites learned"
        ~id:Printexc.to_string [ Failure "boom"; Invalid_argument "boom" ]
        (fun e ->
          let again =
            match e with
            | Failure _ ->
                Failure
                  "Norn_model.v: the model raised after 1 site (a normal of \
                   shape []): boom"
            | _ ->
                Invalid_argument
                  "Norn_model.v: the model raised after 1 site (a normal of \
                   shape []): boom"
          in
          raises again (fun () ->
              M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                  ignore (M.sample std);
                  raise e)));
      test "another exception propagates with its backtrace" (fun () ->
          match
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                ignore (M.sample std);
                boom ())
          with
          | _ -> fail "v returned"
          | exception Boom line ->
              let locations =
                Printexc.get_raw_backtrace ()
                |> Printexc.backtrace_slots
                |> Option.fold ~none:[] ~some:Array.to_list
                |> List.filter_map Printexc.Slot.location
                |> List.map (fun (l : Printexc.location) ->
                    (Filename.basename l.filename, l.line_number))
              in
              mem (pair string int) ("test_model.ml", line) locations);
      test "a sample missing from the result is refused" (fun () ->
          raises_match
            (Exn.invalid_arg
               ~sub:"the 3rd sample, a normal of shape [], is not in the result")
            (fun () ->
              M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
                  let a = M.sample std and b = M.sample std in
                  let _ = M.sample std in
                  ({ a; b }, M.sample std))));
      test "a value returned twice is refused naming the site" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Norn_model.v: site a appears twice")
            (fun () ->
              M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
                  let a = M.sample std in
                  ({ a; b = a }, M.sample std))));
      test "a discrete latent site is refused naming the site" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn_model.v: site 1:") (fun () ->
              M.v Nx.float64
                Nx.Ptree.(pair tensor tensor)
                Nx.Ptree.tensor
                (fun () ->
                  let a = M.sample std in
                  let b = M.sample (D.poisson ~rate:(f64 1.)) in
                  ((a, b), M.sample std))));
      test "a model that samples nothing is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn_model.v") (fun () ->
              M.v Nx.float64 Nx.Ptree.unit Nx.Ptree.unit (fun () -> ((), ()))));
      test "a model lies where its sites' distributions are" (fun () ->
          let module P = (val two_devices ()) in
          let elsewhere v = Nx.place P.on (f64 v) in
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.unit (fun () ->
                ( M.sample (D.normal ~loc:(elsewhere 0.) ~scale:(elsewhere 1.)),
                  () ))
          in
          (* A key on one device: the values lie where the model does. *)
          let one =
            Nx.Placement.device P.v (List.hd (Nx.Devices.to_list P.v))
          in
          let key = Nx.Rng.place one (Nx.Rng.key 0) in
          let pp p = Format.asprintf "%a" Nx.Placement.pp p in
          let x, () = M.simulate m key in
          equal ~msg:"simulate" string (pp P.on) (pp (Nx.placement x));
          let l = M.log_density m () (M.from_prior m () ~n:2 key) in
          equal ~msg:"log_density" string (pp P.on) (pp (Nx.placement l)));
      cases "every interpreter refuses a site on another set, naming it"
        ~id:Fun.id interpreters (fun name ->
          let module P = (val two_devices ()) in
          let m, c, p = changed (straying_site P.on) in
          (* The stray sample is the run's first, at site a's place. *)
          raises_match
            (Exn.invalid_arg
               ~sub:"site a: the distribution lies on another device set")
            (fun () -> interpret name m ~c ~p));
      cases "every interpreter refuses a factor on another set" ~id:Fun.id
        interpreters (fun name ->
          let module P = (val two_devices ()) in
          let m, c, p = changed (straying_factor P.on) in
          raises_match
            (Exn.invalid_arg ~sub:"a factor lies on another device set")
            (fun () -> interpret name m ~c ~p));
    ]

(* sample, factor *)

let sample_tests =
  group "sample"
    [
      test "outside an interpreter is unhandled" (fun () ->
          raises_match
            (function Effect.Unhandled _ -> true | _ -> false)
            (fun () -> M.sample std));
    ]

let factor_tests =
  group "factor"
    [
      test "outside an interpreter is unhandled" (fun () ->
          raises_match
            (function Effect.Unhandled _ -> true | _ -> false)
            (fun () -> M.factor (f64 0.)));
      cases "its axis 0 indexes its points, a scalar being one point"
        ~id:(fun (id, _, _) -> id)
        [
          ("a scalar", f64 3., [| 3. |]);
          ( "a [2; 3] matrix",
            Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |],
            [| 6.; 15. |] );
        ]
        (fun (_, l, points) ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let a = M.sample std in
                M.factor l;
                (a, M.sample (D.normal ~loc:a ~scale:(f64 1.))))
          in
          equal
            (array (float 1e-12))
            (Array.append points [| log_std 0. |])
            (floats (M.pointwise m (f64 0.) (f64 0.))));
    ]

(* coords, constrain, unconstrain *)

let coords_tests =
  group "coords"
    [
      test "a site has as many coordinates as degrees of freedom" (fun () ->
          let m =
            M.fix
              (fun p -> List.nth p 3)
              (vec [| 1.; 2. |])
              (M.v Nx.float64 (Nx.Ptree.list Nx.Ptree.tensor) Nx.Ptree.tensor
                 (fun () ->
                   let w =
                     M.sample
                       (D.dirichlet ~concentration:(vec [| 1.; 2.; 3. |]))
                   in
                   let v = M.sample (D.iid [| 8 |] std) in
                   let s = M.sample (D.sorted 2 std) in
                   let f = M.sample (D.iid [| 2 |] std) in
                   ([ w; v; s; f ], M.sample std)))
          in
          let shapes = List.map Nx.shape (one m :> Nx.host Nx.float64_t list) in
          (* A simplex of 3 has 2; 8 iid normals 8; 2 sorted normals 2; a fixed
             site none. *)
          equal (list (array int)) [ [| 2 |]; [| 8 |]; [| 2 |]; [| 0 |] ] shapes);
    ]

let constrain_tests =
  group "constrain"
    [
      test "coordinates of another shape are refused naming the site" (fun () ->
          let m = Lazy.force bounded in
          let c =
            Nx.Ptree.map (M.coords m) (fun _ t -> Nx.reshape [| 1 |] t) (one m)
          in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Norn_model.constrain: the coordinates do not match the \
                  model: at b") (fun () -> M.constrain m c));
    ]

(* Values: elementwise round trips lose a few ulps of each element, and
   noncentred's (θ - μ) / τ loses u (|θ| + |μ|) / τ, which stays below 1e-12
   relative for values of the prior's scale. *)
let values_close = float_rel ~rel:1e-11 ~abs:1e-11

(* Coordinates in (-5, 5): the inverse of each bijector loses at most u |value|
   / |∂value/∂coordinate|; the worst is interval's at 5, where sigmoid' =
   0.0066, and noncentred's at τ = e⁻⁵, both below 1e-12. *)
let coords_close = float_rel ~rel:1e-10 ~abs:1e-10

let unconstrain_tests =
  group "unconstrain"
    [
      prop "constrain undoes it" ~count:60
        ~examples:(List.map (fun m -> (m, 0)) laws)
        simulations
        (fun ((Model l as model), key) ->
          covers laws model;
          let m = Lazy.force l.model in
          Law.round_trip
            (Testable.contramap
               (fun p -> floats (vector l.latent p))
               (array values_close))
            pass (M.unconstrain m) (M.constrain m)
            (fst (M.simulate m (Nx.Rng.key key))));
      prop "it undoes constrain" ~count:60 ~examples:(at_origin laws ~chains:1)
        (positions laws ~chains:1 ~box:5.) (fun ((Model l as model), v) ->
          covers laws model;
          let m = Lazy.force l.model in
          let c = instance m (at m ~chains:1 (vec v)) in
          equal (array coords_close)
            (floats (vector (M.coords m) c))
            (floats (vector (M.coords m) (M.unconstrain m (M.constrain m c)))));
      test "values of another shape are refused naming the site" (fun () ->
          let m = Lazy.force centred in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Norn_model.unconstrain: the values do not match the model: \
                  at theta") (fun () ->
              M.unconstrain m
                {
                  mu = f64 0.;
                  tau = f64 1.;
                  theta = Nx.zeros Nx.float64 [| 7 |];
                }));
    ]

(* log_density *)

(* The models whose constrain has a square Jacobian. The simplex's K - 1
   coordinates give K values, and a fixed site has a value and no coordinates:
   their densities are Bij's and fix's promises. *)
let square =
  List.filter
    (fun m ->
      List.mem (name m) [ "centred"; "non-centred"; "affine"; "bounded" ])
    laws

let jacobian_law ((Model l as model), v) =
  covers square model;
  let m = Lazy.force l.model in
  let c = at m ~chains:1 (vec v) in
  let values v =
    vector l.latent (M.constrain m (instance m (at m ~chains:1 v)))
  in
  let _, logdet = Nx.slogdet Nx.float64 (Rune.jacfwd' values (vec v)) in
  let joint = item (M.log_joint m l.data (M.constrain m (instance m c))) in
  equal
    (agrees (scale m l.data c).(0))
    (joint +. item logdet)
    (Nx.item [ 0 ] (M.log_density m l.data c))

let rows_law ((Model l as model), v) =
  covers laws model;
  let m = Lazy.force l.model in
  let c = at m ~chains:3 (vec v) in
  let alone i = Nx.item [ 0 ] (M.log_density m l.data (chain m c i)) in
  densities_agree ~scale:(scale m l.data c)
    (floats (M.log_density m l.data c))
    (Array.init 3 alone)

(* A scale vector whose middle element is the latent [a]. *)
let scaled () =
  M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
      let a = M.sample std in
      let scale =
        Nx.concatenate ~axis:0
          [ vec [| 1. |]; Nx.reshape [| 1 |] a; vec [| 1. |] ]
      in
      let b = M.sample (D.normal ~loc:(f64 0.) ~scale) in
      ({ a; b }, M.sample std))

(* A Poisson site whose rate is [exp f]: at [f = 1000] the rate overflows. *)
let counts () =
  M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
      let f = M.sample std in
      (f, M.sample (D.poisson ~rate:(Nx.exp f))))

let log_density_tests =
  group "log_density"
    [
      prop "is the values' joint density times the Jacobian" ~count:40
        ~examples:(at_origin square ~chains:1)
        (positions square ~chains:1 ~box:3.)
        jacobian_law;
      prop "a chain's row is the density of a position of that chain alone"
        ~examples:(saturating laws)
        (positions laws ~chains:3 ~box:5.)
        rows_law;
      test "data of the wrong shape are refused at once" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.log_density: the observations do not match the \
                model: at y, data of shape [7], the site's shape [8]")
            (fun () ->
              M.log_density (Lazy.force centred) (Nx.zeros Nx.float64 [| 7 |])));
      test "data wrong at two sites are refused naming the first performed"
        (fun () ->
          raises
            (Invalid_argument
               "Norn_model.log_density: the observations do not match the \
                model: at 0, data of shape [2], the site's shape [3]")
            (fun () ->
              M.log_density (Lazy.force bounded)
                (Nx.zeros Nx.int32 [| 2 |], Nx.zeros Nx.float64 [| 3 |])));
      test "a position of another shape is refused naming the site" (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let x = M.sample (D.iid [| 3 |] std) in
                (x, M.sample (D.normal ~loc:(Nx.sum x) ~scale:(f64 1.))))
          in
          let c =
            Nx.Ptree.map (M.coords m)
              (fun _ t -> Nx.slice [ Nx.R (0, 2); Nx.R (0, 2) ] t)
              (at m ~chains:2 (Nx.zeros Nx.float64 [| 6 |]))
          in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Norn_model.log_density: the position does not match the \
                  model: at p") (fun () -> M.log_density m (f64 0.) c));
      test "a parameter outside its domain makes the term -inf" (fun () ->
          (* Chain 1 has a = -2, a negative scale. *)
          let m = scaled () in
          let c = at m ~chains:2 (vec [| 0.5; -2.; 0.; 0.; 0.; 0.; 0.; 0. |]) in
          finite_then_neg_inf (M.log_density m (f64 0.) c));
      test "an overflowed exponential rate makes the term -inf" (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let f = M.sample std in
                (f, M.sample (D.exponential ~rate:(Nx.exp f))))
          in
          finite_then_neg_inf
            (M.log_density m (f64 0.5) (holding m [| 0.; 1000. |])));
      test "an overflowed Poisson rate makes the term -inf" (fun () ->
          let m = counts () in
          finite_then_neg_inf
            (M.log_density m (Nx.scalar Nx.int32 3l)
               (holding m [| 0.; 1000. |])));
      test "a Cholesky factor holding NaN makes the term -inf" (fun () ->
          let m = correlated () in
          finite_then_neg_inf
            (M.log_density m (vec [| 0.5; -0.5 |]) (holding m [| 0.5; 2. |])));
      test "a NaN term with parameters in their domain raises naming the site"
        (fun () ->
          let m = Lazy.force centred in
          let y =
            Nx.concatenate ~axis:0
              [ vec [| Float.nan |]; Nx.slice [ Nx.R (1, 8) ] y_schools ]
          in
          raises_match (Exn.invalid_arg ~sub:"Norn_model.log_density: site y:")
            (fun () ->
              M.log_density m y (at m ~chains:1 (Nx.zeros Nx.float64 [| 10 |]))));
      test "an overflowed trajectory is a divergence" (fun () ->
          let m = counts () in
          let lp = M.log_density m (Nx.scalar Nx.int32 3l) in
          let u = M.coords m in
          (* A geometry a thousand wide sends the first step past the rate's
             overflow. *)
          let geometry =
            Norn.Gaussian.diagonal u Nx.float64
              ~mean:(instance m (holding m [| 0. |]))
              ~scale:(instance m (holding m [| 1e3 |]))
          in
          let s =
            Norn.Nuts.init u ~geometry lp (holding m [| 0.; 0.; 0.; 0. |])
          in
          let s = Norn.Nuts.step u lp (Nx.Rng.key 0) s in
          equal (array bool)
            [| true; true; true; true |]
            (Nx.to_array s.stats.diverging));
    ]

(* log_prior, log_likelihood *)

(* An observed value [y] read by the latent site [a]: y ~ N(0, 1), a ~ N(y,
   1). *)
let downstream =
  lazy
    (M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
         let y = M.sample std in
         (M.sample (D.normal ~loc:y ~scale:(f64 1.)), y)))

let log_prior_tests =
  group "log_prior"
    [
      test "answers the observed sites with the data" (fun () ->
          (* At y = 1, a = 0: log N(0; 1, 1). *)
          let m = Lazy.force downstream in
          equal (float 1e-12) (log_std 1.)
            (Nx.item [ 0 ] (M.log_prior m (f64 1.) (holding m [| 0. |]))));
    ]

let log_likelihood_tests =
  group "log_likelihood"
    [
      prop "with log_prior adds up to log_density" ~examples:(saturating laws)
        (positions laws ~chains:3 ~box:5.) (fun ((Model l as model), v) ->
          covers laws model;
          let m = Lazy.force l.model in
          let c = at m ~chains:3 (vec v) in
          densities_agree ~scale:(scale m l.data c)
            (floats (M.log_density m l.data c))
            (floats
               (Nx.add (M.log_prior m l.data c) (M.log_likelihood m l.data c))));
      test
        "with log_prior adds up to log_density where a latent site reads data"
        (fun () ->
          (* At y = 1, a = 0: log N(1; 0, 1) + log N(0; 1, 1). *)
          let m = Lazy.force downstream and y = f64 1. in
          let c = holding m [| 0. |] in
          let sum = Nx.add (M.log_prior m y c) (M.log_likelihood m y c) in
          equal (float 1e-12) (2. *. log_std 1.) (Nx.item [ 0 ] sum);
          equal (float 1e-12)
            (Nx.item [ 0 ] (M.log_density m y c))
            (Nx.item [ 0 ] sum));
      test "raises as log_density does" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.log_likelihood: the observations do not match the \
                model: at y, data of shape [7], the site's shape [8]")
            (fun () ->
              M.log_likelihood (Lazy.force centred)
                (Nx.zeros Nx.float64 [| 7 |])));
    ]

(* from_prior *)

(* [by_chain u c] is [c]'s coordinates as a matrix, one row per chain. *)
let by_chain u c =
  let rows t = Nx.reshape [| (Nx.shape t).(0); -1 |] (Nx.cast Nx.float64 t) in
  Nx.concatenate ~axis:1
    (List.rev (Nx.Ptree.fold u (fun _ t acc -> rows t :: acc) c []))

let from_prior_tests =
  group "from_prior"
    [
      cases "refuses n < 1" ~id:string_of_int [ 0; -1 ] (fun n ->
          raises_match (Exn.invalid_arg ~sub:"Norn_model.from_prior") (fun () ->
              M.from_prior (Lazy.force centred) y_schools ~n (Nx.Rng.key 0)));
      prop "a draw does not depend on n"
        Gen.(triple (int_range 0 999) (int_range 1 3) (int_range 1 3))
        (fun (key, n, more) ->
          let m = Lazy.force centred and k = Nx.Rng.key key in
          let draws n = by_chain (M.coords m) (M.from_prior m y_schools ~n k) in
          equal (array float_exact)
            (floats (draws n))
            (floats (Nx.slice [ Nx.R (0, n) ] (draws (n + more)))));
      prop "draw i is the coordinates of simulate's values from row i's key"
        Gen.(int_range 0 999)
        (fun key ->
          let m = Lazy.force centred and k = Nx.Rng.key key in
          let simulated k_i =
            vector (M.coords m) (M.unconstrain m (fst (M.simulate m k_i)))
          in
          equal (array float_exact)
            (floats
               (Nx.stack
                  (Array.to_list (Array.map simulated (Nx.Rng.split ~n:3 k)))))
            (floats (by_chain (M.coords m) (M.from_prior m y_schools ~n:3 k))));
      test "answers the observed sites with the data" (fun () ->
          (* Draw i of a is drawn from N(y, 1) with fold_in k_i 1, k_i row i of
             split_batch ~n:3 k; y = 1 answers site 0. *)
          let m = Lazy.force downstream and y = f64 1. and k = Nx.Rng.key 4 in
          let expected =
            Array.map
              (fun k_i ->
                item
                  (D.sample (Nx.Rng.fold_in k_i 1)
                     (D.normal ~loc:y ~scale:(f64 1.))))
              (Nx.Rng.split ~n:3 k)
          in
          equal (array float_exact) expected
            (floats (M.from_prior m y ~n:3 k :> Nx.host Nx.float64_t)));
    ]

(* Init, init *)

(* [p_uniform ranks ~draws] is the smallest of the elements' uniformity p-values
   of [ranks], rows of ranks in 0 .. draws. *)
let p_uniform ranks ~draws =
  let rows =
    List.init (Nx.shape ranks).(0) (fun i -> Nx.slice [ Nx.I i ] ranks)
  in
  item (Nx.min (Norn.Diag.rank_uniformity Nx.Ptree.tensor ~draws rows))

(* [ranked u] is the rank among 0 .. 19 of values [u] uniform on (0, 1): uniform
   on 0 .. 19 where [u] is. *)
let ranked u = Nx.floor (Nx.mul_s u 20.)

(* Draws whose probability-integral transform is uniform give uniform ranks.
   [Diag.rank_uniformity]'s p-value is below a level with at most that
   probability under uniformity: below 1e-7 on each of at most ten elements
   fails a correct start at most 1e-6 of the time. *)
let uniform_level = 1e-7

(* A latent normal and an observation finite everywhere: every candidate is
   finite, so a start is its first candidate. *)
let everywhere_finite () =
  M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
      let mu = M.sample std in
      (mu, M.sample (D.normal ~loc:mu ~scale:(f64 1.))))

let init_strategies =
  group "Init"
    [
      prop "uniform draws every coordinate in (-2, 2)"
        Gen.(pair (int_range 0 999) (int_range 1 8))
        (fun (key, chains) ->
          let m = Lazy.force centred in
          let c = M.init m y_schools ~chains (Nx.Rng.key key) in
          satisfies ~claim:"every coordinate in (-2, 2)" (array float_exact)
            (Array.for_all (fun x -> x > -2. && x < 2.))
            (floats (vector (M.coords m) c)));
      test "uniform starts are uniform" (fun () ->
          let m = Lazy.force centred in
          let c = M.init m y_schools ~chains:1000 (Nx.Rng.key 6) in
          let u = Nx.div_s (Nx.add_s (by_chain (M.coords m) c) 2.) 4. in
          at_least float_exact ~than:uniform_level
            (p_uniform (ranked u) ~draws:19));
      test "prior starts follow the prior" (fun () ->
          let m = everywhere_finite () in
          let c =
            M.init ~from:M.Init.prior m (f64 0.3) ~chains:1000 (Nx.Rng.key 7)
          in
          let u = Nx.ndtr (by_chain (M.coords m) c) in
          at_least float_exact ~than:uniform_level
            (p_uniform (ranked u) ~draws:19));
      test "near starts lie within 0.1 of the values' coordinates" (fun () ->
          let m = Lazy.force centred in
          let p, _ = M.simulate m (Nx.Rng.key 2) in
          let c =
            M.init ~from:(M.Init.near p) m y_schools ~chains:4 (Nx.Rng.key 4)
          in
          let target = vector (M.coords m) (M.unconstrain m p) in
          less float_exact ~than:0.1
            (item (Nx.max (Nx.abs (Nx.sub (by_chain (M.coords m) c) target)))));
    ]

(* [x] latent, an observation uniform on (x, x + 1) at 0.5: finite for x in
   (-0.5, 0.5), a quarter of the uniform candidates. *)
let narrow () =
  M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
      let x = M.sample std in
      (x, M.sample (D.uniform ~low:x ~high:(Nx.add_s x 1.))))

let init_tests =
  group "init"
    [
      test "every chain starts where the density is finite" (fun () ->
          let m = narrow () and y = f64 0.5 in
          let c = M.init m y ~chains:8 (Nx.Rng.key 0) in
          satisfies ~claim:"finite" (array float_exact)
            (Array.for_all Float.is_finite)
            (floats (M.log_density m y c)));
      prop "a chain's start does not depend on the chain count"
        Gen.(triple (int_range 0 999) (int_range 1 3) (int_range 1 3))
        (fun (key, chains, more) ->
          let m = Lazy.force centred and k = Nx.Rng.key key in
          let starts chains =
            by_chain (M.coords m) (M.init m y_schools ~chains k)
          in
          equal (array float_exact)
            (floats (starts chains))
            (floats (Nx.slice [ Nx.R (0, chains) ] (starts (chains + more)))));
      test "starts from uniform by default" (fun () ->
          let m = Lazy.force centred and k = Nx.Rng.key 3 in
          let u = M.coords m in
          equal (array float_exact)
            (floats
               (by_chain u
                  (M.init ~from:M.Init.uniform m y_schools ~chains:3 k)))
            (floats (by_chain u (M.init m y_schools ~chains:3 k))));
      cases "refuses chains < 1" ~id:string_of_int [ 0; -1 ] (fun chains ->
          raises_match (Exn.invalid_arg ~sub:"Norn_model.init") (fun () ->
              M.init (Lazy.force centred) y_schools ~chains (Nx.Rng.key 0)));
      test "no finite candidate raises naming the site and the terms" (fun () ->
          let m =
            M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
                let a = M.sample std and b = M.sample std in
                ({ a; b }, M.sample (D.uniform ~low:(f64 0.) ~high:(f64 1.))))
          in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Norn_model.init: no finite log density in 100 candidates; at \
                  the last, site y has log density -inf: it is outside the \
                  support of uniform, (0, 1); the terms are a ") (fun () ->
              M.init m (f64 5.) ~chains:2 (Nx.Rng.key 0)));
      test "two sites outside their support name the first performed" (fun () ->
          let unit = D.uniform ~low:(f64 0.) ~high:(f64 1.) in
          let m =
            M.v Nx.float64 Nx.Ptree.tensor
              Nx.Ptree.(pair tensor tensor)
              (fun () ->
                let a = M.sample std in
                let w = M.sample unit in
                let u = M.sample unit in
                (a, (u, w)))
          in
          raises_match
            (Exn.invalid_arg ~sub:"at the last, site 1 has log density -inf")
            (fun () -> M.init m (f64 5., f64 5.) ~chains:1 (Nx.Rng.key 0)));
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
               ~sub:
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
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Norn_model.init: no finite log density in 100 candidates; at \
                  the last, site p has log density -inf: its coordinates \
                  saturate: the log-determinant is -inf; the terms are ")
            (fun () ->
              M.init
                ~from:(M.Init.near (f64 1e-320))
                m (f64 0.) ~chains:1 (Nx.Rng.key 0)));
      test "a compiled start names the support from its arguments" (fun () ->
          Compiled.here ();
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
                (M.init (m low high) y ~chains:1 k :> Nx.host Nx.float64_t))
          in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "site y has log density -inf: it is outside the support of \
                  uniform, (2, 3)") (fun () ->
              start (Nx.Rng.key 0) (f64 2.) (f64 3.) (f64 5.)));
      test "a support bounded above is named as one" (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let x = M.sample std in
                let below = Norn.Bij.less ~high:(f64 2.) in
                ( x,
                  M.sample (D.transform below (D.normal ~loc:x ~scale:(f64 1.)))
                ))
          in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "site y has log density -inf: it is outside the support of \
                  transform, (-inf, 2)") (fun () ->
              M.init m (f64 5.) ~chains:1 (Nx.Rng.key 0)));
      test "a parameter outside its domain everywhere is named with the site"
        (fun () ->
          let m =
            M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
                let a = M.sample std in
                let b =
                  M.sample
                    (D.normal ~loc:(f64 0.)
                       ~scale:(Nx.sub_s (Nx.neg (Nx.abs a)) 1.))
                in
                ({ a; b }, M.sample std))
          in
          raises_match
            (Exn.invalid_arg
               ~sub:
                 "Norn_model.init: no finite log density in 100 candidates; at \
                  the last, site b: normal: scale is -") (fun () ->
              M.init m (f64 0.) ~chains:2 (Nx.Rng.key 0)));
    ]

(* log_joint *)

let schools_joint y p =
  let terms =
    [
      D.log_density (D.normal ~loc:(f64 0.) ~scale:(f64 5.)) p.mu;
      D.log_density (D.half_cauchy ~scale:(f64 5.)) p.tau;
      D.log_density (D.iid [| 8 |] (D.normal ~loc:p.mu ~scale:p.tau)) p.theta;
      D.log_density (D.normal ~loc:p.theta ~scale:sigma) y;
    ]
  in
  ( List.fold_left (fun a t -> a +. item t) 0. terms,
    List.fold_left (fun a t -> a +. Float.abs (item t)) 0. terms )

let log_joint_tests =
  group "log_joint"
    [
      prop "is the sum of each site's log density"
        Gen.(int_range 0 999)
        (fun key ->
          let m = Lazy.force centred in
          let p, y = M.simulate m (Nx.Rng.key key) in
          let expected, s = schools_joint y p in
          equal (agrees s) expected (item (M.log_joint m y p)));
      test "adds the factors to each site's log density" (fun () ->
          let m = Lazy.force bounded and counts, z = bounded_data in
          let p, _ = M.simulate m (Nx.Rng.key 5) in
          let terms =
            [
              D.log_density (D.truncated ~high:(f64 2.) std) p.b;
              D.log_density (D.uniform ~low:(f64 (-1.)) ~high:(f64 1.)) p.a;
              D.log_density
                (D.iid [| 3 |] (D.poisson ~rate:(Nx.exp p.a)))
                counts;
              Nx.neg (Nx.add (Nx.square p.b) (Nx.square p.a));
              D.log_density
                (D.iid [| 2 |] (D.normal ~loc:p.b ~scale:(f64 1.)))
                z;
            ]
          in
          let s =
            List.fold_left (fun a t -> a +. Float.abs (item t)) 0. terms
          in
          equal (agrees s)
            (List.fold_left (fun a t -> a +. item t) 0. terms)
            (item (M.log_joint m bounded_data p)));
      test "is a mixture's log density over trials" (fun () ->
          let trials = Nx.create Nx.int32 [| 2 |] [| 3l; 7l |] in
          let mix l =
            D.mixture
              ~logits:(vec [| 0.; 0. |])
              (D.binomial ~trials ~logits:(Nx.broadcast_to [| 2 |] l))
          in
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let l = M.sample std in
                (l, M.sample (mix l)))
          in
          let l = f64 0.3 and y = Nx.scalar Nx.int32 5l in
          equal (float 1e-12)
            (item (D.log_density std l) +. item (D.log_density (mix l) y))
            (item (M.log_joint m y l)));
      test "refuses a parameter outside its domain naming the site" (fun () ->
          raises
            (Invalid_argument
               "Norn_model.log_joint: site b: normal: scale at [1] is -2, not \
                in (0, inf)") (fun () ->
              M.log_joint (scaled ()) (f64 0.)
                { a = f64 (-2.); b = vec [| 0.; 0.; 0. |] }));
      test "after a factor of -inf takes a bad parameter as -inf" (fun () ->
          let m =
            M.v Nx.float64 two Nx.Ptree.tensor (fun () ->
                let a = M.sample std in
                M.factor (f64 Float.neg_infinity);
                let b =
                  M.sample
                    (D.normal ~loc:(f64 0.) ~scale:(Nx.sub_s (Nx.abs a) 10.))
                in
                ({ a; b }, M.sample std))
          in
          equal float_exact Float.neg_infinity
            (item (M.log_joint m (f64 0.) { a = f64 1.; b = f64 0. })));
    ]

(* pointwise *)

(* A regression on covariates [x]: y ~ N(beta x, 1). *)
let regression x =
  M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
      let beta = M.sample std in
      (beta, M.sample (D.normal ~loc:(Nx.mul beta x) ~scale:(f64 1.))))

let pointwise_tests =
  group "pointwise"
    [
      prop "sums to the likelihood" ~examples:(at_origin laws ~chains:1)
        (positions laws ~chains:1 ~box:5.) (fun ((Model l as model), v) ->
          covers laws model;
          let m = Lazy.force l.model in
          let c = at m ~chains:1 (vec v) in
          let pw = M.pointwise m l.data (M.constrain m (instance m c)) in
          let s = (scale m l.data c).(0) in
          equal (agrees s)
            (Nx.item [ 0 ] (M.log_likelihood m l.data c))
            (item (Nx.sum pw)));
      test "gives each point in the order the sites and factors are performed"
        (fun () ->
          (* counts' three points, the factor's two, z's two. *)
          let m = Lazy.force bounded and counts, z = bounded_data in
          let p, _ = M.simulate m (Nx.Rng.key 5) in
          let expected =
            Nx.concatenate ~axis:0
              [
                D.factors (D.iid [| 3 |] (D.poisson ~rate:(Nx.exp p.a))) counts;
                Nx.stack [ Nx.neg (Nx.square p.b); Nx.neg (Nx.square p.a) ];
                D.factors (D.iid [| 2 |] (D.normal ~loc:p.b ~scale:(f64 1.))) z;
              ]
          in
          equal
            (array (float 1e-12))
            (floats expected)
            (floats (M.pointwise m bounded_data p)));
      test "over draws has a point axis after the draws" (fun () ->
          (* Every point is log N(0; 0, 1) at mu = 0 and y = 0. *)
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let mu = M.sample std in
                (mu, M.sample (D.iid [| 5 |] (D.normal ~loc:mu ~scale:(f64 1.)))))
          in
          let draws =
            Norn.Draws.v Nx.Ptree.tensor (Nx.zeros Nx.float64 [| 2; 3 |])
          in
          let ll =
            Norn.Draws.map Nx.Ptree.tensor Nx.Ptree.tensor
              (fun mu -> M.pointwise m (Nx.zeros Nx.float64 [| 5 |]) mu)
              draws
          in
          let ll = (ll :> Nx.host Nx.float64_t) in
          equal (array int) [| 2; 3; 5 |] (Nx.shape ll);
          equal (array (float 1e-12)) (Array.make 30 (log_std 0.)) (floats ll));
      test "reads values, so a model on new covariates takes them" (fun () ->
          let beta, _ =
            M.simulate (regression (vec [| 1.; 2. |])) (Nx.Rng.key 1)
          in
          let x = vec [| 0.5; -1.; 3. |] and y = vec [| 0.2; 0.1; -0.4 |] in
          equal
            (array (float 1e-12))
            (floats
               (D.factors (D.normal ~loc:(Nx.mul beta x) ~scale:(f64 1.)) y))
            (floats (M.pointwise (regression x) y beta)));
      test "refuses an observed site's parameter outside its domain" (fun () ->
          let m =
            M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
                let a = M.sample std in
                (a, M.sample (D.normal ~loc:(f64 0.) ~scale:(Nx.sub_s a 10.))))
          in
          raises_match
            (Exn.invalid_arg ~sub:"Norn_model.pointwise: site y: normal: scale")
            (fun () -> M.pointwise m (f64 0.) (f64 0.)));
    ]

(* simulate, predict *)

(* A latent normal and its observation, with a factor [l] between them. *)
let with_factor l =
  M.v Nx.float64 Nx.Ptree.tensor Nx.Ptree.tensor (fun () ->
      let a = M.sample std in
      M.factor l;
      (a, M.sample (D.normal ~loc:a ~scale:(f64 1.))))

let simulate_tests =
  group "simulate"
    [
      prop "site i draws from fold_in k i"
        Gen.(int_range 0 999)
        (fun key ->
          let k = Nx.Rng.key key in
          let p, y = M.simulate (Lazy.force centred) k in
          let draw i d = D.sample (Nx.Rng.fold_in k i) d in
          let mu = draw 0 (D.normal ~loc:(f64 0.) ~scale:(f64 5.)) in
          let tau = draw 1 (D.half_cauchy ~scale:(f64 5.)) in
          let theta = draw 2 (D.iid [| 8 |] (D.normal ~loc:mu ~scale:tau)) in
          let y' = draw 3 (D.normal ~loc:theta ~scale:sigma) in
          equal (array float_exact)
            ( floats (vector schools { mu; tau; theta }) |> fun a ->
              Array.append a (floats y') )
            (Array.append (floats (vector schools p)) (floats y)));
      prop "ignores factors"
        Gen.(int_range 0 999)
        (fun key ->
          let k = Nx.Rng.key key in
          let without, y = M.simulate (with_factor (vec [||])) k in
          let with_nan, y' = M.simulate (with_factor (f64 Float.nan)) k in
          equal (array float_exact)
            [| item without; item y |]
            [| item with_nan; item y' |]);
    ]

let predict_tests =
  group "predict"
    [
      prop "from simulate's values is simulate's data" ~count:60
        ~examples:(List.map (fun m -> (m, 0)) laws)
        simulations
        (fun ((Model l as model), key) ->
          covers laws model;
          let m = Lazy.force l.model and k = Nx.Rng.key key in
          let p, y = M.simulate m k in
          equal (array float_exact)
            (floats (vector l.observed y))
            (floats (vector l.observed (M.predict m k p))));
    ]

(* reparam, noncentre, fix *)

let reparam_tests =
  group "reparam"
    [
      prop "keeps the density of the values"
        Gen.(int_range 0 999)
        (fun key ->
          let p, y = M.simulate (Lazy.force centred) (Nx.Rng.key key) in
          let joint m = item (M.log_joint (Lazy.force m) y p) in
          equal (array float_exact)
            [| joint centred; joint centred |]
            [| joint noncentred; joint affine |]);
      test "refuses a selector that computes" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn_model.reparam") (fun () ->
              M.reparam
                (fun p -> Nx.exp p.theta)
                (fun _ -> Norn.Bij.identity)
                (Lazy.force centred)));
    ]

let noncentre_tests =
  group "noncentre"
    [
      test "coordinates are the standardised values" (fun () ->
          let p, _ = M.simulate (Lazy.force centred) (Nx.Rng.key 2) in
          let c =
            (M.unconstrain (Lazy.force noncentred) p
              :> Nx.host Nx.float64_t schools)
          in
          let z = Nx.div (Nx.sub p.theta p.mu) p.tau in
          equal (array values_close) (floats z) (floats c.theta));
    ]

let fix_tests =
  group "fix"
    [
      test "a fixed site has coordinates of shape [0]" (fun () ->
          let m = Lazy.force fixed in
          equal (array int) [| 0 |]
            (Nx.shape (one m :> Nx.host Nx.float64_t schools).tau));
      test "a fixed site keeps its value and adds no density term" (fun () ->
          let m = Lazy.force fixed in
          let c = at m ~chains:1 (Nx.zeros Nx.float64 [| 9 |]) in
          let p = M.constrain m (instance m c) in
          equal ~msg:"value" float_exact 2. (item p.tau);
          let tau_term =
            item (D.log_density (D.half_cauchy ~scale:(f64 5.)) (f64 2.))
          in
          let with_tau = item (M.log_joint (Lazy.force centred) y_schools p) in
          equal
            (agrees (Float.abs with_tau))
            (with_tau -. tau_term)
            (Nx.item [ 0 ] (M.log_density m y_schools c)));
      test "refuses a value of another shape" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn_model.fix: site tau")
            (fun () ->
              M.fix
                (fun p -> p.tau)
                (Nx.zeros Nx.float64 [| 3 |])
                (Lazy.force centred)));
      test "refuses a selector that computes" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn_model.fix") (fun () ->
              M.fix (fun p -> Nx.exp p.tau) (f64 2.) (Lazy.force centred)));
    ]

(* terms, pp *)

let terms_tests =
  group "terms"
    [
      prop "sum to log_density" ~examples:(saturating laws)
        (positions laws ~chains:3 ~box:5.) (fun ((Model l as model), v) ->
          covers laws model;
          let m = Lazy.force l.model in
          let c = at m ~chains:3 (vec v) in
          let sites, factors = M.terms m l.data c in
          let sum = List.fold_left (fun a (_, t) -> Nx.add a t) factors sites in
          densities_agree ~scale:(scale m l.data c)
            (floats (M.log_density m l.data c))
            (floats sum));
      test "name each site in the order performed, then the factors' sum"
        (fun () ->
          let m = Lazy.force bounded in
          let c = at m ~chains:2 (vec [| 0.3; -0.4; 0.5; 1.2 |]) in
          let p = M.constrain m (instance m c) in
          let sites, factors = M.terms m bounded_data c in
          equal (list string) [ "b"; "a"; "0"; "1" ] (List.map fst sites);
          equal (array int) [| 2 |] (Nx.shape factors);
          equal (float 1e-12)
            (-.(item (Nx.square p.b) +. item (Nx.square p.a)))
            (Nx.item [ 0 ] factors));
      test "a fixed site's term is zero" (fun () ->
          let m = Lazy.force fixed in
          let sites, _ =
            M.terms m y_schools (at m ~chains:2 (Nx.zeros Nx.float64 [| 18 |]))
          in
          equal (array float_exact) [| 0.; 0. |]
            (floats (List.assoc "tau" sites)));
    ]

let pp_tests =
  group "pp"
    [
      test "the site table" (fun () ->
          expect
            (Format.asprintf "%a@.%a" M.pp (Lazy.force fixed) M.pp
               (Lazy.force bounded))
          @@ __POS_OF__
               {|
            site   role      family       shape  points  support      coordinates
            mu     latent    normal       []             (-inf, inf)  identity
            tau    fixed     half_cauchy  []             (0, inf)
            theta  latent    normal       [8]            (-inf, inf)  identity
            y      observed  normal       [8]    8       (-inf, inf)
            site  role      family     shape  points  support         coordinates
            b     latent    truncated  []             (-inf, 2)       less
            a     latent    uniform    []             (-1, 1)         interval
            0     observed  poisson    [3]    3       {0, 1, 2, ...}
            1     observed  normal     [2]    2       (-inf, inf)
            |});
    ]

let () =
  exit
    (run "Norn_model"
       [
         result_rule;
         static_models;
         interpreters_under_rune;
         v_tests;
         sample_tests;
         factor_tests;
         coords_tests;
         constrain_tests;
         unconstrain_tests;
         log_density_tests;
         log_prior_tests;
         log_likelihood_tests;
         from_prior_tests;
         init_strategies;
         init_tests;
         log_joint_tests;
         pointwise_tests;
         simulate_tests;
         predict_tests;
         reparam_tests;
         noncentre_tests;
         fix_tests;
         terms_tests;
         pp_tests;
       ])
