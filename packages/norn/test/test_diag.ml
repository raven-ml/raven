(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Diag = Norn.Diag
module Draws = Norn.Draws

let t = Nx.Ptree.tensor
let item x = Nx.item [] x
let f64 = Nx.scalar Nx.float64
let chains c n xs = Draws.v t (Nx.create Nx.float64 [| c; n |] xs)

(* Within rounding of the two implementations' summation orders. *)
let close = float_rel ~rel:1e-9 ~abs:1e-12

let nan_or_close =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b) || Testable.equal close a b)

(* [transitions ~energy ~diverging] is the statistics of transitions with these
   energies and divergences, the other fields zero. *)
let transitions ~energy ~diverging =
  let z = Nx.zeros_like energy in
  Draws.v
    (Norn.Stats.ptree Nx.float64)
    Norn.Stats.
      {
        lp = z;
        acceptance = z;
        step_size = z;
        n_steps = Nx.zeros Nx.int32 (Nx.shape energy);
        diverging;
        saturated = Nx.zeros Nx.bool (Nx.shape energy);
        energy;
      }

(* Goldens: chains and their diagnostics as ArviZ computes them, written by
   gen/diag.py. A case is [case NAME CHAINS DRAWS SUPERCHAINS], then [data] and
   the draws, then a line per diagnostic. *)

type golden = {
  name : string;
  chains : int;
  draws : int;
  superchains : int;
  data : float array;
  values : (string * float array) list;
}

let read_goldens () =
  let ic = open_in "golden/diag.golden" in
  let rec lines acc =
    match input_line ic with
    | l -> lines (if String.length l > 0 && l.[0] = '#' then acc else l :: acc)
    | exception End_of_file ->
        close_in ic;
        List.rev acc
  in
  let floats ws = Array.of_list (List.map float_of_string ws) in
  let rec cases acc = function
    | [] -> List.rev acc
    | l :: rest -> (
        match String.split_on_char ' ' l with
        | [ "case"; name; c; n; k ] ->
            let rec body values data = function
              | l :: rest when not (String.starts_with ~prefix:"case " l) -> (
                  match String.split_on_char ' ' l with
                  | "data" :: ws -> body values (floats ws) rest
                  | key :: ws -> body ((key, floats ws) :: values) data rest
                  | [] -> body values data rest)
              | rest -> (values, data, rest)
            in
            let values, data, rest = body [] [||] rest in
            let g =
              {
                name;
                chains = int_of_string c;
                draws = int_of_string n;
                superchains = int_of_string k;
                data;
                values;
              }
            in
            cases (g :: acc) rest
        | _ -> failwith ("diag.golden: unexpected line " ^ l))
  in
  cases [] (lines [])

let goldens = lazy (read_goldens ())
let draws_of g = chains g.chains g.draws g.data

(* [against_arviz key f] compares [f] of each golden's draws with ArviZ's [key],
   each value named by its case. *)
let against_arviz ?(only = fun _ -> true) key f =
  let gs = List.filter only (Lazy.force goldens) in
  equal
    (list (pair string (array nan_or_close)))
    (List.map (fun g -> (g.name, List.assoc key g.values)) gs)
    (List.map (fun g -> (g.name, Nx.to_array (f g))) gs)

(* Chains on a grid of 1/8, which [exp] and the affine maps below keep distinct,
   ties included, with a map [a x + b]. *)
let chains_gen =
  Gen.(
    let* c, n = pair (int_range 1 4) (int_range 4 40) in
    let+ xs =
      array
        ~size:(constant (c * n))
        (map (fun k -> float_of_int k /. 8.) (int_range (-40) 40))
    and+ a = of_list [ -3.; -0.5; 0.5; 2. ]
    and+ b = map float_of_int (int_range (-4) 4) in
    (c, n, xs, a, b))
  |> Gen.with_pp (fun ppf (c, n, xs, a, b) ->
      Format.fprintf ppf "%d chains of %d, map %g x + %g: [%s]" c n a b
        (String.concat "; "
           (Array.to_list (Array.map (Printf.sprintf "%h") xs))))

(* [mapped case] is the case's chains, and their image under its map. *)
let mapped (c, n, xs, a, b) =
  cover "one chain" (c = 1);
  cover "a decreasing map" (a < 0.);
  (chains c n xs, chains c n (Array.map (fun x -> (a *. x) +. b) xs))

(* Elements *)

let no_diagnostic =
  cases
    "has no diagnostic where the draws are constant, not all finite, or of \
     chains of fewer than 4"
    ~id:fst
    [
      ("constant", (4, 10, Array.make 40 1.));
      ( "a NaN draw",
        ( 2,
          5,
          Array.init 10 (fun i -> if i = 3 then Float.nan else float_of_int i)
        ) );
      ( "an infinite draw",
        ( 2,
          5,
          Array.init 10 (fun i ->
              if i = 3 then Float.infinity else float_of_int i) ) );
      ("chains of 3", (2, 3, Array.init 6 float_of_int));
    ]
    (fun (_, (c, n, xs)) ->
      let d = chains c n xs in
      List.iter
        (fun f ->
          satisfies ~claim:"nan" float_exact Float.is_nan (item (f t d)))
        [ Diag.rhat; Diag.ess_bulk; Diag.ess_tail; Diag.mcse_mean ])

let elements =
  group "elements"
    [
      test "each element of each tensor has its own diagnostic" (fun () ->
          let x = Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float64 [| 4; 50 |] in
          let y = Nx.stack ~axis:2 [ Nx.neg x; Nx.mul_s x 2. ] in
          let two = Nx.Ptree.pair t t in
          let rx, ry = Diag.rhat two (Draws.v two (x, y)) in
          equal (array close) [| item rx; item rx |] (Nx.to_array ry));
      test "a diagnostic keeps the element's dtype" (fun () ->
          let x = Nx.Rng.uniform ~key:(Nx.Rng.key 0) Nx.float32 [| 2; 10 |] in
          equal string "float32"
            (Nx.Dtype.name (Nx.dtype (Diag.rhat t (Draws.v t x)))));
      no_diagnostic;
      test "chains of 4 draws have diagnostics" (fun () ->
          let d = chains 2 4 [| 0.; 3.; 1.; 2.; 5.; 4.; 7.; 6. |] in
          satisfies ~claim:"finite" float_exact Float.is_finite
            (item (Diag.ess_bulk t d)));
      test "raises naming a tensor that is not of a float dtype" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"int32") (fun () ->
              Diag.rhat t (Draws.v t (Nx.zeros Nx.int32 [| 2; 10 |]))));
    ]

(* Convergence *)

let rhat =
  group "rhat"
    [
      test "agrees with ArviZ" (fun () ->
          against_arviz "rhat" (fun g -> Diag.rhat t (draws_of g)));
      prop "does not change under an affine map" chains_gen (fun case ->
          let d, d' = mapped case in
          equal nan_or_close (item (Diag.rhat t d)) (item (Diag.rhat t d')));
    ]

let nested_rhat =
  group "nested_rhat"
    [
      test "agrees with ArviZ" (fun () ->
          against_arviz
            ~only:(fun g -> g.superchains > 0)
            "nested_rhat"
            (fun g ->
              Diag.nested_rhat t ~superchains:g.superchains (draws_of g)));
      cases "raises for superchains below 2 or not dividing the chains"
        ~id:string_of_int [ 1; 3 ] (fun k ->
          let x = Nx.Rng.uniform ~key:(Nx.Rng.key 0) Nx.float64 [| 4; 10 |] in
          raises_match (Exn.invalid_arg ~sub:"Norn.Diag.nested_rhat") (fun () ->
              Diag.nested_rhat t ~superchains:k (Draws.v t x)));
    ]

let ess =
  group "ess_bulk and ess_tail"
    [
      test "ess_bulk agrees with ArviZ" (fun () ->
          against_arviz "ess_bulk" (fun g -> Diag.ess_bulk t (draws_of g)));
      test "ess_tail agrees with ArviZ" (fun () ->
          against_arviz "ess_tail" (fun g -> Diag.ess_tail t (draws_of g)));
      prop "do not change under an increasing map" chains_gen
        (fun (c, n, xs, _, _) ->
          cover "one chain" (c = 1);
          let d = chains c n xs in
          let d' = chains c n (Array.map (fun x -> Float.exp (x *. 0.5)) xs) in
          let same f = equal nan_or_close (item (f t d)) (item (f t d')) in
          same Diag.ess_bulk;
          same Diag.ess_tail);
    ]

let mcse_mean =
  group "mcse_mean"
    [
      test "agrees with ArviZ" (fun () ->
          against_arviz "mcse_mean" (fun g -> Diag.mcse_mean t (draws_of g)));
      prop "scales by |a| under a x + b" chains_gen
        (fun ((_, _, _, a, _) as case) ->
          let d, d' = mapped case in
          equal nan_or_close
            (Float.abs a *. item (Diag.mcse_mean t d))
            (item (Diag.mcse_mean t d')));
    ]

(* Hamiltonian transitions *)

let ebfmi =
  group "ebfmi"
    [
      test "agrees with ArviZ" (fun () ->
          against_arviz "ebfmi" (fun g ->
              let shape = [| g.chains; g.draws |] in
              Diag.ebfmi
                (transitions
                   ~energy:(Nx.create Nx.float64 shape g.data)
                   ~diverging:(Nx.zeros Nx.bool shape))));
      test "raises for chains of one draw" (fun () ->
          let s =
            transitions
              ~energy:(Nx.zeros Nx.float64 [| 2; 1 |])
              ~diverging:(Nx.zeros Nx.bool [| 2; 1 |])
          in
          raises_match (Exn.invalid_arg ~sub:"Norn.Diag.ebfmi") (fun () ->
              Diag.ebfmi s));
    ]

let divergent_shift =
  let x = chains 1 4 [| 0.; 1.; 2.; 5. |] in
  let at diverging =
    transitions
      ~energy:(Nx.zeros Nx.float64 [| 1; 4 |])
      ~diverging:(Nx.create Nx.bool [| 1; 4 |] diverging)
  in
  group "divergent_shift"
    [
      test "is the diverged draws' mean less the mean, in sds" (fun () ->
          (* Mean 2, sd sqrt (14 / 3) of 3 degrees of freedom, the diverged
             draws' mean 3.5. *)
          let s = at [| false; false; true; true |] in
          equal close
            (1.5 /. Float.sqrt (14. /. 3.))
            (item (Diag.divergent_shift t s x)));
      test "is NaN with no divergence" (fun () ->
          let s = at [| false; false; false; false |] in
          satisfies ~claim:"nan" float_exact Float.is_nan
            (item (Diag.divergent_shift t s x)));
      test "raises for other chains or draws" (fun () ->
          let s = at [| false; false; true; true |] in
          raises_match (Exn.invalid_arg ~sub:"Norn.Diag.divergent_shift")
            (fun () ->
              Diag.divergent_shift t s (chains 1 5 [| 0.; 1.; 2.; 3.; 4. |])));
    ]

(* Calibration *)

let rank =
  group "rank"
    [
      cases "counts the draws strictly below the truth"
        ~id:(fun (truth, _) -> Printf.sprintf "%g" truth)
        [ (-1., 0.); (2., 2.); (2.5, 3.); (9., 6.) ]
        (fun (truth, expected) ->
          let d = chains 2 3 [| 1.; 5.; 3.; 2.; 4.; 0. |] in
          equal float_exact expected (item (Diag.rank t ~truth:(f64 truth) d)));
      test "raises past float16's 2048 exact integers" (fun () ->
          let d = Draws.v t (Nx.zeros Nx.float16 [| 4; 600 |]) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Diag.rank") (fun () ->
              Diag.rank t ~truth:(Nx.scalar Nx.float16 0.) d));
      test "raises past float4's 4 exact integers" (fun () ->
          let d = Draws.v t (Nx.zeros Nx.float4_e2m1fn [| 1; 5 |]) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Diag.rank") (fun () ->
              Diag.rank t ~truth:(Nx.scalar Nx.float4_e2m1fn 0.) d));
    ]

(* [enumerated ~ranks:n ~draws:l] is the simultaneous p-value of every tuple of
   [n] ranks in [0, ..., l], tuple [t] read in base [l + 1], computed in
   integers over the [(l + 1)^n] equally likely tuples. The count below point
   [i] is binomial [(n, i / (l + 1))], a count's pointwise p-value twice its
   smaller tail, at most 1 (Säilynoja et al. 2022, eq. 13), and a tuple's
   p-value the share of tuples whose least pointwise p-value is at most its own,
   borders included (eq. 15). Every quantity is scaled by [(l + 1)^n]. *)
let enumerated ~ranks:n ~draws:l =
  let rec pow a b = if b = 0 then 1 else a * pow a (b - 1) in
  let rec choose n k =
    if k = 0 then 1 else choose n (k - 1) * (n - k + 1) / k
  in
  let d = pow (l + 1) n in
  let rank t r = t / pow (l + 1) r mod (l + 1) in
  let tail i lo hi =
    let s = ref 0 in
    for k = lo to hi do
      s := !s + (choose n k * pow i k * pow (l + 1 - i) (n - k))
    done;
    !s
  in
  let least t =
    let g = ref d in
    for i = 1 to l do
      let c =
        List.length (List.filter (fun r -> rank t r < i) (List.init n Fun.id))
      in
      g := min !g (min d (2 * min (tail i 0 c) (tail i c n)))
    done;
    !g
  in
  let least = Array.init d least in
  let p g = Array.fold_left (fun k h -> if h <= g then k + 1 else k) 0 least in
  ( List.init n (fun r ->
        Nx.create Nx.float64 [| d |]
          (Array.init d (fun t -> float_of_int (rank t r)))),
    Array.map (fun g -> float_of_int (p g) /. float_of_int d) least )

let rank_uniformity =
  group "rank_uniformity"
    [
      prop "is the share of uniform ranks at least as far, borders included"
        Gen.(pair (int_range 1 5) (int_range 1 4))
        (fun (n, l) ->
          let ranks, expected = enumerated ~ranks:n ~draws:l in
          equal
            (array (float 1e-12))
            expected
            (Nx.to_array (Diag.rank_uniformity t ~draws:l ranks)));
      test "uniform ranks have a p-value below 5% at most 5% of the time"
        (fun () ->
          (* 10000 replications as 2000 elements of 40 ranks each, out of 30
             draws, at each of five keys. Rejections are binomial with
             probability at most 5%; 557 is its 99.5% quantile at 5%. *)
          let rejected =
            List.fold_left
              (fun n key ->
                let k = Nx.Rng.key key in
                let ranks =
                  List.init 40 (fun i ->
                      Nx.cast Nx.float64
                        (Nx.Rng.randint ~key:(Nx.Rng.fold_in k i)
                           (Nx.full Nx.int32 [| 2000 |] 31l)))
                in
                let stat =
                  Nx.to_array (Diag.rank_uniformity t ~draws:30 ranks)
                in
                Array.fold_left
                  (fun n p -> if p < 0.05 then n + 1 else n)
                  n stat)
              0 [ 11; 12; 13; 14; 15 ]
          in
          at_most int ~than:557 rejected;
          (* Not so conservative it never rejects: 340 is below the 0.5%
             quantile at a rate of 4%. *)
          at_least int ~than:340 rejected);
      cases "rejects ranks piled at zero or in the middle" ~id:fst
        [
          ("20 at zero", List.init 20 (fun _ -> 0.));
          ( "100 at 4 to 6",
            List.init 100 (fun i -> float_of_int (4 + (i mod 3))) );
        ]
        (fun (_, ranks) ->
          less float_exact ~than:1e-6
            (item (Diag.rank_uniformity t ~draws:10 (List.map f64 ranks))));
      cases "raises for no ranks, draws below 1, or a rank not in 0 to draws"
        ~id:fst
        [
          ("no ranks", (10, []));
          ("draws 0", (0, [ 0. ]));
          ("rank 11", (10, [ 11. ]));
          ("rank 1.5", (10, [ 1.5 ]));
          ("rank -1", (10, [ -1. ]));
        ]
        (fun (_, (draws, ranks)) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Diag.rank_uniformity")
            (fun () -> Diag.rank_uniformity t ~draws (List.map f64 ranks)));
    ]

let () =
  exit
    (run "Norn.Diag"
       [
         elements;
         rhat;
         nested_rhat;
         ess;
         mcse_mean;
         ebfmi;
         divergent_shift;
         rank;
         rank_uniformity;
       ])
