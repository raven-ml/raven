(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Diag = Norn.Diag
module Draws = Norn.Draws

let t = Nx.Ptree.tensor
let item x = Nx.item [] x

(* Goldens *)

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
let draws_of g = Draws.v t (Nx.create Nx.float64 [| g.chains; g.draws |] g.data)

(* Within rounding of the transforms' different summation orders. *)
let close = float_rel ~rel:1e-9 ~abs:1e-12

let nan_or_close =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b) || Testable.equal close a b)

let check_golden g =
  let d = draws_of g in
  let value key = (List.assoc key g.values).(0) in
  subtest "rhat" (fun () ->
      equal nan_or_close (value "rhat") (item (Diag.rhat t d)));
  subtest "ess_bulk" (fun () ->
      equal nan_or_close (value "ess_bulk") (item (Diag.ess_bulk t d)));
  subtest "ess_tail" (fun () ->
      equal nan_or_close (value "ess_tail") (item (Diag.ess_tail t d)));
  subtest "mcse_mean" (fun () ->
      equal nan_or_close (value "mcse_mean") (item (Diag.mcse_mean t d)));
  if g.superchains > 0 then
    subtest "nested_rhat" (fun () ->
        equal nan_or_close (value "nested_rhat")
          (item (Diag.nested_rhat t ~superchains:g.superchains d)));
  subtest "ebfmi" (fun () ->
      let n = g.chains in
      let z = Nx.zeros Nx.float64 [| n; g.draws |] in
      let zi = Nx.zeros Nx.int32 [| n; g.draws |] in
      let zb = Nx.zeros Nx.bool [| n; g.draws |] in
      let energy = Nx.create Nx.float64 [| n; g.draws |] g.data in
      let s =
        Draws.v
          (Norn.Stats.ptree Nx.float64)
          Norn.Stats.
            {
              lp = z;
              acceptance = z;
              step_size = z;
              n_steps = zi;
              diverging = zb;
              saturated = zb;
              energy;
            }
      in
      equal (array nan_or_close)
        (List.assoc "ebfmi" g.values)
        (Nx.to_array (Diag.ebfmi s)))

let reference =
  group "against ArviZ"
    (List.map
       (fun g -> test g.name (fun () -> check_golden g))
       (Lazy.force goldens))

(* Laws *)

type 'a two = { x : 'a; y : 'a }

module Two = struct
  type 'a t = 'a two

  let walk c { x; y } =
    let open Nx.Ptree.Walk in
    let x = field c "x" leaf x in
    let y = field c "y" leaf y in
    { x; y }
end

let two : Nx.float64_t two Nx.Ptree.t = Nx.Ptree.instantiate (module Two)

let chains_gen =
  Gen.(
    let* c, n = pair (int_range 1 4) (int_range 4 40) in
    (* A grid that [exp] keeps distinct, ties included. *)
    let+ xs =
      array
        ~size:(constant (c * n))
        (map (fun k -> float_of_int k /. 8.) (int_range (-40) 40))
    in
    (c, n, xs))
  |> Gen.with_pp (fun ppf (c, n, xs) ->
      Format.fprintf ppf "%d chains of %d: [%s]" c n
        (String.concat "; "
           (Array.to_list (Array.map (Printf.sprintf "%h") xs))))

let invariance =
  prop "effective sizes do not change under an increasing map" chains_gen
    (fun (c, n, xs) ->
      let x = Nx.create Nx.float64 [| c; n |] xs in
      let d = Draws.v t x and d' = Draws.v t (Nx.exp (Nx.mul_s x 0.5)) in
      let same f = equal nan_or_close (item (f t d)) (item (f t d')) in
      same Diag.ess_bulk;
      same Diag.ess_tail)

let structures =
  group "structures"
    [
      test "each element of each tensor has its own diagnostic" (fun () ->
          let g = List.hd (Lazy.force goldens) in
          let x = Nx.create Nx.float64 [| g.chains; g.draws |] g.data in
          let y = Nx.stack ~axis:2 [ Nx.neg x; Nx.mul_s x 2. ] in
          let r = Diag.rhat two (Draws.v two { x; y }) in
          equal (array int) [| 2 |] (Nx.shape r.y);
          equal close (item r.x) (Nx.item [ 0 ] r.y);
          equal close (item r.x) (Nx.item [ 1 ] r.y));
      test "a constant element has no R-hat" (fun () ->
          let d = Draws.v t (Nx.ones Nx.float64 [| 4; 10 |]) in
          satisfies ~claim:"nan" float_exact Float.is_nan (item (Diag.rhat t d)));
      test "chains of three draws have no effective size" (fun () ->
          let d =
            Draws.v t (Nx.arange_f Nx.float64 0. 6. 1. |> Nx.reshape [| 2; 3 |])
          in
          satisfies ~claim:"nan" float_exact Float.is_nan
            (item (Diag.ess_bulk t d)));
      test "a diagnostic keeps the element's dtype" (fun () ->
          let d = Draws.v t (Nx.rand Nx.float32 [| 2; 10 |]) in
          equal string "float32" (Nx_dtype.to_string (Nx.dtype (Diag.rhat t d))));
      test "integer draws are refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"int32 draws") (fun () ->
              Diag.rhat t (Draws.v t (Nx.zeros Nx.int32 [| 2; 10 |]))));
      test "superchains divide the chains" (fun () ->
          raises_match
            (Exn.invalid_arg ~substring:"3 superchains do not divide 4")
            (fun () ->
              Diag.nested_rhat t ~superchains:3
                (Draws.v t (Nx.rand Nx.float64 [| 4; 10 |]))));
    ]

let divergences =
  test "divergent_shift is the divergent draws' mean shift in sds" (fun () ->
      let x = Nx.create Nx.float64 [| 1; 4 |] [| 0.; 1.; 2.; 5. |] in
      let z = Nx.zeros Nx.float64 [| 1; 4 |] in
      let zi = Nx.zeros Nx.int32 [| 1; 4 |] in
      let diverging =
        Nx.create Nx.bool [| 1; 4 |] [| false; false; true; true |]
      in
      let s =
        Draws.v
          (Norn.Stats.ptree Nx.float64)
          Norn.Stats.
            {
              lp = z;
              acceptance = z;
              step_size = z;
              n_steps = zi;
              diverging;
              saturated = Nx.zeros Nx.bool [| 1; 4 |];
              energy = z;
            }
      in
      (* Mean 2, sd sqrt (14 / 3), divergent mean 3.5. *)
      let expected = 1.5 /. Float.sqrt (14. /. 3.) in
      equal close expected (item (Diag.divergent_shift t s (Draws.v t x))))

(* Calibration *)

let ranks =
  group "rank"
    [
      test "more draws than exact integers are refused" (fun () ->
          let d = Draws.v t (Nx.zeros Nx.float16 [| 4; 600 |]) in
          raises
            (Invalid_argument
               "Norn.Diag.rank: : 2400 draws exceed float16's exact integers, \
                2048") (fun () ->
              Diag.rank t ~truth:(Nx.scalar Nx.float16 0.) d));
      test "a rank counts the draws below the truth" (fun () ->
          let d =
            Draws.v t
              (Nx.create Nx.float64 [| 2; 3 |] [| 1.; 5.; 3.; 2.; 4.; 0. |])
          in
          equal float_exact 3.
            (item (Diag.rank t ~truth:(Nx.scalar Nx.float64 2.5) d));
          equal float_exact 0.
            (item (Diag.rank t ~truth:(Nx.scalar Nx.float64 (-1.)) d));
          equal float_exact 6.
            (item (Diag.rank t ~truth:(Nx.scalar Nx.float64 9.) d)));
    ]

let uniformity =
  group "rank_uniformity"
    [
      test "two ranks at zero of one draw have a p-value of one half" (fun () ->
          (* The count below 1 is binomial (2, 1/2): 0, 1, 2 with 1/4, 1/2, 1/4,
             pointwise p-values 1/2, 1, 1/2. Both ranks at 0 count 2, and a
             count as far is 0 or 2, of probability 1/2. *)
          let r =
            Diag.rank_uniformity t ~draws:1
              [ Nx.scalar Nx.float64 0.; Nx.scalar Nx.float64 0. ]
          in
          equal close 0.5 (item r));
      test "a count at its median has a p-value of one" (fun () ->
          let r =
            Diag.rank_uniformity t ~draws:1
              [ Nx.scalar Nx.float64 0.; Nx.scalar Nx.float64 1. ]
          in
          equal close 1. (item r));
      slow "uniform ranks have a p-value below 5% at most 5% of the time"
        (fun () ->
          (* 2000 replications as 2000 elements of 40 ranks each, out of 30
             draws. Rejections are binomial with probability at most 5%; 126 is
             its 99.5% quantile at 5%. *)
          let k = Nx.Rng.key 11 in
          let ranks =
            List.init 40 (fun i ->
                Nx.cast Nx.float64
                  (Nx.Rng.randint (Nx.Rng.fold_in k i) ~high:31 [| 2000 |]))
          in
          let stat = Nx.to_array (Diag.rank_uniformity t ~draws:30 ranks) in
          let rejected =
            Array.fold_left (fun n p -> if p < 0.05 then n + 1 else n) 0 stat
          in
          at_most int ~than:126 rejected;
          (* Not so conservative it never rejects: 50 is far below the 0.5%
             quantile at a rate of 4%. *)
          at_least int ~than:50 rejected);
      test "ranks piled at zero are rejected" (fun () ->
          let ranks = List.init 20 (fun _ -> Nx.scalar Nx.float64 0.) in
          less float_exact ~than:1e-6
            (item (Diag.rank_uniformity t ~draws:10 ranks)));
      test "ranks piled in the middle are rejected" (fun () ->
          let ranks =
            List.init 100 (fun i ->
                Nx.scalar Nx.float64 (float_of_int (4 + (i mod 3))))
          in
          less float_exact ~than:1e-6
            (item (Diag.rank_uniformity t ~draws:10 ranks)));
      test "a rank above the draws is refused" (fun () ->
          raises_match
            (Exn.invalid_arg ~substring:"rank 11 is not in 0, ..., 10")
            (fun () ->
              Diag.rank_uniformity t ~draws:10 [ Nx.scalar Nx.float64 11. ]));
      test "no ranks are refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"no ranks") (fun () ->
              Diag.rank_uniformity t ~draws:10 []));
    ]

let () =
  exit
    (run "Norn.Diag"
       [
         reference;
         group "laws" [ invariance ];
         structures;
         group "divergences" [ divergences ];
         ranks;
         uniformity;
       ])
