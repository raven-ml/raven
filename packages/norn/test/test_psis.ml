(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Diag = Norn.Diag
module Draws = Norn.Draws

let t = Nx.Ptree.tensor
let item x = Nx.item [] x
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let floats x = Nx.to_array (Nx.cast Nx.float64 x)

let log_sum_exp xs =
  let m = Array.fold_left Float.max Float.neg_infinity xs in
  m +. Float.log (Array.fold_left (fun s x -> s +. Float.exp (x -. m)) 0. xs)

let normalise xs =
  let z = log_sum_exp xs in
  Array.map (fun x -> x -. z) xs

(* [tail_length s] is [floor (min (0.2 s, 3 sqrt s))]: the largest [m] with
   [5 m <= s] and [m² <= 9 s]. *)
let tail_length s =
  let m = ref 0 in
  while 5 * (!m + 1) <= s && (!m + 1) * (!m + 1) <= 9 * s do
    incr m
  done;
  !m

let pp_floats ppf xs =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map (Printf.sprintf "%h") xs)))

(* Bits, so that NaN equals NaN and -0. differs from 0. *)
let same_floats =
  let bits x = Int64.bits_of_float x in
  Testable.make ~pp:pp_floats ~equal:(fun a b ->
      Array.length a = Array.length b
      && Array.for_all2 (fun x y -> Int64.equal (bits x) (bits y)) a b)

let is_nan x = satisfies ~claim:"nan" float_exact Float.is_nan x

(* Goldens *)

type case = {
  kind : string;
  name : string;
  dims : int list;
  lines : (string * float array) list;
}

let read_goldens () =
  let ic = open_in "golden/psis.golden" in
  let rec lines acc =
    match input_line ic with
    | l -> lines (if String.length l > 0 && l.[0] = '#' then acc else l :: acc)
    | exception End_of_file ->
        close_in ic;
        List.rev acc
  in
  let floats ws = Array.of_list (List.map float_of_string ws) in
  let header l =
    match String.split_on_char ' ' l with
    | ("psis" | "loo") as kind :: name :: dims ->
        Some (kind, name, List.map int_of_string dims)
    | _ -> None
  in
  let rec cases acc = function
    | [] -> List.rev acc
    | l :: rest -> (
        match header l with
        | None -> failwith ("psis.golden: unexpected line " ^ l)
        | Some (kind, name, dims) ->
            let rec body values = function
              | l :: rest when header l = None -> (
                  match String.split_on_char ' ' l with
                  | key :: ws -> body ((key, floats ws) :: values) rest
                  | [] -> body values rest)
              | rest -> (List.rev values, rest)
            in
            let lines, rest = body [] rest in
            cases ({ kind; name; dims; lines } :: acc) rest)
  in
  cases [] (lines [])

let goldens = lazy (read_goldens ())

(* Within rounding of the two implementations' summation orders. *)
let close = float_rel ~rel:1e-9 ~abs:1e-12

let check_psis g =
  let v key = List.assoc key g.lines in
  let { Diag.log_weights = w; khat = k } = Diag.psis (vec (v "data")) in
  equal close (v "khat").(0) (item k);
  equal (array close) (v "weights") (normalise (floats w))

let check_loo g =
  let v key = List.assoc key g.lines in
  let c, s, n =
    match g.dims with [ c; s; n ] -> (c, s, n) | _ -> failwith g.name
  in
  let ll = Nx.create Nx.float64 [| c; s; n |] (v "data") in
  let r = Diag.loo (Draws.v t ll) in
  equal (array close) (v "elpd_i") (floats r.pointwise);
  equal (array close) (v "khat") (floats r.khat);
  equal close (v "elpd").(0) (item r.elpd);
  equal close (v "se").(0) (item r.se);
  equal close (v "p").(0) (item r.p);
  let p_i = v "p_i" in
  equal close (Array.fold_left ( +. ) 0. p_i) (item r.p)

let reference =
  group "against ArviZ"
    (List.map
       (fun g ->
         test (g.kind ^ " " ^ g.name) (fun () ->
             if g.kind = "psis" then check_psis g else check_loo g))
       (Lazy.force goldens))

(* Laws *)

(* Log ratios that break the smoothing: few draws and many, a scale from
   nearly equal ratios to a tail that spans hundreds of nats, ties from a coarse
   grid, and zero weights. *)
let ratios_gen =
  Gen.(
    let* s = one_of [ int_range 0 30; int_range 25 400 ] in
    let* scale = of_list [ 1e-6; 0.3; 1.; 4.; 60. ] in
    let* coarse = bool in
    let* zeros = int_range 0 (s / 10) in
    let+ xs = array ~size:(constant s) (float_range (-3.) 3.) in
    let xs =
      Array.map
        (fun x ->
          let x = scale *. Float.exp x in
          if coarse then Float.round (x *. 2.) /. 2. else x)
        xs
    in
    for i = 0 to zeros - 1 do
      xs.(i) <- Float.neg_infinity
    done;
    xs)
  |> Gen.with_pp pp_floats

let sorted xs =
  let ys = Array.copy xs in
  Array.sort Float.compare ys;
  ys

let smoothing =
  prop "smoothing keeps the S - M smallest ratios, the order and the bounds"
    ratios_gen (fun lr ->
      let s = Array.length lr in
      let m = tail_length s in
      let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
      let w = floats w and k = item k in
      cover "fewer than 25 draws" (s < 25);
      cover "a tail fitted" (Float.is_finite k);
      cover "a bounded tail" (k = Float.neg_infinity);
      if Float.is_nan k then equal same_floats lr w
      else begin
        let body a = Array.sub (sorted a) 0 (s - m) in
        equal same_floats (body lr) (body w);
        let top = Array.fold_left Float.max Float.neg_infinity lr in
        let largest = Array.fold_left Float.max Float.neg_infinity w in
        at_most float_exact ~than:top largest;
        (* Pairs whose ratios are ordered and whose weights are not. *)
        let crossed = ref [] in
        Array.iteri
          (fun i wi ->
            Array.iteri
              (fun j wj ->
                if lr.(i) < lr.(j) && wi > wj then
                  crossed := (i, j) :: !crossed)
              w)
          w;
        equal (list (pair int int)) [] !crossed
      end)

(* The ratios on a grid of 2^-20, so that adding an integer is exact and the
   two sets differ by the constant alone. *)
let shift =
  prop "adding a constant to the log ratios adds it to the weights"
    Gen.(pair ratios_gen (map float_of_int (int_range (-500) 500)))
    (fun (lr, c) ->
      let grid = 1048576. in
      let lr = Array.map (fun x -> Float.round (x *. grid) /. grid) lr in
      let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
      let shifted = vec (Array.map (fun x -> x +. c) lr) in
      let { Diag.log_weights = w'; khat = k' } = Diag.psis shifted in
      equal same_floats [| item k |] [| item k' |];
      let shifted = Array.map (fun x -> x +. c) (floats w) in
      equal (array (float_rel ~rel:1e-15 ~abs:1e-12)) shifted (floats w'))

(* Ratios [U^(-k)], U uniform, have a Pareto tail of shape [k]: their excesses
   over any threshold are generalized Pareto of shape [k]. A likelihood
   estimate from [M] excesses spreads by about [(1 + k) / sqrt M]; five of
   those, beside the prior's pull [10 (1/2 - k) / (M + 10)], bound a false
   failure at 6e-7 a case under normality. *)
let tail_index =
  let draws = 100_000 in
  prop ~count:20 "k-hat recovers a Pareto tail's shape"
    Gen.(pair (float_range 0.1 0.9) (int_range 0 1_000_000))
    (fun (k, seed) ->
      let u = Nx.Rng.uniform ~key:(Nx.Rng.key seed) Nx.float64 [| draws |] in
      let lr = Nx.mul_s (Nx.log u) (-.k) in
      let khat = item (Diag.psis lr).khat in
      let m = float_of_int (tail_length draws) in
      let bias = 10. *. (0.5 -. k) /. (m +. 10.) in
      cover "a tail with finite variance" (k < 0.5);
      cover "a tail with a mean and no variance" (k > 0.5);
      equal (float (5. *. (1. +. k) /. Float.sqrt m)) (k +. bias) khat)

(* Edges *)

let edges =
  group "edges"
    [
      test "24 draws have no tail" (fun () ->
          let lr = Array.init 24 (fun i -> Float.of_int i /. 3.) in
          let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
          is_nan (item k);
          equal same_floats lr (floats w));
      test "25 draws fit a tail of 5" (fun () ->
          let lr = Array.init 25 (fun i -> Float.of_int (i * i) /. 50.) in
          let k = (Diag.psis (vec lr)).khat in
          satisfies ~claim:"finite" float_exact Float.is_finite (item k));
      cases ~id:fst "a ratio that is NaN or +inf leaves the set unsmoothed"
        [ ("nan", Float.nan); ("+inf", Float.infinity) ]
        (fun (_, bad) ->
          let lr = Array.init 100 (fun i -> Float.of_int i /. 10.) in
          lr.(42) <- bad;
          let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
          is_nan (item k);
          equal same_floats lr (floats w));
      test "every ratio -inf has no tail" (fun () ->
          let lr = Array.make 100 Float.neg_infinity in
          let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
          is_nan (item k);
          equal same_floats lr (floats w));
      test "equal ratios have a bounded tail" (fun () ->
          let lr = Array.make 100 (-2.5) in
          let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
          equal float_exact Float.neg_infinity (item k);
          equal same_floats lr (floats w));
      test "a quarter of the tail at the cutoff is no Pareto tail" (fun () ->
          (* M = 20, and the tail's 5 smallest, its first quartile, equal the
             largest of the others. *)
          let lr =
            Array.init 100 (fun i ->
                if i < 75 then Float.of_int i /. 100.
                else if i < 85 then 1.
                else 1. +. (Float.of_int (i - 84) /. 10.))
          in
          let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
          is_nan (item k);
          equal same_floats lr (floats w));
      test "zero weights stay zero" (fun () ->
          let lr =
            Array.init 200 (fun i ->
                if i < 30 then Float.neg_infinity
                else Float.sqrt (Float.of_int i))
          in
          let { Diag.log_weights = w; khat = k } = Diag.psis (vec lr) in
          satisfies ~claim:"finite" float_exact Float.is_finite (item k);
          equal same_floats (Array.sub lr 0 30) (Array.sub (floats w) 0 30));
      test "no draws have no tail" (fun () ->
          let lr = Nx.zeros Nx.float64 [| 0; 3 |] in
          let { Diag.log_weights = w; khat = k } = Diag.psis lr in
          equal (array int) [| 0; 3 |] (Nx.shape w);
          equal (array int) [| 3 |] (Nx.shape k);
          Array.iter is_nan (floats k));
      test "a scalar is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"no axis of draws") (fun () ->
              Diag.psis (Nx.scalar Nx.float64 1.)));
      test "each index past axis 0 is its own set" (fun () ->
          let k = Nx.Rng.key 7 in
          let x = Nx.Rng.normal ~key:k Nx.float64 [| 300; 2; 3 |] in
          let x = Nx.mul x x in
          let { Diag.log_weights = w; khat = k } = Diag.psis x in
          equal (array int) [| 2; 3 |] (Nx.shape k);
          for i = 0 to 1 do
            for j = 0 to 2 do
              let one = Nx.slice [ Nx.A; Nx.I i; Nx.I j ] x in
              let { Diag.log_weights = w1; khat = k1 } = Diag.psis one in
              equal same_floats [| item k1 |] [| Nx.item [ i; j ] k |];
              equal same_floats (floats w1)
                (floats (Nx.slice [ Nx.A; Nx.I i; Nx.I j ] w))
            done
          done);
      test "the weights and k-hat keep the ratios' dtype" (fun () ->
          let x = Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float32 [| 100 |] in
          let { Diag.log_weights = w; khat = k } = Diag.psis x in
          equal string "float32" (Nx.Dtype.name (Nx.dtype w));
          equal string "float32" (Nx.Dtype.name (Nx.dtype k)));
    ]

let loo_edges =
  group "loo edges"
    [
      test "a point not finite at a draw has NaN terms" (fun () ->
          let shape = [| 2; 100; 3 |] in
          let a = floats (Nx.Rng.normal ~key:(Nx.Rng.key 2) Nx.float64 shape) in
          (* Point 1 at draw 17 of chain 0. *)
          a.((3 * 17) + 1) <- Float.neg_infinity;
          let r = Diag.loo (Draws.v t (Nx.create Nx.float64 shape a)) in
          let pw = floats r.pointwise and k = floats r.khat in
          is_nan pw.(1);
          is_nan k.(1);
          satisfies ~claim:"finite" float_exact Float.is_finite pw.(0);
          is_nan (item r.elpd);
          is_nan (item r.p));
      test "one point has no standard error" (fun () ->
          let ll = Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| 2; 50 |] in
          let r = Diag.loo (Draws.v t ll) in
          equal (array int) [||] (Nx.shape r.pointwise);
          equal close (item r.pointwise) (item r.elpd);
          is_nan (item r.se));
      test "a constant likelihood leaves nothing out" (fun () ->
          (* Every ratio equal: elpd_i is the likelihood, p_loo is 0. *)
          let ll = Nx.full Nx.float64 [| 1; 40; 2 |] (-1.25) in
          let r = Diag.loo (Draws.v t ll) in
          equal (array close) [| -1.25; -1.25 |] (floats r.pointwise);
          equal (array float_exact) [| Float.neg_infinity; Float.neg_infinity |]
            (floats r.khat);
          equal close 0. (item r.p);
          equal close 0. (item r.se));
    ]

let () =
  exit
    (run "Norn.Diag.psis"
       [
         reference;
         group "laws" [ smoothing; shift; tail_index ];
         edges;
         loo_edges;
       ])
