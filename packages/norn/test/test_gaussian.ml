(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module G = Norn.Gaussian

type 'a pos = { a : 'a; b : 'a }

module Pos = struct
  type 'a t = 'a pos

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let pos : Nx.float64_t pos Nx.Ptree.t = Nx.Ptree.instantiate (module Pos)
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let floats x = Nx.to_array x
let close = float_rel ~rel:1e-10 ~abs:1e-12

(* A position of 4 elements: [a] of 1, [b] of 3. *)
let of_vec v =
  { a = Nx.slice [ Nx.R (0, 1) ] v; b = Nx.slice [ Nx.R (1, 4) ] v }

let to_vec p = Nx.concatenate ~axis:0 [ p.a; p.b ]
let mean = of_vec (vec [| 0.5; -1.; 2.; 0.25 |])
let scale = of_vec (vec [| 2.; 0.5; 1.; 3. |])

(* Two orthonormal directions in 4 elements, and their variances. *)
let u1 = vec [| 0.5; 0.5; 0.5; 0.5 |]
let u2 = vec [| 0.5; -0.5; 0.5; -0.5 |]

let stacked us =
  let m = Nx.stack ~axis:0 us in
  {
    a = Nx.slice [ Nx.R (0, List.length us); Nx.R (0, 1) ] m;
    b = Nx.slice [ Nx.R (0, List.length us); Nx.R (1, 4) ] m;
  }

let variances = vec [| 4.; 0.25 |]
let g = G.low_rank pos ~mean ~scale ~directions:(stacked [ u1; u2 ]) ~variances

(* The dense covariance [S (I + Σ (v - 1) u uᵀ) S]. *)
let dense =
  let s = Nx.diag (to_vec scale) in
  let inner =
    List.fold_left2
      (fun acc u v ->
        Nx.add acc
          (Nx.mul_s
             (Nx.matmul
                (Nx.unsqueeze ~axes:[ 1 ] u)
                (Nx.unsqueeze ~axes:[ 0 ] u))
             (v -. 1.)))
      (Nx.eye Nx.float64 4) [ u1; u2 ] [ 4.; 0.25 ]
  in
  Nx.matmul s (Nx.matmul inner s)

let gaussian_log_density x =
  let d = Nx.sub x (to_vec mean) in
  let _, logdet = Nx.slogdet dense in
  let q = Nx.item [] (Nx.vdot d (Nx.solve dense d)) in
  (-0.5 *. q) -. (0.5 *. Nx.item [] logdet) -. (2. *. Float.log (2. *. Float.pi))

let points =
  Gen.(
    with_pp
      (fun ppf xs ->
        Format.fprintf ppf "[%s]"
          (String.concat "; " (Array.to_list (Array.map string_of_float xs))))
      (array ~size:(constant 4) (float_range (-5.) 5.)))

let laws =
  group "laws"
    [
      prop "the log density is the dense Gaussian's" points (fun x ->
          let x = vec x in
          let row = Nx.unsqueeze ~axes:[ 0 ] x in
          let lp =
            G.log_density pos g
              {
                a = Nx.slice [ Nx.R (0, 1); Nx.R (0, 1) ] row;
                b = Nx.slice [ Nx.R (0, 1); Nx.R (1, 4) ] row;
              }
          in
          equal close (gaussian_log_density x) (Nx.item [ 0 ] lp));
      test "the variance is the dense covariance's diagonal" (fun () ->
          equal (array close)
            (floats (Nx.diagonal dense))
            (floats (to_vec (G.variance pos g))));
      test "directions are orthonormalised" (fun () ->
          let g' =
            G.low_rank pos ~mean ~scale
              ~directions:
                (stacked [ Nx.mul_s u1 3.; Nx.add u2 (Nx.mul_s u1 0.7) ])
              ~variances
          in
          equal (array close)
            (floats (to_vec (G.variance pos g)))
            (floats (to_vec (G.variance pos g'))));
      test "a precision's Gaussian has its inverse as covariance" (fun () ->
          let p = of_vec (vec [| 1.; 4.; 0.5; 2. |]) in
          let w = stacked [ vec [| 1.; 0.; 1.; 0. |] ] in
          let l = vec [| 3. |] in
          let g = G.of_precision pos Nx.float64 ~low_rank:(w, l) ~mean p in
          let precision =
            Nx.add
              (Nx.diag (to_vec p))
              (Nx.mul_s
                 (Nx.matmul
                    (Nx.reshape [| 4; 1 |] (vec [| 1.; 0.; 1.; 0. |]))
                    (Nx.reshape [| 1; 4 |] (vec [| 1.; 0.; 1.; 0. |])))
                 3.)
          in
          let cov = Nx.inv precision in
          equal (array close)
            (floats (Nx.diagonal cov))
            (floats (to_vec (G.variance pos g))));
      slow "draws have the Gaussian's moments" (fun () ->
          let n = 200000 in
          let x = G.sample pos (Nx.Rng.key 4) ~n g in
          let x = Nx.concatenate ~axis:1 [ x.a; x.b ] in
          let m = Nx.mean ~axes:[ 0 ] x in
          (* Each mean is within six standard errors, a false alarm of 2e-9. *)
          let se = Nx.sqrt (Nx.div_s (Nx.diagonal dense) (float_of_int n)) in
          Array.iteri
            (fun i d ->
              less (float 1e-12) ~than:(6. *. (floats se).(i)) (Float.abs d))
            (floats (Nx.sub m (to_vec mean)));
          let c =
            Nx.div_s
              (Nx.matmul (Nx.transpose (Nx.sub x m)) (Nx.sub x m))
              (float_of_int n)
          in
          equal
            (array (float 0.1))
            (floats (Nx.reshape [| 16 |] dense))
            (floats (Nx.reshape [| 16 |] c)));
    ]

let errors =
  group "errors"
    [
      test "a precision that is not positive names its path" (fun () ->
          raises
            (Invalid_argument
               "Norn.Gaussian.of_precision: b at [1] is -1, not in (0, inf)")
            (fun () ->
              G.of_precision pos Nx.float64 ~mean
                (of_vec (vec [| 1.; 1.; -1.; 1. |]))));
      test "a variance that is not positive raises" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"variances at [1] is 0")
            (fun () ->
              G.low_rank pos ~mean ~scale
                ~directions:(stacked [ u1; u2 ])
                ~variances:(vec [| 1.; 0. |])));
      test "a diagonal Gaussian has rank 0" (fun () ->
          equal string "gaussian(dimension 4, rank 0)"
            (Format.asprintf "%a" (G.pp pos)
               (G.diagonal pos Nx.float64 ~mean ~scale)));
    ]

let () = exit (run "Norn.Gaussian" [ laws; errors ])
