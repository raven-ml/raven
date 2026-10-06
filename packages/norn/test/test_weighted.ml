(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module W = Norn.Weighted

let t = Nx.Ptree.tensor

let weighted lw =
  let n = Array.length lw in
  W.
    {
      values = Nx.arange_f Nx.float64 0. (float_of_int n) 1.;
      log_weights = Nx.create Nx.float64 [| n |] lw;
    }

(* Log weights in [-10, 2], some of them [-inf]: a draw of no weight. *)
let log_weights =
  Gen.array ~size:(Gen.int_range 1 40)
    (Gen.frequency
       [ (4, Gen.float_range (-10.) 2.); (1, Gen.constant Float.neg_infinity) ])
  |> Gen.such_that (Array.exists Float.is_finite)

let normalised lw =
  let w = Array.map Float.exp lw in
  let s = Array.fold_left ( +. ) 0. w in
  Array.map (fun w -> w /. s) w

let ess =
  group "ess"
    [
      prop "is Kish's (Σ w)² / Σ w²" log_weights (fun lw ->
          let w = Array.map Float.exp lw in
          let s = Array.fold_left ( +. ) 0. w in
          let s2 = Array.fold_left (fun a w -> a +. (w *. w)) 0. w in
          equal (float 1e-9) (s *. s /. s2) (Nx.item [] (W.ess (weighted lw))));
      test "equal weights count every draw" (fun () ->
          equal (float 1e-12) 7.
            (Nx.item [] (W.ess (weighted (Array.make 7 3.)))));
    ]

let resample =
  group "resample"
    [
      prop "copies a draw floor (n w) or ceil (n w) times"
        (Gen.triple log_weights (Gen.int_range 1 60) Gen.int)
        (fun (lw, n, seed) ->
          let x = W.resample t (Nx.Rng.key seed) ~n (weighted lw) in
          let counts = Array.make (Array.length lw) 0 in
          Array.iter
            (fun v -> counts.(int_of_float v) <- counts.(int_of_float v) + 1)
            (Nx.to_array x);
          Array.iteri
            (fun i w ->
              let e = float_of_int n *. w in
              satisfies ~claim:"floor (n w) <= count <= ceil (n w)" int
                (fun c ->
                  float_of_int c >= Float.floor e -. 1e-9
                  && float_of_int c <= Float.ceil e +. 1e-9)
                counts.(i))
            (normalised lw));
      test "n below 1 is refused" (fun () ->
          raises
            (Invalid_argument "Norn.Weighted.resample: n = 0 is not positive")
            (fun () ->
              ignore (W.resample t (Nx.Rng.key 1) ~n:0 (weighted [| 0. |]))));
    ]

let () = exit (run "Norn.Weighted" [ ess; resample ])
