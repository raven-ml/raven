(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A compiled call gives the same bits whatever the process compiled before it
   (rune.mli, Numerics), so a program compiled first in its process, in a
   process of its own, compiles to eager's bits too. Four nested scans meet
   there storage numbered below the innermost call's argument count, which once
   ran such a call evaluated or out of bounds. *)

open Windtrap
open Rune_support
open Witness

let f64 = Nx.float64

(* Three scans deep under a fourth. The innermost step reads the outermost
   carry [w] through [parts] values, each a loop invariant passed to the step's
   call, and the result of a scan [z] that the second level runs first. Halves
   and sums of halves keep every value exact. *)
let parts = 24

let nested rows x =
  let inner w z g _ =
    let acc = ref (Nx.add g z) in
    for i = 0 to parts - 1 do
      acc := Nx.add !acc (Nx.mul_s (Nx.get [ i ] w) 0.5)
    done;
    (!acc, g)
  in
  let level w cw _ =
    let z, _ =
      Rune.scan' ~init:x (fun c q -> (Nx.add (Nx.mul_s c 0.5) q, c)) rows
    in
    let b, _ =
      Rune.scan' ~init:(Nx.add z cw)
        (fun e _ -> (fst (Rune.scan' ~init:e (inner w z) rows), e))
        rows
    in
    (Nx.add cw b, b)
  in
  let top w _ =
    let cw, _ = Rune.scan' ~init:(Nx.zeros f64 [| 1 |]) (level w) rows in
    (Nx.add (Nx.mul_s w 0.5) cw, cw)
  in
  fst (Rune.scan' ~init:(Nx.full f64 [| 32 |] 0.25) top rows)

let () =
  exit
    (run "rune first compile"
       [
         test "nested scans compiled first in a process give eager's bits"
           (fun () ->
             let rows = Nx.zeros f64 [| 2; 1 |] in
             let x = Nx.create f64 [| 1 |] [| 0.25 |] in
             let expected = bits (nested rows x) in
             let calls = Rune.evaluations () in
             let got = bits (Rune.jit' (nested rows) x) in
             Compiled.here ();
             equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - calls);
             equal bits_t expected got);
       ])
