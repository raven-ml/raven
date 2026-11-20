(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let same_shape want got = equal (array int) (Nx.shape want) (Nx.shape got)

let close tol want got =
  same_shape want got;
  equal (array (float tol)) (floats want) (floats got)

(* NaN equals NaN here: bit-for-bit comparison of float arrays. *)
let bits x =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (floats x)

let same want got =
  same_shape want got;
  equal (array int64) (bits want) (bits got)

let transforms =
  test "transforms" (fun () ->
      let module T = V.Transform in
      let p = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.; 2.; 3. |] in
      let t = Nx.create Nx.float64 [| 2 |] [| 0.5; -1. |] in
      equal (array float_exact) [| 1.5; -1.; 2.5; 2. |]
        (floats (T.apply (T.translation t) p));
      equal (array float_exact) [| 2.; 0.; 4.; 9. |]
        (floats
           (T.apply (T.scaling (Nx.create Nx.float64 [| 2 |] [| 2.; 3. |])) p));
      equal (array float_exact) [| 1.; 0.5; 2.; 4. |]
        (floats
           (T.apply (T.shear (Nx.create Nx.float64 [| 2 |] [| 0.; 0.5 |])) p));
      (* A quarter turn takes +x to +y. *)
      close 1e-15
        (Nx.create Nx.float64 [| 2; 2 |] [| 0.; 1.; -3.; 2. |])
        (T.apply (T.rotation (Nx.scalar Nx.float64 (Float.pi /. 2.))) p);
      (* matmul a b applies b first. *)
      let a = T.translation t
      and b = T.scaling (Nx.create Nx.float64 [| 2 |] [| 2.; 3. |]) in
      same (T.apply a (T.apply b p)) (T.apply (Nx.matmul a b) p);
      (* A projective transform divides. *)
      let m =
        Nx.create Nx.float64 [| 3; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0.; 1.; 0.; 1. |]
      in
      same
        (Nx.create Nx.float64 [| 2; 2 |] [| 0.5; 0.; 2. /. 3.; 1. |])
        (T.apply m p))

let () = exit (run "mynd.transform" [ transforms ])
