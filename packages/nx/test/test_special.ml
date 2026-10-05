(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's special functions against correctly rounded references (golden/special/,
   written by gen/special.py), at the bounds nx.mli states per function and
   dtype; a zero result is held to its sign. *)

open Windtrap
open Nx_test.Special

let golden name = "golden/special/" ^ name ^ ".golden"

type u = { u : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

let unary name ~bound ?scale { u } =
  let f = { f = (fun a -> u a.(0)) } in
  let scale = Option.map (fun { u } -> { f = (fun a -> u a.(0)) }) scale in
  group name
    [
      group "against the goldens" (check ~bound (golden name) f);
      group "at every narrow float" (narrow ~bound ?scale f);
    ]

let everywhere b _ = b

(* [erfinv]'s condition number at [p]: [|p sqrt pi e^(x^2) / (2x)|] at [x =
   erfinv p], 1 at 0. *)
let erfinv_kappa =
  {
    u =
      (fun p ->
        let x = Nx.erfinv p in
        let zero = Nx.equal_s p 0. in
        let k =
          Nx.div
            (Nx.mul_s (Nx.mul p (Nx.exp (Nx.square x))) (Float.sqrt Float.pi))
            (Nx.mul_s (Nx.where zero (Nx.ones_like x) x) 2.)
        in
        Nx.where zero (Nx.ones_like p) (Nx.abs k));
  }

let error_function =
  group "error function"
    [
      unary "erf" ~bound:(everywhere (Ulps 2)) { u = Nx.erf };
      unary "erfinv"
        ~bound:(everywhere (Inverse (4, 8)))
        ~scale:erfinv_kappa { u = Nx.erfinv };
    ]

let () = exit (run "nx special" [ error_function ])
