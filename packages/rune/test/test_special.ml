(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's special functions compiled, against the goldens nx's suite holds them to
   (nx's golden/special/, written by its gen/special.py), on the host and on the
   Metal device where the machine has one. Metal computes no float64 and flushes
   float32 subnormals, so it is held to the float32 rows whose arguments and
   value are normal. *)

open Windtrap
open Nx_test.Special

let golden name = "../../nx/test/golden/special/" ^ name ^ ".golden"

type u = { u : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

(* [compiled at u] is [u] compiled, its argument placed at [at] and its result
   read back on the host. *)
let compiled at { u } =
  {
    f =
      (fun a ->
        let x = Nx.place at a.(0) in
        Nx.place Nx.Placement.host (Rune.jit' u x));
  }

let metal = Result.to_option (Nx_metal.get 0)

let unary name ~bound u =
  let host = check ~bound (golden name) (compiled Nx.Placement.host u) in
  let metal =
    match metal with
    | Some m ->
        check
          ~keep:(fun r -> not (subnormal r))
          ~f32_only:true ~bound (golden name)
          (compiled (Nx.Placement.on m) u)
    | None -> [ test "metal" (fun () -> skip ~reason:"no Metal device" ()) ]
  in
  group name [ group "on the host" host; group "on Metal" metal ]

let everywhere b _ = b

let error_function =
  group "error function"
    [
      unary "erf" ~bound:(everywhere (Ulps 2)) { u = Nx.erf };
      unary "erfinv" ~bound:(everywhere (Inverse (4, 8))) { u = Nx.erfinv };
    ]

let () = exit (run "rune special" [ error_function ])
