(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The library's initialisation, measured between the initialisers of the probes
   linked around it (support/dune): its tables are static data, and the one
   value it computes is Gpu.key. *)

open Windtrap
module B = Device_nv_abi_before

(* The words a reading of B.allocated costs, and what initialising device_nv_abi
   allocated besides. *)
let reading = B.before -. B.start
let init = Device_nv_abi_after.after -. B.before -. reading

let key_words () =
  let a = B.allocated () in
  let b = B.allocated () in
  ignore (Sys.opaque_identity (Type.Id.make () : int Type.Id.t));
  let c = B.allocated () in
  c -. b -. (b -. a)

let tests =
  group ~timeout:10. "initialisation"
    [
      xfail
        ~reason:
          "Method computes interrupt at initialisation: 8 words besides \
           Gpu.key's 5"
        (test "initialising the library allocates only Gpu.key" (fun () ->
             equal float_exact (key_words ()) init));
    ]

let () = exit (run "device_nv_abi.init" [ tests ])
