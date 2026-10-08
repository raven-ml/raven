(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The library's initialisation, measured between the initialisers of the probes
   linked around it (support/dune): it computes no value, and allocates only
   what its declarations make when a module starts: Gpu.key, the exception
   Packet.template stops at a hole with, and the one Cubin.of_string stops at a
   truncated attribute with. *)

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

let exception_words () =
  let a = B.allocated () in
  let b = B.allocated () in
  let exception E in
  ignore (Sys.opaque_identity E);
  let c = B.allocated () in
  c -. b -. (b -. a)

let tests =
  group ~timeout:10. "initialisation"
    [
      test "initialising the library allocates only its declarations" (fun () ->
          equal float_exact (key_words () +. (2. *. exception_words ())) init);
    ]

let () = exit (run "device_nv_abi.init" [ tests ])
