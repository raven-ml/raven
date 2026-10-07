(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The library's initialisation, measured between the initialisers of the probes
   linked around it (support/dune): its tables are static data, and the one
   value it computes is Capability.key. Reading the tables builds nothing that
   outlives the reading. *)

open Windtrap
open Device_amd_abi
module B = Device_amd_abi_before
module S = Device_amd_abi_support

let timeout = S.timeout

(* The words a reading of B.allocated costs, and what initialising
   device_amd_abi allocated besides. *)
let reading = B.before -. B.start
let init = Device_amd_abi_after.after -. B.before -. reading

let key_words () =
  let a = B.allocated () in
  let b = B.allocated () in
  ignore (Sys.opaque_identity (Type.Id.make () : int Type.Id.t));
  let c = B.allocated () in
  c -. b -. (b -. a)

(* Every reader of a table, on each generation. *)
let read_tables () =
  ignore (Sys.opaque_identity (Code_object.of_string ""));
  List.iter
    (fun gc ->
      let g = S.gpu gc in
      ignore (Sys.opaque_identity (Gpu.processor g));
      ignore (Sys.opaque_identity (Register.registers g));
      ignore (Sys.opaque_identity (Scratch.tmpring g 0));
      ignore (Sys.opaque_identity (Scratch.descriptor g ~base:0 0));
      ignore (Sys.opaque_identity (Pm4.run g []));
      ignore (Sys.opaque_identity (Sdma.copy g ~dst:0 ~src:0 1));
      ignore (Sys.opaque_identity (Thread_trace.start g ~size:4096 Fun.id));
      ignore (Sys.opaque_identity (Thread_trace.stop g Fun.id));
      ignore
        (Sys.opaque_identity (Thread_trace.waves g (String.make 64 '\001'))))
    S.families

let live () =
  Gc.full_major ();
  (Gc.stat ()).live_words

let tests =
  group ~timeout "initialisation"
    [
      xfail
        ~reason:
          "initialisation allocates 19 words where Capability.key takes 5: \
           Packet's exception Hole takes 3 and Thread_trace's bits, a partial \
           application, 6"
        (test "initialising the library allocates only Capability.key"
           (fun () -> equal float_exact (key_words ()) init));
      test "reading every table keeps no word live" (fun () ->
          read_tables ();
          let l0 = live () in
          read_tables ();
          equal int l0 (live ()));
    ]

let () = exit (run "device_amd_abi.init" [ tests ])
