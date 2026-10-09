(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type backend = {
  name : string;
  kernels : (module Nx_kernel.S);
  device : Rig.t;
  computes : Nx_kernel.Prog.op1 list;
  around : 'a. (unit -> 'a) -> 'a;
}

(* nx.cpu's target tables: the host's, and the one its kernels run. *)
external targets : unit -> string list = "nx_kernels_support_targets"
external current : unit -> string = "nx_kernels_support_current"
external use : string -> unit = "nx_kernels_support_use"

let with_target t f =
  let before = current () in
  use t;
  Fun.protect ~finally:(fun () -> use before) f

let cpu target =
  {
    name = "cpu/" ^ target;
    kernels = (module Nx_cpu);
    device = Rig.host;
    computes = [ Copy; Cast ];
    around = (fun f -> with_target target f);
  }

let backends = List.map cpu (targets ())
