(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kind =
  | K0 of Nx_kernel.Prog.op0
  | K1 of Nx_kernel.Prog.op1
  | K2 of Nx_kernel.Prog.op2
  | K3 of Nx_kernel.Prog.op3

type backend = {
  name : string;
  kernels : (module Nx_kernel.S);
  device : Rig.t;
  computes : kind -> Nx_array.Dtype.any -> bool;
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

(* nx_cpu.mli's promise: [Copy] and [Cast] at every dtype; [Fill] and
   [Where] at every dtype of a byte or more; the other kinds of no, two and
   three operands at the base dtypes. *)
let base (Nx_array.Dtype.Any dt) =
  let module D = Nx_array.Dtype in
  match dt with
  | D.Float32 | D.Float64 | D.Int8 | D.Uint8 | D.Int16 | D.Uint16 | D.Int32
  | D.Uint32 | D.Int64 | D.Uint64 | D.Bool ->
      true
  | _ -> false

let cpu_computes k (Nx_array.Dtype.Any dt as d) =
  match k with
  | K1 (Copy | Cast) -> true
  | K1 (Unary _ | Bitcast) -> false
  | K0 (Fill _) | K3 Where -> Nx_array.Dtype.bits dt >= 8
  | K0 (Iota _) | K2 _ | K3 Fma -> base d

let cpu target =
  {
    name = "cpu/" ^ target;
    kernels = (module Nx_cpu);
    device = Rig.host;
    computes = cpu_computes;
    around = (fun f -> with_target target f);
  }

let backends = List.map cpu (targets ())
