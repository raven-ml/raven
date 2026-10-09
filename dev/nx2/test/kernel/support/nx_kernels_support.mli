(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernel libraries as nx.kernel's suite runs them. *)

type backend = {
  name : string;  (** The backend's name in the suite, as ["cpu/base"]. *)
  kernels : (module Nx_kernel.S);  (** Its kernels. *)
  device : Rig.t;  (** The device whose memory its laws compute on. *)
  computes : Nx_kernel.Prog.op1 list;
      (** The kinds of one operand it states it computes: the suite fails when
          it declines one. *)
  around : 'a. (unit -> 'a) -> 'a;
      (** [around f] is [f ()] run as the backend: nx.cpu's under its target
          table. *)
}
(** The type for kernel libraries in one configuration. *)

val backends : backend list
(** [backends] is nx.cpu under each target table the host runs, base first. *)
