(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx.cpu's suite needs beyond the library's interface. *)

val targets : unit -> string list
(** [targets ()] names the target tables the host runs, the one the kernels run
    unless {!with_target} says otherwise last. *)

val with_target : string -> (unit -> 'a) -> 'a
(** [with_target t f] is [f ()] with the kernels running the table [t], one of
    {!targets}. *)
