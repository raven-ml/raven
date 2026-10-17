(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Interception: the implementation of {!Nx.Op.intercept} and
    {!Nx.Op.eval}.

    An interception delivers the operations of its fiber that its
    interpreter claims, as effects. A global count of the interceptions live
    on any domain gates every perform. While it is zero, nothing performs an
    effect to find an interception. It is global because a suspended fiber
    may resume on another domain. *)

type interpreter = { run : 'r. 'r Op.t -> 'r; claims : 'r. 'r Op.t -> bool }
(** {!Nx.Op.interpreter}. *)

val intercept : interpreter -> (unit -> 'a) -> 'a
val intercepted : unit -> bool
val eval : 'r Op.t -> 'r

val intercepting : unit -> bool
(** [intercepting ()] is [true] iff an interception is live on any domain.
    It reads one atomic, and a caller tests it before {!perform}. *)

val perform : 'r Op.t -> 'r
(** [perform op] delivers [op] to the interception around the calling fiber,
    and answers it with {!Dispatch.direct} when there is none. *)
