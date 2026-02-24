(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Instrumentation hooks for tensor operation tracing.

    The frontend consults {!val-current_hook} to wrap operations in spans. Hooks
    can be installed globally with {!val-set_hook} or temporarily with
    {!val-with_hook}. *)

(** {1:events Events} *)

type event = {
  op : string;  (** Operation name. *)
  kind : [ `Info | `Warn | `Error ];  (** Event severity. *)
  msg : string;  (** Human-readable message. *)
  attrs : (string * string) list;  (** Structured event attributes. *)
}
(** The type for emitted instrumentation events. *)

(** {1:hooks Hooks} *)

type hook = {
  enabled : bool;  (** Enables span wrapping when [true]. *)
  with_span :
    'a.
    op:string -> ?attrs:(unit -> (string * string) list) -> (unit -> 'a) -> 'a;
      (** [with_span ~op ?attrs f] executes [f] inside a span. *)
  emit : event -> unit;  (** [emit e] records an event. *)
}
(** The type for instrumentation hooks. *)

val null_hook : hook
(** [null_hook] is a disabled hook with no-op span and emit handlers. *)

val current_hook : hook ref
(** [current_hook] is the globally active hook reference. *)

val set_hook : hook option -> unit
(** [set_hook h] installs [h] as the current hook.

    [set_hook None] installs {!null_hook}. *)

val with_hook : hook option -> (unit -> 'a) -> 'a
(** [with_hook h f] evaluates [f] with temporary hook [h], restoring the
    previous hook even if [f] raises. *)
