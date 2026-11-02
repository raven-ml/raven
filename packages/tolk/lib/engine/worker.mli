(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Parallel work.

    Compiling a schedule's kernels and searching a kernel's optimizations apply
    one function to many independent inputs. {!map} spreads these applications
    over as many domains as {!Setting.parallel} allows, and returns what a
    serial map returns. {!iter} hands each result, as it is computed, to a
    consumer on the calling domain while others are computed. *)

val map : ('a -> 'b) -> 'a list -> 'b list
(** [map f l] is [List.map f l], computed on the calling domain and on up to
    [p - 1] other domains, where [p] is the value of {!Setting.parallel}. The
    calls running at once share these domains, so the program runs at most
    [p - 1] of them. A call, one made from [f] included, takes the domains left
    free when it starts, one fewer than the elements of [l] at most, and does
    with fewer, down to its calling domain alone, when the runtime refuses to
    spawn them. Every application of [f] sees the settings of the calling domain
    at the time of the call ({!Setting.context}).

    If applications of [f] raise, [map f l] raises the exception, with its
    backtrace, of the first element of [l] whose application raises, as
    [List.map] does. The elements after it may or may not have been applied.
    Either way, [map f l] returns or raises once every application it started
    has ended.

    [f] may run on several domains at once, and must be safe to. *)

val iter : ('a -> 'b) -> ('a -> 'b -> unit) -> 'a list -> unit
(** [iter f g l] applies [g x (f x)] to each element [x] of [l], [g] on the
    calling domain, in the order the applications of [f] end, while up to
    [p - 1] other domains apply [f] to the elements not yet started. [p], and
    how calls share these domains, are as for {!map}. The calling domain applies
    [f] to an element no domain has started when no application has ended for
    it to take, so a call that spawns no domain is [List.iter]'s order of
    applications. Every application of [f] sees the settings of the calling
    domain at the time of the call ({!Setting.context}).

    If an application of [f] or [g] raises, [iter f g l] raises its exception,
    with its backtrace: the first that [g]'s order meets, after which [g] is
    applied to no element. Elements not yet started then may or may not be
    applied [f]. [iter f g l] returns or raises once every application of [f] it
    started has ended.

    [f] may run on several domains at once, and at once with [g], and must be
    safe to. *)
