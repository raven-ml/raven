(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The OCaml primitives of [rig_edge.h]: the host's clock, rig's array of
    timeline words, the count of the locks a thread holds and its signal
    mask, which rig, its drivers and their C share.

    Each value calls the C function of [rig_edge.h] it names, declared once
    with the attributes that function allows. A library calls these and
    declares none of the C functions itself. Those marked [[@@noalloc]]
    allocate nothing, raise nothing and run no OCaml code. The other four
    do one of these: {!words} allocates its pair, {!unmask} and
    {!unshield} release and take back the runtime as the outermost pair
    ends, and {!block} runs OCaml code and may raise. A declaration that
    marks any of the four [[@@noalloc]] is unsafe.

    [Rig_edge]'s Drivers section ({!Rig_edge.section-drivers}) states when
    a driver counts a lock and when it blocks signals. *)

(** {1:clock Clock and words} *)

external now : unit -> (int[@untagged])
  = "caml_rig_edge_now_byte" "caml_rig_edge_now"
[@@noalloc]
(** [now ()] is the host's clock ([rig_now_ns]): nanoseconds of the
    system's monotonic clock from an unspecified start, so only differences
    and comparisons of its values mean anything. It is the clock of
    [Rig.Profile.now], and a driver writes a value's times and measures its
    waits on it ({!Rig_edge.section-calls}). *)

external words : unit -> int * int = "caml_rig_edge_words"
(** [words ()] is [(p, n)] where [p] is the host address of rig's array of
    timeline words and [n] its bytes ([rig_words] and [RIG_WORDS_BYTES]),
    for a driver that maps the array whole. *)

(** {1:locks Locks} *)

external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]
(** [holds ()] is the number of locks the calling thread holds
    ([rig_holds]). *)

external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
(** [hold ()] counts one more lock of the calling thread, from before its
    take ([rig_hold]). *)

external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]
(** [unhold ()] counts the give of a lock that {!hold} counted, after the
    give ([rig_unhold]). *)

(** {1:signals Signals}

    {!mask} and {!unmask} are a pair, as are {!shield} and {!unshield}, and
    pairs nest. {!unblock} and {!block} bracket a wait's blocking step inside
    a pair. On Windows none of them blocks a signal. *)

external mask : unit -> unit = "caml_rig_edge_mask" [@@noalloc]
(** [mask ()] blocks on the calling thread every signal but those a fault
    raises ([rig_mask]). *)

external unmask : unit -> unit = "caml_rig_edge_unmask"
(** [unmask ()] ends a {!mask} ([rig_unmask]). The outermost gives the
    thread back the mask it had before the pair, then releases and takes
    back the runtime, which re-arms the next poll point. A signal that came
    during the pair runs its handler there. *)

external shield : unit -> unit = "caml_rig_edge_shield" [@@noalloc]
(** [shield ()] starts a pair as {!mask} does, inside which {!unblock}
    gives nothing back, for code that must take no signal even in its waits,
    such as a finaliser or an exit handler ([rig_shield]). *)

external unshield : unit -> unit = "caml_rig_edge_unshield"
(** [unshield ()] ends a {!shield} as {!unmask} ends a {!mask}, re-arming
    the runtime at the outermost ([rig_unshield]). *)

external unblock : unit -> unit = "caml_rig_edge_unblock" [@@noalloc]
(** [unblock ()] gives the calling thread back the mask it had before the
    outermost {!mask}, right before a wait's blocking step ([rig_unblock]).
    Outside every pair, and inside a {!shield}, it does nothing. *)

external block : unit -> unit = "caml_rig_edge_block"
(** [block ()] runs what a poll point runs (the handlers of the signals
    pending, finalisers, [Gc.Memprof] callbacks, a switch to another
    thread), then blocks again what {!unblock} gave back ([rig_block]). A
    wait calls it right after its blocking step, whether the step returned
    or raised, so that a signal that came during the wait raises its
    exception there. Outside every pair it only runs what a poll point
    runs.

    Raises the exception that the code it runs raises, such as a signal's
    handler. *)
