(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the remote bench and its two-machine measurement share: the floors in
    C, the caller's half of a rail's run, and the request the rows time.

    Every function that waits releases the runtime. A failure crosses as a
    negated code: errno, or WSAGetLastError on Windows. *)

(** {1:sockets Sockets} *)

external tune : Unix.file_descr -> unit = "rig_remote_bench_tune"
(** [tune fd] sets on [fd] the options a link sets on its socket: no delay for
    small segments and, where the system needs an option for it, no SIGPIPE. *)

(** {1:floors Floors} *)

external ask : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_ask"
(** [ask fd buf out back] sends [out] bytes of [buf] on [fd] and receives [back]
    into it: [0], [1] if the stream ended, or a negated code. *)

external echo : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_echo"
(** [echo fd buf in out] answers each [in] bytes received on [fd] with [out]
    bytes, through [buf], until the stream ends: [1] then, or a negated code. *)

external stream_open : Unix.file_descr -> Unix.file_descr -> int -> nativeint
  = "rig_remote_bench_stream_open"
(** [stream_open out in n] is a stream of [n > 0] bytes a run from [out] to
    [in], the two ends of one connection, whose own thread receives each run
    whole; or [0n] if memory or a thread ran out. *)

external stream_run : nativeint -> int = "rig_remote_bench_stream_run"
(** [stream_run st] sends a run's bytes and waits until they arrived: [0], [1]
    if the stream ended, or a negated code. *)

external stream_close : nativeint -> unit = "rig_remote_bench_stream_close"
(** [stream_close st] shuts the connection down, ends the receiving thread and
    frees [st]. The caller closes the sockets. *)

(** {1:rails Rail runs} *)

val rail_run : Rig_remote_abi.end_ -> Rig_remote_abi.end_ -> int -> int
(** [rail_run sender receiver c] advances [sender]'s [ready] to [c] through its
    C function and waits until [receiver]'s [arrived] reaches [c]: [0], or [1]
    if the job failed meanwhile. *)

(** {1:requests Requests} *)

val alloc : bool Rig_remote_proxy.Wire.request
(** [alloc] is the request the rows time: 4 KiB of an agent's device. *)

val request_bytes : int
(** [request_bytes] is the bytes of {!alloc}'s frame: 35. *)

val answer_bytes : int
(** [answer_bytes] is the bytes of its answer's frame: 11. *)
