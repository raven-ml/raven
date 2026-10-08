(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the AMD suites share. *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once the process holds the machine's GPU lock, which
    it keeps until it exits, or at once if the machine has no AMD GPU. The lock
    is [flock] on [/tmp/raven-rig-gpu.lock], the file every suite and bench that
    acts on a GPU of the machine locks; its holder writes its executable and
    process id into it. A suite calls [hold_gpu] before [Windtrap.run], so that
    the wait counts against no test's timeout, and {!gpu} calls it again. A
    bench calls it before [Thumper.run], so that the workers it forks run under
    the lock: [hold_gpu] starts no vendor library, which a process must not
    start before it forks. It returns at once, taking nothing, if the variable
    [RIG_GPU_LOCK_HELD] is set: the process that started this one holds the
    lock for it, as a timing run takes it before the host's timing locks. A
    machine counts a GPU as {!gpus} does.

    Raises [Failure] naming the holder if another process still holds the lock
    after 300 s, or naming the errno if the file cannot be locked. *)

val driverless : unit -> bool
(** [driverless ()] is [true] iff {!open_gpu} opens the GPU with no kernel
    driver: the variable [RIG_AMD_PCI_FIRMWARE] is set. *)

val gpus : unit -> int
(** [gpus ()] is the number of AMD GPUs the suites can open: amdgpu's
    ({!Rig_amd_amdgpu.count}), or, where the variable [RIG_AMD_PCI_FIRMWARE]
    is set, the machine's ({!Rig_amd_pci.count}). *)

val open_gpu : unit -> (Rig_amd.t, string) result
(** [open_gpu ()] opens GPU [0] through amdgpu ({!Rig_amd_amdgpu.open_}), or,
    where the variable [RIG_AMD_PCI_FIRMWARE] lists directories separated by
    [:], with no kernel driver, its firmware read from them
    ({!Rig_amd_pci.open_}): the GPU detached ({!Rig_amd_pci.detach}) and the
    process privileged to take it. *)

val gpu : unit -> Rig_amd.t
(** [gpu ()] is AMD GPU [0], opened by {!open_gpu} and handed to rig
    under a name of its own ({!rig}), after stopping the device an earlier
    {!gpu} opened if no {!stop} stopped it, as a failed test leaves it, while
    the process holds the machine's GPU lock ({!hold_gpu}). It skips the test if
    the machine has no AMD GPU. *)

val rig : Rig_amd.t -> Rig.t
(** [rig g] is rig's device over [g], which {!gpu} opened.

    Raises [Invalid_argument] if [g] was stopped, lost or opened otherwise. *)

val submit : Rig_amd.t -> Rig.Submission.part array -> int
(** [submit g ps] submits [ps] through rig on [g] and is their value. Raises
    what {!Rig.submit} raises; once it raised {!Rig.Lost}, {!with_gpu} does not
    stop [g] again. *)

val stop : Rig_amd.t -> unit
(** [stop g] is [Rig_amd.stop g]. Tests stop the devices {!gpu} opened through
    it. *)

val with_gpu : (Rig_amd.t -> 'a) -> 'a
(** [with_gpu f] is [f (gpu ())], the device stopped after unless rig lost it.
*)

val wait : Rig_amd.t -> int -> unit
(** [wait g v] returns once [g]'s timeline word reaches [v], sleeping on the
    device between reads. *)

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for about [ms] milliseconds of CPU time, and
    asserts under [w] that each read is [x]. *)

val now_ns : unit -> int
(** [now_ns ()] is the monotonic clock, in nanoseconds: the clock a device's
    [hang_ms] bound counts. *)

val read : int -> int -> string
(** [read a n] is the [n] bytes of host memory at [a]. *)

val write : int -> string -> unit
(** [write a s] writes [s] to host memory at [a]. *)

val pages : int -> int
(** [pages n] is the host address of [n] new zeroed bytes on pages of their own.
*)

val free_pages : int -> int -> unit
(** [free_pages a n] gives back the [n] bytes at [a] that {!pages} gave. *)

(** {1:fills Fills} *)

type fill
(** The type for fills: a C function and its argument, a buffer of host memory.
*)

val fill :
  ?code:int ->
  ?split:int ->
  Rig_amd.capability ->
  int array ->
  bytes:int ->
  fill
(** [fill ~code ~split c ws ~bytes] is a fill that places the words [ws] with
    [c]'s [place], in two calls, the first of the first [split] words, if
    [0 < split < Array.length ws] (defaults to [0]: one call), then takes
    [bytes] bytes of the argument segment with [c]'s [segment] if [bytes > 0],
    and returns the first failure of these, else [code] (defaults to [0]). *)

val fill_address : fill -> int
(** [fill_address f] is the GPU address of the segment bytes [f] took at its
    last call, or [0] if it took none. *)

val fill_part :
  queue:string ->
  ?after:int array ->
  fill ->
  units:int ->
  bytes:int ->
  Rig.Submission.part
(** [fill_part ~queue ~after f ~units ~bytes] is [f] as a part for rig,
    declaring [units] ring words and [bytes] segment bytes. *)

val words_part :
  queue:string -> ?after:int array -> int array -> Rig.Submission.part
(** [words_part ~queue ~after ws] is the words [ws], each integer's low 32 bits,
    as a part for rig, in a host buffer of their own. *)

(** {1:edge The C entries}

    Work handed to a device's {!Rig_amd.room_entry} and {!Rig_amd.submit_entry}
    directly, for what rig does not express: the room check's answer, a value
    the test numbers itself, waits on any word. *)

module Edge : sig
  type part
  (** The type for parts. *)

  val words : queue:string -> ?after:int array -> int array -> part
  (** [words ~queue ~after ws] is the words [ws] on [queue]. *)

  val fill :
    queue:string -> ?after:int array -> fill -> units:int -> bytes:int -> part
  (** [fill ~queue ~after f ~units ~bytes] is [f] on [queue], declaring [units]
      ring words and [bytes] segment bytes. *)

  val copy : ?after:int array -> dst:int -> src:int -> int -> part
  (** [copy ~after ~dst ~src n] is a copy of [n] bytes from the GPU address
      [src] to [dst], on ["COPY:0"]. *)

  val raw :
    queue:int ->
    ?words:int ->
    ?fill:bool ->
    ?copy:int ->
    ?after:int array ->
    unit ->
    part
  (** [raw ~queue ~words ~fill ~copy ~after ()] is a part on the queue of index
      [queue] with [words] zero words (defaults to none), the support's fill
      function with no argument if [fill], and a copy of [copy] bytes between
      address [0] and itself: a part a device may refuse. *)

  val room : Rig_amd.t -> part array -> [ `Fits | `Later | `Never ]
  (** [room g ps] is what [g]'s room entry answers for [ps]. *)

  val submit :
    Rig_amd.t ->
    v:int ->
    ?waits:(int * int) array ->
    part array ->
    [ `Ok | `Failed of string ]
  (** [submit g ~v ~waits ps] is what [g]'s submit entry answers for [ps] as the
      value [v], after the waits [(a, w)]: the 64-bit word at [a] holds at least
      [w]. *)
end
