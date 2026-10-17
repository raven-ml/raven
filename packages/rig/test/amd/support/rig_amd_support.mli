(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the AMD suites share. *)

val driverless : unit -> bool
(** [driverless ()] is [true] iff {!open_gpu} opens the GPU with no kernel
    driver: the variable [RIG_AMD_PCI_FIRMWARE] is set. *)

val gpus : unit -> int
(** [gpus ()] is the number of AMD GPUs the suites can open: amdgpu's
    ({!Rig_amd_amdgpu.count}), or, where the variable [RIG_AMD_PCI_FIRMWARE] is
    set, the machine's ({!Rig_amd_pci.count}). *)

val gpu : unit -> Rig_amd.t Rig_edge.opener
(** [gpu ()] describes GPU [0] through the selected kernel or PCI path. *)

val open_gpu : word:nativeint -> (Rig_amd.t, string) result
(** [open_gpu ~word] opens GPU [0] on the timeline word at [word], through
    amdgpu ({!Rig_amd_amdgpu.gpu}), or,
    where the variable [RIG_AMD_PCI_FIRMWARE] lists directories separated by
    [:], with no kernel driver, its firmware read from them
    ({!Rig_amd_pci.gpu}): the GPU detached ({!Rig_amd_pci.detach}) and the
    process privileged to take it. *)

val vram : int -> int option
(** [vram i] is the bytes of GPU memory KFD counts this process as holding
    on AMD GPU [i], over each of the GPU's nodes ([vram_<gpu_id>] under
    [/sys/class/kfd/kfd/proc/<pid>]), or [None] where KFD shows no count,
    as before the process opened KFD or in a container whose process ids
    are not the host's. KFD counts the GPU memory its allocations made and
    its frees gave back. *)

(** {1:gpu The GPU} *)

include Rig_gpu_support.Conformance with module D = Rig_amd
(** GPU [0], opened by {!open_gpu}; the second device is GPU [0] opened again
    through amdgpu, none without a kernel driver. *)

val reached : Rig_amd.t -> int -> unit
(** [reached g v] returns once [g]'s timeline word reaches [v], sleeping on the
    device between reads: the wait of values handed over at the C entries
    ({!Edge}), which rig does not number. *)

val words_part : int array -> Rig.Submission.part
(** [words_part ws] is the words [ws], each integer's low 32 bits, as a part:
    PM4 words, which the device's compute queue runs. *)

(** {1:edge The C entries}

    Work handed to a device's room check and submit ([edge] of {!Rig_amd.facts})
    directly, for what rig does not express: the room check's answer, a value
    the test numbers itself, waits on any word. *)

module Edge : sig
  type part
  (** The type for parts. *)

  val words : queue:string -> ?after:int array -> int array -> part
  (** [words ~queue ~after ws] is the words [ws] on [queue]. *)

  val copy : ?after:int array -> dst:int -> src:int -> int -> part
  (** [copy ~after ~dst ~src n] is a copy of [n] bytes from the GPU address
      [src] to [dst], on ["COPY:0"]. *)

  val launch :
    ?after:int array ->
    Rig_edge.entry ->
    groups:int * int * int ->
    threads:int * int * int ->
    ?shared:int ->
    string ->
    (int * int) list ->
    part
  (** [launch ~after e ~groups ~threads ~shared params refs] is a launch on
      ["COMPUTE:0"] of the function whose entry is [e], over [groups] of
      [threads], its groups taking [shared] bytes of shared memory (defaults to
      [0]), with the parameters [params], in which the 8 bytes at each
      [(at, slot)] of [refs] are an offset into the submit's buffer [slot]
      ({!submit}'s [slots]). *)

  val raw :
    queue:int ->
    ?work:[ `None | `Words of int | `Fill | `Copy of int ] ->
    ?after:int array ->
    unit ->
    part
  (** [raw ~queue ~work ~after ()] is a part on the queue of index [queue] whose
      work is [work] (defaults to [`None], a part of no kind): [n] zero words, a
      fill with a null function, or a copy of [n] bytes between address [0] and
      itself. It is a part a device may refuse. *)

  val room : Rig_amd.t -> part array -> [ `Fits | `Later | `Never ]
  (** [room g ps] is what [g]'s room entry answers for [ps]. *)

  val submit :
    Rig_amd.t ->
    v:int ->
    ?waits:(int * int) array ->
    ?slots:int array ->
    part array ->
    [ `Ok | `Failed of string ]
  (** [submit g ~v ~waits ~slots ps] is what [g]'s submit entry answers for [ps]
      as the value [v], after the waits [(a, w)]: the 64-bit word at [a] holds
      at least [w]. [slots] are the addresses of the buffers its launches' refs
      name (defaults to none). *)
end
