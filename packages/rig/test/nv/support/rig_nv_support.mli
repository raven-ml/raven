(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NV suites share. *)

(** {1:gpu The GPU} *)

include Rig_gpu_support.Conformance with module D = Rig_nv
(** GPU [0], opened on its {!type-path}. The second device is GPU [1], where
    it is on the same path. *)

(** The type for the ways the suites open a GPU. *)
type path =
  | Kernel  (** Through {!Rig_nv_nvidia}, for a GPU NVIDIA's driver holds. *)
  | Pci
      (** Through {!Rig_nv_pci}, for a GPU unbound or on [vfio-pci] with the
          GSP's firmware found ({!firmware}). Behind an IOMMU that
          translates, an unbound GPU's open fails, naming why. *)

val path : unit -> path option
(** [path ()] is GPU [0]'s path, from the driver bound to it, or [None] if
    neither opens it. *)

val opener : int -> Rig_nv.t Rig_edge.opener
(** [opener i] opens GPU [i] on GPU [0]'s path. *)

val firmware : unit -> string list option
(** [firmware ()] is the directories an open over PCI reads NVIDIA's firmware
    from, the variable [RIG_NV_PCI_FIRMWARE]'s, separated by [:], then rig's
    firmware directory and [/lib/firmware], if one of them holds the GSP's
    firmware with its pinned digest. *)

(** {1:work Work through rig} *)

module Sub := Rig.Submission

val run : t -> Sub.part array -> unit
(** [run t ps] submits [ps] and waits for their value. *)

val words : int array -> Sub.part
(** [words ws] is the ring words [ws], which the device's compute queue
    runs. *)

val copy : dst:Rig.Buffer.t -> Rig.Buffer.t -> Sub.part
(** [copy ~dst src] is a copy of [src] into [dst]. *)

val shared : t -> int -> Rig.Buffer.t * int
(** [shared t n] is [(b, a)]: [n] bytes of host memory at [a], which [b], a
    buffer of [t]'s, borrows. *)

val watchdog : string -> (unit -> 'a) -> 'a
(** [watchdog what f] is [f ()]. If [f] has not returned after 10 seconds, it
    prints [what] and ends the process: [f] blocks in C, where no timeout of the
    test reaches it. *)

(** {1:edge The C entries} *)

module Edge : sig
  val submit : Rig_nv.t -> v:int -> (int * int array) array -> unit
  (** [submit g ~v ps] hands [g]'s C room check and submit the parts [ps],
      each [(queue, words)]: the ring words [words], each integer's low 32
      bits, on the queue of index [queue], as the value [v] with no waits. It
      expresses work rig does not, such as words on the copy queue. The caller
      numbers the values of a device rig never took ({!driver}). Fails the
      test if the room check answers other than [RIG_FITS] or the submit
      fails. *)

  val room : Rig_nv.t -> (int * int array) array -> string
  (** [room g ps] is what [g]'s C room check answers for the parts [ps], as
      {!submit} hands them: ["FITS"], ["LATER"], ["NEVER"] or ["GROW"]. A
      ["GROW"] leaves its request for {!Rig_nv.grow} on the calling
      thread. *)
end

(** {1:host Host memory} *)

val host : Rig_nv.region -> int
(** [host r] is [r]'s host address. Fails the test if the host does not address
    [r]. *)

val address : Rig_nv.region -> int
(** [address r] is [r]'s GPU address. *)

val word : Rig_nv.t -> Rig_nv.region
(** [word g] is [g]'s timeline word. *)

val mapped : Rig_nv.t -> int -> Rig_nv.region
(** [mapped g n] is [n] bytes of [g]'s [Mapped] memory, or of its [Pinned]
    memory where it has no [Mapped] memory left, as rig allocates. *)

val pattern : int -> int -> int -> unit
(** [pattern a n seed] writes the [n] bytes at [a] with the pattern of [seed]:
    byte [i] a hash of [i] and [seed], so that a byte shifted, repeated or left
    from another pattern differs. *)

val mismatch : int -> int -> int -> int
(** [mismatch a n seed] is the index of the first of the [n] bytes at [a] that
    differs from the pattern of [seed], or [-1] if none does. *)

(** {1:launches Kernels and launches} *)

val fixture : ?dir:string -> string -> string
(** [fixture ~dir f] is the contents of the file [f] of [dir] (defaults to
    ["fixtures"]). *)

type kernels
(** The type for a cubin loaded on a device. *)

val kernels : ?dir:string -> ?file:string -> t -> kernels
(** [kernels ~dir ~file t] is the cubin [file] (defaults to
    ["kernels_sm89.cubin"]) of [dir] ({!fixture}), loaded on [t] by
    {!Rig.Image.load}.
    ["kernels_sm89.cubin"] holds the kernels of ["kernels.cu"]. *)

val image : Rig_nv.t -> string -> Rig_nv.image * Rig_nv.region * string
(** [image g bin] is the cubin [bin] loaded on [g] by the driver alone: the
    image, its new [Device] code region and the bytes to write there. *)

type launches
(** The type for driver memory that holds launches: their descriptors, constant
    banks and ring segments. *)

val launches : Rig_nv.t -> launches
(** [launches g] is new memory for launches on [g]. *)

val launch :
  launches -> kernels -> string -> blocks:int -> int list -> int array
(** [launch l k f ~blocks args] is the ring entry, as two words, of a segment
    that schedules the kernel [f] of [k] over [blocks] blocks of 256 threads,
    with the 64-bit parameters [args], after making the local memory of [l]'s
    device serve it. The memory it uses is [l]'s until {!reset}. *)

val launch_at :
  launches ->
  code:int ->
  string ->
  string ->
  blocks:int ->
  int list ->
  int array
(** [launch_at l ~code bin f ~blocks args] is {!launch} for the kernel [f] of
    the cubin [bin] whose image the caller placed at the address [code]. *)

val segment : launches -> Rig_nv_abi.Template.t -> int array
(** [segment l p] is the ring entry, as two words, of a segment of the words
    [p]. *)

val reset : launches -> unit
(** [reset l] makes [l]'s memory free for new launches. The caller resets [l]
    once no work uses its launches. *)

val free_launches : launches -> unit
(** [free_launches l] frees [l]'s memory. *)
