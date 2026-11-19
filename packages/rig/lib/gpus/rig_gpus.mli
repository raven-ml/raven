(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPUs of this machine, and its devices by name.

    A {e path} is one way rig reaches a vendor's GPUs, a library of its own:
    through the vendor's runtime, through its kernel driver, or through no
    driver. One card is a GPU of each path that sees it, and its PCI bus
    address says which GPUs of the paths are one card. A path lists its GPUs
    only when asked, since listing loads the vendor's library or reads the
    system's devices.

    A program that takes whichever GPU is here opens the first that opens
    ({!first}):
    {[
    let gpu = Rig_gpus.first ()
    ]}
    and a program that takes the device its user names opens it by its name
    ({!section-names}):
    {[
    let gpu = Rig_gpus.open_ "cuda:1"
    ]} *)

type gpu = private {
  name : string;  (** Its device's name once open, such as ["CUDA:1"]. *)
  bus : string option;
      (** Its PCI bus address, such as ["0000:01:00.0"], the card's identity
          across paths, where its path names one. Metal's GPUs have none. *)
  open_ : unit -> (Rig.t, string) result;
      (** Opens the GPU with {!Rig.open_}. *)
}
(** The type for a GPU a path sees. *)

type path = private {
  kind : string;
      (** Its name, which is its GPU [0]'s device's, such as ["CUDA"]. *)
  gpus : unit -> (gpu list, string) result;
      (** Lists its GPUs on this machine in its own order, or [Error why] if
          it cannot. Each call lists them again and opens no GPU. *)
}
(** The type for a path of rig. *)

val paths : ?firmware:string list -> unit -> path list
(** [paths ~firmware ()] is each path of rig. It lists no GPU. [firmware]
    (defaults to [[]]) is where the driver-less paths look for firmware
    first ({!Rig_nv_pci.gpu}, {!Rig_amd_pci.gpu}).

    Paths come in this order:
    + ["METAL"] ({!Rig_metal}) and ["CUDA"] ({!Rig_cuda}), through the vendor's
      runtime;
    + ["NV"] ({!Rig_nv_nvidia}) and ["AMD"] ({!Rig_amd_amdgpu}), through the
      vendor's kernel driver;
    + ["NV-PCI"] ({!Rig_nv_pci}) and ["AMD-PCI"] ({!Rig_amd_pci}), through no
      driver: the GPUs no kernel driver holds.

    The vendor's runtime comes first, then its kernel driver, then no driver:
    each step down, rig does more of the vendor's work itself. One card
    appears under each path that sees it, with one [bus]. A GPU that a kernel
    driver holds appears under its runtime path and its kernel path, and
    under its driver-less path, whose open then fails. A program that uses
    every card opens one GPU per [bus]. Opening has a cost and takes the
    GPU, so a program opens only the GPUs it uses.

    A path's [gpus] raises [Invalid_argument] as {!Rig_cuda.buses} does, from
    a callback inside a lock ([Rig]'s domains section). *)

val first : ?firmware:string list -> unit -> (Rig.t, string) result
(** [first ~firmware ()] is the first GPU of {!paths} that opens, or
    [Error why], where [why] names each path's refusal and each GPU's. It
    lists a path only once every path before it has opened no GPU.

    Raises [Invalid_argument] as a path's [gpus] does. *)

(** {1:names Names}

    A {e name} is a device's name, as {!Rig.name} gives it: [KIND] or
    [KIND:N], where [KIND] is ["CPU"] or a path's kind ({!paths}), in any
    case, and [N] is a decimal number with no sign and no leading zero.
    [KIND:0] names [KIND]. So:
    - ["CPU"] and ["CPU:0"] name {!Rig.host}, and ["CPU:N"], for [N >= 1],
      the software device of that name ({!Rig.software});
    - ["CUDA"], ["cuda:0"] and ["CUDA:0"] name GPU [0] of the path of kind
      ["CUDA"], and ["CUDA:1"] its GPU [1], in the path's order.

    ["CUDA:01"], ["CUDA:"], ["CUDA:-1"], ["CPU:0:1"] and ["VULKAN"] name no
    device. A name is plain data, so a process hands it to the children it
    forks, where a device does not cross a fork ({!Rig.section-stops}).

    A driver's library opens a device of another name, such as a GPU of a
    driver-less path with its firmware ({!Rig_nv_pci.gpu}), with
    {!Rig.open_}. *)

val open_ : string -> (Rig.t, string) result
(** [open_ name] is the device [name] names, opened: {!Rig.host},
    {!Rig.software}[ name], or {!Rig.open_} of GPU [N]'s opener of its path.
    For every device [d] it opens, [open_ (Rig.name d)] is [Ok d].

    It lists no path: the opener's own [Error] says when the path has no
    GPU [N]. Opening a GPU starts its driver in this process, so that a
    child this process forks may not open it. A program that forks children
    to use a GPU, such as a benchmark whose cases run in forked workers,
    asks {!opens} instead.

    [Error why] if [name] is no name, [why] listing the kinds, or with the
    open's [Error]. Raises as {!Rig.open_} does. *)

val open_all : string -> (Rig.t list, string) result
(** [open_all names] is the devices [names] names, separated by commas, in
    order, each opened by {!val-open_} as {!String.trim} leaves it: a
    program's list of devices, as a user writes it, such as ["cuda"] or
    ["CPU:1,CPU:2"]. A program mints a set over them ([Nx.devices]).

    [Error why] with {!val-open_}'s [why] for the first name that opens
    nothing, an empty one included, or naming two names of one device, such
    as ["cpu"] and ["CPU:0"]. The devices opened before it stay open. *)

val opens : string -> (unit, string) result
(** [opens name] is [Ok ()] if {!val-open_}[ name] succeeds in a child of
    [Unix.fork], and [Error why] with the child's [why] otherwise. The child
    opens the device, writes its answer on a pipe and ends with
    [Unix._exit], running no [at_exit]. This process starts no driver, so
    the children it forks later open the device as that child did.

    [Error why] if [name] is no name, as {!val-open_} says, before any fork; if
    the child dies, [why] says how. Raises as [Unix.fork] does: [Failure]
    if the process has ever spawned a domain, and wherever the system has no
    [fork], such as on Windows. A process that cannot fork calls {!val-open_}.
    Call it before the process starts a thread: in a child of a process with
    several threads, only async-signal-safe calls are defined. *)
