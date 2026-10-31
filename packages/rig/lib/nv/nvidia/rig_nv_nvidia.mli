(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs of this machine, opened through NVIDIA's kernel driver.

    This library is the {!Rig_nv} path for a GPU that NVIDIA's Linux kernel
    driver holds. It reaches the GPU's resource manager through the driver's
    control file [/dev/nvidiactl] and the GPU's file [/dev/nvidiaN], and maps
    memory into the GPU's address space through the driver's unified memory
    module, [/dev/nvidia-uvm]. The kernel driver owns the GPU and shares it
    with other programs. The RM reports the faults of the GPU's work
    ({!Rig_nv.section-faults}). While the RM's watchdog runs on the GPU, it
    ends work that holds the GPU too long, and the device's [hang_ms] is
    [None]. The watchdog does not run on a GPU without a display. There the
    device's [hang_ms] is 30 s, and rig loses the device once its work has
    made no progress for that long.

    A program opens GPU [i] ({!Rig_nv.is_gpu}) with its opener:
    {[
    let d = Result.get_ok (Rig.open_ (Rig_nv_nvidia.gpu 0)) in
    Printf.printf "%s\n" (Rig.arch d)
    ]}

    A device's life goes in four steps:
    + The first open, or {!usage}, that succeeds in creating the process's
      client of the kernel driver keeps it for the process: the control
      file and the RM's client. The first open that succeeds in creating the
      process's unified memory keeps it too: two files of the unified memory
      driver and a reservation of the process's addresses from 384 GiB to
      [2{^40}], where the GPUs' memory that the host maps too lies. Every
      later open uses both, whether or not that open succeeds.
    + The first open of a GPU makes the GPU's objects in the RM, which stay
      for the process. Later opens of that GPU reuse them.
    + The open makes the device ({!Rig_nv.make}). A GPU has one device at a
      time, since the unified memory module registers a GPU once per
      process.
    + When the device stops, this path cannot tell whether the GPU's work
      has ended, so the memory the device holds itself returns only if the
      RM freed its channels ({!Rig_nv.section-faults}). The GPU may then be
      opened again.

    [Pinned] and [Mapped] memory lie at the same address for the host and
    the GPU. [Mapped] memory lies in the GPU's BAR1, whose size the kernel
    driver sets. [Device] memory lies at GPU addresses that the process
    reserves for each allocation, with no access, wherever its address space
    has room outside the client's reservation, so it may lie above
    [2{^40}]. Host memory
    that {!Rig_nv.val-map_host} maps lies at a GPU address of its own, since
    the host's address may lie higher. The kernel driver pins those pages
    for writing while they are mapped, for the long term on x86-64. From
    Linux 6.5 on, Linux refuses such a pin of a shared mapping of a file
    whose file system tracks its dirty pages, such as ext4's but not
    tmpfs's, so there {!Rig_nv.val-map_host} returns [None] for those pages,
    as for unmapped ones.

    {!Rig_nv.val-dma} exports the RM object a region lies in as a new
    dma-buf ({!Rig_edge.Dmabuf}), through the GPU's file, at the region's
    offset in it. The dma-buf holds the object until its last reference
    goes, freed or not. It is [Error] where the kernel driver exports none,
    as only NVIDIA's open kernel modules do, for an [fn] that is no function
    under [/sys/bus/pci/devices], and for an object that is no whole number
    of the host's pages. The kernel driver maps the memory for the importer
    where the importer registers the dma-buf, and refuses there a topology
    it does not support. The export takes the driver's default mapping,
    which serves every importer: through BAR1, or on a GPU coherent with the
    host as the host's own pages, over the GPU's link to the host, so that a
    NIC behind the GPU's own PCIe switch reaches it through the host's
    memory system, at that link's bandwidth, where a BAR1 mapping would
    cross only the switch. The device's facts name the GPU's function as
    [bus].

    {!buses}, {!count}, {!usage} and an opener's [make] read the machine's
    files: its PCI functions under [/sys/bus/pci], and the kernel driver's
    files under [/dev]. Listing GPUs changes nothing on the machine.

    Any domain may call any function, at the same time as others. *)

val buses : unit -> (string list, string) result
(** [buses ()] is the bus addresses of this machine's NVIDIA GPUs
    ({!Rig_nv.is_gpu}), in bus order: by domain, bus, device and function,
    each as a number. GPU [i] is the [i]th. It is [Ok []] where
    [/sys/bus/pci] does not exist, such as off Linux.

    A function whose files cannot be read, as while the kernel removes it, is
    left out.

    [Error why] if the machine's PCI functions cannot be listed, such as
    when the process has no file left. *)

val count : unit -> (int, string) result
(** [count ()] is the number of {!buses}, or their [Error]. A machine
    with no NVIDIA GPU is [Ok 0]. *)

val gpu : int -> Rig_nv.t Rig_edge.opener
(** [gpu i] is the opener of GPU [i], named ["NV"] for [0] and ["NV:i"]
    otherwise. Its [make] opens the GPU that is [i]th when it runs, and
    changes only the kernel driver's state for this process.

    [make] returns [Error msg], having given back what it took for GPU [i],
    but what the kernel driver keeps, which stays for the process and which
    [msg] names ({!Rig_nv.make}), if:
    - the machine has no GPU [i] ({!count}), saying how many there are;
    - the machine's PCI functions cannot be read, saying why;
    - the kernel driver does not hold GPU [i], or cannot be opened;
    - the kernel driver's release is none of those {!Rig_nv.make} accepts;
    - a device of GPU [i] is open and not stopped;
    - the kernel driver refuses an object or memory, with its refusal.

    Raises [Invalid_argument] if [i < 0]. [make] raises it, having opened
    nothing, from a callback inside a lock ([Rig]'s domains section). *)

(** {1:usage Usage} *)

type usage = {
  memory_total : int;  (** The GPU's memory, in bytes. *)
  memory_free : int;
      (** The bytes of it that the RM has free to allocate. The rest is held
          by every process on the GPU, and by the RM and its firmware. *)
  busy : float option;
      (** How busy the GPU's graphics and compute engine was, from [0.0] to
          [100.0]: the mean of the RM's samples of it taken in the window
          {!usage} reads, by the GPU's own clock. [None] if the RM took none
          in that time, or takes none on this GPU. *)
}
(** The type for what a GPU's RM reports of its use, by every process. *)

val usage : window:float -> int -> (usage, string) result
(** [usage ~window i] is GPU [i]'s usage, [busy] over the last [window]
    seconds. The RM keeps samples of about the last 14 seconds, so a longer
    window reads those.

    It reads through the process's client of the kernel driver, the GPU's
    device file and the RM's device and subdevice of the GPU, which its first
    call makes and which stay for the process, as an open's do. It makes no
    channel, address space or mapping, and opens no device. The GPU's device
    file stays open until the process exits: the kernel driver keeps its
    state of the GPU, as its persistence mode does, rather than tear it down
    at the last close. Unbinding the GPU from the driver, as to bind it to
    [vfio-pci], waits until the process exits, and unloading the driver or
    resetting the GPU is refused until then.

    [Error msg] if the machine has no GPU [i]; if the GPU's PCI function is
    not active by its [power/runtime_status], or that cannot be read, since
    reading could resume a GPU the kernel suspended for being idle; if the
    kernel driver does not hold the GPU, cannot be opened, or is of a release
    {!Rig_nv.make} does not accept; or with the RM's refusal of a command.

    Raises [Invalid_argument] if [i < 0] or [window] is not positive (or is
    NaN), or, having read nothing, from a callback inside a lock ([Rig]'s
    domains section). *)

(**/**)

(* [buses_under ~root], [count_under ~root] and [gpu_under ~root] are
   {!buses}, {!count} and {!gpu} for the machine whose files are under the
   directory [root] in place of ["/"], such as a container's view of its host
   or a test's tree. Every [root] a process uses must show the same kernel
   driver. The process's client of the kernel driver is made from the
   [root/dev] of the first open that makes it, and every later open uses it,
   whatever its [root]. *)
val buses_under : root:string -> (string list, string) result
val count_under : root:string -> (int, string) result
val gpu_under : root:string -> int -> Rig_nv.t Rig_edge.opener

type mem
(* The type for the memory of this library's paths. *)

val gpu_with :
  wrap:(mem Rig_nv.path -> mem Rig_nv.path) -> int -> Rig_nv.t Rig_edge.opener
(* [gpu_with ~wrap i] is {!gpu}[ i], whose make hands {!Rig_nv.make} the path
   [wrap p] in place of the path [p] it made, for the suites that fail a call
   of a path on a GPU. *)
