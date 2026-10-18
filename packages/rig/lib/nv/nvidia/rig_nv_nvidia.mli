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
    + The first open that succeeds in creating the process's client of the
      kernel driver keeps it for the process, made from that open's
      [root/dev]. The client is three open files and a reservation of the
      process's addresses from 384 GiB to [2{^40}], where the GPUs' memory
      that the host maps too lies. Every later open uses it, whatever its
      [root] and whether or not that open succeeds.
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
    as for unmapped ones. The RM also bounds the host memory mapped for a GPU
    at once, which this path maps in 4 KiB pages: past the bound it refuses
    every further mapping ([NV_ERR_NO_MEMORY]), however the memory is split,
    and {!Rig_nv.val-map_host} returns [None] until mappings are freed.

    {!buses}, {!count} and an opener's [make] read the machine's files under
    the directory [root] (defaults to ["/"]): its PCI functions under
    [root/sys/bus/pci], and the kernel driver's files under [root/dev]. A
    container may see its host's files under a [root] of its own. Every
    [root] a program passes must show the same kernel driver. Listing GPUs
    changes nothing on the machine.

    Any domain may call any function, at the same time as others. *)

val buses : ?root:string -> unit -> (string list, string) result
(** [buses ~root ()] is the bus addresses of this machine's NVIDIA GPUs
    ({!Rig_nv.is_gpu}) under [root] (defaults to ["/"]), in bus order: by
    domain, bus, device and function, each as a number. GPU [i] is the
    [i]th. It is [Ok []] where [root/sys/bus/pci] does not exist, such as off
    Linux.

    A function whose files cannot be read, as while the kernel removes it, is
    left out.

    [Error why] if the machine's PCI functions cannot be listed, such as
    when the process has no file left. *)

val count : ?root:string -> unit -> (int, string) result
(** [count ~root ()] is the number of {!buses}, or their [Error]. A machine
    with no NVIDIA GPU is [Ok 0]. *)

val gpu : ?root:string -> int -> Rig_nv.t Rig_edge.opener
(** [gpu ~root i] is the opener of GPU [i] under [root] (defaults to
    ["/"]), named ["NV"] for [0] and ["NV:i"] otherwise. Its [make] opens the
    GPU that is [i]th when it runs, and changes only the kernel driver's
    state for this process.

    [make] returns [Error msg], having given back what it took for GPU [i],
    if:
    - the machine has no GPU [i] ({!count}), saying how many there are;
    - the machine's PCI functions cannot be read, saying why;
    - the kernel driver does not hold GPU [i], or cannot be opened;
    - the kernel driver's release is none of those {!Rig_nv.make} accepts;
    - a device of GPU [i] is open and not stopped;
    - the kernel driver refuses an object or memory, with its refusal.

    Raises [Invalid_argument] if [i < 0]. [make] raises it, having opened
    nothing, from a callback inside a lock ([Rig]'s domains section). *)
