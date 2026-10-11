(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ConnectX NICs of this machine, opened through Linux's RDMA subsystem.

    The kernel's [mlx5_ib] driver owns the NIC and shares it with other
    programs. This library opens a NIC through its user verbs file,
    [/dev/infiniband/uverbsN], has the kernel make the NIC's objects with the
    verbs ioctls, and gives them to {!Rig_mlx5}, which drives the NIC
    ({!Rig_mlx5.make}).
    {[
    let nic = Result.get_ok (Rig_mlx5_uverbs.open_ "mlx5_0") in
    Rig_mlx5.link nic (* `Infiniband *)
    ]}

    A NIC is named as the kernel names its RDMA device, such as ["mlx5_0"]. A
    ConnectX function has one port, and this library opens port 1.

    {b The machine's files.} {!names} and {!open_} read this machine's files
    under the directory [root] (defaults to ["/"]): the RDMA devices under
    [root/sys/class/infiniband] and [root/sys/class/infiniband_verbs], and the
    verbs files under [root/dev/infiniband]. Off Linux this machine has none
    of these files, so [names ()] is [[]], and {!open_} and {!path} return
    [Error].

    {b Privileges.} Opening needs read and write access to the NIC's verbs file
    and no other privilege. Registering memory locks its pages, so the process's
    locked-memory limit ([RLIMIT_MEMLOCK]) bounds the bytes it registers unless
    the process has [CAP_IPC_LOCK].

    {b Kernels.} Registering dma-buf memory needs Linux 5.12 or later.

    {b References.}
    - The Linux kernel's
      {{:https://docs.kernel.org/infiniband/user_verbs.html}Userspace verbs
       access} documentation: the verbs files and the pinning of registered
      memory. *)

val names : ?root:string -> unit -> string list
(** [names ~root ()] is the names of this machine's RDMA devices under [root]
    (defaults to ["/"]) whose PCI function the [mlx5_core] driver holds, sorted
    as strings, so ["mlx5_10"] comes before ["mlx5_2"]. It is [[]] where
    [root/sys/class/infiniband] does not exist. Listing them changes nothing on
    the machine. *)

val open_ : ?root:string -> string -> (Rig_mlx5.t, string) result
(** [open_ ~root name] opens port 1 of the NIC [name] under [root] (defaults to
    ["/"]).

    The result is [Error msg] if:
    - no RDMA device [name] exists, or [mlx5_core] does not hold its PCI
      function;
    - it has no verbs file, or its verbs file cannot be opened, [msg] naming the
      file and the reason;
    - the kernel refuses a call, [msg] naming the call and the reason;
    - port 1 is not active, or is an Ethernet port with no RoCE v2 global
      identifier;
    - {!Rig_mlx5.make} returns [Error msg]. *)

val path : ?root:string -> string -> (Rig_mlx5.path, string) result
(** [path ~root name] is a path through the verbs file of the RDMA device
    [name] under [root] (defaults to ["/"]), on port 1, whatever driver holds
    the device. The path passes the driver data it is given to the kernel
    unchanged, and names the driver the kernel lists for [name].
    [open_ ~root name] is {!Rig_mlx5.make} of this path for a NIC that
    [mlx5_core] holds. The caller must close a path it does not make into a
    NIC.

    The result is [Error msg] if no RDMA device [name] exists, it has no verbs
    file, its verbs file cannot be opened ([msg] names the file and the
    reason), or the kernel lists no driver for [name]. *)
