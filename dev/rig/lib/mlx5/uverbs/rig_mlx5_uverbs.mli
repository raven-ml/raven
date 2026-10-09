(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** ConnectX NICs of this machine, opened through Linux's RDMA subsystem.

    The kernel's [mlx5_ib] driver owns the NIC and shares it with other
    programs. This library opens a NIC through its user verbs file,
    [/dev/infiniband/uverbsN], makes the NIC's objects with the kernel's verbs
    ioctls, carrying the driver data {!Rig_mlx5} gives as bytes, and gives them
    to {!Rig_mlx5}, which drives the NIC ({!Rig_mlx5.make}):
    {[
    let nic = Result.get_ok (Rig_mlx5_uverbs.open_ "mlx5_0") in
    Rig_mlx5.link nic (* `Infiniband *)
    ]}

    {b Naming.} A NIC is named as the kernel names its RDMA device, such as
    ["mlx5_0"]. A ConnectX function has one port; this library opens port 1.

    {b The machine's files.} {!names} and {!open_} read this machine's files
    under the directory [root] (defaults to ["/"]): the RDMA devices under
    [root/sys/class/infiniband] and [root/sys/class/infiniband_verbs], and the
    verbs files under [root/dev/infiniband].

    {b Privileges.} Opening needs read and write access to the NIC's verbs file,
    which the default rules of most distributions give every user, and no other
    privilege. Registering memory locks its pages: the process's locked-memory
    limit ([RLIMIT_MEMLOCK]) bounds the bytes it registers.

    {b Kernels.} Linux 5.12 or later: dma-buf regions need its
    [UVERBS_METHOD_REG_DMABUF_MR].

    {b References.}
    - Linux v6.12's [include/uapi/rdma]: [rdma_user_ioctl_cmds.h] (the ioctl's
      header and attributes), [ib_user_ioctl_cmds.h] (objects, methods and
      attributes) and [ib_user_verbs.h] (the commands the ioctl invokes and
      their responses).
    - Linux v6.12's [drivers/infiniband/core]: [uverbs_ioctl.c] (how the kernel
      reads the ioctl), [uverbs_std_types_device.c] (invoking a command),
      [uverbs_cmd.c] and [uverbs_std_types_mr.c].
    - rdma-core v56.0's [libibverbs/cmd_*.c]: the same calls as its library
      makes them. *)

val names : ?root:string -> unit -> string list
(** [names ~root ()] is the names of this machine's RDMA devices under [root]
    (defaults to ["/"]) that the [mlx5] driver holds, in the kernel's name
    order. It is [[]] where [root/sys/class/infiniband] does not exist, such as
    off Linux. Listing them changes nothing on the machine. *)

val open_ : ?root:string -> string -> (Rig_mlx5.t, string) result
(** [open_ ~root name] opens port 1 of the NIC [name] under [root] (defaults to
    ["/"]). The result is [Error msg] if no RDMA device [name] exists, the
    [mlx5] driver does not hold it, its port is not active, has no RoCE v2
    global identifier on Ethernet, if its verbs file cannot be opened, naming it
    and the reason, if the kernel refuses a call, naming it and the reason, or
    with {!Rig_mlx5.make}'s message. *)
