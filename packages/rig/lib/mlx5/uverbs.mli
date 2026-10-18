(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The path through Linux's RDMA subsystem.

    It lists the machine's RDMA devices from their files, and makes a device's
    verbs file a {!Path.t}. The path passes the driver data its caller gives to
    the kernel as bytes, so the same calls make the objects of a device of
    another driver, given that driver's data. *)

(** {1:files The machine's files} *)

val devices : string -> (string list, string) result
(** [devices root] is the names of the RDMA devices under [root], in name order,
    or [Ok []] if [root/sys/class/infiniband] does not exist. It is
    [Error msg] if that directory cannot be read, [msg] naming it and the
    reason. *)

val exists : string -> string -> bool
(** [exists root name] is [true] iff the RDMA device [name] exists under [root].
*)

val driver_name : string -> string -> string option
(** [driver_name root name] is the kernel driver of the PCI function of the RDMA
    device [name] under [root], such as ["mlx5_core"], or [None] if the device's
    [device/uevent] file names none, as for a device with no PCI function. *)

(** {1:path The path} *)

val open_ : root:string -> string -> (Path.t, string) result
(** [open_ ~root name] is {!Rig_mlx5.val-path}[ ~root name] for a device
    [name] that exists under [root], which it does not check. The kernel's
    RDMA netlink lists the driver the path names. *)
