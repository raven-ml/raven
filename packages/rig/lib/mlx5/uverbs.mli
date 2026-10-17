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
(** [open_ ~root name] opens the verbs file of the RDMA device [name] under
    [root] and is the path through it, which names the device's driver as the
    kernel's RDMA netlink lists it. The path works on port 1.

    It is [Error msg] if the device has no verbs file, the file cannot be
    opened, or the kernel lists no driver of [name], [msg] naming the file or
    the device and the reason. *)
