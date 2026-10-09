(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The path through Linux's RDMA subsystem, for any driver: the machine's RDMA
    devices as its files list them, and a device's verbs file made into a
    {!Rig_mlx5.type-path}. The path carries the driver data its caller gives as
    bytes, so the same calls make the objects of a device of another driver,
    with that driver's data. *)

(** {1:files The machine's files} *)

val devices : string -> string list
(** [devices root] is the names of the RDMA devices under [root], in name order,
    [[]] if [root/sys/class/infiniband] does not exist. *)

val exists : string -> string -> bool
(** [exists root name] is [true] iff the RDMA device [name] exists under [root].
*)

val driver_name : string -> string -> string option
(** [driver_name root name] is the kernel driver of the PCI function of the RDMA
    device [name] under [root], such as ["mlx5_core"], or [None] if it has no
    PCI function. *)

(** {1:path The path} *)

val open_ :
  driver:int -> root:string -> string -> (Rig_mlx5.path, string) result
(** [open_ ~driver ~root name] opens the verbs file of the RDMA device [name]
    under [root], whose driver the kernel knows by [driver] (an [RDMA_DRIVER_*]
    value), and is the path through it. It is [Error msg] if the device has no
    verbs file, or it cannot be opened, naming the file and the reason. *)
