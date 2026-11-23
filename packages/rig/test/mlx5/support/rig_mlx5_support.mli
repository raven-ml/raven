(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the suites on this machine's NIC share. *)

val nic_name : unit -> string
(** [nic_name ()] is the first NIC the mlx5 driver holds whose port 1 carries RDMA,
    as the kernel's files say before anything opens it, and whose verbs file
    the process reaches. A port carries RDMA on InfiniBand, or on Ethernet
    with a RoCE v2 global identifier: a ConnectX function whose RoCE is off,
    as a cloud machine's virtual function, lists an Ethernet port with none,
    which {!Rig_mlx5.open_} refuses. The verbs file is the character device
    [/dev/infiniband/uverbsN] that [/sys/class/infiniband_verbs] names for
    the NIC: a container may show the host's NICs in sysfs and lack it.

    Skips the running test ({!Windtrap.skip}) where no NIC qualifies, saying
    why. A NIC that qualifies and fails to open fails its test. *)
