(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Host memory as windows, and fake GPUs, for the suites. *)

val window : int -> Rig_pci.Window.t
(** [window n] is a window on [n] new zeroed bytes of host memory, kept alive
    for the process. *)

(** {1:gpus Fake GPUs} *)

type gpu = {
  machine : Rig_pci.Machine.t;
      (** A machine reached through a fake transport whose one function is an
          NVIDIA GPU at [0000:01:00.0]. Its BAR 0 is the transport's first
          [16 MiB], zeroed, where the GPU's registers read as written. *)
  far : int;  (** The far machine of the transport ({!Rig_pci_support.far}). *)
  regs : Rig_pci.Window.t;  (** BAR 0. *)
  released : int ref;  (** The number of releases of the GPU's function. *)
}
(** The type for fake GPUs. *)

val gpu : ?vendor:(unit -> int) -> unit -> gpu
(** [gpu ()] is a new fake GPU whose vendor ID reads [vendor ()] (defaults to
    NVIDIA's, [0x10de]). *)
