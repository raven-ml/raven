(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices: the implementation of {!Nx.Device}.

    A device is a memory and the backend that computes on it eagerly, if
    any. {!make} and {!with_backend} check that the backend computes on the
    memory, and siblings build devices only through them. *)

type t = private { memory : Rig.t; backend : Nx_backend.t option }
(** {!Nx.Device.t}. [backend] is [None] on a memory nothing computes on
    eagerly, such as a GPU without a backend or the disk. *)

val host : t
val cpu : int -> t
val with_backend : Nx_backend.t -> t -> t
val make : ?backend:Nx_backend.t -> Rig.t -> t
val memory : t -> Rig.t
val name : t -> string
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
