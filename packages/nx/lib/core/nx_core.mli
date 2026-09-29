(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Core modules for [nx].

    This module re-exports core building blocks used by backends and the
    high-level [Nx] frontend. *)

module Shape = Shape
(** Concrete shape operations. *)

module View = View
(** Strided tensor views. *)

module Elements = Elements
(** Elements of host buffers as values of a dtype. *)

module Backend_intf = Backend_intf
(** The operations a backend implements. *)
