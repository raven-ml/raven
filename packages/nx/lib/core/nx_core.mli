(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Core modules for [nx].

    This module re-exports core building blocks used by backends and the
    high-level [Nx] frontend. *)

module Dtype = Dtype
(** Tensor element dtypes. *)

module Shape = Shape
(** Concrete shape operations. *)

module Symbolic_shape = Symbolic_shape
(** Symbolic shape expressions. *)

module View = View
(** Strided tensor views. *)

module Backend_intf = Backend_intf
(** Backend interface used by frontend functors. *)

module Rng = Rng
(** RNG key utilities. *)

module Make_frontend = Frontend.Make
(** Frontend functor parameterized by a backend implementation. *)
