(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrays over runtime buffers.

    An array is typed elements of one {!Nx_device.Buffer.t} through a strided
    view. It is what nx's kernels compute on: they read their operands' arrays
    and write their results' arrays, on the arrays' device. *)

module Shape = Shape
(** Concrete shape operations. *)

module View = View
(** Strided views: the map from an array's indices to its buffer's elements. *)

module Elements = Elements
(** Elements of host buffers as values of a dtype. *)

module Backend_intf = Backend_intf
(** The operations a backend implements. *)

type ('a, 'b) t = {
  dtype : ('a, 'b) Nx_dtype.t;  (** The type of the elements. *)
  view : View.t;
      (** The array's shape, and where each of its elements lies in [buffer],
          in elements of [dtype]. *)
  buffer : Nx_device.Buffer.t;  (** The memory the elements are in. *)
}
(** The type for arrays of elements of type ['a], stored as ['b]. [buffer]'s
    format is [Nx_dtype.Scalar.of_dtype dtype], and every element the view
    reaches lies in it. *)
