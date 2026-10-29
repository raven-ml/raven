(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HDUs from bytes, checksums, and writing files: the implementation of
    {!Ymir_fits.Fits.v}, {!Ymir_fits.Fits.with_header},
    {!Ymir_fits.Fits.verify} and {!Ymir_fits.Fits.write}. *)

val v : Header.t -> (int, Nx.int8_unsigned_elt, Nx.host) Nx.t -> Hdu.t
val with_header : Header.t -> Hdu.t -> Hdu.t
val verify : Hdu.t -> (unit, string) result
val write : string -> Hdu.t list -> (unit, string) result
