(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images: the implementation of {!Ymir_fits.Fits.Image}, and the element
    and byte-order conversions that tables share with it. *)

type t
(** {!Ymir_fits.Fits.Image.t}. *)

val of_hdu : Hdu.t -> (t, string) result
val shape : t -> int array
val element : t -> Nx_dtype.Scalar.t
val scaled : t -> bool
val pp : Format.formatter -> t -> unit

val raw :
  ?window:(int * int) array ->
  ('a, 'b) Nx.dtype ->
  Hdu.t ->
  (('a, 'b) Nx.t, string) result

val values :
  ?window:(int * int) array ->
  (float, 'b) Nx.dtype ->
  Hdu.t ->
  ((float, 'b) Nx.t, string) result

val validity :
  ?window:(int * int) array -> Hdu.t -> (Nx.bit_t option, string) result

val hdu : ?tiles:int array -> Header.t -> ('a, 'b) Nx.t -> Hdu.t

val quantized :
  ?tiles:int array -> float -> Header.t -> (float, 'b) Nx.t -> Hdu.t

(** {1:elements Elements} *)

val check_holds : Err.place -> Nx_dtype.Scalar.t -> Nx_dtype.Scalar.t -> unit
(** [check_holds place element dtype] raises {!Err.Fail} at [place], naming
    the dtypes that hold [element], unless [dtype] holds every value of
    [element] exactly. *)

val with_offset : Nx_dtype.Scalar.t -> Nx.packed -> Nx.packed
(** [with_offset element s] is the stored numbers [s] as [element]: the
    offsets of FITS 4.0 Table 11 added, modulo the width, which flips the
    sign bit. [s] is unchanged unless [element] is an offset one of [s]'s
    stored type. *)

val to_stored : ('a, 'b) Nx.t -> Nx.packed
(** [to_stored t] is the stored numbers of [t]'s elements: for [int8],
    [uint16], [uint32] and [uint64], the offset subtracted, modulo the
    width, in the stored type; [t] itself otherwise. *)

(** {1:bytes Byte order} *)

val of_big_endian :
  ('a, 'b) Nx.dtype -> int array -> Rig.Buffer.t -> ('a, 'b) Nx.t
(** [of_big_endian dtype shape bytes] is the C-order elements of [dtype]
    held big-endian in the host buffer [bytes], of shape [shape]. [shape]
    must hold as many elements as [bytes]. *)

val to_big_endian : ('a, 'b) Nx.t -> Rig.Buffer.t
(** [to_big_endian t] is a host buffer of [t]'s elements in C order, each
    big-endian. It may be [t]'s own storage when [t]'s elements are bytes
    or the host is big-endian. *)
