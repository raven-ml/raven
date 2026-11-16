(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NumPy's npy format: headers, and arrays over their data. *)

val check : by:string -> ('v, 's) Nx_array.Dtype.t -> unit
(** [check ~by dt] is [()] if npy has [dt].

    Raises [Invalid_argument] naming [by] otherwise. *)

type header = private {
  dtype : Nx_array.Dtype.any;
  shape : int array;
  fortran : bool;  (** Whether the first axis varies fastest. *)
  swap : bool;  (** Whether the elements' byte order differs from the host's. *)
  data : int;  (** The offset of the data from the header's start. *)
  size : int;  (** The data's length in bytes. *)
}
(** The type for what a header states. *)

val header : (int -> int -> string) -> length:int -> header
(** [header read ~length] is the header of the npy stream of [length] bytes that
    [read pos n] reads, [n] bytes from [pos], at most what remains.

    Raises [Failure] if the header is malformed, or its data is not [length]
    less the header's bytes. *)

val value : header -> Rig.Buffer.t -> Nx.host Nx.packed
(** [value h b] is the array of the stream [b] whose header is [h]
    ({!Storage.value}). *)

val copied : header -> Rig.Buffer.t -> Nx.host Nx.packed
(** [copied h b] is {!value} over a copy of the array's bytes that the value
    owns ({!Storage.copied}). *)

val rows :
  by:string ->
  header ->
  Rig.Buffer.t ->
  row:int array ->
  rows:int array ->
  Nx.host Nx.packed
(** [rows ~by h file ~row ~rows] is [Nx_io.load_npy_rows]'s array of the
    stream [file] whose header is [h], for windows of non-negative bounds
    and [row] and [rows] of one length.

    Raises [Invalid_argument] naming [by] if a window ends past [h]'s rows or
    the windows hold more than 2^53 elements, or rows where a row holds none,
    and [Failure] if [h]'s array has no axis or lays out a row's elements
    apart. *)

val encode : ?length:int -> ('v, 's) Nx_array.Dtype.t -> int array -> string
(** [encode dt s] is the header of a C-order little-endian stream of [dt]
    elements at shape [s]. [dt] must pass {!check}. With [~length] it is
    padded to [length] bytes, which must be a multiple of 64 at least its
    own padded length. *)
