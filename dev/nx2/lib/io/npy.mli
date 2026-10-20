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

val encode : ('v, 's) Nx_array.Dtype.t -> int array -> string
(** [encode dt s] is the header of a C-order little-endian stream of [dt]
    elements at shape [s]. [dt] passes {!check}. *)
