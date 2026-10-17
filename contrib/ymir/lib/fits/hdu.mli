(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HDUs: the implementation of {!Ymir_fits.Fits.hdu}, its reading and its
    lookup.

    An HDU is a header and the bytes of its data unit in a buffer: the disk
    buffer of a file, the host buffer of bytes held elsewhere, or the
    encoding of a constructed HDU. A constructed HDU computes its header and
    data on first use ({!Once}), so any function that reaches them may
    encode it, and blocks while another domain does.

    The functions on headers that read structural keywords raise
    {!Err.Fail} at the card that fails, for the public function's
    {!Err.catch}. *)

(** {1:hdus HDUs} *)

type store = { buffer : Rig.Buffer.t; offset : int; size : int }
(** The type for a data unit's bytes: [size] of them from byte [offset] of
    [buffer], padding excluded. *)

type derived
(** The type for values computed from an HDU ({!derive}). *)

type t = {
  name : string;  (** the file's path, {!of_bytes}'s name, or [""] *)
  index : int;  (** position in the file; [0] for a constructed HDU *)
  digest : string Once.t;  (** of the file's headers *)
  header : Header.t Once.t;
  data : store Once.t;
  header_bytes : (int * int) option;
      (** a read HDU's header, as an offset and a length in [buffer] *)
  stream : stream Once.t option;
      (** a constructed HDU whose data {!Files.write} streams without
          encoding it whole *)
  derived : derived list Atomic.t;  (** what {!derive} computed *)
}
(** {!Ymir_fits.Fits.hdu}. *)

(** The type for how {!Files.write} streams a constructed HDU: [provisional]
    holds the final header's records, so the data unit's place is known
    before it is written; [write sink] writes the data unit and is the final
    header. *)
and stream = { provisional : Header.t; write : sink -> Header.t }

(** The type for where a stream writes: [append] adds host bytes at the end
    of the data unit; [patch off s] writes [s] over bytes from [off] that
    were appended as zeros. *)
and sink = { append : Rig.Buffer.t -> unit; patch : int -> string -> unit }

val header : t -> Header.t
val name : t -> string
val digest : t -> string
val data : t -> (int, Nx.uint8_elt) Nx.t

val structure : t -> Header.t
(** [structure h] is a header holding [h]'s structure, without encoding a
    constructed HDU that {!Files.write} would stream. *)

val place : t -> Err.place
(** [place h] is the place of [h]'s header. *)

val store : t -> store
(** [store h] is [h]'s data unit. *)

val derive : 'a Type.Id.t -> (t -> 'a) -> t -> 'a
(** [derive key f h] is [f h], computed on the first ask for [key] and kept
    with [h]. Domains asking concurrently may each compute it, and all get
    the kept one. *)

(** {1:reading Reading} *)

val read : string -> (t list, string) result
val of_bytes :
  name:string -> (int, Nx.uint8_elt) Nx.t -> (t list, string) result

val buffer_of : (int, Nx.uint8_elt) Nx.t -> Rig.Buffer.t
(** [buffer_of t] is the C-order bytes of [t] in a host buffer: [t]'s own
    storage on the host, a host copy otherwise. *)

(** {1:lookup Lookup} *)

val get : ?ver:int -> string -> t list -> (t, string) result

val extname : t -> string option
(** [extname h] is [h]'s [EXTNAME], or [None] if it is absent or does not
    read. *)

val extver : t -> int
(** [extver h] is [h]'s [EXTVER], or [1] if it is absent or does not
    read. *)

(** {1:host Host bytes} *)

val host_bytes : int -> Rig.Buffer.t
(** [host_bytes n] is a new host buffer of [n] bytes. *)

val bigbytes : Rig.Buffer.t -> Checksum.bigbytes
(** [bigbytes b] is the bytes of [b], a host buffer of ymir's own, exported
    for good ({!Rig.Buffer.export}). *)

val reading : Rig.Buffer.t -> (Checksum.bigbytes -> 'a) -> 'a
(** [reading b f] is [f] of the bytes of the host buffer [b], which may be a
    value's storage, under a read claim so that no donation writes them
    while [f] reads ({!Rig.Buffer.read}). *)

val copy_in :
  Rig.Buffer.t -> offset:int -> Rig.Buffer.t -> at:int -> int -> unit
(** [copy_in src ~offset dst ~at n] copies [n] bytes from byte [offset] of
    [src] to byte [at] of the host buffer [dst]. *)

val read_string : Rig.Buffer.t -> offset:int -> int -> string
(** [read_string src ~offset n] is the [n] bytes of [src] from byte
    [offset]. *)

val host_store : Rig.Buffer.t -> store
(** [host_store b] is the data unit of all of [b]'s bytes. *)

(** {1:structure Structural keywords} *)

val hdu_place : string -> int -> Header.t -> Err.place
(** [hdu_place name index h] is the place of HDU [index] of the file
    [name], with [h]'s [EXTNAME] when it reads. *)

val get_struct : Header.t -> 'a Value.t -> string -> 'a
(** [get_struct h v k] is [k]'s value read with [v]. Raises {!Err.Fail} if
    [k] is absent or does not read. *)

val find_struct : Header.t -> 'a Value.t -> string -> 'a option
(** [find_struct h v k] is [k]'s value read with [v], if any. Raises
    {!Err.Fail} if it does not read. *)

val card_fail : Header.t -> string -> string -> 'a
(** [card_fail h k what] raises {!Err.Fail} with [what] at [k]'s first card,
    or at [h]'s place if [h] has no [k]. *)

val bitpix : Header.t -> int
(** [bitpix h] is [BITPIX]. Raises {!Err.Fail} if it is not 8, 16, 32, 64,
    -32 or -64. *)

val naxes : Header.t -> int array
(** [naxes h] is [NAXIS1] to [NAXISn], in file order. Raises {!Err.Fail} if
    [NAXIS] is outside 0 to 999 or an axis is negative. *)

(** The type for an HDU's kind. *)
type kind =
  | Primary of { groups : bool }  (** [GROUPS = T] marks random groups *)
  | Extension of string  (** [XTENSION], trailing spaces dropped *)

val kind_of : bool -> Header.t -> kind
(** [kind_of first h] is the kind of the HDU of header [h], the file's
    primary when [first]. Raises {!Err.Fail} if a primary's [SIMPLE] is
    [F]. *)

val data_size : kind -> Header.t -> int
(** [data_size kind h] is the size in bytes of the data unit [h] describes
    (FITS 4.0 Eq. 1, 2 and 4), padding excluded. Raises {!Err.Fail} if a
    size overflows or a count is negative. *)

val padded : int -> int
(** [padded n] is [n] rounded up to a whole number of blocks. *)

(** {1:constructed Constructed HDUs} *)

val constructed : ?stream:stream Once.t -> Header.t -> store Once.t -> t
(** [constructed ~stream h data] is the HDU of header [h] and data unit
    [data], without a file. *)

val constructed_lazy :
  ?stream:stream Once.t -> (Header.t * store) Once.t -> t
(** [constructed_lazy ~stream encoded] is the HDU whose header and data
    [encoded] computes when first asked, as a tiled image's, whose [PCOUNT]
    is its compressed size. *)
