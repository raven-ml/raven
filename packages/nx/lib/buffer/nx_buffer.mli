(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Flat buffers for tensor storage.

    Flat, C-layout, one-dimensional buffers of every {!Nx_dtype.t}: the standard
    Bigarray element types and the extended ones (bfloat16, bool, int4, float8,
    uint32, uint64).

    The buffer type {!t} is abstract in this interface. Conversions to and from
    {!Bigarray} are explicit via {!of_bigarray1}, {!to_bigarray1},
    {!of_genarray}, and {!to_genarray}.

    {b Warning.} Buffers must not be marshalled. Marshalling silently loses the
    extended dtype (a bfloat16 buffer round-trips as float16) and reads out of
    bounds for int4 dtypes. *)

(** {1:buf Buffer type and operations} *)

type ('a, 'b) t
(** [('a, 'b) t] is a flat, C-layout, one-dimensional buffer. *)

(** {2:create Creation} *)

val create : ('a, 'b) Nx_dtype.t -> int -> ('a, 'b) t
(** [create dtype n] allocates a buffer of [n] elements. Its contents are
    unspecified: write every element, or {!fill} it, before reading. *)

(** {2:props Properties} *)

val dtype : ('a, 'b) t -> ('a, 'b) Nx_dtype.t
(** [dtype buf] is the dtype of [buf]'s elements. *)

val length : ('a, 'b) t -> int
(** [length buf] is the number of elements in [buf]. *)

(** {2:access Element access} *)

val get : ('a, 'b) t -> int -> 'a
(** [get buf i] is the element at index [i].

    Raises [Invalid_argument] if [i] is out of bounds. *)

val set : ('a, 'b) t -> int -> 'a -> unit
(** [set buf i v] sets the element at index [i] to [v].

    Raises [Invalid_argument] if [i] is out of bounds. *)

val unsafe_get : ('a, 'b) t -> int -> 'a
(** [unsafe_get buf i] is like {!get} without bounds checking. *)

val unsafe_set : ('a, 'b) t -> int -> 'a -> unit
(** [unsafe_set buf i v] is like {!set} without bounds checking. *)

val unsafe_data_ptr : ('a, 'b) t -> nativeint
(** [unsafe_data_ptr buf] is the address of [buf]'s first element. The buffer's
    storage lives outside the OCaml heap and is never moved, so the address
    stays valid for as long as [buf] is reachable — keep a reference to [buf]
    alive while the pointer is in use, and do not use it afterwards. Not
    available on JavaScript. *)

(** {2:reinterpret Reinterpretation} *)

val reinterpret : ('a, 'b) Nx_dtype.t -> ('c, 'd) t -> ('a, 'b) t
(** [reinterpret dtype buf] is [buf]'s memory read as elements of [dtype],
    without a copy. Its length is [buf]'s size in bytes divided by
    [Nx_dtype.itemsize dtype]. Elements are read in the machine's byte order.

    The result and [buf] share their storage, as the results of
    {!Bigarray.Array1.sub} do: a write through either is seen through the other,
    and storage that the runtime manages, allocated or mapped from a file, lives
    until both are unreachable. Over memory some other owner manages, the owner
    stays the caller's concern.

    This is the only way to view existing memory at an extended dtype, such as
    the bytes of a mapped file as [bfloat16].

    Raises [Invalid_argument] if [buf]'s size in bytes is not a multiple of
    [Nx_dtype.itemsize dtype], if [buf]'s address is not a multiple of it, or if
    [dtype] or [buf]'s dtype is [Int4] or [UInt4]. *)

(** {2:files Mapped files}

    A buffer over a mapped file can say where in the file its bytes are, so that
    code that needs the bytes elsewhere, such as an upload to a device, can read
    them from the file with ordinary reads instead of faulting them in through
    the mapping page by page. *)

type file = {
  path : string;  (** The path the file was opened by. *)
  size : int;  (** Its size in bytes when it was mapped. *)
  mtime : float;  (** Its modification time when it was mapped. *)
  inode : int;  (** Its inode number when it was mapped, [0] where none. *)
}
(** The type for the identity of a mapped file. A path may name another file
    later: before reading through [path], check that the file opened has this
    size, modification time and inode. *)

val register_file : file -> (int, Nx_dtype.uint8_elt) t -> unit
(** [register_file file buf] records that [buf] is a mapping of the whole of
    [file], from its first byte: [buf] is the result of [Unix.map_file] at
    position [0], before any sub-array or other view of it was made. From then
    on {!file_range} answers for [buf] and for every buffer derived from it. The
    record is dropped when the file is unmapped, which the runtime does once the
    last buffer over the mapping has been collected.

    Raises [Invalid_argument] if [buf] is not a mapped file or already has
    views. *)

val file_range : ('a, 'b) t -> (file * int) option
(** [file_range buf] is the file [buf]'s memory is a mapping of and the byte
    offset in that file of [buf]'s first element, if [buf]'s memory lies inside
    a mapping recorded with {!register_file}. Results of {!Bigarray.Array1.sub},
    {!reinterpret} and the other views answer by their address. It is [None] for
    any other buffer, and always on JavaScript. *)

(** {2:bulk Bulk operations} *)

val fill : ('a, 'b) t -> 'a -> unit
(** [fill buf v] sets every element of [buf] to [v]. *)

val blit : src:('a, 'b) t -> dst:('a, 'b) t -> unit
(** [blit ~src ~dst] copies all elements from [src] to [dst].

    Raises [Invalid_argument] if dimensions differ. *)

val blit_from_bytes :
  ?src_off:int -> ?dst_off:int -> ?len:int -> bytes -> ('a, 'b) t -> unit
(** [blit_from_bytes ?src_off ?dst_off ?len bytes buf] copies [len] elements
    from [bytes] into [buf]. Offsets and length are in elements; [bytes] holds
    the elements in their storage representation. [src_off] and [dst_off]
    default to [0]. [len] defaults to [length buf - dst_off].

    For [Int4] and [UInt4], [bytes] holds two elements packed per byte;
    [src_off] and [dst_off] must be even, and [len] must be even unless the copy
    reaches the end of [buf]. Raises [Invalid_argument] otherwise. *)

val blit_to_bytes :
  ?src_off:int -> ?dst_off:int -> ?len:int -> ('a, 'b) t -> bytes -> unit
(** [blit_to_bytes ?src_off ?dst_off ?len buf bytes] copies [len] elements from
    [buf] into [bytes]. Offsets and length are in elements; [bytes] receives the
    elements in their storage representation. [src_off] and [dst_off] default to
    [0]. [len] defaults to [length buf - src_off].

    For [Int4] and [UInt4], [bytes] holds two elements packed per byte, and
    [src_off] and [dst_off] must be even; if [len] is odd the last byte written
    carries a padding upper nibble. Raises [Invalid_argument] on odd offsets. *)

(** {1:ba Bigarray conversions}

    Only buffers of standard dtypes can be viewed as one-dimensional bigarrays:
    an extended dtype has no faithful {!Bigarray.kind}, so exposing one would
    let standard bigarray operations misread its contents. *)

val of_bigarray1 : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> ('a, 'b) t
(** [of_bigarray1 ba] is [ba] viewed as a buffer. Zero-copy.

    Raises [Invalid_argument] if [ba]'s kind is [Char], [Int] or [Nativeint],
    which buffers do not support. *)

val to_bigarray1 : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [to_bigarray1 buf] is [buf] viewed as a one-dimensional bigarray. Zero-copy.

    Raises [Invalid_argument] if [buf]'s dtype has no {!Bigarray.kind}
    ([Nx_dtype.to_bigarray_kind (dtype buf) = None]). *)

val to_genarray :
  ('a, 'b) t -> int array -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_genarray buf shape] reshapes [buf] into a genarray with [shape]. The
    product of [shape] must equal [length buf]. Zero-copy.

    For a buffer of an extended dtype, the resulting genarray carries [buf]'s
    dtype in storage flags that only the functions of this module understand;
    see the {{!section:ga}genarray bridge}. *)

val of_genarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> ('a, 'b) t
(** [of_genarray ga] flattens [ga] into a one-dimensional buffer.

    Raises [Invalid_argument] if [ga]'s kind is [Char], [Int] or [Nativeint],
    which buffers do not support. *)

(** {1:ga Genarray bridge}

    Operations on {!Bigarray.Genarray.t} that handle extended dtypes. Used by
    I/O modules (npy, safetensors, images).

    A genarray obtained from {!to_genarray} or {!genarray_create} with an
    extended dtype is only meaningful to the functions below and to
    {!of_genarray}; standard {!Bigarray} operations misread its contents (and
    read out of bounds for int4 dtypes). Keep such values inside I/O plumbing.
*)

val genarray_create :
  ('a, 'b) Nx_dtype.t ->
  'c Bigarray.layout ->
  int array ->
  ('a, 'b, 'c) Bigarray.Genarray.t
(** [genarray_create dtype layout dims] allocates a genarray of [dtype]'s
    elements. *)

val genarray_dtype : ('a, 'b, 'c) Bigarray.Genarray.t -> ('a, 'b) Nx_dtype.t
(** [genarray_dtype ga] is the dtype of [ga]'s elements, extended dtypes
    included. *)

val genarray_dims : ('a, 'b, 'c) Bigarray.Genarray.t -> int array
(** [genarray_dims ga] is the dimensions of [ga]. *)

val genarray_blit :
  ('a, 'b, 'c) Bigarray.Genarray.t -> ('a, 'b, 'c) Bigarray.Genarray.t -> unit
(** [genarray_blit src dst] copies [src] to [dst]. Handles extended dtypes. *)

val genarray_change_layout :
  ('a, 'b, 'c) Bigarray.Genarray.t ->
  'd Bigarray.layout ->
  ('a, 'b, 'd) Bigarray.Genarray.t
(** [genarray_change_layout ga layout] changes the layout of [ga]. *)
