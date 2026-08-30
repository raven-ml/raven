(** Typed access to the generated NVIDIA driver tables.

    The generated [Nv_defs] module holds the data shared by every supported
    driver generation: class ids, escape and UVM ioctl command numbers, flag
    and method constants, parameter-structure layouts, and the QMD bitfield
    tables. [Nv_defs_versions] holds the handful of parameter-structure
    layouts that differ across driver generations, as one record value per
    generation. This module selects the generation record and provides the
    byte-blob helpers used to build parameter structures for the driver's
    escape and UVM ioctls.

    Structure layouts are (byte offset, byte size) pairs read and written
    over [blob] values, off-heap byte buffers whose storage never moves, so
    a blob's address may be embedded into another blob when a parameter
    structure carries a pointer to a second one. *)

module Defs : module type of Nv_defs
(** Raw generation-stable data: class ids, escape numbers, allocation and
    mapping flags, channel method ids and argument bit ranges, control
    commands, UVM command numbers, parameter-structure layouts, MMU-fault
    name tables, and the QMD bitfield tables. *)

module Versions : module type of Nv_defs_versions
(** The per-generation parameter-structure layouts and status-code names,
    as three values of the record type [Nv_defs_versions.t]. *)

val defs_for_driver : major:int -> Nv_defs_versions.t
(** [defs_for_driver ~major] is the layout record for the installed driver's
    major version: 610 and above select the 610 layouts, 580 to 609 the 580
    layouts, and anything older the 570 layouts. *)

type blob =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** Off-heap storage for one driver parameter structure. *)

val create_blob : int -> blob
(** [create_blob size] is a zero-filled blob of [size] bytes. *)

val get_field : ?base:int -> blob -> int * int -> int
(** [get_field ?base b (off, size)] reads the unsigned little-endian field
    of [size] bytes (1, 2, 4, or 8) at byte [base + off]. [base] defaults to
    0; pass it to address a structure embedded at a nonzero offset or an
    array element. Raises [Invalid_argument] when [size] is unsupported,
    the field lies outside [b], or an 8-byte value exceeds [max_int]. *)

val set_field : ?base:int -> blob -> int * int -> int -> unit
(** [set_field ?base b (off, size) v] writes [v] little-endian into the
    field of [size] bytes (1, 2, 4, or 8) at byte [base + off], truncating
    [v] to the field width, so [-1] writes an all-ones field.
    Raises [Invalid_argument] when [size] is unsupported or the field lies
    outside [b]. *)

val escape_code : nr:int -> size:int -> int
(** [escape_code ~nr ~size] is the ioctl request number for driver escape
    call [nr] carrying a read-write parameter structure of [size] bytes.
    UVM commands are not escapes; their raw command numbers are the request
    numbers. *)

val blob_addr : blob -> nativeint
(** [blob_addr b] is the address of [b]'s storage, for embedding into a
    parameter structure that points at a second structure. Keep [b] live
    for as long as the address may be dereferenced. *)

val ioctl : fd:int -> request:int -> blob -> int
(** [ioctl ~fd ~request b] issues the ioctl [request] on [fd] with [b] as
    its argument and returns the ioctl's return value; driver-level status
    is reported inside the blob, not by the return value. The call holds
    the OCaml runtime lock: no call on this path blocks in the kernel.
    Raises [Failure] on syscall failure, and on platforms other than
    Linux. *)
