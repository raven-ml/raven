(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The verbs ioctls the path makes: each a method of a kernel object with its
    attributes, laid out as the kernel reads them, and the readers of what the
    kernel writes back.

    A command of the kernel's older interface (alloc a protection domain,
    register memory, make a queue) is a method too: the device's [INVOKE_WRITE],
    whose attributes carry the command's request, its answer and the driver's
    data. *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for bytes outside the OCaml heap, which the kernel reads and writes
    while the runtime is released. *)

val params : int -> params
(** [params n] is [n] zero bytes. *)

val of_string : string -> params
(** [of_string s] is [s]'s bytes. *)

val to_string : params -> string
(** [to_string p] is [p]'s bytes. *)

(** The type for the attributes of a method. *)
type attr =
  | Value of int * int  (** An 8-byte input, inline: a constant or flags. *)
  | Word of int * int  (** A 4-byte input, inline. *)
  | In of int * params
      (** An input of the bytes: inline if at most 8, by address otherwise. *)
  | Handle of int * int  (** An object the kernel reads, by its handle. *)
  | Made of int  (** An object the kernel makes: it writes the handle. *)
  | Out of int * params  (** Bytes the kernel writes, by address. *)

type t
(** The type for requests: an ioctl's header and attributes, and the bytes its
    attributes name, which it keeps alive. *)

val number : int
(** [number] is the verbs ioctl's request number. *)

val call : driver:int -> obj:int -> meth:int -> attr list -> t
(** [call ~driver ~obj ~meth attrs] calls the method [meth] of the object [obj]
    with [attrs], on a device whose driver is [driver]. Every attribute is
    mandatory: a kernel that does not know one refuses the call.

    Raises [Invalid_argument] if an attribute is longer than 65535 bytes or the
    header is longer than a page of 4096 bytes. *)

val bytes : t -> params
(** [bytes r] is the ioctl's header and attributes, which the ioctl reads. *)

val made : t -> int -> int
(** [made r id] is the handle the kernel wrote for the attribute [Made id] of
    [r]. *)

(** {1:commands Commands} *)

val write :
  driver:int ->
  cmd:int ->
  params ->
  out:params ->
  uhw:string ->
  uhw_out:params ->
  t
(** [write ~driver ~cmd req ~out ~uhw ~uhw_out] invokes the command [cmd] with
    the request [req] (its [response] field included, which the kernel ignores),
    its answer in [out], the driver data [uhw] and the driver's answer in
    [uhw_out]. An empty [out], [uhw] or [uhw_out] is left out. *)

val field : params -> int * int -> int
(** [field p (at, n)] is the field of [n] bytes at byte [at] of [p], [n] 1, 2, 4
    or 8, as an unsigned integer. *)

val set : params -> int * int -> int -> unit
(** [set p (at, n) v] writes [v] in that field. *)
