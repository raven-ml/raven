(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dispatch: operations answered with no interpretation.

    Operands all on the host run nx.cpu directly, with no operation built.
    Other operands run where their route gives ({!Route.route}), once per
    device of that placement, by its backend, on that device's arrays. nx
    first places there each operand that is not there, such as a host value,
    a value on the disk, or a split value that a device needs whole
    ({!Place.move_to}). nx answers movements, placing and reading itself.

    An operation function [f ...] takes the payload of its {!Op.t}
    constructor, in the constructor's order. It raises [Invalid_argument]
    as {!Route.route} does, and before any work if a device of the route has
    no backend. A backend that refuses its operands ({!Nx_backend.Refused})
    makes it raise [Invalid_argument] naming the backend, the device and the
    reason. A traced operand raises as {!Value.outside_trace}. *)

val direct : 'r Op.t -> 'r
(** [direct op] answers [op] with no interpretation. *)

(** {1:ops Operations} *)

val unary :
  Nx_backend.unary -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val binary :
  Nx_backend.binary ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val compare :
  Nx_backend.compare ->
  ('a, 'b) Value.t ->
  ('a, 'b) Value.t -> (bool, Nx_dtype.bool_elt) Value.t
val where :
  (bool, Nx_dtype.bool_elt) Value.t ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val fma :
  ('a, 'b) Value.t ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val reduce :
  Nx_backend.reduce ->
  int array -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val scan :
  Nx_backend.reduce ->
  int -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val arg_reduce :
  Nx_backend.arg_reduce ->
  int ->
  ('a, 'b) Value.t -> (int64, Nx_dtype.int64_elt) Value.t
val sort : bool -> int -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val argsort :
  bool ->
  int ->
  ('a, 'b) Value.t -> (int64, Nx_dtype.int64_elt) Value.t
val group :
  string ->
  (int64, Nx_dtype.uint64_elt) Value.t ->
  (int64, Nx_dtype.int64_elt) Value.t
val pad :
  (int * int) array ->
  'a -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val cat : int -> ('a, 'b) Value.t list -> ('a, 'b) Value.t
val convert :
  Op.conversion ->
  ('c, 'd) Nx_dtype.t -> ('a, 'b) Value.t -> ('c, 'd) Value.t
val threefry :
  (int32, Nx_dtype.int32_elt) Value.t ->
  (int32, Nx_dtype.int32_elt) Value.t ->
  (int32, Nx_dtype.int32_elt) Value.t
val gather :
  int ->
  (int64, Nx_dtype.int64_elt) Value.t ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t
val scatter :
  Nx_backend.scatter ->
  bool ->
  int ->
  (int64, Nx_dtype.int64_elt) Value.t ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val update :
  ('a, 'b) Value.t ->
  (int64, Nx_dtype.int64_elt) Value.t ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t
val unfold :
  int array ->
  int array ->
  int array ->
  (int * int) array -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val fold :
  int array ->
  int array ->
  int array ->
  int array ->
  (int * int) array -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val matmul :
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val fft :
  bool ->
  int array ->
  (Complex.t, 'a) Value.t -> (Complex.t, 'a) Value.t
val rfft :
  (Complex.t, 'a) Nx_dtype.t ->
  int array -> (float, 'b) Value.t -> (Complex.t, 'a) Value.t
val irfft :
  (float, 'a) Nx_dtype.t ->
  int array ->
  int array option ->
  (Complex.t, 'b) Value.t -> (float, 'a) Value.t
val cholesky : bool -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val solve_triangular :
  bool ->
  bool ->
  bool ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t

val copy : ('a, 'b) Value.t -> ('a, 'b) Value.t
(** [copy x] is [x] in storage of its own, C-contiguous from its first
    element, as [Contiguous x]. *)

val move : ('a, 'b) Value.t -> Move.t -> ('a, 'b) Value.t
(** [move x m] is [x] moved by [m], as a view of [x]'s storage, or of a
    C-order copy of [x] when [x]'s strides cannot express a reshape. *)

(** {1:reads Reads} *)

val read_elements_of : ('a, 'b) Value.t -> Rig.Buffer.t
(** [read_elements_of x] is the elements of [x]'s view in C order, in a host
    buffer. It is [x]'s own storage when [x] is a host value whose elements
    are one run of it, and a copy otherwise. *)

val check_elements :
  (bool, Nx_dtype.bool_elt) Value.t -> Value.packed list ->
  (int array -> Value.packed list -> exn) -> unit
(** [check_elements ok data fail] raises [fail i d] if [ok] has a false
    element, where [i] is the index of the first in C order and [d] each of
    [data], of [ok]'s shape, copied at its element [i]. *)
