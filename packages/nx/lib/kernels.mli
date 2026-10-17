(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels: each operation's results, allocated by nx and written by a
    backend's kernels on one device.

    A kernel function [f e ...] takes the payload of its {!Op.t} constructor,
    in the constructor's order, as arrays on [e]'s device. It allocates each
    result with [e.alloc], C-contiguous from its first element, and has [e]'s
    backend write it. While the backend runs, the operands' memory is under
    read claims, taken on every operand or on none, so that no compiled call
    on another domain donates it; the claims are released however the kernel
    ends. A kernel on a memory the host runs on first waits for the work that
    wrote its operands.

    A kernel function raises what the backend raises, such as
    {!Nx_backend.Refused}. It raises [Invalid_argument] if an operand's
    memory is dead or held exclusive by a consuming call, and [Rig.Lost] if
    it depends on a lost device. *)

(** Where kernels run. *)
type env = {
  device : Device.t;
  kernels : (module Nx_backend.S);
  alloc : 'a 'b. ('a, 'b) Nx_dtype.t -> int array -> ('a, 'b) Nx_array.t;
      (** Allocates a result on [device]. *)
  arr : 'a 'b. ('a, 'b) Value.t -> ('a, 'b) Nx_array.t;
      (** An operand's array on [device]. *)
}

val host : env
(** [host] runs nx.cpu on the host. Its [arr] reads a placed value to the
    host ({!Place.host_of}). *)

val on : (module Nx_backend.S) -> Device.t -> env
(** [on kernels d] runs [kernels] on [d]. Its results lie in [d]'s memory,
    with the bits of a result's last byte past its last element at [0]. Its
    [arr] is a placed value's shard on [d], and raises [Invalid_argument] on
    a host value. *)

val shape_of : ('a, 'b) Nx_array.t -> int array
(** [shape_of a] is the shape of [a]'s view. *)

val narrowed : Nx_array.View.t -> int -> Nx_array.View.t
(** [narrowed v k] is [v] with each element read as [k] along a new last
    axis, as a [k] times narrower dtype reads it. *)

val bitcast :
  env -> ('c, 'd) Nx_dtype.t -> ('a, 'b) Nx_array.t -> ('c, 'd) Nx_array.t
(** [bitcast e dtype a] is [a]'s bits read as elements of [dtype], counted in
    bits, so that [bit] is an eighth of [uint8]. At [a]'s width it is the same
    view, and [k] times narrower each element as [k] along a new last axis
    ({!narrowed}). [k] times wider, it reads each run of [k] along the last
    axis, of [k], as one element, a view of [a]'s memory when [a] is
    C-contiguous from a first element whose bits start in memory at a
    multiple of [dtype]'s width, and of a C-contiguous copy that [e] makes
    otherwise. For a [dtype] of whole bytes, that start is an address
    aligned to its size in bytes, so an [a] whose first element lies at an
    unaligned address is copied. *)

(** {1:kernels Kernels} *)

val unary :
  env -> Nx_backend.unary -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val binary :
  env ->
  Nx_backend.binary ->
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val compare :
  env ->
  Nx_backend.compare ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t -> (bool, Nx_dtype.bool_elt) Nx_array.t
val where :
  env ->
  (bool, Nx_dtype.bool_elt) Nx_array.t ->
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val fma :
  env ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val reduce :
  env ->
  Nx_backend.reduce ->
  int array -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val scan :
  env ->
  Nx_backend.reduce -> int -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val arg_reduce :
  env ->
  Nx_backend.arg_reduce ->
  int -> ('a, 'b) Nx_array.t -> (int64, Nx_dtype.int64_elt) Nx_array.t
val sort :
  env -> bool -> int -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val argsort :
  env ->
  bool ->
  int -> ('a, 'b) Nx_array.t -> (int64, Nx_dtype.int64_elt) Nx_array.t
val group :
  env ->
  (int64, Nx_dtype.uint64_elt) Nx_array.t ->
  (int64, Nx_dtype.int64_elt) Nx_array.t
val pad :
  env ->
  (int * int) array -> 'a -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val cat : env -> int -> ('a, 'b) Nx_array.t list -> ('a, 'b) Nx_array.t
val cast :
  env ->
  ('a, 'b) Nx_dtype.t -> ('c, 'd) Nx_array.t -> ('a, 'b) Nx_array.t
val threefry :
  env ->
  (int32, Nx_dtype.int32_elt) Nx_array.t ->
  (int32, Nx_dtype.int32_elt) Nx_array.t ->
  (int32, Nx_dtype.int32_elt) Nx_array.t
val gather :
  env ->
  int ->
  Nx_backend.index_array -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val scatter :
  env ->
  Nx_backend.scatter ->
  bool ->
  int ->
  Nx_backend.index_array ->
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val update :
  env ->
  ('a, 'b) Nx_array.t ->
  Nx_backend.index_array -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val unfold :
  env ->
  int array ->
  int array ->
  int array ->
  (int * int) array -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val fold :
  env ->
  int array ->
  int array ->
  int array ->
  int array ->
  (int * int) array -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val matmul :
  env ->
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val fft :
  env ->
  bool ->
  int array -> (Complex.t, 'a) Nx_array.t -> (Complex.t, 'a) Nx_array.t
val rfft :
  env ->
  (Complex.t, 'a) Nx_dtype.t ->
  int array -> (float, 'b) Nx_array.t -> (Complex.t, 'a) Nx_array.t
val irfft :
  env ->
  (float, 'a) Nx_dtype.t ->
  int array ->
  int array option ->
  (Complex.t, 'b) Nx_array.t -> (float, 'a) Nx_array.t
val contiguous : env -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val cholesky : env -> bool -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
val qr :
  env ->
  bool ->
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t * ('a, 'b) Nx_array.t
val lu :
  env ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t * (int64, Nx_dtype.int64_elt) Nx_array.t *
  (int64, Nx_dtype.int64_elt) Nx_array.t
val svd :
  env ->
  bool ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t * (float, Nx_dtype.float64_elt) Nx_array.t *
  ('a, 'b) Nx_array.t
val eig :
  env ->
  bool ->
  ('a, 'b) Nx_array.t ->
  (Complex.t, Nx_dtype.complex64_elt) Nx_array.t *
  (Complex.t, Nx_dtype.complex64_elt) Nx_array.t option
val eigh :
  env ->
  bool ->
  ('a, 'b) Nx_array.t ->
  (float, Nx_dtype.float64_elt) Nx_array.t * ('a, 'b) Nx_array.t option
val solve_triangular :
  env ->
  bool ->
  bool ->
  bool ->
  ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t -> ('a, 'b) Nx_array.t
