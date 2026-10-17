(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Entry functions: one per constructor of {!Op.t}, on which the frontend
    builds.

    While no interception is live on any domain ({!Intercept.intercepting}),
    an entry function answers with {!Dispatch} and builds no operation.
    Otherwise it performs its operation ({!Intercept.perform}). {!qr},
    {!lu}, {!svd} and the eigen functions always build theirs
    ({!Intercept.eval}).

    A [bit] operand reaches an operation only where backends compute on
    bits: casts, bitcasts, [And], [Or], [Xor], [Maximum] and [Minimum], the
    [Max] and [Min] reductions, and the moves ([pad], [cat], [gather],
    [scatter] with [`Set], [update] and copies). {!sort} and {!argsort} pass
    a [bit] operand to the backend as it is, and nx.cpu raises
    [Invalid_argument] on a packed dtype. Every other entry function reads
    its [bit] operands as [bool] and stores a result of their dtype as
    [bit], so it answers as it does on [bool], raising where that raises. *)

(** {1:ops Operations} *)

val cast :
  ('a, 'b) Nx_dtype.t -> ('c, 'd) Value.t -> ('a, 'b) Value.t
val bitcast :
  ('a, 'b) Nx_dtype.t -> ('c, 'd) Value.t -> ('a, 'b) Value.t
val unary :
  Nx_backend.unary -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val binary :
  Nx_backend.binary ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val cmp :
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
  axes:int array -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val scan :
  Nx_backend.reduce ->
  axis:int -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val arg_reduce :
  Nx_backend.arg_reduce ->
  axis:int -> ('a, 'b) Value.t -> Op.int64_t
val sort :
  descending:bool ->
  axis:int -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val argsort :
  descending:bool -> axis:int -> ('a, 'b) Value.t -> Op.int64_t
val group :
  by:string ->
  (int64, Nx_dtype.uint64_elt) Value.t -> Op.int64_t
val pad :
  (int * int) array ->
  'a -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val cat : axis:int -> ('a, 'b) Value.t list -> ('a, 'b) Value.t
val threefry : Op.int32_t -> Op.int32_t -> Op.int32_t
val gather :
  axis:int ->
  Op.int64_t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val scatter :
  mode:Nx_backend.scatter ->
  unique:bool ->
  axis:int ->
  indices:Op.int64_t ->
  updates:('a, 'b) Value.t ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t
val update :
  ('a, 'b) Value.t ->
  starts:Op.int64_t ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t
val unfold :
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t
val fold :
  output_size:int array ->
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t
val matmul :
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val fft :
  inverse:bool ->
  axes:int array ->
  (Complex.t, 'a) Value.t -> (Complex.t, 'a) Value.t
val rfft :
  (Complex.t, 'a) Nx_dtype.t ->
  axes:int array ->
  (float, 'b) Value.t -> (Complex.t, 'a) Value.t
val irfft :
  ?s:int array ->
  (float, 'a) Nx_dtype.t ->
  axes:int array ->
  (Complex.t, 'b) Value.t -> (float, 'a) Value.t
val cholesky :
  upper:bool -> ('a, 'b) Value.t -> ('a, 'b) Value.t
val qr :
  reduced:bool ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t * ('a, 'b) Value.t
val lu :
  ('a, 'b) Value.t ->
  ('a, 'b) Value.t * Op.int64_t * Op.int64_t
val svd :
  full_matrices:bool ->
  ('a, 'b) Value.t ->
  ('a, 'b) Value.t * (float, Nx_dtype.float64_elt) Value.t *
  ('a, 'b) Value.t
val eigvals :
  ('a, 'b) Value.t ->
  (Complex.t, Nx_dtype.complex64_elt) Value.t
val eigvalsh :
  ('a, 'b) Value.t -> (float, Nx_dtype.float64_elt) Value.t

val eig :
  ('a, 'b) Value.t ->
  (Complex.t, Nx_dtype.complex64_elt) Value.t *
  (Complex.t, Nx_dtype.complex64_elt) Value.t
(** [eig x] is [x]'s eigenvalues and eigenvectors.

    Raises [Invalid_argument] if the interpretation answers no
    eigenvectors. *)

val eigh :
  ('a, 'b) Value.t ->
  (float, Nx_dtype.float64_elt) Value.t * ('a, 'b) Value.t
(** [eigh x] is as {!eig} for a symmetric or Hermitian [x]. *)

val solve_triangular :
  upper:bool ->
  transpose:bool ->
  unit_diag:bool ->
  ('a, 'b) Value.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t

(** {1:moves Movements}

    Each is a view of its operand's storage when no interception is live,
    or of a C-order copy when its strides cannot express a reshape
    ({!Dispatch.move}). *)

val reshape : ('a, 'b) Value.t -> int array -> ('a, 'b) Value.t
val expand : ('a, 'b) Value.t -> int array -> ('a, 'b) Value.t
val permute : ('a, 'b) Value.t -> int array -> ('a, 'b) Value.t
val shrink :
  ('a, 'b) Value.t -> (int * int) array -> ('a, 'b) Value.t
val flip : ('a, 'b) Value.t -> bool array -> ('a, 'b) Value.t
val sliding_window :
  ('a, 'b) Value.t ->
  axis:int -> window:int -> step:int -> ('a, 'b) Value.t

val broadcast :
  ('a, 'b) Value.t -> Nx_array.Shape.t -> ('a, 'b) Value.t
(** [broadcast x shape] is the one-element [x] reshaped to [shape]'s rank
    and expanded to [shape]. [broadcast x [||]] is [x] unchanged. *)

(** {1:reads Reads and placing} *)

val read : by:string -> ('a, 'b) Value.t -> Rig.Buffer.t
(** [read ~by x] is the elements of [x]'s view in C order, in a host buffer,
    as {!Dispatch.read_elements_of}. [by] names the surface function that
    reads, for an interpretation. *)

val check :
  (bool, Nx_dtype.bool_elt) Value.t ->
  Value.packed list ->
  (int array -> Value.packed list -> exn) -> unit
(** [check ok data fail] is {!Dispatch.check_elements}[ ok data fail]. *)

val place :
  Placement.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
(** {!Nx.place}. A value already at [p] is returned as it is once a read
    claim on its storage has checked it, which raises as {!Value.reading}
    does. *)

val copy : ('a, 'b) Value.t -> ('a, 'b) Value.t
(** [copy x] is [x] in storage of its own, C-contiguous from its first
    element. It always copies. An interpretation answers a traced value's
    copy. *)

val contiguous : ('a, 'b) Value.t -> ('a, 'b) Value.t
(** [contiguous x] is [x] itself when its view is C-contiguous from its
    first element, and {!copy}[ x] otherwise. *)

val to_buffer : ('a, 'b) Value.t -> Rig.Buffer.t

(** {1:creation Creation} *)

val full :
  Value.context ->
  ('a, 'b) Nx_dtype.t -> Nx_array.Shape.t -> 'a -> ('a, 'b) Value.t
(** [full ctx dtype shape v] is the constant [v] of [shape], one element on
    each device of [ctx] (the host's included) expanded as a view to
    [shape]. A kernel reads it with stride 0, and a compiled call folds it
    into its kernels. *)

val from_host :
  Value.context ->
  ('a, 'b) Nx_dtype.t -> int -> Rig.Buffer.t -> ('a, 'b) Value.t
(** [from_host ctx dtype n b] is the value of the [n] elements of [dtype] in
    the host buffer [b], placed at [ctx].

    Raises [Invalid_argument] if [b] is off the host. *)
