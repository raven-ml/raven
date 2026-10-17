(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations as values: the implementation of {!Nx.Op}, without its
    interpretation ({!Intercept}) and placement ({!Route}).

    Besides {!Nx.Op}'s functions, it computes the shapes of results from
    their operands', which the kernels allocate before they write them. These
    shape functions check nothing beyond what they state, since their callers
    have checked the operands. *)

(** {1:ops Operations} *)

type conversion = Cast | Bitcast  (** {!Nx.Op.conversion}. *)

type move = Move.t =
  | Reshape of int array
  | Expand of int array
  | Permute of int array
  | Shrink of (int * int) array
  | Flip of bool array
  | Window of { axis : int; size : int; step : int }

type int32_t = (int32, Nx_dtype.int32_elt) Value.t
type int64_t = (int64, Nx_dtype.int64_elt) Value.t

(** {!Nx.Op.t}. *)
type _ t =
  | Unary : Nx_backend.unary * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Binary :
      Nx_backend.binary * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Compare :
      Nx_backend.compare * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> (bool, Nx_dtype.bool_elt) Value.t t
  | Where :
      (bool, Nx_dtype.bool_elt) Value.t * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Fma :
      ('a, 'b) Value.t * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Reduce :
      Nx_backend.reduce * int array * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Scan : Nx_backend.reduce * int * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Arg_reduce : Nx_backend.arg_reduce * int * ('a, 'b) Value.t -> int64_t t
  | Sort : {
      descending : bool;
      axis : int;
      x : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Argsort : {
      descending : bool;
      axis : int;
      x : ('a, 'b) Value.t;
    }
      -> int64_t t
  | Group : {
      by : string;
      x : (int64, Nx_dtype.uint64_elt) Value.t;
    }
      -> int64_t t
  | Pad : (int * int) array * 'a * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Cat : int * ('a, 'b) Value.t list -> ('a, 'b) Value.t t
  | Convert :
      conversion * ('c, 'd) Nx_dtype.t * ('a, 'b) Value.t
      -> ('c, 'd) Value.t t
  | Threefry : int32_t * int32_t -> int32_t t
  | Gather : int * int64_t * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Scatter : {
      mode : Nx_backend.scatter;
      unique : bool;
      axis : int;
      indices : int64_t;
      updates : ('a, 'b) Value.t;
      into : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Update :
      ('a, 'b) Value.t * int64_t * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Unfold : {
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
      x : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Fold : {
      output_size : int array;
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
      x : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Matmul : ('a, 'b) Value.t * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Fft : {
      inverse : bool;
      axes : int array;
      x : (Complex.t, 'b) Value.t;
    }
      -> (Complex.t, 'b) Value.t t
  | Rfft : {
      dtype : (Complex.t, 'c) Nx_dtype.t;
      axes : int array;
      x : (float, 'b) Value.t;
    }
      -> (Complex.t, 'c) Value.t t
  | Irfft : {
      dtype : (float, 'c) Nx_dtype.t;
      axes : int array;
      s : int array option;
      x : (Complex.t, 'b) Value.t;
    }
      -> (float, 'c) Value.t t
  | Contiguous : ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Cholesky : { upper : bool; x : ('a, 'b) Value.t } -> ('a, 'b) Value.t t
  | Qr : {
      reduced : bool;
      x : ('a, 'b) Value.t;
    }
      -> (('a, 'b) Value.t * ('a, 'b) Value.t) t
  | Lu : ('a, 'b) Value.t -> (('a, 'b) Value.t * int64_t * int64_t) t
  | Svd : {
      full_matrices : bool;
      x : ('a, 'b) Value.t;
    }
      -> (('a, 'b) Value.t
         * (float, Nx_dtype.float64_elt) Value.t
         * ('a, 'b) Value.t)
         t
  | Eig : {
      vectors : bool;
      x : ('a, 'b) Value.t;
    }
      -> ((Complex.t, Nx_dtype.complex64_elt) Value.t
         * (Complex.t, Nx_dtype.complex64_elt) Value.t option)
         t
  | Eigh : {
      vectors : bool;
      x : ('a, 'b) Value.t;
    }
      -> ((float, Nx_dtype.float64_elt) Value.t * ('a, 'b) Value.t option) t
  | Solve_triangular : {
      upper : bool;
      transpose : bool;
      unit_diag : bool;
      a : ('a, 'b) Value.t;
      b : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Move : ('a, 'b) Value.t * move -> ('a, 'b) Value.t t
  | Place : Placement.t * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Read : { by : string; x : ('a, 'b) Value.t } -> Rig.Buffer.t t
  | Check : {
      ok : (bool, Nx_dtype.bool_elt) Value.t;
      data : Value.packed list;
      fail : int array -> Value.packed list -> exn;
    }
      -> unit t

val name : 'r t -> string
val operands : 'r t -> Value.packed list

type mapper = { f : 'a 'b. ('a, 'b) Value.t -> ('a, 'b) Value.t }

val map_operands : mapper -> 'r t -> 'r t
val pp : Format.formatter -> 'r t -> unit

val shape : ('a, 'b) Value.t t -> int array
(** {!Nx.Op.shape}. A [Read] has no shape and must not be passed. *)

val dtype : ('a, 'b) Value.t t -> ('a, 'b) Nx_dtype.t
(** {!Nx.Op.dtype}. A [Read] has no dtype and must not be passed. *)

(** {1:shapes Result shapes} *)

val pad_shape : (int * int) array -> int array -> int array
(** [pad_shape padding s] is [s] with [before + after] added to each axis,
    [padding] holding [(before, after)] per axis. *)

val cat_shape : int -> int array list -> int array
(** [cat_shape axis shapes] is the first of [shapes] with its [axis] the sum
    of theirs.

    Raises [Invalid_argument] if [shapes] is empty. *)

val window_counts :
  int array -> int array -> int array -> (int * int) array -> int array ->
  int array
(** [window_counts kernel_size stride dilation padding spatial] is the number
    of windows along each axis of extents [spatial], and [0] where the
    dilated kernel is longer than the padded extent. *)

val unfold_shape :
  int array -> int array -> int array -> (int * int) array -> int array ->
  int array
(** [unfold_shape kernel_size stride dilation padding s] is [s] with its last
    [Array.length kernel_size] axes replaced by two, the number of the
    kernel's elements and the number of windows ({!window_counts}). *)

val fold_shape : 'a array -> 'a array -> 'a array
(** [fold_shape output_size s] is [s] with its last two axes replaced by
    [output_size]. *)

val matmul_shape : int array -> int array -> int array
(** [matmul_shape sa sb] is the shape of the product of [sa] and [sb], whose
    leading batch axes broadcast, each of rank at least 2. *)

val rfft_shape : int array -> int array -> int array
(** [rfft_shape axes s] is [s] whose last transformed axis of [n] holds
    [n / 2 + 1] values. *)

val irfft_shape : int array -> int array option -> int array -> int array
(** [irfft_shape axes sizes s] is [s] whose last transformed axis holds the
    last of [sizes], or [2 (n - 1)] for an axis of [n] without them. *)
