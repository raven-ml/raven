(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Composed operations.

    Higher-level operations built from movement, element-wise, and reduction
    primitives. *)

(** {1 Broadcasting} *)

val broadcasted : ?reverse:bool -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
(** [broadcasted a b] is [(a, b)] broadcast to a common shape and promoted to a
    common dtype; with [~reverse:true] the pair is returned swapped. Linking
    this module installs it as {!Tensor.broadcasted}, which the element-wise
    operations use. *)

(** {1 Assignment} *)

val assign : Tensor.t -> Tensor.t -> Tensor.t
(** [assign t x] writes the values of [x] into the storage of [t] and returns
    [t]. The write is recorded in the graph as an effect on [t]'s buffer:
    nothing executes until a realization, and every read of the buffer built
    after the assignment observes the written values. When [t] is a view (a
    slice of a larger tensor), the write lands in the viewed region and every
    live tensor aliasing the underlying buffer is repointed to depend on it.
    [x] is broadcast to the shape of [t]; assigning a tensor to itself is a
    no-op.

    @raise Invalid_argument
      if the dtypes differ or the tensors live on different devices. *)

(** {1 Statistics} *)

val mean : ?axis:int list -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [mean t] is the arithmetic mean along the reduced axes (default: all).
    Integer inputs produce a float result. *)

val var : ?axis:int list -> ?keepdim:bool -> ?correction:int -> Tensor.t -> Tensor.t
(** [var t] is the variance along the reduced axes. [correction] (default [1])
    is subtracted from the element count in the denominator, giving the
    unbiased estimator by default. *)

val std : ?axis:int list -> ?keepdim:bool -> ?correction:int -> Tensor.t -> Tensor.t
(** [std t] is the standard deviation, the square root of {!var}. *)

val layernorm : ?axis:int list -> ?eps:float -> Tensor.t -> Tensor.t
(** [layernorm t] normalises [t] along [axis] (default [\[-1\]]) to zero mean
    and unit variance, computed with the biased variance estimator and
    stabilised by [eps] (default [1e-5]) inside the square root. *)

(** {1 Joining} *)

val cat : ?dim:int -> Tensor.t -> Tensor.t list -> Tensor.t
(** [cat t others] concatenates [t] and [others] along axis [dim]. All tensors
    must share every axis except [dim]. *)

val stack : ?dim:int -> Tensor.t -> Tensor.t list -> Tensor.t
(** [stack t others] joins equally shaped tensors along a new axis inserted at
    [dim]. *)

(** {1 Matrix multiplication} *)

val dot : ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.t -> Tensor.t
(** [dot a w] contracts the last axis of [a] with the matching axis of [w]:
    the last axis of [w] when [w] is 1-D, otherwise its second-to-last. Leading
    axes broadcast, giving batched matrix multiplication. [dtype] sets the
    accumulation dtype. *)

val matmul : ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.t -> Tensor.t
(** [matmul a b] is [dot a b]. *)

(** {1 Padding} *)

val pad_constant :
  Tensor.t -> (int * int) option list -> Tensor.scalar -> Tensor.t
(** [pad_constant t padding value] pads [t] with the constant [value].
    [padding] gives, per axis in order, the count to add before and after;
    [None] leaves an axis unchanged, and negative counts shrink. When [value]
    is non-zero the result dtype is promoted to hold it. *)

type pad_mode =
  | Constant  (** Fill new positions with a constant. *)
  | Reflect  (** Mirror the edge values, excluding the edge itself. *)
  | Replicate  (** Repeat the edge value. *)
  | Circular  (** Wrap around to the opposite edge. *)

val pad :
  ?mode:pad_mode -> ?value:Tensor.scalar -> Tensor.t ->
  (int * int) option list -> Tensor.t
(** [pad t padding] pads [t] as in {!pad_constant}, with [mode] choosing how new
    positions are filled ([value] applies only to {!Constant}, the default).
    Negative counts shrink the axis after padding.

    @raise Invalid_argument
      for {!Reflect} if a pad count is not smaller than its axis, or for
      {!Circular} if a pad count exceeds its axis (wrapping more than once). *)

(** {1 Scans} *)

val cumsum : ?axis:int -> Tensor.t -> Tensor.t
(** [cumsum t] is the cumulative sum along [axis] (default [0]); the result has
    the same shape as [t]. *)

val cumprod : ?axis:int -> Tensor.t -> Tensor.t
(** [cumprod t] is the cumulative product along [axis] (default [0]). *)

val cummax : ?axis:int -> Tensor.t -> Tensor.t * Tensor.t
(** [cummax t] is [(values, indices)] of the cumulative maximum along [axis]
    (default [0]): [values.(i)] is the maximum of the prefix up to [i], and
    [indices.(i)] is the position of that maximum. *)

val cummin : ?axis:int -> Tensor.t -> Tensor.t * Tensor.t
(** [cummin t] is [(values, indices)] of the cumulative minimum along [axis]. *)

(** {1 Indexing} *)

val getitem : Tensor.t -> Movement.index list -> Tensor.t
(** [getitem t indices] selects a sub-tensor of [t], with one {!Movement.index}
    per axis applied from the outermost inward. Integer indices ({!Movement.I})
    drop their axis, slices ({!Movement.R}, {!Movement.All}) keep a strided
    range, {!Movement.New} inserts a size-[1] axis, and {!Movement.Ellipsis}
    fills the unaddressed axes.

    An integer-tensor index ({!Movement.T}) performs advanced indexing: its
    elements gather positions along the axis, and several such indices broadcast
    against each other into a shared leading block of axes. Out-of-bounds
    gathered positions read as [0].

    @raise Invalid_argument
      if the indices are malformed for the rank of [t] (see
      {!Movement.normalize_indices} and {!Movement.parse_view_index}) or an
      index tensor is not integer-typed. *)

val one_hot : Tensor.t -> int -> Tensor.t
(** [one_hot index num_classes] adds a trailing axis of length [num_classes]
    and sets, for each index value, a [1] at that position and [0] elsewhere.
    [index] must be an integer tensor. *)

val argmax : ?axis:int -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [argmax t] is the integer index of the maximum along [axis]. With no [axis]
    the whole tensor is flattened first. On ties the last occurrence wins. *)

val argmin : ?axis:int -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [argmin t] is the index of the minimum, as {!argmax} on the reflected
    tensor. *)

val sort : ?dim:int -> ?descending:bool -> Tensor.t -> Tensor.t * Tensor.t
(** [sort t] is [(values, indices)] sorting [t] along [dim] (default [-1]),
    ascending unless [descending]. [values] is [t] reordered and [indices] gives
    each sorted element's original position along [dim]. Equal elements keep
    their input order. *)

val argsort : ?dim:int -> ?descending:bool -> Tensor.t -> Tensor.t
(** [argsort t] is the [indices] of {!sort}: the permutation that orders [t]
    along [dim]. *)

val topk : ?dim:int -> ?largest:bool -> ?sorted_:bool -> Tensor.t -> int -> Tensor.t * Tensor.t
(** [topk t k] is [(values, indices)] of the [k] largest elements of [t] along
    [dim] (default [-1]), or the smallest when [largest] is [false], ordered.

    @raise Invalid_argument
      if [k] exceeds the axis length, or [sorted_] is [false] (unordered
      selection is not supported). *)

val gather : Tensor.t -> dim:int -> Tensor.t -> Tensor.t
(** [gather t ~dim index] selects along [dim] using [index]: the output has the
    shape of [index], and output position [p] holds
    [t] at [p] with its [dim] coordinate replaced by [index.(p)]. [index] must
    be an integer tensor with the same rank as [t], and no larger than [t] on
    every axis other than [dim]. *)

val scatter : Tensor.t -> dim:int -> Tensor.t -> Tensor.t -> Tensor.t
(** [scatter t ~dim index src] is a copy of [t] with each element of [src]
    written at the position given by the matching element of [index] along
    [dim]. [index] and [src] must have the same rank as [t]; [index] must be no
    larger than [src], and no larger than [t] off [dim]. When an index repeats,
    the last write wins. Use {!scatter_reduce} to combine colliding writes. *)

val scatter_reduce :
  Tensor.t -> dim:int -> Tensor.t -> Tensor.t ->
  reduce:[ `Sum | `Prod | `Mean | `Amax | `Amin ] ->
  ?include_self:bool -> unit -> Tensor.t
(** [scatter_reduce t ~dim index src ~reduce ()] scatters [src] into [t] along
    [dim] like {!scatter}, but reduces all values landing on the same position
    with [reduce] (sum, product, mean, maximum, or minimum). The original [t]
    value participates in the reduction unless [include_self] is [false]. *)

val masked_select : ?fill_value:Tensor.scalar -> Tensor.t -> Tensor.t -> size:int -> Tensor.t
(** [masked_select t mask ~size] is the 1-D tensor of the elements of [t] where
    [mask] is true, in row-major order, packed into a fixed length [size].
    [mask] must be boolean and broadcast to the shape of [t]. If fewer than
    [size] elements are kept the remainder is filled with [fill_value] (default
    [0]); if more, the excess is dropped. The fixed [size] keeps the result
    shape static (and thus jittable). *)

val nonzero : ?fill_value:Tensor.scalar -> Tensor.t -> size:int -> Tensor.t
(** [nonzero t ~size] is a [size]-by-[rank] integer tensor whose rows are the
    coordinates of the non-zero elements of [t], in row-major order. Rows past
    the number of non-zero elements are filled with [fill_value] (default [0]);
    excess coordinates are dropped. As with {!masked_select}, the fixed [size]
    keeps the shape static. *)

(** {1 Triangular masks} *)

val triu : ?diagonal:int -> Tensor.t -> Tensor.t
(** [triu t] keeps the upper triangle of the last two axes of [t] and zeros the
    rest. [diagonal] shifts the boundary: [0] is the main diagonal, positive
    moves it up, negative down. *)

val tril : ?diagonal:int -> Tensor.t -> Tensor.t
(** [tril t] keeps the lower triangle of the last two axes of [t] and zeros the
    rest, with [diagonal] as in {!triu}. *)

(** {1 Log-space reductions}

    Numerically stable reductions that work in log space, subtracting the
    running maximum before exponentiating. Scalar inputs are returned
    unchanged. *)

val logsumexp : ?axis:int -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [logsumexp t] is [log (sum (exp t))] over [axis] (default: all axes). *)

val softmax : ?axis:int -> ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.t
(** [softmax t] rescales [t] along [axis] (default [-1]) into non-negative
    values summing to [1]. [dtype] casts the shifted input before
    exponentiating. *)

val log_softmax : ?axis:int -> ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.t
(** [log_softmax t] is the logarithm of {!softmax}, computed stably. *)

val logcumsumexp : ?axis:int -> Tensor.t -> Tensor.t
(** [logcumsumexp t] is the cumulative {!logsumexp} along [axis] (default [0]):
    output position [k] is [log (sum (exp t.(0..k)))]. *)

(** {1 Attention} *)

val scaled_dot_product_attention :
  ?attn_mask:Tensor.t -> ?is_causal:bool -> Tensor.t -> Tensor.t -> Tensor.t ->
  Tensor.t
(** [scaled_dot_product_attention q k v] is
    [softmax (q @ k^T / sqrt d) @ v] over the last two axes, where [d] is the
    size of the last axis of [q]. The score product accumulates in at least
    [float32] and the softmax is applied at [q]'s dtype. [attn_mask] is added
    to the scores before the softmax; a boolean mask contributes [0] where
    true and negative infinity where false. [is_causal] (default [false])
    instead masks each query position from attending past its own, as if a
    lower-triangular boolean mask were given.

    @raise Invalid_argument if both [attn_mask] and [is_causal] are given. *)

(** {1 Convolution and pooling}

    These operate on tensors shaped [(batch, channels, spatial...)]. The number
    of spatial axes follows the kernel, so the same functions cover 1-D, 2-D,
    and higher convolutions. [stride] and [dilation] accept a single-element
    list, broadcast to every spatial axis. [padding] is either a single value
    (all sides), one value per spatial axis, or two per spatial axis. *)

val conv2d :
  ?bias:Tensor.t -> ?groups:int -> ?stride:int list -> ?dilation:int list ->
  ?padding:int list -> ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.t ->
  Tensor.t
(** [conv2d x weight] convolves [x] with [weight] (shape
    [(out_channels, in_channels/groups, kernel...)]), optionally adding [bias]
    and splitting channels into [groups]. [dtype] sets the accumulation dtype. *)

val avg_pool2d :
  ?kernel_size:int list -> ?stride:int list -> ?dilation:int list ->
  ?padding:int list -> Tensor.t -> Tensor.t
(** [avg_pool2d x] averages each sliding window over the spatial axes.
    [stride] defaults to [kernel_size] (non-overlapping windows). *)

val max_pool2d :
  ?kernel_size:int list -> ?stride:int list -> ?dilation:int list ->
  ?padding:int list -> Tensor.t -> Tensor.t
(** [max_pool2d x] takes the maximum of each sliding window over the spatial
    axes. [stride] defaults to [kernel_size].

    @raise Invalid_argument if [x] is not floating point. *)

(** {1 Ranges} *)

val arange :
  ?stop:int -> ?step:int -> ?dtype:Tolk_uop.Dtype.t -> int -> Tensor.t
(** [arange start] is the 1-D tensor [\[0; 1; ...; start-1\]]. [arange start
    ~stop] ranges over [\[start, stop)], and [~step] sets the spacing (which
    may be negative). The length is [ceil((stop - start) / step)], clamped to
    zero. [dtype] defaults to the default integer type. *)

val linspace :
  ?dtype:Tolk_uop.Dtype.t -> float -> float -> int -> Tensor.t
(** [linspace start stop steps] is the 1-D tensor of [steps] values evenly
    spaced over [\[start, stop\]], inclusive of both ends. [dtype] defaults to
    the default float type.

    @raise Invalid_argument if [steps] is negative or [dtype] is boolean. *)

val eye : ?m:int -> ?dtype:Tolk_uop.Dtype.t -> int -> Tensor.t
(** [eye n] is the [n]x[n] identity matrix. [~m] sets a different column count,
    giving an [n]x[m] matrix with ones on the main diagonal. [dtype] defaults
    to the default float type.

    @raise Invalid_argument if [n] or [m] is negative. *)
