(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Composed operations.

    Higher-level operations built from movement, element-wise, and reduction
    primitives. *)

(** {1 Assignment} *)

val assign : Tensor.t -> Tensor.t -> Tensor.t
(** [assign t x] writes the values of [x] into the storage of [t] and returns
    [t]. The write is recorded in the graph as an effect on [t]'s buffer:
    nothing executes until a realization. A view of storage, including a
    bitcast view, writes into the viewed region and repoints live aliases to
    depend on the write. A partial write into a pending contiguous tensor
    first materializes that tensor's storage.

    If [t] is a pending value without storage, its old computation is discarded
    and [x] initializes fresh storage. This also applies when overwriting a
    whole pending contiguous tensor. [x] is broadcast to the shape of [t].
    A weak [t] first acquires fresh storage at the dtype selected by
    {!Tolk_uop.Uop.commit_dtype}. A weak [x]
    promotes with [t]'s dtype; the result must match [t]'s dtype. Once [t] has
    a concrete dtype, assigning it to itself is a no-op. For a storage write,
    a value on another device is copied to the destination during preparation.

    @raise Invalid_argument
      if the dtypes differ after weak promotion, or the tensors have
      incompatible sharding axes. *)

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
    must share every axis except [dim].

    @raise Invalid_argument
      if the shapes differ off [dim], or if [dim] is symbolic and the extents
      along it are not all equal. *)

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

val pad_to : ?value:Tensor.scalar -> Tensor.t -> int option list -> Tensor.t
(** [pad_to t dims] pads [t] at the end of each axis to the corresponding
    size in [dims], filling new elements with [value] (default [0]). [None]
    keeps the current size. An unchanged shape returns [t].

    @raise Invalid_argument if the ranks differ or a target size is smaller
      than its current size. *)

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
    fills the unaddressed axes. Axis lengths may be symbolic. A slice whose
    bounds remain symbolic requires a unit step and a provably non-negative
    length; negative integer bounds count from the symbolic axis length.
    Use {!Movement.symbolic_shrink} for explicit symbolic bound expressions.

    An integer-tensor index ({!Movement.T}) performs advanced indexing: its
    elements gather positions along the axis, and several such indices broadcast
    against each other into a shared leading block of axes. Index-tensor
    shapes and unindexed axes may be symbolic; indexed axes must have concrete
    lengths. Out-of-bounds gathered positions read as [0].

    @raise Invalid_argument
      if the indices are malformed for the rank of [t] (see
      {!Movement.normalize_indices} and {!Movement.parse_view_index}) or an
      index tensor is not integer-typed, is on a different device, or indexes
      a symbolic-size axis. *)

val one_hot : Tensor.t -> int -> Tensor.t
(** [one_hot index num_classes] adds a trailing axis of length [num_classes]
    and sets, for each index value, a [1] at that position and [0] elsewhere.
    [index] must be an integer tensor. *)

val argmax : ?axis:int -> ?keepdim:bool -> Tensor.t -> Tensor.t
(** [argmax t] is the integer index of the maximum along [axis]. With no [axis]
    the whole tensor is flattened first. On ties the first occurrence wins. *)

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

val scatter_indexed :
  Tensor.t -> dim:int -> Tensor.t -> Tensor.t -> mode:[ `Set | `Add ] ->
  unique:bool -> Tensor.t
(** [scatter_indexed t ~dim index src ~mode ~unique] is [t] after each element
    of [src] has been written into it at the position given by the matching
    element of [index] along [dim], at a cost in the number of updates instead
    of the size of [t]. The write lands in [t]'s own storage, as {!assign}
    does: [t] is a buffer, or a buffer after the effects that wrote it, and
    every later read of that storage sees the write. To keep [t], scatter into
    a {!Creation.clone} of it.

    [index] and [src] have the rank of [t]; [src] has the extent of [t] on
    every axis but [dim], where it has the extent of [index]; off [dim],
    [index] has the extent of [t] or one, and an index broadcast along an axis
    is read once for that axis. A weak [src] promotes with [t]'s dtype before
    the write; the result must match [t]'s dtype. A concrete [src] must already
    have that dtype.

    Under [`Set] the last of the updates aimed at one position wins, in the
    order of [index] along [dim]; under [`Add] they all accumulate onto [t]'s
    value, in that order. An index outside \[[0];[n-1]\], with [n] the extent
    of [t] along [dim], writes nothing. [unique] promises that no two updates
    share a position, which lets them run in any order. Where the promise is
    broken, a position aimed at more than once holds an unspecified one of its
    updates under [`Set] and an unspecified value under [`Add]; every other
    position is exact.

    {!scatter} and {!scatter_reduce} compute the same values for in-range
    indices, as a new tensor, at a cost in the size of [t] times the number of
    updates.

    When [t] is split across devices, each device writes its own slice. Split
    off [dim], [index] and [src] are split along the same axis, or [index] has
    extent one there; split along [dim], they are whole on every device, and
    each device keeps the updates that land in its rows.

    @raise Invalid_argument
      if [t] is not storage, if ranks or extents disagree as above, if [index]
      or [src] is split otherwise than [t] allows, if [src] does not have the
      dtype of [t], or if [index] is not an integer
      tensor. *)

val quant_matmul :
  ?ids:Tensor.t -> Tensor.t -> codes:Tensor.t -> scales:Tensor.t -> Tensor.t
(** [quant_matmul ?ids x ~codes ~scales] multiplies rows of [x] by the
    transposes of MXFP4 matrices, decoding the weights in registers.

    [codes] is [\[e; n; k/2\]] uint8, two 4-bit e2m1 codes per byte, the low
    nibble first, and [scales] is [\[e; n; k/32\]] uint8, one power-of-two
    scale 2{^ s-127} per 32 values along [k], 255 standing for NaN. [x] is
    [\[ix; m; k\]] at float32, bfloat16 or float16, and the result is
    [\[i; m; n\]] at [x]'s dtype, where [i] is the length of [ids], or [e]
    without them, and [ix] divides [i]. Instance [t] of the result is block
    [t / (i / ix)] of [x] times the transpose of matrix [ids.(t)], or of
    matrix [t] without [ids]. An id outside \[[0];[e-1]\] selects no matrix:
    its instance is exactly zero, whatever [x] holds there. Products and sums
    are float32 on every device, so a product of a bfloat16 or float16 input
    and a code is exact; each group's partial sum is multiplied by its scale,
    and the result is rounded once.

    Every operand is read as whole storage: a view is copied first. On a GPU,
    when [k] is at least 64, an instance whose id selects no matrix reads its
    id and nothing else, and runs no multiply-adds; elsewhere it reads matrix
    0 and its result is zeroed.

    Over operands split across devices, each device computes its own slice of
    the result, which is split alike: instances split along the first axis of
    [ids] (or of [codes] without ids) and of [x] (unless [ix] is one), rows
    along [x]'s second axis, or columns along the second axis of [codes] and
    [scales]. Whole instances over matrices split along the first axis of
    [codes] and [scales], or over inputs split along [x]'s last axis and the
    last axis of [codes] and [scales], leave each device a partial product,
    which the result sums across the devices at float32 before rounding once:
    ids name matrices by their position in the whole, and each device
    multiplies the instances whose matrix it holds. Without [ids], instances
    split along the first axis of [x] may read matrices split alike.

    @raise Invalid_argument
      if the shapes disagree as above, if the operands are split otherwise, if
      [k] is not a multiple of 32, if
      [codes] or [scales] is not uint8, if [x]'s dtype is not one of the
      three, if [ids] is not an integer tensor, or if no operand is placed on
      a device. *)

val quant_row_bound : Tolk.Renderer.t -> int
(** [quant_row_bound ren] is the most rows a matrix of {!quant_matmul} should
    meet on a device rendering with [ren]: beyond it, decoding the matrix and
    multiplying it with {!block_matmul} costs less. One value per device, at
    every dtype; [0] on a device whose options are not measured. *)

val quant_row_tile : Tolk.Renderer.t -> int
(** [quant_row_tile ren] is the most rows of [x] {!quant_matmul} reads per load
    of a group on a device rendering with [ren]: [1] on a device whose options
    are not measured. A block of more rows costs more than one of this many. *)

val block_row_tiles : Tolk.Renderer.t -> n:int -> k:int -> int list
(** [block_row_tiles ren ~n ~k] is, largest first, the rows a block of
    {!block_matmul} may have for its pinned options to fill their tiles, with
    [n] outputs and [k] inputs on a device rendering with [ren]: empty where no
    options are pinned, and no number of rows fills a tile. *)

val block_matmul :
  ?transpose:bool -> Tensor.t -> Tensor.t -> ids:Tensor.t -> Tensor.t
(** [block_matmul x w ~ids] multiplies each block of rows of [x] by the matrix
    of [w] its id addresses: block [b] of the result is [matmul x.(b)
    w.(ids.(b))] when [ids.(b)] is in \[[0];[e-1]\], and exactly zero
    otherwise, whatever [x.(b)] holds. [x] is [\[nb; m; k\]], [ids] is
    [\[nb\]] and the result is [\[nb; m; n\]] at [x]'s dtype, a float of at
    most 32 bits. [w] is [\[e; k; n\]]; with [transpose] (default [false]) it
    is [\[e; n; k\]] and each block is multiplied by the transpose of its
    matrix, the layout of a linear layer's weight. Products and sums are
    float32 on every device, tensor cores included, so a product of two values
    of a float dtype narrower than float32 is exact, and the result is rounded
    once.

    Each block reads its own matrix in place, where [matmul] over [w] gathered
    by [ids] would copy the gathered matrices. On a GPU, when [k] is at least
    2, a block whose id is out of range reads its id and nothing else and runs
    no multiply-adds; with one input its load is gated instead.

    Over operands split across devices, each device computes its own slice of
    the result, which is split alike: blocks split along the first axis of [x]
    and [ids], rows along [x]'s second axis, or columns along [w]'s output
    axis. Whole blocks over [w] split along its first axis, or over inputs split
    along [x]'s last axis and [w]'s input axis, leave each device a partial
    product, which the result sums across the devices at float32 before
    rounding once: ids name matrices by their position in the whole, and each
    device multiplies the blocks whose matrix it holds.

    @raise Invalid_argument
      if the shapes disagree as above, if the operands are split otherwise, if
      [x] and [w] differ in dtype or are
      not floats of at most 32 bits, if [ids] is not an integer tensor, or if
      no operand is placed on a device. *)

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

val softmin : ?axis:int -> ?dtype:Tolk_uop.Dtype.t -> Tensor.t -> Tensor.t
(** [softmin t] is {!softmax} of the negated input: the same rescaling, with
    the smallest element receiving the largest weight. *)

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
    axes. [stride] defaults to [kernel_size]. *)

(** {1 Ranges} *)

val arange :
  ?stop:int -> ?step:int -> ?dtype:Tolk_uop.Dtype.t -> int -> Tensor.t
(** [arange start] is the 1-D tensor [\[0; 1; ...; start-1\]]. [arange start
    ~stop] ranges over [\[start, stop)], and [~step] sets the spacing (which
    may be negative). The length is [ceil((stop - start) / step)], clamped to
    zero. [dtype] defaults to the default integer type, or [int64] when the
    endpoints require it. Intermediate endpoint and length calculations are
    exact.

    @raise Invalid_argument if [step] is zero, the range is not representable
    in [dtype], or its length exceeds [max_int]. *)

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
