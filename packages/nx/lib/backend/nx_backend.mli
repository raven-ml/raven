(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What computes on arrays.

    A backend is kernels over the arrays of one device: one function per
    operation nx computes, each reading its operands and writing its result into
    [dst], an array nx allocated on the same device, C-contiguous from its first
    element. nx places the operands, allocates the results and calls the backend
    once per device, so a backend never sees nx's values, a placement or another
    device. The kinds of nx's operations are here too: operations of one kind
    share their operands' and results' types, and the kind names the
    mathematical function, after the [Nx] function it implements. *)

(** {1:kinds Kinds} *)

(** The type for elementwise functions of one operand, of its dtype. *)
type unary =
  | Neg
  | Recip
  | Abs
      (** On complex dtypes, the modulus in the real component and zero in the
          imaginary one, computed without intermediate overflow: components
          whose [re² + im²] would saturate still give a finite modulus. *)
  | Sqrt
  | Sign  (** [-1], [0] or [1]; NaN for a NaN. *)
  | Exp
  | Log
  | Sin  (** In radians, as every trigonometric function. *)
  | Cos
  | Tan
  | Asin  (** In \[[-π/2], [π/2]\]. *)
  | Acos  (** In \[[0], [π]\]. *)
  | Atan  (** In \[[-π/2], [π/2]\]. *)
  | Sinh
  | Cosh
  | Tanh
  | Trunc  (** Toward zero. The four roundings are the identity on integers. *)
  | Ceil  (** Toward positive infinity. *)
  | Floor  (** Toward negative infinity. *)
  | Round  (** To the nearest integer, half away from zero (C's [round]). *)
  | Erf  (** The error function, [2/√π ∫₀ˣ e^(-t²) dt]. *)

(** The type for elementwise functions of two operands of one shape and dtype,
    of that dtype. *)
type binary =
  | Add
  | Sub
  | Mul
  | Fdiv
      (** The IEEE 754 quotient, of float and complex operands. nx picks [Fdiv]
          or [Idiv] by dtype; a kernel never inspects it here. *)
  | Idiv  (** The integer quotient truncated toward zero, of integers. *)
  | Mod
      (** The remainder of the division, whose sign follows the dividend: C's
          [%] on integers, [fmod] on floats. *)
  | Pow  (** [a] to the power [b]. *)
  | Atan2  (** The angle of [(b, a)] in radians, in \]-π, π\]. *)
  | Maximum
      (** The IEEE 754 maximum on floats: NaN propagates and [-0] orders below
          [+0], as for [Minimum], [Max] and [Min]. *)
  | Minimum
  | And  (** Bitwise on integers, logical on booleans, as [Or] and [Xor]. *)
  | Or
  | Xor

(** The type for elementwise comparisons of two operands of one shape and dtype,
    to booleans. *)
type compare = Equal | Not_equal | Less | Less_equal

(** The type for associative reductions, over axes ([Nx.sum]) or as running
    values along one axis ([Nx.cumsum]). *)
type reduce = Sum | Prod | Max | Min

(** The type for the position of an extreme along one axis, as [int32]. The
    first of equal extremes is taken. *)
type arg_reduce = Argmax | Argmin

(** {1:backends Backends} *)

exception Refused of string
(** [Refused reason] is raised by a kernel that does not run its arguments,
    before it writes anything. [reason] names the backend, the operation and
    what it refuses, as in ["counting: matmul: no float64"]. *)

exception
  Linalg_error of {
    op : string;
    kind : [ `Not_positive_definite | `Singular | `No_convergence ];
  }
(** [Linalg_error { op; kind }] is raised by a linear-algebra kernel whose
    computation fails on its values: [op] is the operation, as in ["cholesky"],
    and [kind] the failure: a matrix that is not positive-definite, a singular
    matrix, or an iteration that did not converge. A precondition on shapes or
    dtypes that slips past nx raises [Invalid_argument] or [Failure]. *)

type int32_array = (int32, Nx_dtype.int32_elt) Nx_array.t
(** The type for int32 arrays: indices, and Threefry's words. Every index a
    kernel reads or writes is an [int32]: axes longer than [2{^ 31} - 1] are
    unsupported, and the limit is not checked. *)

(** The type for backend implementations.

    Every function but [name] and [runs_on] is a kernel. Its operands and its
    destinations are arrays in the memory of one device the backend runs on; it
    writes each destination whole, or raises {!Refused} before it writes
    anything. Operands may be strided, broadcast (zero strides) or offset;
    destinations are C-contiguous from their first element and share no memory
    with the operands. A kernel may submit its work and return: the device's
    timeline orders it.

    nx guarantees, for every kernel: operands have the shapes and dtypes the
    operation's [Nx] function gives them (the two operands of an elementwise
    kernel have one shape, after broadcasting, and one dtype, after promotion);
    axes are in range, non-negative and, where there are several, distinct; the
    reduced axes of [Max], [Min], [Argmax] and [Argmin] are not empty; and each
    destination has the result's shape and dtype. *)
module type S = sig
  val name : string
  (** [name] is the backend's name, as placements print it. *)

  val runs_on : Nx_device.t -> bool
  (** [runs_on d] is [true] iff the kernels compute on arrays in [d]'s memory.
  *)

  (** {1:elementwise Elementwise} *)

  val unary : unary -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit
  (** [unary k x ~dst] writes [k] of each element of [x] into [dst]. *)

  val binary :
    binary ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [binary k a b ~dst] writes [k] of the elements of [a] and [b] at each
      index into [dst]. *)

  val compare :
    compare ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:(bool, Nx_dtype.bool_elt) Nx_array.t ->
    unit
  (** [compare k a b ~dst] writes the comparison [k] of the elements of [a] and
      [b] at each index into [dst]. *)

  val where :
    (bool, Nx_dtype.bool_elt) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [where c a b ~dst] writes [a]'s element where [c]'s is [true] and [b]'s
      elsewhere into [dst]. The three operands have one shape. *)

  val cast : ('a, 'b) Nx_array.t -> dst:('c, 'd) Nx_array.t -> unit
  (** [cast x ~dst] writes [x]'s elements converted to [dst]'s dtype into [dst].
      A float becomes an integer truncated toward zero; an integer becomes a
      float rounded, which may lose precision. A complex becomes a real by its
      real component, and a real becomes a complex with a zero imaginary one:
      nx's complex accessors rely on both, so a cast through the modulus would
      change their results. *)

  val threefry : int32_array -> int32_array -> dst:int32_array -> unit
  (** [threefry key counter ~dst] writes the Threefry-2x32 hash of each word
      pair of [counter] under [key] into [dst]. This is normative: 20 rounds,
      with the standard rotation constants and key schedule, so that a
      [(key, counter)] pair gives bit-identical words on every backend and under
      every lowering, eager or compiled. *)

  (** {1:reductions Reductions and scans} *)

  val reduce :
    reduce ->
    axes:int array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [reduce k ~axes x ~dst] writes [x] reduced by [k] over [axes] into [dst],
      which drops the reduced axes: nx reinserts them when a caller keeps them.
  *)

  val scan :
    reduce -> axis:int -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit
  (** [scan k ~axis x ~dst] writes the inclusive running [k] of [x] along [axis]
      into [dst], of [x]'s shape. *)

  val arg_reduce :
    arg_reduce -> axis:int -> ('a, 'b) Nx_array.t -> dst:int32_array -> unit
  (** [arg_reduce k ~axis x ~dst] writes the position of the extreme of [x]
      along [axis], the first of equal ones, into [dst], which drops [axis]. *)

  (** {1:sorting Sorting} *)

  val sort :
    descending:bool ->
    axis:int ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [sort ~descending ~axis x ~dst] writes [x]'s elements along [axis] in the
      order {!argsort} gives into [dst], bit for bit. *)

  val argsort :
    descending:bool ->
    axis:int ->
    ('a, 'b) Nx_array.t ->
    dst:int32_array ->
    unit
  (** [argsort ~descending ~axis x ~dst] writes the positions that sort [x]
      along [axis] into [dst]. The sort is stable, [-0] orders below [+0], and
      NaNs come last in either direction. *)

  (** {1:assembly Assembly} *)

  val pad :
    (int * int) array ->
    'a ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [pad padding v x ~dst] writes [x] with [fst padding.(i)] elements of value
      [v] before it and [snd padding.(i)] after it along each axis [i] into
      [dst]. *)

  val cat :
    axis:int -> ('a, 'b) Nx_array.t list -> dst:('a, 'b) Nx_array.t -> unit
  (** [cat ~axis xs ~dst] writes the arrays of [xs] one after the other along
      [axis] into [dst]. [xs] is not empty, and its arrays have one shape but
      along [axis]. *)

  val contiguous : ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit
  (** [contiguous x ~dst] copies [x]'s elements into [dst]. *)

  (** {1:indexed Indexed access} *)

  val gather :
    axis:int ->
    int32_array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [gather ~axis indices x ~dst] writes, at each index of [indices], [x]'s
      element at that index with its [axis] component replaced by the index
      there, into [dst], of [indices]' shape. [indices] and [x] have one rank.
      An index outside \[[0], [n]), [n] being [x]'s size along [axis], negative
      included, reads zero, and the kernel touches no memory outside [x]. *)

  val scatter :
    mode:[ `Set | `Add ] ->
    unique:bool ->
    axis:int ->
    indices:int32_array ->
    updates:('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [scatter ~mode ~unique ~axis ~indices ~updates x ~dst] writes [x] into
      [dst], with each element of [updates] set ([`Set]) or added ([`Add]) at
      the position of [indices] at the same index along [axis]. [indices] and
      [updates] have one shape and [x]'s rank. Under [`Set] the last of
      duplicate positions wins; under [`Add] every update adds. With [unique],
      the caller asserts the positions are distinct, so updates may land in any
      order: a position selected more than once then holds an unspecified one of
      its updates, or an unspecified sum, and every other position is exact. An
      update at an index outside \[[0], [n]), [n] being [x]'s size along [axis],
      negative included, is dropped, and the kernel touches no memory outside
      [dst]. *)

  val update :
    ('a, 'b) Nx_array.t ->
    starts:int32_array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [update x ~starts v ~dst] writes [x] with [v] at the window whose corner
      is [starts] and whose extent is [v]'s shape into [dst]. [starts] is a
      vector of [x]'s rank, read when the kernel runs, and already clamped so
      that the window fits; [v] has [x]'s rank and dtype. *)

  (** {1:windows Windows}

      Sliding windows over the last [k] axes, [k] being the length of
      [kernel_size], and their inverse: convolutions are [unfold], a reshape and
      a product, and pooling [unfold] and a reduction. Every array parameter has
      [k] elements, all positive but the padding. *)

  val unfold :
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [unfold ~kernel_size ~stride ~dilation ~padding x ~dst] writes the windows
      of [x] of shape [(leading..., spatial...)] into [dst] of shape
      [(leading..., product kernel_size, l)], [l] being the number of windows:
      [((spatial + before + after - (dilation (kernel - 1) + 1)) / stride + 1)]
      along each axis, multiplied. *)

  val fold :
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [fold ~output_size ~kernel_size ~stride ~dilation ~padding x ~dst] writes
      the windows of [x] of shape [(leading..., product kernel_size, l)] back
      into [dst] of shape [(leading..., output_size...)], summing where windows
      overlap. The parameters are those of an {!unfold} to [x]'s shape. *)

  (** {1:products Products} *)

  val matmul :
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [matmul a b ~dst] writes the product of the matrices of the last two axes
      of [a] and [b] into [dst], their leading axes broadcast. [a]'s last axis
      has [b]'s next-to-last size. *)

  (** {1:fourier Fourier transforms}

      The transforms are unnormalized: nx applies the normalization. *)

  val fft :
    inverse:bool ->
    axes:int array ->
    (Complex.t, 'b) Nx_array.t ->
    dst:(Complex.t, 'b) Nx_array.t ->
    unit
  (** [fft ~inverse ~axes x ~dst] writes the discrete Fourier transform of [x]
      over [axes], or its inverse, into [dst]. *)

  val rfft :
    axes:int array ->
    (float, 'b) Nx_array.t ->
    dst:(Complex.t, 'c) Nx_array.t ->
    unit
  (** [rfft ~axes x ~dst] writes the transform of real [x] over [axes] into
      [dst]: the non-redundant half, [n / 2 + 1] bins, along the last of [axes].
  *)

  val irfft :
    axes:int array ->
    s:int array option ->
    (Complex.t, 'b) Nx_array.t ->
    dst:(float, 'c) Nx_array.t ->
    unit
  (** [irfft ~axes ~s x ~dst] writes the inverse transform over [axes] of the
      half-spectrum [x] into real [dst]. [s] gives the output sizes along
      [axes]; with none, [dst]'s last transformed size is [2 (m - 1)], [m] being
      [x]'s. A last size that needs fewer or more bins than [x] has truncates or
      zero-pads the half-spectrum.

      The result is defined for every [x], symmetric or not: the real part of
      the inverse transform of the spectrum whose last axis is extended with the
      conjugated mirror of bins [1] to [ceil (n / 2) - 1], [n] being the last
      output size. The imaginary parts of bin [0] and, for an even [n], bin
      [n / 2] never contribute. A kernel implements exactly this linear map:
      differentiation rules are its transpose, off the symmetric subspace too.
  *)

  (** {1:linalg Linear algebra}

      The last two axes hold the matrices and the leading ones are batch axes.
      nx gives the matrices their required shapes: square where a kernel needs
      it, and matching for a solve. *)

  val cholesky :
    upper:bool -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit
  (** [cholesky ~upper x ~dst] writes the factor of the positive-definite [x]
      into [dst]: [L] with [x = L Lᴴ], or [U] with [x = Uᴴ U] under [upper].

      Raises {!Linalg_error} [`Not_positive_definite] if [x] is not
      positive-definite. *)

  val qr :
    reduced:bool ->
    ('a, 'b) Nx_array.t ->
    q:('a, 'b) Nx_array.t ->
    r:('a, 'b) Nx_array.t ->
    unit
  (** [qr ~reduced x ~q ~r] writes the factors of [x = Q R] into [q] and [r]:
      [Q] with orthonormal columns, unitary on complex matrices, and [R] upper
      triangular. With [reduced], for [x] of [m] rows and [n] columns and
      [k = min m n], [q] has [k] columns and [r] [k] rows; otherwise [q] is
      square.

      Raises {!Linalg_error} [`No_convergence] if the factorization does not
      converge. *)

  val lu :
    ('a, 'b) Nx_array.t ->
    lu:('a, 'b) Nx_array.t ->
    pivots:int32_array ->
    perm:int32_array ->
    unit
  (** [lu x ~lu ~pivots ~perm] writes the factorization of [x] with partial
      pivoting. For [x] of [m] rows, [n] columns and [k = min m n]:
      - [lu] has [x]'s shape and packs both factors: the unit lower-triangular
        [L] strictly below the diagonal (its ones are not stored), and the
        upper-triangular [U] on and above it.
      - [pivots] has [k] elements per matrix: at step [j], rows [j] and
        [pivots.{j}] were interchanged, the row of largest magnitude in column
        [j] at or below the diagonal ([|re| + |im|] on complex), the first of
        equal ones.
      - [perm] has [m] elements per matrix: the row order the interchanges
        produce, so that row [i] of [L U] is row [perm.{i}] of [x].

      A singular [x] is not an error: a zero pivot stays in [U] and its column
      of [L] is left unscaled. *)

  val svd :
    ('a, 'b) Nx_array.t ->
    u:('a, 'b) Nx_array.t ->
    s:(float, Nx_dtype.float64_elt) Nx_array.t ->
    vt:('a, 'b) Nx_array.t ->
    unit
  (** [svd x ~u ~s ~vt] writes [x = U diag(S) Vᴴ] into [u], [s] and [vt]. [s]
      holds the singular values in descending order. The shapes of [u] and [vt]
      say whether the factors are full, square, or thin, of [min m n] columns
      and rows.

      Raises {!Linalg_error} [`No_convergence] if the iteration does not
      converge. *)

  val eig :
    ('a, 'b) Nx_array.t ->
    values:(Complex.t, Nx_dtype.complex64_elt) Nx_array.t ->
    vectors:(Complex.t, Nx_dtype.complex64_elt) Nx_array.t option ->
    unit
  (** [eig x ~values ~vectors] writes the eigenvalues of the general square [x]
      into [values] and, when [vectors] is given, its eigenvectors into them.
      Without [vectors] the kernel does not accumulate them.

      Raises {!Linalg_error} [`No_convergence] if the iteration does not
      converge. *)

  val eigh :
    ('a, 'b) Nx_array.t ->
    values:(float, Nx_dtype.float64_elt) Nx_array.t ->
    vectors:('a, 'b) Nx_array.t option ->
    unit
  (** [eigh x ~values ~vectors] writes the eigenvalues of the symmetric or
      Hermitian [x] into [values] and, when [vectors] is given, its
      eigenvectors, of [x]'s dtype, into them. Without [vectors] the kernel does
      not accumulate them.

      Raises {!Linalg_error} [`No_convergence] if the iteration does not
      converge. *)

  val solve_triangular :
    upper:bool ->
    transpose:bool ->
    unit_diag:bool ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
  (** [solve_triangular ~upper ~transpose ~unit_diag a b ~dst] writes the
      solution [x] of [A x = b] into [dst], [A] being [a], upper triangular
      under [upper] and lower otherwise, or of [Aᴴ x = b] under [transpose]: the
      conjugate transpose on complex, the plain one on reals. Under [unit_diag]
      the diagonal of [a] is taken as ones. [b] is a vector of shape [(..., n)]
      or right-hand sides of shape [(..., n, nrhs)].

      Raises {!Linalg_error} [`Singular] if [a] is singular: a zero on the
      diagonal when [unit_diag] is [false]. *)
end

type t
(** The type for backends. *)

val make : (module S) -> t
(** [make k] is the backend of kernels [k]. Each [make] is a backend of its own:
    a library makes its backend once and shares the value. *)

val kernels : t -> (module S)
(** [kernels b] is [b]'s kernels, for nx to call. *)

val name : t -> string
(** [name b] is [b]'s name. *)

val runs_on : t -> Nx_device.t -> bool
(** [runs_on b d] is [true] iff [b] computes on arrays in [d]'s memory. *)

val equal : t -> t -> bool
(** [equal b b'] is [true] iff [b] and [b'] are the same {!make}. *)
