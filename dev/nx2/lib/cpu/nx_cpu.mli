(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for host memory.

    [computes_on d] is {!Rig.shares_host_memory}[ d]. Each kernel computes from
    operands of [dst]'s shape and any layouts, on memory the host addresses. It
    claims every operand through the door of [nx_array.h] for the extent of its
    call. It answers [Done] once its work is done, or, before any write, a
    refusal: the door's, or [Shape_mismatch] if the operands' shapes differ.

    [apply0] computes [Fill] at every dtype of a byte or more and [Iota] at
    float32, float64 and the 8- to 64-bit integers, and declines the others.
    [apply1] to [apply3] compute every kind at every dtype of its domain. A
    [Copy] whose operand's dtype is not [dst]'s is refused with
    [Wrong_dtype]. [map] computes a program of plain loads whose outputs,
    loads and coordinate axes number at most [NX_MAX_OPERANDS] of
    [nx_array.h], and whose nodes, evaluated in the program's order, hold at
    most 256 values at once, a value held from the node that computes it
    through the last node that reads it, or to the end for an output; it
    answers [Declined] for the others and for a program that bitcasts a
    sub-byte dtype.

    [contract] computes contractions in [float32] or [float64] whose [dst]
    has their [acc]'s dtype, whose [a], [b] and [init] each convert exactly
    into [acc], and whose layouts {!Nx_kernel.Contract_view} groups; it
    answers [Declined] for the others. [float32] holds [bool], [bit], the
    integers of at most 16 bits and the floats of at most 32; [float64] also
    [int32], [uint32] and [float64].

    A float contraction adds in its own order ({!Nx_kernel.section-orders}).
    With no products, an output is [init]'s bits, or [+0]. Otherwise a NaN
    result's payload is unspecified.

    [reduce] and [scan] compute one [Sum], [Prod], [Max] or [Min] of a
    program's one operand into its own dtype, read plain or padded, at
    [float32], [float64] and the 8- to 64-bit integers, and [Max] and [Min]
    at [bool]; [scan] also computes [Moments] so at [float32] and
    [float64]. They answer [Wrong_dtype] where {!Nx_kernel.accepts} is
    [false] of a reduction and its output's dtype, and [Declined] for the
    other cases. A padded operand is
    first copied padded into host memory, which the call holds until it
    returns. They refuse with [Shape_mismatch] what {!Nx_kernel.Shape.padded}
    and {!Nx_kernel.Shape.reduce} or {!Nx_kernel.Shape.scan} answer [Error]
    for, and destinations whose shapes are not the ones they give.

    A float sum or product adds each output's terms in lane order
    ({!Nx_kernel.section-orders}). A scan cuts each slice along its
    axis into chunks of 4096 from its start. A chunk's total is its terms
    reduced in that order. The carry into the first chunk is [+0] ([1]), into
    the next the carry into this one combined with this one's total; each
    result of a chunk is its carry combined with the chunk's terms up to it,
    left to right.

    [gather] and [scatter] compute every case, at every dtype, and refuse with
    [Shape_mismatch] operands and destinations whose shapes do not fit. A
    scatter's [Add] adds a target's updates left to right.

    [sort] computes every dtype of a byte or more and declines the sub-byte
    ones. [fold] computes [float32], [float64] and the 8- to 64-bit integers
    and declines the others. [assemble] computes every dtype. The three
    refuse with [Shape_mismatch] operands and destinations whose shapes do
    not fit.

    [c2c], [r2c], [c2r] and the routines of matrices answer [Wrong_dtype]
    for dtypes {!Nx_kernel.S} does not take. The transforms decline the
    others. [cholesky], [lu], [qr], [solve_triangular], [eigh] and [svd]
    compute them; [eig] declines them. The six refuse with
    [Shape_mismatch] what {!Nx_kernel.Shape} answers [Error] for and results
    of other shapes. A matrix's results are a function of its values alone,
    whatever its batch, its layout or the threads. With
    [γ(k) = k u / (1 - k u)] and [ε(k) = k η], [u] the dtype's unit
    roundoff and [η] its least positive subnormal (2{^-149} or 2{^-1074}),
    [k] doubled on complex dtypes, and [|·|] taken elementwise, the bounds
    below hold. Their [ε] terms bound what rounding below the least normal
    number adds:
    - [cholesky]: [L Lᴴ = a + Δ] with
      [|Δ_ij| <= γ(n + 1) (|L| |Lᴴ|)_ij + ε(n + 1) (1 + L_jj)]. The
      [Upper] factor is the [Lower] one's conjugate transpose, bit for bit.
    - [lu]: [L U = P a + Δ] with
      [|Δ_ij| <= γ(2k) (|L| |U|)_ij + ε(2k) (1 + |U_pp|)], [p = min(i, j)]
      and [P] the rows' permutation.
    - [qr]: with [g = γ(k (2m + 10))] and [h = ε(k (2m + 10))], each column
      [a_j] has [‖a_j - q r_j‖₂ <= g ‖a_j‖₂ + √m h], and each element of
      [qᴴq - I] is at most [g] in magnitude. [q] is the product of [k]
      Householder reflectors [I - τ v vᴴ], its first [k] columns when
      reduced, whole when complete, and [r]'s diagonal is real, of either
      sign.
    - [solve_triangular]: [(op a + Δ) x = b + δ] with [|Δ| <= γ(2n) |op a|]
      and [|δ_i| <= ε(2n) (1 + |(op a)_ii|)] for each column of [b], [op a]
      the triangle read, conjugate-transposed under [adjoint], ones on its
      diagonal under [unit_diagonal].
    - [eigh]: with [g = γ(16n)] and [h = ε(16n)],
      [‖a - v diag w vᴴ‖_F <= g ‖a‖_F + h] and [‖vᴴv - I‖_F <= g]; each
      eigenvalue within [g ‖a‖₂ + h] of the exact one, with [v] or without.
    - [svd]: with [g = γ(16 max(m, n) + 100)] and
      [h = ε(16 max(m, n) + 100)], [‖a - u diag s vh‖_F <= g ‖a‖_F + h]
      over [u]'s first [k] columns and [vh]'s first [k] rows,
      [‖uᴴu - I‖_F <= g] and [‖vh vhᴴ - I‖_F <= g], reduced or complete;
      each singular value within [g ‖a‖₂ + h] of the exact one, with
      [vectors] or without. The 100 is the iteration's test of convergence
      on the bidiagonal, which drops a superdiagonal element below
      [u{^7/8}] times its neighbours. Values alone may differ from the
      vectors' call's in their last bits. *)

include Nx_kernel.S

(**/**)

val reduce_on :
  threads:int ->
  Nx_kernel.Prog.t ->
  axes:int array ->
  (Nx_kernel.reduction * int) array ->
  dsts:Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer
(** [reduce_on ~threads] is {!reduce} on one thread where [threads] is [1],
    on as many as its job gives where it is [0]. The bits are the same. *)

val scan_on :
  threads:int ->
  Nx_kernel.Prog.t ->
  axis:int ->
  Nx_kernel.reduction * int ->
  dsts:Nx_array.any array ->
  Nx_kernel.load array ->
  Nx_array.answer
(** [scan_on ~threads] is {!scan}, threaded as {!reduce_on}. *)

(**/**)
