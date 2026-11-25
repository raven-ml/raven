(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernel libraries as nx.kernel's suite runs them. *)

(** The type for kinds of no operand. *)
type op0 =
  | Fill of string  (** A [Const] node of the result's dtype and these bits. *)
  | Iota of int  (** A [Coord] node of this axis. *)

(** The type for a kind of any arity. *)
type kind =
  | K0 of op0
  | K1 of Nx_kernel.Prog.op1
  | K2 of Nx_kernel.Prog.op2
  | K3 of Nx_kernel.Prog.op3

type attention = {
  factor : Nx_array.Dtype.any -> Nx_array.Dtype.any -> int -> float;
      (** [factor q acc k] is its error factor [e(k)] for queries, keys and
          values of [q] in the accumulator [acc]
          ({!Nx_kernel.section-bounds}). *)
  epsilon : Nx_array.Dtype.any -> int -> float -> float;
      (** [epsilon acc t lse] is its attention's [ε] in [acc] for a row of [t]
          keys whose [lse] is [lse] ({!Nx_kernel.section-attention}). *)
  zero_weights : Nx_array.Dtype.any -> int -> float;
      (** [zero_weights acc tk] bounds the [|δ|] of each weight of a row whose
          [tk] keys, a power of two, are visible with outputs [0]. The
          contract allows [(1 + u) ε + u ln tk + 5u'], [u' = u / (1 - u)],
          where the library states nothing tighter. *)
  rounds_weights : bool;
      (** [rounds_weights] is [true] iff it may apply weights through a
          rounding to [float16] or [bfloat16] values' dtype. *)
  float64 : bool;
      (** [float64] is [true] iff it computes attention in a [float64]
          accumulator. *)
}
(** The type for what a kernel library states of its attention's errors. *)

(** The type for an axis of a transform, as a library's bound counts it. *)
type axis =
  | Complex  (** A [c2c]'s, or an [r2c]'s or a [c2r]'s but its last. *)
  | R2c_last  (** An [r2c]'s last, of [n] real points. *)
  | C2r_last  (** A [c2r]'s last, of [n] real points. *)

(** The type for how a library states it adds a float contraction's
    products ({!Nx_kernel.section-bounds}). *)
type sums =
  | Blocks
      (** In blocks of at most 256: nx_cpu.mli's and nx_cuda.mli's. *)
  | Running  (** Into one running sum a range: nx_amd.mli's. *)

(** The type for what a library gives at float32 and bfloat16 where an
    operand is subnormal or an exact result is below the least normal
    number, [2{^-126}] in magnitude. *)
type underflow =
  | Gradual  (** Subnormal operands and results, as IEEE 754 states. *)
  | Flush
      (** A subnormal operand reads as a zero of its sign, and a nonzero
          exact result below [2{^-126}] gives a zero of its sign, in every
          kind that computes a float; the kinds whose result is an integer
          value, a boolean or an operand's bits are exact. *)

type backend = {
  name : string;  (** The backend's name in the suite, as ["cpu/base"]. *)
  kernels : (module Nx_kernel.S);  (** Its kernels. *)
  device : Rig.t;  (** The device whose memory its laws compute on. *)
  computes : kind -> Nx_array.Dtype.any -> bool;
      (** [computes k dt] is [true] iff it states it computes [k] at [dt]: the
          result's dtype for [K0], the operand's for [K1] and [K2] (for a
          [Cast], it states a cast computed where [computes] holds at both
          dtypes, or the two are one), and for [K3]
          the second operand's, with a [bool] condition. The suite fails when it
          declines one. *)
  arith : Nx_array.Dtype.any -> bool;
      (** [arith dt] is [true] iff it states it computes on [dt]'s values;
          where it does not, it only moves them, and declines every call
          that computes on one. *)
  around : 'a. (unit -> 'a) -> 'a;
      (** [around f] is [f ()] run as the backend: nx.cpu's under its target
          table. *)
  attention : attention option;
      (** What it states of its attention's errors: [None] iff it declines
          every attention. *)
  lane_order : bool;
      (** [lane_order] is [true] iff it states that its reductions
          associate as nx.cpu's do, in nx.cpu's lane order, and its scans
          in an order of the dtypes, the extents and the scanned axis
          alone, so that neither depends on how a program reads its terms:
          nx_cpu.mli's sums and products and nx_cuda.mli's loops. *)
  transform_factor : axis -> int -> float;
      (** [transform_factor a n] is the [e(n)] it states for an axis [a] of
          [n] points, whose sum over a transform's axes is the [ε] of its
          bound [‖ŷ - y‖₂ <= ε u √N ‖x‖₂]: nx_cpu.mli's and nx_cuda.mli's
          transforms. *)
  scan_order : bool;
      (** [scan_order] is [true] iff it also states that every scan
          associates as nx.cpu's does, in its chunks: nx.cpu's. nx_cuda.mli
          states its own association for a [Sum], [Max] or [Min] scan along
          the destination's innermost axis. *)
  contract_sums : sums;
      (** [contract_sums] is how it states it adds a float contraction's
          products. *)
  underflow : underflow;
      (** [underflow] is what it states it gives at float32 and bfloat16
          below the least normal number. *)
}
(** The type for kernel libraries in one configuration. *)

val contract_beta : backend -> int -> int
(** [contract_beta b n] is nx_kernel.mli's [β(n)] for a float contraction's
    sum of [n] terms on [b]: [⌈5 min(n, 256) / 4⌉ + ⌈n / 128⌉] for
    {!Blocks}, [2n + 1] for {!Running}. *)

val cpus : backend list
(** [cpus] is nx.cpu under each target table the host runs, base first.
    Only the contractions' and linear algebra's kernels differ by table. *)

val cpu : backend
(** [cpu] is nx.cpu under the table it runs from the start: the best the
    host runs. *)

val backends : unit -> backend list
(** [backends ()] is each kernel library under its own configuration:
    {!cpu}, then {!gpus}[ ()]. *)

val gpus : unit -> backend list
(** [gpus ()] is nx.cuda on the machine's NVIDIA GPU 0 through CUDA, as
    ["cuda"], where CUDA opens one and the environment variable
    [NX_KERNELS] is not ["cpu"], opened at the first call. *)

val nv : unit -> backend option
(** [nv ()] is nx.cuda on GPU 0 through NVIDIA's kernel driver and rig.nv,
    as ["cuda/nv"], where {!gpus}[ ()] has GPU 0 and rig.nv opens it too,
    opened at the first call. Its kernels are the cubin CUDA loads, so its
    suites check that its results are CUDA's bits. *)

val amd : unit -> backend option
(** [amd ()] is nx.amd on the machine's AMD GPU 0 through Linux's amdgpu
    driver, as ["amd"], where rig opens one, nx.amd computes on it and the
    environment variable [NX_KERNELS] is not ["cpu"], opened at the first
    call. Each suite but attention's, whose calls nx.amd declines, runs
    its laws on it. Its reductions and scans associate as nx.cpu's. *)

val metal : unit -> backend option
(** [metal ()] is nx.metal on the machine's Metal GPU 0, as ["metal"],
    where rig sees one, nx.metal computes on it and the environment
    variable [NX_KERNELS] is not ["cpu"], opened at the first call. A
    suite runs its laws on it for the families nx.metal computes. *)

val elements : Nx_array.any -> string array
(** [elements x] is [x]'s elements in C order of indices, each as its
    bytes in the host's order, a sub-byte element as its code in one byte,
    read where the host reads them whatever [x]'s device. *)

val same_bytes : Nx_array.any -> Nx_array.any -> bool
(** [same_bytes x y] is [true] iff [x] and [y], of one dtype and shape,
    have the same {!elements}. *)

(** {1:devices Devices} *)

val on : backend -> ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t
(** [on b x] is [x] on [b]'s device with [x]'s layout: [x] itself if it lies
    there, otherwise [x]'s dtype and layout over a copy of its buffer's bytes
    up to the end of its span, so that a kernel reads its offset, gaps and
    overlaps as they lie on the source. *)

val gapped : Nx_array.Layout.t -> bool
(** [gapped l] is [true] iff [l] has an element and its least position is
    past [0] or its span is longer than the product of its extents along
    axes of nonzero stride. Such a layout's buffer holds bytes before or
    among its elements that none of them reaches. An overlapping layout can
    have such bytes and be [false]: shape [[3; 3]] at strides [[2; 2]].
    {!Nx_array.to_device} moves a [gapped] layout's elements to other
    positions; {!on} keeps them where they lie. *)

val host : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t
(** [host x] is [x] where the host reads it: [x] itself on {!Rig.host}, a
    copy there otherwise. *)

(** {1:guards Guard pages} *)

(** The type for the side of an array's bytes an inaccessible page lies on. *)
type side =
  | Before  (** The first byte of its span lies just after the page. *)
  | After  (** The last byte of its span lies just before the page. *)

val guarded :
  side ->
  ('v, 's) Nx_array.Dtype.t ->
  Nx_array.Layout.t ->
  ('v, 's) Nx_array.t * (unit -> unit)
(** [guarded side dt l] is an array of [dt] over [l] on host memory of its own,
    zeroed, whose span's bytes ({!Nx_array.Layout.span}) lie against an
    inaccessible page on [side], so that a kernel reading a byte past them on
    that side faults; and the function that frees the memory, after which the
    array is not used. [l]'s positions start at the memory's first byte. *)

val guard : side -> ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t * (unit -> unit)
(** [guard side x] is {!guarded} [side] over [x]'s layout, holding the bytes of
    [x]'s span: [x]'s elements, bit for bit, against the page, strided,
    reversed or broadcast as [x] is. With the function that frees it. [x] is
    on host memory. *)

(** {1:single One-node maps}

    A kind of the elementwise kernel, {!Nx_kernel.S.map} of the program
    {!Nx_kernel.Prog.of_node} makes of its one node, each operand read
    plain, into one destination. Each raises [Invalid_argument] where the
    kind does not take the operands' dtypes and the result's, as
    {!Nx_kernel.Prog.v} does. *)

val map0 :
  (module Nx_kernel.S) -> op0 -> dst:('v, 's) Nx_array.t -> Nx_array.answer
(** [map0 k op ~dst] stores [op] at each index of [dst]. *)

val map1 :
  (module Nx_kernel.S) ->
  Nx_kernel.Prog.op1 ->
  dst:('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer
(** [map1 k op ~dst x] stores [op] of each element of [x] into [dst]. *)

val map2 :
  (module Nx_kernel.S) ->
  Nx_kernel.Prog.op2 ->
  dst:('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer
(** [map2 k op ~dst x y] stores [op] of each element of [x] and [y] into
    [dst]. *)

val map3 :
  (module Nx_kernel.S) ->
  Nx_kernel.Prog.op3 ->
  dst:('v, 's) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer
(** [map3 k op ~dst c x y] stores [op] of each element of [c], [x] and [y]
    into [dst]. *)

(** {1:held Device memory} *)

val settled : Rig.t -> int
(** [settled d] is the bytes [d] holds once collected arrays are freed and
    its free memory returned, after rounds of both: a chain of finalisers
    frees its memory one round late. *)

val loop_bytes : Rig.t -> int -> (unit -> unit) -> int array
(** [loop_bytes d n f] runs [f], which waits for its work, once uncounted,
    then [n] times with no collection between, and is the bytes [d] holds
    past {!settled}'s count after each. *)

(** {1:memory Address space} *)

val cap_memory : int -> bool
(** [cap_memory n] limits the calling process's address space to [n] bytes
    above its size, so that an allocation past them fails. It is [true] iff
    the limit is set, which only Linux does. The limit stays for the
    process's life: only a forked child calls it. *)

(** {1:order Orders of a contraction's sum} *)

val ordered :
  chain:bool ->
  f32:bool ->
  init:float option ->
  float array ->
  float array ->
  float
(** [ordered ~chain ~f32 ~init a b] is the sum of the products [a.(t) b.(t)] in
    nx_cpu.mli's chain order where [chain], else in its lane order, then [init]
    added: each fused multiply-add and each addition rounded once to float32
    where [f32], to float64 otherwise. [a] and [b] are of one length and, where
    [f32], hold float32 values. With no product it is [init], or [+0]. *)
