(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dtypes, layouts and arrays over device buffers.

    An array is three facts over a {!Rig.Buffer.t}, whose bytes it reads:
    - a {e dtype} ({!Dtype}), what one element is: its storage format and the
      OCaml type its values read as;
    - a {e layout} ({!Layout}), where elements lie: a map from an index to an
      element position in the buffer;
    - the buffer, which places the array on a device ({!device}).

    A {e movement} ({!Move}) changes a layout without moving an element, and
    {!move} applies one to an array. {!v} makes an array from its parts and
    checks that its layout reaches only bits of its buffer, which every movement
    keeps; {!create} and {!of_array} make fresh ones.

    Host kernels are C and read arrays only through the door of [nx_array.h],
    which claims every operand of a call or none, waits under the claims for
    unfinished device work on them, and answers an {!answer}. A kernel's caller
    hands a refusal to {!refused}, which raises. The door may run OCaml code
    while it waits, so a kernel's external is never [[@@noalloc]]:
    {[
    external add_kernel :
      ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t ->
      Nx_array.answer = "nx_cpu_add"

    let add z x y =
      match add_kernel z x y with
      | Done -> z
      | Declined -> add_by_parts z x y
      | refusal -> Nx_array.refused "Nx.add" refusal [ Any z; Any x; Any y ]
    ]}
    Kernels that submit device work from OCaml claim their arrays through
    {!door}, which answers the same refusals. [nx_dtype.h] holds the dtypes'
    codes, their facts and every conversion into them for C, CUDA, HIP and
    Metal sources. *)

(** {1:dtypes Dtypes} *)

module Dtype = Dtype
(** Element formats. *)

(** {1:layouts Layouts} *)

module Move = Move
(** Movements. *)

module Layout = Layout
(** Layouts. *)

(** {1:arrays Arrays} *)

type ('v, 's) t
(** The type for arrays of elements of storage format ['s] read as ['v]. An
    array's layout reaches only bits of its buffer, at non-negative positions,
    and its first element lies on a multiple of its storage's alignment: its
    width for byte-wide dtypes, one component's for complex ones. *)

(** The type for arrays whose dtype is chosen at run time. {!expect} recovers
    the static type. *)
type any = Any : ('v, 's) t -> any

val v : ('v, 's) Dtype.t -> Layout.t -> Rig.Buffer.t -> ('v, 's) t
(** [v dt l b] is the array of [dt] elements laid out by [l] over [b]'s bytes.

    Raises [Invalid_argument] if [b] is dead, [l] reaches a bit past [b]'s
    bytes, or [l] has an element and its first element's byte offset into [b]'s
    memory, or for host memory its address, is not a multiple of [dt]'s
    alignment. *)

val create :
  ?memory:Rig.Buffer.memory ->
  Rig.t ->
  ('v, 's) Dtype.t ->
  int array ->
  ('v, 's) t
(** [create d dt s] is a fresh C-contiguous array of shape [s] on [d]'s memory
    [memory] (defaults to [Device]), at offset 0, with unspecified elements. Its
    buffer holds [Dtype.bytes dt n] bytes for [n] elements; the bits of a
    sub-byte array's last byte past its last element are zero.

    Raises [Invalid_argument] as {!Layout.contiguous} does, and what
    {!Rig.Buffer.create} raises. *)

val dtype : ('v, 's) t -> ('v, 's) Dtype.t
(** [dtype a] is [a]'s dtype. *)

val layout : ('v, 's) t -> Layout.t
(** [layout a] is [a]'s layout. *)

val buffer : ('v, 's) t -> Rig.Buffer.t
(** [buffer a] is [a]'s buffer. *)

val device : ('v, 's) t -> Rig.t
(** [device a] is the device of [a]'s buffer, the device [a] lives on. *)

val expect : ('w, 'r) Dtype.t -> any -> ('w, 'r) t
(** [expect dt (Any a)] is [a] at type [dt] if its dtype is [dt].

    Raises [Invalid_argument] naming both dtypes otherwise. *)

(** {1:views Views}

    {!move} and {!bitcast} make arrays over their argument's buffer. *)

val move : Move.t -> ('v, 's) t -> ('v, 's) t option
(** [move m a] is the array over [a]'s buffer laid out by
    [Layout.move m (layout a)], or [None] where that is [None].

    Raises [Invalid_argument] as {!Move.shape} does. *)

val bitcast : ('w, 'r) Dtype.t -> ('v, 's) t -> ('w, 'r) t option
(** [bitcast dt a] is [a]'s bits read as elements of [dt], over [a]'s buffer.
    With [r] the ratio of the two widths:
    - equal widths keep the layout;
    - to a narrower dtype, a trailing axis of extent [r] and stride 1 is
      appended, and the other strides and the offset are multiplied by [r];
    - to a wider dtype, [a] needs a trailing axis of extent [r] and stride 1,
      and an offset and other strides that are multiples of [r]; the axis is
      removed and they are divided by [r]. An array with no element needs only
      the trailing axis of extent [r].

    It is [None] where widening's conditions fail or the result's first element
    is not on a multiple of [dt]'s alignment.

    Raises [Invalid_argument] if [a]'s buffer is dead, and on a narrowing of an
    array of rank {!Layout.max_rank}. *)

(** {1:elements Elements}

    A function that reads or writes bytes on the host claims the buffer's
    memory, waits for the device work the access must follow
    ({!Rig.Buffer.wait}), and holds the claim while it runs. It raises
    [Invalid_argument] if the buffer is dead, its memory is held exclusive, or
    the host does not address its memory, and {!Rig.Lost} as {!Rig.Buffer.wait}
    does. *)

val get : ('v, 's) t -> int array -> 'v
(** [get a i] is [a]'s element at index [i].

    Raises [Invalid_argument] unless [i] has [Layout.rank (layout a)] entries,
    each in [\[0, Layout.dim (layout a) j)] for entry [j]. *)

val set : ('v, 's) t -> int array -> 'v -> unit
(** [set a i x] stores [x] at [a]'s index [i]. Stores to the other elements of a
    sub-byte element's byte, from any domain, are kept.

    Raises [Invalid_argument] as {!get} does, if [a] reaches an element twice
    ({!Layout.is_distinct}), if [a]'s memory is [Read], and if [x] is an [int]
    below {!Dtype.min_value} or above {!Dtype.max_value} of [a]'s dtype. *)

val to_array : ('v, 's) t -> 'v array
(** [to_array a] is [a]'s elements in C order of indices. [float] elements fill
    a flat [float array]; [int32], [int64] and [Complex.t] elements are boxed.
*)

val of_array : ('v, 's) Dtype.t -> int array -> 'v array -> ('v, 's) t
(** [of_array dt s xs] is a fresh C-contiguous array of shape [s] on {!Rig.host}
    holding [xs] in C order of indices, each stored as {!Dtype.of_float} says.

    Raises [Invalid_argument] as {!Layout.contiguous} does, if [xs] does not
    have one value per index of [s], or if a value is an [int] outside its
    dtype's range. *)

val copy : ('v, 's) t -> ('v, 's) t
(** [copy a] is a fresh C-contiguous array on [a]'s device holding [a]'s
    elements bit for bit, NaN payloads included. *)

(** {1:placement Placement} *)

val to_device : Rig.t -> ('v, 's) t -> ('v, 's) t
(** [to_device d a] is [a] over a fresh buffer on [d], made by
    {!Rig.Buffer.copy} of the bytes [a]'s layout reaches between its first and
    last position; its layout is [a]'s, shifted to the copy. It copies on [a]'s
    own device too, runs no kernel and keeps strided and broadcast layouts. For
    a sub-byte array, the copy's first and last bytes keep the bits of the
    source's neighbouring elements, as found before or after a store another
    domain makes to them while it copies.

    {!borrow} shares the memory instead, where [d] maps it.

    Raises [Invalid_argument] if [a]'s buffer is dead or its memory is held
    exclusive, and what {!Rig.Buffer.create} and {!Rig.Buffer.copy} raise. *)

val borrow : Rig.t -> ('v, 's) t -> ('v, 's) t option
(** [borrow d a] is [Some a'], [a]'s elements on [d] without a copy: [a]'s
    dtype and layout over a borrow of its buffer ({!Rig.Buffer.borrow}), which
    shares [a]'s memory, so writes through either show in the other. It is
    [Some a] for [a] on [d], and [None] where [d] cannot map [a]'s memory.

    Raises [Invalid_argument] if [a]'s buffer is dead, and what
    {!Rig.Buffer.borrow} raises: {!Rig.Lost} for a lost device. *)

(** {1:bigarrays Bigarrays}

    {!bigarray} and {!of_bigarray} share bytes with the value they take, without
    a copy. *)

val bigarray :
  ('v, 's) Bigarray.kind ->
  ('v, 's) t ->
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t option
(** [bigarray k a] is [Some] bigarray over [a]'s own bytes iff [a] is on
    {!Rig.host}, C-contiguous ({!Layout.is_contiguous}) and of rank at most 16.
    Writes through it write [a], and are allowed only if [a]'s memory admits
    them ({!Rig.Buffer.val-access}). From then on [a]'s memory is never held
    exclusive again ({!Rig.Buffer.bigarray}). It waits for nothing: access
    through the bigarray follows device work only after {!Rig.Buffer.wait}. A
    format Bigarray lacks is bitcast first to one of its width ({!bitcast}).

    Reads and writes through the bigarray are Bigarray's: an [int] out of range
    keeps its low bits, and a [float] stored into a [Float16] bigarray is
    rounded through binary32 first, so it can differ from {!Dtype.of_float} by
    one unit in the last place.

    Raises [Invalid_argument] if [a]'s buffer is dead, or its memory is held
    exclusive by claims that have not consumed it. *)

val of_bigarray :
  ('v, 's) Dtype.t ->
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t ->
  ('v, 's) t
(** [of_bigarray dt b] is the C-contiguous array of [dt] elements over [b]'s
    bytes, of [b]'s shape. The types tie [dt] to [b]'s kind: a kind no dtype
    stores, as [Char], [Int] or [Nativeint], is a type error.

    Raises [Invalid_argument] as {!v} does if [b]'s data is not aligned for its
    elements. *)

(** {1:kernels Kernels} *)

(** The type for what a kernel answers, and the door. A kernel answers [Done]
    once its work is done on the host or queued on the device's timeline,
    [Declined] for a case it does not compute, and otherwise, before any write,
    a refusal: what is wrong with an operand. C answers [Val_int] of
    [nx_array.h]'s code, whose enum lists the constructors in this order. *)
type answer =
  | Done
  | Declined
  | Wrong_dtype  (** An operand's dtype is not the one the kernel loads. *)
  | Dead_buffer  (** An operand's buffer is dead. *)
  | Off_host  (** The host does not address an operand's memory. *)
  | Held_exclusive  (** An operand's memory is held exclusive. *)
  | Read_only  (** A written operand's memory is [Read]. *)
  | Repeated_elements  (** A written operand reaches an element twice. *)
  | Overlapping  (** A written operand shares a byte with another operand. *)
  | Bad_layout
      (** An operand's layout is not a layout: a guard for C callers, which an
          array made by this library never meets. *)
  | Shape_mismatch  (** The operands of one loop have different shapes. *)
  | Bad_arity
      (** No operand, or more than a loop takes: a guard for C callers, which a
          kernel of this library's contract never meets. *)

val door :
  written:any array -> read:any array -> ('a -> unit) -> 'a -> answer
(** [door ~written ~read f x] claims the memory of [written] for writing and of
    [read] for reading, every claim or none, runs [f x], and releases the claims
    when [f] returns or raises; an exception of [f] propagates once they are
    released. It answers [Done] once [f] returns, or, having claimed nothing
    and run nothing, [Repeated_elements] or [Overlapping] for a written array,
    [Dead_buffer] or [Held_exclusive] for an array, or [Read_only] for a written
    array on [Read] memory.

    It waits for no device work. [f] touches the arrays' elements only through
    work it submits with {!Rig.submit}, naming each array of [written] in its
    writes and each of [read] in its reads or writes; rig orders that work after
    the work before it on every device. [f] leaves the OCaml arrays [written]
    and [read] holding the arrays they held: the door releases what they hold
    when [f] ends. With a [f] that is not a closure and arrays the caller
    reuses, it allocates nothing.

    Raises {!Rig.Lost}, having claimed nothing and run nothing, if, when it
    claims, an array's memory is a lost device's or must follow work a lost
    device did not finish. A device lost while [f] runs reaches the caller as
    [f]'s exception. *)

val refused : string -> answer -> any list -> 'a
(** [refused name r operands] raises [Invalid_argument] for the refusal [r]
    that a kernel answered for [operands], naming [name], the reason [r] gives,
    and each operand's dtype and shape. [name] is the function the user called,
    as ["Nx.add"]. For [Dead_buffer] the reason names each operand whose
    buffer is dead, numbered from 1 in [operands]' order, with the reason its
    consumer gave ({!Rig.Buffer.dead}), as
    [Nx.add: operand 2 was consumed, donated (Nx.donate) (float32 [3], …)].
    [Done] and [Declined] are no refusal, and raise [Invalid_argument] saying
    so. *)
