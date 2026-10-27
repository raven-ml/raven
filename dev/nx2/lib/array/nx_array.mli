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
      | Declined -> invalid_arg "Nx.add: nx_cpu_add does not compute this case"
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

type ('v, 's) t = private {
  dtype : ('v, 's) Dtype.t;
  layout : Layout.t;
  buffer : Rig.Buffer.t;
}
(** The type for arrays of elements of storage format ['s] read as ['v]. An
    array's layout reaches only bits of its buffer, at non-negative positions,
    and its first element lies on a multiple of its storage's alignment: its
    width for dtypes of a byte or more, one byte for narrower ones, one
    component's for complex ones. The record is private. Only this module's
    functions make an array, so every array keeps these facts. *)

(** The type for arrays whose dtype is chosen at run time. {!expect} recovers
    the static type. [Any a] is [a] at run time. Packing allocates nothing. *)
type any = Any : ('v, 's) t -> any [@@unboxed]

val v : ('v, 's) Dtype.t -> Layout.t -> Rig.Buffer.t -> ('v, 's) t
(** [v dt l b] is the array of [dt] elements laid out by [l] over [b]'s bytes.

    Raises [Invalid_argument] if [b] is dead, [l] reaches a bit past [b]'s
    bytes, or [l] has an element and its first element's byte offset into [b]'s
    memory, or for host memory its address, is not a multiple of [dt]'s
    alignment. *)

val create :
  ?memory:Rig.Buffer.kind ->
  Rig.t ->
  ('v, 's) Dtype.t ->
  int iarray ->
  ('v, 's) t
(** [create d dt s] is a fresh C-contiguous array of shape [s] in [d]'s memory
    of kind [memory] (defaults to [Device]), at offset 0, with unspecified
    elements. Its buffer holds [Dtype.bytes dt n] bytes for [n] elements; the
    bits of a sub-byte array's last byte past its last element are zero.

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

val get : ('v, 's) t -> int iarray -> 'v
(** [get a i] is [a]'s element at index [i].

    Raises [Invalid_argument] unless [i] has [Layout.rank (layout a)] entries,
    each in [\[0, Layout.dim (layout a) j)] for entry [j]. *)

val set : ('v, 's) t -> int iarray -> 'v -> unit
(** [set a i x] stores [x] at [a]'s index [i]. Stores to the other elements of a
    sub-byte element's byte, from any domain, are kept.

    Raises [Invalid_argument] as {!get} does, if [a] reaches an element twice
    ({!Layout.is_distinct}), if [a]'s memory is [Read], and if [x] is an [int]
    below {!Dtype.min_value} or above {!Dtype.max_value} of [a]'s dtype. *)

val to_array : ('v, 's) t -> 'v array
(** [to_array a] is [a]'s elements in C order of indices. [float] elements fill
    a flat [float array]; [int32], [int64] and [Complex.t] elements are boxed.
*)

val of_array : ('v, 's) Dtype.t -> int iarray -> 'v array -> ('v, 's) t
(** [of_array dt s xs] is a fresh C-contiguous array of shape [s] on {!Rig.host}
    holding [xs] in C order of indices, each stored as {!Dtype.of_float} says.

    Raises [Invalid_argument] as {!Layout.contiguous} does, if [xs] does not
    have one value per index of [s], or if a value is an [int] outside its
    dtype's range. *)

(** {1:placement Placement}

    {!to_device}, {!transfer} and {!copy} store an array's elements bit for
    bit, NaN payloads included, on any devices, and run no kernel. {!borrow}
    shares an array's memory with another device instead, where that device
    maps it. {!to_device} and {!transfer} also raise what
    {!Rig.Buffer.create}, {!Rig.Buffer.copy} and {!Rig.submit} raise. *)

val to_device : Rig.t -> ('v, 's) t -> ('v, 's) t
(** [to_device d a] is [a]'s elements on [d], also when [a] is on [d], in a
    fresh buffer of at most [Dtype.bytes] of one element per index of [a], an
    axis of stride [0] counted once. Where the bytes from [a]'s first to its
    last position are no more than those, the copy keeps [a]'s layout shifted
    to it, and a sub-byte copy's first and last bytes keep the source's bits
    beside the elements, each as found before or after a store another domain
    makes to it meanwhile. Otherwise the copy's axes go by decreasing magnitude
    of [a]'s stride, ties in axis order, the last at stride [1] and each other
    at the product of the extents after it. Each takes the sign of [a]'s
    stride, an axis of stride [0] keeps [0], and the least position is at
    [0]. Shape [[3; 2]] at strides [[2; 2]] gets strides [[2; 1]].

    Raises [Invalid_argument] if [a]'s buffer is dead or its memory is held
    exclusive. *)

val transfer : src:('v, 's) t -> dst:('v, 's) t -> unit
(** [transfer ~src ~dst] stores [src]'s elements into [dst], of the same shape.
    It writes [dst]'s elements and no other bit of its buffer, a sub-byte
    neighbour included.

    Raises [Invalid_argument] if the shapes differ, a buffer is dead or its
    memory is held exclusive, [dst] reaches an element twice
    ({!Layout.is_distinct}), or [dst]'s elements are narrower than a byte and
    the host does not address its memory, and, where the host addresses both
    arrays' memory, if [dst]'s memory is [Read] or [dst] shares a byte with
    [src] without being identical to it ({!section-kernels}). *)

val copy : ('v, 's) t -> ('v, 's) t
(** [copy a] is a fresh C-contiguous array on [a]'s device holding [a]'s
    elements, the array {!create} makes there filled by {!transfer}.

    Raises what {!create} and {!transfer} raise. *)

val borrow : Rig.t -> ('v, 's) t -> ('v, 's) t option
(** [borrow d a] is [Some a'], [a]'s elements on [d] without a copy: [a]'s
    dtype and layout over a borrow of its buffer ({!Rig.Buffer.borrow}), which
    shares [a]'s memory, so writes through either show in the other. It is
    [Some a] for [a] on [d], and [None] where [d] cannot map [a]'s memory.

    Raises [Invalid_argument] if [a]'s buffer is dead, and what
    {!Rig.Buffer.borrow} raises: {!Rig.Lost} for a lost device. *)

(** {1:bigarrays Bigarrays}

    {!bigarray}, {!read} and {!of_bigarray} share bytes with the value they
    take, without a copy. {!bigarray} and {!read} raise [Invalid_argument]
    unless the array [a] they take is C-contiguous ({!Layout.is_contiguous})
    and of rank at most 16, the most axes a Bigarray has, and if [a]'s buffer
    is dead or its memory is held exclusive. They raise {!Rig.Lost} as
    {!Rig.Buffer.wait} does. *)

val bigarray :
  ('v, 's) Bigarray.kind ->
  ('v, 's) t ->
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t
(** [bigarray k a] is a bigarray of [a]'s shape over [a]'s own bytes. Writes
    through it write [a], and are allowed only if [a]'s memory admits them
    ({!Rig.Buffer.val-access}). It waits once for the work on [a]'s memory, and
    from then on that memory is never held exclusive again
    ({!Rig.Buffer.export}). Access through the bigarray follows later device
    work only after {!Rig.Buffer.wait}. A format Bigarray lacks is bitcast
    first to one of its width ({!bitcast}).

    Reads and writes through the bigarray are Bigarray's: an [int] out of range
    keeps its low bits, and a [float] stored into a [Float16] bigarray is
    rounded through binary32 first, so it can differ from {!Dtype.of_float} by
    one unit in the last place.

    Raises [Invalid_argument] naming [Nx_array.bigarray], and unless [a] is
    on {!Rig.host}. *)

val read :
  ('v, 's) Bigarray.kind ->
  ('v, 's) t ->
  (('v, 's, Bigarray.c_layout) Bigarray.Genarray.t -> 'a) ->
  'a
(** [read k a f] is [f b], where [b] is a bigarray of [a]'s shape over [a]'s own
    bytes; an [a] with no element gives an empty [b], whatever its memory. It
    claims [a]'s memory for reading, waits for the device work a read must
    follow ({!Rig.Buffer.wait}), and ends the claim when [f] returns or raises.
    The memory stays inside the claims, so a donation can hold it exclusive once
    [f] returns, which {!bigarray} prevents for good.

    [f] must not write through [b], since work that reads the memory may still
    run, and must not keep [b] after it returns, since a donation or
    {!Rig.Buffer.free} can then give the memory to another array. The claim
    orders [b] only against donations. The program must order [b] against
    work another domain submits while [f] runs.

    Raises [Invalid_argument] naming [Nx_array.read], and if [a] has an
    element and the host does not address its memory. *)

val of_bigarray :
  ('v, 's) Dtype.t ->
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t ->
  ('v, 's) t
(** [of_bigarray dt b] is the C-contiguous array of [dt] elements over [b]'s
    bytes, of [b]'s shape. The types tie [dt] to [b]'s kind: a kind no dtype
    stores, as [Char], [Int] or [Nativeint], is a type error.

    Raises [Invalid_argument] as {!v} does if [b]'s data is not aligned for its
    elements. *)

(** {1:kernels Kernels}

    Two arrays are {e identical} when they have one element width and place
    every index at the same bits, which two memories may hold, as an io memory
    and a bigarray over its pages do ({!Rig.Buffer.overlaps}). A written array
    may share bytes with read arrays identical to it, and with no other.

    The door and the kernels that read through it keep other domains
    waiting to collect for a bounded time, which does not grow with the
    number of arrays they claim or release. Past {!max_operands} arrays, other
    threads of the calling domain may run OCaml code while the door claims or
    releases them. *)

val max_operands : int
(** [max_operands] is [18], [nx_array.h]'s [NX_MAX_OPERANDS]: the length of
    the runs in which the door claims and releases arrays, and the most
    layouts {!Layout.coalesce} and [nx_coalesce] take. *)

(** The type for what a kernel answers, and the door. A kernel answers [Done]
    once its work is done on the host or queued on the device's timeline,
    [Declined] for a case it does not compute, and otherwise, before any write,
    a refusal: what is wrong with an operand. [Dead_buffer], [Held_exclusive]
    and [Read_only], which claiming an operand answers, come before [Off_host],
    [Repeated_elements] and [Overlapping], so a dead operand is [Dead_buffer]
    whatever else is wrong with it. C answers [Val_int] of [nx_array.h]'s code,
    whose enum lists the constructors in this order. *)
type answer =
  | Done
  | Declined
  | Wrong_dtype  (** An operand's dtype is not the one the kernel loads. *)
  | Dead_buffer  (** An operand's buffer is dead. *)
  | Off_host  (** The host does not address an operand's memory. *)
  | Held_exclusive  (** An operand's memory is held exclusive. *)
  | Read_only  (** A written operand's memory is [Read]. *)
  | Repeated_elements  (** A written operand reaches an element twice. *)
  | Overlapping
      (** A written operand shares a byte with another written operand, or
          with a read operand not identical to it ({!section-kernels}). *)
  | Shape_mismatch
      (** The operands do not fit the operation: their number, or their
          shapes. Operands of one loop of different shapes, shapes its
          descriptor refuses, or other destinations than an operation names,
          as a reduction given other than its one or two. *)

val door :
  written:any array -> read:any array -> ('a -> unit) -> 'a -> answer
(** [door ~written ~read f x] claims the memory of [written] for writing and of
    [read] for reading, every claim or none, runs [f x], and releases the claims
    when [f] returns or raises, before an exception of [f] propagates. It waits
    for no device work. It answers [Done] once [f] returns, and otherwise,
    without running [f] or holding a claim, [Dead_buffer] or [Held_exclusive]
    for an array, or [Read_only], [Repeated_elements] or [Overlapping] for a
    written one.

    [f] must touch the arrays' elements only through work it submits with
    {!Rig.submit}, naming each array of [written] in its writes and each of
    [read] in its reads or writes, and must leave [written] and [read] holding
    the arrays they held. The door allocates nothing, and nor does a caller
    that passes an [f] that is not a closure and stores [Any a] into the same
    [written] and [read] each call.

    Raises {!Rig.Lost}, without running [f] or holding a claim, if an array's
    memory is a lost device's or must follow work a lost device did not
    finish, and as [f]'s exception for a device lost while [f] runs. *)

val refused : string -> answer -> any list -> 'a
(** [refused name r operands] raises [Invalid_argument] for the refusal [r]
    that a kernel answered for [operands], naming [name], the reason [r] gives,
    and each operand's dtype and shape. [name] is the function the user called,
    as ["Nx.add"]. For [Dead_buffer], and for [Held_exclusive] once the
    holder consumed the memory, the reason names each operand whose buffer is
    dead, in [operands]' order, as [<dtype> <shape> was <why>], [why] the
    reason its consumer gave ({!Rig.Buffer.dead}):
    [Nx.add: float32 [3] was donated to Nx.mul (float32 [3], …)].
    For [Done] and [Declined] it raises [Invalid_argument] saying they refuse
    nothing. *)
