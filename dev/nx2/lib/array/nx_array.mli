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
    unfinished device work on them, and answers a code. A kernel's OCaml wrapper
    hands any code but [NX_OK] to {!refused}, which raises. The door may run
    OCaml code while it waits, so a kernel's external is never [[@@noalloc]]:
    {[
    external add_kernel :
      ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> int
      = "nx_cpu_add"

    let add z x y =
      let e = add_kernel z x y in
      if e <> 0 then Nx_array.refused "Nx.add" e [ Any z; Any x; Any y ]
    ]}
    GPU kernel libraries bind arrays in OCaml, through their typed signatures or
    {!expect}. [nx_dtype.h] holds the dtypes' codes, their facts and every
    conversion into them for C, CUDA, HIP and Metal sources. *)

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

    Raises [Invalid_argument] if [a]'s buffer is dead or its memory is held
    exclusive, and what {!Rig.Buffer.create} and {!Rig.Buffer.copy} raise. *)

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

val refused : string -> int -> any list -> 'a
(** [refused name code operands] raises [Invalid_argument] for [code], a code
    of [nx_array.h] other than [NX_OK] that a kernel answered for [operands],
    naming [name], the reason the code gives, and each operand's dtype and
    shape. [name] is the function the user called, as ["Nx.add"]. *)
