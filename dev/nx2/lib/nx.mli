(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** N-dimensional arrays as values on device sets.

    A value of type [('v, 's, 'd) t] is elements of one dtype, read as ['v] and
    stored as ['s], with a shape and a placement over the device set ['d].
    Nothing changes a value.

    A device set is a module: [module Gpu = (val Nx.devices [ d0; d1 ])] mints a
    brand [Gpu.d], and values of two sets never meet in typed code. A value
    lies at a {e placement}: on the host, on one device of its set, whole on
    every device of it, or cut into {e windows} across them, where each
    device holds one window of the value. A value is made where it is asked:
    on the host, at a placement with an [_at] form, or beside another value
    with a [_like] form ({!section-creation}).

    An operation computes where its operands lie and moves none of them.
    {!place} moves a value, and an operation whose operands do not lie
    together raises, naming the {!place} to write ({!section-routing}). The
    partial results of a fold over a cut axis, and the parts of the result
    they close into, are the only bytes an operation moves between devices
    ({!section-reductions}). *)

(** {1:values Values}

    A value is a shape, a dtype and the devices its elements lie on. A shape
    has at most {!Nx_array.Layout.max_rank} axes, no negative extent, and
    extents other than [0] that multiply to at most
    {!Nx_array.Layout.max_numel}. A function given or making another shape
    raises [Invalid_argument] naming itself. *)

type (!'v, !'s, +!'d) t
(** The type for arrays of ['s]-format elements read as ['v], on the device set
    ['d]. *)

type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t
(** The type for dtypes whose elements OCaml reads as ['v] and which store them
    as ['s]. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is [x]'s extents, a fresh array. *)

val dtype : ('v, 's, 'd) t -> ('v, 's) dtype
(** [dtype x] is [x]'s element format. *)

(** {1:dtypes Dtypes}

    A dtype says how a value stores its elements and how OCaml reads them:
    [float32] stores 32-bit floats and reads them as [float], [int8] stores
    8-bit integers and reads them as [int]. The operands of an operation share a
    dtype by their types; {!cast} changes it. There is one value per constructor
    of {!Dtype.t}, named in lowercase. *)

module Dtype = Nx_array.Dtype
(** Dtypes: their constructors, element types and properties. *)

(** {!Dtype}'s element types, under the same names: [float32_elt] is
    [Dtype.float32_elt]. *)

type float64_elt = Dtype.float64_elt
type float32_elt = Dtype.float32_elt
type float16_elt = Dtype.float16_elt
type bfloat16_elt = Dtype.bfloat16_elt
type float8_e4m3fn_elt = Dtype.float8_e4m3fn_elt
type float8_e5m2_elt = Dtype.float8_e5m2_elt
type float4_e2m1fn_elt = Dtype.float4_e2m1fn_elt
type int64_elt = Dtype.int64_elt
type uint64_elt = Dtype.uint64_elt
type int32_elt = Dtype.int32_elt
type uint32_elt = Dtype.uint32_elt
type int16_signed_elt = Dtype.int16_signed_elt
type int16_unsigned_elt = Dtype.int16_unsigned_elt
type int8_signed_elt = Dtype.int8_signed_elt
type int8_unsigned_elt = Dtype.int8_unsigned_elt
type int4_elt = Dtype.int4_elt
type uint4_elt = Dtype.uint4_elt
type complex64_elt = Dtype.complex64_elt
type complex32_elt = Dtype.complex32_elt
type bool_elt = Dtype.bool_elt
type bit_elt = Dtype.bit_elt

val float64 : (float, Dtype.float64_elt) dtype
val float32 : (float, Dtype.float32_elt) dtype
val float16 : (float, Dtype.float16_elt) dtype
val bfloat16 : (float, Dtype.bfloat16_elt) dtype
val float8_e4m3fn : (float, Dtype.float8_e4m3fn_elt) dtype
val float8_e5m2 : (float, Dtype.float8_e5m2_elt) dtype
val float4_e2m1fn : (float, Dtype.float4_e2m1fn_elt) dtype
val int64 : (int64, Dtype.int64_elt) dtype
val uint64 : (int64, Dtype.uint64_elt) dtype
val int32 : (int32, Dtype.int32_elt) dtype
val uint32 : (int32, Dtype.uint32_elt) dtype
val int16 : (int, Dtype.int16_signed_elt) dtype
val uint16 : (int, Dtype.int16_unsigned_elt) dtype
val int8 : (int, Dtype.int8_signed_elt) dtype
val uint8 : (int, Dtype.int8_unsigned_elt) dtype
val int4 : (int, Dtype.int4_elt) dtype
val uint4 : (int, Dtype.uint4_elt) dtype
val complex128 : (Complex.t, Dtype.complex64_elt) dtype
val complex64 : (Complex.t, Dtype.complex32_elt) dtype
val bool : (bool, Dtype.bool_elt) dtype
val bit : (bool, Dtype.bit_elt) dtype

(** Each dtype value [dt] names the type of its values, [dt_t]: a
    ['d float32_t] is a float32 value on the set ['d]. *)

type 'd float64_t = (float, Dtype.float64_elt, 'd) t
type 'd float32_t = (float, Dtype.float32_elt, 'd) t
type 'd float16_t = (float, Dtype.float16_elt, 'd) t
type 'd bfloat16_t = (float, Dtype.bfloat16_elt, 'd) t
type 'd float8_e4m3fn_t = (float, Dtype.float8_e4m3fn_elt, 'd) t
type 'd float8_e5m2_t = (float, Dtype.float8_e5m2_elt, 'd) t
type 'd float4_e2m1fn_t = (float, Dtype.float4_e2m1fn_elt, 'd) t
type 'd int64_t = (int64, Dtype.int64_elt, 'd) t
type 'd uint64_t = (int64, Dtype.uint64_elt, 'd) t
type 'd int32_t = (int32, Dtype.int32_elt, 'd) t
type 'd uint32_t = (int32, Dtype.uint32_elt, 'd) t
type 'd int16_t = (int, Dtype.int16_signed_elt, 'd) t
type 'd uint16_t = (int, Dtype.int16_unsigned_elt, 'd) t
type 'd int8_t = (int, Dtype.int8_signed_elt, 'd) t
type 'd uint8_t = (int, Dtype.int8_unsigned_elt, 'd) t
type 'd int4_t = (int, Dtype.int4_elt, 'd) t
type 'd uint4_t = (int, Dtype.uint4_elt, 'd) t
type 'd complex128_t = (Complex.t, Dtype.complex64_elt, 'd) t
type 'd complex64_t = (Complex.t, Dtype.complex32_elt, 'd) t
type 'd bool_t = (bool, Dtype.bool_elt, 'd) t
type 'd bit_t = (bool, Dtype.bit_elt, 'd) t

(** The type for values whose dtype is hidden, such as a file's named values.
    Match on [P] for the value; {!unpack} also fixes its dtype. *)
type 'd packed = P : ('v, 's, 'd) t -> 'd packed

val unpack : ('v, 's) dtype -> 'd packed -> ('v, 's, 'd) t
(** [unpack dt p] is the value in [p] if its dtype is [dt].

    Raises [Invalid_argument] naming both dtypes otherwise. *)

(** {1:devices Device sets and placement}

    A value lies on a device set, a module minted by {!devices} whose brand ['d]
    keeps its values apart from other sets'. {!Placement} says where a value
    lies within its set, and {!place} moves a value between sets, and
    between placements of one. The host is a set like any other, {!Host}. *)

type +'d devices
(** The type for device sets of brand ['d]: distinct devices, and the kernels
    that compute on them. *)

val rigs : 'd devices -> Rig.t list
(** [rigs s] is [s]'s devices in order. *)

(** Named axes over a set's devices. *)
module Mesh : sig
  type +'d t
  (** The type for named axes over a set of brand ['d]. *)

  val v : 'd devices -> (string * int) list -> 'd t
  (** [v s axes] lays [s]'s devices in row-major order over the named [axes].

      Raises [Invalid_argument] unless the extents are positive and multiply to
      the number of [s]'s devices and the names are distinct. *)
end

(** Where a value's elements lie over its set. *)
module Placement : sig
  type +'d t
  (** The type for placements over a set of brand ['d]. *)

  val on : 'd devices -> 'd t
  (** [on s] holds the whole value on every device of [s]. *)

  val device : 'd devices -> Rig.t -> 'd t
  (** [device s d] holds the whole value on [d] alone, a device of [s];
      [on s] where [s] has [d] alone.

      Raises [Invalid_argument] naming [Nx.Placement.device] unless [d]
      is a device of [s] ({!Rig.equal}). *)

  val split : axis:int -> 'd devices -> 'd t
  (** [split ~axis s] cuts [axis] into one window per device of [s], in
      order, as evenly as its extent allows: of [n] elements over [k] devices,
      the first [n mod k] windows hold [n / k + 1] and the others [n / k]. A
      window is empty only where [n < k].

      Raises [Invalid_argument] if [axis] is negative or not below
      {!Nx_array.Layout.max_rank}. *)

  val mesh : 'd Mesh.t -> (int * string list) list -> 'd t
  (** [mesh m cuts] cuts each axis [a] of [(a, names)] over the mesh axes
      [names], major first; a mesh axis no cut names holds copies. An axis cut
      over several mesh axes is cut as {!split} cuts it over as many devices
      as their extents multiply to, numbered major first: over extents [e1]
      and [e2], the device at [i1] along the first and [i2] along the second
      holds window [i1 × e2 + i2].
      [split ~axis s] is [mesh (Mesh.v s [ ("x", n) ]) [ (axis, [ "x" ]) ]].

      Raises [Invalid_argument] for a name [m] lacks, an axis or a name in two
      cuts, or an axis that is negative or not below
      {!Nx_array.Layout.max_rank}. *)

  val devices : 'd t -> 'd devices
  (** [devices p] is [p]'s set. *)

  val equal : 'd t -> 'e t -> bool
  (** [equal p q] is [true] iff [p] and [q] are of one set and every device
      holds the same window of a value at [p] and at [q], whatever their
      brands. *)

  val uncut : 'd t -> 'd t
  (** [uncut p] is [p] with every axis whole on each of its devices, each
      mesh axis that cut one holding copies. It is [p] itself if [p] cuts no
      axis. *)

  val pp : Format.formatter -> 'd t -> unit
  (** [pp] formats a placement on one device as the device's name, others as
      [on [CUDA:0; CUDA:1]], [split ~axis:0 [CUDA:0; CUDA:1]] or a mesh's
      extents and cuts. *)

  (**/**)

  (** For transformations that map over a leading axis, as a batching
      interpreter places a lane's operation for all lanes in one
      operation. *)

  val with_leading_axis : 'd t -> 'd t
  (** [with_leading_axis p] is the placement of a value of one more axis in
      front, cut as [p] cuts the others: each device holds the whole of the
      new axis by its window of the rest. It is [p] itself where [p] cuts no
      axis.

      Raises [Invalid_argument] if [p] cuts axis
      [Nx_array.Layout.max_rank - 1], which would become an axis past the
      most a layout has. *)

  val without_leading_axis : 'd t -> 'd t
  (** [without_leading_axis p] is the placement of a value without its first
      axis, cut as [p] cuts the others, each moved down by one: the mesh
      axes that cut the first axis hold copies. It is [p] itself where [p]
      cuts no axis. *)

  (**/**)
end

(** The type for device sets as modules: a brand and its placements. *)
module type Devices = sig
  type d
  (** The set's brand. *)

  val v : d devices
  (** [v] is the set. *)

  val on : d Placement.t
  (** [on] is [Placement.on v]. *)

  val split : axis:int -> d Placement.t
  (** [split ~axis] is [Placement.split ~axis v]. *)
end

type host
(** The brand of {!Host}. *)

module Host : Devices with type d = host
(** The process's host, {!Rig.host}, computed by nx.cpu. *)

val devices : ?kernels:(module Nx_kernel.S) -> Rig.t list -> (module Devices)
(** [devices ~kernels ds] mints a new set over [ds], in their order, with a
    brand no other set has, computed eagerly by [kernels]. Without [kernels], a
    set whose every device runs on the host ({!Rig.runs_on_host}) is computed by
    nx.cpu, and any other set by none.

    Raises [Invalid_argument] if [ds] is empty or repeats a device, or if
    [kernels] does not compute on one of [ds]. *)

(** {2:routing Routing}

    Each device computes its window of an operation's result from the
    windows of the operands it holds. The result lies at the union of the
    operands' cuts, joined in operand order. A cut of an axis that becomes
    a result axis cuts that axis over the same devices. A cut of an axis
    read whole cuts nothing. A cut of an axis that a reduction or a
    contraction sums makes the devices compute by partials where every
    operand holding the axis is cut alike, and cuts nothing in the result;
    one along a scan's axis stays in it. An operand whole on every device
    is read by a view of each device's window.

    An operation raises [Invalid_argument] where two operands cut one
    result axis over different devices or tiles, where a device needs a
    window of an operand it does not hold, and where operands lie alone on
    two devices or on two sets. The message names the function, the
    operand, numbered from [0] in the order {!Prim.t} lists them, where it
    lies, where it is read, and the {!place} that puts it there, writing
    the set as [s], a device by its name, a mesh's axes as ["g0"], ["g1"],
    … in the read placement's order, and the operands as [x], [y], [z],
    then [x3], [x4], …:

    {v
Nx.matmul: operand 1 lies at split ~axis:1 [CUDA:0; CUDA:1] and is read
at split ~axis:0 [CUDA:0; CUDA:1]; Nx.place (Placement.split ~axis:0 s) y
    v}

    A traced operation raises it as eager execution would
    ({!Prim.results}). *)

val place : 'e Placement.t -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place p x] is [x]'s elements at [p]. Each device of [p] holds its window
    over [x]'s own memory where [x] already lies there, or where the host
    addresses both [x]'s memory and that device's ({!Rig.reaches});
    otherwise it holds a copy of its window, allocating at most the window's
    bytes. Across sets it changes the brand; within one, the arrangement.
    Placing onto {!Host} from another set is a {{!section-waiting}wait} on
    [x]'s set.

    Raises [Invalid_argument] if [p] cuts an axis [x] does not have,
    {!Rig.Lost} for a lost device, and a failed check as a wait does. *)

val placement : ('v, 's, 'd) t -> 'd Placement.t
(** [placement x] is where [x]'s bytes lie. *)

(** {1:creation Creation}

    A value made from a dtype, a shape and numbers is a host value. The [_at]
    forms make it at a placement, each device computing its window; the
    [_like] forms make it beside a value, at that value's placement, or whole
    on each of its devices where the shape has no axis the placement cuts. A
    creation holds its elements from the call on.

    An [_at] form raises as its host form does, and raises
    [Invalid_argument] naming itself where its placement cuts an axis the
    shape lacks. *)

val zeros : ('v, 's) dtype -> int array -> ('v, 's, host) t
(** [zeros dt s] is the host value of shape [s] whose every element is zero.

    Raises [Invalid_argument] if an extent is negative. *)

val zeros_at : 'd Placement.t -> ('v, 's) dtype -> int array -> ('v, 's, 'd) t
(** [zeros_at p dt s] is {!zeros}[ dt s] at [p]. *)

val ones : ('v, 's) dtype -> int array -> ('v, 's, host) t
(** [ones dt s] is the host value of shape [s] whose every element is one
    ([true] for booleans).

    Raises [Invalid_argument] if an extent is negative. *)

val ones_at : 'd Placement.t -> ('v, 's) dtype -> int array -> ('v, 's, 'd) t
(** [ones_at p dt s] is {!ones}[ dt s] at [p]. *)

val full : ('v, 's) dtype -> int array -> 'v -> ('v, 's, host) t
(** [full dt s v] is the host value of shape [s] whose every element is [v], a
    float stored as {!Dtype.of_float} says.

    Raises [Invalid_argument] if an extent is negative or [v] is an [int] outside
    [dt]'s range. *)

val full_at :
  'd Placement.t -> ('v, 's) dtype -> int array -> 'v -> ('v, 's, 'd) t
(** [full_at p dt s v] is {!full}[ dt s v] at [p]. *)

val scalar : ('v, 's) dtype -> 'v -> ('v, 's, host) t
(** [scalar dt v] is the 0-d host value [v].

    Raises [Invalid_argument] if [v] is an [int] outside [dt]'s range. *)

val scalar_at : 'd Placement.t -> ('v, 's) dtype -> 'v -> ('v, 's, 'd) t
(** [scalar_at p dt v] is {!scalar}[ dt v] at {!Placement.uncut}[ p], whole on
    each of [p]'s devices however [p] cuts, since a 0-d value has no axis to
    cut. *)

val arange : ('v, 's) dtype -> int -> int -> int -> ('v, 's, host) t
(** [arange dt start stop step] is the 1-d host value [start], [start + step],
    … before [stop], empty where [stop] is not past [start] in [step]'s
    direction. Each value is exact, then stored in [dt] as {!cast} stores an
    [int64].

    Raises [Invalid_argument] if [step = 0], the range holds more than
    [max_int] values, or a value is outside [dt]'s range for an integer
    dtype, or other than [0] and [1] for a boolean one. *)

val arange_at :
  'd Placement.t -> ('v, 's) dtype -> int -> int -> int -> ('v, 's, 'd) t
(** [arange_at p dt start stop step] is {!arange}[ dt start stop step] at
    [p]. *)

val arange_f :
  (float, 's) dtype -> float -> float -> float -> (float, 's, host) t
(** [arange_f dt start stop step] is the 1-d host value of the [⌈(stop -
    start) / step⌉] values [start + i step], each computed at float64 with one
    rounding, then stored in [dt]; empty where that count is not positive.

    Raises [Invalid_argument] if [step = 0.] or the count is not finite. *)

val arange_f_at :
  'd Placement.t ->
  (float, 's) dtype ->
  float ->
  float ->
  float ->
  (float, 's, 'd) t
(** [arange_f_at p dt start stop step] is {!arange_f}[ dt start stop step] at
    [p]. *)

val linspace :
  ('v, 's) dtype -> ?endpoint:bool -> float -> float -> int -> ('v, 's, host) t
(** [linspace dt ~endpoint start stop n] is the host value of [n] values evenly
    spaced from [start] to [stop], [stop] included where [endpoint] (default
    [true]). They are computed at float64 and stored in [dt]: the first is
    [start] and, with [endpoint] and [n >= 2], the last is [stop], each as
    [dt] stores it, and for finite [start] and [stop] every value lies between
    them.

    Raises [Invalid_argument] if [n < 0]. *)

val linspace_at :
  'd Placement.t ->
  ('v, 's) dtype ->
  ?endpoint:bool ->
  float ->
  float ->
  int ->
  ('v, 's, 'd) t
(** [linspace_at p dt ~endpoint start stop n] is
    {!linspace}[ dt ~endpoint start stop n] at [p]. *)

val logspace :
  (float, 's) dtype ->
  ?endpoint:bool ->
  ?base:float ->
  float ->
  float ->
  int ->
  (float, 's, host) t
(** [logspace dt ~endpoint ~base start stop n] is the host value of [base]
    (default [10.]) to the powers [linspace float64 ~endpoint start stop n],
    each power computed at float64 and stored once in [dt]:
    [logspace float32 0. 2. 3] is [1], [10] and [100].

    Raises [Invalid_argument] if [n < 0]. *)

val logspace_at :
  'd Placement.t ->
  (float, 's) dtype ->
  ?endpoint:bool ->
  ?base:float ->
  float ->
  float ->
  int ->
  (float, 's, 'd) t
(** [logspace_at p dt ~endpoint ~base start stop n] is
    {!logspace}[ dt ~endpoint ~base start stop n] at [p]. *)

val geomspace :
  (float, 's) dtype ->
  ?endpoint:bool ->
  float ->
  float ->
  int ->
  (float, 's, host) t
(** [geomspace dt ~endpoint start stop n] is the host value of [n] values in
    geometric progression from [start] to [stop], [stop] included where
    [endpoint] (default [true]): the value at [i] is
    [start (stop / start)^(i / (n - 1))], or [^(i / n)] without [endpoint].
    They are computed at float64 and stored in [dt]: the first is [start]
    and, with [endpoint] and [n >= 2], the last is [stop], each as [dt]
    stores it. [geomspace float32 1. 1000. 4] is [1], [10], [100] and
    [1000].

    Raises [Invalid_argument] if [n < 0], or if [start] or [stop] is not
    positive and finite. *)

val geomspace_at :
  'd Placement.t ->
  (float, 's) dtype ->
  ?endpoint:bool ->
  float ->
  float ->
  int ->
  (float, 's, 'd) t
(** [geomspace_at p dt ~endpoint start stop n] is
    {!geomspace}[ dt ~endpoint start stop n] at [p]. *)

val eye : ?m:int -> ?k:int -> ('v, 's) dtype -> int -> ('v, 's, host) t
(** [eye ~m ~k dt n] is the [n × m] host value ([m] defaults to [n]) whose
    element at row [i] and column [j] is one where [j - i = k] (default [0])
    and zero elsewhere: [k > 0] is a diagonal above the main one.

    Raises [Invalid_argument] if [n] or [m] is negative. *)

val eye_at :
  ?m:int -> ?k:int -> 'd Placement.t -> ('v, 's) dtype -> int -> ('v, 's, 'd) t
(** [eye_at ~m ~k p dt n] is {!eye}[ ~m ~k dt n] at [p]. *)

(** {2:like From a reference}

    These make a value of their argument's dtype where it lies. Its elements
    are never read. *)

val zeros_like : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [zeros_like x] is {!zeros}[ (dtype x) (shape x)]. *)

val ones_like : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [ones_like x] is {!ones}[ (dtype x) (shape x)]. *)

val full_like : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [full_like x v] is {!full}[ (dtype x) (shape x) v].

    Raises [Invalid_argument] as {!full} does. *)

val scalar_like : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [scalar_like x v] is {!scalar_at}[ (placement x) (dtype x) v], whole on
    each device of [x]'s placement.

    Raises [Invalid_argument] as {!scalar} does. *)

(** {2:moving Donation and copies}

    A program gives a value up with {!donate}, which makes a {e handle} of
    it: a value that one operation reads and may write its result over. The
    value given up is the handle's {e donor}. The operation that reads a
    handle {e consumes} it and is its {e consumer}: once the operation has
    checked every operand, the donor and every handle of it are {e dead},
    whatever its set's kernels then answer, and a later operation on any of
    them raises [Invalid_argument] naming the consumer, while their shape,
    dtype and placement still answer. An operation refused by its checks
    kills nothing. One whose kernels decline raises with them dead and
    returns their memory to its device, as a consumer that raises does
    ({!Prim.Consumer.with_}).

    A movement, which rearranges elements without computing them
    ({!reshape}, {!transpose}, {!flip}, {!broadcast_to}, slicing by ranges,
    {!sliding_window}), passes a new handle of the donor to its result,
    whether it makes a view or a copy, and kills nothing; the deaths wait
    for the final consumer. The handle it read stays a handle of the donor:
    once an operation consumes either, the other raises naming that
    consumer.

    A call consumes each operand that reaches one operation, through
    movements: the call is the consumer the deaths name, as {!trace} is for
    the diagonal it sums. An operand that several operations read is
    {e lent}: the call reads it as a value over the donor's memory, which is
    shared from then on, and the donor and its handles live on, as [a] does
    in [lstsq dt (donate a) b]. That memory returns to its device once no
    value holds it. A sampler's key is lent ({!Rng}). Two handles of one
    donor reaching one operation, the same handle twice, two donations of
    one value or a handle and one a movement passed on, are read as the
    donor beside its handle is: as the donor was when called, and it dies
    once. So [add (donate x) x] is [x + x], and [mul h h] is [x * x] for
    [h = donate x]. Where they reach different operations of a call, which
    run in an order the call fixes, one that reads the donor or a handle
    after another consumed it raises naming the call.

    An interpretation that receives an operation ({!Prim.interpret}) reads
    each handle as its arrays and kills nothing: the donor lives on, and its
    memory is shared from then on.

    On any number of domains, no value is consumed twice, and a call on
    another domain that overlaps a donor's consumer either reads the donor's
    elements as they were or raises [Invalid_argument]; it never reads the
    consumer's result. No consumer overwrites elements another call is
    reading: one that claims a handle while a call reads its memory computes
    into fresh memory. Racing consumers on several domains for the handles
    of one donor is a misuse: any of them may raise, naming another, and
    leave dead the other donors it claimed.

    The consumer may write its result into the donated memory when it reads
    that memory only at the result's own index, through the handle or
    through the donor itself (an elementwise operand, the target of {!set}
    or {!scatter}, {!contract}'s [init]), the memory is C-contiguous with
    the result's shape and dtype, and the memory is not shared. Memory is
    {e shared} once a value outside the donor's handles has been made over
    it (a view, a borrow by {!place}, a {!Repr} crossing, an
    interpretation's read, a read by a function of several operations), and
    stays shared. Otherwise the consumer computes into fresh memory, and a
    value that shares the memory keeps it. Its result is the same either
    way. *)

val donate : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [donate x] is a handle of [x] ({!section-moving}). [x] stays usable
    until an operation consumes a handle of [x]; then [x] and every handle
    of [x] are dead. A traced [x] is returned unchanged, and nothing dies.

    Raises [Invalid_argument] if [x] is dead. *)

val copy : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [copy x] is [x] stored afresh: equal elements in new memory. *)

(** {2:ocaml From OCaml values}

    A value made from OCaml data lies on the host, the set {!Host}. *)

val create : ('v, 's) dtype -> int array -> 'v array -> ('v, 's, host) t
(** [create dt s vs] is the host value of shape [s] holding [vs] in C order,
    a float stored as {!Dtype.of_float} says.

    Raises [Invalid_argument] if an extent is negative, [Array.length vs] is not
    the product of [s], or an element is an [int] outside [dt]'s range. *)

val init : ('v, 's) dtype -> int array -> (int array -> 'v) -> ('v, 's, host) t
(** [init dt s f] is the host value of shape [s] whose element at index [i] is
    [f i]. [f] is called once per index, in C order, each time with a fresh
    array.

    Raises [Invalid_argument] as {!create} does. *)

val map_item : ('v -> 'v) -> ('v, 's, 'd) t -> ('v, 's, host) t
(** [map_item f x] is the host value of [x]'s shape whose element at each index
    is [f] of [x]'s element there, a float stored as {!Dtype.of_float} says.
    [f] is an OCaml function, so it runs on the host, once per element in C
    order. [x], at any brand, is read as {!to_array} reads it.

    Raises [Invalid_argument] if [f] gives an [int] outside [x]'s dtype's
    range, and as {!to_array} does. *)

val of_bigarray :
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t -> ('v, 's, host) t
(** [of_bigarray b] is a host value holding a copy of [b]'s elements, of [b]'s
    shape and the dtype that stores its kind. Later writes to [b] do not reach
    it.

    The copy is one pass over [b], into memory the value owns, so its
    donation may write in place ({!donate}). To read [b]'s bytes in place
    instead,
    [Repr.of_array Host.v (Nx_array.of_bigarray dt b)] is the value over them
    ({!Nx_array.of_bigarray}, {!Repr.of_array}): a write to [b] from then on
    leaves what that value reads unspecified.

    Raises [Invalid_argument] if [b]'s kind is [char], [int] or [nativeint],
    which no dtype stores. *)

(** {1:reads Reads}

    A read gives a value's elements to OCaml. It works at every brand: it is a
    {{!section-waiting}wait} on the value's set, which may raise a failed
    check, then copies the value's elements to the host and leaves every
    placement as it was. A read reaches no interpretation.

    Each raises [Invalid_argument] naming itself for a dead value, and for a
    traced one, as [Nx.item: the value is traced by Rune.jit]; {!item} alone
    reads a traced value that has a twin, as its contract says. *)

val to_array : ('v, 's, 'd) t -> 'v array
(** [to_array x] is [x]'s elements in C order. *)

val item : int list -> ('v, 's, 'd) t -> 'v
(** [item i x] is [x]'s element at [i], one position per axis, a negative one
    counting from the end; [item [] x] reads a 0-d value. A traced [x] with a
    twin ({!Prim.twinned}) reads as its twin where [x]'s interpretation is
    around the call and no [Extent] interpretation lies between them, and
    raises as any traced value elsewhere.

    Raises [Invalid_argument] if [i] does not have one position per axis of [x]
    or a position is outside its axis. *)

val to_bigarray :
  ('v, 's) Bigarray.kind ->
  ('v, 's, 'd) t ->
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_bigarray k x] is a fresh bigarray of [x]'s shape holding its elements.
    A dtype Bigarray has no kind for is a type error.

    Raises [Invalid_argument] if [x] has more than 16 axes, Bigarray's most. *)

(** {1:waiting Waiting and checks}

    A set's devices may compute while the caller goes on: an operation returns
    once its work is submitted. The host waits for a set only at a wait:
    {!sync}, a {{!section-reads}read}, or {!place} onto {!Host}.

    A check is a condition on data whose failure raises the caller's exception:
    one stated with {!check}, a sampler's parameter outside its domain
    ({!Rng}), or a {!Prim.Check}. On
    the host it raises at the call. On another set it is computed where its data
    lies, and its failure raises at the next wait on that set on the domain that
    issued it, so that no check makes the host wait, except on a device of
    another machine, where reading whether a check passed waits for the work
    submitted there. The systhreads of a domain
    share its checks. A wait drains the checks its domain issued on the set
    before it and raises the earliest failure among them; the later ones are
    dropped. A wait on one set raises nothing of another's checks; a check no
    wait follows never raises. A check keeps its data's memory until a wait
    drains it, or until a later check on its set is issued once its devices
    report its flags computed, so a donation of that data meanwhile computes
    into fresh memory. A device reports its work at most a fixed number of
    submissions late while work is submitted on it ({!Rig.section-timeline}),
    and a check on a device that gets no more work is found computed when the
    next check is issued, so the checks of a loop that never waits keep the
    data of a bounded window of steps. A failing check's data is read when
    its failure is found, at a wait or at a later check's issue, and that
    read may wait for the set. Once a check is known to fail, the checks
    after it release their data, and a check issued later on its set is not
    kept. *)

val sync : ('v, 's, 'd) t -> unit
(** [sync x] waits for the work [x] depends on, then drains the checks pending
    on [x]'s set, raising the earliest failure, as stated above.

    Raises [Invalid_argument] naming [Nx.sync] for a dead value, and for a
    traced one: [Nx.sync: the value is traced by Rune.jit]. *)

(** {1:shapes Shapes, broadcasting and movements}

    Axes count from [0], and a negative axis from the end: [-1] is the last.
    Shapes broadcast from their last axes: aligned there, each pair of extents
    is equal or one of them is [1], and missing leading axes count as [1].

    A movement rearranges elements without computing on them. Each says whether
    it makes a {e view}, a value over its operand's memory that costs no copy.
    That is a cost, never part of the result's meaning: a value never changes,
    so a view and a copy read the same.

    Over a value cut across devices, a movement keeps the cut of an axis it
    keeps in order, as a permutation, a broadcast that does not stretch the
    axis and a slice of all of it do. A reshape keeps the cut of an axis it
    factors or merges where the tiles divide both extents, each window then
    holding whole rows. Any other axis it reads whole ({!section-routing}). *)

val ndim : ('v, 's, 'd) t -> int
(** [ndim x] is [x]'s number of axes. *)

val dim : int -> ('v, 's, 'd) t -> int
(** [dim a x] is the extent of [x]'s axis [a].

    Raises [Invalid_argument] if [a] is not an axis of [x]. *)

val numel : ('v, 's, 'd) t -> int
(** [numel x] is [x]'s number of elements: the product of its extents, [1] for a
    0-d value. *)

val nbytes : ('v, 's, 'd) t -> int
(** [nbytes x] is the bytes [x]'s elements fill when stored contiguously,
    [(numel x * b + 7) / 8] for a dtype of [b] bits. *)

val reshape : int array -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [reshape s x] is [x]'s elements in C order at shape [s]. One extent of [s]
    may be [-1]; it is inferred from the others. A view where strides express
    it, a copy otherwise.

    Raises [Invalid_argument] if the element counts differ, an extent is below
    [-1], or two are [-1]. *)

val broadcast_to : int array -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [broadcast_to s x] is [x] stretched to shape [s]: an axis of extent [1], or
    one [x] lacks, repeats [x]'s elements along it. A view.

    Raises [Invalid_argument] unless [x]'s shape broadcasts with [s] to [s]. *)

val broadcast_shapes : int array list -> int array
(** [broadcast_shapes ss] is the shape that [ss] broadcast to, [[||]] for [[]].

    Raises [Invalid_argument] if an extent is negative or two of them do not
    broadcast. *)

val broadcast_arrays : ('v, 's, 'd) t list -> ('v, 's, 'd) t list
(** [broadcast_arrays xs] is each of [xs] stretched to {!broadcast_shapes} of
    their shapes, in order. Views.

    Raises [Invalid_argument] if two of their shapes do not broadcast. *)

val meshgrid :
  indexing:[ `Xy | `Ij ] -> ('v, 's, 'd) t list -> ('v, 's, 'd) t list
(** [meshgrid ~indexing [x1; …; xn]] is the coordinate grids of the 1-d [x1]
    to [xn]: [n] views of one shape, the [i]th holding [xi]'s element [j]
    wherever its axis for [xi] is at [j]. With [`Ij], [xi]'s axis is axis
    [i], of shape [[|n1; …; nn|]]. With [`Xy], the first two exchange, so
    that [x1] runs along columns and [x2] along rows as a plane's x and y
    do: [[|n2; n1; n3; …; nn|]].

    Raises [Invalid_argument] if an operand is not 1-d. *)

val squeeze : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [squeeze ~axes x] is [x] without the axes [axes]; without [axes], without
    every axis of extent [1]. A view.

    Raises [Invalid_argument] if an axis of [axes] is not [x]'s, repeats, or has
    an extent other than [1]. *)

val unsqueeze : axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [unsqueeze ~axes x] is [x] with an axis of extent [1] at each position of
    [axes], positions of the result: [unsqueeze ~axes:[ 0; -1 ]] of a [[3]]
    value is [[1; 3; 1]]. A view.

    Raises [Invalid_argument] if a position is not an axis of the result or
    repeats. *)

val flatten : ?start_dim:int -> ?end_dim:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [flatten ~start_dim ~end_dim x] is [x] with its axes [start_dim] to
    [end_dim], both included, merged into one. [start_dim] defaults to [0] and
    [end_dim] to [-1]; a 0-d [x] flattens as the [[1]] value it holds. A view
    where strides express it, a copy otherwise.

    Raises [Invalid_argument] if either is not an axis of [x] or [start_dim]
    comes after [end_dim]. *)

val transpose : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [transpose ~axes x] has [x]'s axis [List.nth axes i] as its axis [i];
    without [axes], [x]'s axes in reverse order. A view.

    Raises [Invalid_argument] unless [axes] lists each of [x]'s axes once. *)

val moveaxis : int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [moveaxis a b x] is [x] with its axis [a] moved to position [b], the other
    axes in their order. A view.

    Raises [Invalid_argument] if [a] or [b] is not an axis of [x]. *)

val swapaxes : int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [swapaxes a b x] is [x] with its axes [a] and [b] exchanged. A view.

    Raises [Invalid_argument] if [a] or [b] is not an axis of [x]. *)

val flip : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [flip ~axes x] is [x] with the order of elements along each axis of [axes]
    reversed; without [axes], along every axis. A view.

    Raises [Invalid_argument] if an axis of [axes] is not [x]'s or repeats. *)

val sliding_window :
  ?axis:int -> window:int -> ?step:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sliding_window ~axis ~window ~step x] is [x]'s windows of [window]
    consecutive elements along [axis] (default [-1]), one every [step] (default
    [1]). Axis [axis] becomes the number of windows, [(d - window) / step + 1]
    for an extent [d], or [0] where [window] exceeds [d], and an axis of extent
    [window] is appended: window [w]'s element [j] is [x]'s element
    [w * step + j] along [axis]. A view, whose windows share elements where
    [step < window].

    Raises [Invalid_argument] if [axis] is not [x]'s, or [window] or [step] is
    below [1]. *)

val split : axis:int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t list
(** [split ~axis n x] is [x] cut along [axis] into [n] runs of consecutive
    elements, in order. With [d] the extent, the first [d mod n] runs have
    [d / n + 1] elements and the others [d / n]. Views.

    Raises [Invalid_argument] if [axis] is not [x]'s or [n < 1]. *)

val concatenate : axis:int -> ('v, 's, 'd) t list -> ('v, 's, 'd) t
(** [concatenate ~axis xs] is [xs] joined along their axis [axis], in order. A
    copy.

    Raises [Invalid_argument] if [xs] is empty, [axis] is not an axis of its
    first value, or their shapes differ off [axis]. *)

val stack : ?axis:int -> ('v, 's, 'd) t list -> ('v, 's, 'd) t
(** [stack ~axis xs] is [xs] joined along a new axis at position [axis] of the
    result (default [0]), in order. A copy.

    Raises [Invalid_argument] if [xs] is empty, their shapes differ, or [axis]
    is not an axis of the result. *)

val pad : (int * int) array -> 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [pad widths v x] is [x] with [fst widths.(i)] elements [v] before its axis
    [i] and [snd widths.(i)] after it. A copy.

    Raises [Invalid_argument] if [widths] does not have one pair per axis of
    [x], a width is negative, or [v] is an [int] outside [x]'s dtype's range. *)

val roll : ?axis:int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [roll ~axis k x] is [x] with its elements along [axis] shifted [k] places
    toward the end, those past it wrapping to the start; a negative [k] shifts
    toward the start. Without [axis], [flatten x] shifted, at [x]'s shape. A
    copy.

    Raises [Invalid_argument] if [axis] is not an axis of [x]. *)

val tile : int array -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tile reps x] is [x] repeated [reps.(i)] times, end to end, along its axis
    [i]. With more entries in [reps] than [x] has axes, [x] first gains leading
    axes of extent [1]. A view where strides express it, a copy otherwise.

    Raises [Invalid_argument] if [reps] has fewer entries than [x] has axes or a
    negative one. *)

val repeat : ?axis:int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [repeat ~axis n x] is [x] with each element repeated [n] times in place
    along [axis]: [repeat ~axis:0 2] of [[a; b]] is [[a; a; b; b]]. Without
    [axis], [flatten x] repeated. A view where strides express it, a copy
    otherwise.

    Raises [Invalid_argument] if [axis] is not an axis of [x] or [n < 0]. *)

(** {1:patterns Axis patterns}

    A pattern states a layout in words. Names are words separated by spaces,
    parentheses group axes merged into one, [1] is an axis of extent [1], and
    [...] stands for the axes no name covers. One grammar serves {!rearrange},
    with one operand, {!contract}, with two, and {!einsum}, with two or more:

    {v
    pattern  ::= operands "->" layout [ "|" name { name } ]
    operands ::= layout { "," layout }
    layout   ::= { axis }
    axis     ::= name | "(" name { name } ")" | "1" | "..."
    name     ::= a letter, then letters, digits and "_"
    v}

    {[
    let split_heads = Nx.Pattern.v "b t (h d) -> b h t d"
    let merge_heads = Nx.Pattern.inverse split_heads
    let q = Nx.rearrange ~sizes:[ ("h", heads) ] split_heads y
    ]} *)

(** Axis patterns. *)
module Pattern : sig
  type t
  (** The type for axis patterns, parsed and checked. *)

  val v : string -> t
  (** [v s] is the pattern [s]. Each call parses [s]; a pattern bound at a
      module's top level is parsed once, as the module starts.

      With one operand, every name appears once on each side; a group on the
      left splits an axis and on the right merges axes; [1] drops a unit axis on
      the left and adds one on the right; [...] is the same axes on both sides.

      With two or more, a name in the result is a batch axis where two operands
      or more hold it and a free axis where one does, and the names after [|]
      are summed: each is in two operands or more and not in the result. Every
      operand name is in the result or summed. A group in an operand splits
      that operand's axis, and an extent a group leaves unknown comes from
      another operand that holds the name. [...] stands for the same leading
      axes in every layout, kept as batch axes; [1] drops a unit axis of an
      operand or adds one to the result.

      Sums are written out because a name dropped from the result is easy to
      miss. In ["b h q d, b h k d -> h q k | d"], [b] was left out by mistake;
      summing it would give a result of plausible shape and wrong values, so
      the pattern raises.

      Raises [Invalid_argument] naming [s] if it is not in the grammar, a name
      repeats within a layout, [...] appears twice in a layout, or a name breaks
      the rules above. A string in NumPy's einsum notation, such as
      ["bik,bkj->bij"], raises with the pattern it means:
      ["b i k, b k j -> b i j | k"]. *)

  val inverse : t -> t
  (** [inverse p] is the one-operand pattern that undoes [p]: its sides
      exchanged.

      Raises [Invalid_argument] if [p] has more than one operand. *)
end

val rearrange :
  ?sizes:(string * int) list -> Pattern.t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rearrange ~sizes p x] is [x] split, permuted and merged as [p] says: the
    result's element at an index is [x]'s element where each name takes the same
    position. A name alone on the left takes its axis's extent, and [sizes]
    gives others; a group on the left may hold one name whose extent neither
    gives, the quotient of its axis by the others. A view where strides express
    it, a copy otherwise.

    Raises [Invalid_argument] if [p] has more than one operand, [p]'s left side
    names more axes than [x] has or, without [...], other than [x]'s rank, a
    [1] on the left meets an extent other than [1], a group's extents do not
    divide or multiply to its axis, a group has two unknown extents, a size
    disagrees with the extent [x] gives its name, a name in [sizes] is not in
    [p] or is there twice, or a size is negative. *)

(** {1:indexing Indexing and slicing}

    A position written in the program ([I], [L], the ranges) is checked: one
    outside its axis raises, and a negative one counts from the end. A position
    held in data ([T], {!take}, {!take_along_axis}, {!scatter}) never raises and
    never counts from the end: outside [\[0, d)], negative included, it reads
    zero ([+0.], [false], [0 + 0i]) and its write is dropped. That makes a
    gather and an add-scatter adjoint; a program that wants a loud failure
    states it with a check. A [D] start, also held in data, clamps into
    [\[0, d - n\]] instead, so its window lies in the axis. *)

(** The type for selections along one axis. *)
type 'd index =
  | I of int  (** One position; the axis goes. *)
  | L of int list  (** Positions written in the program; a gather. *)
  | T of 'd int64_t
      (** Positions held in data, replacing the axis by their axes; a gather. *)
  | R of int * int
      (** [R (start, stop)]: [start] to [stop - 1]. A negative [start] or [stop]
          counts from the end; then both clip into [\[0, d\]]. *)
  | Rs of int * int * int
      (** [Rs (start, stop, step)]: [start], [start + step], … toward [stop],
          [stop] excluded. A negative [start] or [stop] counts from the end;
          then both clip into [\[0, d\]] for [step > 0] and into
          [\[-1, d - 1\]] for [step < 0], so [Rs (-1, -d - 1, -1)] is the axis
          reversed. *)
  | A  (** The whole axis. *)
  | N  (** A new axis of extent [1]; it selects along no axis of [x]. *)
  | D of 'd int64_t * int
      (** [D (start, n)]: [n] consecutive positions from the 0-d [start] held
          in data, the start clamped into [\[0, d - n\]]; a gather. *)
  | M of 'd bool_t
      (** A mask over as many axes as it has, from the next one, of their
          shape: the positions where it holds, in C order, replace those
          axes by one; a gather. A 0-d mask covers no axis and adds one of
          extent [1] or [0]. Its count is a length that is data
          ({!section-counted}). *)

val slice : 'd index list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [slice idx x] selects from [x] axis by axis: each entry selects along the
    next axis, or a mask along its axes, independently of the others, [N]
    excepted, and axes past the list are whole. [I], [R], [Rs], [A] and [N]
    give a view; [L], [T], [D] and [M] gather.

    Raises [Invalid_argument] if [idx] addresses more axes than [x] has, a
    written position is outside its axis, a step is [0], a [D] start is not
    0-d, a [D] length is negative or exceeds its axis, a mask's shape is not
    that of the axes it covers, or a mask is traced. Under a trace,
    [T (positions ~size m)] selects, or {!where} computes without
    selecting. *)

val get : int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [get p x] is [slice (List.map (fun i -> I i) p) x], a view. *)

val take : ?axis:int -> 'd int64_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [take ~axis p x] replaces [x]'s axis [axis] by [p]'s axes, reading [x] at
    each position [p] holds; without [axis], [flatten x] is read. A position
    outside [\[0, d)] reads zero. A gather.

    Raises [Invalid_argument] if [axis] is not an axis of [x]. *)

val take_along_axis :
  axis:int -> 'd int64_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [take_along_axis ~axis p x] has [p]'s shape broadcast with [x]'s off
    [axis]; its element at [j] is [x]'s element at [j] with axis [axis]
    replaced by [p]'s element at [j]. A position outside [\[0, d)] reads zero.
    A gather.

    Raises [Invalid_argument] if [axis] is not an axis of [x], [p]'s rank is
    not [x]'s, or the shapes off [axis] do not broadcast. *)

val unravel_index : 'd int64_t -> int array -> 'd int64_t list
(** [unravel_index i s] is the coordinates in the shape [s], in C order, of
    each flat index of [i]: one value of [i]'s shape per axis of [s], [[]]
    for a 0-d [s]. An index [k] in [\[0, n)], [n] the number of elements of
    [s], has [(k / t) mod s.(j)] along axis [j], [t] the product of the
    extents after [j]. An index outside [\[0, n)] has each axis' extent,
    outside it, so {!take} reads zero there and {!scatter} writes nothing.

    Raises [Invalid_argument] if [s] is not a shape. *)

(** {1:updates Functional updates} *)

val set : 'd index list -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [set idx v x] is [x] with [slice idx x] replaced by [v], broadcast to the
    selection's shape. [x] is unchanged. Without {!donate} it costs a copy of
    [x]; [set idx v (donate x)] writes only the selection, into [x]'s memory,
    where {!donate} lets the consumer write there. A position in [T] outside its
    axis writes nothing; where [T] repeats a position, the last update, in C
    order of the selection, wins. A [D] window clamps its start as {!slice}
    does, so a start past [d - n] writes the last [n] positions: a decode step
    whose position reaches the cache's capacity overwrites its last row, and a
    program that can pass capacity states the bound with a check. A mask
    writes the positions where it holds. [v] is checked against its count
    only once the mask is read: a [v] that fits the selection but for that
    count raises with a donated mask consumed.

    Where one entry selects by positions and every other is [A], [N] or a
    range of its whole axis, the update is a {!scatter} along that axis.
    Otherwise the positions of each [L], [T] and [D] entry are read
    broadcast to the selection's shape, beside those of the entries before
    it, at the union of their cuts in the order of [idx]
    ({!section-routing}). A refusal names the entry's positions as operand
    [1], broadcast to the selection, a value the program does not hold,
    beside the entries before it as operand [0]; the {!place} to write puts
    that entry's positions, as given, whole on each device,
    [Placement.on s].

    Raises [Invalid_argument] as {!slice} does, if [v] does not broadcast to
    the selection's shape, or if an [L] names one position twice, counting
    from the end. *)

(** The type for how {!scatter} combines a target with its updates. *)
type combine = Nx_kernel.combine =
  | Set  (** The last update. *)
  | Add  (** The target plus its updates. *)
  | Max  (** The largest of the target and its updates, or a NaN. *)
  | Min  (** The smallest, or a NaN. *)

val scatter :
  ?combine:combine ->
  axis:int ->
  'd int64_t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [scatter ~combine ~axis p u x] is [x] with each element of [u] combined
    into [x] at its own index with axis [axis] replaced by [p]'s element there;
    [p] and [u] have one shape. A target's updates apply in C order of [u]:
    [Set] (default) keeps the last; [Add] gives the element plus its updates,
    with no [+0] added, so a [-0.] element and [-0.] updates give [-0.], a
    narrow float summed in float32 and rounded once; [Max] and
    [Min] give the largest or smallest of the element and its updates, or a
    NaN where one of them is. Where no update lands, the element keeps
    its bits, a signalling NaN's included. A position outside [\[0, d)]
    writes nothing. A float [Add] gives the same bits on every run, adding in
    an order its kernel library states. For finite operands, barring
    overflow, it is within [γ(n + 1) (|x| + Σ |ui|)] of the exact sum of an
    element [x] and its [n] updates before a narrow float's rounding:
    [γ(k) = k u / (1 - k u)], [u] the unit roundoff of float32 for a narrow
    float, else of the dtype's parts. Without
    {!donate} it costs a copy of [x]; [scatter ~axis p u (donate x)] writes
    only the updates, into [x]'s memory, where {!donate} lets the consumer
    write there.

    Raises [Invalid_argument] if [axis] is not an axis of [x], [p] and [u]
    differ in shape, their shape differs from [x]'s off [axis], or [combine]
    is [Add] and the dtype is boolean. *)

(** {1:arith Arithmetic}

    Elementwise operations compute where their operands lie and give a new
    value. Binary operations broadcast: aligned at their last axes, two extents
    are equal or one of them is [1], which stretches. Integers wrap. The float
    formats narrower than 32 bits compute at [float32] and round once.

    Each raises [Invalid_argument] naming itself if the shapes do not broadcast,
    or if its operation does not take the operands' dtype: what each takes is
    stated with it. The host computes every dtype stated. Where a device's
    kernels do not compute one, as a GPU's may not compute complex numbers,
    the function raises naming itself. *)

val add : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [add a b] is the elementwise sum. Every dtype but booleans. *)

val sub : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sub a b] is [a - b], as {!add}. *)

val mul : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mul a b] is the elementwise product, as {!add}. *)

val div : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [div a b] is [a / b]: the IEEE 754 quotient on floats and complex numbers;
    on integers the quotient truncated toward zero, [0] for a divisor of [0],
    and a signed dtype's least value for that value by [-1]. Every dtype but
    booleans. *)

val mod_ : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mod_ a b] is the remainder of [a / b], of [a]'s sign: [fmod] on floats, and
    [a] for a divisor of [0] on integers, so that
    [a = add (mul b (div a b)) (mod_ a b)] on integers. Floats and integers. *)

val pow : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [pow a b] is [a] to the power [b]. On complex numbers it is the principal
    value [exp (b * log a)]: [1] where [b] is [0], [0] at [a = 0] where
    [b]'s real part is positive, with [log]'s branch cut
    ({!section-transcendental}); its error grows with [|b log a|], within
    [(3 + 3|b log a|) u |a{^b}|]. On floats it is within 4 ulps at float32
    and 2 at float64, with Annex F's special values
    ({!section-transcendental}). Every dtype but booleans. *)

val fma : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [fma a b c] is [a * b + c] rounded once. Floats and integers. *)

val lerp :
  (float, 's, 'd) t ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [lerp a b w] is [a + w (b - a)], by one {!fma}: [a + w (b - a)] where
    [w < 0.5], else [b + (w - 1) (b - a)], [b - a] rounded once first. Where
    [b - a] is finite, [w = 0]
    gives [a] and [w = 1] gives [b], and [lerp a a w] is [a] for a finite
    [w], each equal as a number: a [-0.] there may give [+0.]. Floats. *)

val maximum : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [maximum a b] is the greater element by the dtype's order ({!less}). On
    floats it is the IEEE 754 maximum: a NaN gives NaN, and [-0.] orders below
    [+0.]. Every dtype. *)

val minimum : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [minimum a b] is the lesser element, as {!maximum}. *)

val neg : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [neg x] is [-x]. Every dtype but booleans. *)

val square : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [square x] is [mul x x]. Every dtype but booleans. *)

val recip : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [recip x] is [1 / x]; on integers, [x] for [1] and [-1] and [0] otherwise.
    Every dtype but booleans. *)

val abs : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [abs x] is [x]'s absolute value. Floats and integers: a complex number's
    modulus is a float, {!magnitude}. *)

val sign : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sign x] is [-1], [0] or [1] by [x]'s sign, and NaN for a NaN. On complex
    numbers it is [x / |x|], of modulus [1], and [x] itself at [0]; at an
    infinite part, the direction its parts point to:
    [sign (inf + 2i) = 1 + 0i]; within [3 u] as {!section-transcendental}
    measures it. Every dtype but booleans. *)

(** {2:scalars Scalar forms}

    [f_s x c] is [f x (scalar_like x c)] and [rf_s c x] is
    [f (scalar_like x c) x], [c] stored in [x]'s dtype as
    {!Dtype.of_float} says for a float. [c] is written into the program of
    one operation over [x] alone, as {!full} writes its value. Each raises
    as its function does, naming itself, and [Invalid_argument] if [c] is an
    [int] outside [x]'s dtype's range. *)

val add_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [add_s x c] is [add x c]. *)

val sub_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [sub_s x c] is [sub x c]. *)

val mul_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [mul_s x c] is [mul x c]. *)

val div_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [div_s x c] is [div x c]. *)

val pow_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [pow_s x c] is [pow x c]. *)

val mod_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [mod_s x c] is [mod_ x c]. *)

val maximum_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [maximum_s x c] is [maximum x c]. *)

val minimum_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [minimum_s x c] is [minimum x c]. *)

val rsub_s : 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rsub_s c x] is [sub c x]. *)

val rdiv_s : 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rdiv_s c x] is [div c x]. *)

val rpow_s : 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rpow_s c x] is [pow c x]. *)

val equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [equal_s x c] is [equal x c]. *)

val not_equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [not_equal_s x c] is [not_equal x c]. *)

val less_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [less_s x c] is [less x c]. *)

val less_equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [less_equal_s x c] is [less_equal x c]. *)

val greater_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [greater_s x c] is [greater x c]. *)

val greater_equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [greater_equal_s x c] is [greater_equal x c]. *)

(** {2:operators Operators}

    The arithmetic operators, for a local open: [Nx.(sin x * x +$ 1.)].
    Comparisons and logical operators live in {!Infix}, so that [Nx.( … )]
    keeps OCaml's own. The remainder has no operator: {!mod_} takes the
    dividend's sign. *)

val ( + ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a + b] is [add a b]. *)

val ( - ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a - b] is [sub a b]. *)

val ( * ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a * b] is [mul a b]. *)

val ( / ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a / b] is [div a b]. *)

val ( ** ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a ** b] is [pow a b]. *)

val ( ~- ) : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [-x] is [neg x]. *)

val ( +$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x +$ c] is [add_s x c]. *)

val ( -$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x -$ c] is [sub_s x c]. *)

val ( *$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x *$ c] is [mul_s x c]. *)

val ( /$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x /$ c] is [div_s x c]. *)

val ( **$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x **$ c] is [pow_s x c]. *)

val ( land ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a land b] is [bitwise_and a b]. *)

val ( lor ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a lor b] is [bitwise_or a b]. *)

val ( lxor ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a lxor b] is [bitwise_xor a b]. *)

val ( *@ ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a *@ b] is [matmul a b]. *)

val ( .%{} ) : ('v, 's, 'd) t -> int list -> ('v, 's, 'd) t
(** [x.%{p}] is [get p x]. *)

val ( .${} ) : ('v, 's, 'd) t -> 'd index list -> ('v, 's, 'd) t
(** [x.${idx}] is [slice idx x]. *)

(** Comparisons and logical operators, for [open Nx.Infix]. *)
module Infix : sig
  val ( = ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a = b] is [equal a b]. *)

  val ( <> ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a <> b] is [not_equal a b]. *)

  val ( < ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a < b] is [less a b]. *)

  val ( <= ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a <= b] is [less_equal a b]. *)

  val ( > ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a > b] is [greater a b]. *)

  val ( >= ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a >= b] is [greater_equal a b]. *)

  val ( =$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x =$ c] is [equal_s x c]. *)

  val ( <>$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x <>$ c] is [not_equal_s x c]. *)

  val ( <$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x <$ c] is [less_s x c]. *)

  val ( <=$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x <=$ c] is [less_equal_s x c]. *)

  val ( >$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x >$ c] is [greater_s x c]. *)

  val ( >=$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x >=$ c] is [greater_equal_s x c]. *)

  val ( && ) : 'd bool_t -> 'd bool_t -> 'd bool_t
  (** [a && b] is [logical_and a b]. Both operands are computed, whatever
      [a] holds. *)

  val ( || ) : 'd bool_t -> 'd bool_t -> 'd bool_t
  (** [a || b] is [logical_or a b], as [( && )]. *)
end

(** {2:transcendental Powers, exponentials and trigonometry}

    Floats and complex numbers, where a function does not say floats. A stated
    range is the mathematical one, which rounding can leave at an end: in
    [float8_e4m3fn], [π] rounds to [3.25].

    On floats the {e error} of a result is its distance in floats of the
    dtype from the exact value rounded to the dtype: a neighbour is one
    away, and [-0.] and [+0.] are one apart. Each function states a bound on
    it at float32 and float64. Whatever the bound, a result is exact at the
    special values of C99's Annex F (ISO/IEC 9899, F.10): [exp (-inf)] is
    [+0.], [log] of [±0.] is [-inf], [sin] of an infinity is NaN and
    [atan inf] is [π/2] rounded. Where the exact value rounds to an infinity
    or to zero, the result is that infinity or zero, of the exact value's
    sign: [exp 710.] is [inf] and [pow (-2.) (-1101.)] is [-0.]. Another zero
    result has the sign Annex F gives it: [sin (-0.)] is [-0.]. A NaN operand
    gives a NaN, but where Annex F gives a number for every value of that
    operand: [pow nan 0.] and [pow 1. nan] are [1.]. Which NaN, its sign and
    its payload are unspecified.

    The formats narrower than 32 bits compute at float32 and round once
    ({!section-arith}). Where a bound is in ulps, their result is within one
    ulp of the exact value rounded to their dtype. A format without
    infinities or NaN stores such a value as {!cast} does: [log] of a
    [float8_e4m3fn] zero is [-448.].

    nx.cpu keeps subnormal numbers ({!Nx_cpu}). A device whose kernel
    library flushes them to zero reads a subnormal operand as zero and
    writes a subnormal result as zero; there each bound, of this section and
    of {!section-special}, holds where the operands and the result are
    normal.

    On complex numbers each is the principal value C99 defines (ISO/IEC 9899,
    7.3), with its special values at infinities, NaNs and signed zeros (Annex
    G). Where a result's sign comes from a NaN part, as the imaginary [∞] of
    [sqrt (-∞ + NaN i)] does, that sign is unspecified, as Annex G allows. A
    function whose inverse is many-valued has branch cuts on an axis, and the
    sign of a zero part names the side it continues from: [sqrt (-4 + 0i)] is
    [2i] and [sqrt (-4 - 0i)] is [-2i]. A result [f'] is within [k u |f|] of
    the value [f], where [|f|] is a normal float: [u] is the parts' unit
    roundoff, [2{^-24}] for complex64 and [2{^-53}] for complex128, and [k] is
    [3] for {!sqrt}, [4] for {!exp}, {!exp2}, {!log} and {!log2}, [5] for
    {!asin}, {!acos}, {!atan}, {!asinh}, {!acosh} and {!atanh}, and [6] for
    {!sin}, {!cos}, {!tan}, {!sinh}, {!cosh} and {!tanh}. On the real axis, at
    a non-zero real part where the float function is defined, a function of
    one operand but {!asinh}, {!acosh} and {!atanh} gives the float function's
    result as its real part; {!pow} does at a positive base and a real
    exponent.

    These bounds, and those the functions below state, hold where a set's
    kernels compute. Under a compiler such as [Rune.jit], the compiler's
    bounds hold. *)

val sqrt : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sqrt x] is the square root, NaN below [-0.], correctly rounded on
    floats. On complex numbers its real part is never negative, and its
    branch cut is the negative real axis. *)

val rsqrt : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rsqrt x] is [recip (sqrt x)]: [inf] at [+0.], [-inf] at [-0.], NaN below.
    Within 2 ulps at float32 and float64. On complex numbers it is {!recip}
    of {!sqrt}, NaN at [0]. *)

val hypot : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [hypot x y] is [sqrt (x{^2} + y{^2})] with no intermediate overflow: finite
    wherever the result is. It is [inf] where [x] or [y] is an infinity, a NaN
    beside it included, and NaN where either is a NaN otherwise. Within
    3 ulps at float32 and float64. Floats. *)

val exp : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [exp x] is [e{^x}]. Within 2 ulps at float32 and float64. *)

val exp2 : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [exp2 x] is [2{^x}]. Within 2 ulps at float32 and float64. *)

val expm1 : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [expm1 x] is [e{^x} - 1], accurate near [0]: [-0.] at [-0.] and [-1.] at
    [-inf]. Within 1 ulp at float32 and 2 at float64. Floats. *)

val log : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [log x] is the natural logarithm: [-inf] at [0.], NaN below. Within 1
    ulp at float32 and 2 at float64. On complex numbers it is
    [log |x| + i angle x], its imaginary part in \[[-π], [π]\] and its
    branch cut the negative real axis. *)

val log2 : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [log2 x] is the base-2 logarithm. Within 1 ulp at float32 and 2 at
    float64. *)

val log1p : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [log1p x] is [log (1 + x)], accurate near [0]: [-0.] at [-0.], [-inf] at
    [-1.], NaN below. Within 1 ulp at float32 and 2 at float64. Floats. *)

val sin : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sin x] is the sine of [x] radians. Within 2 ulps at float32 and
    float64. *)

val cos : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cos x] is the cosine of [x] radians. Within 2 ulps at float32 and
    float64. *)

val tan : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tan x] is the tangent of [x] radians. Within 4 ulps at float32 and 2
    at float64. *)

val asin : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [asin x] is the arcsine, in \[[-π/2], [π/2]\]: on complex numbers its real
    part, with branch cuts on the real axis beyond [-1] and [1]. Within
    2 ulps at float32 and float64. *)

val acos : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [acos x] is the arccosine, in \[[0], [π]\]: on complex numbers its real
    part, with {!asin}'s branch cuts. Within 2 ulps at float32 and
    float64. *)

val atan : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atan x] is the arctangent, in \[[-π/2], [π/2]\]: on complex numbers its
    real part, with branch cuts on the imaginary axis beyond [-i] and [i].
    Within 2 ulps at float32 and float64. *)

val atan2 : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atan2 y x] is the angle of the point [(x, y)], in \[[-π], [π]\]:
    [-π] for [y = -0.] and [x < 0]. Within 3 ulps at float32 and 2 at
    float64. Floats. *)

val sinh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sinh x] is the hyperbolic sine. Within 3 ulps at float32 and 2 at
    float64. *)

val cosh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cosh x] is the hyperbolic cosine. Within 2 ulps at float32 and
    float64. *)

val tanh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tanh x] is the hyperbolic tangent. Within 2 ulps at float32 and
    float64. *)

val asinh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [asinh x] is the inverse hyperbolic sine, odd, [x] itself near [0.]: within
    3 ulps at float32 and float64. On complex numbers its imaginary part is in
    \[[-π/2], [π/2]\], with branch cuts on the imaginary axis beyond [-i]
    and [i]. *)

val acosh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [acosh x] is the inverse hyperbolic cosine: [0.] at [1.], NaN below [1.],
    within 3 ulps at float32 and float64. On complex numbers its real part is
    not negative and its imaginary part is in \[[-π], [π]\], with a branch
    cut on the real axis below [1]. *)

val atanh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atanh x] is the inverse hyperbolic tangent: [inf] at [1.], [-inf] at
    [-1.], NaN beyond them, odd, [x] itself near [0.]; within 3 ulps at
    float32 and float64. On complex numbers its imaginary part is in
    \[[-π/2], [π/2]\], with branch cuts on the real axis beyond [-1] and
    [1]. *)

val erf : (float, 's, 'd) t -> (float, 's, 'd) t
(** [erf x] is the error function [2/√π ∫₀ˣ e{^-t²} dt]: [-1.] at [-inf]
    and [1.] at [inf]. Within 2 ulps at float32 and float64. *)

(** {2:special Special functions}

    The laws of the distributions {!Rng} samples. Arguments broadcast,
    parameters first and the variable last: [gammainc a] is the distribution
    function of the gamma distribution of shape [a]. Outside a function's
    domain, and at a NaN argument, the value is NaN; at a pole it is the
    function's limit there.

    Each function states a bound at float32 and float64 on its error
    ({!section-transcendental}). A bound of [k] ulps {e absolute near zeros}
    also holds where the exact value is below [1.] in magnitude and the result
    within [kε] of it, [ε] being the distance from [1.] to the next float of
    the dtype. An inverse's bound adds [κ] times its forward function's bound,
    where [κ = |p / (x F'(x))|] at its result [x], for [F] the forward
    function, is its condition number. A subnormal result is held to its
    bound in units of the least subnormal, plus half of one. These bounds
    hold where a set's kernels compute. Under a compiler such as
    [Rune.jit], the compiler's bounds hold.

    The formats narrower than 32 bits follow {!section-transcendental};
    where a bound is absolute, their result is within that bound at
    [float32] plus half an ulp of their dtype. *)

val erfc : (float, 's, 'd) t -> (float, 's, 'd) t
(** [erfc x] is [1 - erf x], to its own relative precision where [erf x]
    rounds to [1.]: [erfc 10.] is [2.088e-45] at float64. It is [2.] at
    [-inf] and [0.] at [inf]. Within 8 ulps. *)

val erfinv : (float, 's, 'd) t -> (float, 's, 'd) t
(** [erfinv p] is the [x] with [erf x = p], on \[[-1.], [1.]\]: [-inf] at
    [-1.], [-0.] at [-0.] and [inf] at [1.]. Within [4 + 8κ] ulps, where
    [κ = |p √π e{^x²} / (2x)|] is [1] at [0.] and grows as [p] nears [±1.]. *)

val ndtr : (float, 's, 'd) t -> (float, 's, 'd) t
(** [ndtr x] is the standard normal distribution function
    [Φ(x) = erfc (-x/√2) / 2]. Its lower tail keeps its relative precision
    down to the least subnormal, at [x] about [-38.5] at float64 and [-14.]
    at float32. It is [0.] at [-inf] and [1.] at [inf]. Within 16 ulps. *)

val log_ndtr : (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_ndtr x] is [log (ndtr x)], finite where [ndtr x] underflows while
    [x²/2] is finite: down to [x] about [-1.9e154] at float64 and [-2.6e19]
    at float32, and [-inf] below. It is [-inf] at [-inf] and [-0.] at [inf].
    Within 32 ulps. *)

val ndtri : (float, 's, 'd) t -> (float, 's, 'd) t
(** [ndtri p] is the [x] with [ndtr x = p], on \[[0.], [1.]\]: [-inf] at [0.]
    and [inf] at [1.]. A [p] above [1/2] is read as one minus the upper tail,
    so the result keeps the precision of [p] near either end:
    [ndtri 1e-300] is [-37.047]. Within [4 + 16κ] ulps, where
    [κ = |p / (x Φ'(x))|]. *)

val lgamma : (float, 's, 'd) t -> (float, 's, 'd) t
(** [lgamma x] is [log |Γ(x)|], on every float: [inf] at [0.], the negative
    integers and [inf], NaN at [-inf]. For [x > 0] within 16 ulps, absolute
    near its zeros at [1.] and [2.]; below [0.], within 16 ulps or
    [16ε (1 + lgamma (1 - x))] absolutely, the precision its reflection
    [log π - log |sin πx| - lgamma (1 - x)] keeps. *)

val digamma : (float, 's, 'd) t -> (float, 's, 'd) t
(** [digamma x] is [ψ(x) = Γ'(x) / Γ(x)], the mean of [log X] for [X] a
    gamma draw of shape [x], on every float: [-inf] at [+0.], [inf] at [-0.]
    and [inf], NaN at the negative integers and [-inf]. For [x > 0] within
    16 ulps, absolute near its root at [1.4616]; below [0.], within 16 ulps
    or [16ε (1 + |π cot πx|)] absolutely. *)

val lbeta : (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [lbeta a b] is [log B(a, b) = lgamma a + lgamma b - lgamma (a + b)] for
    [a, b ≥ 0], computed without that sum's cancellation: [lbeta 1e10 1.] is
    [-log 1e10]. It is [inf] where either is [0.] and the other finite,
    [-inf] where either is [inf] and the other positive, and NaN where one is
    [0.] and the other [inf]. Within 256 ulps, or [512ε] absolutely where it
    is below [1.] in magnitude. *)

val gammainc : (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammainc a x] is the regularised lower incomplete gamma function
    [P(a, x) = (1/Γ(a)) ∫₀ˣ t{^a-1} e{^-t} dt], the distribution function at
    [x] of the gamma distribution of shape [a] and unit rate, for [a, x ≥ 0].
    It is [0.] at [x = 0.] and where [a] is [inf] for finite [x], [1.] at
    [x = inf] and, its limit, at [a = 0.] for [x > 0]; NaN at [a = x = 0.]
    and [a = x = inf], where the limits disagree. For [0 < a ≤ 2{^20}]
    within [32 (1 + |log P|) + 4] ulps; above, its error grows with [a]. *)

val gammaincc : (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammaincc a x] is [Q(a, x) = 1 - P(a, x)] ({!gammainc}), the upper
    tail, to its own relative precision where [P] rounds to [1.]:
    [gammaincc 1. 40.] is [e{^-40}]. Its edges are the complements of
    {!gammainc}'s. Within [32 (1 + |log Q|) + 4] ulps for
    [0 < a ≤ 2{^20}]. *)

val log_gammainc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_gammainc a x] is [log P(a, x)] ({!gammainc}), finite for [a, x > 0]
    where [P] underflows: [log_gammainc 1000. 1.] is [-5913.13]. It is [-inf]
    at [x = 0.] and where [a] is [inf], [0.] at [x = inf], and [-0.] at
    [a = 0.] for [x > 0], where [P] reaches [1.] from below. For
    [0 < a ≤ 2{^20}] within 16 ulps, absolute near zeros. *)

val log_gammaincc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_gammaincc a x] is [log Q(a, x)] ({!gammaincc}), finite for [a > 0]
    wherever [x] is. It is [0.] at [x = 0.], [-0.] where [a] is [inf] for
    [x > 0], where [Q] reaches [1.] from below, and [-inf] at [x = inf] and
    at [a = 0.]. Within the bound of {!log_gammainc}. *)

val gammaincinv :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammaincinv a p] is the [x] with [gammainc a x = p], the gamma
    distribution's quantile, for [a ≥ 0] and [p] in \[[0.], [1.]\]. A [p]
    above [1/2] is read as one minus the upper tail, so the result keeps its
    precision near either end. It is [0.] at [p = 0.], [inf] at [p = 1.] and
    where [a] is [inf] for [p > 0], and at [a = 0.], where the mass is at
    [0.], [0.] for [p < 1]. For [0 < a ≤ 2{^20}] within
    [4 + κ (32 (1 + |log p|) + 4)] ulps, where [κ = |p / (x ∂ₓP(a, x))|]. *)

val gammainccinv :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammainccinv a q] is the [x] with [gammaincc a x = q], as
    {!gammaincinv} from the upper tail: [inf] at [q = 0.], [0.] at [q = 1.],
    and [0.] at [a = 0.] for [q > 0]. Within the bound of {!gammaincinv},
    in [q]. *)

val betainc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [betainc a b x] is the regularised incomplete beta function
    [I_x(a, b) = (1/B(a, b)) ∫₀ˣ t{^a-1} (1 - t){^b-1} dt], the distribution
    function at [x] of the beta distribution of shapes [a] and [b], for
    [a, b ≥ 0] and [x] in \[[0.], [1.]\]. It is [0.] at [x = 0.] and [1.] at
    [x = 1.]; its limits, [1.] at [a = 0.] and where [b] is [inf], and [0.]
    at [b = 0.] and where [a] is [inf]; NaN where two of these disagree. For
    [0 < a, b ≤ 2{^20}] within [64 (1 + |log I|) + 4] ulps; above, its
    error grows with [a] and [b]. *)

val betaincc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [betaincc a b x] is [1 - I_x(a, b)] ({!betainc}), the upper tail, to its
    own relative precision where [I] rounds to [1.]: [betaincc 1. 60. 0.5] is
    [2{^-60}]. For [x ≥ 1/2] it is [betainc b a (1 - x)] bit for bit. Its
    edges are the complements of {!betainc}'s. Within
    [64 (1 + |log (1 - I)|) + 4] ulps for [0 < a, b ≤ 2{^20}]. *)

val log_betainc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [log_betainc a b x] is [log I_x(a, b)] ({!betainc}), finite for finite
    [a, b > 0] and [x] in (0, 1\], where {!betainc} underflows included. It
    is [-inf] where {!betainc} is [0.], [0.] at [x = 1.], and [-0.] at
    [a = 0.] and where [b] is [inf] for [x] in (0, 1), where the limit is
    reached from below. For [0 < a, b ≤ 2{^20}] within 32 ulps, absolute
    near zeros. *)

val log_betaincc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [log_betaincc a b x] is [log (1 - I_x(a, b))] ({!betaincc}), finite for
    finite [a, b > 0] and [x] in \[0, 1). It is [-inf] where {!betaincc} is
    [0.], [0.] at [x = 0.], and [-0.] at [b = 0.] and where [a] is [inf] for
    [x] in (0, 1). Within the bound of {!log_betainc}. *)

val i0e : (float, 's, 'd) t -> (float, 's, 'd) t
(** [i0e x] is [e{^-|x|} I₀(x)], [I₀] the modified Bessel function of the
    first kind of order 0: even, [1.] at [0.] and [0.] at [±inf]. The von
    Mises normaliser [2π I₀(k)] has the logarithm [k + log (2π i0e k)], finite
    where [I₀(k)] overflows, past [k] about [714] at float64 and [92] at
    float32. Within 8 ulps. *)

val i1e : (float, 's, 'd) t -> (float, 's, 'd) t
(** [i1e x] is [e{^-|x|} I₁(x)], [I₁] the modified Bessel function of the
    first kind of order 1: odd, so [-0.] at [-0.] and at [-inf], and [0.] at
    [inf]. [i1e k / i0e k] is the mean of [cos θ] under the von Mises
    distribution of concentration [k]. Within 8 ulps. *)

(** {2:rounding Rounding}

    Floats and integers; the identity on integers. *)

val floor : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [floor x] rounds toward negative infinity. *)

val ceil : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [ceil x] rounds toward positive infinity. *)

val round : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [round x] rounds to the nearest integer, halves away from zero. *)

val trunc : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [trunc x] rounds toward zero. *)

(** {2:complex Complex numbers}

    A complex value's parts are floats of half its width: float32 for
    complex64, float64 for complex128. {!real}, {!imag}, {!magnitude} and
    {!angle} take the dtype of their result first. *)

val real : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [real dt z] is [z]'s real parts, stored in [dt] by the conversion rule
    ({!cast}): exact where [dt] is as wide as the parts. *)

val imag : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [imag dt z] is [z]'s imaginary parts, as {!real}. *)

val magnitude : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [magnitude dt z] is [z]'s modulus, {!hypot} of its parts computed in the
    wider of [dt] and the parts' format, then stored in [dt]. *)

val angle : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [angle dt z] is [z]'s argument in radians, {!atan2} of its imaginary and
    real parts, computed as {!magnitude} is. The negative real axis is a branch
    cut whose side the sign of a zero imaginary part picks: [-1 + 0i] has
    angle [π], [-1 - 0i] has [-π]; [0] has angle [0]. *)

val complex :
  (Complex.t, 'c) dtype ->
  re:(float, 'a, 'd) t ->
  im:(float, 'a, 'd) t ->
  (Complex.t, 'c, 'd) t
(** [complex dt ~re ~im] is the complex value of parts [re] and [im],
    broadcast together, each stored in [dt]'s part format by the conversion
    rule: infinities, NaN and signed zeros stay, and one part never affects the
    other. [complex dt ~re:(real f z) ~im:(imag f z)] is [z] for [f] the
    parts' format.

    Raises [Invalid_argument] if the shapes do not broadcast. *)

val conjugate : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [conjugate x] is [x] with each imaginary part negated, [0.] to [-0.]
    included; [x] itself for a dtype that is not complex. *)

(** {2:compare Comparisons and bits}

    Every dtype has one order: [false < true]; integers by value, unsigned
    dtypes unsigned; floats by value, [-0.] below [+0.]; complex numbers by real
    part, then imaginary part. A comparison is [true] where it holds; it
    compares floats by value, so [-0.] equals [+0.], and it is [false] where an
    element is a NaN, or a complex number with a NaN part, but for {!not_equal},
    which is [true] there. Comparisons take every dtype. *)

val equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [equal a b] is [a = b]. *)

val not_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [not_equal a b] is [a <> b]. *)

val less : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [less a b] is [a < b]. *)

val less_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [less_equal a b] is [a <= b]. *)

val greater : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [greater a b] is [less b a]. *)

val greater_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [greater_equal a b] is [less_equal b a]. *)

val array_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [array_equal a b] is the 0-d [true] iff [a] and [b] have one shape and
    {!equal} holds at every index: [false] where an element is a NaN, and
    [true] for two values without elements. Values of two shapes give [false]
    and do not broadcast. *)

val bitwise_and : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_and a b] is the bitwise and of integers, the logical and of
    booleans. Integers and booleans. *)

val bitwise_or : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_or a b] is the bitwise or, as {!bitwise_and}. *)

val bitwise_xor : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_xor a b] is the bitwise exclusive or, as {!bitwise_and}. *)

val bitwise_not : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_not x] is every bit of [x] flipped, the logical not of booleans.
    Integers and booleans. *)

val lshift : ('v, 's, 'd) t -> int -> ('v, 's, 'd) t
(** [lshift x n] is [x] shifted [n] bits toward its high end: [x 2{^n}] modulo
    [2{^w}] for a dtype of [w] bits, [0] once [n] reaches [w]. Integers.

    Raises [Invalid_argument] if [n < 0]. *)

val rshift : ('v, 's, 'd) t -> int -> ('v, 's, 'd) t
(** [rshift x n] is [x] shifted [n] bits toward its low end: [x / 2{^n}]
    rounded toward negative infinity, so a signed [x] keeps its sign; [0], or
    [-1] for a negative [x], once [n] reaches the width. Integers.

    Raises [Invalid_argument] if [n < 0]. *)

val logical_and : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_and a b] is one where [a] and [b] are both not zero, zero
    elsewhere, in their dtype. A NaN is not zero. Every dtype. *)

val logical_or : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_or a b] is one where [a] or [b] is not zero, as {!logical_and}. *)

val logical_xor : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_xor a b] is one where exactly one of [a] and [b] is not zero, as
    {!logical_and}. *)

val logical_not : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_not x] is one where [x] is zero, zero elsewhere, as
    {!logical_and}. *)

val isnan : ('v, 's, 'd) t -> 'd bool_t
(** [isnan x] is [true] where [x] is a NaN, or a complex number with a NaN
    part. Every dtype: [false] for the integers and booleans. *)

val isinf : ('v, 's, 'd) t -> 'd bool_t
(** [isinf x] is [true] where [x] is an infinity, or a complex number with an
    infinite part. Every dtype: [false] for the integers, the booleans and the
    float formats without infinities. *)

val isfinite : ('v, 's, 'd) t -> 'd bool_t
(** [isfinite x] is [true] where [x] is neither an infinity nor a NaN, and for
    a complex number where both parts are. Every dtype: [true] for the integers
    and booleans. *)

val clamp : ?min:'v -> ?max:'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [clamp ~min ~max x] is [minimum (maximum x min) max] with each bound a
    constant, a bound left out not applied: [max] where [min > max], and a NaN
    stays a NaN. [x] itself without either. Every dtype.

    Raises [Invalid_argument] if a bound is an [int] outside [x]'s dtype's
    range. *)

val where : 'd bool_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [where c x y] is [x]'s element where [c] is [true] and [y]'s elsewhere.

    Raises [Invalid_argument] if the shapes do not broadcast. *)

(** {2:reductions Reductions and scans}

    A reduction folds [x]'s elements along [axes], every axis by default, a
    negative one counted from the end, and drops them, or keeps each with extent
    [1] where [keepdims]. On one device it gives the same bits on every run
    of the same computation, for the same values, shapes, dtypes, placement
    and devices ({!Nx_kernel}); on nx.cpu ({!Nx_cpu.section-sums}) its
    order of association depends on the shape alone. Over axes cut across
    devices, each device reduces the part its window holds, and only
    partial results move instead of [x]'s elements: one per output and
    window holding terms, stacked in C order of the windows' first indices
    and reduced over the stack by the kernels; for {!argmax} and {!argmin},
    each device's extreme and its position. Integer and boolean reductions,
    extremes and their positions keep the bits they have on one device. A
    float or complex {!sum}, {!prod}, {!mean}, {!var}, {!std}, {!logsumexp}
    or {!logmeanexp} over cut axes has the kernels' association over each
    window's terms and over the stacked partials. It may differ from the
    result on one device within the bounds below, and is the same on every
    run for a given placement and set of devices. Barring overflow and
    underflow, a float sum of [n] terms stays within [γ(n - 1) Σ|x|] of the
    exact sum and a product within a relative [γ(n - 1)], as on one device
    ({!Nx_kernel.section-bounds}); a complex product within a normwise
    relative [γ(2n - 2)]. A {!mean} is such a sum divided by [n]. {!var} and
    {!std} are within the bound {!var} states. A {!logsumexp} is within
    [ε + ε{_t}] of the exact one, [ε] the largest bound the kernels' library
    states for the logsumexp of a window's terms and [ε{_t}] its bound for
    one of [t] terms, [t] the windows holding terms, and {!logmeanexp} is
    that less [log n].

    A scan along an axis cut across devices scans each device's window
    alone. The windows' totals, one row each, move to every device that
    holds a window of the same rows, and scan once in the windows' order;
    each window but the first combines each of its results with the
    totals' scan of the windows before it. Over integers and booleans, and
    for {!cummax} and {!cummin}, it keeps the bits it has on one device;
    {!cumsum} and {!cumprod} of floats and complex numbers keep the bounds
    of a reduction of each result's terms. {!ewma} and {!logcumsumexp} read
    their axis whole on each device, so a value cut along it raises
    ({!section-routing}).

    A float result that a NaN reaches is a NaN; which NaN, its sign and its
    payload are unspecified. A complex number with a NaN part is a NaN to
    {!prod}, {!max}, {!min}, their positions and scans; {!sum} and
    {!cumsum} add the parts apart. The float
    formats narrower than 32 bits accumulate at [float32] and round once.

    Each raises [Invalid_argument] naming itself for an axis outside [x]'s rank
    or repeated, or a dtype it does not take. The host computes every dtype
    stated. Where a device's kernels do not compute one, as a GPU's may not
    compute complex numbers, the function raises naming itself. *)

val sum : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sum ~axes x] adds the elements from [+0]: a sum of [-0.] terms is [+0.],
    and a sum of none is [+0.]. Integers wrap. Every dtype but booleans. *)

val prod : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [prod ~axes x] multiplies the elements from the first, as {!sum}, and a
    product of none is [1]. Floats, integers and complex numbers. *)

val max : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [max ~axes x] is the greatest element, as {!maximum} orders them: [-0.]
    below [+0.], any on booleans. Every dtype.

    Raises [Invalid_argument] where an axis it reduces is empty and the result
    has an element, which has then nothing to choose from. *)

val min : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [min ~axes x] is the least element, as {!max}; all on booleans. *)

val mean : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mean ~axes x] is the {!sum} divided by the number of elements. Over floats
    and complex numbers it rounds once: NaN for none. Over integers it is the
    exact mean truncated toward zero, as {!div} rounds: it lies between the
    least and greatest element, so it never wraps. Every dtype but booleans.

    Raises [Invalid_argument] over integers where an axis it reduces is empty
    and the result has an element, since the mean of nothing has no integer
    value, or where the axes hold more elements than an integer mean computes
    exactly: [2{^31}] or more at 64 bits, [2{^(64 - b)}] or more at [b]
    bits. *)

val all : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd bool_t
(** [all ~axes x] is [true] where every element is non-zero, as {!cast} into
    {!bool} reads it: a NaN is non-zero, and a complex number is where a part
    is. Of no element it is [true]. Every dtype. *)

val any : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd bool_t
(** [any ~axes x] is [true] where some element is non-zero, as {!all}. Of no
    element it is [false]. *)

val count : ?axes:int list -> ?keepdims:bool -> (bool, 's, 'd) t -> 'd int64_t
(** [count ~axes m] is the number of [true] elements of [m], a {!bool} or a
    {!bit} value: [0] of none. *)

val argmax : ?axis:int -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd int64_t
(** [argmax ~axis x] is the position along [axis] of the first element
    greatest as {!max} orders them: the first NaN where there is one, and
    [1] for [[-0.; 0.]]. Without [axis], the position in C order among all of
    [x]'s elements, read as {!reshape} to one axis reads them, of rank [0],
    or every axis of extent [1] where [keepdims]. Every dtype.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank,
    or where the result has an element with no element to choose from: [x]
    is empty without [axis], or [axis] is empty and the result is not. *)

val argmin : ?axis:int -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd int64_t
(** [argmin ~axis x] is the position of the first element least as {!min}
    orders them: the first NaN where there is one, and [1] for [[0.; -0.]].
    As {!argmax} otherwise. *)

val var :
  ?axes:int list ->
  ?keepdims:bool ->
  ?ddof:int ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [var ~axes ~ddof x] is the variance [Σ (x - m)² / (n - ddof)] of the [n]
    elements, [m] their mean; [ddof] defaults to [0], the population
    variance, and [1] gives the sample variance. NaN of no element.

    On one device, the population variance is the kernels' moments, within
    the bound their library states. Over axes cut across devices, it is the
    mean, then the deviations [d] from it and their squares, each summed
    as {!sum} sums, and [(Σ d² - Σ d (Σ d / n)) / n], [+0] where that is
    negative and [+∞] where [Σ d²] overflows: on nx.cpu's kernels, within
    nx.cpu's bound for its moments ({!Nx_cpu.section-states}) with
    [L + D + L{_t} + D{_t}] in place of its [L + D], [L{_t}] and [D{_t}]
    nx.cpu's [L] and [D] for [t] terms, [t] the windows holding terms.
    A [ddof] scales it by a relative [2u] more, then one rounding to the
    dtype.

    Raises [Invalid_argument] naming itself for a negative [ddof], or one of
    at least [n] where [n > 0]. *)

val std :
  ?axes:int list ->
  ?keepdims:bool ->
  ?ddof:int ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [std ~axes ~ddof x] is the standard deviation, the square root of
    {!var}, taken before the one rounding to the dtype. As {!var} otherwise. *)

val logsumexp :
  ?axes:int list -> ?keepdims:bool -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [logsumexp ~axes x] is [log Σ exp x], computed as
    [m + log Σ exp (x - m)] for the elements' greatest [m], so no term
    overflows: [-∞] of no element or of elements all [-∞], [+∞] where an
    element is [+∞] and none is NaN. On one device, its error is the bound
    the kernels' library states; over axes cut across devices, the bound of
    {!section-reductions}. *)

val logmeanexp :
  ?axes:int list -> ?keepdims:bool -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [logmeanexp ~axes x] is [log (Σ exp x / n)]: {!logsumexp} less [log n],
    taken before the one rounding to the dtype. NaN of no element. *)

val cumsum : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cumsum ~axis x] is, at each index, the sum of [x]'s elements along [axis]
    up to that index, inclusive; without [axis], of its elements in C order up
    to that one, in [x]'s shape. Every dtype but booleans.

    On one device, or along an axis no device cuts, the result at an index
    is a function of the elements up to it alone: the cumsum of a prefix of
    the axis is the prefix of the cumsum, bit for bit.
    Its last element need not equal {!sum}'s of the same elements.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank or a
    dtype it does not take. *)

val cumprod : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cumprod ~axis x] is the running product, as {!cumsum}. Floats, integers
    and complex numbers, as {!prod}. *)

val cummax : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cummax ~axis x] is the running {!max}, as {!cumsum}. Every dtype. *)

val cummin : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cummin ~axis x] is the running {!min}, as {!cummax}. *)

val logcumsumexp : ?axis:int -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [logcumsumexp ~axis x] is, at each index [i] along [axis],
    [log Σ_{k ≤ i} exp x_k]; without [axis], over its elements in C order up
    to that one, in [x]'s shape. It carries the elements' running maximum
    [m] and the sum of [exp (x_k - m)], at most [i + 1], so it stays finite
    where [log (cumsum (exp x))] overflows once an [exp x_k] does. Floats.

    From the first NaN element on, every result is a NaN. With no NaN, a
    result is [+∞] from a [+∞] element on, and [-∞] while every element is
    [-∞].

    It is the kernels' [Logsumexp] scan ({!Nx_kernel.section-scans}),
    within the bound their library states. The result at an index is a
    function of the elements up to it alone: the logcumsumexp of a prefix
    of the axis is the prefix of the logcumsumexp, bit for bit. On nx.cpu
    ({!Nx_cpu.section-state_scans}), where the elements up to [i] have a
    finite maximum, the result before its rounding to the dtype is within
    [(7 D + 6 ln (i + 1)) u + u|r|] of the exact [r], to first order, with
    [D = max 0 (2⌈log₂ (i + 1)⌉ - 1)] and [u] the unit roundoff of
    [float32] for the floats narrower than it, else of the dtype.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s
    rank. *)

val ewma : ?axis:int -> alpha:float -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [ewma ~alpha x] is the exponentially weighted moving average of [x]
    along [axis]: [y₀ = x₀], then [yᵢ = (1 - alpha) yᵢ₋₁ + alpha xᵢ];
    without [axis], of its elements in C order, in [x]'s shape.

    The average of a prefix of [x] is the prefix of its average, bit for
    bit. Each device reads [axis] whole. Element [i] passes through at
    most [max(0, 2⌈log₂(i+1)⌉ - 1)] compositions of the recurrence's affine
    maps [y ↦ a y + b], in [O(n)] work for [n] elements. A NaN reaches every
    later element, and an infinity as the recurrence carries it; a finite
    term whose weight underflows to [0] drops out.

    Raises [Invalid_argument] naming itself unless [0 < alpha <= 1], or for
    an axis outside [x]'s rank. *)

(** {2:activations Activations}

    Floats only, each computed in [float32] for the formats narrower than 32
    bits and rounded once. *)

val sigmoid : (float, 's, 'd) t -> (float, 's, 'd) t
(** [sigmoid x] is [1 / (1 + exp (-x))], in [[0, 1]]: [+0.] and [1.] only
    where the exact value rounds there, a subnormal where it is one. Within
    3 ulps at float32 and float64. *)

val softplus : (float, 's, 'd) t -> (float, 's, 'd) t
(** [softplus x] is [log (1 + exp x)], computed as
    [max x 0 + log1p (exp (-|x|))]: [exp x] to the format's precision far
    below [0.], [x] far above it, never overflowing. Within 3
    ulps at float32 and float64. *)

val softmax :
  ?axes:int list -> ?scale:float -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [softmax ~axes ~scale x] is [exp (scale x) / Σ exp (scale x)] along
    [axes], [[-1]] by default; [scale] defaults to [1.]. The exponents are
    shifted by the greatest [scale x] of each lane, so none overflows. A
    lane holding a NaN, or whose greatest [scale x] is infinite, is NaN.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank
    or repeated. *)

val log_softmax :
  ?axes:int list -> ?scale:float -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_softmax ~axes ~scale x] is the logarithm of {!softmax}, computed as
    [(scale x - m) - log Σ exp (scale x - m)] for the lane's greatest [m]:
    an entry far below [m] keeps its digits where [log (softmax x)] would be
    [-∞]. As {!softmax} otherwise. *)

val rms_norm :
  ?axes:int list -> ?eps:float -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [rms_norm ~axes ~eps x] is [x / sqrt (mean x² + eps)], the mean along
    [axes], [[-1]] by default; [eps] defaults to [1e-6]. It computes
    [x · (1 / sqrt (Σ x² / n + eps))] for the [n] elements of each lane:
    the squares summed as {!sum} sums them, each later step rounded once.
    At [float32] and [float64] these are the bits of
    [mul x (rsqrt (add_s (mean ~axes ~keepdims:true (square x)) eps))].

    Barring overflow and underflow, and for [k u < 1/4], it is within
    [k u / (1 - 4 k u)] of the exact value, relatively, where
    [k = (n + 8) / 2] and [u] is the unit roundoff of the format it computes
    in, before a narrow float's rounding: the squares and their sum take [n]
    roundings, the mean and [eps] two more, which the root halves, and the
    root, the reciprocal and the product one each.

    A lane whose sum of squares overflows is [x · 0]: zeros, NaN at an
    infinity. With [~eps:0.], a lane whose squares sum to [0] is [x · ∞]:
    [[nan; nan]] of [[0.; 0.]], and [[inf]] of a [float32] [[1e-30]], whose
    square underflows. A lane holding a NaN is NaN.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank
    or repeated, or if [eps] is negative or NaN. *)

(** {2:conversion Conversion} *)

val cast : ('w, 'r) dtype -> ('v, 's, 'd) t -> ('w, 'r, 'd) t
(** [cast dt x] is [x]'s elements stored in [dt]; [x] itself where [dt] is [x]'s
    dtype.

    A float rounds once to the nearest value of a float format, ties to even.
    Past the format's largest finite value it is an infinity in [float64],
    [float32], [float16], [bfloat16] and [float8_e5m2], and saturates in the
    formats with none, [float8_e4m3fn] and [float4_e2m1fn]. NaN stays NaN, but in [float4_e2m1fn], where it is [+0.]. Into an
    integer, a float truncates toward zero and saturates to the range, NaN
    giving [0]; an integer is kept modulo the width; a boolean is [0] or [1].
    Into a boolean, any element is [true] where it is not zero, NaN included. An
    integer into a float rounds once from its exact value. Into a complex dtype
    these rules give the real part, and the imaginary part is [0.]; a complex
    number stores part by part into a complex dtype, as [true] into a boolean if
    a part is not zero, and by its real part into any other dtype. *)

val bitcast : ('w, 'r) dtype -> ('v, 's, 'd) t -> ('w, 'r, 'd) t
(** [bitcast dt x] is [x]'s bits read as elements of [dt], NaN payloads
    included: [x] itself where [dt] is [x]'s dtype. At one width the shape is
    [x]'s. A [dt] [k] times narrower appends an axis of [k], its elements the
    pieces of [x]'s element, in the host's byte order; one [k] times wider reads
    [x]'s last axis, of [k] elements, as one element, and drops it.

    Raises [Invalid_argument] where [dt] is wider and [x]'s last axis does not
    have [k] elements, and where [dt] is narrower and [x] has the greatest rank
    already. *)

(** {1:contraction Contraction}

    A contraction multiplies two operands along their paired axes and sums the
    products over the summed ones. On one device it gives the same bits on
    every run of the same computation, as a reduction does
    ({!section-reductions}); on nx.cpu its order of association depends on the
    shapes alone. A NaN result's sign and payload are unspecified.

    Over operands cut across devices, each device computes its window of
    the result from its windows of the operands. Where both operands' paired
    summed axes are cut alike, the devices that hold one window of the
    result compute it together: each contracts the products its windows
    hold, and only these partial sums move, in the accumulator, stacked in
    C order of the windows' first indices and summed over the stack by the
    kernels, then [init] added and the sum rounded once. An operand whole on
    every device is read at the other's cut; summed axes cut otherwise raise
    ({!section-routing}). Integer contractions keep the bits they have on
    one device. A float or complex one may differ from it within the bound
    {!contract} states, and is the same on every run for a given placement
    and set of devices: float32 [[1e7; -1e7; 0.1; 0]] by ones gives [0.] on
    one nx.cpu device and [0.1] cut two ways along its summed axis. *)

val contract :
  ?sizes:(string * int) list ->
  ?acc:('w, 'q) dtype ->
  ?init:('v, 's, 'd) t ->
  ('v, 's) dtype ->
  Pattern.t ->
  ('a, 'b, 'd) t ->
  ('c, 'e, 'd) t ->
  ('v, 's, 'd) t
(** [contract dt p a b] is [round_dt (init + Σ a · b)] over [p]'s summed names,
    summed in [acc], for each index of [p]'s result. [sizes] gives the extents
    of names inside groups that neither operand's shape gives alone.

    [acc] defaults to [float32] for float operands of 32 bits or fewer,
    [float64], [complex64] or [complex128] at the operands' width otherwise, and
    [dt] for integers, which wrap. An integer operand with a float one has no
    default: [acc] is required. The sum starts from [init], or [+0] without it,
    and associates in an order each kernel library states, never a function
    of the order a pattern lists its names. Before rounding it is within
    [γ(K + 1, 2u) (|init| + Σ|a||b|)] of the exact sum of its [K] products,
    wherever its operands lie, where [γ(n, v) = n v / (1 - n v)] and [u] is
    [acc]'s unit roundoff.

    A donated [init] receives the result in its memory where {!donate} lets
    the consumer write there and the result's names are the batch names, then
    [a]'s others, then [b]'s, each in its operand's order, as in
    ["n i, n o -> i o | n"]; otherwise the result is fresh.

    Raises [Invalid_argument] if [dt] is a boolean; [p] has other than two
    operands; an operand's rank is not the number of axes [p] gives it, or
    fewer with [...]; [...] covers different numbers of axes in the two
    operands; paired extents
    differ; a group's extents do not multiply to its axis, or it leaves
    two extents unknown; a unit axis's extent is not [1]; a name in [sizes] is
    not in [p] or is there twice, or its size is negative; [acc] is omitted for
    an integer and a float operand, or is a boolean, a float narrower than
    [float32] or than a float operand, or of another kind than [dt]; or
    [init]'s shape is not the result's. *)

val einsum :
  ?sizes:(string * int) list ->
  ?acc:('w, 'q) dtype ->
  Pattern.t ->
  ('v, 's, 'd) t list ->
  ('v, 's, 'd) t
(** [einsum ~sizes ~acc p [x1; …; xn]] is [round (Σ x1 ⋯ xn)] over [p]'s
    summed names, for each index of [p]'s result, at the operands' dtype. It
    is a tree of {!contract}s of two operands, each summing in [acc] and
    leaving its value in [acc], so that the result rounds once. [acc] defaults
    as {!contract}'s for two operands of that dtype, and [sizes] is as
    {!contract}'s.

    Of the trees whose values have at most {!Nx_array.Layout.max_rank} axes,
    the tree is the one whose contractions do the fewest multiply-adds in
    all, a contraction doing the product of the extents of the names its two
    parts hold. A part's value holds the names that the result or an operand
    outside it needs, and each of its elements costs at least one
    multiply-add, so the work bounds the memory too. Of trees doing as many,
    each contraction's left part holds the earliest of its operands; of such
    parts, the larger, then the one holding the earliest operand the other
    lacks: equal work gives [((x1 x2) x3) x4]. The tree is a function of [p]
    and the shapes alone, and so, on nx.cpu, is the result's order of
    association. Choosing it takes time in [3^n], so [p] has at most 11
    operands.

    {[
    (* For square matrices, a (b v): j·k multiply-adds, then i·j, where
       (a b) v does i·j·k, then i·k. *)
    let mmv = Nx.Pattern.v "i j, j k, k -> i | j k"
    let y = Nx.einsum mmv [ a; b; v ]
    ]}

    Before rounding the result is within [γ(N, 2u) Σ |x1| ⋯ |xn|] of the exact
    sum, where [N = Σ (K + 1)] over the tree's [n - 1] contractions, each
    summing [K] products per element, and [γ] and [u] are as {!contract}
    states.

    Raises [Invalid_argument] if [p] has one operand, more than 11, or other
    than [n]; the dtype is a boolean; an operand's rank is not the number of
    axes [p] gives it, or fewer with [...]; [...] covers different numbers of
    axes in two operands; a name's extents differ between operands; every tree
    holds a value of more than {!Nx_array.Layout.max_rank} axes; a group's
    extents do not multiply to its axis, or it leaves two extents unknown; a
    unit axis's extent is not [1]; a name in [sizes] is not in [p] or is there
    twice, or its size is negative; or [acc] is a boolean, a float narrower
    than [float32] or than the operands, or of another kind than theirs. *)

val matmul : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [matmul a b] is the matrix product over the last two axes, leading axes
    broadcast, at [a]'s dtype and accumulated as {!contract} does by default. A
    1-d [a] is a row and a 1-d [b] a column, and their unit axis is dropped from
    the result: two 1-d operands give their 0-d inner product.

    Raises [Invalid_argument] if an operand is 0-d, the inner extents differ,
    the leading axes do not broadcast, or the dtype is a boolean. *)

val dot : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [dot a b] is the sum of the products of [a]'s last axis with [b]'s only
    axis, for a 1-d [b], or its second to last otherwise. The result's axes are [a]'s
    others, then [b]'s: two vectors give their 0-d inner product, two matrices
    their product. It does not broadcast, and accumulates as {!matmul}.

    Raises [Invalid_argument] if an operand is 0-d, the summed extents differ,
    or the dtype is a boolean. *)

val vdot : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [vdot a b] is the 0-d sum of the products of [conjugate a] and [b], both
    flattened, accumulated as {!matmul}.

    Raises [Invalid_argument] if [a] and [b] have different numbers of
    elements, or the dtype is a boolean. *)

val cross : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cross ~axis a b] is the cross product of the 3-vectors of [a] and [b]
    along [axis] (default [-1]) of their broadcast shape: component [i] is
    [a(i+1) b(i+2) - a(i+2) b(i+1)], indices modulo 3. A float format of at
    most 32 bits gives that difference within one ulp, finite wherever it fits
    the dtype; float64 within 2 ulps where no product overflows or
    underflows, and IEEE 754's difference of an infinite product. A zero
    difference has IEEE 754's sign. Integers wrap.

    Raises [Invalid_argument] if [axis] is not an axis of the broadcast shape,
    if [a] or [b] does not have 3 elements along it, or for booleans. *)

val conv :
  ?stride:int array ->
  ?padding:(int * int) array ->
  ?dilation:int array ->
  ?groups:int ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [conv x w] slides the kernel [w], [[o; k…; c / groups]], over [x],
    [[n; s…; c]], channels last, for [K] spatial axes [s…] and kernel extents
    [k…]: the result, [[n; t…; o]], is at [(b, p…, j)] the sum over [i] and
    the taps [q…] of [x[b, p · stride + q · dilation - lo…, i]] times
    [w[j, q…, i]], an element outside [x] reading zero. The kernel is not
    flipped. [x] is padded along each spatial axis by [padding]'s [(lo, hi)]
    (default none), a negative one cutting; [stride] and [dilation] default
    to ones; and [t = (s + lo + hi - e) / stride + 1], none where the padded
    axis is shorter than [e], the dilated kernel's extent
    [dilation (k - 1) + 1]. A kernel extent of [0] sums no term: [e] is [0]
    and each element is [+0]. Under [groups]
    (default [1]) the channels split into [groups] runs, output channels
    [g · o / groups] to [(g + 1) · o / groups - 1] reading input run [g]
    alone.

    Each output sums its [K] products, one per tap and channel of its group,
    from [+0], in the accumulator {!contract} takes by default, and rounds
    once: the float formats narrower than 32 bits sum at [float32]. Integers
    wrap. Before rounding it is within [γ(K + 1, 2u) Σ|x||w|] of the exact
    sum, [γ] and [u] as {!contract} states. On one device it gives the same
    bits on every run of the same computation ({!section-reductions}),
    computing as follows:
    - With other than one input channel a group, [c' = c / groups], it is
      one {!contract} of [x]'s windows with [w], batched over the groups:
      an output's products are numbered [q · c' + i], [q] the tap's place
      in C order of the taps' indices and [i] the channel among the [c']
      its group reads, and summed as {!contract} sums them. [x] padded is a
      copy of [x], or [x] itself where nothing pads it, and its windows a
      view of it.
    - With one, as a depthwise convolution has, it is one reduction over
      [x]'s padded windows, never stored, of the products numbered
      [i · Q + q], [Q] the taps, summed as {!sum} does.

    Raises [Invalid_argument] if [x] has fewer than three axes; [w] has
    another rank; [stride], [padding] or [dilation] do not have [K] entries;
    a stride or a dilation is below [1]; [groups] is below [1] or does not
    divide [o]; [x]'s channels are not [groups] times [w]'s last extent; or
    the dtype is a boolean. *)

val conv_transpose :
  ?stride:int array ->
  ?padding:(int * int) array ->
  ?dilation:int array ->
  ?groups:int ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [conv_transpose x w] is {!conv}'s transpose with respect to its input:
    for the [z] of the least spatial extents whose [conv z w], under the same
    arguments, has [x]'s shape [[n; t…; o]], it is the [[n; s…; c]] value
    with [sum (mul (conv z w) x) = sum (mul z (conv_transpose x w))] for
    every such [z]: [s = (t - 1) stride + e - lo - hi], [e] as {!conv}
    states, or [0] where that is negative or [t = 0]. It is {!conv} at stride
    one of [x] with [stride - 1] zeros between its elements, by [w] reversed
    along its spatial axes and with its channels exchanged, and sums as
    {!conv} states with [o / groups] channels a group: one {!contract} over
    [x] spread and padded, a copy where zeros spread or pad it, or, with
    one channel a group, one reduction that never stores it.

    Raises [Invalid_argument] as {!conv}, reading [o] from [x]'s last axis, or
    if no input has a [conv] of [x]'s shape. *)

(** {2:products Other products}

    Products of tensors, vectors and chains of matrices, each a {!contract},
    a chain of them or an elementwise {!mul}. Each takes every dtype but
    booleans, for which it raises [Invalid_argument]. *)

val tensordot :
  ?axes:int list * int list ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [tensordot ~axes:(la, lb) a b] is the sum of the products of [a] and [b]
    over the axes [la] of [a], paired in order with the axes [lb] of [b]
    (default [a]'s last two with [b]'s first two). The result's axes are [a]'s others,
    then [b]'s, each in order: [~axes:([], [])] gives the outer product. It
    accumulates as {!matmul}.

    Raises [Invalid_argument] if [la] and [lb] differ in length, an axis is
    not one of its operand's or repeats, or paired extents differ. *)

val inner : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [inner a b] is the sum of the products of [a]'s last axis with [b]'s
    last axis; the result's axes are [a]'s others, then [b]'s. A 0-d operand gives
    {!mul}[ a b]. It accumulates as {!matmul}.

    Raises [Invalid_argument] if the last extents differ. *)

val outer : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [outer a b] is of shape [[|numel a; numel b|]], its element [(i, j)] the
    product of [a]'s [i]th element and [b]'s [j]th, in C order. *)

val vecdot : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [vecdot ~axis a b] is the sum of the products of [conjugate a] and [b]
    along each operand's axis [axis] (default [-1]); their other axes broadcast to the
    result's shape. It accumulates as {!matmul}.

    Raises [Invalid_argument] if [axis] is not an axis of both, their extents
    along it differ, or their other axes do not broadcast. *)

val kron : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [kron a b] is the Kronecker product. With the operand of fewer axes given
    leading axes of extent [1], each axis has [a]'s extent times [b]'s, and
    the element at [i sb + j] along each axis, [sb] [b]'s extent there, is
    [a]'s element at the [i]s times [b]'s at the [j]s.

    Raises [Invalid_argument] if the axes of extent other than [1] of [a] and
    of [b] number more than {!Nx_array.Layout.max_rank} together. *)

val multi_dot : ('v, 's, 'd) t list -> ('v, 's, 'd) t
(** [multi_dot [x1; …; xn]] is the matrix product [x1 x2 ⋯ xn] of [n >= 2]
    operands. The middle ones are matrices; a 1-d [x1] is a row and a 1-d
    [xn] a column, and the result drops their unit axis. It multiplies in
    the order of fewest multiply-adds, an [m × k] by a [k × p] doing
    [m k p]; of orders doing as many, the one that splits the chain earliest,
    within each part too, so the order is a function of the shapes alone.
    Each product sums in {!contract}'s default accumulator and leaves its
    value there, so the result rounds once, within the bound {!einsum}
    states for its tree.

    Raises [Invalid_argument] if [n < 2], an operand has another rank, or
    adjacent inner extents differ. *)

val matrix_power : ('v, 's, 'd) t -> int -> ('v, 's, 'd) t
(** [matrix_power a n] is each square matrix of the batch to the power [n]:
    the identity for [n = 0], and [inv a] to the power [-n] for a negative
    [n]. It squares repeatedly, [⌊log₂ |n|⌋] squarings and one product per
    further set bit of [|n|], each a {!matmul} rounded to the dtype.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not
    square, and, for a negative [n], for a dtype {!inv} does not take. *)

(** {2:attention Attention}

    Attention weighs values by how well their keys match a query: for each
    query row, a softmax over the row's scores against the keys, times the
    values. Masks and score rewrites are plain Nx code over the coordinates
    of one query-key pair. *)

(** Keys and values, by key head [hk] and key [tk]. *)
type ('v, 's, 'd) kv =
  | Dense of { k : ('v, 's, 'd) t; v : ('v, 's, 'd) t }
      (** [k] [[lead…; hk; tk; dk]], [v] [[lead…; hk; tk; dv]]. *)
  | Paged of { k : ('v, 's, 'd) t; v : ('v, 's, 'd) t; table : 'd int64_t }
      (** Pools [k] [[pages; hk; page; dk]] and [v] [[pages; hk; page; dv]]
          read through [table] [[lead…; max_pages]]: key [t] of lead index
          [b] is the pools' entry [table[b; t / page]] at [t mod page], and
          [tk = max_pages · page]. *)

type 'd pair = { q : 'd int64_t; k : 'd int64_t; len : 'd int64_t; tq : int }
(** The 0-d coordinates of one query-key pair: the query row, the key column,
    the row's visible length, and the number of query rows. *)

val causal : 'd pair -> 'd bool_t
(** [causal p] holds where [p.k <= p.q + (p.len - p.tq)]: the last query row
    sees every key before the row's length, and each row before it one key
    fewer. *)

val attention :
  ?scale:float ->
  ?lengths:'d int64_t ->
  ?mask:('d pair -> 'd bool_t) ->
  ?score:('d pair -> 'd float32_t -> 'd float32_t) ->
  kv:('v, 's, 'd) kv ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t * 'd float32_t
(** [attention ~kv q] is [(o, lse)] for [q] [[lead…; hq; tq; dk]], [hk]
    dividing [hq] (zero heads divide zero), query head [h] reading key head
    [h / (hq / hk)]. [o], [[lead…; hq; tq; dv]], is the softmax over each
    row's visible keys of [scale · q kᵀ] (default [1 / √dk]), rewritten by
    [score], times [v], in [q]'s dtype; [lse], [[lead…; hq; tq]], is the
    row's log-sum-exp in float32. A pair is visible where [mask] holds, its
    key is before its lead index's entry of [lengths] ([[lead…]], default
    [tk]), and a paged key's table entry is in [[0, pages)]. A row with no
    visible key, or only [-∞] scores, gives [o = 0] and [lse = -∞].

    A hidden pair's score never changes [o] or [lse], and a table entry
    outside [[0, pages)] reads no page. A finite value at a hidden key never
    changes [o]: its weight is exactly zero. Values past a row's length are
    zeroed, so they never change [o]. A NaN or an infinity at a key hidden
    from a row but within its length makes that row's [o] NaN, since [o]
    reads it at weight zero; a kernel library that skips hidden blocks may
    give a finite [o] instead, and says so in its own documentation. The
    weights are float32, and a visible key whose weight underflows to zero
    still counts: a NaN in its value makes the row's [o] NaN. A sum of zeros
    is [+0].

    [mask] and [score] are plain [Nx] code over 0-d values, applied once to
    stand-ins and kept as programs that run at each pair. Their results, and
    the arrays they capture, broadcast to [[lead…; hq; tq; tk]]; a captured
    array is read at each pair. On the pair's values they apply elementwise
    functions and broadcasts; another operation on them, or a read of one,
    raises naming [Nx.attention].

    {[
    let window p = Nx.(logical_and (causal p) (less p.q (add_s p.k 128L)))
    let alibi slopes p s = Nx.(s - cast float32 (p.q - p.k) * slopes)
    (* slopes [hq; 1; 1] *)
    let o, lse = Nx.attention ~mask:window ~score:(alibi slopes) ~kv q
    ]}

    [q]'s dtype is float16, bfloat16 or float32. Scores, their maxima and
    sums, and [lse] compute in float32, and [o] rounds once to [q]'s dtype.

    It computes as a contraction for the scores, a map and a reduction for
    the visible scores' log-sum-exp, a map for the weights and a contraction
    for [o]. Each call stores three float32 arrays of
    [[lead…; hq; tq; tk]]: the scores, the visible scores and the weights.
    With [lengths], it copies [v] with its keys past each row's length
    zeroed; with [Paged], it copies the keys and values the table reads into
    dense [[lead…; hk; tk; dk]] and [[lead…; hk; tk; dv]] arrays. An
    interpretation receives these operations; none is attention's own.

    Raises [Invalid_argument] if [q]'s dtype is another, [q] has fewer than
    three axes, extents disagree, [hk] does not divide [hq], [lengths] is not
    of shape [[lead…]], [dk] is [0] without [scale], or a result or capture
    of [mask] or [score] does not broadcast to [[lead…; hq; tq; tk]]. *)

(** {2:matrices Matrices}

    A value of two axes or more is a batch of matrices over its last two:
    rows, then columns. *)

val matrix_transpose : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [matrix_transpose x] is [x] with its last two axes exchanged; [x] itself
    below two axes. A view. *)

val tril : ?k:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tril ~k x] is [x] where column - row [<= k] (default [0]) and zero above,
    in each matrix of the batch.

    Raises [Invalid_argument] if [x] has fewer than two axes. *)

val triu : ?k:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [triu ~k x] is [x] where column - row [>= k] (default [0]) and zero below,
    as {!tril}. *)

val diagonal :
  ?offset:int -> ?axis1:int -> ?axis2:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [diagonal ~offset ~axis1 ~axis2 x] is the elements of [x] at index [i]
    along [axis1] and [i + offset] along [axis2] (default the last two axes,
    and [0]), for each [i] where both lie in their axes; that axis replaces the
    two, last, the others kept in order. A view where [x]'s two axes merge
    without a copy.

    Raises [Invalid_argument] if [x] has fewer than two axes, or [axis1] and
    [axis2] are not two distinct axes of [x]. *)

val trace : ?offset:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [trace ~offset x] is the {!sum} of [diagonal ~offset x] along its last
    axis: one value per matrix of the batch, [+0] for an empty diagonal.

    Raises [Invalid_argument] as {!diagonal} and {!sum} do. *)

val diag : ?k:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [diag ~k v] is, for a vector [v] of [n] elements, the square of
    [n + |k|] whose [k]th diagonal (default [0], above the main one where
    positive) holds [v] and whose other elements are zero; for a matrix, its
    diagonal [diagonal ~offset:k v].

    Raises [Invalid_argument] if [v] has neither one nor two axes. *)

(** {1:linalg Linear algebra}

    Factorisations, solves and norms of the matrices of a batch, laid out as
    {!section-matrices} says, [m] rows, [n] columns and [k = min m n]. They
    compute [float32], [float64], [complex64] and [complex128]: another dtype
    raises [Invalid_argument]. {!vector_norm}, {!matrix_norm},
    {!matrix_rank} and {!cond} also take the float formats narrower than 32
    bits. A dtype's {e precision} is [float64]'s for [float64] and
    [complex128], and [float32]'s for the other floats and [complex64].
    [aᴴ] is the conjugate transpose, the transpose of a real [a].

    A result has its operand's dtype, except a real result ({!svd}'s
    singular values, {!eigh}'s eigenvalues, {!slogdet}'s logarithm,
    {!lstsq}'s residuals and singular values, the norms {!vector_norm} and
    {!matrix_norm}, and {!cond}) and {!eig}'s complex results, each of
    which takes its dtype first, of a real or a complex operand. A
    norm or condition number is computed at the wider of its operand's
    precision and its dtype's, every other result at its operand's
    precision. Each is then rounded once to its dtype by the conversion
    rule, as {!real} stores: a dtype without NaN or infinities holds what
    {!cast} gives for the NaN and infinities each function states.

    Each matrix of a batch is computed on its own, and data never raises: a
    matrix whose factorisation fails, as each function states, is NaN in its
    own results alone. For a finite [a],
    [all ~axes:[ -2; -1 ] (isfinite (cholesky a))] tests which matrices of a
    batch are positive definite.

    Each kernel library states its error bounds. A factorisation leaves some
    choices open: the order of {!eig}'s eigenvalues; {!qr}'s factors, unique
    up to a unit factor per column of [q] and row of [r] where [r]'s diagonal
    has no zero; the vectors of {!svd}, {!eigh} and {!eig}, unique up to a
    unit factor where their value is simple and up to a unitary basis of its
    space where it repeats; and the columns [Complete] factors add. *)

(** The type for a square matrix's triangles, each with its diagonal: the
    elements on and below it, or on and above it. *)
type triangle = Nx_kernel.triangle = Lower | Upper

(** The type for how much of a matrix's unitary factors a factorisation gives:
    [k] columns of the left one and [k] rows of the right one, or the whole
    squares, of [m] and of [n]. *)
type factors = Nx_kernel.factors = Reduced | Complete

val cholesky : ?triangle:triangle -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cholesky a] is the factor of the Hermitian positive definite matrix that
    [a]'s [triangle] (default [Lower]) and the real parts of its diagonal name,
    which alone are read: [L], lower triangular with a real and positive
    diagonal and [a = L Lᴴ], or under [~triangle:Upper] [U = Lᴴ], upper
    triangular with [a = Uᴴ U]. The other triangle is [+0]. A matrix whose
    factorisation meets a pivot that is not positive, or NaN, is NaN.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val qr : ?factors:factors -> ('v, 's, 'd) t -> ('v, 's, 'd) t * ('v, 's, 'd) t
(** [qr a] is [(q, r)] with [a = q r], [q]'s columns orthonormal and [r] upper
    triangular: [q] is [[…; m; k]] and [r] [[…; k; n]], or under
    [~factors:Complete] [[…; m; m]] and [[…; m; n]].

    Raises [Invalid_argument] if [a] has fewer than two axes, or for its
    dtype. *)

val lu : ('v, 's, 'd) t -> 'd int64_t * ('v, 's, 'd) t * ('v, 's, 'd) t
(** [lu a] is [(perm, l, u)] where row [i] of [matmul l u] is row [perm.{i}] of
    [a], [l] is [[…; m; k]], lower triangular with ones on its diagonal, and [u]
    is [[…; k; n]], upper triangular; [perm] is [[…; m]].

    The factorisation pivots partially: at each column it moves up the row at
    or below the diagonal whose magnitude ([|re| + |im|] for a complex) is
    greatest, the first such row on a tie, so each element of a real [l] has a
    magnitude of at most [1]. A NaN is never greater, and nothing is greater
    than a NaN. A singular [a] factors as any other, its zero pivots on [u]'s
    diagonal.

    Raises [Invalid_argument] if [a] has fewer than two axes, or for its
    dtype. *)

val svd :
  ?vectors:bool ->
  ?factors:factors ->
  (float, 'r) dtype ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t * (float, 'r, 'd) t * ('v, 's, 'd) t
(** [svd dt a] is [(u, s, vh)], [a] the product of [u], the diagonal matrix
    of [s] and [vh], [s] the [[…; k]]
    singular values in [dt], descending and not negative, a zero one [+0];
    [u] and [vh] have orthonormal columns and rows: [[…; m; k]] and
    [[…; k; n]], or under [~factors:Complete] [[…; m; m]] and [[…; n; n]].
    Under [~vectors:false] (default [true]) neither [u] nor [vh] is computed:
    they are [[…; m; 0]] and [[…; 0; n]]. A matrix that holds NaN or an
    infinity, or on which the iteration does not converge, is NaN in all three.

    Raises [Invalid_argument] if [a] has fewer than two axes, or for its
    dtype. *)

val eig :
  ?vectors:bool ->
  (Complex.t, 'c) dtype ->
  ('v, 's, 'd) t ->
  (Complex.t, 'c, 'd) t * (Complex.t, 'c, 'd) t
(** [eig dt a] is [(w, v)] in [dt], the [[…; n]] eigenvalues of the real or
    complex [a] and columns of unit norm [v], [matmul a v] being [v] with
    each column scaled by its eigenvalue. Under
    [~vectors:false] (default [true]) [v] is not computed and is [[…; n; 0]].
    NaN as for {!svd}.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not
    square, or for its dtype. *)

val eigh :
  ?vectors:bool ->
  ?triangle:triangle ->
  (float, 'r) dtype ->
  ('v, 's, 'd) t ->
  (float, 'r, 'd) t * ('v, 's, 'd) t
(** [eigh dt a] is [(w, v)], the [[…; n]] eigenvalues in [dt] of the
    Hermitian [a], ascending, and orthonormal columns [v], [matmul a v]
    being [v] with each column scaled by its eigenvalue. [a] is the
    Hermitian
    matrix its [triangle] (default [Lower]) and the real parts of its diagonal
    name, which alone are read. Under [~vectors:false] (default [true]) [v] is
    not computed and is [[…; n; 0]]. A matrix whose read part holds NaN or an
    infinity, or on which the iteration does not converge, is NaN in both.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val det : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [det a] is the determinant of each matrix of [a], the product of the
    diagonal of {!lu}'s [u] negated once per row interchange. A zero
    determinant is [+0], and a matrix with no row has determinant [1]. The
    product is taken in the dtype and overflows or underflows where a partial
    product does. {!slogdet} holds determinants beyond the dtype's range.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val slogdet :
  (float, 'r) dtype -> ('v, 's, 'd) t -> ('v, 's, 'd) t * (float, 'r, 'd) t
(** [slogdet dt a] is [(sign, logabs)] with [det a = sign · exp logabs], from
    {!lu}'s pivots: [logabs] in [dt] is the sum of their magnitudes'
    logarithms, and [sign] is [1.] or [-1.] for a real [a] and a complex
    number of magnitude [1.] for a complex one. A singular matrix, one with
    a zero pivot, has [sign] [+0] and [logabs] [neg_infinity] whatever its
    other pivots.

    Raises [Invalid_argument] as {!det}. *)

(** The type for the matrix a solve applies: [aᵀ], or [aᴴ]. For a real [a]
    they are one. *)
type transpose = Transpose | Adjoint

val solve_triangular :
  ?triangle:triangle ->
  ?transpose:transpose ->
  ?unit_diag:bool ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [solve_triangular a b] is [x] with [matmul a x = b], or under [transpose]
    with [matmul aᵀ x = b] or [matmul aᴴ x = b], reading [a]'s [triangle]
    (default [Lower]) alone, without its diagonal under [~unit_diag:true]
    (default [false]), whose ones it takes instead. A [b]
    with one axis fewer than [a] is a vector, [[…; n]], and [x] has its shape;
    otherwise [b] is [[…; n; r]], [r] right-hand sides. [a]'s and [b]'s
    leading axes broadcast. A zero on a diagonal it reads makes that matrix's
    [x] NaN.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    for its dtype, if [b] has neither [a]'s rank nor one axis fewer, if [b]'s
    rows are not [n], or if the leading axes do not broadcast. *)

val solve : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [solve a b] is [x] with [matmul a x = b], for a square [a], from {!lu}'s
    factors and two triangular solves; [b] is as {!solve_triangular} takes
    it. A matrix with a pivot of exactly zero makes its [x] NaN. A matrix
    singular in exact arithmetic whose pivot rounds to a tiny nonzero value
    gives large, infinite or NaN elements instead.

    Raises [Invalid_argument] as {!solve_triangular}. *)

val inv : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [inv a] is [solve a] of the identity: NaN for a matrix {!solve} finds
    singular.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val lstsq :
  ?rtol:float ->
  (float, 'r) dtype ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t * (float, 'r, 'd) t * 'd int64_t * (float, 'r, 'd) t
(** [lstsq dt a b] is [(x, residuals, rank, s)]: [x] the least-squares
    solution of [matmul a x ≈ b] of least norm, whatever [a]'s shape and
    rank, from [s], {!svd}'s singular values of [a] in [dt]. A singular value
    at or below [rtol] times [max m n] times its matrix's greatest counts as
    zero; [rtol] defaults to [a]'s machine epsilon. [rank] is the
    number of the others, [-1] for a matrix on which {!svd} fails, whose [x]
    is NaN. [residuals] is the squared norm of each column of
    [b - matmul a x] in [dt]: [b]'s shape without its rows. [b] is as
    {!solve_triangular} takes it, with [m] rows.

    Raises [Invalid_argument] if [a] has fewer than two axes, for its dtype,
    if [b] has neither [a]'s rank nor one axis fewer, if [b]'s rows are not
    [m], if the leading axes do not broadcast, or if [rtol] is negative or
    NaN. *)

val pinv : ?rtol:float -> ?hermitian:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [pinv a] is the [[…; n; m]] Moore-Penrose pseudoinverse of [a], from
    {!svd}, or from {!eigh} of the Hermitian [a] its lower triangle names under
    [~hermitian:true] (default [false]). A singular value, or an eigenvalue's
    magnitude, at or below [rtol] times [max m n] times its matrix's greatest
    counts as zero; [rtol] defaults to the dtype's machine epsilon. A matrix on
    which that factorisation fails is NaN.

    Raises [Invalid_argument] if [a] has fewer than two axes, is not square
    under [~hermitian:true], for its dtype, or if [rtol] is negative or NaN. *)

val vector_norm :
  (float, 'r) dtype ->
  ?ord:float ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('v, 's, 'd) t ->
  (float, 'r, 'd) t
(** [vector_norm dt x] is, in [dt], the norm of order [ord] (default [2.])
    of the vector of the elements of [x] along [axes] (default all of them),
    dropping those axes unless [keepdims]. Of the magnitudes [|x|], the
    moduli of a complex [x], it is
    [(Σ |x|^ord)^(1/ord)] for a nonzero [ord], the count of nonzero elements
    for [0.], and the greatest or least magnitude for [infinity] or
    [neg_infinity]. A negative [ord] over no element is [infinity].

    The sums of powers are scaled by the greatest magnitude, or the least for
    a negative [ord], so that a norm the dtype holds never overflows or
    underflows on the way: the norm of [[1e200; 1e200]] is [1.414e200]. An
    element whose magnitude is NaN makes its norm NaN.

    Raises [Invalid_argument] if an axis is outside [x]'s rank or repeats,
    if [ord] is NaN, or if [ord] is [infinity] or [neg_infinity] and a
    vector has no element. *)

val matrix_norm :
  (float, 'r) dtype ->
  ?ord:[ `Fro | `Nuc | `One | `Two | `Inf | `NegOne | `NegTwo | `NegInf ] ->
  ?keepdims:bool ->
  ('v, 's, 'd) t ->
  (float, 'r, 'd) t
(** [matrix_norm dt x] is, in [dt], the norm of order [ord] (default [`Fro])
    of each matrix of [x], dropping its last two axes unless [keepdims]. Of
    a matrix:
    - [`Fro] is [sqrt (Σ |x|²)], scaled as {!vector_norm}'s, and [`Nuc] the
      sum of its singular values.
    - [`Two] is its greatest singular value and [`NegTwo] its least.
    - [`One] is the greatest sum of a column's magnitudes and [`NegOne] the
      least; [`Inf] and [`NegInf] are the same of its rows.

    Raises [Invalid_argument] if [x] has fewer than two axes, or if [ord]
    takes a greatest or least singular value, column or row of a matrix
    that has none. *)

val matrix_rank : ?rtol:float -> ('v, 's, 'd) t -> 'd int64_t
(** [matrix_rank a] is the rank of each matrix of [a], of [a]'s batch shape:
    the number of its singular values above [rtol] times [max m n] times
    the greatest, as {!lstsq} counts them, [rtol] defaulting to the dtype's
    machine epsilon, float32's for a narrower float. A matrix of no element
    has rank [0]. A matrix on which {!svd} is NaN, one that holds NaN or an
    infinity or on which the iteration does not converge, has rank [-1]: its
    rank is undefined.

    Raises [Invalid_argument] if [a] has fewer than two axes, for its dtype,
    or if [rtol] is negative or NaN. *)

val cond :
  (float, 'r) dtype ->
  ?p:[ `One | `Two | `Inf ] ->
  ('v, 's, 'd) t ->
  (float, 'r, 'd) t
(** [cond dt a] is, in [dt], the condition number of each matrix of [a] in
    the norm [p] (default [`Two]), dropping its last two axes: under [`Two]
    its greatest singular value over its least, under [`One] and [`Inf] its
    {!matrix_norm} of order [p] times that of its {!inv}. A singular matrix,
    whose least singular value is zero or whose {!inv} is NaN, has an
    infinite condition number; one singular only to rounding has one near
    [1 / ε] or above, [ε] the machine epsilon of the precision it is
    computed at. A matrix that holds NaN or an infinity, or on which
    {!svd} fails, has NaN.

    Raises [Invalid_argument] if [a] has fewer than two axes or a matrix of
    no element, if [dt] has no infinities, or if [p] is [`One] or [`Inf]
    and [a] is not square. *)

(** {1:fft Fourier transforms}

    Along an axis of [n] points, the forward transform is
    [y[k] = Σ_j x[j] e^(-2πi jk/n)] and the inverse the same sum with
    [e^(+2πi jk/n)]. A transform along several axes is the transform along
    each in turn. A transform of no point is the empty sum, zero.

    [axes] defaults to the last axis. [s] gives the number of points along each
    of [axes], in order: the operand is cut to it, or extended with zeros. A
    transform of [N] points in all is scaled by [norm]: under [Backward]
    (default) the inverse divides by [N], under [Forward] the forward one does,
    and under [Ortho] each divides by [√N].

    Each raises [Invalid_argument] if the operand is 0-d, if [axes] is empty,
    if an axis is outside the operand's rank or repeats, if [s] does not give
    one length per axis, or if a length is negative. *)

(** The type for the scaling of a transform and its inverse. *)
type fft_norm = Backward | Forward | Ortho

val fft :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 's, 'd) t ->
  (Complex.t, 's, 'd) t
(** [fft x] is the forward transform of [x] along [axes]. *)

val ifft :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 's, 'd) t ->
  (Complex.t, 's, 'd) t
(** [ifft x] is the inverse transform of [x] along [axes]: under one [norm],
    [ifft (fft x)] is [x] to rounding. *)

val rfft :
  (Complex.t, 'c) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (Complex.t, 'c, 'd) t
(** [rfft dt x] is the forward transform of the real [x] along [axes], the
    last of them keeping bins [0] to [n/2] of its [n] points, the others
    being redundant: one bin for [n = 0]. It computes at the wider of [x]'s
    precision and [dt]'s, the formats narrower than 32 bits at [float32], and
    rounds the scaled result once to [dt]. *)

val irfft :
  (float, 'r) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 's, 'd) t ->
  (float, 'r, 'd) t
(** [irfft dt x] is the real inverse of {!rfft} along [axes], of [n] points
    along the last of them from its first [n/2 + 1] bins, which are cut or
    extended with zeros to that many. [s] gives the points of the result;
    without it, the other axes keep their extents and [n] is [2 (m - 1)] for
    [m] bins. Along the last axis, bin [k] above [n/2] reads as the conjugate
    of bin [n - k], and the imaginary parts of bin [0] and, for an even [n],
    of bin [n/2] are not read: it is this linear map for every operand,
    Hermitian or not. It computes and rounds as {!rfft}.

    Raises [Invalid_argument] also if the last axis has no bin and [s] does
    not give its points. *)

val hfft :
  (float, 'r) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 's, 'd) t ->
  (float, 'r, 'd) t
(** [hfft dt z] is the forward transform over [axes] of the Hermitian signal
    [h], with [h(-j) = conj h(j)] for every index [j] over [axes] taken modulo
    their lengths, whose first [n/2 + 1] samples along the last of [axes] are
    [z], read as {!irfft} reads its bins: its spectrum is real. Under one
    [norm] it is [irfft dt (conjugate z)] scaled as a forward transform, so
    [Backward] does not scale it. It takes [axes] and [s], computes and
    rounds as {!irfft}, and raises as it does. *)

val ihfft :
  (Complex.t, 'c) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (Complex.t, 'c, 'd) t
(** [ihfft dt x] is the first [n/2 + 1] samples, along the last of [axes], of
    the Hermitian signal whose transform over [axes] is the real [x]:
    [conjugate (rfft dt x)] scaled as an inverse transform. Under one [norm],
    [hfft ~s (ihfft x)] is [x] to rounding, [s] the lengths of [x]'s [axes].
    It takes [axes] and [s], computes and rounds as {!rfft}, and raises as it
    does. *)

(** The type for the four discrete cosine and sine transforms, I to IV. *)
type trig_type = Type_1 | Type_2 | Type_3 | Type_4

val dct :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [dct x] is the discrete cosine transform of type [type_] (default
    [Type_2]) of [x] along each of [axes]: of [n] points [x[j]], type II is
    [y[k] = 2 Σ_j x[j] cos (πk(2j + 1) / 2n)], type I
    [y[k] = x[0] + (-1)^k x[n-1] + 2 Σ_(0<j<n-1) x[j] cos (πjk / (n-1))],
    type III [y[k] = x[0] + 2 Σ_(j>0) x[j] cos (πj(2k + 1) / 2n)] and type IV
    [y[k] = 2 Σ_j x[j] cos (π(2j + 1)(2k + 1) / 4n)]. Under [Forward] it
    divides by the transform's logical length, [2 (n - 1)] for type I and
    [2n] otherwise, and under [Ortho] it is the orthonormal transform. A
    copy of [x]'s shape and dtype. It computes the formats narrower than 32
    bits at [float32] and rounds the result once.

    Raises [Invalid_argument] also if type I is asked of an axis of one
    element. *)

val idct :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [idct x] is the inverse of {!dct} under the same [type_] and [norm]: types
    I and IV are their own inverses up to scale, II and III each other's.

    Raises [Invalid_argument] as {!dct}. *)

val dst :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [dst x] is the discrete sine transform of type [type_] (default [Type_2])
    along each of [axes]: type II is
    [y[k] = 2 Σ_j x[j] sin (π(k + 1)(2j + 1) / 2n)], type I
    [y[k] = 2 Σ_j x[j] sin (π(j + 1)(k + 1) / (n + 1))], type III
    [y[k] = (-1)^k x[n-1] + 2 Σ_(j<n-1) x[j] sin (π(j + 1)(2k + 1) / 2n)] and
    type IV [y[k] = 2 Σ_j x[j] sin (π(2j + 1)(2k + 1) / 4n)]. Its logical
    length is [2 (n + 1)] for type I and [2n] otherwise, and [norm] scales and
    narrow formats compute as {!dct}'s. *)

val idst :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [idst x] is the inverse of {!dst} under the same [type_] and [norm]. *)

val fftfreq : (float, 's) dtype -> ?d:float -> int -> (float, 's, host) t
(** [fftfreq dt ~d n] is the host value of the frequency of each of {!fft}'s
    [n] bins for samples [d] apart (default [1.]), in cycles per unit of [d]:
    [i / (d n)] for [i] in [[0, (n - 1) / 2]], then [(i - n) / (d n)] for the
    rest, the negative frequencies. Each is computed at float64 from [d n]
    rounded once, then stored in [dt]: [fftfreq float64 4] is [0], [0.25],
    [-0.5] and [-0.25].

    Raises [Invalid_argument] if [n < 1] or [d] is zero or not finite. *)

val fftfreq_at :
  'd Placement.t -> (float, 's) dtype -> ?d:float -> int -> (float, 's, 'd) t
(** [fftfreq_at p dt ~d n] is {!fftfreq}[ dt ~d n] at [p], an [_at] form as
    {!section-creation} states. *)

val rfftfreq : (float, 's) dtype -> ?d:float -> int -> (float, 's, host) t
(** [rfftfreq dt ~d n] is the host value of the frequency of each of
    {!rfft}'s [n / 2 + 1] bins for [n] samples [d] apart: [i / (d n)] for [i]
    in [[0, n / 2]], computed as {!fftfreq}'s.

    Raises [Invalid_argument] as {!fftfreq}. *)

val rfftfreq_at :
  'd Placement.t -> (float, 's) dtype -> ?d:float -> int -> (float, 's, 'd) t
(** [rfftfreq_at p dt ~d n] is {!rfftfreq}[ dt ~d n] at [p], an [_at] form as
    {!section-creation} states. *)

val fftshift : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [fftshift x] is [x] rolled along each of [axes] (default the last) by
    [n / 2] places toward the end, [n] its extent there: the zero frequency
    of a transform moves to the middle, after the negative frequencies.
    [fftshift (fftfreq float64 5)] is [-0.4], [-0.2], [0], [0.2] and [0.4]. A
    copy.

    Raises [Invalid_argument] if [x] is 0-d, [axes] is empty, or an axis is
    outside [x]'s rank or repeats. *)

val ifftshift : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [ifftshift x] undoes {!fftshift} along the same [axes]: [x] rolled by
    [n / 2] places toward the start. The two differ only along an axis of odd
    extent.

    Raises [Invalid_argument] as {!fftshift}. *)

val hann : (float, 's) dtype -> int -> (float, 's, host) t
(** [hann dt n] is the host value of the periodic Hann window of [n] points,
    [0.5 - 0.5 cos (2πi / n)], computed at float64 and rounded once: the point
    that would open the next period is dropped, so shifted copies sum to a
    constant.

    Raises [Invalid_argument] if [n] is negative. *)

val hann_at : 'd Placement.t -> (float, 's) dtype -> int -> (float, 's, 'd) t
(** [hann_at p dt n] is {!hann}[ dt n] at [p], an [_at] form as
    {!section-creation} states. *)

val stft :
  (Complex.t, 'c) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 's, 'd) t ->
  (float, 's, 'd) t ->
  (Complex.t, 'c, 'd) t
(** [stft dt ~window x] is the short-time transform of [x] along its last
    axis: frames of [window] samples, one every [step] (default [window / 4],
    at least [1]), each multiplied by [win] and transformed by {!rfft}.
    [[…; n]] gives [[…; frames; window / 2 + 1]], time first, with
    [frames = (n - window) / step + 1]; samples past the last whole frame are
    dropped. [win] defaults to [hann (dtype x) window], and
    [ones (dtype x) [| window |]] frames without a taper.

    Raises [Invalid_argument] if [x] is 0-d, [window] or [step] is below [1],
    [window] exceeds the last axis, or [win] is not [[window]]. *)

val istft :
  (float, 'r) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 'r, 'd) t ->
  ?length:int ->
  (Complex.t, 'c, 'd) t ->
  (float, 'r, 'd) t
(** [istft dt ~window z] is the signal whose {!stft} under the same [window],
    [step] and [win] is [z]: each frame inverted by {!irfft}, multiplied by
    [win], summed back where it was taken, and divided by the sum of [win]'s
    squares landing there. [[…; frames; window / 2 + 1]] gives
    [[…; (frames - 1) step + window]], or [length] samples, cut or extended
    with zeros. A sample every frame's [win] holds at zero carries nothing and
    is [0.]: under the default taper, sample [0].

    Raises [Invalid_argument] if [z] has fewer than two axes, [window] is below
    [1], [step] is outside [[1, window]], the last axis is not
    [window / 2 + 1], [win] is not [[window]], or [length < 1]. *)

(** {1:order Order and counted lengths}

    Sorting, selection, searching and grouping order elements by one order, the
    {e sort order}:
    - floats as [-∞ < … < -0. < +0. < … < +∞ < NaN], every NaN equal to every
      other whatever its sign and payload;
    - [false < true], and integers by value, unsigned ones as unsigned;
    - complex numbers by real part, then imaginary part, each as a float; one
      with a NaN part equals every other such and is above every other complex
      number.

    A descending sort orders by its exact reverse: NaN first, [+0.] before
    [-0.]. Keys are elements, or rows of a 2-d value compared column [0] first:
    {!lexsort} sorts them, {!searchsorted} searches them and {!unique} groups
    them. Complex numbers have no keys. *)

val sort :
  ?descending:bool -> ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t * 'd int64_t
(** [sort ~descending ~axis x] is [x]'s elements along [axis] (default [-1]) in
    the sort order, or its reverse where [descending] (default [false]), and the
    positions they come from. The sort is stable in both directions: equal
    elements keep their order, NaNs included. The values are
    [take_along_axis ~axis positions x], bit for bit.

    Raises [Invalid_argument] if [axis] is not an axis of [x]. *)

val argsort : ?descending:bool -> ?axis:int -> ('v, 's, 'd) t -> 'd int64_t
(** [argsort ~descending ~axis x] is [snd (sort ~descending ~axis x)]. *)

val top_k : k:int -> ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t * 'd int64_t
(** [top_k ~k ~axis x] is the first [k] entries along [axis] (default [-1]) of
    [sort ~descending:true ~axis x]: the greatest first, equal ones lowest
    position first, a NaN before every number. One sort that keeps [k].

    Raises [Invalid_argument] if [axis] is not an axis of [x] or [k] is outside
    [[0, n]], [n] the axis' extent. *)

val order_key : ('k, 'r) dtype -> ('v, 's, 'd) t -> ('k, 'r, 'd) t
(** [order_key kd x] is, for each element of [x], an unsigned integer of [kd]
    whose order is the sort order: two keys compare, read unsigned, as their
    elements sort, and are equal iff the elements are equal in the sort order.
    [kd] is {!uint8}, {!uint16}, {!uint32} or {!uint64}, at least as wide as
    [x]'s elements; [uint64] keys every dtype but the complex ones.

    The key of an element is:
    - for a signed integer, its value in the signed integer of [kd]'s width with
      the sign bit flipped;
    - for an unsigned integer, its value, and for a boolean [0] or [1];
    - for a float, the bits of its value in the float of [kd]'s width, all of
      them flipped where the sign bit is set and only the sign bit elsewhere, so
      [-0.] keys just below [+0.]; every NaN keys to all ones. The float of a
      width is [x]'s own dtype at its width, bfloat16 for bfloat16 at 16 bits,
      float8_e4m3fn for float4_e2m1fn at 8 bits, and otherwise float16, float32
      or float64.

    An exact cast to a dtype no wider than [kd] keeps the key wherever the two
    dtypes key through the same float: [order_key uint64 (cast int64 x)] is
    [order_key uint64 x] for every signed integer [x]. A key of several columns
    is a row of order keys made with {!stack} [~axis:1]; a column that sorts
    descending is {!bitwise_not} of its key.

    Raises [Invalid_argument] if [x] is complex, or if [kd] is not one of the
    four or is narrower than [x]'s elements. *)

val lexsort : ('v, 's, 'd) t -> 'd int64_t
(** [lexsort keys] is the stable permutation that sorts the keys of [keys]: [n]
    elements of a 1-d [keys], or [n] rows of a 2-d one, compared column [0]
    first, each in the sort order. [take ~axis:0 (lexsort keys) keys] is in
    order and equal keys keep their order; a 2-d [keys] with no column leaves
    every row in place. It sorts one column at a time, the last first: [w] sorts
    for [w] columns.

    Raises [Invalid_argument] if [keys] is neither 1-d nor 2-d, or is complex.
*)

val searchsorted :
  side:[ `Left | `Right ] -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd int64_t
(** [searchsorted ~side s v] is, for each key of [v], the number of keys of [s]
    before it ([`Left]), or before or equal to it ([`Right]). Keys compare in
    the sort order, except that [-0.] equals [+0.], as {!less} compares numbers.
    - A 1-d [s] holds [m] keys of one element; [v] holds keys of one element in
      any shape, and the result has [v]'s shape.
    - A 2-d [s] of shape [[|m; w|]] holds [m] rows; [v] has shape [[|n; w|]],
      and the result shape [[|n|]].

    It is a binary search of [⌈log₂ (m + 1)⌉] rounds, each a gather of one key
    of [s] per key of [v]. Where [s] is not in order, each result is still in
    [[0, m]].

    Raises [Invalid_argument] if [s] is neither 1-d nor 2-d, is complex, or is
    2-d and [v] is not 2-d with as many columns. *)

(** {2:counted Lengths that are data}

    A length that is data is read once, on the host, as the function's one read:
    a placed operand's set synchronises once. A traced value cannot be read, so
    there the function raises naming itself, unless [size] fixes the length, in
    which case nothing is read. *)

val positions : ?size:int -> ('v, 's, 'd) t -> 'd int64_t
(** [positions c] is each index [i] of the 1-d [c] repeated [c.{i}] times, in
    increasing order. A boolean counts [1] where it holds, so [positions c] is
    the indices where [c] holds; a negative count counts zero, as does a
    [uint64] count past [int64]'s range. Without [size], the result's length is
    the counts' sum, read on the host. With [size], the result has [size]
    entries and reads nothing: the first positions, then [dim 0 c]. Along an
    axis of [c]'s length that pad is out of range, so {!take} reads zero there
    and {!scatter} writes nothing; along a longer axis it is a real position.

    Raises [Invalid_argument] if [c] is not 1-d, is a float or a complex, or if
    [size] is negative. *)

val argwhere : ?size:int -> ('v, 's, 'd) t -> 'd int64_t
(** [argwhere t] is the coordinates of [t]'s elements other than zero, a row
    each in C order: of shape [[k; ndim t]] for [k] such elements, whose row
    [i] is the coordinates in [t]'s shape of the [i]th entry of
    [positions (flatten (not_equal t (zeros_like t)))]. A NaN is not zero,
    [-0.] is, and a 0-d [t] other than zero has one row of no coordinate.
    Without [size], [k] is read on the host. With [size], the result has
    [size] rows and reads nothing: the first rows, then rows of [t]'s
    extents, out of range along every axis.

    Raises [Invalid_argument] if [size] is negative. *)

val nonzero : ?size:int -> ('v, 's, 'd) t -> 'd int64_t list
(** [nonzero ~size t] is the coordinates of [t]'s elements other than zero,
    one 1-d value per axis: the [j]th is column [j] of [argwhere ~size t], a
    view. With [size], a pad entry is the axis' extent, outside it.

    Raises [Invalid_argument] if [t] is 0-d or [size] is negative. *)

val extract :
  ?size:int -> ?fill:'v -> ('c, 'r, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [extract c x] is the elements of [x] where [c] is not zero, in C order,
    [c] and [x] flattened: a NaN is not zero, [-0.] is. Without [size], its
    length is read on the host. With [size], it has [size] entries and reads
    nothing: the first such elements, then [fill] (default zero).

    Raises [Invalid_argument] if [c] and [x] differ in their number of
    elements, [size] is negative, or [fill] is outside [x]'s dtype. *)

type 'd groups = {
  ids : 'd int64_t;  (** Each key's group, [n] entries. *)
  first : 'd int64_t;  (** Each group's first key, ascending. *)
  counts : 'd int64_t;  (** Each group's number of keys. *)
  count : 'd int64_t;  (** [k], 0-d. *)
}
(** The type for [n] keys in [k] groups of equal keys, numbered in order of
    first appearance. *)

val unique : ?size:int -> ('v, 's, 'd) t -> 'd groups
(** [unique keys] groups the keys of [keys] (elements of a 1-d [keys], rows of a
    2-d one) that are equal in the sort order: every NaN is one group, [-0.] and
    [+0.] two. For [g = unique keys], [g.ids.{0}] is [0],
    [take ~axis:0 g.first keys] is the distinct keys in order of first
    appearance, and key [i]'s group starts at [g.first.{g.ids.{i}}], at most
    [i].

    Without [size], [first] and [counts] have [k] entries, [k] read on the host.
    With [size] they have [size] entries and nothing is read: past [k], [first]
    holds [n] and [counts] [0]; below [k], the later groups are cut from them,
    and [ids] still numbers those groups from [size].

    It is one stable sort of the keys and a few passes over them.

    Raises [Invalid_argument] if [keys] is neither 1-d nor 2-d, is complex, or
    if [size] is negative. *)

val quantile :
  ?axis:int -> float array -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [quantile ~axis qs x] is the quantiles [qs] of [x] along [axis], or of all
    its elements without [axis]. The result leads with an axis of one entry per
    probability, followed by [x]'s axes without [axis].

    The quantile [q] of [n] elements interpolates linearly between the order
    statistics around [h = q (n - 1)] in the sort order: with [a] and [b] the
    elements at [floor h] and [floor h + 1], it is [a + (h - floor h) (b - a)],
    and [a] exactly where [h] is an integer or [a] equals [b]. NaN sorts last,
    so it reaches the upper quantiles. The narrow floats interpolate in float32
    and round once. One sort serves every probability.

    Raises [Invalid_argument] if a probability is outside [[0, 1]] or NaN, if
    [axis] is not an axis of [x], or if there is no element along it. *)

val histogram :
  ?weights:(float, 's, 'd) t ->
  ((float, 's, 'd) t * (float, 's, 'd) t) list ->
  (float, 's, 'd) t
(** [histogram [ (e1, x1); …; (ed, xd) ]] counts points into the cells of the
    edges [e1] to [ed]: point [i] is [(x1.{i}, …, xd.{i})], one per element of
    the coordinates, which share one shape. The result has one axis per
    dimension, of [dim 0 ek - 1] bins; cell [(j1, …, jd)] counts the points
    whose coordinate [xk] is in bin [jk] of [ek] for every [k].
    - Bin [j] holds the coordinates from [ek.{j}] included to [ek.{j+1}]
      excluded; the last bin includes its right edge too. Coordinates compare as
      {!less} does, so [-0.] is [+0.].
    - A point outside the edges along a dimension, NaN included, is in no cell.
    - Counts accumulate in [int64] and convert once to the dtype. With
      [weights], of the coordinates' shape, a cell holds the sum of its points'
      weights, as {!scatter}'s [Add] sums them.

    Edges must ascend; along edges that do not, a point lands in some cell or in
    none.

    Raises [Invalid_argument] if no dimension is given, if an edge vector is not
    1-d of two or more edges, if the coordinates' shapes differ, or if [weights]
    has another shape. *)

(** {1:windows Windows and segments}

    A function here reads its operand in groups of elements: classes, windows
    along its last axes, or rows grouped by positions held in data. Each
    computes where its operands lie.

    The {e spatial} axes are an operand's last [K] axes, and the axes before
    them are batch axes, each computed alone. Along a spatial axis of [n]
    elements, padded with [lo] zeros before and [hi] after, a window of [k]
    elements with stride [s] and dilation [d] holds the padded axis' elements
    [w s + j d], [j < k], for window [w]. There are
    [(n + lo + hi - d (k - 1) - 1) / s + 1] windows, and none where
    [d (k - 1) + 1] exceeds [n + lo + hi]. Windows across several spatial axes
    are every combination of one window per axis, in C order, and so are
    their taps [j]. Except for {!extract_patches}, whose result they are,
    these functions read the windows where they lie and store none. *)

val one_hot : num_classes:int -> ('v, 's, 'd) t -> 'd uint8_t
(** [one_hot ~num_classes i] is [i]'s shape with an axis of [num_classes]
    appended, holding [1] at the class each element of [i] names and [0] at the
    others: an element outside [\[0, num_classes)] gives a row of zeros.
    Integers only.

    Raises [Invalid_argument] if [i] is not of an integer dtype or
    [num_classes] is negative. *)

val extract_patches :
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [extract_patches ~kernel_size ~stride ~dilation ~padding x] is the windows
    of [x] over its last [K = Array.length kernel_size] axes, padded with zeros
    by [padding]'s [(lo, hi)] pairs. A value of shape [[lead…; spatial…]]
    gives [[lead…; P; L]], [P] the product of [kernel_size] and [L] the number
    of windows: its element at [(…, p, l)] is window [l]'s tap [p]. A copy.

    Raises [Invalid_argument] if [kernel_size] is empty; [stride], [dilation]
    and [padding] do not have one entry per kernel axis; a size, stride or
    dilation is below [1] or a padding negative; or [x] has fewer axes than
    [kernel_size]. *)

val combine_patches :
  output_size:int array ->
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding p]
    is the adjoint of {!extract_patches} from spatial axes [output_size]: [p],
    of shape [[lead…; P; L]], is summed into [[lead…; output_size…]], each of
    its elements added at the position that {!extract_patches} reads it from,
    and one that lands in the padding dropped. An element no window covers is
    [+0]. With one geometry, the sum of [extract_patches x * y] is the sum of
    [x * combine_patches y].

    The sums add from [+0] in an order the kernel library states, the
    float formats narrower than 32 bits at [float32], rounded once. A float
    sum keeps the bound of a {!sum} of its terms ({!section-reductions}).
    Integers wrap.
    Every dtype but booleans.

    Raises [Invalid_argument] as {!extract_patches} does on the geometry, if
    [output_size] does not have one entry per kernel axis or has a negative
    one, if [p] has fewer than two axes or its last two are not [P] and [L],
    or if [p] is boolean. *)

val correlate :
  ?padding:[ `Full | `Same | `Valid ] ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [correlate ~padding x kernel] is the cross-correlation of [x] with
    [kernel] over the last [K = ndim kernel] axes of [x], each sum of products
    taken without flipping [kernel]. [padding] defaults to [`Valid].

    Along each spatial axis, of [n] elements in [x] and [k] in [kernel],
    value [i] correlates [kernel] with the elements of [x] from [i - k + 1],
    those outside [x] being zero. [padding] keeps a run of consecutive values.
    With [m = min n k]:
    - [`Full] keeps the [n + k - 1] from value [0];
    - [`Valid] keeps the [max n k - m + 1] where the shorter of [x] and
      [kernel] lies within the longer, from value [m - 1]. An empty [x] lies
      within [kernel] at [k + 1] positions, so it gives [k + 1] zeros;
    - [`Same] keeps [max n k], from value [(m - 1) / 2] if [k <= n] and from
      value [m / 2] if [k > n]. With [k <= n], value [i] correlates [kernel]
      with the elements of [x] from [i - k / 2] to [i + (k - 1) / 2]: an odd
      kernel is centred on element [i].

    For [x = [|1.; 2.; 3.; 4.; 5.; 6.; 7.|]] and
    [kernel = [|1.; 10.; 100.; 1000.|]], [`Same] gives
    [[|2100.; 3210.; 4321.; 5432.; 6543.; 7654.; 765.|]].

    Each value sums one product per element of [kernel], from [+0], in an
    order fixed by the shapes, as {!sum} adds. The float formats narrower
    than 32 bits multiply and sum at [float32] and round once. Integers wrap.
    Every dtype but booleans.

    Raises [Invalid_argument] if [x] has fewer axes than [kernel], [kernel] has
    an axis of no element, or the dtype is boolean. *)

(** {2:segments Segments and ranges}

    A {e row} of [x] is its value at one position of its first axis. A
    {e segment} is the rows whose id, held in data, is one number; a
    {e range} is the rows between two bounds held in data. As positions held
    in data, an id or a bound never raises: an id outside the segments drops
    its row, and a bound clips into the rows.

    {!reduce_segments} and {!reduce_ranges} fold each segment's or range's
    rows, in row order, by a {!combine}:
    - [Add] sums from [+0], so a sum of [-0.] rows is [+0.]; the float formats
      narrower than 32 bits sum at [float32] and round once, and integers
      wrap. Every dtype but booleans.
    - [Max] and [Min] keep the greatest or least row as {!maximum} orders
      them, or a NaN where a row's is one, as {!max} does. Every dtype.
    - [Set] keeps the last row.

    A segment or range of no row holds [Add]'s and [Set]'s zero, or [Max]'s
    least and [Min]'s greatest value in the dtype's order: [neg_infinity] and
    [infinity], [-448.] and [448.] in [float8_e4m3fn], which has no infinity,
    the least and greatest integers, [false] and [true], and for complex
    numbers both parts so. *)

val reduce_segments :
  combine -> segments:int -> 'd int64_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [reduce_segments c ~segments ids x] is [segments] rows, row [s] the rows [i]
    of [x] with [ids.{i} = s] folded by [c]. [ids] holds one id per row: give
    a row that belongs to no segment the id [-1]. It is
    [scatter ~combine:c ~axis:0] of [x]'s rows into [segments] rows of the
    identity, so its sums have {!scatter}'s bits.

    {[
    let ids = Nx.create Nx.int64 [| 5 |] [| 0L; 2L; 0L; -1L; 2L |]
    let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    (* [|4.; 0.; 7.|] and [|3.; neg_infinity; 5.|] *)
    let sums = Nx.reduce_segments Add ~segments:3 ids x
    let maxima = Nx.reduce_segments Max ~segments:3 ids x
    ]}

    Counts are [Add] of ones, a mean is [Add] divided by the counts, and each
    segment's first row is [Min] of the row positions, then a {!take}.

    Raises [Invalid_argument] if [segments] is negative, [x] is 0-d, [ids] is
    not 1-d with one id per row of [x], or [c] is [Add] and [x] is boolean. *)

val reduce_ranges :
  combine ->
  lo:'d int64_t ->
  hi:'d int64_t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [reduce_ranges c ~lo ~hi x] is one row per range, row [i] the rows of [x]
    from [lo.{i}] included to [hi.{i}] excluded folded by [c]. Bounds clip into
    [\[0, n\]] for [n] rows, so a range may reach past either end of [x], and
    one whose clipped [lo] is at or after its clipped [hi] holds the identity.

    A row's bits depend on its range's bounds and rows alone: rows outside it
    never enter it, so a large row outside a range never cancels into its sum.
    A growing window, a running fold, is the ranges from [0] to [i + 1]. Its
    work is [O((n + m) log n)] for [m] ranges, and its memory [O(n + m)].

    {[
    let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    let lo = Nx.create Nx.int64 [| 4 |] [| -2L; 0L; 3L; 4L |]
    let hi = Nx.create Nx.int64 [| 4 |] [| 1L; 3L; 3L; 9L |]
    (* [|1.; 6.; 0.; 5.|] and [|1.; 3.; neg_infinity; 5.|] *)
    let sums = Nx.reduce_ranges Add ~lo ~hi x
    let maxima = Nx.reduce_ranges Max ~lo ~hi x
    ]}

    Raises [Invalid_argument] if [x] is 0-d, [lo] and [hi] are not 1-d of one
    length, or [c] is [Add] and [x] is boolean. *)

val map_segments :
  segments:int ->
  'd int64_t ->
  ('d int64_t -> ('v, 's, 'd) t -> ('w, 'r, 'd) t) ->
  ('v, 's, 'd) t ->
  ('w, 'r, 'd) t
(** [map_segments ~segments ids f x] is [f] applied to each position alone,
    the per-position map: at each position [p] of [ids], the row [f] gives
    [p]'s row of [x] with owner [ids.{p}] where [ids.{p}] is in
    [\[0, segments)], and zeros otherwise, whatever [x] holds at [p]. [f] is
    called once on the rows grouped by id, so what [f] reads by owner is read
    once per block, where a per-position map reads it once per position.

    {b Shapes.} Positions are the indices of [ids], of shape [[| s… |]]. The
    first [ndim ids] axes of [x] broadcast to [[| s… |]], and [p]'s row is [x]
    at [p] after that broadcast, of shape [[| r… |]]. The result has shape
    [[| s…; r'… |]]. Ids may repeat. Results agree with the per-position map
    up to how [f] rounds rows it computes together.

    {b What [f] receives.} [f owners rows] is called exactly once. [rows] is
    [[| g; c; r… |]] at [x]'s dtype, [g] blocks of [c] rows, and [owners] is
    [[| g |]], every owner in range. Every row of block [b] is the row of an
    in-range position whose id is [owners.{b}]: each such position fills one
    row, and the block's other rows repeat rows it already holds. A block no
    position fills takes the owner and row of an in-range position of its
    group. A position whose id is out of range never reaches [f]. In a group
    with no in-range position, every row is zeros. The order of blocks, and
    of rows in a block, is unspecified. [f] returns [[| g; c; r'… |]], with
    [r'] independent of [g] and [c], and computes each row of its result from
    that row and its block's owner alone.

    {b Selection.} Results are selected, never multiplied by a mask: what [f]
    gives a repeated row never reaches the result, and a position out of range
    reaches neither [f] nor the result, whatever its row holds.

    {b Groups and blocks.} Positions are grouped per device. [g] and [c]
    depend on shapes, [segments] and placement, never on the ids' values, so
    the shapes [f] sees never depend on the routing. [c] is [1] when a group
    holds few positions per segment, as one token over many experts: then each
    position is its own block and nothing is sorted. With [segments] zero or
    no positions, [g] is [0] and the result is zeros. [ids] and [x] may be
    split over devices along their first axis only, identically or one of them
    whole on each device; [rows], [owners] and [f]'s result are split along
    their first axis as the positions are.

    {[
    let ids = Nx.create Nx.int64 [| 4 |] [| 1L; -1L; 0L; 1L |]
    let x = Nx.create Nx.float64 [| 4; 1 |] [| 1.; 2.; 3.; 4. |]
    let scale = Nx.create Nx.float64 [| 2; 1; 1 |] [| 10.; 100. |]
    (* [|100.; 0.; 30.; 400.|] *)
    let y =
      Nx.map_segments ~segments:2 ids
        (fun owners rows -> Nx.mul rows (Nx.take ~axis:0 owners scale))
        x
    ]}

    Raises [Invalid_argument] if [segments] is negative, [x] has fewer axes
    than [ids] or its leading axes do not broadcast to [ids]', [ids] or [x] is
    split along another axis or in another number of windows, or [f]'s result
    does not have shape [[| g; c; … |]]. What [f] raises reaches the caller
    as raised, and the operations [f] makes are named by the functions it
    calls. *)

(** {1:structures Structures}

    A structure is an OCaml type whose values hold arrays: a model's parameters,
    an optimiser's state, a cache. {!Ptree} describes one by the function that
    walks it, and {!check} and {!associative_scan} take its description. *)

type ('v, 's, 'd) tensor = ('v, 's, 'd) t
(** [tensor] is {!t}, named for {!Ptree}'s signatures, in which [t] is a
    description. *)

(** Structures of arrays.

    A structure is a type [('a, 'd) t] with one function, [walk], that walks the
    parts of a value with a {!Walk.cursor}: the positions of its parameter ['a],
    its arrays of a fixed type, its {e fixed arrays}, on the set ['d], and the
    data a program depends on (a window, a list's length, a variant's case).
    Every operation here runs that function, so all of them visit the parts in
    one order: maps and folds, casts, placement, the names under which arrays
    are saved and the lists of arrays that interpreters trace.

    A {e description}, [('s, 'd) t], is a structure at one type ['s] whose
    arrays are all on ['d]. A map's function may then close over an array of
    ['d]. Changing the set of a value, as changing a dtype, goes through the
    structure's module: {!place} and {!cast} take one.

    A description is made where it is used. Every description but {!tensor} and
    {!unit} is a function's result, so OCaml fixes its dtype and its set at its
    first use, and a set minted in a local scope has no description outside it.
    A library therefore exports a structure's module, or a walk such as
    {!Rng.walk}, and its users describe the structure at their dtype and set.

    {[
    type 'a linear = { w : 'a; b : 'a option }

    module Linear = struct
      type ('a, _) t = 'a linear

      let walk c { w; b } =
        let open Nx.Ptree.Walk in
        let w = field c "w" leaf w in
        let b = field c "b" (option leaf) b in
        { w; b }
    end

    let params = { w = Nx.ones Nx.float32 [| 4; 2 |]; b = None }
    let linear = Nx.Ptree.instantiate (module Linear)
    let zeros = Nx.Ptree.map linear (fun _ x -> Nx.zeros_like x) params
    let on_gpu = Nx.Ptree.place (module Linear) Gpu.on params
    ]} *)
module Ptree : sig
  (** {1:paths Paths} *)

  (** Paths from a value's root to one of its parts. *)
  module Path : sig
    (** The type for a path's steps. *)
    type seg =
      | Field of string  (** A named part, such as a record's field. *)
      | Index of int
          (** A numbered part: a list's or array's element, a tuple's component,
              a pair's side. *)

    type t
    (** The type for paths. [Some] and a variant's case add no step. *)

    val root : t
    (** [root] is the path of no step. *)

    val v : seg list -> t
    (** [v segs] is the path of [segs], from the root. *)

    val add : seg -> t -> t
    (** [add seg p] is [p] then [seg], in constant time, sharing [p]. *)

    val segments : t -> seg list
    (** [segments p] is [p]'s steps, from the root. *)

    val equal : t -> t -> bool
    (** [equal p q] is [true] iff [p] and [q] have equal steps: the one field
        ["a.b"] and the field [a] then [b] differ, though they print alike. *)

    val to_string : t -> string
    (** [to_string p] is [p]'s steps joined by ["."], as ["blocks.3.fc.b"]: the
        name of the part at [p] in a saved value. The root is [""]. *)

    val pp : Format.formatter -> t -> unit
    (** [pp] formats a path as {!to_string}. *)
  end

  (** {1:walk Writing a structure} *)

  type ('s, 'd) t
  (** The type for descriptions of structures ['s] whose arrays are on ['d]. *)

  (** The functions a [walk] is written with, one per kind of part.

      Each walks one part at the cursor's path. A sub-structure is walked by its
      own [walk], and a variant names its case before its other parts:

      {[
      type ('a, 'd) weight =
        | Float of 'a
        | Mxfp4 of { blocks : 'd uint8_t; scales : 'd uint8_t }

      let walk_weight c =
        let open Nx.Ptree.Walk in
        function
        | Float w ->
            case c "float";
            Float (leaf c w)
        | Mxfp4 { blocks; scales } ->
            case c "mxfp4";
            let blocks = field c "blocks" tensor blocks in
            let scales = field c "scales" tensor scales in
            Mxfp4 { blocks; scales }
      ]} *)
  module Walk : sig
    type ('a, 'b, 'd, 'e) cursor
    (** The type for cursors of a walk that turns the parameter's positions from
        ['a] into ['b] and the fixed arrays from the set ['d] to ['e]. A cursor
        stands at a path of the value walked. *)

    val field :
      ('a, 'b, 'd, 'e) cursor ->
      string ->
      (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
      'u ->
      'w
    (** [field c name walk x] is [walk c' x], [c'] at [c]'s path then
        [Field name]. *)

    val index :
      ('a, 'b, 'd, 'e) cursor ->
      int ->
      (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
      'u ->
      'w
    (** [index c i walk x] is [walk c' x], [c'] at [c]'s path then [Index i].
        Tuples and arrays number their parts with it; an array also reports its
        length with {!int}. *)

    val leaf : ('a, 'b, 'd, 'e) cursor -> 'a -> 'b
    (** [leaf c x] walks [x], a position of the parameter, at [c]'s path. *)

    val tensor :
      ('a, 'b, 'd, 'e) cursor -> ('v, 's, 'd) tensor -> ('v, 's, 'e) tensor
    (** [tensor c x] walks [x], a fixed array, at [c]'s path. A description
        treats it as any other array; {!Payload}'s maps and {!cast} keep it, and
        {!place} places it. *)

    val int : ('a, 'b, 'd, 'e) cursor -> int -> int
    (** [int c n] reports [n] at [c]'s path and is [n]. A bool is reported as
        [0] or [1]. *)

    val case : ('a, 'b, 'd, 'e) cursor -> string -> unit
    (** [case c tag] reports the variant case [tag] at [c]'s path. *)

    val option :
      (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
      ('a, 'b, 'd, 'e) cursor ->
      'u option ->
      'w option
    (** [option walk c x] reports whether [x] is present at [c]'s path and, if
        it is, walks its content with [walk] at the same path. *)

    val list :
      (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
      ('a, 'b, 'd, 'e) cursor ->
      'u list ->
      'w list
    (** [list walk c l] reports [l]'s length at [c]'s path, then walks its
        [i]-th element with [walk] at [c]'s path then [Index i], from the first.
    *)

    val structure : ('s, 'd) t -> ('a, 'b, 'd, 'd) cursor -> 's -> 's
    (** [structure s c x] walks [x], a value of the description [s], at [c]'s
        path: each of its arrays as {!tensor} walks one, each of its reports as
        [s] makes it, at [s]'s paths after [c]'s. [tensor c x] is
        [structure Nx.Ptree.tensor c x].

        A description is at one set, so a walk that calls [structure] is too: it
        makes a description with {!v}, never a structure's module, and {!place}
        cannot move what it walks. A part of a fixed type that moves between
        sets is walked by its own walk, as {!Rng.walk} walks a key:

        {[
        type ('a, 'd) state = {
          params : 'a linear;
          key : 'd Nx.Rng.key;
          step : 'd Nx.int32_t;
        }

        module State = struct
          type ('a, 'd) t = ('a, 'd) state

          let walk c s =
            let open Nx.Ptree.Walk in
            let params = field c "params" Linear.walk s.params in
            let key = field c "key" Nx.Rng.walk s.key in
            let step = field c "step" tensor s.step in
            { params; key; step }
        end
        ]}

        [Nx.Ptree.place (module State) Gpu.on s] places the parameters, the key
        and the step. *)
  end

  (** Structures.

      [walk c x] walks every part of [x] once and rebuilds [x] from what each
      walk returns:
      - each position of the parameter with {!Walk.leaf}, each fixed array with
        {!Walk.tensor}, and each sub-structure with its own [walk];
      - each integer that changes what a program computes (a window, a block
        shape, a layer kind) is reported with {!Walk.int}; other scalar data a
        program depends on belongs in an array;
      - a variant reports its case with {!Walk.case} before its case's other
        parts, with a tag no other case of the type uses;
      - options and lists are walked with {!Walk.option} and {!Walk.list}, which
        report presence and length, so a part with no array still counts;
      - a dictionary reports its size with {!Walk.int} and each key with
        {!Walk.case} before walking the key's value with {!Walk.field};
      - a walk that returns each part unchanged rebuilds an equal value.

      The type checker ensures no position of the parameter is forgotten, since
      {!Walk.leaf} changes its type. A forgotten fixed array, integer or case
      compiles; a test of the structure's {!val-visits} catches it.

      A record literal evaluates its fields in an order OCaml leaves
      unspecified, and every operation runs the same [walk]: write the fields as
      a sequence of [let]s where the order must show.

      ['d] is the set of the fixed arrays: a structure whose every array is at
      its parameter ignores it, as [type ('a, _) t = 'a linear]. *)
  module type S = sig
    type ('a, 'd) t
    (** The type for values with parameter ['a] and fixed arrays on ['d]. *)

    val walk : ('a, 'b, 'd, 'e) Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
    (** [walk c x] walks every part of [x] with [c] and is [x] rebuilt from what
        the walks return. *)
  end

  (** {1:descriptions Descriptions} *)

  val instantiate : (module U : S) -> ((('v, 's, 'd) tensor, 'd) U.t, 'd) t
  (** [instantiate (module U)] is [U] with arrays at its parameter's positions:
      [nest (module U) tensor]. A binding is at one dtype and one set, as every
      description ({!Ptree}), annotated only where nothing else in its
      compilation unit fixes them. *)

  val nest : (module U : S) -> ('s, 'd) t -> (('s, 'd) U.t, 'd) t
  (** [nest (module U) s] is [U] whose parameter's positions are values of [s],
      walked by [s] with their paths after the position's. An optimiser's state
      over a model is nested. *)

  val v : ('a 'b. ('a, 'b, 'd, 'd) Walk.cursor -> 's -> 's) -> ('s, 'd) t
  (** [v walk] is the description whose values [walk] walks: a structure without
      a parameter, on one set, such as a record of parts walked with
      {!Walk.structure}. [walk] keeps the contract of {!S}'s [walk], with no
      position of a parameter. {!place} cannot move its values: a structure that
      moves between sets has a module. *)

  val tensor : (('v, 's, 'd) tensor, 'd) t
  (** [tensor] is one array, at the root. *)

  val unit : (unit, 'd) t
  (** [unit] is [()], which has no part. *)

  val pair : ('a, 'd) t -> ('b, 'd) t -> ('a * 'b, 'd) t
  (** [pair a b] walks a pair's first component with [a] at [Index 0], then its
      second with [b] at [Index 1]. *)

  val option : ('a, 'd) t -> ('a option, 'd) t
  (** [option a] reports presence as {!Walk.option} does and walks the content
      with [a] at the same path. *)

  val list : ('a, 'd) t -> ('a list, 'd) t
  (** [list a] reports the length as {!Walk.list} does and walks the [i]-th
      element with [a] at [Index i]. *)

  (** {1:derived Maps and folds}

      These pass every array its path, in walk order. Each array has its own
      dtype, so their functions are polymorphic; fixed arrays are arrays like
      the others.

      Two values that differ raise [Invalid_argument] with the message
      [fn: p: a in the first value, b in the second], [p] the path of the first
      visit at which they differ and [a] and [b] what each holds there, as
      {!pp_visit} words it, or [nothing] for a value with no more visits. Where
      the two visits are at different paths, [b] reads [b at q], [q] the
      second's path; paths that print alike print as their steps. *)

  val map :
    ('s, 'd) t ->
    ('v 'r. Path.t -> ('v, 'r, 'd) tensor -> ('v, 'r, 'd) tensor) ->
    's ->
    's
  (** [map s f x] is [x] with each array [a] at path [p] replaced by [f p a]. *)

  val donate : ('s, 'd) t -> 's -> 's
  (** [donate s x] is [x] with each array given up by its owner: {!Nx.donate}
      of each, in walk order. A consumer of the arrays, an operation or a
      compiled call, may then write its results into their memory.

      Raises [Invalid_argument] naming [Nx.Ptree.donate] and the path of the
      first dead array, as
      [Nx.Ptree.donate: w: float32 [2] was donated to Nx.copy]. *)

  val map2 :
    ('s, 'd) t ->
    ('v 'r.
     Path.t -> ('v, 'r, 'd) tensor -> ('v, 'r, 'd) tensor -> ('v, 'r, 'd) tensor) ->
    's ->
    's ->
    's
  (** [map2 s f x y] is [x] with each array [a] at path [p] replaced by
      [f p a b], [b] [y]'s array at [p].

      Raises [Invalid_argument] before applying [f] if [x] and [y] have
      different {!val-visits}, with the message of {!section-derived} prefixed
      by ["Nx.Ptree.map2"]; and, after applying [f] to the arrays before [p], if
      the arrays at [p] differ in dtype, as in
      ["Nx.Ptree.map2: w: float32 in the first value, int32 in the second"], or
      if [s] walks one value two ways, which breaks {!S}'s contract. *)

  val fold :
    ('s, 'd) t ->
    ('v 'r. Path.t -> ('v, 'r, 'd) tensor -> 'acc -> 'acc) ->
    's ->
    'acc ->
    'acc
  (** [fold s f x acc] is [f pn an (… (f p1 a1 acc))], [a1], …, [an] [x]'s
      arrays and [p1], …, [pn] their paths. *)

  (** {1:payloads Sets, dtypes and payloads}

      A structure's [walk] also changes the type of its parameter's positions
      and the set of its fixed arrays. These take the structure's module, since
      only a module names its type constructor. *)

  val place :
    (module U : S) ->
    'e Placement.t ->
    (('v, 's, 'd) tensor, 'd) U.t ->
    (('v, 's, 'e) tensor, 'e) U.t
  (** [place (module U) p x] is [x] with each array, fixed ones included, at
      [p], as {!Nx.place}.

      Raises what a walk of [x] raises: {!Rng.walk} refuses a key at a [p]
      that cuts its words' axis. Place such keys apart, at
      [Placement.uncut p] by {!Rng.place}. *)

  val cast :
    (module U : S) ->
    ('w, 'r) dtype ->
    (('v, 's, 'd) tensor, 'd) U.t ->
    (('w, 'r, 'd) tensor, 'd) U.t
  (** [cast (module U) dt x] is [x] with each array at a position of [U]'s
      parameter cast to [dt], as {!Nx.cast}. Fixed arrays are kept. *)

  (** Maps and folds at any payload: a mask, a learning rate per array, a plan,
      dimensions. The positions of the parameter hold the payloads; fixed
      arrays, and those of parts walked with {!Walk.structure}, are kept from
      the first value, and neither passed nor compared. *)
  module Payload : sig
    val map :
      (module U : S) -> (Path.t -> 'a -> 'b) -> ('a, 'd) U.t -> ('b, 'd) U.t
    (** [map (module U) f x] is [x] with each payload [a] at path [p] replaced
        by [f p a], in walk order. *)

    val map2 :
      (module U : S) ->
      (Path.t -> 'a -> 'b -> 'c) -> ('a, 'd) U.t -> ('b, 'd) U.t -> ('c, 'd) U.t
    (** [map2 (module U) f x y] is [x] with each payload [a] at path [p]
        replaced by [f p a b], [b] [y]'s payload at [p].

        Raises [Invalid_argument] before applying [f] if [x] and [y] differ in
        their payloads' paths or in their reports, with the message of
        {!section-derived} prefixed by ["Nx.Ptree.Payload.map2"], and if
        [U.walk] walks one value two ways. *)
  end

  (** {1:arithmetic Arithmetic}

      A value's float and complex arrays are one real vector, each complex
      element two of its coordinates, its real and imaginary parts: the inner
      product of two complex arrays is [Re (conj a · b)]. These functions are
      the vector's linear algebra, the steps of optimisers, integrators and
      samplers written over a structure. An array of another dtype, a counter or
      a key, is not part of the vector: {!dot} and {!norm} skip it, {!scale} and
      {!axpy} keep it.

      The functions of two values raise [Invalid_argument] as {!map2} does, with
      their own name, and where the arrays at one path differ in shape, as in
      ["Nx.Ptree.dot: w: [2; 3] in the first value, [3; 2] in the second"]. *)

  val dot :
    ('s, 'd) t -> (float, 'c) dtype -> 's -> 's -> (float, 'c, 'd) tensor
  (** [dot s dt x y] is the 0-d inner product of [x] and [y] at [dt]: from [0],
      the sum in walk order, over each array [a] of the vector and [y]'s array
      [b] at its path, of [vdot (cast dt a) (cast dt b)] for floats and of the
      same over the real parts, then the imaginary parts, stored in [dt], for
      complex arrays. The [0] it starts from lies whole on each device of the
      placement of [x]'s first array in walk order.

      Raises [Invalid_argument] naming [Nx.Ptree.dot] if [x] holds no
      array. *)

  val norm : ('s, 'd) t -> (float, 'c) dtype -> 's -> (float, 'c, 'd) tensor
  (** [norm s dt x] is [x]'s Euclidean norm, [sqrt (dot s dt x x)] in exact
      arithmetic, computed so that no square overflows: each coordinate is
      divided by their greatest magnitude [m], and [m] multiplies the root of
      the quotients' sum of squares. It computes in [dt]'s carrier, float32 for
      a narrower float, and rounds once to [dt], so it is finite wherever the
      norm is: [+∞] where a coordinate is infinite and none is NaN, NaN where
      one is NaN, [0] of no coordinate. Its constants lie as {!dot}'s [0]
      does.

      Raises [Invalid_argument] naming [Nx.Ptree.norm] if [x] holds no
      array. *)

  val scale : ('s, 'd) t -> (float, 'c, 'd) tensor -> 's -> 's
  (** [scale s k x] is [x] with each array [a] of the vector replaced by
      [k' * a], [k'] [k] cast to [a]'s format, part by part for a complex [a].

      Raises [Invalid_argument] naming [Nx.Ptree.scale] unless [k] is 0-d. *)

  val axpy : ('s, 'd) t -> (float, 'c, 'd) tensor -> 's -> 's -> 's
  (** [axpy s k x y] is [x] with each array [a] replaced by [k' * a + b], [b]
      [y]'s array at its path and [k'] [k] cast to [a]'s format, the product and
      the sum each rounded to it, a complex [a] part by part; and by [b] where
      [a] is not in the vector.

      Raises [Invalid_argument] naming [Nx.Ptree.axpy] unless [k] is 0-d. *)

  (** {1:visits Visits} *)

  (** The type for what a walk reports. *)
  type report =
    | Int of int  (** From {!Walk.int}. *)
    | Case of string  (** From {!Walk.case}. *)
    | Present of bool  (** From {!Walk.option} and {!option}. *)
    | Length of int  (** From {!Walk.list} and {!list}. *)

  (** The type for the steps of a walk. *)
  type visit =
    | Leaf of Path.t  (** An array, at its path. *)
    | Report of Path.t * report  (** A report, at the path it is made at. *)

  val visits : ('s, 'd) t -> 's -> visit list
  (** [visits s x] is the steps of [s]'s walk of [x], in walk order. A
      structure's test compares them with the arrays, integers and cases its
      [walk] must visit. *)

  val pp_visit : Format.formatter -> visit -> unit
  (** [pp_visit ppf v] formats [v] as its path and what is there:
      ["blocks.3.moe: case \"mxfp4\""], ["0.keys: a leaf"],
      ["layers: length 4"], ["bias: Some"], ["window: int 8"]. The root is
      formatted ["the root"]. *)

  (** {1:signatures Signatures of compiled functions}

      A signature has the shape of the function an interpreter compiles:
      [tensor @-> index @-> caches @-> returns (pair tensor caches)]. Each
      argument has its own type, so [tensor @-> tensor @-> …] takes arrays of
      two dtypes. A caller gives up an argument by donating it at the call
      ({!donate}). *)

  (** The type for signatures of functions of type ['f] on the set ['d]. *)
  type (_, 'd) fn = private
    | Returns : ('a, 'd) t -> ('a, 'd) fn  (** A result. *)
    | Arg : ('a, 'd) t * ('b, 'd) fn -> ('a -> 'b, 'd) fn
        (** An argument, then the rest of the signature. *)

  val ( @-> ) : ('a, 'd) t -> ('b, 'd) fn -> ('a -> 'b, 'd) fn
  (** [a @-> f] takes an argument of [a], then is [f]. *)

  val returns : ('a, 'd) t -> ('a, 'd) fn
  (** [returns a] is a result of [a]. *)

  (** {1:flattening Flattening}

      {b Low level.} A value as its arrays and its skeleton: what interpreters
      key compiled programs on and put results back with. *)

  (** Skeletons of values. *)
  module Skeleton : sig
    type t
    (** The type for a value's skeleton: its {!val-visits}. *)

    val equal : t -> t -> bool
    (** [equal k k'] is [true] iff [k] and [k'] have equal visits, their paths
        compared by {!Path.equal}. *)

    val hash : t -> int
    (** [hash k] hashes [k]: equal skeletons hash equal. *)

    val diff : this:string -> t -> that:string -> t -> string option
    (** [diff ~this k ~that k'] is [None] iff [equal k k'], and otherwise the
        message of {!section-derived} for their first differing visit, [this]
        and [that] placing each side: with [~this:"here"] and
        [~that:"in the previous key"],
        ["window: int 8 here, int 4 in the previous key"]. *)
  end

  val flatten : ('s, 'd) t -> 's -> 'd packed list * Skeleton.t
  (** [flatten s x] is [x]'s arrays in walk order and [x]'s skeleton. *)

  val rebuild : ('s, 'd) t -> like:'s -> 'd packed list -> 's
  (** [rebuild s ~like xs] is [like] with its arrays replaced by [xs], in walk
      order. [rebuild s ~like (fst (flatten s like))] is [like].

      Raises [Invalid_argument] naming [Nx.Ptree.rebuild] if [xs] has fewer
      arrays than [like], naming the path of the first left without one; if it
      has more, naming both counts; or if an array has another dtype than
      [like]'s at its position, naming the path and both dtypes. *)
end

val check :
  ('a, 'd) Ptree.t -> 'd bool_t -> 'a -> (int array -> 'a -> exn) -> unit
(** [check s ok data fail] raises [fail i d] if an element of [ok] is [false],
    [i] the index of the first in C order and [d] the structure [data] with each
    array replaced by its element at [i], a 0-d value whole on each device of
    that array's placement ({!scalar_like}). Each array of [data] broadcasts
    to [ok]'s shape; an empty [ok] passes. A check with no data passes
    {!Ptree.unit} and [()].

    It is a {{!section-waiting}check}: on the host it raises at the call, and on
    another set at the next wait on that set. It computes on [ok]'s set.

    [s]'s walk must rebuild [data] with each array replaced by a 0-d one, as
    [fail]'s [d] is built; a walk that checks its arrays' shapes, as {!Rng.walk}
    does, may refuse. [check] tries that rebuild at the call over stand-ins
    that hold no bytes, so that trying computes nothing on the devices. Each
    stand-in is 0-d, of its array's dtype, and whole on each device of that
    array's placement. A walk may read their shapes, dtypes and placements.
    An operation on a stand-in raises ["a value traced by Nx.check reached
    eager execution"], and a read of its elements raises too.

    Raises [Invalid_argument] naming [Nx.check] at the call if an array of
    [data] does not broadcast to [ok]'s shape, or if [s]'s walk raises it over
    the stand-ins, with the walk's message. *)

val associative_scan :
  ?axis:int -> ('s, 'd) Ptree.t -> ('s -> 's -> 's) -> 's -> 's
(** [associative_scan s f x] is the inclusive scan of [f] along [axis] over the
    structure [x]: its element [i] combines [x]'s elements [0] to [i] by [f],
    each earlier run as [f]'s first operand, an element being the structure of
    every array's slice at one index of its [axis]. [axis] defaults to [0] and
    counts from each array's last axis when negative.

    [f] must be associative and elementwise: [f (f a b) c] equals [f a (f b c)],
    and an element of [f a b] depends on [a]'s and [b]'s elements at its own
    index alone. The combinations follow Brent and Kung's circuit: of the
    elements [x₀], …, [xₙ₋₁], the pairs [pₖ = f x₂ₖ x₂ₖ₊₁] are scanned so into
    [q], then [y₀ = x₀], [y₂ₖ₊₁ = qₖ] and [y₂ₖ = f qₖ₋₁ x₂ₖ]. Element [i] is
    thus a tree of [f] whose shape depends on [i] alone, of depth at most
    [max(0, 2⌈log₂(i+1)⌉ - 1)]: the scan of a prefix of [x] is the prefix of the
    scan of [x], bit for bit, where [f] computes each element from its operands'
    elements at its index alone, whatever their slices' extents. For a float [f]
    these are not the left fold's bits. [f] is called at most [2⌈log₂ n⌉] times
    for [n] elements, each time over slices of every array, and combines fewer
    than [2n] pairs of elements in all; [f] needs no identity.

    Other scans are lifts of [f]: an exclusive scan is the inclusive one shifted
    by one along [axis], [f]'s identity first; a reversed scan is the scan of
    [x] flipped along [axis], flipped back; a segmented scan is the scan of
    pairs [(start, v)], [start] [true] at each segment's first element, by
    [fun (s1, v1) (s2, v2) -> (logical_or s1 s2, where s2 v2 (f v1 v2))].

    Raises [Invalid_argument] naming [Nx.associative_scan] if [axis] is not an
    axis of an array of [x], if two arrays of [x] have different extents along
    it, or if [f] returns a structure whose visits, dtypes or shapes differ from
    its operands'. *)

(** {1:random Random numbers} *)

(** Random numbers from one Threefry-2x32 generator.

    A key is the generator's whole state: every draw is a function of its key,
    dtype and shape, the same at every placement. Fresh draws come from fresh
    keys, by {!split} or {!fold_in}. A key is spent once it is used: its own
    draws, its {!split} keys, its {!fold_in} keys and the keys of a scope
    rooted at it all
    come from one stream of blocks under it, so using a key two of these ways
    can repeat bits. A scope lets one key stand for a region: a sampler called
    without [~key] takes the scope's next key. A scope may instead refuse such
    draws ({!refusing}), so that a region draws only from keys it is given.

    A key's two words lie whole on one device: a placement that cuts the
    words' axis holds no key. {!key} makes a host key, which {!place} moves
    to a placement that keeps the words whole. {!split} and {!fold_in}
    compute where their key lies, and {!fold_in_tensor} where its key and
    index lie together. A scope's
    keys are host keys, computed on the host when a draw takes one; no
    interpretation receives that computation. *)
module Rng : sig
  type +'d key = private 'd int32_t
  (** The type for keys, of shape [[|2|]], and batches of keys, of shape
      [[|…; 2|]]: two 32-bit words per key. Arithmetic on a key gives an array,
      which no sampler takes. *)

  (** {1:keys Keys} *)

  val key : int -> host key
  (** [key seed] is the host key of [seed]. Equal seeds give equal keys. *)

  val of_tensor : 'd int32_t -> 'd key
  (** [of_tensor t] is the key, or batch of keys, whose words are [t], for words
      that were saved: the inverse of {!to_tensor}.

      Raises [Invalid_argument] unless [t]'s last axis has extent [2], and
      if [t]'s placement cuts the last axis. *)

  val to_tensor : 'd key -> 'd int32_t
  (** [to_tensor k] is [k]'s words: the inverse of {!of_tensor}, and what the
      coercion of [k] to an array is. *)

  val place : 'e Placement.t -> 'd key -> 'e key
  (** [place p k] is [k] at [p], as {!Nx.place}.

      Raises [Invalid_argument] if [p] cuts the last axis, and as
      {!Nx.place} does. *)

  val walk : ('a, 'b, 'd, 'e) Ptree.Walk.cursor -> 'd key -> 'e key
  (** [walk c k] walks [k], a key or a batch of keys, at [c]'s path as one
      fixed array ({!Ptree.Walk.tensor}), and rebuilds it as {!of_tensor}
      does. A structure holding a key walks it with
      [field c "key" Nx.Rng.walk k]; [Nx.Ptree.v Nx.Rng.walk] describes a key
      alone. *)

  val split : ?n:int -> 'd key -> 'd key array
  (** [split ~n k] is [n] keys derived from [k] ([n] defaults to [2]),
      independent of each other. They are [k]'s blocks [0] to [n - 1], so a key
      split is spent ({!Rng}).

      Raises [Invalid_argument] if [n < 1] or [k] is a batch. *)

  val split_batch : n:int -> 'd key -> 'd key
  (** [split_batch ~n k] is [split ~n k] as one batch of shape [[|n; 2|]]: row
      [i] holds [(split ~n k).(i)]'s words.

      Raises [Invalid_argument] if [n < 1] or [k] is a batch. *)

  val fold_in : 'd key -> int -> 'd key
  (** [fold_in k i] is the key of [k] indexed by [i]: distinct [i] give
      independent keys. A batch of keys gives a batch. *)

  val fold_in_tensor : 'd key -> 'd int32_t -> 'd key
  (** [fold_in_tensor k i] is [fold_in k] of the indices held in [i], for an
      index known only as data. Its shape is [k]'s batch broadcast with [i]'s,
      then [2]; where [i] holds [n] it is [fold_in k n]. It lies at the union
      of [k]'s and [i]'s placements, each key whole on its device
      ({!Nx.section-routing}).

      Raises [Invalid_argument] if the shapes do not broadcast, and as an
      operation does where [k] and [i] cannot be read together. *)

  (** {1:samplers Samplers}

      A sampler draws from [~key], or without it from the scope's next key,
      whose two words it writes into its program as {!full} writes its
      value, so that no bytes move. A sampler without an array parameter
      draws on the host, and its [_at] form at the placement it is given,
      each device computing its window: [placement (uniform_at p dt s)] is
      [p]. An [_at] form raises as its host form does, and raises
      [Invalid_argument] naming itself where [p] cuts an axis the draw's
      shape lacks, and where a device of [p] does not hold [key], naming the
      {!place} to write. A sampler with array parameters draws where they
      lie, reading [key] beside them as an operation reads its operands
      ({!Nx.section-routing}). A key with batch axes [b] draws
      [b] followed by the draw's shape, one draw per key, each the draw that
      key alone gives. Float draws compute at float64 for float64 and at
      float32 for the other floats, and round once to their dtype.

      A donated key reaching a sampler is lent ({!Nx.section-moving}), and
      lives on.

      A sampler's parameters are arrays, and the draw has their broadcast
      shape. Each parameter is checked against its domain: an element outside
      raises [Invalid_argument] naming the sampler, the parameter, the element's
      index and its value, as in
      [Nx.Rng.bernoulli: p at [3] is 1.5, not in [0, 1]]. The check is a
      {{!Nx.section-waiting}check}: it raises at the call for a parameter on the
      host, and for one on another set at the next wait on that set, so that a
      draw never waits for its parameters. Under an
      interpretation it is a [Check] operation, which raises where the
      interpretation computes it. *)

  val bits : ?key:host key -> int array -> host int32_t
  (** [bits shape] is uniformly random 32-bit words: word [j] in C order is word
      [j mod 2] of the generator's block [j / 2].

      Raises [Invalid_argument] if an extent is negative. *)

  val bits_at : 'd Placement.t -> ?key:'d key -> int array -> 'd int32_t
  (** [bits_at p shape] is {!bits}[ shape] at [p]. *)

  val uniform :
    ?key:host key -> (float, 's) dtype -> int array -> (float, 's, host) t
  (** [uniform dt shape] is draws from [\[0, 1)]: multiples of [2{^-p}], each
      equally likely, [p] the largest such that every multiple of [2{^-p}] in
      [\[0, 1)] is a value of [dt]: [dt]'s significand width, from float64's
      53 down to float8_e5m2's 3; float4_e2m1fn's values below 1 are 0 and
      0.5, so its [p] is 1.

      Raises [Invalid_argument] if an extent is negative. *)

  val uniform_at :
    'd Placement.t ->
    ?key:'d key ->
    (float, 's) dtype ->
    int array ->
    (float, 's, 'd) t
  (** [uniform_at p dt shape] is {!uniform}[ dt shape] at [p]. *)

  val normal :
    ?key:host key -> (float, 's) dtype -> int array -> (float, 's, host) t
  (** [normal dt shape] is standard normal draws, by the Box-Muller transform:
      draws [2q] and [2q + 1] in C order are the cosine's and the sine's of one
      transform, of the {!uniform} draws [2q] and [2q + 1] at the compute
      dtype.

      Raises [Invalid_argument] if an extent is negative. *)

  val normal_at :
    'd Placement.t ->
    ?key:'d key ->
    (float, 's) dtype ->
    int array ->
    (float, 's, 'd) t
  (** [normal_at p dt shape] is {!normal}[ dt shape] at [p]. *)

  val exponential :
    ?key:host key -> (float, 's) dtype -> int array -> (float, 's, host) t
  (** [exponential dt shape] is exponential draws of rate 1, by inverting the
      distribution at [1 - u] for a uniform [u]: finite, and never negative.

      Raises [Invalid_argument] if an extent is negative. *)

  val exponential_at :
    'd Placement.t ->
    ?key:'d key ->
    (float, 's) dtype ->
    int array ->
    (float, 's, 'd) t
  (** [exponential_at p dt shape] is {!exponential}[ dt shape] at [p]. *)

  val gumbel :
    ?key:host key -> (float, 's) dtype -> int array -> (float, 's, host) t
  (** [gumbel dt shape] is standard Gumbel draws, of distribution
      [exp (-exp (-x))]: [-log e] for an exponential draw [e] of 53 random
      bits at float64 and 32 otherwise, computed so that [e] keeps its
      relative precision near [0.], so the draws are finite. Added to
      unnormalised
      log-probabilities, its argmax draws from the distribution they
      describe, as {!categorical} does; a softmax in place of the argmax
      gives that draw's relaxed, differentiable form.

      Raises [Invalid_argument] if an extent is negative. *)

  val gumbel_at :
    'd Placement.t ->
    ?key:'d key ->
    (float, 's) dtype ->
    int array ->
    (float, 's, 'd) t
  (** [gumbel_at p dt shape] is {!gumbel}[ dt shape] at [p]. *)

  val randint :
    ?key:host key ->
    ?low:int ->
    high:int ->
    int array ->
    host int32_t
  (** [randint ~low ~high shape] is integers drawn uniformly from
      [\[low, high)] ([low] defaults to [0]): a 64-bit draw times [high - low],
      shifted down by 64 bits, whose biased low part, at most one draw in
      [2{^32}], takes a second draw (Lemire's multiply-shift).

      Raises [Invalid_argument] if [low >= high], a bound is outside int32, or
      an extent is negative. *)

  val randint_at :
    'd Placement.t ->
    ?key:'d key ->
    ?low:int ->
    high:int ->
    int array ->
    'd int32_t
  (** [randint_at p ~low ~high shape] is {!randint}[ ~low ~high shape] at
      [p]. *)

  val bernoulli :
    ?key:'d key -> (float, 's, 'd) t -> 'd bool_t
  (** [bernoulli p] is [true] with probability [p] rounded up to a multiple of
      [2{^-53}], elementwise: 53 random bits, read as an integer, below
      [p 2{^53}].

      Raises [Invalid_argument] if an element of [p] is outside [[0, 1]]. *)

  val gamma : ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [gamma a] is gamma draws of concentration [a] and unit rate, elementwise:
      Marsaglia and Tsang's rejection, below a concentration of 1 through
      [Gamma(a) = Gamma(a + 1) U{^1/a}]. Divide by a rate for the two-parameter
      family.

      It is not exact: a draw takes the first of ten rounds that accepts, each
      accepting more than 95% of proposals, and the one element in about
      [10{^13}] that no round accepts is the distribution's mean from a
      concentration of 1 up, and [(a + 1) U{^1/a}] below.

      Raises [Invalid_argument] if an element of [a] is outside [(0, inf)]. *)

  val beta :
    ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [beta a b] is beta draws on [[0, 1]] of concentrations [a] and [b],
      elementwise over their broadcast shape: [G(a) / (G(a) + G(b))] for two
      independent {!gamma} draws, formed from their logarithms so that two draws
      that underflow keep their ratio. Draw [j] in C order is formed from the
      draws [2j] and [2j + 1] that {!gamma} gives from its key over [a] and [b]
      interleaved. It inherits {!gamma}'s rounds.

      Raises [Invalid_argument] if the shapes do not broadcast or an element of
      [a] or [b] is outside [(0, inf)]. *)

  val von_mises : ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [von_mises k] is von Mises draws of mean direction 0 and concentration
      [k], elementwise, as angles in [[-π, π]]: uniform on the circle at
      [k = 0], near a normal of variance [1 / k] for a large [k].

      It is not exact: Best and Fisher's rejection from a wrapped Cauchy
      envelope runs twenty rounds, and the one element in about [2·10{^9}] that
      no round accepts is a draw from the envelope. Add a mean direction,
      wrapped into [\[-π, π\]], for the general family.

      Raises [Invalid_argument] if an element of [k] is outside [\[0, inf)]. *)

  val poisson :
    ?key:'d key -> (float, 's, 'd) t -> 'd int32_t
  (** [poisson rate] is Poisson counts of [rate], elementwise; a rate of 0 gives
      0. Below 10 a count inverts the distribution with one uniform draw, with
      no fallback. From 10 up it is Hörmann's transformed rejection over
      sixteen rounds, and the one element in about [5·10{^9}] that no round
      accepts takes its last proposal, a count near the rate clamped to at
      least 0. A count past int32's largest saturates there. A float32 rate
      places its proposals exactly up to about [10{^5}]; beyond, float32's
      spacing near the rate exceeds a hundredth of a count, so proposals drift
      off their counts and the distribution is biased: give a float64 rate.

      Raises [Invalid_argument] if an element of [rate] is outside
      [\[0, 2{^31})]. *)

  val binomial :
    ?key:'d key ->
    'd int32_t ->
    (float, 's, 'd) t ->
    'd int32_t
  (** [binomial n p] is the number of successes in [n] trials that each succeed
      with probability [p], elementwise over their broadcast shape: 0 at
      [p = 0], [n] at [p = 1]. Where the mean of the rarer outcome is below 10 a
      count inverts the distribution with one uniform draw, with no fallback;
      from 10 up it is Hörmann's transformed rejection over eighteen rounds,
      and the one element in about [4·10{^9}] that no round accepts takes its
      last proposal, clamped into [\[0, n\]]. A float32 [p] places its
      proposals exactly up to a mean of about [10{^5}]; beyond, float32's
      spacing near the mean exceeds a hundredth of a count, so proposals drift
      off their counts and the distribution is biased: give a float64 [p].

      Raises [Invalid_argument] if the shapes do not broadcast, an element of
      [n] is negative, or one of [p] is outside [[0, 1]]. *)

  val truncated_normal :
    ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [truncated_normal lower upper] is standard normal draws conditioned on
      [[lower, upper]], elementwise over their broadcast shape. The bounds may
      come in either order, an infinite one leaves its side open, and equal
      ones give the bound. Draw [j] in C order is the [x] with
      [Φ(x) = (1 - u) Φ(lower) + u Φ(upper)], [Φ] the normal distribution
      function ({!ndtr}), and [u] {!uniform}'s draw [j] from the same key at
      float64 for float64 and at float32 otherwise, floored at its least
      positive value. It inverts the smaller of the masses below and above
      [x], held as logarithms in the tails, so that bounds far out on one side
      keep their digits, and its cost does not depend on the interval's mass.
      A draw lies in the interval, and at float32 within [6·2{^-23}] plus 3
      units in the last place of the exact solution. The masses at each bound
      are computed once per element of the bound as given, so a scalar bound
      costs one.

      Raises [Invalid_argument] if the shapes do not broadcast or an element of
      [lower] or [upper] is NaN. *)

  val dirichlet : ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [dirichlet a] is Dirichlet draws of concentrations [a] along its last
      axis, one draw per row: the {!gamma} draws of [a] from the same key, each
      row divided by its sum. The quotient is formed from the draws'
      logarithms, so that a row of concentrations so small that its gamma draws
      underflow keeps their ratios: each row sums to 1 and has an element of at
      least [1 / n], [n] its length, up to rounding. It inherits {!gamma}'s
      rounds.

      Raises [Invalid_argument] if [a] has no axis, or an element of [a] is
      outside [(0, inf)]. *)

  val categorical : ?key:'d key -> ?axis:int -> (float, 's, 'd) t -> 'd int64_t
  (** [categorical logits] is a position along [axis] of each row of [logits],
      drawn with probability proportional to the exponential of its logit:
      [axis] is the last by default, a negative one counted from the end. A
      logit of [-inf] is never drawn. The result has [logits]' shape without
      [axis]; broadcast [logits] for several draws of one row.

      It is the {!argmax} of the logits plus Gumbel noise [-log e], computed at
      float64 for float64 and float32 for the other floats, [e] an exponential
      draw formed from [b] random bits, 53 at float64 and 32 otherwise, that
      keeps its relative precision near 0. The noise is bounded, so a category
      whose logit lies more than [(b + 1) ln 2 + ln ((b + 1) ln 2)] below its
      row's largest, about 26 at float32 and 41 at float64, is never drawn:
      its probability is below [e{^-26}] or [e{^-41}].

      Raises [Invalid_argument] if [axis] is not an axis of [logits] or has no
      element, or if a row's largest logit is not finite: [+inf], NaN, or a row
      of [-inf]. *)

  val permutation : ?key:host key -> int -> host int64_t
  (** [permutation n] is a uniformly random ordering of [0], …, [n - 1]: the
      positions that sort [n] random 64-bit words. Two words collide, and
      their positions keep their order, with probability below
      [n{^2} / 2{^65}].

      Raises [Invalid_argument] if [n] is negative. *)

  val permutation_at : 'd Placement.t -> ?key:'d key -> int -> 'd int64_t
  (** [permutation_at p n] is {!permutation}[ n] at [p]. A sort reads its
      axis whole, so where [p] cuts it each device of [p] draws and sorts
      all [n] words and keeps its window. *)

  (** {1:scope Scope} *)

  val with_key : host key -> (unit -> 'a) -> 'a
  (** [with_key k f] runs [f] in a scope rooted at [k]: the samplers called in
      [f] without [~key] take successive keys of [k], [fold_in k 0],
      [fold_in k 1], …. Scopes nest, the inner replacing the outer. A scope is
      per fiber and per domain.

      Raises [Invalid_argument] if [k] is dead or traced: a scope's keys are
      host values computed outside any interpretation. *)

  val refusing : string -> (unit -> 'a) -> 'a
  (** [refusing name f] runs [f] in a scope that gives no keys. Scopes nest
      as {!with_key} says: a scope inside [f] gives its own keys.

      A sampler called in [f] without [~key], and {!next_key}, raise
      [Invalid_argument] naming themselves and [name], as
      [Nx.Rng.normal: no key inside Rune.scan, which refuses keyless draws]. *)

  val next_key : unit -> host key
  (** [next_key ()] is the scope's next key, a host key computed at the call.
      Outside every scope, it is the next key of the domain's own, seeded from
      the system's entropy.

      Raises [Invalid_argument] inside {!refusing}. *)
end

(** {1:printing Printing} *)

val pp : Format.formatter -> ('v, 's, 'd) t -> unit
(** [pp ppf x] prints [x]'s dtype and shape, [on] its set off the host, then
    its elements as nested rows, as [float32 [2; 3] [[1, 2, 3], [4, 5, 6]]].
    Elements print as {!Dtype.pp_value} does: a float in the shortest form that
    reads back to it in its format, an unsigned integer unsigned. Past 1000
    elements an axis longer than 6 shows its first and last 3 around [...]. A
    value with elements is read as {!to_array} reads it, so it raises a failed
    check pending on its set ({!section-waiting}).

    A traced value prints [traced by] its interpretation; a dead one the
    function it was donated to; one on a lost device the loss. None of these
    raises. *)

val to_string : ('v, 's, 'd) t -> string
(** [to_string x] is [x] printed by {!pp}. *)

val print : ('v, 's, 'd) t -> unit
(** [print x] prints [x] by {!pp} on standard output, then a newline. *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape ppf s] prints [s] as nx's messages do: [[2; 3]]. *)

val pp_dtype : Format.formatter -> ('v, 's) dtype -> unit
(** [pp_dtype] is {!Dtype.pp}: [float32]. *)

(** {1:errors Errors}

    Every misuse raises [Invalid_argument] whose message is the function the
    user called, a colon, the reason and, where they matter, the operands as
    [dtype shape on set], as in
    [Nx.add: shapes [2; 3] and [4] do not broadcast]. The function is the one
    the user called whatever functions it computes through. A function the
    user gives, as {!init}'s [f] or an interpretation's rule, keeps its own
    errors: what it raises reaches the caller as raised. A lost device
    raises {!Rig.Lost}, and a device's memory that runs out
    {!Rig.Out_of_memory}. Host memory that runs out raises OCaml's
    [Out_of_memory]. *)

(** {1:repr Arrays}

    The crossing to the array layer, for libraries that read or make bytes:
    formats, C bindings, compilers. Layouts and buffers are visible here and
    nowhere else in this module. It shares memory, copying nothing: a value and
    the arrays crossing with it read the same bytes, so a caller who writes them
    breaks "nothing changes a value". {!copy} gives a value memory of its own.
*)

module Repr : sig
  val of_array : 'd devices -> ('v, 's) Nx_array.t -> ('v, 's, 'd) t
  (** [of_array s a] is the value whose bytes are [a]'s. The value reads [a]'s
      memory from then on: a write to it through any handle leaves what values
      over it read unspecified. The memory stays its caller's: no donation of
      the value writes it ({!donate}).

      Raises [Invalid_argument] naming [Nx.Repr.of_array] unless [a] lies on a
      device of [s]. *)

  val unsafe_of_array : 'd devices -> ('v, 's) Nx_array.t -> ('v, 's, 'd) t
  (** [unsafe_of_array s a] is {!of_array}[ s a] for an [a] its caller gives
      up: the value owns [a]'s memory, as it owns memory nx allocated for it,
      so a donation of the value may write its result there ({!donate}).

      {b Warning.} After the call, the caller must not read, write or keep
      [a], an array over its memory or a bigarray over it, and no other value
      must read that memory: a donation of the value would change what they
      read.

      Raises [Invalid_argument] naming [Nx.Repr.unsafe_of_array] unless [a]
      lies on a device of [s]. *)

  val array : ('v, 's, 'd) t -> ('v, 's) Nx_array.t option
  (** [array x] is [Some a] iff [x] lies on one device, [a] its array, and
      [None] for a value on several devices. [a] may be strided or offset. It
      is for reading: writing through it changes [x].

      Raises [Invalid_argument] naming [Nx.Repr.array] for a dead or a traced
      [x]. *)

  val read : ('v, 's, 'd) t -> (('v, 's) Nx_array.t -> 'a) -> 'a
  (** [read x f] is [f a], [a] an array on the host of [x]'s elements for the
      extent of [f]: over [x]'s own memory where [x] lies on one device whose
      memory the host addresses (the host, a memory device, Metal), and
      otherwise a copy on the host, made for any placement, including a value
      cut across devices. It is a {{!section-reads}read}: it waits for [x]'s
      set and raises a failed check pending on it. [a] may be strided or
      offset.

      Unlike {!array}, it leaves [x]'s memory unshared, so a donation of [x]
      after [f] returns may write its result there ({!donate}). [a]'s memory
      is claimed for the extent of [f]: a donation of [x] meanwhile computes
      into fresh memory. [f] must not write through [a], and must not keep
      [a], or a bigarray over it, after it returns. [x] stays reachable until
      [f] returns.

      Raises [Invalid_argument] naming [Nx.Repr.read] for a dead or a traced
      [x], or if [x]'s memory is held exclusive, and a failed check as a wait
      does. *)

  val of_shards :
    'd Placement.t -> ('v, 's) Nx_array.t iarray -> ('v, 's, 'd) t
  (** [of_shards p arrays] is the value whose bytes are [arrays], one per
      device of [p] in order, each that device's window of one shape: along an
      axis [p] cuts, the sum of its windows' extents, and along any other,
      each array's.

      Raises [Invalid_argument] naming [Nx.Repr.of_shards] unless there is
      one array per device of [p], each on its device, all of one rank that
      has every axis [p] cuts, and each array is its device's window of that
      shape. *)

  val shards : ('v, 's, 'd) t -> ('v, 's) Nx_array.t iarray
  (** [shards x] is [x]'s arrays, one per device of [x]'s placement, in
      order: [[| a |]] for a value on one device. The arrays are for reading,
      as {!array}'s.

      Raises [Invalid_argument] naming [Nx.Repr.shards] for a dead or a
      traced [x]. *)
end

(**/**)

(** Operations as data, for interpreters such as rune's.

    An operation goes to the innermost interpretation around it that reaches
    it. An interpretation is around the operations of its extent: those applied
    by the function {!interpret} calls, on whatever domain it resumes, and by
    the fibers that a scheduler installed in the extent runs on the domain the
    extent runs on. Outside the extent are a systhread or a domain started in
    it, a fiber forked by a scheduler installed outside the interpretation, and
    a fiber that a scheduler installed in the extent resumes on another domain:
    there, an operation on a traced value raises, and an [Extent]
    interpretation reaches none of the other operations. An [Adopting]
    interpretation is around its extent on the domain it started on, and on
    another only where the interpretation it adopts is around too.
    Every interpretation reaches the operations on its traced values; an
    [Extent] interpretation reaches every operation of its extent, and an
    [Adopting] one those on the values of the interpretations it adopts. An
    operation no interpretation reaches computes now on its operands' set's
    kernels, and raises if the set has none, but a check of one flag that has
    bytes ([Check]).

    An interpretation's rule receives each operation whole, a [Map] of
    several nodes or outputs included, and gives it its whole meaning,
    evaluating pieces it builds, as one-node maps from a [Map]'s program,
    through {!eval} on the interpretations around its own.

    A rule runs outside its own interpretation and outside every handler
    installed in its extent, with the extent suspended until it returns: its
    operations go to the interpretations around its own, never to its own,
    and an effect it performs reaches the handlers around the interpretation.
    So a rule is never entered while it runs, and its state needs no lock. A
    rule that two interpretations share, as an adopting one may share its
    parent's, is entered for one while it runs for the other: by an operation
    it applies, or while its fiber is suspended, never on two domains at
    once.

    An operation on a traced value raises [Invalid_argument] naming [by] and
    the interpretation where the interpretation is not around it: after it
    returned, in its own rule, or outside its extent.

    Every value has a placement, a traced one its form's. A creation is a
    [Map] of no loads that names where it lies, so an [Extent]
    interpretation's rule receives it at that placement, which {!results}
    gives its result. *)
module Prim : sig
  type ('v, 's, 'd) nx := ('v, 's, 'd) t

  type ('v, 's, 'd) form = {
    dtype : ('v, 's) dtype;
    layout : Nx_array.Layout.t;
    placement : 'd Placement.t;
  }
  (** A value without its bytes. A value cut over devices has the C-contiguous
      layout of the whole. *)

  type 'd any = Any : ('v, 's, 'd) nx -> 'd any

  type 'd load =
    | Plain : ('v, 's, 'd) nx -> 'd load
        (** How a loop reads an operand: through its layout. *)
    | Padded : {
        x : ('v, 's, 'd) nx;
        fill : 'v;
        pad : Nx_kernel.pad;
      }
        -> 'd load
        (** [x] padded with [fill] and windowed as [pad] says
            ({!Nx_kernel.load}): the loop's shape is the padded one. *)

  type ('d, 'e) data =
    | [] : ('d, unit) data
    | ( :: ) : ('v, 's, 'd) nx * ('d, 'e) data -> ('d, 'v * 'e) data
        (** A check's data: values whose elements at one index its failure
            receives, in order. *)

  (** What a loop's reduction makes of one output of its program. It computes
      in the output's carrier ({!Nx_kernel.carrier}). *)
  type ('d, _) reduction =
    | Monoid :
        Nx_kernel.monoid * int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx) reduction
        (** [Monoid (m, k, dt)] folds output [k] by [m] in its carrier and
            rounds once to [dt]. *)
    | Logsumexp : int * ('v, 's) dtype -> ('d, ('v, 's, 'd) nx) reduction
        (** [Logsumexp (k, dt)] is [log Σ exp] of output [k], in its carrier,
            rounded once to [dt]. *)
    | Moments :
        int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx * ('v, 's, 'd) nx) reduction
        (** The population mean of output [k], then its variance. *)
    | Arg :
        Nx_kernel.extreme * int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx * (int64, Dtype.int64_elt, 'd) nx) reduction
        (** The extreme of output [k], then its first position in C order of the
            reduced indices. *)

  type ('d, 'r) reductions =
    | [] : ('d, unit) reductions
    | ( :: ) :
        ('d, 'a) reduction * ('d, 'r) reductions
        -> ('d, 'a * 'r) reductions
        (** A reduction's reductions, its results in order. *)

  type ('d, 'r) outs =
    | [] : ('d, unit) outs
    | ( :: ) : ('v, 's) dtype * ('d, 'r) outs -> ('d, ('v, 's, 'd) nx * 'r) outs
        (** The dtypes of a map's results. *)

  type ('d, _) fft =
    | C2c : {
        direction : Nx_kernel.direction;
        axes : int iarray;
        x : (Complex.t, 's, 'd) nx;
      }
        -> ('d, (Complex.t, 's, 'd) nx) fft
        (** The transform [direction] of [x] along each of [axes]. *)
    | R2c : {
        dtype : (Complex.t, 'c) dtype;
        axes : int iarray;
        x : (float, 's, 'd) nx;
      }
        -> ('d, (Complex.t, 'c, 'd) nx) fft
        (** The forward transform of a real [x] along each of [axes], bins [0]
            to [n/2] along the last; [dtype] has [x]'s precision. *)
    | C2r : {
        dtype : (float, 'r) dtype;
        n : int;
        axes : int iarray;
        x : (Complex.t, 's, 'd) nx;
      }
        -> ('d, (float, 'r, 'd) nx) fft
        (** The real inverse transform of [n] points along the last of [axes],
            from [n/2 + 1] bins; [dtype] has [x]'s precision. *)

  (** Each constructor is one routine of {!Nx_kernel.S}, its results in that
      routine's order, each of [a]'s dtype but [Lu]'s positions and the
      results of a [dtype] field, which is of [a]'s
      {!Nx_kernel.same_precision}. *)
  type ('d, _) linalg =
    | Cholesky : {
        triangle : Nx_kernel.triangle;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx) linalg
    | Lu :
        ('v, 's, 'd) nx
        -> ( 'd,
             ('v, 's, 'd) nx
             * (int64, Dtype.int64_elt, 'd) nx
             * (int64, Dtype.int64_elt, 'd) nx )
           linalg
        (** [lu], [pivots], [perm]. *)
    | Qr : {
        factors : Nx_kernel.factors;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx * ('v, 's, 'd) nx) linalg  (** [q], [r]. *)
    | Svd_values : {
        dtype : (float, 'r) dtype;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, (float, 'r, 'd) nx) linalg
        (** [s] alone, of [dtype]. *)
    | Svd : {
        factors : Nx_kernel.factors;
        dtype : (float, 'r) dtype;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, (float, 'r, 'd) nx * ('v, 's, 'd) nx * ('v, 's, 'd) nx) linalg
        (** [s] of [dtype], [u], [vh]. *)
    | Eigh_values : {
        dtype : (float, 'r) dtype;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, (float, 'r, 'd) nx) linalg
        (** [w] alone, of [dtype]. *)
    | Eigh : {
        dtype : (float, 'r) dtype;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, (float, 'r, 'd) nx * ('v, 's, 'd) nx) linalg
        (** [w] of [dtype], [v]. *)
    | Eig_values : {
        dtype : (Complex.t, 'c) dtype;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, (Complex.t, 'c, 'd) nx) linalg
        (** [w] alone, of [dtype]. *)
    | Eig : {
        dtype : (Complex.t, 'c) dtype;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, (Complex.t, 'c, 'd) nx * (Complex.t, 'c, 'd) nx) linalg
        (** [w], [v], of [dtype]. *)
    | Solve_triangular : {
        triangle : Nx_kernel.triangle;
        adjoint : bool;
        unit_diagonal : bool;
        a : ('v, 's, 'd) nx;
        b : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx) linalg

  (** A contraction operand's block scale, an array of its rank whose
      extents divide its own ({!Nx_kernel.type-scale}). *)
  type 'd scale =
    | Exponent of (int, Dtype.int8_unsigned_elt, 'd) nx
        (** Code [e] scales its block by [2^(e - 127)]; [255] makes it NaN. *)
    | Value : ('v, 's, 'd) nx -> 'd scale
        (** Each scale read in its own dtype into the accumulator. *)

  (** The operations. Their arrays are immutable, so an operation once built
      never changes, whoever keeps it. A loop's loads have exactly its
      iteration shape: no operation broadcasts or promotes. *)
  type 'r t =
    | Map : {
        layout : Nx_array.Layout.t;
        prog : Nx_kernel.Prog.t;
        outs : ('d, 'r) outs;
        loads : 'd load iarray;
        at : 'd Placement.t option;
            (** [Some p] exactly for a map of no loads, a creation, which lies
                at [p], each device of [p] computing its window; [None] for a
                map with loads, which lies where its operands put it. *)
      }
        -> 'r t
        (** [prog] at every index of [layout]'s shape, reading load [i] as its
            operand [i]; result [k] is its output [k], laid out as [layout],
            which is C-contiguous. A one-result map is ['v * unit]. Its
            [Coord] nodes read axes of that shape: a program that reads one
            past its rank is refused, as are a map of no loads and [None], one
            with loads and [Some _], and an [at] that cuts an axis the shape
            lacks. *)
    | Reduce : {
        layout : Nx_array.Layout.t;
        axes : int iarray;
        prog : Nx_kernel.Prog.t;
        reductions : ('d, 'r) reductions;
        loads : 'd load iarray;
      }
        -> 'r t
        (** [prog] at every index of [layout]'s shape, as a map's, each
            reduction folding its output along [axes], strictly increasing, as
            {!Nx_kernel.S.reduce} says. Results drop [axes] and are
            C-contiguous. A [Max], [Min] or [Arg] is refused where a result
            has an element and [axes] hold no term, as
            {!Nx_kernel.Shape.reduce} says. *)
    | Scan : {
        layout : Nx_array.Layout.t;
        axis : int;
        prog : Nx_kernel.Prog.t;
        reduction : ('d, 'r) reduction;
        loads : 'd load iarray;
        carry : 'd any option;
      }
        -> 'r t
        (** The reduction of [prog]'s output over the indices along [axis] up to
            each index's, inclusive, as {!Nx_kernel.S.scan} says: a running
            [Moments] is the running mean and variance. Its results have
            [layout]'s shape and are C-contiguous. On one device, or along
            an axis no device cuts, the scan of a prefix of the axis is the
            prefix of the scan, bit for bit. [prog] reads
            coordinates as a map's. With [Some c], of a [Sum], [Prod], [Max]
            or [Min], each slice continues from its element of [c], [S.scan]'s
            carry, of the results' shape but extent [1] along [axis] and of
            the output's carrier ({!Nx_kernel.carrier}). *)
    | Gather : {
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) nx;
        x : ('v, 's, 'd) nx;
      }
        -> ('v, 's, 'd) nx t
        (** [x] read at the positions [idx] holds along [axis]: [idx] has [x]'s
            rank and its extents off [axis], and the result [idx]'s shape. A
            position outside [x]'s axis reads the element of zero bits. *)
    | Scatter : {
        combine : Nx_kernel.combine;
        unique : bool;
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) nx;
        updates : ('v, 's, 'd) nx;
        into : ('v, 's, 'd) nx;
      }
        -> ('v, 's, 'd) nx t
        (** [into] with each update combined at its own index with axis [axis]
            replaced by [idx]'s element there, in C order of [updates]
            ({!Nx_kernel.S.scatter}); [idx] and [updates] have one shape,
            [into]'s off [axis]. [unique] promises distinct targets. *)
    | Sort : {
        axis : int;
        descending : bool;
        k : int option;
        x : ('v, 's, 'd) nx;
      }
        -> (('v, 's, 'd) nx * (int64, Dtype.int64_elt, 'd) nx) t
        (** [x]'s elements along [axis], stably, in {!Nx_kernel.S.sort}'s
            order, reversed where [descending], then their positions along
            [axis]; with [Some k], the first [k] of each slice. *)
    | Fold : {
        shape : int iarray;
        pad : Nx_kernel.pad;
        x : ('v, 's, 'd) nx;
      }
        -> ('v, 's, 'd) nx t
        (** The adjoint of a [Padded] load by [pad] of a value of [shape]
            ({!Nx_kernel.S.fold}). *)
    | Assemble : {
        dtype : ('v, 's) dtype;
        shape : int iarray;
        fill : 'v;
        pieces : (Nx_array.Move.range iarray * ('v, 's, 'd) nx) list;
      }
        -> ('v, 's, 'd) nx t
        (** The value of [shape] whose element at an index is the last piece's
            whose region, a [Slice] of [shape] by its ranges, holds it, and
            [fill] where none does. Each piece has its region's shape. *)
    | Contract : {
        batch : (int * int) iarray;
        contracting : (int * int) iarray;
        acc : Dtype.any;
            (** A dtype {!Nx_kernel.accumulates} takes. *)
        out : ('v, 's) dtype;
        a : ('a, 'b, 'd) nx;
        b : ('c, 'e, 'd) nx;
        init : ('v, 's, 'd) nx option;  (** Of the result's shape. *)
        a_scale : 'd scale option;
        b_scale : 'd scale option;
        segments : (int64, Dtype.int64_elt, 'd) nx Nx_kernel.segments option;
      }
        -> ('v, 's, 'd) nx t
        (** The contraction of [a] with [b] by the pairs [batch] and
            [contracting], from [init], each operand's elements times its
            scale's, each group of [segments] as {!Nx_kernel.type-segments}
            states, computed in [acc] and rounded once to [out]
            ({!Nx_kernel.S.contract}): its result C-contiguous, of the shape
            {!Nx_kernel.Shape.contract} gives. *)
    | Fft : ('d, 'r) fft -> 'r t
        (** A Fourier transform ({!Nx_kernel.S.c2c} and its siblings). *)
    | Linalg : ('d, 'r) linalg -> 'r t
        (** A factorisation or a triangular solve ({!Nx_kernel.S.cholesky} and
            its siblings). *)
    | Copy : ('v, 's, 'd) nx -> ('v, 's, 'd) nx t
        (** The value stored afresh, C-contiguous. *)
    | Move : Nx_array.Move.t * ('v, 's, 'd) nx -> ('v, 's, 'd) nx t
    | Bitcast : ('w, 'r) dtype * ('v, 's, 'd) nx -> ('w, 'r, 'd) nx t
        (** The bits read in another dtype: one width keeps the shape, a
            narrower one appends an axis of the ratio, a wider one consumes a
            trailing axis of it. *)
    | Place : 'e Placement.t * ('v, 's, 'd) nx -> ('v, 's, 'e) nx t
    | Check : {
        ok : (bool, Dtype.bool_elt, 'd) nx;
        data : ('d, 'e) data;
        fail : int array -> 'e -> exn;
      }
        -> unit t
        (** Raises [fail i e] at the first index [i], in C order, where [ok]
            is [false], [e] each of [data]'s elements at [i], in order: OCaml
            values, which belong to no device set. Each value of [data] has
            [ok]'s shape. An empty [ok] passes. It is computed where [ok]'s
            bytes lie: at the call on the host, and otherwise at the next
            {{!section-waiting}wait} on [ok]'s set. A check of one element
            reduces nothing: its flag is that element, read on the device
            that holds it, and its first index is [0], so it needs no
            kernels. *)

  type operands = Operands : 'd any list -> operands

  val name : 'r t -> string
  (** [name op] is [op]'s constructor, as ["Map"]. *)

  val pp : Format.formatter -> 'r t -> unit
  (** [pp] formats an operation: its name, its programs as expressions over its
      operands [x0], [x1], …, and each operand's dtype, shape and placement. A
      program node read more than once prints once, as [nK = …] before the
      outputs, and as [nK] where it is read. *)

  val operands : 'r t -> operands
  (** [operands op] is [op]'s operands in order: a loop's loads, then a scan's
      carry; a gather's [idx] then [x]; a scatter's [idx], [updates] then
      [into]; an assembly's pieces; a contraction's [a], [b], then those it
      has of [init], [a_scale]'s array, [b_scale]'s and [segments]'
      offsets; a triangular solve's [a] then [b]; [Check]'s [ok] then its
      data; the one operand of the others. *)

  val map : ('v 's 'd. ('v, 's, 'd) nx -> ('v, 's, 'd) nx) -> 'r t -> 'r t
  (** [map m op] is [op] with each operand [x] replaced by [m x]. *)

  val form : ('v, 's, 'd) nx -> ('v, 's, 'd) form
  (** [form x] is [x] without its bytes. *)

  val results :
    by:string ->
    ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) nx) ->
    'r t ->
    'r
  (** [results ~by m op] is [op]'s result, its value at position [k] made by
      [m k f], [f] the form eager execution gives it, at the placement its
      route gives ({!section-routing}); [()] for [Check].

      Raises [Invalid_argument] naming [by], before [m] is called, where [op]'s
      operands break its rule or its route, as eager execution raises. *)

  val iter_results : ('v 's 'd. ('v, 's, 'd) nx -> unit) -> 'r t -> 'r -> unit
  (** [iter_results f op r] is [f y] for each value [y] of [op]'s result [r]
      at positions [0], [1], … in turn, the positions {!results} passes its
      maker. A [Check] has none. *)

  (** {1:interpretations Interpretations} *)

  type interpretation
  (** The type for interpretations. *)

  type reach =
    | Values  (** Reaches the operations on its traced values. *)
    | Extent  (** Also reaches every operation of its extent. *)
    | Adopting of interpretation
        (** [Adopting p] also reaches the operations on the traced values of
            [p] and of what [p] adopts, before [p]. *)

  type ('v, 's, +'d) payload = ..
  (** What an interpretation keeps in its traced values. A payload holds values,
      never a function of ['d]. *)

  val interpret :
    name:string ->
    reach ->
    ('r. interpretation -> by:string -> 'r t -> 'r) ->
    (interpretation -> 'a) ->
    'a
  (** [interpret ~name reach rule f] is [f i], [i] a new interpretation that
      gives the operations it reaches the meaning [rule i ~by op], live until
      [f] returns or raises. [name] names it in messages, as ["Rune.grad"].
      [by] is the function the user called, whatever functions that one
      computes through: each operation of {!trace} reaches [rule] as
      ["Nx.trace"]. [rule] is the user's function: the operations it makes are
      named by the functions it calls, and what it raises reaches the caller
      as raised.

      [rule] receives a donated handle ({!Nx.donate}) as a live value over
      its arrays, whose memory is shared from then on, so that [rule] may keep
      it: no later donation writes there. Its donor lives on.

      With [Adopting p], [i] receives the operations on [p]'s values as on its
      own, without asking whether [p] is around them: [p] is around [i]'s
      extent, since [interpret] checks it is around the caller, so an
      operation on [p]'s values costs [i] what one on its own does. Since
      [rule] may share [p]'s state, [i] answers on another domain than the one
      [interpret] was called on only where [p] is around there too, its
      extent resumed there with [i]'s; elsewhere an operation of [i]'s extent
      on a value of [i] or of what it adopts raises as one outside their
      extents. {!payload}[ i] is [None] for [p]'s values: [rule] reads them
      through {!owner}.

      Raises [Invalid_argument] naming [name] and [p], with [Adopting p], if
      [p] is an [Extent], returned, or is not around the caller, as from its
      own rule or on another domain. *)

  val traced :
    interpretation ->
    ('v, 's, 'd) form ->
    ('v, 's, 'd) payload ->
    ('v, 's, 'd) nx
  (** [traced i f p] is a value of form [f] that [i] owns, keeping [p]. Its
      form, payload and owner answer after [i] returns; any operation on it then
      raises. *)

  val twinned :
    interpretation -> ('v, 's, 'd) nx -> ('v, 's, 'd) payload -> ('v, 's, 'd) nx
  (** [twinned i c p] is {!traced}[ i (form c) p] with the {e twin} [c]:
      {!item} of it reads [c] where [i] is around the call and no [Extent]
      interpretation lies between them, so that code [i] runs may branch on
      a value it computes, as an eager derivative reads its primal. Elsewhere
      (after [i] returned, in [i]'s rule, outside its extent, inside an
      [Extent] interpretation [i] is around, which stages the values it
      reaches) [item] raises as for any traced value. Every other read
      raises: an array read out would hold the twin's elements without what
      [i] tracks. Operations on it go to [i] as on any value [i] owns. *)

  val payload : interpretation -> ('v, 's, 'd) nx -> ('v, 's, 'd) payload option
  (** [payload i x] is [Some p] iff [i] owns [x], [p] what it keeps: [None]
      for a value [i] adopts ([Adopting]), whose owner keeps its payload. *)

  val quiet : unit -> bool
  (** [quiet ()] is [true] only if no [Extent] interpretation is live, on any
      domain: an operation that no [Values] or [Adopting] interpretation
      reaches then computes now. It may be [false] with none live: a
      continuation dropped inside an extent keeps it [false] for the life of
      the process. It reads one atomic. *)

  val owner : ('v, 's, 'd) nx -> interpretation option
  (** [owner x] is the interpretation that owns [x], if [x] is traced. *)

  val donated : ('v, 's, 'd) nx -> bool
  (** [donated x] is [true] iff [x] is a handle {!Nx.donate} made, live or
      not. It allocates nothing. *)

  val eval : by:string -> 'r t -> 'r
  (** [eval ~by op] is [op]'s meaning under the rule, as the vocabulary's
      functions apply it. *)

  (** {1:consumers Consumers}

      A {e consumer} is a function of several operations that consumes the
      donated handles among its operands as one operation does ({!Nx.donate}),
      such as a compiled call: the deaths name it, it reads each handle as the
      donor was when it was claimed, and it may write a result into a handle's
      memory. {!Consumer.with_} runs one: it checks every operand, then claims
      the handles among them, which kills their donors for good, runs the
      consumer, and ends the handles when it returns or raises, so that
      memory no other value reads, and that no result took, returns to its
      device once the work on it is done.

      The consumer reads its operands by their position in the list
      {!Consumer.with_} takes. As for an operation, two handles of one donor,
      or a handle and its donor, are read as the donor and kill it once. The
      arrays it reads are its own until it returns: it must not keep one, or
      a value over one, past that. The arrays {!Consumer.take} gives are the
      exception: their memory is the consumer's from then on. *)

  module Consumer : sig
    type 'd t
    (** The type for consumers of operands of brand ['d]. *)

    val with_ : by:string -> 'd any list -> ('d t -> 'a) -> 'a
    (** [with_ ~by xs f] is [f c], [c] a consumer named [by] of the operands
        [xs], whose handles it claimed first. The handles end when [f]
        returns or raises.

        Raises [Invalid_argument] naming [by] and the earlier consumer if a
        value of [xs] is dead, claiming nothing then, and, as a racing
        consumer may, if a donor died to another domain's consumer while it
        claimed, leaving dead the donors it claimed before. Raises what [f]
        raises. *)

    val arrays : 'd t -> int -> Nx_array.any array
    (** [arrays c i] is the arrays of [c]'s operand [i], one per device of its
        placement, in order.

        Raises [Invalid_argument] naming [c]'s [by] for a traced operand, a
        position no operand has, or once {!with_} returned. *)

    val take :
      'd t ->
      int ->
      ('v, 's) dtype ->
      Nx_array.Layout.t ->
      ('v, 's) Nx_array.t option
    (** [take c i dt l] is [Some a], [a] an array of [dt] laid out as [l] over
        the memory of [c]'s operand [i], for a result to be written there:
        where the operand is a handle [c] claimed, on one device, its array of
        [dt] and [l] over all of its memory, no operand at another position
        shares that memory, and no other value reads it. [None] otherwise,
        and for an operand {!take} gave before.

        [a] holds the operand's elements until [c] writes it. The operand's
        arrays and every array made over them are dead from then on, so [c]
        reads the operand through [a] alone, and a [c] that writes a result
        there while it reads the operand must read each element before it
        writes over it, as an elementwise update does.

        Raises as {!arrays} does. *)
  end

  (**/**)

  val coerce : ('v, 's, 'd) form -> ('w, 'r, 'e) nx -> ('v, 's, 'd) nx option
  (** [coerce f x] is [Some x] at [f]'s type iff [x]'s dtype is [f]'s and [x]
      lies on the set [f]'s placement names. Layouts and placements within the
      set are not compared. Each brand names one set, so a value on that set
      is a value of the brand.

      For an interpretation that binds the values it makes where OCaml cannot
      relate their types: one that binds its stand-ins makes each binding at
      the stand-in's placement, and reads it back at the stand-in's form,
      where the check holds. *)

  (**/**)
end

(**/**)
