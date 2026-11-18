(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** {!Nx.attention} under {!Rune.jit} against a float64 reference.

    A case draws query rows, keys and values on the host, in float32 rounded
    to its dtype, places them on a set of devices and compiles the call
    there. The reference is the host's eager call of the same rounded values
    in float64, and [M], each element's [Σ p |v|], the same call over [|v|].
    The compiled [o] and [lse] hold nx.mli's bound read with rune.mli's
    constants: the compiled [Nx.logsumexp]'s [ε], and, for a narrow float,
    the weights rounded once to [v]'s dtype. *)

(** The type for masks. *)
type mask =
  | Unmasked
  | Causal
  | Window  (** Causal over the last 8 keys. *)
  | Documents  (** Captured document ids of queries and keys, sorted. *)
  | Positions
      (** Captured positions of queries and keys, as a key cache's: each
          query sees the keys at or before its own position, the first
          query's at the keys' middle. *)
  | Blocks  (** A captured table of visible blocks of 4 by 4 pairs. *)
  | Opaque
      (** Causal through floats, which a tile's ranges cannot class: each
          tile of the diagonal votes. *)
  | Spans
      (** Captured bounds of each query's run of 161 keys, along the lead
          axes and the queries, rising with a jitter, of int64 for an even
          seed and uint16 for an odd one: tiles hidden, shown and in
          part. *)
  | Reach
      (** The keys within a captured scalar's reach of the query,
          [k > q - w]: at an even seed [w] is [2{^62}], wider than every
          position, so that the mask shows every pair; at an odd seed [w]
          is [100] and the mask causal too. *)

type t = {
  dtype : Nx.Dtype.any;  (** float16, bfloat16, float32 or float64. *)
  lead : int;
  hq : int;
  hk : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  page : int option;  (** Paged by pages of this many keys. *)
  holes : bool;  (** Some table entries out of range. *)
  lengths : int array option;  (** One a lead index, within [[0, tk]]. *)
  mask : mask;
  alibi : bool;  (** A score program reading captured slopes. *)
  rising : bool;  (** Scores that rise along the keys. *)
  positive : bool;  (** Values all positive. *)
  nans : bool;
      (** NaN in the values and keys past each row's length, which no row
          reads. *)
  zeros : bool;
      (** Scores 2000 lower at keys before 64 and values [-0] from key 64
          on: where a row sees a key of each, every weighted value is zero
          in the accumulator. Dense keys only. *)
  faint : bool;
      (** Every key's score but the first's 11 below it, whatever the query
          and key, and the first key's values [0]: each other weight is
          about [2{^-15.9}] of the first's, below float16's smallest normal,
          and [o] is their sum alone. Dense keys only. *)
  seed : int;
}
(** The type for cases. *)

val pp : Format.formatter -> t -> unit

val gen : Nx.Dtype.any list -> t Windtrap.Gen.t
(** [gen dtypes] draws cases of [dtypes]: one to three lead indices, heads
    of [1/1], [8/8], [8/2] and [8/1], [tq] and [tk] around the tiles' edges
    ([0] to [2], [7] to [9], [15] to [17], [31] to [33], [63] to [65]), each
    mask, pages of 16 keys and of fewer with entries out of range, lengths of
    [0], [1] and [tk], and the adversarial rows: positive values, rising
    scores, sums of zeros, faint weights. *)

val examples : Nx.Dtype.any list -> t list
(** [examples dts] is cases each draw may miss, of [dts] in turn: a decode
    step of 64 query heads over 8 key heads and 1000 keys, which splits its
    keys; prefills of 300 rows with no mask and rising scores, under
    [Window] with positive values, [Documents], [Blocks] with ALiBi, [Opaque]
    with lengths and NaN past them, and [Spans] at an odd and an even seed;
    zero-size shapes, values of no column, 40 rows over paged keys with
    entries out of range, and keys of 768 dimensions. Then, of [dts]'s
    float32, bfloat16 and float16 where it has them: a float32 prefill of 300
    rows under [Blocks] with ALiBi, whose gradients underflow; gpt-oss's
    decode step under captured positions over 200 keys; 1000 faint weights on
    positive values, which fails the bound where a product drops float16's
    subnormal weights; and a prefill of 300 rows under [Reach]'s causal
    window of 100 keys. Last, of each dtype, a prefill of 128 rows of sums of
    zeros. *)

val kernel_examples : Nx.Dtype.any list -> t list
(** [kernel_examples dts] is cases of a GPU kernel's shapes, of [dts]'s
    bfloat16 and float16 where it has them: keys of 24 dimensions and values
    of 72, a tensor core's step of K; keys and values of 96 and 128
    dimensions, the derivative's kernels of wider heads; rows and keys of 7
    and 9 with a key head a query head, either side of Metal's tiles of 8;
    and prefills of 300 rows under [Reach], whose tiles a tensor core's mask
    shows whole hide pairs by a row's length, or by page entries out of
    range over 320 keys. *)

val heads : Nx.Dtype.any list -> t list
(** [heads dts] is cases of each of [dts] whose keys and values have [36], [64],
    [110], [160], [256] and [1024] dimensions, then [256] over [36] and [64]
    over [1024], each under a mask of its own and with lengths, over two tiles
    of keys and of rows. Dense keys only. *)

val law : ?composed:bool -> (module Nx.Devices) -> t -> unit
(** [law s c] checks [c] compiled on [s]'s devices, and that the call is
    compiled, not evaluated. Where an element of [o] is a sum of zeros,
    every [p |v|] of the reference zero, it is [+0]. Where [composed]
    (default [false]), the reference is the softmax of [c]'s dense scores,
    masked and rewritten, times [v], in float64, of dense keys only, and
    the law makes no demand of {!Windtrap.cover}. *)

val zero_sums : (module Nx.Devices) -> Nx.Dtype.any -> unit
(** [zero_sums s dt] checks that each of 32768 rows of [dt], float32 or
    narrower, compiled on [s]'s devices in a float32 accumulator, is [+0]
    where its keys [0] to [63] score 200 below its keys [64] to [127]: the
    first keys' weights underflow to zero, their values are [-1] and the
    others' [-0]. With so many rows, every target runs each row's keys in
    one kernel, which no combine of key splits follows. *)

val grad_law : ?composed:bool -> (module Nx.Devices) -> vmap:bool -> t -> unit
(** [grad_law s ~vmap c] checks the gradient in the query, keys and values of
    a loss over [c]'s [o] and [lse], compiled on [s]'s devices, against the
    host's eager float64 gradient within the bound rune.mli's numerics
    compose, and that the call is compiled. Where [vmap], the compiled
    gradient is that of each lead index, mapped by {!Rune.vmap}.

    Where [composed], the reference is {!law}'s composition's.

    Raises [Invalid_argument] if [vmap] and [c] has pages, lengths or a
    [Spans] mask, whose arrays vary along the lead axes. *)

val sizes : (module Nx.Devices) -> Nx.Dtype.any -> unit
(** [sizes s dt] checks that the kernels of attention's own that {!grad_law}
    compiles on [s]'s devices for [dt], keys and values of [256] dimensions
    each, list as many loads, stores and matrix products each as with either or
    both of [1024]: a kernel's code is bounded by its tiles. *)

val uncopied :
  ?dims:int -> (module Nx.Devices) -> grad:bool -> Nx.Dtype.any -> unit
(** [uncopied ~dims s ~grad dt] checks that a compiled call of attention over
    [dt] on [s]'s devices, keys and values of [dims] dimensions (default
    [64]), and where [grad] its compiled gradient in the query, keys and
    values of [o] and [lse], compile only kernels of attention's own:
    each result is the storage its kernel wrote, uncopied. *)

val lse_only :
  (module Nx.Devices) -> rows:int -> dims:int -> Nx.Dtype.any -> string list
(** [lse_only s ~rows ~dims dt] checks that a compiled call of attention over
    [rows] batches of 4 heads of 256 rows of [dims] dimensions of [dt], its
    values of no column, gives the [lse] bits of the same call over values of
    [dims] columns, and from an [l̄se] cotangent alone the query's and keys'
    cotangents of that call with [ō = 0], and is the names of the kernels it
    compiles. *)

val apart : (module Nx.Devices) -> Nx.Dtype.any -> unit
(** [apart s dt] checks that a compiled call returning one attention's [o]
    twice, and transposed, on [s]'s devices, returns three arrays of [o]'s
    elements, no two sharing memory. *)

val stacked : ?axis:int -> (module Nx.Devices) -> Nx.Dtype.any -> unit
(** [stacked ~axis s dt] checks that a compiled call stacking two attentions'
    [o] over [dt] on [s]'s devices along [axis] (default [0]) gives the bits
    of each attention compiled on its own, and, along the first axis,
    compiles only kernels of attention's own, each writing its row of the
    stack. *)

val donated_twice : (module Nx.Devices) -> Nx.Dtype.any -> unit
(** [donated_twice s dt] checks that a compiled call returning one attention
    over [dt] on [s]'s devices twice, each result taking the memory of a
    donated argument it does not read, gives the undonated call's bits in
    both, apart. *)

val donated : (module Nx.Devices) -> Nx.Dtype.any -> unit
(** [donated s dt] checks that a compiled call of attention over [dt] on [s]'s
    devices whose result takes the memory of a donated argument it does not
    read compiles only kernels of attention's own, holds the result's bytes
    less than the undonated call, gives the undonated call's bits and, where
    the host computes on the device ({!Rig.runs_on_host}), writes the result
    in the argument's memory. *)
