(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Where a call's tokens sit in a cache.

    A cache is any record of {e pools}. A pool of [slots] slots is one tensor of
    [slots + 1] rows whose axis 0 is the slot axis, with no batch axis: slot [s]
    is row [s] of every pool, and the last row is scratch: {!pool} builds one. A
    cache index says, for one call, which position each token holds and which
    slots hold the positions of its sequence. One contiguous run of slots per
    sequence ({!rows}), paged allocation, a prefix shared by two sequences and a
    forked beam are all values of an index, and a layer is the same for each. A
    model passes the index it was given to every layer and never looks inside; a
    layer calls {!extend} on each of its pools and attends under {!mask}.

    For a reader coming from serving systems: the table is a block table whose
    blocks hold one position each, and the slot a token stores at, the one its
    table names at its position, is what those systems call its slot mapping.

    {1:laws Laws}

    + {b Column is position.} Column [j] of a sequence's row of the table is the
      slot holding position [j], and within a row a slot appears once. What a
      layer stores may depend on the position and on every token before it (a
      rotated key above the first block), so two sequences share a slot only at
      the same position, after the same tokens. Every column a token sees names
      a slot that this call or an earlier one stored; a hole inside that range
      reads as a zero key with the weight of a zero score. Under a window a
      column is freed and becomes [-1] once it is below every window still to be
      fed.
    + {b [-1] addresses nothing.} A position of [-1] is padding, a table entry
      of [-1] is an unallocated column, and a lane whose [row] is [-1] is
      padding throughout. A slot outside the pool and a row outside the table
      address nothing either. No index into a tensor that a cache index computes
      is out of range, so an eager run and a compiled run agree on every cache
      index.
    + {b The scratch row is never observed.} What addresses nothing is written
      to the last row of a pool, and what is unallocated is read from it and
      replaced by zero. It is last so that slot numbers [0] to [slots - 1] are
      the allocator's: an engine takes [slots] from whoever built the cache,
      never from a pool's extent, and slot [slots] addresses nothing.
    + {b A token stores at its own column:} the slot its table names at its
      position. A token whose position is past the last column stores nothing.
    + {b Write targets are distinct.} Within a call the targets that address a
      slot are distinct, and a slot another sequence's table names is never one
      of them: a shared slot is one an earlier call wrote. Two tokens aimed at
      one slot leave it holding, element by element, an unspecified one of their
      stores, and an eager and a compiled run may differ there; every other slot
      is exact. These are obligations on the table alone.
    + {b Padding stores nothing.}
    + {b A column no token of its lane sees contributes exactly zero,} whatever
      its slot holds: {!extend} replaces it by zero before any weight multiplies
      it. That covers an unallocated column, a column past every position of the
      lane, and a column below the index's {!window} for every token of the
      lane. The zeroing and the {!mask} read one window, the index's, so a layer
      cannot zero with one and mask with another.
    + {b Values vary, shapes do not.} Positions and tables are tensors: under
      {!Rune.jit} they are inputs of the step, and one compiled program serves
      every position and every allocation. *)

type t
(** The type for cache indices of [batch] lanes of [seq] tokens each. *)

(** {1:constructors Constructors} *)

val make : ?row:Nx.int32_t -> pos:Nx.int32_t -> table:Nx.int32_t -> unit -> t
(** [make ~pos ~table ()] is the index of tokens at positions [pos] in sequences
    held at the slots [table], with:

    - [pos], of shape [[| batch; seq |]]: [pos.(b).(i)] is the position of token
      [i] of lane [b] in its sequence, [-1] for padding.
    - [table], of shape [[| rows; context |]]: [table.(r).(j)] is the slot
      holding position [j] of sequence [r], [-1] when none is allocated.
    - [row], of shape [[| batch |]]: the sequence of each lane, [-1] for a lane
      of padding. Without it lane [b] is sequence [b] and [rows] must equal
      [batch]. Several lanes may name one sequence: its tokens, one per lane,
      see each other as they would in one lane.

    A token whose position is [context] or more stores nothing and sees every
    column without itself: its output is unspecified, and an engine retires the
    sequence first.

    Raises [Invalid_argument] if a shape is wrong or an axis is empty. *)

val rows : context:int -> int array -> t
(** [rows ~context lens] gives sequence [b] its own run of [context] slots,
    [b * context] to [b * context + context - 1], and places its [lens.(b)]
    tokens at positions [0] to [lens.(b) - 1]. Lanes are padded on the left to
    the longest, so [seq] is the largest length and the last column is every
    lane's last token; pad the token ids the same way. The matching pools have
    [Array.length lens * context] slots.

    Raises [Invalid_argument] if there is no lane, [context] is not positive, or
    a length is negative or exceeds [context]. *)

val whole : ?lens:int array -> batch:int -> seq:int -> unit -> t
(** [whole ~batch ~seq ()] is the index of whole sequences at positions [0] to
    [seq - 1]: nothing is read and nothing is kept, so a layer given it leaves
    its cache as it was. With [lens], lane [b] holds [lens.(b)] tokens padded on
    the left, as in {!rows}.

    Raises [Invalid_argument] if [batch] or [seq] is not positive or a length
    does not fit. *)

val window : int -> t -> t
(** [window w index] is [index] whose tokens see only the last [w] positions at
    or before their own, themselves included. It replaces the window [index]
    had. An index is built without a window; a model whose layers differ passes
    [window w index] to the sliding ones and [index] to the others, over the
    same pools' slots.

    Raises [Invalid_argument] if [w] is not positive. *)

val advance : t -> t
(** [advance index] is the index of the next token of every lane, with [index]'s
    window: [seq] is [1] and the position is one past the lane's greatest. A
    lane of padding advances to position [0]. It is per lane: where two lanes
    name one sequence it is not the sequence's next position, and the caller
    sets positions itself.

    Raises [Invalid_argument] on a whole index. *)

(** {1:tokens The call's tokens} *)

val batch : t -> int
(** [batch index] is the number of lanes. *)

val seq : t -> int
(** [seq index] is the number of tokens per lane. *)

val context : t -> int
(** [context index] is the number of positions a sequence can hold, the width of
    what {!extend} returns: the table's on an index with one, [seq] on a whole
    index. *)

val positions : t -> Nx.int32_t
(** [positions index], of shape [[| batch; seq |]], is the tokens' positions
    clamped to [0] and [context index - 1], for rotating and for indexing a
    table of position embeddings: padding is [0]. *)

(** {1:pools Pools} *)

val pool : slots:int -> ('a, 'b) Nx.dtype -> int array -> ('a, 'b) Nx.t
(** [pool ~slots dtype shape] is an empty pool of [slots] slots, each of shape
    [shape]: zeros of shape [slots + 1] followed by [shape], the last row being
    the scratch row. Every pool is built here, so nothing else knows about that
    row. [slots] may be [0]: the pool a {!whole} call is given.

    Raises [Invalid_argument] if [slots] is negative. *)

val extend :
  t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t * ('a, 'b) Nx.t
(** [extend index values pool] is [(seen, pool')], with [pool] of shape
    [[| slots + 1; ... |]], of any dtype and width:

    - [pool'] is [pool] with [values], of shape [[| batch; seq; ... |]], stored
      where the call's tokens sit: one {!Nx.scatter}[ ~unique_indices:true] over
      the tokens, in place on a pool donated to {!Rune.jit}. Every slot the call
      does not target is as it was.
    - [seen], of shape [[| batch; context; ... |]], is what those tokens attend
      over, read from [pool']: column [j] is the slot holding position [j], so
      the call's own tokens are in it. A column that is unallocated, past every
      position of its lane, or below the index's {!window} for every token of
      its lane, is zero.

    On a whole index [seen] is [values] and [pool'] is [pool]. Which case
    applies depends on how [index] was built and on no tensor's value, so it
    holds under {!Rune.jit}.

    Costs are in the call's tokens and in [context], never in [slots]. Compiled,
    the slot numbers and the zeroing fuse into the gather: the read is one gated
    load per element. That holds while the slot numbers stay an unevaluated
    expression of the index's tensors; forcing them into storage, as
    {!Nx.contiguous} does, costs a buffer and a kernel per pool per layer.

    Raises [Invalid_argument] if [values] does not have that shape. *)

val mask : t -> Nx.bool_t
(** [mask index], of shape [[| batch; seq; context |]], is which columns of
    {!extend}'s [seen] each token sees: those at or before its position, and
    under the index's {!window} only the last of them. On a whole index the
    columns are the call's tokens, and padded ones are hidden. A padded token
    sees none. *)

(** {1:traversals Traversals}

    Over the index's int32 tensors, for the state of a jitted step. They compute
    nothing and keep the window, which is no tensor. *)

val map : (Nx.int32_t -> Nx.int32_t) -> t -> t
(** [map f index] is [index] with [f] applied to every tensor. *)

val map2 : (Nx.int32_t -> Nx.int32_t -> Nx.int32_t) -> t -> t -> t
(** [map2 f index index'] combines [index] and [index'] tensor by tensor.

    Raises [Invalid_argument] if they were not built the same way or differ in
    their window. *)

val iter : (Nx.int32_t -> unit) -> t -> unit
(** [iter f index] applies [f] to every tensor of [index]. *)
