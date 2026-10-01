(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Where a call's tokens sit in a cache.

    A cache is any record of {e pools}. A pool of [slots] slots is one tensor of
    [slots] rows whose axis 0 is the slot axis, with no batch axis: slot [s] is
    row [s] of every pool. A cache index says, for one call, which position each
    token holds and which slots hold the positions of its sequence. One
    contiguous run of slots per sequence ({!rows}), paged allocation, a prefix
    shared by two sequences, a forked beam, a tree of draft tokens verified in
    one call ({!draft}) and several sequences packed in one lane ({!packed}) are
    all values of an index, and a layer is the same for each. A model passes the
    index it was given to every layer and never looks inside; a layer calls
    {!extend} on each of its pools and attends under {!mask}. A layer that keeps
    one entry per block of positions reads its pools through {!val-every}, and
    one that attends to a few columns per token through {!select}. {!ptree} is
    an index's structure, which a compiled step's signature names.

    For a reader coming from serving systems: the table is a block table whose
    blocks hold one position each, and the slot a token stores at, the one its
    table names at its position, is what those systems call its slot mapping.

    {1:laws Laws}

    + {b Column is position.} Column [j] of a sequence's row of the table is the
      slot holding position [j], and column [j] of a table of blocks of [m]
      positions ({!val-every}) the slot holding block [j], positions [j * m] to
      [j * m + m - 1]. A column {e stands at} its position, or at its block's
      last. Under a {!draft} the columns are the table's, then the call's
      tokens, each standing at its token's position. Within a row a slot appears
      once. What a layer stores may depend on the position and on every token
      before it (a rotated key above the first block), so two sequences share a
      slot only at the same column, after the same tokens. Every column a token
      sees names a slot that this call or an earlier one stored; a hole inside
      that range reads as a zero key with the weight of a zero score. A column
      is freed and becomes [-1] once it is below every reach still to be fed. A
      layer's reach through a table is the furthest back any of its reads looks:
      its window, its selections, and a closing token's read of its block's
      positions.
    + {b [-1] addresses nothing.} A position of [-1] is padding, a table entry
      of [-1] is an unallocated column, and a lane whose [row] is [-1] is
      padding throughout. A slot outside the pool and a row outside the table
      address nothing either. A store at a slot that addresses nothing is
      dropped and a read of one is zero, as {!Nx.scatter} and {!Nx.take} do at
      an index outside a tensor, so an eager run and a compiled run agree on
      every cache index.
    + {b A token stores at its own column:} the slot its table names at the
      column that stands at its position, or under a {!draft} the slot the draft
      names. In blocks of [m] positions only a block's last token has one; the
      others store nothing. A token whose position is past the last column
      stores nothing.
    + {b Write targets are distinct.} Within a call the targets that address a
      slot are distinct, and a slot another sequence's table names is never one
      of them: a shared slot is one an earlier call wrote. Two tokens aimed at
      one slot leave it holding, element by element, an unspecified one of their
      stores, and an eager and a compiled run may differ there; every other slot
      is exact. These are obligations on the table alone.
    + {b Padding stores nothing.}
    + {b A column no token of its lane sees contributes exactly zero,} whatever
      its slot holds: {!extend} replaces it by zero before any weight multiplies
      it. That covers an unallocated column, a column that stands past every
      position of the lane (under a {!draft}, a table column at or past the
      lane's least position), and a column below the index's {!window} for every
      token of the lane. The zeroing and the {!mask} read one window, the
      index's, so a layer cannot zero with one and mask with another. Under a
      selection ({!select}) this holds per chosen column, and a column the token
      did not choose is not read.
    + {b Values vary, shapes do not.} Positions and tables, the tables of blocks
      included, are tensors: under {!Rune.jit} they are inputs of the step, and
      one compiled program serves every position and every allocation. *)

type t
(** The type for cache indices of [batch] lanes of [seq] tokens each. *)

(** {1:constructors Constructors} *)

val make :
  ?row:Nx.int64_t ->
  ?every:(int * Nx.int64_t) list ->
  pos:Nx.int64_t ->
  table:Nx.int64_t ->
  unit ->
  t
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
    - [every], for a layer that keeps one entry per block of [m] positions
      ({!val-every}): pairs [(m, blocks)], one per [m], where [blocks], of shape
      [[| rows; context_m |]], names at [blocks.(r).(j)] the slot holding block
      [j] of sequence [r], positions [j * m] to [j * m + m - 1], [-1] when none
      is allocated. Its slots are those of the pools read in blocks of [m].

    A token whose position is past the last column of the table it stores
    through stores nothing. One at [context] or more sees every column without
    itself, its output is unspecified, and an engine retires the sequence first.

    Raises [Invalid_argument] if a shape is wrong, an axis is empty, an [m] is
    below [2] or two pairs have one [m]. *)

val rows : ?every:int list -> context:int -> int array -> t
(** [rows ~context lens] gives sequence [b] its own run of [context] slots,
    [b * context] to [b * context + context - 1], and places its [lens.(b)]
    tokens at positions [0] to [lens.(b) - 1]. Lanes are padded on the left to
    the longest, so [seq] is the largest length and the last column is every
    lane's last token; pad the token ids the same way. The matching pools have
    [Array.length lens * context] slots. For each [m] of [every], sequence [b]
    also owns the run of [c = (context + m - 1) / m] block slots from [b * c],
    and the pools read in blocks of [m] have [Array.length lens * c] slots. The
    last block of a run is closed only past the context when [m] does not divide
    it.

    Raises [Invalid_argument] if there is no lane, [context] is not positive, a
    length is negative or exceeds [context], or an [m] of [every] is below [2]
    or appears twice. *)

val whole : ?lens:int array -> batch:int -> seq:int -> unit -> t
(** [whole ~batch ~seq ()] is the index of whole sequences at positions [0] to
    [seq - 1]: nothing is read and nothing is kept, so a layer given it leaves
    its cache as it was. With [lens], lane [b] holds [lens.(b)] tokens padded on
    the left, as in {!rows}.

    Raises [Invalid_argument] if [batch] or [seq] is not positive or a length
    does not fit. *)

val packed : seq:int -> int array array -> t
(** [packed ~seq lens] is the whole index of lanes that each hold several
    sequences one after another: lane [b] holds sequences of [lens.(b).(0)],
    [lens.(b).(1)], ... tokens, padded on the left to [seq] as in {!whole}, and
    each sequence's tokens are at positions [0] to [n - 1]. A token sees the
    tokens of its own sequence at or before its own and no other: under {!mask}
    token [j] of a lane is of token [i]'s sequence when [j - p_j = i - p_i],
    where [p] is the position. Nothing is read and nothing is kept.

    [packed ~seq (Array.map (fun n -> [| n |]) lens)] masks, extends and rotates
    as [whole ~lens ~batch ~seq ()]; packing several sequences in a lane gives
    each the outputs it has alone, up to floating-point reassociation. One
    compiled program serves every packing of [batch] lanes of [seq] tokens.

    Raises [Invalid_argument] if there is no lane, [seq] is not positive, a
    length is negative or a lane's lengths sum past [seq]. *)

val window : int -> t -> t
(** [window w index] is [index] whose tokens see only the last [w] positions at
    or before their own, themselves included. It replaces the window [index]
    had. An index is built without a window; a model whose layers differ passes
    [window w index] to the sliding ones and [index] to the others, over the
    same pools' slots.

    Raises [Invalid_argument] if [w] is not positive. *)

val select : Nx.int64_t -> t -> t
(** [select columns index] is [index] whose token [i] of lane [b] reads only the
    columns [columns.(b).(i)], of shape [[| batch; seq; k |]]: {!extend}'s
    [seen] has shape [[| batch; seq; k; ... |]], entry [c] of a token holding
    the column it chose [c]-th, and {!mask} has shape [[| batch; seq; k |]].
    Both read the selection from the index, as they read its window. A chosen
    column the token does not see (after its position, below the window, outside
    the context) reads as zero and is masked; an unallocated one reads as zero,
    a hole as without a selection. A column chosen twice is read twice. What is
    stored does not change. It keeps [index]'s window and replaces any selection
    [index] had. On a whole index a column is a token of the lane, as in
    {!mask}, or a block of the lane under {!val-every}.

    A layer that attends to a few columns per token, chosen from scores or fixed
    as the last [w] positions, reads [k] rows per token whatever the context,
    where {!extend} without a selection reads [context] per lane.

    Raises [Invalid_argument] if [columns] does not have that shape, [k] is [0]
    or [index] is a {!draft}. *)

val draft : slots:Nx.int64_t -> sees:Nx.bool_t -> t -> t
(** [draft ~slots ~sees index] is [index] whose tokens are a draft verified in
    one call, a tree of guesses for instance, with:

    - [slots], of shape [[| batch; seq |]]: [slots.(b).(i)] is the slot token
      [i] of lane [b] stores at, [-1] for none. The caller takes them from the
      slots no table names.
    - [sees], of shape [[| batch; seq; seq |]]: [sees.(b).(i).(j)] says whether
      token [i] of lane [b] sees token [j] of the same lane. For a tree it is
      [true] at the token's ancestors and the token itself.

    Under a draft the table holds each sequence up to the call, and the columns
    are the table's, then the call's tokens: {!context} is the table's width
    plus [seq], and {!extend}'s [seen] is the table read followed by the call's
    [values]. A token sees the table's columns that stand before its lane's
    least position, and token [j] when [sees] says so; of both, only those that
    stand at or before its position and, under the index's {!window}, within it.
    Draft tokens store nothing in the table's columns.

    After verification the caller keeps a path by naming, in the next call's
    table, the slot of each kept token at the column of its position:
    [table.(r).(p_j) <- slots.(b).(j)]; the other slots are free again, and
    nothing is copied. A draft of one chain of tokens is a chunk, which needs no
    draft: tokens at consecutive positions store at their own columns.

    It keeps [index]'s window and replaces any draft [index] had.

    Raises [Invalid_argument] if [slots] or [sees] does not have that shape, or
    if [index] is whole, reads blocks ({!val-every}) or selects columns. *)

val every : int -> t -> t
(** [every m index] is [index] read in blocks of [m] positions, for a layer that
    keeps one entry per block (a compressed key, a summary): column [j] is the
    slot the index's table of blocks of [m] positions names for block [j],
    positions [j * m] to [j * m + m - 1], and it stands at the block's last
    position. That one rule gives each function its meaning in blocks:

    - {!extend} stores the values of the tokens that close a block, at positions
      [j * m + m - 1], and of no other token.
    - A token at [t] sees block [j] once the block is closed, when
      [(j + 1) * m <= t + 1], and under a window [w] while also
      [j * m + m - 1 > t - w]: {!mask}, the zeroing of {!extend} and {!select}
      read that rule.
    - {!context} is the width of the table of blocks, and {!positions} are the
      tokens' positions, as on [index].

    A pool read in blocks of [m] belongs to that table: its slot axis counts the
    table's slots. An unfinished block is no state: what its entry is made of is
    stored per position, in a pool of the positions' table, and the token that
    closes the block reads it there, through {!select} to bound the read.

    On a whole index the columns are the lane's blocks by position,
    [(seq + m - 1) / m] of them, and [seen] holds the values of the tokens that
    close them; a block the lane does not close reads as zero.

    [every m index] is [index] when [index] already reads blocks of [m]
    positions, so [every 1] of a constructor's index is that index. It keeps
    [index]'s window.

    Raises [Invalid_argument] if [m] is not positive, or, when [index] does not
    read blocks of [m] already, if it reads blocks of another size, selects
    columns, is a {!draft} or {!packed}, or has a table and none for blocks of
    [m] positions. *)

val advance : t -> t
(** [advance index] is the index of the next token of every lane. It keeps
    [index]'s window and {!val-every} and drops its selection and its {!draft}:
    [seq] is [1] and the position is one past the lane's greatest. A lane of
    padding advances to position [0]. It is per lane: where two lanes name one
    sequence it is not the sequence's next position, and the caller sets
    positions itself.

    Raises [Invalid_argument] on a whole index. *)

(** {1:tokens The call's tokens} *)

val batch : t -> int
(** [batch index] is the number of lanes. *)

val seq : t -> int
(** [seq index] is the number of tokens per lane. *)

val context : t -> int
(** [context index] is the number of columns a sequence can hold, the width of
    what {!extend} returns: the table's on an index with one, plus [seq] under a
    {!draft}, and [seq] on a whole index. Under {!val-every} a column is a
    block: the width of the table of blocks, or [(seq + m - 1) / m] on a whole
    index. *)

val positions : t -> Nx.int64_t
(** [positions index], of shape [[| batch; seq |]], is the tokens' positions
    clamped to [0] and the last column of the positions' table ([seq - 1] on a
    whole index), for rotating and for indexing a table of position embeddings:
    padding is [0]. It does not depend on {!val-every}. *)

(** {1:pools Pools} *)

val extend :
  t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t * ('a, 'b) Nx.t
(** [extend index values pool] is [(seen, pool')], with [pool] of shape
    [[| slots; ... |]], of any dtype and width:

    - [pool'] is [pool] with [values], of shape [[| batch; seq; ... |]], stored
      where the call's tokens sit, at the column that stands at each one's
      position (under {!val-every}, only a block's last token has one): one
      {!Nx.scatter}[ ~unique_indices:true] over the tokens, in place on a pool a
      compiled step consumes. Every slot the call does not target is as it was.
    - [seen], of shape [[| batch; context; ... |]], is what those tokens attend
      over, read from [pool']: column [j] is the slot holding position [j], or
      block [j] under {!val-every}, so the call's own stores are in it. A column
      that is unallocated, that stands past every position of its lane, or below
      the index's {!window} for every token of its lane, is zero. Under a
      selection it is each token's chosen columns, of shape
      [[| batch; seq; k; ... |]] (see {!select}). Under a {!draft} the draft's
      tokens store at its slots, and [seen] is the table's columns followed by
      [values].

    On a whole index [seen] is [values], the values of the tokens that close
    each block under {!val-every}, or the chosen ones under a selection, and
    [pool'] is [pool]. Which case applies depends on how [index] was built and
    on no tensor's value, so it holds under {!Rune.jit}.

    Costs are in the call's tokens and in [context], never in [slots], but for
    one limit: compiled, the write stores each token's row on its own, up to 16
    tokens a call, and a call of more, such as a long prefill, rewrites the
    whole pool once per token. Compiled, the slot numbers and the zeroing fuse
    into the gather: the read is one gated load per element. That holds while
    the slot numbers stay an unevaluated expression of the index's tensors;
    forcing them into storage, as {!Nx.contiguous} does, costs a buffer and a
    kernel per pool per layer.

    Raises [Invalid_argument] if [values] does not have that shape. *)

val mask : t -> Nx.bool_t
(** [mask index], of shape [[| batch; seq; context |]], is which columns of
    {!extend}'s [seen] each token sees: those that stand at or before its
    position, and under the index's {!window} only the last of them. On a whole
    index the columns are the call's tokens, and a token sees those of its own
    sequence ({!packed}); padded ones are hidden. Under a {!draft} its last
    [seq] columns are the call's tokens, seen as {!draft} says. A padded token
    sees none. Under a selection it has shape [[| batch; seq; k |]] and says
    which of its chosen columns each token sees. *)

(** {1:structure Structure} *)

val ptree : t Nx.Ptree.t
(** [ptree] is the structure of an index, for the signature of a compiled step
    and for {!Nx.Ptree.map} over its tensors. Its leaves are the index's
    tensors, its tables of blocks, its selection and its draft included: int64,
    and bool for a draft's [sees]. It reports what a compiled program depends on
    beyond them. In walk order:
    - [tokens] reports case ["whole"] for an index built by {!whole}, or
      ["packed"] for one built by {!packed}, then walks its positions at
      [tokens.pos]; or case ["tabled"], then whether [row] is present and, if it
      is, [row] itself at [tokens.row], the positions at [tokens.pos], the table
      at [tokens.table], and at [tokens.blocks] the number of tables of blocks,
      each reporting its block size before its table at [tokens.blocks.]{e i};
    - [every] reports the block size the index reads in ({!val-every}), [1] when
      it reads positions;
    - [window] reports whether a {!window} is set and its size;
    - [columns] reports whether a selection ({!select}) is present and walks it;
    - [draft] reports whether a {!draft} is present and walks its slots at
      [draft.slots] and its [sees] at [draft.sees].

    Two indices share a compiled program only if they agree on all of these
    reports; one that differs, such as another window, compiles its own. *)
