(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** C kernels.

    The fused native path of loops that stay in OCaml everywhere else: bytecode
    and js_of_ocaml run the OCaml implementations, which are the reference the
    kernels are held to by the native-vs-bytecode dump differential. *)

val available : bool
(** [available] is [true] iff the C kernels can be used: native code on a
    little-endian machine. *)

(** {1:byte_level The fused byte-level kernel} *)

type byte_level = {
  lead : Bytes.t;  (** {!Pre_tokenizer.lead_class}. *)
  front : Bytes.t;
      (** The state's front pretoken table, probed first; empty when caching is
          off. *)
  front_mask : int;
      (** The front table's entry index mask, [-1] when caching is off. *)
  cache : Bytes.t;
      (** The state's back pretoken table; empty when caching is off. *)
  cache_mask : int;  (** The set index mask, [-1] when caching is off. *)
  byte_ids : int array;  (** The 256 ids of single bytes, [-1] for none. *)
  merge_keys : int array;
      (** The merge map's open-addressing keys, [-1] empty. *)
  merge_values : int array;  (** Packed [(rank lsl 21) lor new_id]. *)
  merge_mask : int;  (** The merge map's slot index mask. *)
  len_table : int array;  (** The bytes each id accounts for. *)
  merge : bool;  (** Whether a miss may be merged in C. *)
}
(** Tables of the fused byte-level kernel: a walker over raw bytes — the GPT-2
    one, or the cl100k or o200k one, by the entry called — the pretoken cache
    and the byte-level BPE merge, built once per [Bpe.state]. Field order is the
    C ABI — [brot_kernels.c] reads the record as its [BROT_BL_*] slots; change
    neither without the other. *)

(** {1:sp The fused SentencePiece kernel} *)

type sp = {
  front : Bytes.t;
      (** The state's front pretoken table, probed first; empty when caching is
          off. *)
  front_mask : int;
      (** The front table's entry index mask, [-1] when caching is off. *)
  cache : Bytes.t;
      (** The state's back pretoken table; empty when caching is off. *)
  cache_mask : int;  (** The set index mask, [-1] when caching is off. *)
  merge_keys : int array;
      (** The merge map's open-addressing keys, [-1] empty. *)
  merge_values : int array;  (** Packed [(rank lsl 21) lor new_id]. *)
  merge_mask : int;  (** The merge map's slot index mask. *)
  len_table : int array;  (** The bytes each id accounts for. *)
  ascii_ids : int array;  (** The 128 ids of single ASCII characters, [-1]. *)
  char_keys : int array;
      (** The multi-byte character map's keys ([Bpe.pack_char_key]), [-1] empty;
          the same open-addressing layout as the merge map. *)
  char_values : int array;  (** The multi-byte characters' ids. *)
  char_mask : int;  (** The character map's slot index mask. *)
  scan : Bytes.t;
      (** What each of the 256 byte values opens: [0] nothing, [1 + slot] a
          punctuation split byte, [9] the ["\xE2"] lead of ["▁"]. *)
  punct_safe : bool array;
      (** Whether a unit may end between a previous byte and a slot's byte, at
          [(slot lsl 8) lor prev] — [Bpe.sp_cut]'s table. *)
  fallback_ids : int array;
      (** The 256 byte fallback ids a character without a piece is spelled
          with, [-1] for none and everywhere without byte fallback. *)
}
(** Tables of the fused SentencePiece kernel: the ▁/punctuation unit walker over
    normalized bytes, the pretoken cache probe and the character-level short
    merge, built once per [Bpe.state]. Field order is the C ABI —
    [brot_kernels.c] reads the record as its [BROT_SP_*] slots; change neither
    without the other. *)

type reason =
  | Done
  | Spans_full
  | Ids_full
  | Class
  | Encode
      (** Why {!byte_level_encode} returned; constructor order is the C [BROT_*]
          enum. [Done]: the range is exhausted. [Spans_full]: the span
          buffer has no room for the span at the resume position, or the call
          has walked 1024 spans, after which it returns so that the runtime,
          which cannot stop a domain inside a call, never waits long for one.
          [Ids_full]: the ids buffer has no room for the span at the resume
          position. [Class]: the
          walker met a code point the Unicode table does not classify yet —
          classify {!code_point} and call again. [Encode]: the span the resume
          position opens was appended to the spans but needs OCaml — over 15
          bytes, a byte without an id, a model whose misses C may not merge, or
          a merge whose result stands for other bytes than its entry's. *)

(** {1:cursor The cursor} *)

val cursor : unit -> Bytes.t
(** [cursor ()] is a fresh kernel cursor: the bytes the kernel reads its input
    counts from and writes its results to, at the byte offsets [0] span count,
    [8] id count, [16] mark count, [24] resume position and [32] code point
    after [Class] or {!unit_stop} after [Encode], then the spans the kernel has
    walked ahead of their probes. A call that resumes on the range the last
    one left at the next of those spans resolves them before walking on. *)

val restart : Bytes.t -> unit
(** [restart cur] drops the spans [cur] holds walked ahead, so that the next
    call walks its range from the start. A driver restarts its cursor before
    every walk it begins: a range of another text can start and stop at the
    positions the last one left. *)

val set : Bytes.t -> spans:int -> ids:int -> marks:int -> unit
(** [set cur ~spans ~ids ~marks] writes the counts a call starts from. *)

val spans : Bytes.t -> int
(** [spans cur] is the span count after a call. *)

val ids : Bytes.t -> int
(** [ids cur] is the id count after a call. *)

val marks : Bytes.t -> int
(** [marks cur] is the mark count after a call. *)

val resume : Bytes.t -> int
(** [resume cur] is the position the walk stopped at: [stop] after [Done], the
    start of the span that did not fit or needs OCaml otherwise. *)

val code_point : Bytes.t -> int
(** [code_point cur] is the code point needing a class, after [Class]. *)

val unit_stop : Bytes.t -> int
(** [unit_stop cur] is the end of the unit or span that needs OCaml, after an
    [Encode] from {!sp_encode} or an ids32 entry; its start is {!resume}.
    Shares its cursor word with {!code_point}: a [Class] exit writes the one,
    an [Encode] exit the other. *)

(** {1:entry The entry} *)

val byte_level_encode :
  string ->
  int ->
  int ->
  Bytes.t ->
  int array ->
  int array ->
  Bytes.t ->
  Bytes.t ->
  byte_level ->
  reason
(** [byte_level_encode text pos stop spans ids marks cursor unicode t] walks
    [text.\[pos..stop)] with the GPT-2 byte-level pattern and appends, from the
    counts [cursor] holds: each pretoken span to [spans] ([Spans.buffer]
    layout), its token ids to [ids] and the id count after it to [marks]
    ([Ints.buffer]), until the range is exhausted, a buffer is full or a span
    needs OCaml — see {!reason}; the exit counts and resume position are in
    [cursor]. [unicode] is {!Char_class.unicode_table}, consulted for every
    non-ASCII code point.

    Unchecked, native only: the caller validates
    [0 <= pos <= stop <= String.length text <= 0xFFFF_FFFF], that the counts in
    [cursor] are within the buffers, and that [marks] has room for as many marks
    as [spans] has room for spans; ids room is checked per span. Reads [text] in
    [\[pos..stop)], except the cache key words, whose masked reads stay within
    the whole of [text] exactly as the OCaml key builder's do. Allocates
    nothing, raises nothing, and holds nothing across calls but what [cursor]
    keeps (see {!cursor}). *)

type ids32 = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for the batch id sink of the ids32 entries: token ids are written
    to it as 32-bit integers, at the same positions the int-array entries would
    write them. *)

val byte_level_encode_ids32 :
  string -> int -> int -> ids32 -> Bytes.t -> Bytes.t -> byte_level -> reason
(** [byte_level_encode_ids32 text pos stop ids cursor unicode t] is
    {!byte_level_encode} without spans or marks: the same walk, probes, table
    stores and ids, stored as [int32] into an {!ids32} buffer, [Spans_full]
    only after 1024 spans. On [Encode] the span at [{!resume}..{!unit_stop})] needs
    OCaml and nothing was appended for it. Ids room is checked per span against
    the buffer's length; every other contract is {!byte_level_encode}'s. *)

val split_encode :
  string ->
  int ->
  int ->
  Bytes.t ->
  int array ->
  int array ->
  Bytes.t ->
  Bytes.t ->
  byte_level ->
  int ->
  reason
(** [split_encode text pos stop spans ids marks cursor unicode t walker] is
    {!byte_level_encode} walking with the cl100k or the o200k walker of
    {!Pre_tokenizer.fill} in place of the GPT-2 pattern: the same probes,
    merges, table stores, exits and contract. [walker] is
    [(digits lsl 2) lor (o200k lsl 1) lor flag]: the longest group of numbers,
    [1] for the o200k walker and [0] for the cl100k one, and its flag, whether
    combining marks are letters for the cl100k walker and whether words take
    contractions for the o200k one. [unicode] tells marks and case by
    {!Char_class}'s bits. *)

val split_encode_ids32 :
  string ->
  int ->
  int ->
  ids32 ->
  Bytes.t ->
  Bytes.t ->
  byte_level ->
  int ->
  reason
(** [split_encode_ids32] is {!split_encode} without spans or marks, as
    {!byte_level_encode_ids32} is {!byte_level_encode}. *)

val sp_encode : string -> int -> int -> int array -> Bytes.t -> sp -> reason
(** [sp_encode text pos stop ids cursor t] cuts [text.\[pos..stop)] at
    SentencePiece unit boundaries — before each ["▁"] whose previous character
    is not itself ["▁"] and before each vocabulary-safe punctuation byte,
    exactly as [Bpe.encode_into]'s walker does — and appends each unit's token
    ids to [ids] from the id count [cursor] holds, probing the pretoken cache
    and short-merging misses character by character. Never returns [Class],
    and [Spans_full] only after 1024 units; on [Encode] the unit at
    [{!resume}..{!unit_stop})] needs OCaml — over 15 bytes, or holding a
    character with neither a piece nor a byte fallback for each of its bytes,
    or a merge result [Bpe.emit_word] would record — and nothing was appended
    for it. No span or mark buffers are involved: the caller records the whole
    stretch as one span, as the OCaml walker's caller does.

    Unchecked, native only: the caller validates
    [0 <= pos <= stop <= String.length text] and that the id count in [cursor]
    is within [ids]; ids room is checked per unit. [pos] must be the start of
    the stretch or a unit boundary of it, which every {!resume} and {!unit_stop}
    is. Reads and allocates as {!byte_level_encode} does. *)

val sp_encode_ids32 : string -> int -> int -> ids32 -> Bytes.t -> sp -> reason
(** [sp_encode_ids32] is {!sp_encode} with the ids sink an {!ids32} buffer
    rather than an [Ints.buffer]: the same cuts, probes, table stores and exits,
    the ids stored as [int32]. Ids room is checked per unit against the buffer's
    length; every other contract is {!sp_encode}'s. *)
