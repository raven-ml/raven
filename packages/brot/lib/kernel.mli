(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** C kernels.

    The fused native path of loops that stay in OCaml everywhere else: bytecode
    and js_of_ocaml run the OCaml implementations, which are the reference the
    kernels are held to by the native-vs-bytecode dump differential. *)

val available : bool
(** [available] is [true] iff the C kernels can be used: native code. *)

(** {1:byte_level The fused byte-level kernel} *)

type byte_level = {
  lead : Bytes.t;  (** {!Pre_tokenizer.lead_class}. *)
  cache : Bytes.t;
      (** The state's pretoken table; empty when caching is off. *)
  cache_mask : int;  (** The set index mask, [-1] when caching is off. *)
  byte_ids : int array;  (** The 256 ids of single bytes, [-1] for none. *)
  merge_keys : int array;
      (** The merge map's open-addressing keys, [-1] empty. *)
  merge_values : int array;  (** Packed [(rank lsl 21) lor new_id]. *)
  merge_mask : int;  (** The merge map's slot index mask. *)
  len_table : int array;  (** The bytes each id accounts for. *)
  merge : bool;  (** Whether a miss may be merged in C. *)
}
(** Tables of the fused byte-level kernel: the GPT-2 walker over raw bytes, the
    pretoken cache and the byte-level BPE merge, built once per [Bpe.state].
    Field order is the C ABI — [brot_kernels.c] reads the record as its
    [BROT_BL_*] slots; change neither without the other. *)

type reason =
  | Done
  | Spans_full
  | Ids_full
  | Class
  | Encode
      (** Why {!byte_level_encode} returned; constructor order is the C [BROT_*]
          enum. [Done]: the range is exhausted. [Spans_full], [Ids_full]: a
          buffer has no room for the span at the resume position. [Class]: the
          walker met a code point the Unicode table does not classify yet —
          classify {!code_point} and call again. [Encode]: the span the resume
          position opens was appended to the spans but needs OCaml — over 15
          bytes, a byte without an id, a model whose misses C may not merge, or
          a merge whose result stands for other bytes than its entry's. *)

(** {1:cursor The cursor} *)

val cursor : unit -> Bytes.t
(** [cursor ()] is a fresh kernel cursor: 40 bytes the kernel reads its input
    counts from and writes its results to, at the byte offsets [0] span count,
    [8] id count, [16] mark count, [24] resume position and [32] code point
    (only after [Class]). *)

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
    nothing, raises nothing, holds nothing across calls. *)
