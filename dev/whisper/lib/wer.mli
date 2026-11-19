(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Word error rates: how far transcripts are from their references, counted in
    words.

    An {e alignment} of a reference's words with a hypothesis' (a transcript's)
    pairs words in order: a pair of equal words is a {e hit}, of unequal ones a
    {e substitution}, a reference word paired with none a {e deletion} and a
    hypothesis word paired with none an {e insertion}. An alignment of least
    edits, substitutions, deletions and insertions together, has as many edits
    as the Levenshtein distance of the two sequences of words. The
    {e word error rate} of a corpus of pairs of a reference and a hypothesis is
    the sum of the edits of their alignments over the sum of their reference
    words: {e corpus} WER, which weighs each utterance by its length, rather
    than the mean of each pair's rate.

    The counts are jiwer's ([jiwer.process_words], whose [wer] is the rate of
    Whisper's evaluation), its default transform splitting the words. Text is
    usually normalised first ({!English.normalize}). *)

type counts = {
  hits : int;
  substitutions : int;
  deletions : int;
  insertions : int;
}
(** The type for the counts of an alignment, or their sums. *)

val words : string -> string array
(** [words s] is the words of the UTF-8 text [s]: each run of two or more
    white-space characters replaced by a space, white space removed at both
    ends, then the text split at each space. A lone tab or newline between two
    words keeps them one word. *)

val align : reference:string array -> hypothesis:string array -> counts
(** [align ~reference ~hypothesis] is the counts of an alignment of least edits
    of the words [reference] with [hypothesis]. Among such alignments it is the
    one that pairs their common first and last words as hits, then, from the end
    of the rest back, prefers a deletion, then an insertion, then a pair of
    words: the counts of rapidfuzz's [Levenshtein.opcodes], which jiwer reads.
*)

val corpus : (string * string) list -> counts
(** [corpus pairs] is the sum over [pairs] of [(reference, hypothesis)] of the
    counts of
    [align ~reference:(words reference) ~hypothesis:(words hypothesis)]. *)

val rate : counts -> float
(** [rate c] is the word error rate of [c]:
    [(substitutions + deletions + insertions) / (hits + substitutions +
     deletions)], the edits over the reference words. With no reference word it
    is [insertions], as jiwer gives it. *)
