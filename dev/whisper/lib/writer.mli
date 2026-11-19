(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transcripts as files: plain text, subtitles, tab-separated values and JSON.

    Each function gives a file's contents in one of the formats of Whisper's
    command line, byte for byte as upstream's writers write them. The texts are
    the segments' ({!Transcript.segment}), every segment of every window in
    order, cleared ones included. A text is {e stripped} of the white space
    Python's [str.strip ()] removes at its ends, Unicode's included. A time is
    written from a frame [f], [f / 100] s, in whole milliseconds, as
    [MM:SS.mmm], preceded by [HH:] from one hour on. Subtitles have no cue when
    every segment is cleared, and a cue per segment otherwise, even when the
    first segment is cleared and a later one is not, where upstream's read word
    timings, which the later segment lacks, and raise.

    This writes a transcript [r] of [talk.flac] as subtitles:
    {[
    Out_channel.with_open_bin "talk.srt" (fun oc ->
        output_string oc (Whisper.Writer.srt r))
    ]} *)

val txt : Transcript.t -> string
(** [txt r] is each segment's stripped text on a line of its own. *)

val vtt : Transcript.t -> string
(** [vtt r] is WebVTT: [WEBVTT] and a blank line, then for each segment a cue,
    [START --> STOP] on a line, its stripped text with each [-->] made [->] on
    the next, and a blank line. *)

val srt : Transcript.t -> string
(** [srt r] is SubRip: for each segment, its number from [1] on a line, then
    [START --> STOP] with the hours always written and a comma before the
    milliseconds, then the text as {!vtt} writes it, and a blank line. *)

val tsv : Transcript.t -> string
(** [tsv r] is tab-separated values: a header [start], [end], [text], then for
    each segment its start and stop in whole milliseconds and its stripped text
    with each tab made a space. *)

val json : Transcript.t -> string
(** [json r] is the JSON object of [r] as Python's [json.dump] writes upstream's
    result: its keys [text], [segments] and [language] in that order, each
    segment an object of [id] (its position from [0]), [seek] (its window's),
    [start] and [end] in seconds, [text], [tokens], and its window's chosen
    attempt's [temperature], [avg_logprob], [compression_ratio] and
    [no_speech_prob], then [words], empty, for a cleared segment. Members are
    separated by [", "] and keys from values by [": "]; a float is written in
    the fewest digits that read back to it, as Python's [repr], with [NaN],
    [Infinity] and [-Infinity] for the others; a character outside printable
    ASCII is escaped, beyond U+FFFF as a pair of surrogates. *)
