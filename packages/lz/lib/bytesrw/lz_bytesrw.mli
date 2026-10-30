(** Bytesrw filters that decompress and compress deflate, zlib, gzip and
    Zstandard streams with {!Lz}.

    A {e decompressing} filter turns a stream ({!Lz}) into its decoded
    bytes, and a {e compressing} filter turns bytes into a stream. A
    {e reader filter} takes a reader [r] and returns a reader of [r]'s
    bytes so turned. A {e writer filter} takes a writer [w] and returns a
    writer that turns the bytes written on it and writes the result on
    [w]. Each format's module has the four filters ({!section-formats}).

    A filter steps its format's {!Lz.Decoder.t} or {!Lz.Encoder.t} over
    the slices it gets, and copies at most 64 KiB of a slice at a time
    into a bigarray, its {e staged input}, for the step. However long the
    stream and its slices, it holds the decoder or encoder, the staged
    input and a copy of one run. Zstandard's window is the only part that
    grows with the input, up to [?window] as for {!Lz.Zstd.decoder}. A
    filter bounds no stream's decoded bytes, so a program that reads a
    stream it does not trust must count the bytes the filter delivers, and
    stop once they pass its limit.

    Unless a value's doc says otherwise, a filter follows bytesrw's
    conventions for filters ({!Bytesrw.Bytes.Reader.filter},
    {!Bytesrw.Bytes.Writer.filter}):
    {ul
    {- The filtering reader's or writer's position starts at [pos]
       (default 0).}
    {- A filtering reader returns slices of at most its [slice_length],
       which defaults to [r]'s.}
    {- A filtering writer's [slice_length] hint defaults to [w]'s, and it
       writes slices of at most [w]'s slice length on [w]. It writes
       [Bytesrw.Bytes.Slice.eod] on [w] after the stream's end if and only
       if [~eod:true].}
    {- Applying a filter raises [Invalid_argument] if [level] or [window] is
       outside the range its format's [encoder] or [decoder] accepts.}}

    This reads the decoded bytes of a [.jsonl.zst] shard, as a reader that
    a JSON parser takes. The filter decodes each frame as the parser reads,
    so it holds at most a window and a run of the decoded bytes at a
    time.
    {[
      let lines ic =
        let r = Bytesrw.Bytes.Reader.of_in_channel ic in
        Lz_bytesrw.Zstd.decompress_reads () r
    ]} *)

(** {1:errors Errors}

    A decompressing filter's read or write raises
    [Bytesrw.Bytes.Stream.Error] with [Error e] when its decoder's step
    returns [Error e] ({!Lz.Decoder}). The error's position is that of
    [e]'s byte in the compressed stream, counted in [r]'s positions for a
    reader and in the filtering writer's for a writer. The message prints
    [e] with {!Lz.pp_error}. The filter delivers the decoded bytes of every
    step before the failing one, and none of that step's.

    An exception raised while the filter steps its decoder or encoder,
    such as one from a write on [w] or [Out_of_memory], propagates. After
    it, or after an error, every later read or write of the filter raises
    it again. Compressing filters raise no [Error]. *)

type Bytesrw.Bytes.Stream.error +=
  | Error of Lz.error
      (** [Error e] is the error [e] that a decompressing filter refused its
          stream with. *)

(** {1:formats Formats}

    A decompressing filter reads the streams its format's [decode] reads.
    A deflate or zlib [decompress_reads] decodes one stream. Once it has
    returned [Bytesrw.Bytes.Slice.eod], [r]'s next read begins at the first
    byte after the stream. A gzip or Zstandard [decompress_reads] decodes
    every member up to [r]'s end.
    [decompress_writes] decodes up to the [eod] written on it. A writer
    cannot hand back bytes written after a deflate or zlib stream, so they
    are [Malformed] at their first byte, with the message
    ["bytes after the stream"].

    [compress_reads] and [compress_writes] write one stream holding every
    byte read from [r] or written on the filter, ended at [r]'s end or at
    the [eod] written. The stream's bytes are those its format's encoder
    writes for the same input, whatever the slices. *)

(** Zstandard frames ({!Lz.Zstd}). *)
module Zstd : sig
  val decompress_reads : ?window:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads ~window () r] is a reader of the decoded bytes of
      [r]'s frames up to its end. [window] is as for {!Lz.Zstd.decoder}. *)

  val decompress_writes : ?window:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes ~window () w] is a writer that decodes the frames
      written on it up to [eod] and writes their decoded bytes on [w].
      [window] is as for {!Lz.Zstd.decoder}. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] is a reader of one frame of [r]'s bytes,
      at [level] as for {!Lz.Zstd.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] is a writer that writes one frame of the
      bytes written on it to [w], at [level] as for {!Lz.Zstd.encoder}. *)
end

(** Gzip members ({!Lz.Gzip}). *)
module Gzip : sig
  val decompress_reads : unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads () r] is a reader of the decoded bytes of [r]'s
      members up to its end. *)

  val decompress_writes : unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes () w] is a writer that decodes the members written
      on it up to [eod] and writes their decoded bytes on [w]. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] is a reader of one member of [r]'s
      bytes, at [level] as for {!Lz.Gzip.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] is a writer that writes one member of
      the bytes written on it to [w], at [level] as for
      {!Lz.Gzip.encoder}. *)
end

(** Zlib streams ({!Lz.Zlib}). *)
module Zlib : sig
  val decompress_reads : unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads () r] is a reader of the decoded bytes of one
      stream read from [r]. *)

  val decompress_writes : unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes () w] is a writer that decodes the stream written
      on it and writes its decoded bytes on [w]. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] is a reader of the stream of [r]'s
      bytes, at [level] as for {!Lz.Zlib.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] is a writer that writes the stream of
      the bytes written on it to [w], at [level] as for
      {!Lz.Zlib.encoder}. *)
end

(** Raw deflate streams ({!Lz.Deflate}). *)
module Deflate : sig
  val decompress_reads : unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads () r] is a reader of the decoded bytes of one
      stream read from [r]. *)

  val decompress_writes : unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes () w] is a writer that decodes the stream written
      on it and writes its decoded bytes on [w]. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] is a reader of the stream of [r]'s
      bytes, at [level] as for {!Lz.Deflate.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] is a writer that writes the stream of
      the bytes written on it to [w], at [level] as for
      {!Lz.Deflate.encoder}. *)
end
