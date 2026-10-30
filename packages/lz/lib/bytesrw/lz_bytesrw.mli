(** Bytesrw filters that decompress and compress deflate, zlib, gzip and
    Zstandard streams with {!Lz}.

    A {e reader filter} takes a reader [r] of one encoding of some bytes
    and returns a reader of the other. A {e writer filter} takes a writer
    [w] and returns a writer that turns the bytes written on it the same
    way and writes the result on [w]. A {e decompressing} filter turns a
    stream ({!Lz}) into the bytes it holds, and a {e compressing} filter
    turns bytes into a stream. Each format's module has the four filters
    ({!section-formats}).

    A filter steps its format's {!Lz.Decoder.t} or {!Lz.Encoder.t} over the
    slices it gets. However long the stream and its slices, it holds the
    decoder or encoder, at most 64 KiB of staged input and a copy of one
    run. Zstandard's window, which [?window] bounds, is the only part that
    grows with the input. A decompressing Zstandard filter allocates the
    window a frame's header declares when the header arrives, before any
    output, up to [?window] (default 128 MiB). A filter over streams the
    program does not trust sets [?window].

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

    This reads the content of a [.jsonl.zst] shard, as a reader that a JSON
    parser takes. The filter decodes each frame as the parser reads, so the
    shard is never held whole in memory.
    {[
      let lines ic =
        let r = Bytesrw.Bytes.Reader.of_in_channel ic in
        Lz_bytesrw.Zstd.decompress_reads () r
    ]} *)

(** {1:errors Errors}

    A decompressing filter's read or write raises
    [Bytesrw.Bytes.Stream.Error] with [Error e], where [e] is the error its
    format's [decode] returns on the bytes read or written, at the same
    offset, except for a Zstandard window above [window]. The error's
    position is that of [e]'s byte in the compressed stream: in [r]'s
    positions for a reader, in the filtering writer's for a writer. The
    message prints [e] with {!Lz.pp_error}.

    After an error, every later read or write of the filter raises it
    again. An exception raised while the filter steps its decoder or
    encoder, such as one from a write on [w] or [Out_of_memory],
    propagates, and every later read or write of the filter raises it
    again too. Compressing filters
    raise no [Error]. *)

type Bytesrw.Bytes.Stream.error +=
  | Error of Lz.error  (** A refusal of the stream's bytes. *)

(** {1:formats Formats}

    Each module has four filters. A decompressing filter reads the streams
    its format's [decode] reads. [decompress_reads] of a deflate or zlib
    stream decodes one stream. Once it has returned
    [Bytesrw.Bytes.Slice.eod], it leaves [r] at the first byte after the
    stream, and a later read of [r] returns the bytes the stream did not
    hold. [decompress_reads] of gzip or Zstandard decodes every member up
    to [r]'s end. [decompress_writes] decodes up to the [eod] written on
    it. Bytes written after a deflate or zlib stream are [Malformed] at
    their first byte, with the message ["bytes after the stream"].

    [compress_reads] and [compress_writes] write one stream holding every
    byte read from [r] or written on the filter, ended at [r]'s end or at
    the [eod] written. The stream's bytes are those of its format's
    [encoder] stepped over the same input with [~last:true] at its end,
    whatever the slices. *)

(** Zstandard frames ({!Lz.Zstd}). *)
module Zstd : sig
  val decompress_reads : ?window:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads ~window () r] reads the frames of [r] up to its end
      and returns their content. A frame whose window passes [window] bytes
      is [Unsupported], as for {!Lz.Zstd.decoder}. *)

  val decompress_writes : ?window:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes ~window () w] decodes the frames written on it up
      to [eod] and writes their content on [w]. [window] is as for
      {!decompress_reads}. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] returns one frame of [r]'s bytes, at
      [level] as for {!Lz.Zstd.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] writes one frame of the bytes written on
      it on [w], at [level] as for {!Lz.Zstd.encoder}. *)
end

(** Gzip members ({!Lz.Gzip}). *)
module Gzip : sig
  val decompress_reads : unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads () r] reads the members of [r] up to its end and
      returns their content. *)

  val decompress_writes : unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes () w] decodes the members written on it up to
      [eod] and writes their content on [w]. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] returns one member of [r]'s bytes, at
      [level] as for {!Lz.Gzip.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] writes one member of the bytes written
      on it on [w], at [level] as for {!Lz.Gzip.encoder}. *)
end

(** Zlib streams ({!Lz.Zlib}). *)
module Zlib : sig
  val decompress_reads : unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads () r] reads one stream from [r] and returns its
      content, leaving [r] at the first byte after the stream. *)

  val decompress_writes : unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes () w] decodes the stream written on it and writes
      its content on [w]. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] returns the stream of [r]'s bytes, at
      [level] as for {!Lz.Zlib.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] writes the stream of the bytes written
      on it on [w], at [level] as for {!Lz.Zlib.encoder}. *)
end

(** Raw deflate streams ({!Lz.Deflate}). *)
module Deflate : sig
  val decompress_reads : unit -> Bytesrw.Bytes.Reader.filter
  (** [decompress_reads () r] reads one stream from [r] and returns its
      content, leaving [r] at the first byte after the stream. *)

  val decompress_writes : unit -> Bytesrw.Bytes.Writer.filter
  (** [decompress_writes () w] decodes the stream written on it and writes
      its content on [w]. *)

  val compress_reads : ?level:int -> unit -> Bytesrw.Bytes.Reader.filter
  (** [compress_reads ~level () r] returns the stream of [r]'s bytes, at
      [level] as for {!Lz.Deflate.encoder}. *)

  val compress_writes : ?level:int -> unit -> Bytesrw.Bytes.Writer.filter
  (** [compress_writes ~level () w] writes the stream of the bytes written
      on it on [w], at [level] as for {!Lz.Deflate.encoder}. *)
end
