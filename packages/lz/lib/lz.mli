(** Lossless compression between byte bigarrays: deflate with its zlib and
    gzip wrappers, Zstandard, LZ4 and Snappy, and CRC-32.

    Every buffer is a {!bigbytes}, a byte bigarray, whose bytes the
    collector never moves. A {e stream} is the bytes that encode one input,
    its {e decoded bytes}, in one format. A gzip stream holds one or more
    {e members}, and a Zstandard or LZ4 stream one or more {e frames},
    which below are members too. An LZ4 or Snappy {e block} is a stream of
    its own. A {e match} encodes decoded bytes as a copy of earlier decoded
    bytes of its member, at most the {e window size} back, and those
    earlier bytes are the member's {e window}. An encoder's {e level} sets
    how long it searches for matches.

    A {e whole-buffer} call is given all of its input
    ({!section-whole}). A format's [decode src dst] decodes the stream at
    the start of [src] into the {e destination} [dst], which is also the
    decode's window. It returns [Ok] of the bytes it read and wrote, or
    [Error] of where and why it refused the input ({!section-errors}). The
    program sizes [dst], usually from a header of the container that holds
    the stream. [encode src dst] writes the stream of [src] into [dst] and
    returns its length, or [None] when it does not fit, and [bound n] is a
    destination size large enough for the stream of any [n] bytes.

    A {e decoder} ({!Decoder.t}) or an {e encoder} ({!Encoder.t}) keeps one
    stream's state across calls, for input that arrives in pieces or whose
    size is not known ({!section-steps}). Each call is a {e step}. A step
    reads its input [src] from the start, {e takes} a prefix of it, and
    applies the program's [f] to its {e run}, the bytes it produced, which
    may be empty ({!Decoder.step}, {!Encoder.step}). It returns [More k]
    when it took [k] bytes and the stream goes on, and [Done k] when the
    stream ended after [k] bytes. A step with [~last:true] says that the
    input ends with [src].

    The run is lent from the decoder's or encoder's own buffer, and is
    valid until [f] returns. A later step writes over it, so [f] must copy
    or write out the bytes the program keeps. A decoder's run is part of
    its window, which later output copies from, so [f] must not write it.
    A write there changes later output, and only a checksum detects it. An
    encoder never reads its run back, and [f] may write it.

    The life cycle of a decoder or an encoder:
    + make it with its format's [decoder] or [encoder];
    + step it on the input in order, each step from the first byte the last
      one did not take;
    + give [~last:true] to the step that reaches the input's end, and to
      each step after it while it returns [More], on the bytes not yet
      taken, an empty input once all are;
    + after [Done], step it on a new stream, or drop it.

    A decoder that returns [Error] is {e dead}, and so is a decoder or an
    encoder whose step raised, from [f] or with [Out_of_memory] when it
    could not allocate. A program that abandons a stream midway makes a
    new decoder or encoder for the next.

    Unless a function's doc says otherwise:
    {ul
    {- a whole-buffer decode verifies every checksum and declared size its
       format carries before it returns [Ok], and a step before it returns
       [Done];}
    {- a size the input declares bounds nothing lz allocates, except a
       Zstandard frame's window size ({!Zstd.decoder});}
    {- an argument outside a stated range raises [Invalid_argument];}
    {- calls may run in parallel from any domains and threads, but a
       program must not run two in parallel when one writes a buffer the
       other reads or writes, and must not, from another domain or thread,
       read or write a buffer a call writes, or write one a call reads.}}

    This decodes a Zstandard page whose header states its decoded size
    [n]. A stream that decodes to more than [n] bytes is an error, and a
    [produced] below [n] means the stream held less than the header said.
    {[
      let page src n =
        let dst = Bigarray.(Array1.create int8_unsigned c_layout n) in
        match Lz.Zstd.decode src dst with
        | Ok { produced; _ } when produced = n -> Ok dst
        | Ok { produced; _ } -> Error (`Short produced)
        | Error e -> Error (`Lz e)
    ]} *)

(** {1:buffers Buffers and the domain lock}

    A thread runs OCaml code while it holds its domain's lock, the
    {e domain lock}. Since the collector never moves a buffer's bytes, a
    whole-buffer call or a step releases the domain lock while it works on
    an input of 16 KiB or more, and holds it under 4 KiB. Between the two,
    the threshold depends on the format and on whether the call decodes or
    encodes. The input is [src], and for a step the bytes of [src] it may
    take ({!section-steps}). The release lets the domain's other threads
    run, and lets a stop-the-world collection proceed without waiting for
    the call. Other domains run whether or not the call holds the lock. *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte buffers. *)

(** {1:errors Errors}

    A decode {e refuses} an input when it returns an {!error}. The error's
    [at] is an offset into the input, for a whole-buffer decode from the
    start of [src], and for a step from the first byte the decoder took
    after it was made or last returned [Done]. It is the first byte of what
    is wrong, such as a field that breaks the format, a checksum or
    declared size that does not match, or a {e code}, a unit of a bitstream
    such as a Huffman code, whose first byte is the one holding its first
    bit. A decode reads its input in order and returns the first error it
    meets, so an error in the bytes a checksum covers comes before the
    checksum's. *)

type why =
  | Truncated
      (** The input ends inside a stream, a cut header included, or is
          empty. It is at the input's length, and more input may complete
          the stream. *)
  | Malformed of string
      (** The bytes at the offset break the format, and no bytes appended
          to the input make it valid. A checksum or declared size that does
          not match is [Malformed]. *)
  | Too_long
      (** The output would pass the destination's end. It is at the first
          byte of the code or field that asks for the output. Only a
          whole-buffer decode returns it. *)
  | Unsupported of string
      (** The stream is well formed and uses a feature that lz
          {e declines}: a dictionary, LZ4's legacy frame ({!Lz4.Frame}), or
          a Zstandard window size above a decoder's bound. It is at the
          field that names the feature. *)
(** The type for the reasons a decode refuses an input. A message comes
    from a fixed set per format, which is zlib's messages and
    ["preset dictionary"] for deflate, zlib and gzip, RFC 8878's field
    names for Zstandard, and lz's own for LZ4 and Snappy. A message is a
    constant string and holds no byte of the input. Messages are for
    printing and may change in a later version, so a program must not parse
    them. *)

type error = { at : int; why : why }
(** The type for decode errors: the offset [at] and the reason [why]. *)

val pp_error : Format.formatter -> error -> unit
(** [pp_error ppf e] prints [e] on [ppf] as its kind, offset and message,
    as in [malformed at byte 1024: invalid distance too far back]. *)

(** {1:whole Whole-buffer calls} *)

(** {2:decoding Decoding}

    [decode src dst] decodes the stream at the start of [src] into [dst]
    from its start, and allocates no memory but its result. [dst]'s
    contents past the bytes it wrote, and all of them after an error, are
    unspecified.

    [decode] raises [Invalid_argument] if [src] and [dst] overlap. *)

type progress = { consumed : int; produced : int }
(** The type for whole-buffer decode results. [produced] counts the bytes
    written to [dst], and [consumed] the bytes read from [src]. A raw
    deflate or zlib stream ends at an exact byte, and [consumed] counts up
    to and including the byte that holds its last bit, so whatever follows
    the stream begins at [consumed]. Every other format is read to the end
    of [src], members one after another and an LZ4 or Snappy block, which
    has no end mark, whole, and [consumed] is [src]'s length. *)

(** {2:encoding Encoding}

    [encode src dst] writes one stream holding all of [src] to the start of
    [dst] and returns [Some] of its length, or [None] when it does not fit,
    with [dst]'s contents unspecified. A destination of
    [bound (Bigarray.Array1.dim src)] bytes always fits, at every level and
    for the empty input. [encode_string s] is the stream of [s] as a
    string, and the empty string encodes to the format's empty stream. An
    encode allocates its tables outside the OCaml heap for the call, and
    raises [Out_of_memory] if it cannot.

    [encode] raises [Invalid_argument] if [src] and [dst] overlap, and
    [bound] on a negative size or one whose bound would pass [max_int]. *)

(** {1:steps Decoders and encoders}

    A step takes all of [src] unless the stream ends inside it, the run
    reaches the decoder's {e run capacity}, or the encoder takes its
    {e quota}, the most input one step takes ({!Decoder}, {!Encoder}).
    The next step begins at the first byte not taken. A step applies [f]
    with the domain lock held.

    A decoder or encoder is {e busy} while a step on it runs, [f]
    included. A step on a busy or dead decoder or encoder, from any domain
    or thread, raises [Invalid_argument] and leaves it as it was. If [f]
    raises, this exception included, the exception propagates and the
    decoder or encoder is dead. *)

type step =
  | Done of int  (** The stream ended after this many bytes of the input. *)
  | More of int  (** This many bytes were taken and the stream goes on. *)
(** The type for step results. *)

(** Decoders.

    A decoder holds its format's tables, about 12 KiB for deflate's and
    24 KiB for Zstandard's, outside the OCaml heap from when it is made,
    and one bigarray, its {e buffer}, which holds its window and its run.
    A deflate, zlib or gzip decoder allocates its buffer at its first
    step, before it reads a byte, with room for a 32 KiB window and a run
    capacity of 256 KiB. A Zstandard decoder allocates its buffer when a
    frame's header arrives, before any output, with room for the frame's
    window size, or the content size the frame declares when smaller, and
    a run capacity of 256 KiB. At its first frame it also allocates about
    256 KiB outside the OCaml heap. It replaces its buffer when a frame
    needs a larger one. A view of a run kept after [f] returns keeps its
    buffer alive, and may read bytes a later step wrote. Each member
    begins with an empty window, and a run may hold the end of one member
    and the start of the next.

    A deflate or zlib decoder returns [Done k] once its stream ends, with
    or without [~last], and has not taken the bytes of [src] from [k] on.
    A gzip or Zstandard decoder returns [Done] only on a step with
    [~last:true], with [k] the length of [src], since without it the
    decoder cannot tell the last member from one that more input would
    follow. A step that returns [More k] with [k] the length of [src] has
    taken all of it, and with [~last:true] has output still due.

    With [~last:true] a stream still open is [Truncated], and a step
    returns [Truncated] only then. Every error a step returns is the one
    [decode] returns, into a large enough destination, on the input taken
    since the decoder was made or last returned [Done], at the same
    offset, except a Zstandard window size above the bound that
    {!Zstd.decoder} sets, which [decode] accepts. How much output [f]
    receives before an error depends on how the input was cut
    ({!Decoder.step}). Until a step returns [Done], the bytes [f] has
    received are unverified, since a later step may find that a checksum
    or declared size covering them does not match.

    lz bounds a decoder's memory and leaves its output unbounded. A
    program that decodes a stream it does not trust must count the bytes
    [f] receives, and stop once they pass its limit. *)
module Decoder : sig
  type t
  (** The type for decoders. *)

  val step :
    t -> last:bool -> bigbytes -> (bigbytes -> unit) -> (step, error) result
  (** [step d ~last src f] decodes a prefix of [src] and returns how much
      of [src] it took. On [Ok] it has applied [f] once to its run, every
      byte the step decoded. On [Error] it has not applied [f], and the
      bytes the step decoded are lost. *)
end

(** Encoders.

    An encoder holds its run's buffer, 384 KiB for deflate, zlib and gzip
    and about 256 KiB for Zstandard, which it allocates when made. Outside
    the OCaml heap it holds its match finder and a window of its input,
    which it allocates by its first step that writes output. They take up
    to about 1.2 MiB for deflate, zlib and gzip, and for Zstandard about
    2.3 MiB at level 1, 6 MiB at level 3 and 50 MiB at level 12. Its quota
    is 64 KiB for deflate, zlib and gzip, and 128 KiB for Zstandard. A
    step's run always has room for the output the step owes.

    An encoder may keep input it has taken until more arrives or a step
    has [~last:true], at most about 300 KiB for deflate, zlib and gzip and
    128 KiB for Zstandard. So a run may be shorter than its input's
    encoding, or empty, and a step's work grows with its input and with
    the kept input it writes out. A step with [~last:true] that takes all
    of [src], an empty [src] included, writes the kept input and the
    stream's end and returns [Done], and only such a step returns [Done].
    On an empty [src] it holds the domain lock while it writes. A fresh
    encoder so stepped on no input writes the format's empty stream. *)
module Encoder : sig
  type t
  (** The type for encoders. *)

  val step : t -> last:bool -> bigbytes -> (bigbytes -> unit) -> step
  (** [step e ~last src f] encodes a prefix of [src] into the stream,
      applies [f] once to its run, and returns how much of [src] it
      took. *)
end

(** {1:formats Formats}

    Each format's module has the whole-buffer calls ({!section-whole}),
    and for deflate, zlib, gzip and Zstandard a [decoder] and an [encoder]
    ({!section-steps}). Its [encode], [encode_string] and [encoder] take a
    [?level] where the format has levels. Its preamble lists what a decode
    refuses, each item [Malformed] at the first byte of the field or code
    it names unless it names another byte or reason.

    A stream's bytes depend only on its decoded bytes, its level and, for
    Zstandard, on whether [encode] or an encoder wrote it ({!Zstd}). They
    never depend on the machine or on how the input was cut into steps,
    and a later version of lz may write different bytes. A decode's output
    is the one its format's specification defines. *)

(** Raw deflate streams (RFC 1951). Levels are 1 to 9, default 6.

    A decode refuses:
    {ul
    {- a block type of 3, at BTYPE;}
    {- a stored block whose NLEN is not the complement of its LEN, at
       NLEN;}
    {- a dynamic block declaring more than 286 literal/length or 30
       distance codes, at HLIT or HDIST;}
    {- code lengths that over-subscribe the code length, literal/length or
       distance code or leave it incomplete, at the length that proves it,
       except a literal/length or distance code that holds a single code
       of length 1, and a distance or code length code whose lengths are
       all 0;}
    {- a repeat of code lengths with no length before it or past their
       count, at the repeat;}
    {- an end-of-block symbol with no code, at the length that leaves it
       none;}
    {- a code that names no symbol, a literal/length symbol 286 or 287, or
       a distance symbol 30 or 31;}
    {- a distance before the stream's first byte, at its code.}}
    A code length code whose lengths are all 0 reads every length as 0.
    Output past [dst]'s end is [Too_long] at the literal/length code that
    asks for it, or in a stored block at the first stored byte that has no
    room. *)
module Deflate : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes the raw deflate stream at [src]'s start
      ({!section-decoding}). *)

  val bound : int -> int
  (** [bound n] is a destination size for the stream of any [n] bytes
      ({!section-encoding}). *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes the stream of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the stream of [s]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] is a new decoder of raw deflate streams. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] is a new encoder of raw deflate streams at
      [level]. *)
end

(** Zlib streams (RFC 1950): a deflate stream with a header and the
    Adler-32 of its decoded bytes. Levels as {!Deflate}'s.

    A decode refuses, besides {!Deflate}'s refusals:
    {ul
    {- a header whose check bits fail, a compression method other than 8,
       or a window size above 32 KiB, at the header's first byte;}
    {- an Adler-32 that does not match.}}
    A preset dictionary is [Unsupported] at the header's second byte. *)
module Zlib : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes the zlib stream at [src]'s start
      ({!section-decoding}). *)

  val bound : int -> int
  (** [bound n] is a destination size for the stream of any [n] bytes
      ({!section-encoding}). *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes the stream of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the stream of [s]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] is a new decoder of zlib streams. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] is a new encoder of zlib streams at [level]. *)
end

(** Gzip (RFC 1952): one or more members, each a deflate stream with a
    header, the CRC-32 of its decoded bytes and their length modulo
    2{^32}. Levels as {!Deflate}'s. The encoder writes one member whose
    header has no name, comment or extra field and a modification time
    of 0.

    A decode refuses, besides {!Deflate}'s refusals:
    {ul
    {- a member that does not start with the magic 1f 8b;}
    {- a compression method other than 8, at CM;}
    {- a reserved flag bit, at FLG;}
    {- a header CRC (FHCRC), CRC-32 or length (ISIZE) that does not
       match.}}
    It checks ID1, ID2 and CM as each arrives, so an input that ends just
    after a wrong one is [Malformed]. A member begins where the last one
    ended, so zero padding after the last member is [Malformed] at its
    first byte. *)
module Gzip : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes every member of [src]
      ({!section-decoding}). *)

  val bound : int -> int
  (** [bound n] is a destination size for the member of any [n] bytes
      ({!section-encoding}). *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes one member of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the member of [s]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] is a new decoder of gzip streams. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] is a new encoder of gzip members at [level]. *)
end

(** Zstandard (RFC 8878): one or more frames, skippable frames among them,
    which produce nothing. Levels are 1 to 12, default 3. The encoder
    always writes the frame's Content_Checksum, an XXH64. It writes
    Frame_Content_Size only in a frame from [encode], so a frame from an
    encoder differs from [encode]'s only in its header. Every frame lz
    writes, from [encode] or an encoder, declares a window size of at most
    4 MiB.

    A decode refuses:
    {ul
    {- a Magic_Number of neither a frame nor a skippable frame, at the
       first byte where no frame begins;}
    {- a reserved Frame_Header_Descriptor bit, Block_Type or
       Symbol_Compression_Modes bits;}
    {- a Block_Size or Regenerated_Size above Block_Maximum_Size, the
       smaller of the window size and 128 KiB;}
    {- a literals or sequences section, Huffman tree or FSE table
       description that runs past its block;}
    {- a Regenerated_Size of 1, 2 or 5 literals in four streams, a
       Compressed_Size too small for its streams, or a Jump_Table past
       Total_Streams_Size;}
    {- a Treeless_Literals_Block or Repeat_Mode with no earlier table;}
    {- a Huffman Weight above 11, or weights that complete to no prefix
       code of at most 11 bits;}
    {- an Accuracy_Log above the format's limit, probabilities past the
       alphabet, or an RLE_Mode symbol above its code's maximum;}
    {- bytes after a Number_of_Sequences of 0;}
    {- a sequence whose offset is 0, above Window_Size or before the
       frame's start, whose Literals_Length passes the literals, or whose
       output passes Block_Maximum_Size;}
    {- a Huffman-coded or sequences bitstream whose last byte is 0, at that
       byte, or that is not read exactly to its start;}
    {- a Frame_Content_Size or Content_Checksum that does not match the
       frame's decoded bytes.}}
    A frame naming a dictionary is [Unsupported] at its Dictionary_ID.
    Output a Zstandard block asks for past [dst]'s end is [Too_long] at the
    block's header. *)
module Zstd : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes every frame of [src] ({!section-decoding}),
      whatever window size it declares, since its window is [dst]. *)

  val bound : int -> int
  (** [bound n] is a destination size for the frame of any [n] bytes
      ({!section-encoding}). *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes one frame of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the frame of [s]. *)

  val decoder : ?window:int -> unit -> Decoder.t
  (** [decoder ~window ()] is a new decoder of Zstandard streams whose
      frames' window sizes are at most [window] bytes (default 128 MiB). A
      frame whose Window_Descriptor declares more than [window] is
      [Unsupported] there, even when its Frame_Content_Size is smaller. A
      frame without a Window_Descriptor declares its Frame_Content_Size as
      its window size, and is [Unsupported] at that field when it passes
      [window]. A program that steps Zstandard streams it does not trust
      sets [window].

      Raises [Invalid_argument] if [window] is below 1 KiB. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] is a new encoder of Zstandard frames at
      [level]. *)
end

(** LZ4 blocks and frames, with whole-buffer calls only. *)
module Lz4 : sig
  (** LZ4 blocks without a size prefix, whose decoded size the program
      stores beside the block. A block is a series of sequences, each a
      one-byte {e token}, literals and a match, and its last sequence holds
      literals only.

      A decode refuses:
      {ul
      {- a match offset of 0 or before the block's start, at the offset
         field;}
      {- last literals that break the format's end-of-block rules (fewer
         than 5, or under 12 bytes from the last match's start to the
         block's end), at their token.}}
      A block whose input ends inside a sequence, or after a match, is
      [Truncated]. Output past [dst] is [Too_long] at the token of the
      sequence whose literals ask for it, or at the offset field of the
      match that does. *)
  module Block : sig
    val decode : bigbytes -> bigbytes -> (progress, error) result
    (** [decode src dst] decodes the block at [src]'s start
        ({!section-decoding}). *)

    val bound : int -> int
    (** [bound n] is a destination size for the block of any [n] bytes
        ({!section-encoding}). *)

    val encode : bigbytes -> bigbytes -> int option
    (** [encode src dst] writes the block of [src] to [dst]. *)

    val encode_string : string -> string
    (** [encode_string s] is the block of [s]. *)
  end

  (** LZ4 frames: one or more, skippable frames among them, read to the
      input's end. A frame's blocks are independent or linked, and a match
      in a linked block may reach back into the earlier blocks of its
      frame. The encoder writes one frame of independent blocks with the
      content size and the content checksum.

      A decode refuses:
      {ul
      {- a magic number of neither a frame, a skippable frame nor a
         {e legacy frame}, the format's older form with magic number
         0x184C2102, at the first byte where no frame begins;}
      {- a version other than 01, a set reserved bit, or a Block Maximum
         Size value below 4;}
      {- a header checksum that does not match;}
      {- a block size above the declared maximum;}
      {- a block checksum, content checksum or content size that does not
         match;}
      {- a block decoding past the declared maximum, at the code that
         does;}
      {- a block ending inside a sequence or after a match, at the token
         of its last sequence;}
      {- {!Block}'s refusals, where a linked block's start is its
         frame's.}}
      A frame naming a dictionary is [Unsupported] at its Dict_ID, and a
      legacy frame at its magic number. *)
  module Frame : sig
    val decode : bigbytes -> bigbytes -> (progress, error) result
    (** [decode src dst] decodes every frame of [src]
        ({!section-decoding}). *)

    val bound : int -> int
    (** [bound n] is a destination size for the frame of any [n] bytes
        ({!section-encoding}). *)

    val encode : bigbytes -> bigbytes -> int option
    (** [encode src dst] writes one frame of [src] to [dst]. *)

    val encode_string : string -> string
    (** [encode_string s] is the frame of [s]. *)
  end
end

(** Snappy blocks, whose preamble states the length of their decoded
    bytes, at most 2{^32} - 1. After the preamble come {e elements}, each a
    literal or a copy, whose first byte is its {e tag}. A decode compares
    the preamble with [dst]'s size before any output, and is [Too_long] at
    the preamble's first byte when [dst] is smaller. It compares it again
    with the output.

    A decode refuses:
    {ul
    {- a preamble longer than 5 bytes or above 2{^32} - 1;}
    {- a copy offset of 0 or before the block's start, or an element whose
       output passes the preamble's length, at the element's tag.}}
    A block whose input ends inside an element, or before its output
    reaches the preamble's length, is [Truncated]. *)
module Snappy : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes the block at [src]'s start
      ({!section-decoding}). *)

  val bound : int -> int
  (** [bound n] is a destination size for the block of any [n] bytes
      ({!section-encoding}).

      Raises [Invalid_argument] if [n] is above 2{^32} - 1. *)

  val encode : bigbytes -> bigbytes -> int option
  (** [encode src dst] writes the block of [src] to [dst].

      Raises [Invalid_argument] if [src] holds more than 2{^32} - 1
      bytes. *)

  val encode_string : string -> string
  (** [encode_string s] is the block of [s].

      Raises [Invalid_argument] as [encode] does. *)
end

(** {1:checksums Checksums} *)

(** CRC-32 as gzip, PNG and ZIP compute it (RFC 1952, section 8), as an
    integer in \[0, 2{^32}). [~crc] continues a running value:
    [string (a ^ b)] is [string ~crc:(string a) b]. {!Crc32.bigbytes}
    releases the domain lock on 1 MiB or more, and {!Crc32.string} holds
    it. *)
module Crc32 : sig
  val bigbytes : ?crc:int -> bigbytes -> int
  (** [bigbytes ~crc b] is the CRC-32 of [b] continued from [crc]
      (default 0). *)

  val string : ?crc:int -> ?first:int -> ?length:int -> string -> int
  (** [string ~crc ~first ~length s] is the CRC-32 of the [length] bytes of
      [s] from [first], continued from [crc]. [first] defaults to 0, [crc]
      to 0, and [length] to the bytes of [s] from [first] on.

      Raises [Invalid_argument] if [first] and [length] do not name bytes of
      [s]. *)
end

(**/**)

(* For the tests: the checksums the formats carry. *)

module Adler32 : sig
  val bigbytes : ?adler:int32 -> bigbytes -> int32
  val string : ?adler:int32 -> ?first:int -> ?length:int -> string -> int32
end

module Xxh32 : sig
  val bigbytes : ?seed:int32 -> bigbytes -> int32
  val string : ?seed:int32 -> ?first:int -> ?length:int -> string -> int32
end

module Xxh64 : sig
  val bigbytes : ?seed:int64 -> bigbytes -> int64
  val string : ?seed:int64 -> ?first:int -> ?length:int -> string -> int64
end

(* For the tests, which hold every path of a kernel to the same bits: the
   paths this host can run, each by its name and run through the same
   driver as the modules above, the generic path first and the one the
   modules above take last. A checksum path takes the running value and
   checksums all of its buffer; a Zstandard path is [Zstd.decode] or
   [Zstd.encode], a deflate path [Deflate.encode]. *)

module Paths : sig
  val crc32 : (string * (int32 -> bigbytes -> int32)) list
  val adler32 : (string * (int32 -> bigbytes -> int32)) list

  val zstd_decode :
    (string * (bigbytes -> bigbytes -> (progress, error) result)) list

  val zstd_encode :
    (string * (level:int -> bigbytes -> bigbytes -> int option)) list

  val deflate_encode :
    (string * (level:int -> bigbytes -> bigbytes -> int option)) list
end

(* For the tests, which hold Zstandard's bytes equal across index limits:
   [Zstd.encode] and [Zstd.encoder] with match finders that rebase their
   indices once a block that starts past the window would end past index
   [limit], in [0, 2^32), rather than past 3 GiB. With a limit of 0, every
   such block rebases. *)

val zstd_encode_limit :
  limit:int -> level:int -> bigbytes -> bigbytes -> int option

val zstd_encoder_limit : limit:int -> level:int -> Encoder.t
