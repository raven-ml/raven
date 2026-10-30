(** Lossless compression between byte bigarrays: deflate with its zlib and
    gzip wrappers, Zstandard, LZ4 and Snappy, and CRC-32.

    Every buffer is a {!bigbytes}, a byte bigarray, whose bytes the
    collector never moves. A {e stream} is one encoded datum of a format: a
    raw deflate or zlib stream, a gzip stream of one or more {e members}, a
    Zstandard or LZ4 stream of one or more frames, or an LZ4 or Snappy
    block. A frame is a member of its stream. A match in a stream copies
    bytes decoded before it, from the {e window} of the latest decoded
    bytes.

    A {e whole-buffer} call is given all of its input
    ({!section-whole}). [decode src dst] decodes the stream at the start of
    [src] into the {e destination} [dst], whose size the program learns
    from the container that holds the stream. The destination is the
    decode's window. [encode src dst] writes the stream of [src] into [dst]
    and returns its length, or [None] when it does not fit, and [bound n]
    is a destination size that the stream of any [n] bytes fits.

    A {e decoder} ({!Decoder.t}) or an {e encoder} ({!Encoder.t}) keeps one
    stream's state across calls, for input that arrives in pieces or whose
    size is not known ({!section-streams}). Each of these calls is a
    {e step}. A step reads from the start of the input it is given and
    {e takes} a prefix of it, all of it unless a decoder's output or an
    encoder's input reached what one step holds. It applies a function [f]
    that the program passes, once, to the output it produced, its {e run},
    which may be empty. The run is lent from the decoder's or encoder's own
    memory. It is valid until [f] returns, and a later step writes over it,
    so [f] copies or writes out what the program keeps. A step with
    [~last:true] says that the input ends with the bytes it is given.

    A decode that fails returns [Error { at; why }], where [why] says what
    is wrong and [at] is the offset of the input byte where it begins
    ({!section-errors}).

    The life cycle of a decoder or an encoder:
    + make it with its format's [decoder] or [encoder];
    + step it on the input in order, each step from the first byte the last
      one did not take;
    + step it with [~last:true] on the input's end, and with an empty input
      again while it returns [More] of the whole input, until [Done];
    + after [Done], step it on a new stream, or drop it.

    A decoder that returns [Error] is {e dead}, and so is a decoder or an
    encoder whose step raised, from [f] or with [Out_of_memory]. A stream
    given up midway gets a new decoder or encoder.

    A whole-buffer call, a step and a checksum of a {!bigbytes} release the
    runtime lock while they work on a long input, so that other domains and
    threads run. {!Crc32.string} holds the lock. A decoder or encoder is
    {e busy} while a step on it runs, [f] included, and takes one step at a
    time. Other calls may run in parallel from any domains.

    Encoded bytes depend on the input, the level, where a stream written by
    steps had [~last:true], and whether [encode] wrote it. They never depend
    on the machine or on how the input was cut into steps, and they may
    change in a later version of this library. Decoded bytes are the
    format's.

    Unless a function's doc says otherwise:
    {ul
    {- a whole-buffer decode verifies every checksum and declared size its
       format carries before it returns [Ok], and a step before it returns
       [Done];}
    {- a size the input declares bounds nothing lz allocates, except a
       Zstandard frame's window ({!Zstd.decoder});}
    {- an argument outside a stated range raises [Invalid_argument];}
    {- a program must not write a buffer from another domain or thread while
       a call reads or writes it.}}

    This decodes a Zstandard page whose header states its decoded size [n].
    A destination of [n] bytes makes a stream that decodes to more
    [Too_long], and [produced] below [n] means the stream held less than
    the header said.
    {[
      let page src n =
        let dst = Bigarray.(Array1.create int8_unsigned c_layout n) in
        match Lz.Zstd.decode src dst with
        | Ok { produced; _ } when produced = n -> Ok dst
        | Ok { produced; _ } -> Error (`Short produced)
        | Error e -> Error (`Lz e)
    ]} *)

(** {1:buffers Buffers} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte buffers. *)

(** {1:errors Errors}

    A decode that fails returns an {!error}. Its [at] is an offset into the
    input. For a whole-buffer decode it counts from the start of [src], and
    for a step from the first byte the decoder took after it was made or
    last returned [Done]. It is the first byte of what is wrong: the byte a
    bad code begins at, the first byte of a checksum or size that does not
    match, the first byte of a declined field, or the input's length when
    the input ends inside a stream. A [Too_long] is at the byte that begins
    the code asking for the output. Each format's module lists what its
    decoders refuse and at which byte. *)

type why =
  | Truncated
      (** The input ends inside a stream, or holds none. A program that can
          read more bytes does. *)
  | Malformed of string
      (** The bytes at the offset are not the format, and no bytes appended
          to the input make it valid. A checksum or declared size that does
          not match is [Malformed]. *)
  | Too_long
      (** The output would pass the destination's end. Only a whole-buffer
          decode returns it. *)
  | Unsupported of string
      (** A well-formed stream this library declines: a preset dictionary,
          LZ4's legacy frame, or a Zstandard window above the decoder's
          bound. *)
(** The type for the reasons a decode fails. The strings come from a fixed
    set per format, zlib's messages for the deflate family and RFC 8878's
    field names for Zstandard. They hold no byte of the input, and numbers
    at most. They are for printing, and a program must not parse them. *)

type error = { at : int; why : why }
(** The type for decode errors: the offset [at] and the reason [why]. *)

val pp_error : Format.formatter -> error -> unit
(** [pp_error] formats an error as its kind, offset and message:
    [malformed at byte 1024: invalid distance too far back]. *)

(** {1:whole Whole-buffer calls} *)

(** {2:decoding Decoding}

    [decode src dst] decodes [src] from its start into [dst] from its
    start, and allocates nothing. A raw deflate or zlib stream ends at an
    exact byte. [consumed] counts up to it, the byte that holds the
    stream's last bit included, and whatever follows the stream in [src]
    begins at [consumed]. Every other format is read to the input's end,
    and [consumed] is [src]'s length. Among gzip members and Zstandard and
    LZ4 frames, a byte that begins no member, at the start or after one, is
    [Malformed] at its offset. An LZ4 or Snappy block has no end mark, and
    bytes after it are read as more of the block.

    An input with no stream, the empty one included, is [Truncated] at 0,
    and one that ends inside a stream is [Truncated] at its length. A match
    reaching before the first byte of its member, frame or independent
    block is [Malformed]. Output that would pass [dst]'s end is
    [Too_long]. [produced] counts the bytes written. [dst]'s contents past
    them, and all of them after an error, are unspecified.

    [decode] raises [Invalid_argument] if [src] and [dst] overlap. *)

type progress = { consumed : int; produced : int }
(** The type for whole-buffer decode results: the input bytes read and the
    output bytes written. *)

(** {2:encoding Encoding}

    [encode ?level src dst] writes one stream holding all of [src] to the
    start of [dst] and returns [Some] of its length, or [None] when it does
    not fit, with [dst]'s contents unspecified. A destination of
    [bound (Bigarray.Array1.dim src)] bytes always fits, at every level and
    for the empty input. [encode_string ?level s] is the stream of [s] as a
    string, and the empty string encodes to the format's empty stream. An
    encode allocates its tables outside the OCaml heap for the call.

    [encode] raises [Invalid_argument] if [src] and [dst] overlap, and
    [bound] on a negative size or one whose bound would pass [max_int]. *)

(** {1:streams Streams}

    A step on a decoder or an encoder reads [src] from its start, takes a
    prefix of it, applies [f] once to the run it produced, and returns a
    {!type-step}. [More k] says that it took [k] bytes and the stream goes
    on. [k] short of [src]'s length means that a decoder's run filled, or
    that an encoder took its {e quota}, the most input one step takes. The
    next step then continues from byte [k]. [Done k] says that the stream
    ended after [k] bytes. The run is a sub-array of the decoder's or
    encoder's own bigarray. A view of it kept after [f] returns reads the
    bytes a later step wrote there.

    A step raises [Invalid_argument] if its decoder or encoder is busy, from
    another domain, another thread or [f], or dead. If [f] raises, the
    exception propagates and the decoder or encoder is dead. *)

type step =
  | Done of int  (** The stream ended after this many bytes of the input. *)
  | More of int  (** This many bytes were taken and the stream goes on. *)
(** The type for step results. *)

(** Streaming decoders.

    A decoder holds its format's tables and one bigarray of its window plus
    a run. It allocates the bigarray when a header fixes the window, and
    grows it to the largest window seen. After [Done] the next step begins
    a new stream of the format, and each member begins with an empty
    window.

    A deflate or zlib stream ends at its exact end, with or without
    [~last]. The step returns [Done k] and has not read the bytes from [k]
    on. A gzip or Zstandard stream ends only on a step with [~last:true],
    with [k] the length of [src]. [More k] with [k] the length of [src]
    means the input is used up. With [~last:false] the decoder may then
    hold a cut header, or a member that ended on the last byte. With
    [~last:true] output is still due, and the program steps again with an
    empty input.

    With [~last:true] the input ends with [src]. A stream still open is
    then [Truncated] at the input's length, and a decoder that has seen no
    stream is [Truncated] at 0. A step returns [Truncated] only then, and
    never [Too_long]. Every error a step returns is the one [decode]
    returns on the bytes taken, at the same offset, except a Zstandard
    window above the decoder's [?window], which [decode] accepts. *)
module Decoder : sig
  type t
  (** The type for decoders. *)

  val step :
    t -> last:bool -> bigbytes -> (bigbytes -> unit) -> (step, error) result
  (** [step d ~last src f] decodes [src] from its start, applies [f] once to
      the run it produced, and returns where it got to. [Ok] means every
      byte decoded has been handed to [f]. On [Error], [f] is not
      applied.

      Raises [Invalid_argument] if [d] is busy or dead. *)
end

(** Streaming encoders.

    An encoder holds its match finder, a window of its input and its output
    buffer. Its quota is a constant of the format, and a step's run holds
    all that the quota produces. A step's cost is linear in [src]. A step
    with [~last:true] that takes the last byte of [src] writes the stream's
    end and returns [Done], and only such a step returns [Done]. The next
    step after [Done] begins a new stream, and a fresh encoder stepped with
    [~last:true] and no input writes the empty stream. *)
module Encoder : sig
  type t
  (** The type for encoders. *)

  val step : t -> last:bool -> bigbytes -> (bigbytes -> unit) -> step
  (** [step e ~last src f] encodes [src] from its start into the stream,
      applies [f] once to the run it produced, and returns where it got to.

      Raises [Invalid_argument] if [e] is busy or dead. *)
end

(** {1:formats Formats}

    Each format's module has the whole-buffer calls and the decoders and
    encoders its format supports. [decode src dst] is its whole-buffer
    decode ({!section-decoding}), and [bound n] the size of a destination
    that the stream of any [n] bytes fits. [encode ?level src dst] and
    [encode_string ?level s] encode at [level] ({!section-encoding}).
    [decoder ()] makes a decoder of the format's streams, and
    [encoder ?level ()] an encoder at [level] ({!section-streams}). Each
    module's preamble lists what its decoders refuse, [Malformed] at the
    byte it names unless it says otherwise. *)

(** Raw deflate streams (RFC 1951). Levels are 1 to 9, default 6, and a
    higher level searches longer for matches.

    A stream is refused, with zlib's message, at the byte holding the first
    bit of what is wrong, for: a block type of 3, at BTYPE; a stored block
    whose NLEN is not the complement of its LEN, at NLEN; a dynamic block
    declaring more than 286 literal/length or 30 distance codes, at HLIT or
    HDIST; code lengths that over-subscribe the code length, literal/length
    or distance code or leave it incomplete, at the length that proves it,
    except a literal/length or distance code of one 1-bit code, and an
    empty distance or code length code; a repeat of code lengths with no
    length before it or past their count, at the repeat; an end-of-block
    symbol with no code, at the length that leaves it none; a code that
    names no symbol, a literal/length symbol 286 or 287, or a distance
    symbol 30 or 31, at the code; a distance before the stream's first
    byte, at its code. An empty code length code reads every length as
    0. *)
module Deflate : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes the stream at [src]'s start. *)

  val bound : int -> int
  (** [bound n] holds the stream of any [n] bytes. *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes the stream of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the stream of [s]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] decodes a sequence of streams, each ending in [Done]. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] writes a stream per [~last:true]. *)
end

(** Zlib streams (RFC 1950): a deflate stream with a header and the
    Adler-32 of its data. Levels as {!Deflate}'s.

    Refused besides {!Deflate}'s, at the header's first byte: a header
    whose check bits fail, a compression method other than 8, a window
    above 32 KiB; and at its first byte, an Adler-32 that does not match. A
    preset dictionary is [Unsupported] at the header's second byte. *)
module Zlib : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes the stream at [src]'s start. *)

  val bound : int -> int
  (** [bound n] holds the stream of any [n] bytes. *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes the stream of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the stream of [s]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] decodes a sequence of streams, each ending in [Done]. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] writes a stream per [~last:true]. *)
end

(** Gzip (RFC 1952): one or more members, each a deflate stream with a
    header, the CRC-32 of its data and its length modulo 2{^32}. Levels as
    {!Deflate}'s. The encoder writes one member whose header has no name,
    comment or extra field and a modification time of 0.

    Refused besides {!Deflate}'s: a member that does not start with the
    magic 1f 8b, at its first byte; a compression method other than 8, at
    CM; a reserved flag bit, at FLG; a header CRC (FHCRC), CRC-32 or length
    (ISIZE) that does not match, at its first byte. ID1, ID2 and CM are
    checked as each arrives, so an input that ends just after a wrong one
    is [Malformed]. Zero padding after the last member begins no member,
    and is [Malformed] at its first byte. *)
module Gzip : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes every member of [src]. *)

  val bound : int -> int
  (** [bound n] holds the member of any [n] bytes. *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes one member of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the member of [s]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] decodes a sequence of members, ending in [Done] on the
      step with [~last:true]. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] writes a member per [~last:true]. *)
end

(** Zstandard (RFC 8878): one or more frames, skippable frames among them,
    which produce nothing. Levels are 1 to 12, default 3. The encoder
    always writes the frame's XXH64 checksum. It writes the content size
    only in a frame from [encode], so a frame written by steps differs from
    [encode]'s only in its header. The encoder's windows are at most
    4 MiB.

    Refused, [Malformed] at the first byte of the field named: a
    Magic_Number of neither a frame nor a skippable frame, at the first
    byte where no frame begins; a reserved Frame_Header_Descriptor bit; a
    reserved Block_Type; a Block_Size above Block_Maximum_Size, the smaller
    of the window and 128 KiB; a literals or sequences section, Huffman
    tree or FSE table description that runs past its block, at its first
    byte; a Regenerated_Size above Block_Maximum_Size, or of 1, 2 or 5
    literals in four streams; a Compressed_Size too small for its streams;
    a Jump_Table past Total_Streams_Size; a Treeless_Literals_Block or
    Repeat_Mode with no earlier table; a Huffman Weight above 11, or
    weights that complete to no prefix code of at most 11 bits; an
    Accuracy_Log above the format's limit, or probabilities past the
    alphabet; reserved Symbol_Compression_Modes bits; an RLE_Mode symbol
    above its code's maximum; bytes after a Number_of_Sequences of 0; a
    sequence whose offset is 0, above Window_Size or before the frame's
    start, whose Literals_Length passes the literals, or whose output
    passes Block_Maximum_Size, at the byte holding its first bit; a
    Huffman-coded or sequences bitstream whose last byte is 0, at that
    byte, or that is not read exactly to its start, at its first byte; a
    Frame_Content_Size or Content_Checksum other than what the frame
    produced. A frame naming a dictionary is [Unsupported] at its
    Dictionary_ID. Output a block asks for past [dst]'s end is [Too_long]
    at the block's header. *)
module Zstd : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes every frame of [src], whatever window it
      declares, since its window is [dst]. *)

  val bound : int -> int
  (** [bound n] holds the frame of any [n] bytes. *)

  val encode : ?level:int -> bigbytes -> bigbytes -> int option
  (** [encode ~level src dst] writes one frame of [src] to [dst]. *)

  val encode_string : ?level:int -> string -> string
  (** [encode_string ~level s] is the frame of [s]. *)

  val decoder : ?window:int -> unit -> Decoder.t
  (** [decoder ~window ()] decodes a sequence of frames, ending in [Done] on
      the step with [~last:true]. It allocates a frame's window when the
      frame's header arrives, before any output. A frame whose window
      passes [window] bytes (default 128 MiB) is [Unsupported] at its
      Window_Descriptor or, for a single-segment frame, whose window is its
      content size, at its Frame_Content_Size. A program stepping Zstandard
      streams it does not trust sets [window].

      Raises [Invalid_argument] if [window] is below 1 KiB. *)

  val encoder : ?level:int -> unit -> Encoder.t
  (** [encoder ~level ()] writes a frame per [~last:true]. *)
end

(** LZ4 blocks and frames, with whole-buffer calls only. *)
module Lz4 : sig
  (** LZ4 blocks without a size prefix, whose decoded size the program
      keeps. A block is refused for a match offset of 0 or before the
      block's start, at the offset field, and for last literals that break
      the format's end-of-block rules (fewer than 5, or under 12 bytes from
      the last match's start to the block's end), at their token. A block
      whose input ends inside a sequence, or after a match, is [Truncated].
      Output past [dst] is [Too_long] at the token of the literals, or the
      offset field of the match, that asks for it. *)
  module Block : sig
    val decode : bigbytes -> bigbytes -> (progress, error) result
    (** [decode src dst] decodes the block at [src]'s start. *)

    val bound : int -> int
    (** [bound n] holds the block of any [n] bytes. *)

    val encode : bigbytes -> bigbytes -> int option
    (** [encode src dst] writes the block of [src] to [dst]. *)

    val encode_string : string -> string
    (** [encode_string s] is the block of [s]. *)
  end

  (** LZ4 frames: one or more, skippable frames among them, read to the
      input's end. The encoder writes one frame of independent blocks with
      the content size and the content checksum. Linked blocks reach back
      into the earlier blocks of their frame.

      Refused, at the field's first byte: a magic number that begins no
      frame; a version other than 01; a set reserved bit; a block size code
      below 4; a header checksum that does not match; a block size above
      the declared maximum; a block, content checksum or content size that
      does not match; a block decoding past the declared maximum, at the
      code that does; a block ending inside a sequence or after a match, at
      the token of its last sequence; and the block refusals of {!Block}. A
      frame naming a dictionary is [Unsupported] at its Dict_ID, and the
      legacy format at its magic number. *)
  module Frame : sig
    val decode : bigbytes -> bigbytes -> (progress, error) result
    (** [decode src dst] decodes every frame of [src]. *)

    val bound : int -> int
    (** [bound n] holds the frame of any [n] bytes. *)

    val encode : bigbytes -> bigbytes -> int option
    (** [encode src dst] writes one frame of [src] to [dst]. *)

    val encode_string : string -> string
    (** [encode_string s] is the frame of [s]. *)
  end
end

(** Snappy blocks, whose preamble states the length of their content, at
    most 2{^32} - 1 bytes. The preamble is checked against [dst],
    [Too_long] at its first byte, and against what the data produces, and
    bounds nothing else. A block is refused for a preamble longer than 5
    bytes or above 2{^32} - 1, at its first byte; and, at the element's
    tag, for a copy offset of 0 or before the block's start, or an element
    whose output passes the preamble's length. A block whose input ends
    inside an element, or before its output reaches the preamble's length,
    is [Truncated]. *)
module Snappy : sig
  val decode : bigbytes -> bigbytes -> (progress, error) result
  (** [decode src dst] decodes the block at [src]'s start. *)

  val bound : int -> int
  (** [bound n] holds the block of any [n] bytes.

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
    [string (a ^ b)] is [string ~crc:(string a) b]. *)
module Crc32 : sig
  val bigbytes : ?crc:int -> bigbytes -> int
  (** [bigbytes ~crc b] is the CRC-32 of [b] continued from [crc]
      (default 0). *)

  val string : ?crc:int -> ?first:int -> ?length:int -> string -> int
  (** [string ~crc ~first ~length s] is the CRC-32 of the [length] bytes of
      [s] from [first] (default: all of [s]), continued from [crc].

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
