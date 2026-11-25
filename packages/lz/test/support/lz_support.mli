(** What lz's suites share: buffers, the corpora that the golden
    generator and the benches also use, drawn inputs that break codecs,
    testables, and the vendor-written goldens. *)

(** {1 Buffers} *)

val of_string : string -> Lz.bigbytes
(** [of_string s] is a fresh buffer holding [s]. *)

val to_string : ?length:int -> Lz.bigbytes -> string
(** [to_string ~length b] is the first [length] bytes of [b] (default all). *)

val create : int -> Lz.bigbytes
(** [create n] is a buffer of [n] bytes, each 0xA5, so that bytes a codec
    did not write are visible. *)

val placed : offset:int -> string -> Lz.bigbytes
(** [placed ~offset s] holds [s] at [offset] bytes into a larger buffer,
    through [Bigarray.Array1.sub]: a view whose data starts at any
    alignment. *)

val room : offset:int -> int -> Lz.bigbytes
(** [room ~offset n] is [create n] placed as {!placed} places it. *)

val stepped :
  ?cut:(int -> int) -> Lz.Decoder.t -> string -> (string * int, Lz.error) result
(** [stepped ~cut d data] steps [d] over [data] as a reader of a stream
    does: step [k] is given at most [cut k] bytes (default all that is
    left) from the first byte not yet taken, with [~last:true] once the
    slice reaches [data]'s end, then empty steps until [Done]. It is the
    runs concatenated and the bytes taken up to [Done], or the error.
    Fails the test if two steps in a row take nothing and lend nothing. *)

(** {1 Corpora}

    The inputs of the goldens, which [gen/raw.exe] writes for
    [gen/generate.py], and of the benches: [corpus kind n] is the first
    [n] bytes of the stream drawn for [kind]. *)

type kind = Text | Columns | Random

val corpus : kind -> int -> string
(** [corpus kind n] is [n] bytes of [kind]: [Text], words of made-up
    syllables with no long-range repeats; [Columns], little-endian int64
    keys with small gaps and float64 prices; [Random], incompressible
    bytes. *)

(** {1 Drawn inputs} *)

(** A recipe for an input that breaks codecs, which prints as itself
    rather than as its bytes. *)
type recipe =
  | Empty
  | Byte of char
  | Run of char * int  (** [n] copies of a byte. *)
  | Noise of int * int  (** [Noise (seed, n)]: [n] random bytes. *)
  | Corpus of kind * int
  | Match of { distance : int; length : int; seed : int }
      (** [distance] random bytes, then [length] bytes copied from
          [distance] back (overlapping when [length > distance]), then 8
          random bytes. *)
  | Concat of recipe list

val build : recipe -> string
(** [build r] is the input [r] describes. *)

val pp_recipe : Format.formatter -> recipe -> unit

val recipe : recipe Windtrap.Gen.t
(** [recipe] draws every recipe up to a few hundred KiB: empty, one byte, runs,
    noise, corpus prefixes, matches, and concatenations of these. A match's
    distance is at the edge of every offset class (1, 2, 3, 4, 7, 8, 15, 16, 31,
    32, 33, 64, 65, 256, 257, 1024, 32 KiB - 1, 32 KiB, 32 KiB + 1, 64 KiB or 1
    MiB) and its length at the edges of the formats' length codes (3, 4, 5, 15,
    16, 19, 258, 259, 273, 65535 or 65536). It draws each constructor with
    probability about 1/7, so a [cover] of one misses about one run in 5 million
    at 100 cases: a law that covers them takes {!every_kind} as [~examples].
    Inputs above 4 MiB are too slow to draw: pass them as [~examples]. *)

val every_kind : recipe list
(** [every_kind] is one recipe of each constructor {!recipe} draws, the corpus
    prefix past 128 KiB + 1 (above Zstandard's block and two of deflate's
    windows), for [~examples]: a law's covers of a constructor, of the empty
    input or of an input past a block hold on every run. *)

val large : recipe list
(** Inputs past the formats' size limits, for [~examples]: 128 KiB + 1
    (Zstandard's block), 4 MiB + 1 (LZ4's largest frame block) and 5 MiB,
    of text, with a 1 MiB distance match. *)

(** {1 Bounds} *)

val largest_bound : (int -> int) -> unit
(** [largest_bound bound] fails the test unless [bound] raises
    [Invalid_argument] exactly on the sizes whose bound would pass [max_int]. It
    checks that the largest size [t] with a bound, found by bisection, has one
    of at least [t] and above [max_int - 8], since a byte more of input adds at
    most 8 to a bound; that [t - 1]'s is no larger; and that [t + 1], [t + 2]
    and [max_int] raise. *)

(** {1 Short destinations} *)

(** How short a destination is of an output of [k] bytes, which prints as
    itself. *)
type short =
  | Short of int  (** [Short d]: [d] bytes short. *)
  | Part of int  (** [Part p]: [p] thousandths of [k], [p] below 1000. *)

val short : short Windtrap.Gen.t
(** [short] draws [Short 1] and [Part 0], an empty destination, each with
    probability 1/6, then [Short] of 2 to 16 bytes and [Part] of 1 to 999
    thousandths. A [cover] of either edge misses about one run in 10{^ 8} at 100
    cases: a law that covers them takes {!short_edges} in its [~examples]. *)

val short_edges : short list
(** [short_edges] is [[Short 1; Part 0]]. *)

val capacity : int -> short -> int
(** [capacity k s] is the size of the destination [s] describes for an
    output of [k] bytes, in \[0, [k]) when [k] is positive. *)

(** {1 Changed bytes} *)

val anywhere : int Windtrap.Gen.t
(** [anywhere] draws an integer in \[0, 2{^ 30}) uniformly, a position in an
    input of [n] bytes as [anywhere mod n]: each byte of an input of a few MiB
    is drawn about as often as any other, its end as often as its header. A law
    that changes bytes passes the first and last as [~examples]. *)

val position : int -> int -> int
(** [position i n] is the byte [i] names in an input of [n] bytes, [n] positive:
    [i mod n], counted from the end when [i] is negative. An example names the
    last byte as [-1], without building the input. *)

(** {1 LZW}

    lz has no LZW encoder: the suites and the benches encode with this one,
    written from the encoder of TIFF 6.0's section 13. A code is a value
    and its width in bits. *)

val lzw_codes : ?last:int -> ?flush:bool -> string -> (int * int) list
(** [lzw_codes ~last ~flush s] is the codes of the LZW stream of [s], in
    order: ClearCode (256), the codes of [s]'s strings, EndOfInformation
    (257). Each code is 9 bits wide until the encoder has added entry 511,
    10 until 1023, 11 until 2047 and 12 after, and a code written without
    adding an entry counts as one that adds, as the section asks. After
    adding entry [last] (default 4094, the section's) the encoder writes
    ClearCode and empties its table. The byte it holds then becomes the
    first code after ClearCode, or with [~flush:true] a code before it,
    which the section allows after entry 4093. A [last] of 4095 widens
    ClearCode past 12 bits, so it is written at 12.

    Raises [Invalid_argument] unless [last] is in \[258, 4095\]. *)

val lzw_iter :
  ?last:int -> ?flush:bool -> string -> (int -> int -> unit) -> unit
(** [lzw_iter ~last ~flush s f] calls [f c w] on each code [c] of
    [lzw_codes ~last ~flush s] and its width [w], in order, without building the
    list.

    Raises [Invalid_argument] as {!lzw_codes} does. *)

val lzw_pack : (int * int) list -> string
(** [lzw_pack codes] is [codes] packed most significant bit first, the
    last byte padded with zero bits. *)

val lzw : ?last:int -> ?flush:bool -> string -> string
(** [lzw ~last ~flush s] is [lzw_pack (lzw_codes ~last ~flush s)], without
    building the list. *)

(** {1 LZF}

    lz has no LZF encoder: the suites encode with this one, written from
    the elements {!Lz.Lzf} states. *)

type lzf =
  | Literal of string  (** A literal run of 1 to 32 bytes. *)
  | Copy of { distance : int; length : int }
      (** A match of 3 to 264 bytes from 1 to 8192 bytes back. *)

val lzf_pack : lzf list -> string
(** [lzf_pack es] is the bytes of the elements [es], in order.

    Raises [Invalid_argument] if an element is outside its range. *)

val lzf_elements : string -> lzf list
(** [lzf_elements s] parses [s] greedily: at each byte, the longest match
    with the last earlier position of the same three bytes, at most 8192
    back, or the byte joins a literal run. A run of [s]'s one byte gives
    matches that overlap their output. *)

val lzf : string -> string
(** [lzf s] is [lzf_pack (lzf_elements s)]. *)

(** {1 Testables} *)

val error : Lz.error Windtrap.testable

val refusal : Lz.error Windtrap.testable
(** Errors equal when their [at] and their constructor agree, printed whole: a
    [Malformed] or [Unsupported] message is not compared, since lz's own
    messages are baselines that each suite keeps in an [expect]. *)

val progress : Lz.progress Windtrap.testable
val step : Lz.step Windtrap.testable

val decoded : (string, Lz.error) result Windtrap.testable
(** Decoded bytes or the error, printing a long string by its length and
    digest. *)

(** {1 Goldens}

    Files the vendors wrote, listed in [golden/manifest.txt] by
    [gen/generate.py]. A suite that reads them depends on the files of
    [golden/] and runs from the test directory. *)

type golden = {
  file : string;  (** The file name under [golden/]. *)
  codec : string;
      (** The decoder that reads it: ["deflate"], ["zlib"], ["gzip"],
          ["zstd"], ["lz4b"], ["lz4f"] or ["snappy"]. *)
  members : int;  (** The members or frames it holds, skippable ones too. *)
  size : int;  (** Its length in bytes, known without reading it. *)
  raw : string Lazy.t;  (** What it decodes to. *)
  data : string Lazy.t;  (** Its bytes. *)
}

val goldens : unit -> golden list
(** [goldens ()] is every golden, in the manifest's order. The manifest is read
    on the first call of the process; a golden's [raw] and [data] are made when
    forced. *)

val goldens_of : string -> golden list
(** [goldens_of codec] is the goldens [codec] reads. *)

val raw_of_spec : string -> string
(** [raw_of_spec spec] is the input a manifest's [raw] field names: [kind:n]
    parts joined by [+]. *)

val decoder : string -> Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result
(** [decoder codec] is the whole-buffer decode of the codec a golden names. *)
