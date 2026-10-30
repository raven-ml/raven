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

val kind_of_string : string -> kind
(** [kind_of_string s] is the kind named ["text"], ["columns"] or
    ["random"]. Raises [Invalid_argument] otherwise. *)

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

val length : recipe -> int
(** [length r] is the length of [build r], without building it. *)

val pp_recipe : Format.formatter -> recipe -> unit

val distances : int list
(** Match distances at the edge of every offset class: 1, 2, 3, 4, 7, 8,
    15, 16, 31, 32, 33, 64, 65, 256, 257, 1024, 32 KiB - 1, 32 KiB,
    32 KiB + 1, 64 KiB and 1 MiB. *)

val lengths : int list
(** Match lengths at the edges of the formats' length codes: 3, 4, 5, 15,
    16, 19, 258, 259, 273, 65535 and 65536. *)

val recipe : recipe Windtrap.Gen.t
(** [recipe] draws every recipe up to a few hundred KiB: empty, one byte,
    runs, noise, corpus prefixes, a match at each of {!distances} with
    each of {!lengths}, and concatenations of these. It draws each
    constructor with probability at least 0.05, so a [cover] of one holds
    over 100 cases from any seed with probability above 0.99. Inputs above
    4 MiB are too slow to draw: pass them as [~examples]. *)

val large : recipe list
(** Inputs past the formats' size limits, for [~examples]: 128 KiB + 1
    (Zstandard's block), 4 MiB + 1 (LZ4's largest frame block) and 5 MiB,
    of text, with a 1 MiB distance match. *)

(** {1 Bounds} *)

val largest_bound : (int -> int) -> Windtrap.test
(** [largest_bound bound] states that [bound] raises [Invalid_argument]
    exactly on the sizes whose bound would pass [max_int]. The largest size
    [t] with a bound, found by bisection, has one of at least [t] and
    above [max_int - 8], since a byte more of input adds at most 8 to a
    bound; [t - 1]'s is no larger; [t + 1], [t + 2] and [max_int]
    raise. *)

(** {1 The runtime lock} *)

val lock_released :
  string ->
  (int -> unit -> 'a) ->
  small:(unit -> unit) ->
  check:('a -> unit) ->
  Windtrap.test
(** [lock_released name long ~small ~check] states that the call
    [long k ()] runs with the runtime lock released. [long k] prepares a
    call on [k] times a base input, long enough to outlast a minor
    collection. The call runs on another domain while this one calls
    [small ()] 100 times, then collects the minor heap, which waits for
    every domain that holds its runtime lock; [check] is applied to the
    call's result. The lock was released if the collection returned while
    the call ran, and held if the collection began while the call ran and
    returned only after it ended. A run where the call ended before the
    collection began proves nothing. A run that proves no release is
    repeated once with [k = 2]: held again fails the test, and proving
    nothing again skips it. *)

(** {1 Short destinations} *)

(** How short a destination is of an output of [k] bytes, which prints as
    itself. *)
type short =
  | Short of int  (** [Short d]: [d] bytes short. *)
  | Part of int  (** [Part p]: [p] thousandths of [k], [p] below 1000. *)

val short : short Windtrap.Gen.t
(** [short] draws [Short 1] and [Part 0], an empty destination, each with
    probability 1/6, then [Short] of 2 to 16 bytes and [Part] of 1 to 999
    thousandths: a [cover] of either edge holds over 100 cases from any
    seed. *)

val capacity : int -> short -> int
(** [capacity k s] is the size of the destination [s] describes for an
    output of [k] bytes, in \[0, [k]) when [k] is positive. *)

(** {1 Checksum paths}

    The CRC-32 and Adler-32 paths this host can run, each with the
    instructions of its name ([Lz.Paths]). {!Lz} reaches only the
    fastest. *)

type path = {
  name : string;
  run : int32 -> Lz.bigbytes -> int32;
      (** [run c b] is the checksum of all of [b] from the running value
          [c]. *)
}

val crc32_paths : path list
(** [crc32_paths] is every CRC-32 path, the generic one first and the one
    {!Lz.Crc32} takes last. *)

val adler32_paths : path list
(** [adler32_paths] is every Adler-32 path, as {!crc32_paths}. *)

(** {1 Zstandard paths}

    The builds of the Zstandard decoder's hot loops this host can run
    ([Lz.Paths]): generic, and BMI2 where the CPU has it.
    {!Lz.Zstd} reaches only the last. *)

type zstd_path = {
  label : string;
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
      (** [decode src dst] is {!Lz.Zstd.decode} [src dst] run with this
          path. *)
}

val zstd_paths : zstd_path list
(** [zstd_paths] is every path, the generic one first and the one
    {!Lz.Zstd} takes last. *)

type zstd_enc_path = {
  label : string;
  encode : level:int -> Lz.bigbytes -> Lz.bigbytes -> int option;
      (** [encode ~level src dst] is {!Lz.Zstd.encode} [~level src dst] run
          with this path. *)
}

val zstd_enc_paths : zstd_enc_path list
(** [zstd_enc_paths] is every path of the Zstandard encoder, the generic
    one first and the one {!Lz.Zstd} takes last. *)

val zstd_encode_limit :
  limit:int -> ?step:int -> level:int -> Lz.bigbytes -> Lz.bigbytes ->
  int option
(** [zstd_encode_limit ~limit ~level src dst] is {!Lz.Zstd.encode}
    [~level src dst] by an encoder whose match finders rebase their indices
    once a block that starts past the window would end past index [limit],
    rather than past 3 GiB: with [~limit:0], every such block. With
    [~step], it is the frame a {!Lz.Zstd.encoder} writes when given at most
    [step] bytes a step. *)

(** {1 Deflate paths}

    The builds of the deflate encoder's matchfinder rebase this host can
    run ([Lz.Paths]): generic, and SSE2 where the compiler targets it. {!Lz.Deflate} reaches only the last. *)

type deflate_enc_path = {
  label : string;
  encode : level:int -> Lz.bigbytes -> Lz.bigbytes -> int option;
      (** [encode ~level src dst] is {!Lz.Deflate.encode} [~level src dst]
          run with this path. *)
}

val deflate_enc_paths : deflate_enc_path list
(** [deflate_enc_paths] is every path, the generic one first and the one
    {!Lz.Deflate} takes last. *)

(** {1 Testables} *)

val why : Lz.why Windtrap.testable
val error : Lz.error Windtrap.testable
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
  vendor : string;  (** The vendor and level that wrote it. *)
  members : int;  (** The members or frames it holds, skippable ones too. *)
  raw : string Lazy.t;  (** What it decodes to. *)
  data : string Lazy.t;  (** Its bytes. *)
}

val goldens : unit -> golden list
(** [goldens ()] is every golden, in the manifest's order. *)

val goldens_of : string -> golden list
(** [goldens_of codec] is the goldens [codec] reads. *)

val raw_of_spec : string -> string
(** [raw_of_spec spec] is the input a manifest's [raw] field names: [kind:n]
    parts joined by [+]. *)

val decoder : string -> Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result
(** [decoder codec] is the whole-buffer decode of the codec a golden names. *)

(** {1 Suites over the goldens}

    Each takes a codec as {!golden.codec} names it. *)

val agreement : string -> Windtrap.test
(** [agreement codec] states that every golden [codec] reads decodes to its
    raw input and consumes all of it. *)

val truncations : string -> Windtrap.test
(** [truncations codec] states that every proper prefix of every golden of
    one member, up to 1 KiB, is [Truncated] at its length. A raw LZ4 block
    has no end to miss: [truncations "lz4b"] is invalid. *)

val mutations : string -> Windtrap.test
(** [mutations codec] states that a golden with any bit flipped (goldens up
    to 512 bytes) or any byte replaced (drawn, goldens up to 64 KiB) decodes
    or fails with an offset inside the input. With the sanitizers it is the
    proof that no mutation reads or writes outside the buffers. *)
