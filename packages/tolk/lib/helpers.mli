(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Compilation targets, integer and list utilities, terminal text, the disk
    cache and programs. *)

(** {1:targets Compilation targets} *)

(** Compilation targets.

    A target selects, for one device, the renderer that generates its code, the
    architecture to generate for, and the interface and device indices to open
    it with. Empty fields are unspecified. *)
module Target : sig
  type t = {
    device : string;  (** The device, e.g. ["AMD"]. *)
    renderer : string;  (** The renderer, e.g. ["HIP"]. *)
    arch : string;  (** The architecture, e.g. ["gfx1100"]. *)
    interface : string;  (** The interface opening the device. *)
    indices : string;  (** The indices of the devices to open. *)
  }
  (** The type for targets. *)

  val of_string : string -> (t, string) result
  (** [of_string s] reads [s], written
      [[INTERFACE[:INDICES]+]DEVICE[:RENDERER[:ARCH]]]. [DEVICE] and [RENDERER]
      are uppercased; the other fields are kept as written. It is an error if
      [s] has more than one [+], naming [s], or if the part after the [+] has
      more than two [:], naming that part. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a target as {!of_string} reads it, without trailing
      separators. *)
end

(** {1:lists Integers and lists} *)

val prod : int list -> int
(** [prod l] is the product of the elements of [l], [1] if [l] is empty. *)

val dedup : (module Hashtbl.HashedType with type t = 'a) -> 'a list -> 'a list
(** [dedup (module H) l] is [l] with each element kept only at its first
    occurrence, elements being compared with [H.equal]. *)

val argsort : int list -> int list
(** [argsort l] is the positions of the elements of [l], ordered by increasing
    element; equal elements keep their order. On a permutation it is the inverse
    permutation. *)

val all_same : ('a -> 'a -> bool) -> 'a list -> bool
(** [all_same equal l] is [true] iff every element of [l] is [equal] to the
    first. It is [true] on [[]]. *)

val get_single_element : 'a list -> 'a
(** [get_single_element l] is the element of [l].

    Raises [Invalid_argument] if [l] does not have exactly one element. *)

val ceildiv : int -> int -> int
(** [ceildiv num amt] is [num / amt] rounded up.

    Raises [Division_by_zero] if [amt] is [0]. *)

val round_up : int -> int -> int
(** [round_up num amt] is the smallest multiple of [amt] greater than or equal
    to [num], for positive [amt].

    Raises [Division_by_zero] if [amt] is [0]. *)

val floordiv : int -> int -> int
(** [floordiv x y] is [x / y] rounded down, and [0] if [y] is [0]. *)

val floormod : int -> int -> int
(** [floormod x y] is [x - floordiv x y * y]: the remainder of the division
    rounded down, with the sign of [y]. It is [x] if [y] is [0]. *)

val lo32 : int -> int
(** [lo32 x] is the low 32 bits of [x]. *)

val hi32 : int -> int
(** [hi32 x] is [x] shifted right by 32 bits, keeping its sign. *)

val data64 : int -> int * int
(** [data64 x] is [(hi32 x, lo32 x)]. *)

val data64_le : int -> int * int
(** [data64_le x] is [(lo32 x, hi32 x)]. *)

(** {1:select Selection} *)

val select_by_name :
  error:string ->
  ('a -> string) ->
  string ->
  'a list ->
  ('a list, string) result
(** [select_by_name ~error name query candidates] is the candidates [c] with
    [name c = query] in their order, or [candidates] if [query] is [""]. If
    there are none, the error is [error], followed by the name of a candidate
    that [query] may have misspelled, if one is similar enough. *)

val select_first_inited :
  error:string -> (unit -> ('a, string) result) list -> ('a, string) result
(** [select_first_inited ~error candidates] is the value of the first of
    [candidates] that initializes, trying them in order. If none does, the error
    is the only candidate's error, or [error] followed by each candidate's
    error, one per line. An exception raised by a candidate is not caught: it
    escapes, and the later candidates are not tried. *)

(** {1:text Terminal text} *)

(** The type for the colors of terminal text. *)
type color =
  | Black
  | Red
  | Green
  | Yellow
  | Blue
  | Magenta
  | Cyan
  | White
  | Bright_black
  | Bright_red
  | Bright_green
  | Bright_yellow
  | Bright_blue
  | Bright_magenta
  | Bright_cyan
  | Bright_white

val colored : ?background:bool -> color -> string -> string
(** [colored c s] is [s] wrapped in the ANSI escape sequences that paint it, or
    its background if [background] is [true], in [c]. It is [s] itself when
    {!Setting.no_color} is set. [background] defaults to [false]. *)

val time_to_str : ?w:int -> float -> string
(** [time_to_str ~w t] is the duration of [t] seconds right-aligned in [w]
    columns with two decimals, followed by a two-column unit: [s ] above 10
    seconds, [ms] above 10 milliseconds, and [us] otherwise. [w] defaults to
    [8]. *)

val size_to_str : int -> string
(** [size_to_str s] is the size of [s] bytes, with two decimals in the largest
    of [GB], [MB] and [KB] it reaches (powers of 1024), or as [s B] below a
    kilobyte. *)

val ansistrip : string -> string
(** [ansistrip s] is [s] without its ANSI escape sequences that erase a line or
    set graphic attributes. The latter run from the escape character and an
    opening bracket to the first ['m'] on the same line. *)

val ansilen : string -> int
(** [ansilen s] is the number of characters of [ansistrip s], counting UTF-8
    encoded characters once. *)

val ansipad : string -> int -> string
(** [ansipad s w] is [s] followed by the spaces that bring its {!ansilen} to
    [w], none if it has that length already. *)

val strip_parens : string -> string
(** [strip_parens s] is [s] without its first and last characters if they are
    parentheses enclosing the whole of [s], and [s] otherwise. *)

val pluralize : string -> int -> string
(** [pluralize st cnt] is [cnt] followed by [st], with an [s] unless [cnt] is
    [1]. *)

val to_function_name : string -> string
(** [to_function_name s] is [ansistrip s] with each character other than an
    ASCII letter, digit or underscore replaced by its Unicode code point in
    uppercase hexadecimal, at least two digits. The result is a valid C
    identifier unless empty or starting with a digit. *)

(** {1:cache Disk cache} *)

val cache_dir : string
(** [cache_dir] is the directory of the files tolk caches: [tolk] in
    [XDG_CACHE_HOME] if that variable is set, and otherwise in
    [~/Library/Caches] on macOS and [~/.cache] elsewhere. *)

val cachedb : string
(** [cachedb] is the directory of the {!Diskcache}, from the variable [CACHEDB].
    Defaults to [cache] in {!cache_dir}, made absolute against the current
    directory when {!cache_dir} is relative. *)

(** Persistent key-value tables.

    A table maps strings to strings across runs and processes. Entries are
    written atomically, so a reader sees a complete entry or none, and the last
    writer of a key wins. Tables are versioned with the library: entries written
    by a library whose cached data meant something else are not seen. Tables and
    keys are any strings; entries stay within {!cachedb} whatever they hold. The
    cache is disabled while {!Setting.cachelevel} is not positive. *)
module Diskcache : sig
  val get : table:string -> string -> string option
  (** [get ~table key] is the value of [key] in [table], if any. It is [None]
      while the cache is disabled.

      Raises [Failure] if the entry exists but is malformed. *)

  val put : table:string -> string -> string -> unit
  (** [put ~table key value] binds [key] to [value] in [table]. It does nothing
      while the cache is disabled.

      Raises [Sys_error] if the entry cannot be written. *)

  val clear : unit -> unit
  (** [clear ()] removes every entry of every table, of every version. Files
      under {!cachedb} that the cache did not write are kept. *)
end

(** {1:exec Programs} *)

val system : ?input:string -> string -> string
(** [system ~input cmd] is the output of the command [cmd], a program and its
    arguments separated by spaces, run with [input] on its standard input or,
    without [input], with the process's: what it writes on its standard output
    and standard error, together, without the white space that starts and ends
    it. With {!Setting.debug} at least 1, it prints how many bytes [cmd]
    returned and how long it took.

    Raises [Failure] naming [cmd] with the reason and output if [cmd] cannot run
    or exits otherwise than with status [0]. *)

val cpu_objdump : string -> unit
(** [cpu_objdump lib] prints the instructions of the object file [lib], as
    [objdump -d] prints them.

    Raises [Failure] as {!system} if [objdump] fails. *)

val find_llvm_objdump : unit -> string
(** [find_llvm_objdump ()] is the [llvm-objdump] to run: Homebrew's on macOS,
    and elsewhere the first of ROCm's and those named [llvm-objdump-21],
    [llvm-objdump-20] and [llvm-objdump] on [PATH].

    Raises [Failure] if there is none. *)

val amdgpu_disassemble : string -> unit
(** [amdgpu_disassemble lib] prints the instructions of the AMD GPU code object
    [lib], as {!find_llvm_objdump}'s [llvm-objdump -d] prints them, without the
    padding instructions that end it ([s_nop 0] and [s_code_end]).

    Raises [Failure] if there is no [llvm-objdump], or as {!system} if it fails.
*)
