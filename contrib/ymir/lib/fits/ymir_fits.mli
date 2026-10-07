(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FITS files.

    FITS is the file format of astronomy: images, catalogues, sky maps. A file
    is a list of HDUs, each a header of 80-byte records and a data unit.

    {[
    let ( let* ) = Result.bind

    let read path =
      let* hdus = Fits.read path in
      let* sci = Fits.get "SCI" hdus in
      Fits.Image.values Nx.float32 sci
    ]}

    {b Errors.} Every failure in data is [Error] of a string whose places come
    first:
    {v <name>: HDU <i> (<EXTNAME>), card <k> (<KEY>), bytes <a>-<b>: <what> v}
    [<name>] is the path or {!Fits.of_bytes}'s name. HDUs count from 0, cards
    within their HDU from 1, and bytes are inclusive file offsets. Each place
    appears when known. A caller's bug, such as an illegal keyword name or a
    dtype FITS cannot store, raises [Invalid_argument]. *)

module Fits : sig
  (** {1:values Values} *)

  (** Value fields. *)
  module Value : sig
    type 'a t
    (** The type for how a value field reads and prints as an ['a]. *)

    val bool : bool t
    (** [bool] is the logical [T] or [F]. *)

    val int : int t
    (** [int] is integer text within [int]'s range. An integer past it reads
        with {!text}. *)

    val float : float t
    (** [float] is integer or real text (FITS 4.0 §4.2.4), read correctly
        rounded. [NAN], [INF], hexadecimal floats, underscores and a real past
        float64's range do not read. It prints the fewest significant digits
        that read back to the float, with a point or an exponent. *)

    val string : string t
    (** [string] is a quoted string, with doubled quotes read as one, the
        [CONTINUE] records after a string ending in [&] joined (§4.2.1.2), and
        trailing spaces dropped: [''] is [""] and [' '] is [" "]. *)

    val text : string t
    (** [text] is the value field as written, trimmed: a string's text keeps its
        quotes. It prints as given, which must be one FITS value. *)

    val map : ('a -> ('b, string) result) -> ('b -> 'a) -> 'a t -> 'b t
    (** [map f g v] reads with [v] then [f], and prints with [g] then [v]. *)
  end

  (** {1:headers Headers} *)

  (** Headers.

      A header is its records in file order, each the 80 bytes the file held:
      order, duplicates, blank records and bytes outside ASCII 32–126 are kept,
      so every record a program does not edit is written back byte for byte. A
      value is parsed when it is asked for, and fails there alone.

      A record is {e commentary} when its keyword is [COMMENT], [HISTORY] or
      blank, or its bytes 9–10 are not ["= "], unless it starts with
      ["HIERARCH "] and holds [=].

      {b Keywords.} A standard keyword is 1 to 8 characters of [A–Z 0–9 - _]. A
      hierarchical keyword is the tokens between ["HIERARCH "] and the first [=]
      of a record, joined by single spaces: two or more tokens, or one longer
      than 8 bytes, each of bytes 33–126 other than [=], in the case written, as
      in ["ESO DET DIT"]. Keywords compare exactly. {!get}, {!find}, {!set} and
      {!remove} raise [Invalid_argument] on a name of neither form, such as
      ["naxis"], and on [COMMENT], [HISTORY], blank and [CONTINUE], which
      {!commentary} reads.

      {b Duplicates.} A keyword repeated with different values is indeterminate
      (§4.1.2.3): {!get} returns the value when every card agrees, and is an
      [Error] naming two cards that disagree otherwise. *)
  module Header : sig
    type t
    (** The type for headers: records in file order, without [END]. *)

    val empty : t
    (** [empty] has no record. *)

    val of_string : ?name:string -> string -> (t, string) result
    (** [of_string ~name s] reads 80-byte records back to back up to [END], or,
        when [s] holds a newline, lines of at most 80 bytes up to [END], as
        [.head] files hold them. [name] starts the header's errors. *)

    val to_string : t -> string
    (** [to_string h] is [h]'s records, [END], and spaces to a multiple of 2880
        bytes. *)

    val records : t -> string list
    (** [records h] is [h]'s records, 80 bytes each. *)

    val equal : t -> t -> bool
    (** [equal h h'] is [true] iff [h] and [h'] have the same records. *)

    val pp : Format.formatter -> t -> unit
    (** [pp] formats a header one record per line, trailing spaces dropped. *)

    val get : 'a Value.t -> string -> t -> ('a, string) result
    (** [get v k h] is the value of keyword [k] read with [v]. It is an [Error]
        if [k] is absent or undefined, if a card does not read, or if its cards
        disagree.

        Raises [Invalid_argument] if [k] is not a keyword (see above). *)

    val find : 'a Value.t -> string -> t -> ('a option, string) result
    (** [find] is {!get} with [Ok None] for an absent or undefined keyword. *)

    val set : ?comment:string -> 'a Value.t -> string -> 'a -> t -> t
    (** [set ~comment v k x h] is [h] with one card giving [k] the value [x], at
        the place of [k]'s first card or appended, its later cards removed.
        [comment] replaces the card's comment; without it the comment is kept,
        cut to fit beside the new value.

        A value that fits bytes 11–30 is written in the fixed format of
        §4.2.2–§4.2.4, others in the free format from byte 11. A string over 68
        bytes continues over [CONTINUE] records. [get v k (set v k x h)] is
        [Ok x] for every finite float and every string without trailing spaces.

        Raises [Invalid_argument] if [k] is not a keyword, on a non-finite
        float, on a byte outside ASCII 32–126, on {!Value.text} that is not one
        FITS value, on a hierarchical keyword that leaves no room for the value,
        and on a [comment] that does not fit beside a value other than a string.
    *)

    val remove : string -> t -> t
    (** [remove k h] is [h] without [k]'s cards and their [CONTINUE] records.

        Raises [Invalid_argument] as {!get} does. *)

    val commentary : string -> t -> string list
    (** [commentary k h] is the text of [h]'s commentary records whose keyword
        is [k] (bytes 9–80, trailing spaces dropped), in order. [k] is
        ["COMMENT"], ["HISTORY"], [""] for blank, or any other keyword whose
        records have no value indicator. *)

    val add_commentary : string -> string -> t -> t
    (** [add_commentary k s h] appends [s] as [k] records of 72 bytes each.

        Raises [Invalid_argument] if [k] is not ["COMMENT"], ["HISTORY"] or
        [""], or if [s] holds a byte outside ASCII 32–126. *)
  end

  (** {1:hdus HDUs and files} *)

  type hdu
  (** The type for HDUs: a header and the bytes of its data unit. *)

  val read : string -> (hdu list, string) result
  (** [read path] is the HDUs of the file [path], primary first: every header
      copied, every data unit checked to lie inside the file, no data read. A
      file's structure fails here alone: a file cut inside a data unit, or
      inside a header of an HDU that starts [XTENSION], is an [Error] naming
      that HDU. A file that ends inside its last data unit's padding reads, as
      do bytes after the last HDU that do not start [XTENSION]. A gzip stream is
      an [Error] naming [Nx_io.gunzip]. *)

  val of_bytes :
    name:string -> (int, Nx.uint8_elt) Nx.t -> (hdu list, string) result
  (** [of_bytes ~name b] is the HDUs of the bytes [b] in C order, as {!read}
      reads a file. [name] starts their errors. *)

  val get : ?ver:int -> string -> hdu list -> (hdu, string) result
  (** [get ~ver name hdus] is the HDU whose [EXTNAME] is [name] and whose
      [EXTVER] is [ver], an absent [EXTVER] being 1. Without [ver], it is the
      one HDU named [name]. [Error] listing the HDUs' names if there is none, or
      several. *)

  val header : hdu -> Header.t
  (** [header hdu] is [hdu]'s header. *)

  val data : hdu -> (int, Nx.uint8_elt) Nx.t
  (** [data hdu] is [hdu]'s data unit as stored, heap included, padding
      excluded: for a read HDU, a view of the file, which must not change while
      the view lives. *)

  val name : hdu -> string
  (** [name hdu] is the path given to {!read}, {!of_bytes}'s name, or [""] for a
      constructed HDU. *)

  val digest : hdu -> string
  (** [digest hdu] is [blake2b-256:<hex>] of every header of [hdu]'s file, each
      as {!Header.to_string} gives it, in file order; of [hdu]'s header for a
      constructed one. Headers state each HDU's sizes and, in files this library
      writes, [DATASUM], so they identify the data; two files with equal headers
      and different data without [DATASUM] share a digest. *)

  val v : Header.t -> (int, Nx.uint8_elt) Nx.t -> hdu
  (** [v h b] is the HDU of any kind with header [h] and data unit [b], as
      {!write} writes it after the first HDU: [DATASUM] and [CHECKSUM] go, and a
      primary image becomes an [IMAGE] extension.

      Raises [Invalid_argument] if [h]'s mandatory keywords do not read, or
      describe a data unit of another size than [b]'s. *)

  val with_header : Header.t -> hdu -> hdu
  (** [with_header h hdu] is [hdu] with [h]'s records, except the structural
      ones ({!write}), which stay [hdu]'s: its mandatory records first, each
      other one where [h] holds that keyword, or after [h]'s records. *)

  val verify : hdu -> (unit, string) result
  (** [verify hdu] reads [hdu] once and checks its checksums (§4.4.2.7):
      [DATASUM], the ones' complement sum of the data unit as big-endian 32-bit
      words, and [CHECKSUM], which makes the whole HDU sum to −0. It is an
      [Error] naming both sums if either is absent or disagrees. *)

  val write : string -> hdu list -> (unit, string) result
  (** [write path hdus] writes [hdus] to a new file beside [path], syncs it and
      renames it over [path], with mode 0640. A failed write leaves [path] as it
      was. Values read by copy from a replaced file do not change.

      {b Structure is the writer's.} The structural keywords are [SIMPLE],
      [XTENSION], [BITPIX], [NAXIS], [NAXISn], [EXTEND], [PCOUNT], [GCOUNT], an
      image's [BZERO] and [BSCALE], a table's [TFIELDS], [THEAP] and
      column-numbered keywords, the tile-compression keywords, [DATASUM] and
      [CHECKSUM]. A record holding the value [write] computes stays at its place
      with its text; others are written at the standard's positions, and
      differing copies are dropped. The first HDU becomes the primary, and an
      empty primary is prepended when it is not an image.

      {b Checksums} are written on every HDU, in the pass that writes its bytes,
      with no date, so output is a function of input. An HDU whose [DATASUM]
      disagrees with its data is an [Error] naming
      [Fits.v (Header.remove "DATASUM" (Fits.header hdu)) (Fits.data hdu)], the
      call that accepts the data as it is. *)

  val pp : Format.formatter -> hdu -> unit
  (** [pp] formats an HDU's kind, [EXTNAME] and [EXTVER], then {!Image.pp}, or
      its data size for other kinds. *)

  (** {1:images Images} *)

  (** Images.

      {b Elements.} [BITPIX] gives the stored format; when [BSCALE] is 1 and
      [BZERO] is the standard's offset for it (Table 11), the format is
      unsigned, or signed for bytes. [BSCALE] and [BZERO] compare as exact
      decimals: [32768], [3.2768E4] and [32768.0] are one offset. Any other
      [BSCALE] or [BZERO] makes the image {e scaled}: its element is [BITPIX]'s
      own format.

      {b Values.} {!raw} returns stored numbers in any dtype that holds every
      value of the element exactly; {!values} returns physical values in a float
      dtype. On a scaled image the physical value of a stored [s] is
      [BZERO + BSCALE × s] computed as [Nx.fma bscale s bzero] in float64,
      [BSCALE] and [BZERO] the float64s nearest their text, then cast to the
      dtype once.

      {b Undefined pixels.} [BLANK] names a stored value of an integer [BITPIX],
      and NaN is undefined in a float image. {!values} returns NaN there and
      {!raw} the stored number.

      {b Windows.} [window] is a [(start, stop)] range per axis, in the tensor's
      axis order, as [Nx.slice] takes [R (start, stop)]. A read copies only the
      rows the window covers. A window of the wrong rank or past the shape is an
      [Error]; a range with [start < 0] or [start > stop] raises
      [Invalid_argument].

      {b Byte order.} FITS stores big-endian. Every result is host memory
      independent of the file. *)
  module Image : sig
    type t
    (** The type for an image's description: shape, stored format, scaling and
        [BLANK]. *)

    val of_hdu : hdu -> (t, string) result
    (** [of_hdu hdu] describes a primary array or an [IMAGE] extension. It is an
        [Error] for other kinds, for random groups, for [NAXIS = 0] and for more
        than 32 axes. *)

    val shape : t -> int array
    (** [shape t] is [[|NAXISn; …; NAXIS1|]]. *)

    val element : t -> Nx_dtype.Scalar.t
    (** [element t] is the stored format, the standard's unsigned offsets
        included. *)

    val scaled : t -> bool
    (** [scaled t] is [true] iff [BSCALE] or [BZERO] scale the stored numbers
        beyond the offsets. *)

    val pp : Format.formatter -> t -> unit
    (** [pp] formats the element and shape, as [float32 [4800; 4600]], and the
        scaling and [BLANK]. *)

    val raw :
      ?window:(int * int) array ->
      ('a, 'b) Nx.dtype ->
      hdu ->
      (('a, 'b) Nx.t, string) result
    (** [raw ~window dtype hdu] is the stored numbers of [hdu]'s image as
        [dtype]. It is an [Error] naming the dtypes that hold the element if
        [dtype] does not hold every value of it exactly. *)

    val values :
      ?window:(int * int) array ->
      (float, 'b) Nx.dtype ->
      hdu ->
      ((float, 'b) Nx.t, string) result
    (** [values ~window dtype hdu] is the physical values of [hdu]'s image, NaN
        where undefined. On an unscaled image it succeeds exactly when {!raw}
        does. *)

    val validity :
      ?window:(int * int) array -> hdu -> (Nx.bit_t option, string) result
    (** [validity ~window hdu] is [false] where a pixel is undefined ([BLANK] or
        NaN), or [None] if none is. *)

    val hdu : ?tiles:int array -> Header.t -> ('a, 'b) Nx.t -> hdu
    (** [hdu h t] is [t] as an image HDU with [h]'s cards that are not
        structural ({!write}); [BLANK] is dropped from a float image. It reads
        back bit for bit, NaN payloads included.

        Raises [Invalid_argument] if [t] is a scalar or of a dtype FITS images
        do not hold (bool, bit, complex, float16, bfloat16, the float8s, int4,
        uint4), naming the cast that stores it. *)
  end
end
